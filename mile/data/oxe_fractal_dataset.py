"""
OXE Fractal Dataset implementation following LeRobot-style layout.
Tailored for datasets like IPEC-COMMUNITY/fractal20220817_data_lerobot.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pydantic import BaseModel
from tqdm import tqdm


class OXEFractalModalityConfig(BaseModel):
    delta_indices: List[int]
    modality_keys: List[str]


class OXEFractalDataset(Dataset):
    """Dataset for OXE Fractal (LeRobot) trajectory data.

    Directory layout under dataset_root (supports HuggingFace snapshots):

    - data/chunk-XYZ/episode_XXXXXXXX.parquet
    - videos/chunk-XYZ/observation.images.image/episode_XXXXXXXX.mp4
    - meta/tasks.jsonl (task_index -> task string)
    """

    CAMERA_MAPPING = {
        # LeRobot OXE naming
        "images.image": "observation.images.image",
    }

    def __init__(
        self,
        dataset_path: str,
        modality_configs: Dict[str, OXEFractalModalityConfig],
        camera: str = "images.image",
        sequence_length: int = 2,
        stride: int = 1,
        cache_videos: bool = False,
        img_resize: Optional[Tuple[int, int]] = None,
        enable_h264_fallback: bool = True,
        video_backend: str = "pyav",
        resolve_snapshot: bool = True,
        max_init_episodes: Optional[int] = None,
        skip_video_on_error: bool = True,
        use_on_the_fly_h264: bool = False,
        h264_cache_dirname: str = "_h264_cache",
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.modality_configs = modality_configs
        self.camera = camera
        self.sequence_length = sequence_length
        self.stride = stride
        self.cache_videos = cache_videos
        self.img_resize = img_resize
        self.enable_h264_fallback = enable_h264_fallback
        self.video_backend = video_backend
        self.skip_video_on_error = skip_video_on_error
        self.use_on_the_fly_h264 = use_on_the_fly_h264
        self.h264_cache_dirname = h264_cache_dirname

        # Resolve snapshot root if needed
        if resolve_snapshot and (not (self.dataset_path / "data").exists() or not (self.dataset_path / "videos").exists()):
            snapshots_dir = self.dataset_path / "snapshots"
            if snapshots_dir.exists() and snapshots_dir.is_dir():
                snapshot_dirs = sorted([p for p in snapshots_dir.iterdir() if p.is_dir()])
                if snapshot_dirs:
                    chosen = snapshot_dirs[-1]
                    print(f"🔎 OXE Fractal: using snapshot {chosen}")
                    self.dataset_path = chosen

        if not (self.dataset_path / "data").exists():
            raise FileNotFoundError(f"Missing data/ under {self.dataset_path}")

        # Load basic metadata
        self.task_idx_mapping = self._load_task_mapping()
        self._episode_data = self._load_episode_data(max_init_episodes=max_init_episodes)
        self._trajectory_ids, self._trajectory_lengths = self._get_trajectories()
        self._all_steps = self._get_all_steps()
        self._modality_keys = self._get_modality_keys()
        self._delta_indices = self._get_delta_indices()

        # Optional video cache
        self.cached_frames: Dict[str, np.ndarray] = {}
        self.start_indices: Optional[np.ndarray] = None
        if self.cache_videos:
            self._cache_all_videos()

        self.curr_traj_id: Optional[str] = None
        self.curr_traj_data: Optional[pd.DataFrame] = None

        print(f"✅ OXE Fractal dataset initialized: {len(self)} samples, {len(self._trajectory_ids)} episodes")

    # ---------- helpers ----------
    def _get_modality_keys(self) -> Dict[str, List[str]]:
        modality_keys = defaultdict(list)
        for modality, config in self.modality_configs.items():
            modality_keys[modality] = config.modality_keys
        return dict(modality_keys)

    def _get_delta_indices(self) -> Dict[str, np.ndarray]:
        delta_indices: Dict[str, np.ndarray] = {}
        for config in self.modality_configs.values():
            for key in config.modality_keys:
                delta_indices[key] = np.array(config.delta_indices)
        return delta_indices

    def _load_task_mapping(self) -> Dict[str, str]:
        tasks_path = self.dataset_path / "meta" / "tasks.jsonl"
        mapping: Dict[str, str] = {}
        if tasks_path.exists():
            with open(tasks_path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    mapping[str(obj.get("task_index"))] = obj.get("task", "")
        return mapping

    def _load_episode_data(self, max_init_episodes: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        episode_data: Dict[str, pd.DataFrame] = {}
        data_dir = self.dataset_path / "data"
        parquet_files = sorted(data_dir.glob("**/episode_*.parquet"))
        if max_init_episodes is not None:
            parquet_files = parquet_files[:max_init_episodes]
        for pq in tqdm(parquet_files, desc="Loading episodes"):
            ep_id = pq.stem  # episode_XXXXXX
            try:
                df = pd.read_parquet(pq)
            except Exception as e:
                print(f"⚠️ Failed to read {pq}: {e}")
                continue

            # Validate minimal columns
            if not {"observation.state", "action"}.issubset(df.columns):
                print(f"⚠️ Episode {ep_id} missing required columns")
                continue
            if df.empty:
                print(f"⚠️ Episode {ep_id} empty")
                continue
            episode_data[ep_id] = df
        print(f"✅ Loaded {len(episode_data)} valid parquet episodes")
        return episode_data

    def _get_trajectories(self) -> Tuple[np.ndarray, np.ndarray]:
        traj_ids: List[str] = []
        traj_lengths: List[int] = []
        for ep_id, df in self._episode_data.items():
            traj_ids.append(ep_id)
            traj_lengths.append(max(0, len(df) - self.sequence_length + 1))
        return np.array(traj_ids), np.array(traj_lengths)

    def _get_all_steps(self) -> List[Tuple[str, int]]:
        steps: List[Tuple[str, int]] = []
        for ep_id, length in zip(self._trajectory_ids, self._trajectory_lengths):
            for s in range(0, int(length), self.stride):
                steps.append((ep_id, s))
        return steps

    # ---------- video io ----------
    def _camera_dir(self) -> str:
        return self.CAMERA_MAPPING.get(self.camera, "observation.images.image")

    def _resolve_video_path(self, episode_id: str, key: str) -> Path:
        ep_num = int(episode_id.split("_")[-1])
        chunk_id = ep_num // 1000
        chunk = f"chunk-{chunk_id:03d}"
        cam_dir = self._camera_dir()
        return self.dataset_path / "videos" / chunk / cam_dir / f"{episode_id}.mp4"

    def _resolve_video_path_h264(self, episode_id: str, key: str) -> Path:
        """Resolve H.264 converted video path in a mirrored directory videos_h264."""
        ep_num = int(episode_id.split("_")[-1])
        chunk_id = ep_num // 1000
        chunk = f"chunk-{chunk_id:03d}"
        cam_dir = self._camera_dir()
        return self.dataset_path / "videos_h264" / chunk / cam_dir / f"{episode_id}.mp4"

    def _resolve_video_path_h264_cache(self, episode_id: str, key: str) -> Path:
        """Resolve on-the-fly H.264 cache path under a private cache directory."""
        ep_num = int(episode_id.split("_")[-1])
        chunk_id = ep_num // 1000
        chunk = f"chunk-{chunk_id:03d}"
        cam_dir = self._camera_dir()
        return self.dataset_path / self.h264_cache_dirname / "videos" / chunk / cam_dir / f"{episode_id}.mp4"

    def _ensure_h264_cache(self, src_path: Path, cache_path: Path) -> Optional[Path]:
        """If cache does not exist, convert src to H.264 into cache.
        Returns cache path if available, else None.
        """
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            return cache_path
        # Convert using ffmpeg if available
        try:
            import subprocess
            cmd = [
                "ffmpeg", "-y",
                "-v", "error",
                "-i", str(src_path),
                "-c:v", "libx264",
                "-crf", "23",
                "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                str(cache_path),
            ]
            subprocess.run(cmd, check=True)
            return cache_path if cache_path.exists() else None
        except Exception:
            return None

    def _load_video_frames_pyav(self, video_path: Path) -> Optional[np.ndarray]:
        import av
        # Reduce threads to improve stability on some AV1 streams
        try:
            container = av.open(str(video_path), options={"threads": "1"})
        except Exception:
            return None
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        frames: List[np.ndarray] = []
        try:
            for frame in container.decode(stream):
                img = frame.to_rgb().to_ndarray()
                if self.img_resize:
                    img = cv2.resize(img, self.img_resize)
                frames.append(img)
        except Exception:
            # PyAV failed
            frames = []
        container.close()
        if not frames:
            return None
        return np.stack(frames, axis=0)

    def _load_video_frames(self, video_path: Path) -> Optional[np.ndarray]:
        if self.video_backend == "pyav":
            frames = self._load_video_frames_pyav(video_path)
            if frames is not None:
                return frames
            # Fallback to OpenCV if PyAV failed
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            frames: List[np.ndarray] = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.img_resize:
                    frame = cv2.resize(frame, self.img_resize)
                frames.append(frame)
            cap.release()
            if not frames:
                return None
            return np.stack(frames, axis=0)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        frames: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.img_resize:
                frame = cv2.resize(frame, self.img_resize)
            frames.append(frame)
        cap.release()
        if not frames:
            return None
        return np.stack(frames, axis=0)

    def _cache_all_videos(self) -> None:
        print("🎬 Caching videos (OXE Fractal)...")
        videos_dir = self.dataset_path / "videos"
        if not videos_dir.exists():
            print("⚠️ No videos directory found")
            return
        cam_dir = self._camera_dir()
        video_files: List[Path] = []
        for chunk_dir in sorted(videos_dir.glob("chunk-*")):
            cpath = chunk_dir / cam_dir
            if cpath.exists():
                video_files.extend(sorted(cpath.glob("episode_*.mp4")))
        if not video_files:
            print("⚠️ No video files found")
            return
        all_frames: List[np.ndarray] = []
        episode_lengths: List[int] = []
        for vf in tqdm(video_files, desc="Caching videos"):
            frames = self._load_video_frames(vf)
            if frames is not None and len(frames) > 0:
                all_frames.append(frames)
                episode_lengths.append(len(frames))
        if all_frames:
            key = f"video.{self.camera}"
            self.cached_frames[key] = np.concatenate(all_frames, axis=0)
            self.start_indices = np.cumsum([0] + episode_lengths[:-1])
            print(f"✅ Cached {len(all_frames)} episodes, total frames: {len(self.cached_frames[key])}")

    # ---------- public api ----------
    def __len__(self) -> int:
        return len(self._all_steps)

    @property
    def trajectory_ids(self) -> np.ndarray:
        """Array of trajectory (episode) IDs."""
        return self._trajectory_ids

    @property
    def trajectory_lengths(self) -> np.ndarray:
        """Array of trajectory lengths (valid sequence count per episode)."""
        return self._trajectory_lengths

    @property
    def all_steps(self) -> List[Tuple[str, int]]:
        """List of (trajectory_id, step_index) tuples for all valid steps."""
        return self._all_steps

    def get_episode_data(self, episode_id: str) -> pd.DataFrame:
        if self.curr_traj_id == episode_id and self.curr_traj_data is not None:
            return self.curr_traj_data
        df = self._episode_data[episode_id]
        self.curr_traj_id = episode_id
        self.curr_traj_data = df
        return df

    def get_video_data(self, episode_id: str, key: str, base_index: int) -> np.ndarray:
        step_indices = self._delta_indices[key] + base_index
        cache_key = f"video.{self.camera}"
        if self.cache_videos and cache_key in self.cached_frames:
            ep_idx = list(self._trajectory_ids).index(episode_id)
            start_idx = self.start_indices[ep_idx]
            absolute = start_idx + step_indices
            return self.cached_frames[cache_key][absolute]
        # Prefer H.264 converted video if available
        video_path = self._resolve_video_path(episode_id, key)
        if self.enable_h264_fallback:
            # 1) Pre-converted videos_h264
            h264_path = self._resolve_video_path_h264(episode_id, key)
            if h264_path.exists():
                video_path = h264_path
            # 2) On-the-fly cache for training (does not modify originals)
            elif self.use_on_the_fly_h264:
                cache_path = self._resolve_video_path_h264_cache(episode_id, key)
                cached = self._ensure_h264_cache(video_path, cache_path)
                if cached is not None and cached.exists():
                    video_path = cached
        frames = self._load_video_frames(video_path)
        if frames is None or len(frames) == 0:
            if self.skip_video_on_error:
                # Return zero frames to skip effect while keeping batch shape consistent
                num = len(step_indices)
                # Heuristic size when unknown
                w, h = (self.img_resize if self.img_resize else (224, 224))
                zeros = np.zeros((num, h, w, 3), dtype=np.uint8)
                print(f"⚠️ Skipping video {video_path} (unreadable). Returning zeros.")
                return zeros
            raise FileNotFoundError(f"Cannot load video for {episode_id}")
        step_indices = np.clip(step_indices, 0, len(frames) - 1)
        return frames[step_indices]

    def get_state_action_data(self, episode_id: str, modality: str, key: str, base_index: int) -> np.ndarray:
        df = self.get_episode_data(episode_id)
        step_indices = self._delta_indices[key] + base_index
        column = "observation.state" if modality == "state" else "action"
        data_series = df[column]
        data_array = np.vstack(data_series.values)
        max_len = len(data_array)
        out: List[np.ndarray] = []
        for idx in step_indices:
            if idx < 0:
                out.append(data_array[0])
            elif idx >= max_len:
                out.append(data_array[-1])
            else:
                out.append(data_array[idx])
        return np.array(out)

    def get_language_data(self, episode_id: str, key: str, base_index: int) -> List[str]:
        df = self.get_episode_data(episode_id)
        step_indices = self._delta_indices[key] + base_index
        task_value = ""
        if "task_index" in df.columns and not df["task_index"].empty:
            task_value = self.task_idx_mapping.get(str(df["task_index"].iloc[0]), "")
        return [task_value] * len(step_indices)

    def get_step_data(self, episode_id: str, base_index: int) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for modality in self._modality_keys:
            for key in self._modality_keys[modality]:
                if modality == "video":
                    data[key] = self.get_video_data(episode_id, key, base_index)
                elif modality in ("state", "action"):
                    data[key] = self.get_state_action_data(episode_id, modality, key, base_index)
                elif modality == "language":
                    data[key] = self.get_language_data(episode_id, key, base_index)
                else:
                    raise ValueError(f"Unknown modality {modality}")
        return data

    def __getitem__(self, index: int) -> Dict[str, Any]:
        episode_id, base_index = self._all_steps[index]
        return self.get_step_data(episode_id, base_index)


class OXEFractalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 8,
        num_workers: int = 4,
        sequence_length: int = 2,
        stride: int = 1,
        camera: str = "images.image",
        cache_videos: bool = False,
        img_resize: Optional[Tuple[int, int]] = None,
        train_split: float = 0.8,
        val_split: float = 0.2,
        enable_h264_fallback: bool = True,
        video_backend: str = "pyav",
        resolve_snapshot: bool = True,
        max_init_episodes: Optional[int] = None,
        skip_video_on_error: bool = True,
        use_on_the_fly_h264_train: bool = True,
        use_on_the_fly_h264_val: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sequence_length = sequence_length
        self.stride = stride
        self.camera = camera
        self.cache_videos = cache_videos
        self.img_resize = img_resize
        self.train_split = train_split
        self.val_split = val_split
        self.enable_h264_fallback = enable_h264_fallback
        self.video_backend = video_backend
        self.resolve_snapshot = resolve_snapshot
        self.max_init_episodes = max_init_episodes
        self.skip_video_on_error = skip_video_on_error
        self.use_on_the_fly_h264_train = use_on_the_fly_h264_train
        self.use_on_the_fly_h264_val = use_on_the_fly_h264_val

        self.modality_configs = {
            "video": OXEFractalModalityConfig(
                delta_indices=list(range(sequence_length)),
                modality_keys=[f"video.{camera}"]
            ),
            "state": OXEFractalModalityConfig(
                delta_indices=list(range(sequence_length)),
                modality_keys=["state.joint_positions"]
            ),
            "action": OXEFractalModalityConfig(
                delta_indices=list(range(sequence_length)),
                modality_keys=["action.joint_positions"]
            ),
            "language": OXEFractalModalityConfig(
                delta_indices=[0],
                modality_keys=["annotation.task"]
            ),
        }

    def setup(self, stage: Optional[str] = None) -> None:
        full = OXEFractalDataset(
            dataset_path=self.dataset_path,
            modality_configs=self.modality_configs,
            camera=self.camera,
            sequence_length=self.sequence_length,
            stride=self.stride,
            cache_videos=self.cache_videos,
            img_resize=self.img_resize,
            enable_h264_fallback=self.enable_h264_fallback,
            video_backend=self.video_backend,
            resolve_snapshot=self.resolve_snapshot,
            max_init_episodes=self.max_init_episodes,
            skip_video_on_error=self.skip_video_on_error,
            use_on_the_fly_h264=False,
        )

        total_episodes = len(full._trajectory_ids)
        train_size = int(total_episodes * self.train_split)
        train_episodes = full._trajectory_ids[:train_size]
        val_episodes = full._trajectory_ids[train_size:]

        def filter_steps(target_episodes: np.ndarray) -> List[Tuple[str, int]]:
            target = set(target_episodes.tolist())
            return [(ep, step) for ep, step in full._all_steps if ep in target]

        self.train_dataset = OXEFractalDataset(
            dataset_path=self.dataset_path,
            modality_configs=self.modality_configs,
            camera=self.camera,
            sequence_length=self.sequence_length,
            stride=self.stride,
            cache_videos=self.cache_videos,
            img_resize=self.img_resize,
            enable_h264_fallback=self.enable_h264_fallback,
            video_backend=self.video_backend,
            resolve_snapshot=self.resolve_snapshot,
            max_init_episodes=self.max_init_episodes,
            skip_video_on_error=self.skip_video_on_error,
            use_on_the_fly_h264=self.use_on_the_fly_h264_train,
        )
        self.train_dataset._all_steps = filter_steps(train_episodes)

        self.val_dataset = OXEFractalDataset(
            dataset_path=self.dataset_path,
            modality_configs=self.modality_configs,
            camera=self.camera,
            sequence_length=self.sequence_length,
            stride=self.stride,
            cache_videos=self.cache_videos,
            img_resize=self.img_resize,
            enable_h264_fallback=self.enable_h264_fallback,
            video_backend=self.video_backend,
            resolve_snapshot=self.resolve_snapshot,
            max_init_episodes=self.max_init_episodes,
            skip_video_on_error=self.skip_video_on_error,
            use_on_the_fly_h264=self.use_on_the_fly_h264_val,
        )
        self.val_dataset._all_steps = filter_steps(val_episodes)

        print(f"📊 OXE Fractal split: {len(self.train_dataset)} train, {len(self.val_dataset)} val")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        collated: Dict[str, Any] = {}
        for key in batch[0].keys():
            values = [sample[key] for sample in batch]
            if isinstance(values[0], np.ndarray):
                collated[key] = torch.from_numpy(np.stack(values)).float()
            elif isinstance(values[0], list):
                collated[key] = values
            else:
                try:
                    collated[key] = torch.stack([torch.tensor(v) for v in values])
                except Exception:
                    collated[key] = values
        return collated


