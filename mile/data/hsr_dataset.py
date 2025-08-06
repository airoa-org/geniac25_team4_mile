"""
HSR Robot Dataset implementation following LeRobot style.
Provides robust data loading with metadata-driven configuration.
"""

import json
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pydantic import BaseModel, ValidationError
from tqdm import tqdm

# Suppress OpenCV/FFmpeg warnings for AV1 codec issues
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*OpenCV.*')
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'


class HSRModalityConfig(BaseModel):
    """Configuration for HSR modalities."""
    delta_indices: List[int]
    modality_keys: List[str]


class HSRVideoMetadata(BaseModel):
    """Video metadata for HSR dataset."""
    resolution: Tuple[int, int]  # (width, height)
    channels: int
    fps: float
    codec: str


class HSRStateActionMetadata(BaseModel):
    """State/Action metadata for HSR dataset."""
    shape: List[int]
    continuous: bool
    absolute: bool
    dtype: str
    start: int
    end: int


class HSRDatasetMetadata(BaseModel):
    """Complete HSR dataset metadata."""
    dataset_name: str
    total_episodes: int
    total_steps: int
    modalities: Dict[str, Dict[str, Any]]
    statistics: Dict[str, Dict[str, Dict[str, List[float]]]]
    embodiment_tag: str


class HSRRobotDataset(Dataset):
    """
    HSR Robot Dataset following LeRobot architecture.
    Supports video caching, robust error handling, and metadata-driven configuration.
    """
    
    # Centralized camera mapping
    CAMERA_MAPPING = {
        "head_rgbd_sensor": "observation.image.head",
        "hand_camera": "observation.image.hand",
    }
    
    def __init__(
        self,
        dataset_path: str,
        modality_configs: Dict[str, HSRModalityConfig],
        embodiment_tag: str = "hsr_robot",
        video_backend: str = "opencv",
        cache_videos: bool = False,
        img_resize: Optional[Tuple[int, int]] = None,
        sequence_length: int = 2,
        stride: int = 1,
        camera: str = "head_rgbd_sensor",
        enable_h264_fallback: bool = True,
        skip_video_on_error: bool = False,
    ):
        """
        Initialize HSR Robot Dataset.
        
        Args:
            dataset_path: Path to HSR dataset root
            modality_configs: Configuration for each modality
            embodiment_tag: Tag for embodiment type
            video_backend: Backend for video reading ('opencv' or 'decord')
            cache_videos: Whether to cache all video frames in memory
            img_resize: Resize videos to (width, height) to save memory
            sequence_length: Length of sequences to return
            stride: Stride between sequences
            camera: Camera name for video data
            enable_h264_fallback: Try H264 converted videos if AV1 fails
            skip_video_on_error: Skip video data instead of failing on errors
        """
        self.dataset_path = Path(dataset_path)
        self.modality_configs = modality_configs
        self.embodiment_tag = embodiment_tag
        self.video_backend = video_backend
        self.cache_videos = cache_videos
        self.img_resize = img_resize
        self.sequence_length = sequence_length
        self.stride = stride
        self.camera = camera
        self.enable_h264_fallback = enable_h264_fallback
        self.skip_video_on_error = skip_video_on_error
        
        # Validate dataset exists
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
        
        # Initialize core components
        self._metadata = self._load_or_create_metadata()
        self._episode_data = self._load_episode_data()
        self._trajectory_ids, self._trajectory_lengths = self._get_trajectories()
        self._all_steps = self._get_all_steps()
        self._modality_keys = self._get_modality_keys()
        self._delta_indices = self._get_delta_indices()
        self.task_idx_mapping = self._get_task_idx_mapping()
        
        # Video caching
        self.cached_frames: Dict[str, np.ndarray] = {}
        self.start_indices: Optional[np.ndarray] = None
        if self.cache_videos:
            self._cache_all_videos()
        
        # Current trajectory cache
        self.curr_traj_data = None
        self.curr_traj_id = None
        
        print(f"✅ Initialized HSR dataset: {len(self)} steps, {len(self._trajectory_ids)} episodes")

    @property
    def metadata(self) -> HSRDatasetMetadata:
        """Dataset metadata."""
        return self._metadata

    @property
    def trajectory_ids(self) -> np.ndarray:
        """Array of trajectory IDs."""
        return self._trajectory_ids

    @property
    def trajectory_lengths(self) -> np.ndarray:
        """Array of trajectory lengths."""
        return self._trajectory_lengths

    @property
    def all_steps(self) -> List[Tuple[str, int]]:
        """List of (trajectory_id, step_index) for all valid steps."""
        return self._all_steps

    @property
    def modality_keys(self) -> Dict[str, List[str]]:
        """Modality keys mapping."""
        return self._modality_keys

    @property
    def delta_indices(self) -> Dict[str, np.ndarray]:
        """Delta indices for temporal sampling."""
        return self._delta_indices

    def _load_or_create_metadata(self) -> HSRDatasetMetadata:
        """Load existing metadata or create new one."""
        metadata_path = self.dataset_path / "hsr_metadata.json"
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata_dict = json.load(f)
                return HSRDatasetMetadata(**metadata_dict)
            except Exception as e:
                print(f"⚠️  Failed to load metadata: {e}")
                print("🔄 Creating new metadata...")
        
        return self._create_metadata()
    
    def _get_task_idx_mapping(self) -> Dict[str, int]:
        """Get task index mapping."""
        task_path = self.dataset_path / "meta" / "tasks.jsonl"
        task_dict = {}
        with open(task_path, "r", encoding="utf-8") as f:   # ← ファイル名を適宜
            for line in f:
                obj = json.loads(line)          # 1行＝1オブジェクト
                task_dict[obj["task_index"]] = obj["task"]
        return task_dict

    def _create_metadata(self) -> HSRDatasetMetadata:
        """Create metadata by analyzing the dataset."""
        print("📊 Analyzing dataset structure...")
        
        # Find all parquet files
        data_dir = self.dataset_path / "data"
        parquet_files = list(data_dir.glob("**/episode_*.parquet"))
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")
        
        # Analyze sample episodes
        sample_files = parquet_files[:min(5, len(parquet_files))]
        all_data = []
        
        for parquet_file in sample_files:
            try:
                df = pd.read_parquet(parquet_file)
                all_data.append(df)
            except Exception as e:
                print(f"⚠️  Failed to read {parquet_file}: {e}")
        
        if not all_data:
            raise RuntimeError("Failed to read any parquet files")
        
        # Combine data for statistics
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Create metadata structure
        # Use proper video key format that matches modality configs
        video_key = f"video.{self.camera}"
        modalities = {
            "video": {
                video_key: {
                    "resolution": [224, 224],  # Default, will be updated from actual videos
                    "channels": 3,
                    "fps": 30.0,
                    "codec": "unknown"
                }
            },
            "state": {},
            "action": {},
            "language": {
                "annotation.task": {
                    "shape": [1],
                    "dtype": "str"
                }
            }
        }
        
        # Analyze joint states and actions
        if 'observation.state' in combined_df.columns:
            joint_dim = len(combined_df['observation.state'].iloc[0])
            modalities["state"]["state.joint_positions"] = {
                "shape": [joint_dim],
                "continuous": True,
                "absolute": True,
                "dtype": "float32",
                "start": 0,
                "end": joint_dim
            }
        
        if 'action' in combined_df.columns:
            action_dim = len(combined_df['action'].iloc[0])
            modalities["action"]["action.joint_positions"] = {
                "shape": [action_dim],
                "continuous": True,
                "absolute": False,
                "dtype": "float32",
                "start": 0,
                "end": action_dim
            }
        
        # Calculate statistics
        statistics = self._calculate_statistics(all_data)
        
        metadata = HSRDatasetMetadata(
            dataset_name=self.dataset_path.name,
            total_episodes=len(parquet_files),
            total_steps=sum(len(df) for df in all_data),
            modalities=modalities,
            statistics=statistics,
            embodiment_tag=self.embodiment_tag
        )
        
        # Save metadata
        metadata_path = self.dataset_path / "hsr_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata.dict(), f, indent=2)
        
        print(f"💾 Saved metadata to {metadata_path}")
        return metadata

    def _calculate_statistics(self, data_list: List[pd.DataFrame]) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
        """Calculate dataset statistics."""
        statistics = {}
        
        # Combine all data
        combined_df = pd.concat(data_list, ignore_index=True)
        
        # Joint states statistics
        if 'observation.state' in combined_df.columns:
            joint_data = np.vstack(combined_df['observation.state'].values)
            statistics["state"] = {
                "state.joint_positions": {
                    "mean": np.mean(joint_data, axis=0).tolist(),
                    "std": np.std(joint_data, axis=0).tolist(),
                    "min": np.min(joint_data, axis=0).tolist(),
                    "max": np.max(joint_data, axis=0).tolist(),
                    "q01": np.quantile(joint_data, 0.01, axis=0).tolist(),
                    "q99": np.quantile(joint_data, 0.99, axis=0).tolist(),
                }
            }
        
        # Action statistics
        if 'action' in combined_df.columns:
            action_data = np.vstack(combined_df['action'].values)
            statistics["action"] = {
                "action.joint_positions": {
                    "mean": np.mean(action_data, axis=0).tolist(),
                    "std": np.std(action_data, axis=0).tolist(),
                    "min": np.min(action_data, axis=0).tolist(),
                    "max": np.max(action_data, axis=0).tolist(),
                    "q01": np.quantile(action_data, 0.01, axis=0).tolist(),
                    "q99": np.quantile(action_data, 0.99, axis=0).tolist(),
                }
            }
        
        return statistics

    def _load_episode_data(self) -> Dict[str, pd.DataFrame]:
        """Load all episode data."""
        print("📂 Loading episode data...")
        episode_data = {}
        
        data_dir = self.dataset_path / "data"
        parquet_files = sorted(data_dir.glob("**/episode_*.parquet"))
        
        failed_episodes = []
        
        for parquet_file in tqdm(parquet_files, desc="Loading episodes"):
            episode_id = parquet_file.stem  # e.g., "episode_000001"
            try:
                df = pd.read_parquet(parquet_file)
                
                # Validate required columns
                required_columns = ['observation.state', 'action']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    print(f"⚠️  Episode {episode_id} missing columns: {missing_columns}")
                    failed_episodes.append(episode_id)
                    continue
                
                # Check if data is not empty
                if df.empty:
                    print(f"⚠️  Episode {episode_id} is empty")
                    failed_episodes.append(episode_id)
                    continue
                
                # Validate observation.state format
                if not df['observation.state'].empty:
                    try:
                        first_state = df['observation.state'].iloc[0]
                        if not isinstance(first_state, (list, np.ndarray)):
                            print(f"⚠️  Episode {episode_id} has invalid observation.state format: {type(first_state)}")
                            failed_episodes.append(episode_id)
                            continue
                    except Exception as e:
                        print(f"⚠️  Episode {episode_id} observation.state validation failed: {e}")
                        failed_episodes.append(episode_id)
                        continue
                
                # Validate action format
                if not df['action'].empty:
                    try:
                        first_action = df['action'].iloc[0]
                        if not isinstance(first_action, (list, np.ndarray)):
                            print(f"⚠️  Episode {episode_id} has invalid action format: {type(first_action)}")
                            failed_episodes.append(episode_id)
                            continue
                    except Exception as e:
                        print(f"⚠️  Episode {episode_id} action validation failed: {e}")
                        failed_episodes.append(episode_id)
                        continue
                
                episode_data[episode_id] = df
                
            except Exception as e:
                print(f"❌ Failed to load {episode_id}: {e}")
                failed_episodes.append(episode_id)
        
        if failed_episodes:
            print(f"⚠️  Excluded {len(failed_episodes)} invalid episodes")
            print(f"   First 10 failed episodes: {failed_episodes[:10]}")
        
        print(f"✅ Loaded {len(episode_data)} valid episodes")
        return episode_data

    def _get_trajectories(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get trajectory IDs and lengths."""
        trajectory_ids = []
        trajectory_lengths = []
        
        for episode_id, df in self._episode_data.items():
            trajectory_ids.append(episode_id)
            # Adjust length for sequence sampling
            valid_length = max(0, len(df) - self.sequence_length + 1)
            trajectory_lengths.append(valid_length)
        
        return np.array(trajectory_ids), np.array(trajectory_lengths)

    def _get_all_steps(self) -> List[Tuple[str, int]]:
        """Get all valid (trajectory_id, step_index) pairs."""
        all_steps = []
        
        for trajectory_id, trajectory_length in zip(self.trajectory_ids, self.trajectory_lengths):
            for step_idx in range(0, trajectory_length, self.stride):
                all_steps.append((trajectory_id, step_idx))
        
        return all_steps

    def _get_modality_keys(self) -> Dict[str, List[str]]:
        """Get modality keys mapping."""
        modality_keys = defaultdict(list)
        for modality, config in self.modality_configs.items():
            modality_keys[modality] = config.modality_keys
        return dict(modality_keys)

    def _get_delta_indices(self) -> Dict[str, np.ndarray]:
        """Get delta indices for temporal sampling."""
        delta_indices = {}
        for config in self.modality_configs.values():
            for key in config.modality_keys:
                delta_indices[key] = np.array(config.delta_indices)
        return delta_indices

    def _cache_all_videos(self):
        """Cache all video frames in memory."""
        print("🎬 Caching video frames...")
        
        videos_dir = self.dataset_path / "videos"
        if not videos_dir.exists():
            print("⚠️  No videos directory found")
            return
        
        # Use centralized camera mapping
        camera_dir = self.CAMERA_MAPPING.get(self.camera, "observation.image.head")
        
        # Find all video files in chunk directories
        video_files = []
        for chunk_dir in sorted(videos_dir.glob("chunk-*")):
            camera_path = chunk_dir / camera_dir
            if camera_path.exists():
                video_files.extend(list(camera_path.glob("episode_*.mp4")))
        
        if not video_files:
            print(f"⚠️  No video files found for camera {self.camera}")
            return
        
        # Sort video files by episode number
        video_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        all_frames = []
        episode_lengths = []
        
        for video_file in tqdm(video_files, desc="Caching videos"):
            try:
                frames = self._load_video_frames(video_file)
                if frames is not None:
                    all_frames.append(frames)
                    episode_lengths.append(len(frames))
                else:
                    print(f"⚠️  Failed to load {video_file}")
            except Exception as e:
                print(f"⚠️  Error loading {video_file}: {e}")
        
        if all_frames:
            # Use the same key format as in modality configs
            video_key = f"video.{self.camera}"
            self.cached_frames[video_key] = np.concatenate(all_frames, axis=0)
            self.start_indices = np.cumsum([0] + episode_lengths[:-1])
            print(f"✅ Cached {len(all_frames)} videos, total frames: {len(self.cached_frames[video_key])}")
        else:
            print("❌ No videos were successfully cached")

    def _load_video_frames(self, video_path: Path) -> Optional[np.ndarray]:
        """Load all frames from a video file. Uses OpenCV by default but can fall back to
        PyAV when AV1 codec is not supported by the OpenCV-embedded FFmpeg.
        """
        # Fast path: explicit PyAV backend
        if self.video_backend == "pyav":
            return self._load_video_frames_pyav(video_path)

        try:
            # Suppress OpenCV errors temporarily
            original_log_level = cv2.getLogLevel()
            cv2.setLogLevel(0)  # Suppress all OpenCV logs
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                cv2.setLogLevel(original_log_level)  # Restore log level
                raise RuntimeError(f"Cannot open video: {video_path}")
            
            frames = []
            frame_count = 0
            max_retries = 5  # Limit retries for corrupted frames
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check if frame is valid
                if frame is None or frame.size == 0:
                    frame_count += 1
                    if frame_count > max_retries:
                        break
                    continue
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize if needed
                if self.img_resize:
                    frame = cv2.resize(frame, self.img_resize)
                
                frames.append(frame)
                frame_count += 1
            
            cap.release()
            cv2.setLogLevel(original_log_level)  # Restore log level
            
            if not frames:
                raise RuntimeError(f"No frames loaded from {video_path}")
            
            return np.array(frames)
            
        except Exception as e:
            # Restore log level in case of exception
            try:
                cv2.setLogLevel(original_log_level)
            except:
                pass
            
            # Check for common video codec issues
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["av1", "decoder", "codec", "pixel format"]):
                raise RuntimeError(f"Video codec issue with {video_path}: {e}")
            else:
                # If the error looks codec-related, automatically retry with PyAV
                if "codec" in error_msg or "av1" in error_msg:
                    try:
                        frames = self._load_video_frames_pyav(video_path)
                        if frames is not None and len(frames) > 0:
                            print(f"✅ Decoded {video_path.name} with PyAV fallback")
                            return frames
                    except Exception as pe:
                        raise RuntimeError(f"OpenCV + PyAV fallback failed for {video_path}: {pe}") from pe
                raise RuntimeError(f"Failed to load video {video_path}: {e}")

    def _load_video_frames_pyav(self, video_path: Path) -> Optional[np.ndarray]:
        """Decode video frames using PyAV (FFmpeg) which has broader codec support
        (e.g. AV1 via libdav1d)."""
        try:
            import av  # PyAV is installed via conda-forge package "av"
            container = av.open(str(video_path))
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            frames = []
            for frame in container.decode(stream):
                img = frame.to_rgb().to_ndarray()
                if self.img_resize:
                    img = cv2.resize(img, self.img_resize)
                frames.append(img)
            container.close()
            if not frames:
                raise RuntimeError(f"No frames decoded via PyAV from {video_path}")
            return np.stack(frames, axis=0)
        except Exception as e:
            raise RuntimeError(f"PyAV decoding error for {video_path}: {e}")

    def get_episode_data(self, episode_id: str) -> pd.DataFrame:
        """Get data for a specific episode."""
        if self.curr_traj_id == episode_id and self.curr_traj_data is not None:
            return self.curr_traj_data
        
        if episode_id not in self._episode_data:
            raise KeyError(f"Episode {episode_id} not found")
        
        self.curr_traj_data = self._episode_data[episode_id]
        self.curr_traj_id = episode_id
        return self.curr_traj_data

    def get_video_data(self, episode_id: str, key: str, base_index: int) -> np.ndarray:
        """Get video data for an episode."""
        step_indices = self.delta_indices[key] + base_index
        
        if self.cache_videos and key in self.cached_frames:
            # Use cached frames - key is already in correct format
            episode_idx = list(self.trajectory_ids).index(episode_id)
            start_idx = self.start_indices[episode_idx]
            absolute_indices = start_idx + step_indices
            return self.cached_frames[key][absolute_indices]
        else:
            # Load frames on demand with fallback strategy
            frames = self._load_video_with_fallback(episode_id, key)
            
            # Ensure indices are within bounds
            step_indices = np.clip(step_indices, 0, len(frames) - 1)
            return frames[step_indices]

    def _load_video_with_fallback(self, episode_id: str, key: str) -> np.ndarray:
        """Load video with AV1 -> H264 fallback strategy."""
        errors = []
        
        # Try original AV1 video first
        video_path = self._resolve_video_path(episode_id, key)
        
        if video_path.exists():
            try:
                frames = self._load_video_frames(video_path)
                if frames is not None and len(frames) > 0:
                    return frames
            except Exception as e:
                errors.append(f"AV1 video ({video_path}): {e}")
                if self.enable_h264_fallback and any(keyword in str(e).lower() 
                                                   for keyword in ["av1", "decoder", "codec", "pixel format"]):
                    print(f"🔄 AV1 failed for {episode_id}, trying H264 fallback...")
                else:
                    raise RuntimeError(f"Failed to load video {video_path}: {e}")
        
        # Try H264 fallback
        if self.enable_h264_fallback:
            h264_path = self._resolve_h264_video_path(episode_id, key)
            if h264_path.exists():
                try:
                    frames = self._load_video_frames(h264_path)
                    if frames is not None and len(frames) > 0:
                        print(f"✅ Successfully loaded H264 fallback for {episode_id}")
                        return frames
                except Exception as e:
                    errors.append(f"H264 video ({h264_path}): {e}")
        
        # No video could be loaded - raise error with all attempts
        error_details = "\n  ".join(errors)
        raise FileNotFoundError(f"No loadable video found for {episode_id}. Attempts:\n  {error_details}")
        
    def _resolve_h264_video_path(self, episode_id: str, key: str) -> Path:
        """Resolve H264 video path for fallback."""
        episode_num = int(episode_id.split('_')[-1])
        chunk_id = episode_num // 1000
        chunk_name = f"chunk-{chunk_id:03d}"
        
        # Extract camera name from key
        camera_name = key.replace("video.", "")
        camera_dir = self.CAMERA_MAPPING.get(camera_name, "observation.image.head")
        
        # H264 converted videos are in _h264 subdirectory
        h264_path = self.dataset_path / "_h264" / "videos" / chunk_name / camera_dir / f"{episode_id}.mp4"
        return h264_path

    def _resolve_video_path(self, episode_id: str, key: str) -> Path:
        """Resolve video path based on chunk structure."""
        # Extract episode number from ID
        episode_num = int(episode_id.split('_')[-1])
        
        # Determine chunk (assuming 1000 episodes per chunk)
        chunk_id = episode_num // 1000
        chunk_name = f"chunk-{chunk_id:03d}"
        
        # Extract camera name from key and use centralized mapping
        camera_name = key.replace("video.", "")
        camera_dir = self.CAMERA_MAPPING.get(camera_name, "observation.image.head")
        
        # Construct video path
        video_path = self.dataset_path / "videos" / chunk_name / camera_dir / f"{episode_id}.mp4"
        
        return video_path

    def get_state_action_data(self, episode_id: str, modality: str, key: str, base_index: int) -> np.ndarray:
        """Get state or action data for an episode."""
        episode_data = self.get_episode_data(episode_id)
        step_indices = self.delta_indices[key] + base_index
        
        # Map key to column name
        if modality == "state" and "joint_positions" in key:
            column_name = "observation.state"
        elif modality == "action" and "joint_positions" in key:
            column_name = "action"
        else:
            raise ValueError(f"Unknown key {key} for modality {modality}")
        
        # Check if column exists
        if column_name not in episode_data.columns:
            available_columns = list(episode_data.columns)
            raise KeyError(f"Column '{column_name}' not found in episode {episode_id}. "
                          f"Available columns: {available_columns}")
        
        # Get data and validate
        try:
            data_series = episode_data[column_name]
            if data_series.empty:
                raise ValueError(f"Column '{column_name}' is empty in episode {episode_id}")
            
            # Check if data is list/array format
            first_item = data_series.iloc[0]
            if not isinstance(first_item, (list, np.ndarray)):
                raise ValueError(f"Column '{column_name}' contains invalid data type in episode {episode_id}: {type(first_item)}")
            
            data_array = np.vstack(data_series.values)
            
            if data_array.size == 0:
                raise ValueError(f"No valid data in column '{column_name}' for episode {episode_id}")
            
            # Check for NaN or infinite values
            if np.any(np.isnan(data_array)) or np.any(np.isinf(data_array)):
                raise ValueError(f"NaN or infinite values found in column '{column_name}' for episode {episode_id}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to process '{column_name}' data for episode {episode_id}: {e}")
        
        # Handle out-of-bounds indices with padding
        max_length = len(data_array)
        padded_data = []
        
        for idx in step_indices:
            if idx < 0:
                padded_data.append(data_array[0])  # First frame padding
            elif idx >= max_length:
                padded_data.append(data_array[-1])  # Last frame padding
            else:
                padded_data.append(data_array[idx])
        
        return np.array(padded_data)

    def get_language_data(self, episode_id: str, key: str, base_index: int) -> List[str]:
        """Get language annotation data for an episode."""
        episode_data = self.get_episode_data(episode_id)
        step_indices = self.delta_indices[key] + base_index
        
        # Check for various possible column names for task annotation
        task_column = "task_index"
        task_value = ""
        
        if task_column in episode_data.columns:
            task_value = self.task_idx_mapping[episode_data[task_column].iloc[0]]
        
        return [task_value] * len(step_indices)

    def get_step_data(self, episode_id: str, base_index: int) -> Dict[str, Any]:
        """Get all data for a single step."""
        data = {}
        
        for modality in self.modality_keys:
            for key in self.modality_keys[modality]:
                if modality == "video":
                    data[key] = self.get_video_data(episode_id, key, base_index)
                elif modality in ["state", "action"]:
                    data[key] = self.get_state_action_data(episode_id, modality, key, base_index)
                elif modality == "language":
                    data[key] = self.get_language_data(episode_id, key, base_index)
                else:
                    raise ValueError(f"Unknown modality: {modality}")
        
        return data

    def __len__(self) -> int:
        """Total number of valid sequences."""
        return len(self.all_steps)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a single data sample."""
        episode_id, base_index = self.all_steps[index]
        
        try:
            return self.get_step_data(episode_id, base_index)
        except Exception as e:
            # Detailed error for debugging
            raise RuntimeError(f"Failed to load sample {index} (episode: {episode_id}, step: {base_index}): {e}")


class HSRDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for HSR data."""
    
    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 8,
        num_workers: int = 4,
        sequence_length: int = 2,
        stride: int = 1,
        camera: str = "head_rgbd_sensor",
        cache_videos: bool = False,
        img_resize: Optional[Tuple[int, int]] = None,
        train_split: float = 0.8,
        val_split: float = 0.2,
        enable_h264_fallback: bool = True,
        skip_video_on_error: bool = False,
        video_backend: str = "opencv",
    ):
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
        self.skip_video_on_error = skip_video_on_error
        self.video_backend = video_backend
        
        # Define modality configurations with proper key naming
        self.modality_configs = {
            "video": HSRModalityConfig(
                delta_indices=list(range(sequence_length)),
                modality_keys=[f"video.{camera}"]
            ),
            "state": HSRModalityConfig(
                delta_indices=list(range(sequence_length)),
                modality_keys=["state.joint_positions"]
            ),
            "action": HSRModalityConfig(
                delta_indices=list(range(sequence_length)),
                modality_keys=["action.joint_positions"]
            ),
            "language": HSRModalityConfig(
                delta_indices=[0],  # Language is typically constant per episode
                modality_keys=["annotation.task"]
            )
        }

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for train/val/test."""
        # Create full dataset
        self.full_dataset = HSRRobotDataset(
            dataset_path=self.dataset_path,
            modality_configs=self.modality_configs,
            cache_videos=self.cache_videos,
            img_resize=self.img_resize,
            sequence_length=self.sequence_length,
            stride=self.stride,
            camera=self.camera,
            enable_h264_fallback=self.enable_h264_fallback,
            skip_video_on_error=self.skip_video_on_error,
            video_backend=self.video_backend,
        )
        
        # Split episodes into train/val
        total_episodes = len(self.full_dataset.trajectory_ids)
        train_size = int(total_episodes * self.train_split)
        
        train_episodes = self.full_dataset.trajectory_ids[:train_size]
        val_episodes = self.full_dataset.trajectory_ids[train_size:]
        
        # Create filtered steps for train/val
        train_steps = [(ep, step) for ep, step in self.full_dataset.all_steps if ep in train_episodes]
        val_steps = [(ep, step) for ep, step in self.full_dataset.all_steps if ep in val_episodes]
        
        # Create train/val datasets by overriding all_steps
        self.train_dataset = HSRRobotDataset(
            dataset_path=self.dataset_path,
            modality_configs=self.modality_configs,
            cache_videos=self.cache_videos,
            img_resize=self.img_resize,
            sequence_length=self.sequence_length,
            stride=self.stride,
            camera=self.camera,
            enable_h264_fallback=self.enable_h264_fallback,
            skip_video_on_error=self.skip_video_on_error,
            video_backend=self.video_backend,
        )
        self.train_dataset._all_steps = train_steps
        
        self.val_dataset = HSRRobotDataset(
            dataset_path=self.dataset_path,
            modality_configs=self.modality_configs,
            cache_videos=self.cache_videos,
            img_resize=self.img_resize,
            sequence_length=self.sequence_length,
            stride=self.stride,
            camera=self.camera,
            enable_h264_fallback=self.enable_h264_fallback,
            skip_video_on_error=self.skip_video_on_error,
            video_backend=self.video_backend,
        )
        self.val_dataset._all_steps = val_steps
        
        print(f"📊 Dataset split: {len(self.train_dataset)} train, {len(self.val_dataset)} val")

    def train_dataloader(self) -> DataLoader:
        """Training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader."""
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
        """Collate function for batching."""
        collated = {}
        
        for key in batch[0].keys():
            values = [sample[key] for sample in batch]
            
            if isinstance(values[0], np.ndarray):
                # Convert numpy arrays to tensors
                collated[key] = torch.from_numpy(np.stack(values)).float()
            elif isinstance(values[0], list):
                # Handle lists (e.g., language data)
                collated[key] = values
            else:
                # Try to stack other types
                try:
                    collated[key] = torch.stack([torch.tensor(v) for v in values])
                except:
                    collated[key] = values
        
        return collated 