#!/usr/bin/env python3
"""
Test script for HSR Dataset v2 with strict error handling (no dummy data).
"""

import argparse
import sys
import traceback
from pathlib import Path

import torch
import numpy as np
from mile.data.hsr_dataset import HSRRobotDataset, HSRDataModule, HSRModalityConfig
from mile.data.oxe_fractal_dataset import (
    OXEFractalDataset,
    OXEFractalDataModule,
    OXEFractalModalityConfig,
)


def test_state_action_only(dataset_path: str, video_backend: str, robot_type: str, camera: str, limit_episodes: int | None):
    """Test dataset with state and action data only (no video)."""
    print("🧪 Testing state and action data only...")
    
    # Define modality config without video
    if robot_type == "oxe_fractal":
        modality_configs = {
            "state": OXEFractalModalityConfig(
                delta_indices=[0, 1],
                modality_keys=["state.joint_positions"],
            ),
            "action": OXEFractalModalityConfig(
                delta_indices=[0, 1],
                modality_keys=["action.joint_positions"],
            ),
            "language": OXEFractalModalityConfig(
                delta_indices=[0],
                modality_keys=["annotation.task"],
            ),
        }
    else:
        modality_configs = {
            "state": HSRModalityConfig(
                delta_indices=[0, 1],
                modality_keys=["state.joint_positions"],
            ),
            "action": HSRModalityConfig(
                delta_indices=[0, 1],
                modality_keys=["action.joint_positions"],
            ),
            "language": HSRModalityConfig(
                delta_indices=[0],
                modality_keys=["annotation.task"],
            ),
        }
    
    try:
        if robot_type == "oxe_fractal":
            dataset = OXEFractalDataset(
                dataset_path=dataset_path,
                modality_configs=modality_configs,
                cache_videos=False,
                sequence_length=2,
                stride=1,
                enable_h264_fallback=True,
                video_backend=video_backend,
                resolve_snapshot=True,
                max_init_episodes=limit_episodes,
            )
        else:
            dataset = HSRRobotDataset(
                dataset_path=dataset_path,
                modality_configs=modality_configs,
                cache_videos=False,
                sequence_length=2,
                stride=1,
                enable_h264_fallback=True,
                skip_video_on_error=False,
                video_backend=video_backend,
            )
        
        print(f"✅ Dataset created successfully!")
        print(f"   Valid episodes: {len(dataset.trajectory_ids)}")
        print(f"   Total steps: {len(dataset)}")
        
        # Test first few samples
        for i in range(min(3, len(dataset))):
            try:
                data = dataset[i]
                print(f"\n📋 Sample {i}:")
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"   {key}: {type(value)} = {value}")
                
            except Exception as e:
                print(f"❌ Sample {i} failed: {e}")
                return False
        
        print("✅ State/action only test passed!")
        return True
        
    except Exception as e:
        print(f"❌ State/action test failed: {e}")
        traceback.print_exc()
        return False


def test_with_h264_video(dataset_path: str, video_backend: str, robot_type: str, camera: str, limit_episodes: int | None):
    """Test dataset with H264 video fallback."""
    print("\n🧪 Testing with H264 video fallback...")
    
    # Full modality config including video
    if robot_type == "oxe_fractal":
        video_key = f"video.{camera}"
        modality_configs = {
            "video": OXEFractalModalityConfig(
                delta_indices=[0, 1],
                modality_keys=[video_key],
            ),
            "state": OXEFractalModalityConfig(
                delta_indices=[0, 1],
                modality_keys=["state.joint_positions"],
            ),
            "action": OXEFractalModalityConfig(
                delta_indices=[0, 1],
                modality_keys=["action.joint_positions"],
            ),
            "language": OXEFractalModalityConfig(
                delta_indices=[0],
                modality_keys=["annotation.task"],
            ),
        }
    else:
        modality_configs = {
            "video": HSRModalityConfig(
                delta_indices=[0, 1],
                modality_keys=["video.head_rgbd_sensor"],
            ),
            "state": HSRModalityConfig(
                delta_indices=[0, 1],
                modality_keys=["state.joint_positions"],
            ),
            "action": HSRModalityConfig(
                delta_indices=[0, 1],
                modality_keys=["action.joint_positions"],
            ),
            "language": HSRModalityConfig(
                delta_indices=[0],
                modality_keys=["annotation.task"],
            ),
        }
    
    try:
        if robot_type == "oxe_fractal":
            dataset = OXEFractalDataset(
                dataset_path=dataset_path,
                modality_configs=modality_configs,
                cache_videos=False,
                sequence_length=2,
                stride=1,
                enable_h264_fallback=True,
                video_backend=video_backend,
                resolve_snapshot=True,
                max_init_episodes=limit_episodes,
            )
        else:
            dataset = HSRRobotDataset(
                dataset_path=dataset_path,
                modality_configs=modality_configs,
                cache_videos=False,
                sequence_length=2,
                stride=1,
                enable_h264_fallback=True,  # Enable H264 fallback
                skip_video_on_error=False,  # No dummy data
                video_backend=video_backend,
            )
        
        print(f"✅ Dataset with video created successfully!")
        print(f"   Valid episodes: {len(dataset.trajectory_ids)}")
        
        # Test first sample with video
        try:
            episode_id = dataset.trajectory_ids[0]
            print(f"   Testing episode: {episode_id}")
            
            data = dataset[0]
            
            print(f"\n📋 Sample 0 with video:")
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
                    if "video" in key:
                        print(f"     Video range: [{value.min():.3f}, {value.max():.3f}]")
                else:
                    print(f"   {key}: {type(value)} = {value}")
            
            print("✅ Video test passed!")
            return True
            
        except Exception as e:
            print(f"❌ Video test failed: {e}")
            print(f"   Error type: {type(e).__name__}")
            if "FileNotFoundError" in str(type(e)):
                print("   → This indicates video file path issues")
            elif "codec" in str(e).lower() or "av1" in str(e).lower():
                print("   → This indicates codec issues")
            return False
            
    except Exception as e:
        print(f"❌ Dataset creation with video failed: {e}")
        traceback.print_exc()
        return False


def test_datamodule_strict(dataset_path: str, video_backend: str, robot_type: str, camera: str, limit_episodes: int | None):
    """Test DataModule with strict error handling."""
    print("\n�� Testing DataModule (strict mode)...")
    
    try:
        if robot_type == "oxe_fractal":
            data_module = OXEFractalDataModule(
                dataset_path=dataset_path,
                batch_size=2,
                num_workers=0,
                sequence_length=2,
                camera=camera,
                cache_videos=False,
                enable_h264_fallback=True,
                video_backend=video_backend,
                resolve_snapshot=True,
                max_init_episodes=limit_episodes,
            )
        else:
            data_module = HSRDataModule(
                dataset_path=dataset_path,
                batch_size=2,  # Small batch for testing
                num_workers=0,  # Avoid multiprocessing in test
                sequence_length=2,
                cache_videos=False,
                enable_h264_fallback=True,
                skip_video_on_error=False,  # Strict mode
                video_backend=video_backend,
            )
        
        data_module.setup()
        
        print(f"✅ DataModule setup successful!")
        print(f"   Train samples: {len(data_module.train_dataset)}")
        print(f"   Val samples: {len(data_module.val_dataset)}")
        
        # Test train dataloader
        train_loader = data_module.train_dataloader()
        
        # Try to get one batch
        try:
            batch = next(iter(train_loader))
            print(f"   ✅ Successfully loaded batch:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"     {key}: {value.shape} ({value.dtype})")
                else:
                    print(f"     {key}: {type(value)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Batch loading failed: {e}")
            print(f"   Error type: {type(e).__name__}")
            return False
        
    except Exception as e:
        print(f"❌ DataModule test failed: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Robot Datasets (HSR / OXE Fractal)")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Path to HSR dataset root")
    parser.add_argument("--robot_type", type=str, default="hsr", choices=["hsr", "oxe_fractal"],
                       help="Dataset type to test")
    parser.add_argument("--camera", type=str, default="images.image",
                       help="Camera key (for fractal: images.image; for hsr: head_rgbd_sensor)")
    parser.add_argument("--test_video", action="store_true",
                       help="Test video loading functionality")
    parser.add_argument("--test_datamodule", action="store_true",
                       help="Test PyTorch Lightning DataModule")
    parser.add_argument("--video_backend", type=str, default="pyav", choices=["opencv", "pyav"],
                       help="Video backend to use (opencv or pyav)")
    parser.add_argument("--limit_episodes", type=int, default=200,
                       help="Limit number of parquet episodes to scan for faster tests (fractal only)")
    
    args = parser.parse_args()
    
    print("🚀 Robot Dataset Test Suite (No Dummy Data)")
    print("=" * 60)
    
    # Validate dataset path
    if not Path(args.data_root).exists():
        print(f"❌ Dataset path not found: {args.data_root}")
        sys.exit(1)
    
    # Check for H264 fallback availability
    h264_path = Path(args.data_root) / "_h264"
    if h264_path.exists():
        print(f"✅ H264 fallback directory found: {h264_path}")
    else:
        print(f"⚠️  H264 fallback directory not found: {h264_path}")
        print("   Video tests may fail if AV1 codec is not supported")
    
    test_results = []
    
    # 1. Test state/action only (safe test)
    success = test_state_action_only(args.data_root, args.video_backend, args.robot_type, args.camera, args.limit_episodes)
    test_results.append(("State/Action Only", success))
    
    # 2. Test with video (if requested)
    if args.test_video:
        success = test_with_h264_video(args.data_root, args.video_backend, args.robot_type, args.camera, args.limit_episodes)
        test_results.append(("Video Loading", success))
    
    # 3. Test DataModule (if requested)
    if args.test_datamodule:
        success = test_datamodule_strict(args.data_root, args.video_backend, args.robot_type, args.camera, args.limit_episodes)
        test_results.append(("DataModule", success))
    
    # Print results summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    for test_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:.<30} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(test_results)} tests passed")
    
    if passed == len(test_results):
        print("🎉 All tests passed! Dataset is ready for strict training.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Check error messages above.")
        print("\n💡 Recommendations:")
        print("  - Ensure H264 converted videos are available")
        print("  - Check for corrupted or missing data files")
        print("  - Validate dataset structure and format")
        sys.exit(1)


if __name__ == "__main__":
    main() 