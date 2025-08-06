#!/usr/bin/env python3
"""
Training script for HSR Robot MILE using the new dataset implementation (v2).
Based on LeRobot architecture with robust error handling and AV1 codec support.
Enhanced with image reconstruction capability following robot_example_with_reconstruction.py
"""

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add the parent directory to sys.path to import mile modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mile.data.hsr_dataset import HSRDataModule, HSRModalityConfig
from mile.models.robot_mile import RobotMile
from mile.models.common import RGBHead
from mile.configs.hsr_config import get_hsr_training_config
from mile.losses import KLLoss


class HSRTrainer:
    """Trainer for HSR Robot MILE with new dataset implementation and reconstruction capability."""
    
    def __init__(
        self,
        config,
        model: RobotMile,
        data_module: HSRDataModule,
        device: torch.device,
        experiment_name: str = "hsr_robot_mile_v2",
        use_wandb: bool = False,
        use_tensorboard: bool = True,
    ):
        self.config = config
        self.model = model.to(device)
        self.data_module = data_module
        self.device = device
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        
        # Setup optimizers
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.TRAIN.LEARNING_RATE,
            weight_decay=config.TRAIN.WEIGHT_DECAY
        )
        
        # Setup loss functions
        self.kl_loss = KLLoss(alpha=config.LOSS.KL_BALANCING_ALPHA)
        
        # Setup logging
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        if self.use_wandb:
            wandb.init(
                project="hsr-robot-mile-v2",
                name=experiment_name,
                config=self._config_to_dict(config)
            )
        
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(f"runs/{experiment_name}")
        
        # Setup checkpointing
        self.checkpoint_dir = Path("checkpoints") / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Trainer initialized: {experiment_name}")
        print(f"   Device: {device}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   WandB logging: {'✅ Enabled' if self.use_wandb else '❌ Disabled'}")
        print(f"   TensorBoard logging: {'✅ Enabled' if self.use_tensorboard else '❌ Disabled'}")


    def _config_to_dict(self, config) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        config_dict = {}
        for attr in dir(config):
            if not attr.startswith('_'):
                value = getattr(config, attr)
                if not callable(value):
                    try:
                        # Try to convert to basic types for logging
                        if hasattr(value, '__dict__'):
                            config_dict[attr] = self._config_to_dict(value)
                        else:
                            config_dict[attr] = value
                    except:
                        config_dict[attr] = str(value)
        return config_dict


    def prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Prepare batch for RobotMile model format."""
        prepared_batch = {}
        
        # Images: convert from (B, T, H, W, C) to (B, T, C, H, W)
        if "video.head_rgbd_sensor" in batch:
            images = batch["video.head_rgbd_sensor"].to(self.device)
            B, T, H, W, C = images.shape
            prepared_batch['image'] = images.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
        
        # Joint states
        if "state.joint_positions" in batch:
            joint_states = batch["state.joint_positions"].to(self.device)
            
            # Debug: log joint states shape on first call
            if not hasattr(self, '_logged_joint_shape'):
                print(f"🔍 Debug: joint_states shape: {joint_states.shape}")
                print(f"🔍 Debug: joint_states dtype: {joint_states.dtype}")
                if joint_states.numel() > 0:
                    print(f"🔍 Debug: joint_states sample: {joint_states[0, 0]}")
                    print(f"🔍 Debug: expected INPUT_DIM from config: {self.config.MODEL.JOINT.INPUT_DIM}")
                self._logged_joint_shape = True
            
            prepared_batch['joint_states'] = joint_states
        
        # Joint actions
        if "action.joint_positions" in batch:
            actions = batch["action.joint_positions"].to(self.device)
            
            # Debug: log action shape on first call
            if not hasattr(self, '_logged_action_shape'):
                print(f"🔍 Debug: actions shape: {actions.shape}")
                print(f"🔍 Debug: actions dtype: {actions.dtype}")
                if actions.numel() > 0:
                    print(f"🔍 Debug: actions sample: {actions[0, 0]}")
                    print(f"🔍 Debug: expected NUM_JOINTS from config: {self.config.MODEL.NUM_JOINTS}")
                self._logged_action_shape = True
            
            prepared_batch['joint_actions'] = actions
        
        # Language instructions - handle properly for LanguageEncoder
        if "annotation.task" in batch:
            # batch["annotation.task"] is a list of lists (B batches, each with T sequence strings)
            # Extract the first string from each batch since task is constant per episode
            text_instructions = []
            annotation_data = batch["annotation.task"]
            
            # Debug: log the structure of annotation data on first call
            if not hasattr(self, '_logged_annotation_structure'):
                print(f"🔍 Debug: annotation_data type: {type(annotation_data)}")
                if isinstance(annotation_data, list) and len(annotation_data) > 0:
                    print(f"🔍 Debug: first annotation type: {type(annotation_data[0])}")
                    print(f"🔍 Debug: first annotation content: {annotation_data[0]}")
                self._logged_annotation_structure = True
            
            # Handle different formats of annotation data
            if isinstance(annotation_data, list):
                for annotation in annotation_data:
                    if isinstance(annotation, list) and len(annotation) > 0:
                        # Take the first task string (task is constant per episode)
                        text_instructions.append(str(annotation[0]) if annotation[0] else "")
                    elif isinstance(annotation, str):
                        text_instructions.append(annotation)
                    else:
                        text_instructions.append("")
            else:
                # Fallback for other formats
                batch_size = len(batch["video.head_rgbd_sensor"]) if "video.head_rgbd_sensor" in batch else 1
                text_instructions = [""] * batch_size
            
            prepared_batch['text_instructions'] = text_instructions
            
            # Debug: log final text instructions on first call
            if not hasattr(self, '_logged_text_instructions'):
                print(f"🔍 Debug: final text_instructions: {text_instructions}")
                self._logged_text_instructions = True
        
        return prepared_batch

    def compute_loss_with_reconstruction(self, output, batch, config):
        """
        Enhanced loss function including image reconstruction
        """
        losses = {}
        
        # Debug: print output keys on first call
        if not hasattr(self, '_logged_output_keys'):
            print(f"🔍 Debug: model output keys: {list(output.keys())}")
            if 'posterior' in output:
                print(f"🔍 Debug: posterior keys: {list(output['posterior'].keys())}")
            if 'prior' in output:
                print(f"🔍 Debug: prior keys: {list(output['prior'].keys())}")
            self._logged_output_keys = True
        
        # Action prediction loss
        if 'joint_actions' in output:
            predicted_actions = output['joint_actions']
            target_actions = batch['joint_actions']
            action_loss = nn.MSELoss()(predicted_actions, target_actions)
            losses['action_loss'] = action_loss
        else:
            losses['action_loss'] = torch.tensor(0.0, device=self.device)
        
        # KL divergence loss - use KLLoss class
        if config.MODEL.TRANSITION.ENABLED and 'posterior' in output and 'prior' in output:
            # Use the proper KLLoss implementation
            posterior = output['posterior']
            prior = output['prior']
            kl_loss = self.kl_loss(prior, posterior)
            losses['kl_loss'] = kl_loss
        else:
            losses['kl_loss'] = torch.tensor(0.0, device=self.device)
        
        # Image reconstruction loss (always computed)
        if 'rgb_1' in output:
            target_images = batch['image']
            predicted_images = output['rgb_1']
            
            # Reshape predicted images for interpolation
            b, s, c, h, w = predicted_images.shape
            predicted_flat = predicted_images.view(b * s, c, h, w)
            target_flat = target_images.view(b * s, target_images.shape[2], 
                                           target_images.shape[3], target_images.shape[4])
            
            # Resize predicted images to match target size
            predicted_resized = nn.functional.interpolate(
                predicted_flat,
                size=(target_images.shape[-2], target_images.shape[-1]),
                mode='bilinear',
                align_corners=False
            )
            
            # Reshape back to (b, s, c, h, w)
            predicted_resized = predicted_resized.view(b, s, target_images.shape[2], 
                                                     target_images.shape[3], target_images.shape[4])
            
            reconstruction_loss = nn.MSELoss()(predicted_resized, target_images)
            losses['reconstruction_loss'] = reconstruction_loss
        else:
            losses['reconstruction_loss'] = torch.tensor(0.0, device=self.device)
        
        # Total loss with weights
        total_loss = (config.LOSS.ACTION_WEIGHT * losses['action_loss'] + 
                      config.LOSS.KL_WEIGHT * losses['kl_loss'] +
                      config.LOSS.RECONSTRUCTION_WEIGHT * losses['reconstruction_loss'])
        losses['total_loss'] = total_loss
        
        return losses

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {}
        num_batches = 0
        
        train_loader = self.data_module.train_dataloader()
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        for batch in pbar:
            self.optimizer.zero_grad()
            
            try:
                # Prepare batch for RobotMile format
                prepared_batch = self.prepare_batch(batch)
                
                # Forward pass
                output = self.model(prepared_batch, deployment=False)
                
                # Compute losses
                losses = self.compute_loss_with_reconstruction(output, prepared_batch, self.config)
                
                # Backward pass
                losses["total_loss"].backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.TRAIN.GRADIENT_CLIP
                )
                
                # Optimizer step
                self.optimizer.step()
                
                # Accumulate losses
                for key, value in losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0.0
                    epoch_losses[key] += value.item()
                
                num_batches += 1
                self.global_step += 1
                
                # Log to wandb/tensorboard every 10 steps
                if self.global_step % 10 == 0:  # Log every 10 steps
                    print(f"📊 Logging step {self.global_step}...")  # Debug print
                    self._log_losses(losses, "train")
                
                # Update progress bar
                pbar.set_postfix({
                    "total_loss": f"{losses['total_loss'].item():.4f}",
                    "action_loss": f"{losses.get('action_loss', torch.tensor(0.0)).item():.4f}",
                    "kl_loss": f"{losses.get('kl_loss', torch.tensor(0.0)).item():.4f}",
                    "recon_loss": f"{losses.get('reconstruction_loss', torch.tensor(0.0)).item():.4f}"
                })
                
            except Exception as e:
                print(f"⚠️  Error in training step: {e}")
                traceback.print_exc()
                continue
        
        # Average losses
        if num_batches > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
        
        return epoch_losses

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_losses = {}
        num_batches = 0
        
        val_loader = self.data_module.val_dataloader()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                try:
                    # Prepare batch for RobotMile format
                    prepared_batch = self.prepare_batch(batch)
                    
                    # Forward pass
                    output = self.model(prepared_batch, deployment=False)
                    
                    # Compute losses
                    losses = self.compute_loss_with_reconstruction(output, prepared_batch, self.config)
                    
                    # Accumulate losses
                    for key, value in losses.items():
                        if key not in epoch_losses:
                            epoch_losses[key] = 0.0
                        epoch_losses[key] += value.item()
                    
                    num_batches += 1
                    
                except Exception as e:
                    print(f"⚠️  Error in validation step: {e}")
                    continue
        
        # Average losses
        if num_batches > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
        
        return epoch_losses

    def _log_losses(self, losses: Dict[str, torch.Tensor], phase: str):
        """Log losses to wandb and tensorboard."""
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            
            if self.use_wandb:
                wandb.log({f"{phase}/{key}": value}, step=self.global_step)
                print(f"🔄 WandB: {phase}/{key}={value:.6f} @ step {self.global_step}")  # Debug
               
            
            if self.use_tensorboard:
                self.tb_writer.add_scalar(f"{phase}/{key}", value, self.global_step)

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best checkpoint: {best_path}")
        
        # Save epoch checkpoint
        if self.epoch % 10 == 0:  # Save every 10 epochs
            epoch_path = self.checkpoint_dir / f"epoch_{self.epoch:04d}.pt"
            torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.best_val_loss = checkpoint['best_val_loss']
            
            print(f"✅ Loaded checkpoint from epoch {self.epoch}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return False

    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """Main training loop."""
        # Resume from checkpoint if specified
        if resume_from and Path(resume_from).exists():
            self.load_checkpoint(resume_from)
        
        print(f"🚀 Starting training for {num_epochs} epochs...")
        print(f"   Starting epoch: {self.epoch}")
        print(f"   Total steps: {self.global_step}")
        
        start_time = time.time()
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            print(f"\n📊 Epoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Training phase
            train_losses = self.train_epoch()
            
            # Validation phase
            val_losses = self.validate_epoch()
            
            # Log epoch results
            print(f"\n📈 Epoch {epoch + 1} Results:")
            print(f"   Train Loss: {train_losses.get('total_loss', 0.0):.6f}")
            print(f"   Val Loss:   {val_losses.get('total_loss', 0.0):.6f}")
            
            if 'action_loss' in train_losses:
                print(f"   Action Loss: {train_losses['action_loss']:.6f} -> {val_losses.get('action_loss', 0.0):.6f}")
            
            if 'kl_loss' in train_losses:
                print(f"   KL Loss:     {train_losses['kl_loss']:.6f} -> {val_losses.get('kl_loss', 0.0):.6f}")
            
            if 'reconstruction_loss' in train_losses:
                print(f"   Recon Loss:  {train_losses['reconstruction_loss']:.6f} -> {val_losses.get('reconstruction_loss', 0.0):.6f}")
            
            # Log to wandb/tensorboard
            print(f"📊 Logging epoch {epoch + 1} results...")  # Debug print
            self._log_losses(train_losses, "train_epoch")
            self._log_losses(val_losses, "val_epoch")
            
            # Check for best model
            val_loss = val_losses.get('total_loss', float('inf'))
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"🏆 New best validation loss: {val_loss:.6f}")
            
            # Save checkpoint
            self.save_checkpoint(is_best=is_best)
        
        total_time = time.time() - start_time
        print(f"\n🎉 Training completed!")
        print(f"   Total time: {total_time / 3600:.2f} hours")
        print(f"   Best val loss: {self.best_val_loss:.6f}")
        print(f"   Checkpoints saved in: {self.checkpoint_dir}")
        
        # Finish WandB run
        if self.use_wandb:
            wandb.finish()
            print(f"📊 WandB run finished")


def create_model(config) -> RobotMile:
    """Create Robot MILE model with HSR configuration and reconstruction capability."""
    model = RobotMile(config)
    return model


def adjust_config_to_data(config, data_module):
    """Automatically adjust config parameters based on actual data dimensions."""
    print("🔧 Adjusting config to match actual data dimensions...")
    
    # Get a sample batch to inspect data dimensions
    data_module.setup()
    train_loader = data_module.train_dataloader()
    sample_batch = next(iter(train_loader))
    
    # Check joint states dimension
    if "state.joint_positions" in sample_batch:
        joint_states = sample_batch["state.joint_positions"]
        actual_joint_dim = joint_states.shape[-1]  # Last dimension is feature dimension
        
        print(f"🔍 Actual joint states dimension: {actual_joint_dim}")
        print(f"🔍 Current config JOINT.INPUT_DIM: {config.MODEL.JOINT.INPUT_DIM}")
        
        if actual_joint_dim != config.MODEL.JOINT.INPUT_DIM:
            print(f"⚙️  Updating JOINT.INPUT_DIM: {config.MODEL.JOINT.INPUT_DIM} -> {actual_joint_dim}")
            config.MODEL.JOINT.INPUT_DIM = actual_joint_dim
    
    # Check action dimension
    if "action.joint_positions" in sample_batch:
        actions = sample_batch["action.joint_positions"]
        actual_action_dim = actions.shape[-1]  # Last dimension is feature dimension
        
        print(f"🔍 Actual action dimension: {actual_action_dim}")
        print(f"🔍 Current config NUM_JOINTS: {config.MODEL.NUM_JOINTS}")
        
        if actual_action_dim != config.MODEL.NUM_JOINTS:
            print(f"⚙️  Updating NUM_JOINTS: {config.MODEL.NUM_JOINTS} -> {actual_action_dim}")
            config.MODEL.NUM_JOINTS = actual_action_dim
    
    print("✅ Config adjustment completed!")
    return config


def main():
    parser = argparse.ArgumentParser(description="Train HSR Robot MILE v2 with Reconstruction")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Path to HSR dataset root")
    parser.add_argument("--experiment_name", type=str, default="hsr_robot_mile_v2_recon",
                       help="Experiment name for logging")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--cache_videos", action="store_true",
                       help="Cache videos in memory for faster training")
    parser.add_argument("--img_resize", type=int, nargs=2, default=None,
                       help="Resize images to (width, height)")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Resume training from checkpoint")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases logging")
    parser.add_argument("--no_tensorboard", action="store_true",
                       help="Disable TensorBoard logging")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda, cpu, or auto)")
    parser.add_argument("--sequence_length", type=int, default=None,
                       help="Sequence length for temporal modeling (overrides config)")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🚀 HSR Robot MILE Training v2 with Reconstruction")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Dataset: {args.data_root}")
    print(f"Experiment: {args.experiment_name}")
    
    try:
        # Load configuration
        config = get_hsr_training_config(args.data_root)
        
        # Override config with command line arguments
        config.TRAIN.BATCH_SIZE = args.batch_size
        config.TRAIN.LEARNING_RATE = args.learning_rate
        config.TRAIN.NUM_EPOCHS = args.epochs
        
        # Override sequence length if provided
        if args.sequence_length is not None:
            config.MODEL.SEQUENCE_LENGTH = args.sequence_length
            print(f"🔧 Overriding sequence length to: {args.sequence_length}")
        
        # Enable reconstruction and set weight
        config.EVAL.RGB_SUPERVISION = True
        if not hasattr(config.LOSS, 'RECONSTRUCTION_WEIGHT'):
            config.LOSS.RECONSTRUCTION_WEIGHT = 0.3  # Default reconstruction weight
        
        print(f"🎨 Image reconstruction enabled with weight: {config.LOSS.RECONSTRUCTION_WEIGHT}")
        
        data_module = HSRDataModule(
            dataset_path=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sequence_length=config.MODEL.SEQUENCE_LENGTH,
            cache_videos=args.cache_videos,
            img_resize=tuple(args.img_resize) if args.img_resize else None,
            enable_h264_fallback=True,  # Enable H264 fallback for AV1 issues
            skip_video_on_error=False,  # No dummy data - fail on errors
            video_backend="pyav",
        )
        
        # Automatically adjust config to match actual data dimensions
        config = adjust_config_to_data(config, data_module)
        
        # Create model with reconstruction capability
        model = create_model(config)
        
        # Create trainer
        trainer = HSRTrainer(
            config=config,
            model=model,
            data_module=data_module,
            device=device,
            experiment_name=args.experiment_name,
            use_wandb=args.use_wandb,
            use_tensorboard=not args.no_tensorboard,
        )
        
        # Start training
        trainer.train(
            num_epochs=args.epochs,
            resume_from=args.resume_from
        )
        
        print("🎉 Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 