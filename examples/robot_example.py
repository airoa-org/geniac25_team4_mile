"""
Example usage of Robot MILE model for language-guided manipulation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List, Dict, Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from mile.models.robot_mile import RobotMile
from mile.configs.robot_config import get_franka_config


class RobotManipulationDataset(Dataset):
    """
    Example dataset for robot manipulation with language instructions
    
    This is a mock dataset for demonstration purposes.
    In practice, you would load your actual robot demonstration data.
    """
    def __init__(self, num_samples=1000, sequence_length=16):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        
        # Mock data parameters
        self.image_size = (224, 224)
        self.num_joints = 7
        
        # Example language instructions
        self.instructions = [
            "Pick up the red block",
            "Move the object to the blue container",
            "Grasp the cup and place it on the table",
            "Open the drawer and put the tool inside",
            "Stack the blocks on top of each other",
            "Pour water from the bottle into the glass",
            "Turn the handle to open the door",
            "Press the button with your finger",
        ]
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Mock image sequence (normally from camera observations)
        images = torch.randn(self.sequence_length, 3, *self.image_size)
        
        # Mock joint states (positions + velocities)
        joint_positions = torch.randn(self.sequence_length, self.num_joints)
        joint_velocities = torch.randn(self.sequence_length, self.num_joints)
        joint_states = torch.cat([joint_positions, joint_velocities], dim=-1)
        
        # Mock joint actions (target positions or velocities)
        joint_actions = torch.randn(self.sequence_length, self.num_joints)
        
        # Random language instruction
        instruction = self.instructions[idx % len(self.instructions)]
        
        return {
            'image': images,
            'joint_states': joint_states,
            'joint_actions': joint_actions,
            'text_instructions': instruction
        }


def collate_fn(batch):
    """
    Custom collate function to handle text instructions
    """
    images = torch.stack([item['image'] for item in batch])
    joint_states = torch.stack([item['joint_states'] for item in batch])
    joint_actions = torch.stack([item['joint_actions'] for item in batch])
    text_instructions = [item['text_instructions'] for item in batch]
    
    return {
        'image': images,
        'joint_states': joint_states,
        'joint_actions': joint_actions,
        'text_instructions': text_instructions
    }


def compute_loss(output, batch, cfg):
    """
    Compute training loss
    """
    losses = {}
    
    # Action prediction loss
    predicted_actions = output['joint_actions']
    target_actions = batch['joint_actions']
    action_loss = nn.MSELoss()(predicted_actions, target_actions)
    losses['action_loss'] = action_loss
    
    # KL divergence loss (if RSSM is enabled)
    if cfg.MODEL.TRANSITION.ENABLED and 'kl_loss' in output:
        kl_loss = output['kl_loss'].mean()
        losses['kl_loss'] = kl_loss
    else:
        losses['kl_loss'] = torch.tensor(0.0)
    
    # Total loss
    total_loss = (cfg.LOSS.ACTION_WEIGHT * losses['action_loss'] + 
                  cfg.LOSS.KL_WEIGHT * losses['kl_loss'])
    losses['total_loss'] = total_loss
    
    return losses


def train_epoch(model, dataloader, optimizer, cfg, device):
    """
    Train for one epoch
    """
    model.train()
    total_losses = {'total_loss': 0, 'action_loss': 0, 'kl_loss': 0}
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        for key in ['image', 'joint_states', 'joint_actions']:
            batch[key] = batch[key].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(batch)
        
        # Compute loss
        losses = compute_loss(output, batch, cfg)
        
        # Backward pass
        losses['total_loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAIN.GRADIENT_CLIP)
        
        optimizer.step()
        
        # Accumulate losses
        for key in total_losses:
            total_losses[key] += losses[key].item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}/{len(dataloader)}, '
                  f'Loss: {losses["total_loss"].item():.4f}, '
                  f'Action: {losses["action_loss"].item():.4f}, '
                  f'KL: {losses["kl_loss"].item():.4f}')
    
    # Average losses
    for key in total_losses:
        total_losses[key] /= num_batches
    
    return total_losses


def validate(model, dataloader, cfg, device):
    """
    Validation
    """
    model.eval()
    total_losses = {'total_loss': 0, 'action_loss': 0, 'kl_loss': 0}
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            for key in ['image', 'joint_states', 'joint_actions']:
                batch[key] = batch[key].to(device)
            
            # Forward pass
            output = model(batch)
            
            # Compute loss
            losses = compute_loss(output, batch, cfg)
            
            # Accumulate losses
            for key in total_losses:
                total_losses[key] += losses[key].item()
            num_batches += 1
    
    # Average losses
    for key in total_losses:
        total_losses[key] /= num_batches
    
    return total_losses


def inference_example(model, cfg, device):
    """
    Example of using the model for inference
    """
    model.eval()
    
    # Create a single example
    batch_size = 1
    sequence_length = 1  # Single timestep for inference
    
    # Mock current observation
    current_image = torch.randn(batch_size, sequence_length, 3, 224, 224).to(device)
    current_joint_state = torch.randn(batch_size, sequence_length, 14).to(device)  # 7 pos + 7 vel
    instruction = ["Pick up the red cube and place it in the box"]
    
    batch = {
        'image': current_image,
        'joint_states': current_joint_state,
        'text_instructions': instruction,
        'action': torch.zeros(batch_size, sequence_length, cfg.MODEL.NUM_JOINTS).to(device)  # Previous action
    }
    
    with torch.no_grad():
        # Get action prediction
        output = model.deployment_forward(batch)
        predicted_action = output['joint_actions']
        
        print(f"Language instruction: {instruction[0]}")
        print(f"Current joint states: {current_joint_state.squeeze().cpu().numpy()[:7]}")  # Positions only
        print(f"Predicted joint actions: {predicted_action.squeeze().cpu().numpy()}")
        
        return predicted_action


def main():
    """
    Main training and inference example
    """
    # Configuration
    cfg = get_franka_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = RobotManipulationDataset(num_samples=800, sequence_length=cfg.EVAL.SEQUENCE_LENGTH)
    val_dataset = RobotManipulationDataset(num_samples=200, sequence_length=cfg.EVAL.SEQUENCE_LENGTH)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.TRAIN.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.EVAL.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Create model
    model = RobotMile(cfg).to(device)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=cfg.TRAIN.LEARNING_RATE, 
        weight_decay=cfg.TRAIN.WEIGHT_DECAY
    )
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training loop (shortened for example)
    num_epochs = 5  # Reduced for demo
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, cfg, device)
        print(f"Train - Total: {train_losses['total_loss']:.4f}, "
              f"Action: {train_losses['action_loss']:.4f}, "
              f"KL: {train_losses['kl_loss']:.4f}")
        
        # Validate
        val_losses = validate(model, val_loader, cfg, device)
        print(f"Val - Total: {val_losses['total_loss']:.4f}, "
              f"Action: {val_losses['action_loss']:.4f}, "
              f"KL: {val_losses['kl_loss']:.4f}")
    
    # Inference example
    print("\n" + "="*50)
    print("INFERENCE EXAMPLE")
    print("="*50)
    
    inference_example(model, cfg, device)
    
    # Save model
    torch.save(model.state_dict(), 'robot_mile_model.pth')
    print("\nModel saved as 'robot_mile_model.pth'")


if __name__ == "__main__":
    main() 