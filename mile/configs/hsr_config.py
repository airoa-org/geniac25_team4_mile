"""
Configuration for HSR Robot MILE
Based on TMC HSR demonstration data
"""

from yacs.config import CfgNode as CN
from mile.configs.robot_config import get_robot_config


def get_hsr_config():
    """Configuration for Toyota HSR robot"""
    cfg = get_robot_config()
    
    # HSR-specific model settings
    cfg.MODEL.NUM_JOINTS = 11  # HSR action dimensions
    cfg.MODEL.JOINT.INPUT_DIM = 8  # 8 positions + 8 velocities
    cfg.MODEL.ACTION_RANGE = (-2.0, 2.0)  # HSR action range
    cfg.MODEL.SEQUENCE_LENGTH = 8  # Match with EVAL.SEQUENCE_LENGTH for consistency
    cfg.MODEL.HIDDEN_STATE_DIM = 512  # Higher dimensional state for complex HSR robot
    
    # HSR has different joint configuration
    cfg.MODEL.JOINT.HIDDEN_DIM = 128
    cfg.MODEL.EMBEDDING_DIM = 512
    
    # Update transition model to match HSR complexity
    cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM = 512  # Match MODEL.HIDDEN_STATE_DIM
    cfg.MODEL.TRANSITION.STATE_DIM = 64  # Larger state dimension for HSR
    
    # Language settings for HSR tasks
    cfg.MODEL.LANGUAGE.MODEL_NAME = 'all-mpnet-base-v2'
    cfg.MODEL.LANGUAGE.HIDDEN_DIM = 256
    
    # Image settings for HSR cameras (hand/head)
    cfg.MODEL.ENCODER.OUT_CHANNELS = 256
    
    # Training settings for HSR data
    cfg.TRAIN.BATCH_SIZE = 32  # Reduced batch size for video data
    cfg.TRAIN.LEARNING_RATE = 1e-4
    cfg.TRAIN.WEIGHT_DECAY = 1e-5
    cfg.TRAIN.GRADIENT_CLIP = 1.0
    cfg.TRAIN.NUM_WORKERS = 4
    
    # Loss weights
    cfg.LOSS.ACTION_WEIGHT = 1.0
    cfg.LOSS.KL_WEIGHT = 1.0
    cfg.LOSS.RECONSTRUCTION_WEIGHT = 1e-4  # Moderate weight for image reconstruction
    cfg.LOSS.KL_BALANCING_ALPHA = 0.75  # For KLLoss balancing between prior and posterior
    
    # Dataset settings
    cfg.DATASET = CN()
    cfg.DATASET.DATAROOT = '/path/to/hsr/data'  # Will be overridden
    cfg.DATASET.STRIDE = 2  # Use every 2nd frame for efficiency
    cfg.DATASET.CAMERA = 'hand'  # Primary camera ('hand' or 'head')
    
    # Evaluation settings
    cfg.EVAL.BATCH_SIZE = 2
    cfg.EVAL.SEQUENCE_LENGTH = 8  # Shorter sequences for efficiency
    cfg.EVAL.RGB_SUPERVISION = True  # Enable image reconstruction
    
    # Standardized robot naming schema
    cfg.ROBOT.JOINT_NAMES = [
        'arm_lift_joint',
        'arm_flex_joint', 
        'arm_roll_joint',
        'wrist_flex_joint',
        'wrist_roll_joint',
        'hand_motor_joint',
        'head_pan_joint',
        'head_tilt_joint'
    ]
    cfg.ROBOT.ACTION_NAMES = [
        'arm_lift', 'arm_flex', 'arm_roll', 'wrist_flex', 'wrist_roll',
        'hand_motor',
        'head_pan', 'head_tilt',
        'base_x', 'base_y', 'base_theta'
    ]
    
    return cfg


def get_hsr_training_config(data_root: str):
    """Get HSR config with specific data path"""
    cfg = get_hsr_config()
    cfg.DATASET.DATAROOT = data_root
    return cfg


def get_hsr_eval_config(data_root: str, model_path: str):
    """Get HSR config for evaluation"""
    cfg = get_hsr_training_config(data_root)
    
    # Evaluation-specific settings
    cfg.EVAL.BATCH_SIZE = 1
    cfg.EVAL.SEQUENCE_LENGTH = 16
    cfg.MODEL.PRETRAINED_PATH = model_path
    
    return cfg 