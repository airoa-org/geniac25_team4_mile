"""
Configuration for OXE/Fractal dataset setup in MILE.
Separated from HSR-specific settings to avoid cross-contamination.
"""

from yacs.config import CfgNode as CN
from mile.configs.robot_config import get_robot_config


def get_fractal_config():
    """Base configuration for Fractal evaluation/training.

    Notes:
    - Fractal dataset may not contain joint states/actions; model/loss code should
      handle missing keys gracefully. Here we keep reasonable defaults from the
      generic robot config and rely on runtime checks to skip unavailable signals.
    """
    cfg = get_robot_config()
    # Standardized robot naming schema (Fractal may not have joints/actions)
    cfg.ROBOT.JOINT_NAMES = ["x", "y", "z", "rx", "ry", "rz", "rw", "gripper"]
    cfg.ROBOT.ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    
    cfg.MODEL.NUM_JOINTS = 7  # Fractal action dimensions
    cfg.MODEL.JOINT.INPUT_DIM = 8

    # Image encoder emphasis (vision-centric)
    cfg.MODEL.ENCODER.OUT_CHANNELS = 256

    # Language settings for text instructions (present in OXE/Fractal)
    cfg.MODEL.LANGUAGE.MODEL_NAME = 'all-mpnet-base-v2'
    cfg.MODEL.LANGUAGE.HIDDEN_DIM = 256

    # Sequence and state sizes can stay moderate
    cfg.MODEL.SEQUENCE_LENGTH = 8
    cfg.MODEL.HIDDEN_STATE_DIM = 256

    # Transition model stays enabled but with moderate dims
    cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM = 256
    cfg.MODEL.TRANSITION.STATE_DIM = 32

    # Training defaults
    cfg.TRAIN.BATCH_SIZE = 8
    cfg.TRAIN.LEARNING_RATE = 1e-4
    cfg.TRAIN.WEIGHT_DECAY = 1e-5
    cfg.TRAIN.GRADIENT_CLIP = 1.0
    cfg.TRAIN.NUM_WORKERS = 4

    # Loss weights; keep some emphasis on reconstruction when enabled
    cfg.LOSS.ACTION_WEIGHT = 1.0
    cfg.LOSS.KL_WEIGHT = 1.0
    cfg.LOSS.RECONSTRUCTION_WEIGHT = 1e-3
    cfg.LOSS.KL_BALANCING_ALPHA = 0.75

    # Dataset settings (Fractal)
    cfg.DATASET = CN()
    cfg.DATASET.DATAROOT = '/path/to/fractal/data'  # Will be overridden
    cfg.DATASET.STRIDE = 1

    # Evaluation
    cfg.EVAL.BATCH_SIZE = 2
    cfg.EVAL.SEQUENCE_LENGTH = 8
    cfg.EVAL.RGB_SUPERVISION = True

    return cfg


def get_fractal_training_config(data_root: str):
    cfg = get_fractal_config()
    cfg.DATASET.DATAROOT = data_root
    return cfg


def get_fractal_eval_config(data_root: str, model_path: str):
    cfg = get_fractal_training_config(data_root)
    cfg.EVAL.BATCH_SIZE = 1
    cfg.EVAL.SEQUENCE_LENGTH = 16
    cfg.MODEL.PRETRAINED_PATH = model_path
    return cfg


