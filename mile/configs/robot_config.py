from yacs.config import CfgNode as CN

def get_robot_config():
    """
    Configuration for Robot Manipulation MILE model
    """
    cfg = CN()

    # General settings
    cfg.RECEPTIVE_FIELD = 8
    cfg.FUTURE_HORIZON = 10
    cfg.CONTROL_FREQUENCY = 10  # Hz
    
    # Dataset settings
    cfg.DATASET = CN()
    cfg.DATASET.STRIDE_SEC = 0.1  # Time between frames in seconds
    
    # Model architecture
    cfg.MODEL = CN()
    cfg.MODEL.NUM_JOINTS = 7  # Number of robot joints (e.g., 7-DOF arm)
    cfg.MODEL.ACTION_RANGE = (-1.0, 1.0)  # Action output range
    cfg.MODEL.EMBEDDING_DIM = 512
    cfg.MODEL.SEQUENCE_LENGTH = 8  # Default sequence length for temporal modeling
    cfg.MODEL.HIDDEN_STATE_DIM = 256  # Hidden state dimension for models
    
    # Image encoder settings
    cfg.MODEL.ENCODER = CN()
    cfg.MODEL.ENCODER.NAME = 'resnet18'
    cfg.MODEL.ENCODER.OUT_CHANNELS = 256
    
    # Language model settings
    cfg.MODEL.LANGUAGE = CN()
    cfg.MODEL.LANGUAGE.MODEL_NAME = 'all-mpnet-base-v2'
    cfg.MODEL.LANGUAGE.HIDDEN_DIM = 256
    
    # Joint state encoder settings
    cfg.MODEL.JOINT = CN()
    cfg.MODEL.JOINT.INPUT_DIM = 14  # Position (7) + velocity (7) for 7-DOF arm
    cfg.MODEL.JOINT.HIDDEN_DIM = 128
    
    # Transition model (RSSM) settings
    cfg.MODEL.TRANSITION = CN()
    cfg.MODEL.TRANSITION.ENABLED = True
    cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM = 256
    cfg.MODEL.TRANSITION.STATE_DIM = 32
    cfg.MODEL.TRANSITION.ACTION_LATENT_DIM = 32
    cfg.MODEL.TRANSITION.USE_DROPOUT = True
    cfg.MODEL.TRANSITION.DROPOUT_PROBABILITY = 0.1
    
    # Robot naming schema (standardized across configs)
    cfg.ROBOT = CN()
    cfg.ROBOT.JOINT_NAMES = []  # Optional: semantic joint names if available
    cfg.ROBOT.ACTION_NAMES = []  # Optional: semantic action names (length == action dims)
    
    # Training settings
    cfg.TRAIN = CN()
    cfg.TRAIN.BATCH_SIZE = 8
    cfg.TRAIN.LEARNING_RATE = 1e-4
    cfg.TRAIN.NUM_EPOCHS = 100
    cfg.TRAIN.GRADIENT_CLIP = 1.0
    cfg.TRAIN.WEIGHT_DECAY = 1e-5
    
    # Loss weights
    cfg.LOSS = CN()
    cfg.LOSS.ACTION_WEIGHT = 1.0
    cfg.LOSS.KL_WEIGHT = 1.0
    cfg.LOSS.RECONSTRUCTION_WEIGHT = 1e-3
    
    # Evaluation settings
    cfg.EVAL = CN()
    cfg.EVAL.BATCH_SIZE = 1
    cfg.EVAL.SEQUENCE_LENGTH = 16
    cfg.EVAL.RGB_SUPERVISION = True  # Enable image reconstruction by default
    
    return cfg


# Example configurations for different robot setups

def get_franka_config():
    """Configuration for Franka Panda 7-DOF arm"""
    cfg = get_robot_config()
    cfg.MODEL.NUM_JOINTS = 7
    cfg.MODEL.JOINT.INPUT_DIM = 14  # 7 positions + 7 velocities
    cfg.MODEL.ACTION_RANGE = (-2.8973, 2.8973)  # Franka joint limits (approx)
    return cfg


def get_ur5_config():
    """Configuration for UR5 6-DOF arm"""
    cfg = get_robot_config()
    cfg.MODEL.NUM_JOINTS = 6
    cfg.MODEL.JOINT.INPUT_DIM = 12  # 6 positions + 6 velocities
    cfg.MODEL.ACTION_RANGE = (-3.14159, 3.14159)  # +-pi for most UR5 joints
    return cfg


def get_dual_arm_config():
    """Configuration for dual-arm setup"""
    cfg = get_robot_config()
    cfg.MODEL.NUM_JOINTS = 14  # 7 + 7 for dual arms
    cfg.MODEL.JOINT.INPUT_DIM = 28  # 14 positions + 14 velocities
    cfg.MODEL.EMBEDDING_DIM = 768  # Larger embedding for more complex setup
    return cfg 