# Robot MILE: Language-Guided Robot Manipulation

This is a modified version of the MILE (Model-based Imitation Learning) architecture adapted for language-guided robot manipulation tasks. The original MILE model was designed for autonomous driving, but this version focuses on robotic manipulation using multimodal inputs: language instructions, visual observations, and joint states.

## Key Features

- **Multimodal Input**: Processes language instructions, camera images, and robot joint states
- **Recurrent State Space Model (RSSM)**: Uses temporal dynamics for sequence modeling
- **Language Understanding**: Integrates pre-trained language models (BERT/DistilBERT)
- **Joint Control**: Outputs joint angle commands for robot manipulation
- **Configurable**: Supports various robot configurations (7-DOF, 6-DOF, dual-arm, etc.)

## Architecture Overview

```
Language Instruction ──┐
                       ├─► Feature Fusion ──► RSSM ──► Policy ──► Joint Actions
Camera Images ────────┤
Joint States ─────────┘
```

### Components

1. **Language Encoder**: Processes natural language instructions using pre-trained transformers
2. **Visual Encoder**: ResNet-based feature extraction from camera observations
3. **Joint Encoder**: Processes current joint positions and velocities
4. **Feature Fusion**: Combines multimodal features into unified representation
5. **RSSM**: Temporal sequence modeling with hidden states
6. **Robot Policy**: Outputs joint angle commands

## Installation

### Dependencies

```bash
# Core dependencies
pip install torch torchvision timm transformers
pip install yacs numpy

# For visualization (optional)
pip install matplotlib opencv-python

# For robot integration (depends on your setup)
# pip install pybullet  # For simulation
# pip install rospy     # For ROS integration
```

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd geniac25_team4_mile
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the environment:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Usage

### Quick Start

```python
from mile.models.robot_mile import RobotMile
from mile.configs.robot_config import get_franka_config

# Load configuration for Franka Panda robot
cfg = get_franka_config()

# Create model
model = RobotMile(cfg)

# Example input batch
batch = {
    'image': torch.randn(1, 1, 3, 224, 224),          # Camera observation
    'joint_states': torch.randn(1, 1, 14),            # Joint pos + vel
    'text_instructions': ["Pick up the red block"],   # Language instruction
}

# Forward pass
output = model(batch, deployment=True)
joint_actions = output['joint_actions']  # Shape: (1, 1, 7)
```

### Training

```python
# Run the example training script
python examples/robot_example.py
```

This will:
- Create mock training data
- Train the model for a few epochs
- Demonstrate inference
- Save the trained model

### Configuration

The model supports various robot configurations:

```python
from mile.configs.robot_config import get_franka_config, get_ur5_config, get_dual_arm_config

# For Franka Panda (7-DOF)
cfg = get_franka_config()

# For UR5 (6-DOF)
cfg = get_ur5_config()

# For dual-arm setup (14-DOF)
cfg = get_dual_arm_config()
```

## Model Configuration

### Key Parameters

- `MODEL.NUM_JOINTS`: Number of robot joints (7 for Franka, 6 for UR5, etc.)
- `MODEL.ACTION_RANGE`: Joint angle limits
- `MODEL.LANGUAGE.MODEL_NAME`: Pre-trained language model
- `MODEL.TRANSITION.ENABLED`: Whether to use RSSM
- `MODEL.EMBEDDING_DIM`: Feature embedding dimension

### Example Configuration

```python
cfg.MODEL.NUM_JOINTS = 7
cfg.MODEL.ACTION_RANGE = (-2.8973, 2.8973)  # Franka joint limits
cfg.MODEL.LANGUAGE.MODEL_NAME = 'distilbert-base-uncased'
cfg.MODEL.TRANSITION.ENABLED = True
```

## Data Format

### Training Data

The model expects training data in the following format:

```python
{
    'image': torch.Tensor,           # Shape: (batch, sequence, 3, H, W)
    'joint_states': torch.Tensor,    # Shape: (batch, sequence, joint_dim)
    'joint_actions': torch.Tensor,   # Shape: (batch, sequence, num_joints)
    'text_instructions': List[str],  # List of instruction strings
}
```

### Joint States

Joint states should include both positions and velocities:
- For 7-DOF robot: `joint_dim = 14` (7 positions + 7 velocities)
- For 6-DOF robot: `joint_dim = 12` (6 positions + 6 velocities)

## Integration with Real Robots

### ROS Integration Example

```python
import rospy
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import String

class RobotMileNode:
    def __init__(self):
        self.model = RobotMile(cfg)
        self.model.load_state_dict(torch.load('robot_mile_model.pth'))
        
        # ROS subscribers
        rospy.Subscriber('/joint_states', JointState, self.joint_callback)
        rospy.Subscriber('/camera/image', Image, self.image_callback)
        rospy.Subscriber('/instruction', String, self.instruction_callback)
        
        # ROS publisher
        self.cmd_pub = rospy.Publisher('/joint_commands', JointState, queue_size=1)
    
    def predict_action(self, image, joint_state, instruction):
        batch = {
            'image': image.unsqueeze(0).unsqueeze(0),
            'joint_states': joint_state.unsqueeze(0).unsqueeze(0),
            'text_instructions': [instruction],
            'action': torch.zeros(1, 1, self.cfg.MODEL.NUM_JOINTS)
        }
        
        with torch.no_grad():
            output = self.model.deployment_forward(batch)
            return output['joint_actions'].squeeze()
```

## Differences from Original MILE

### Removed Components
- BEV (Bird's Eye View) transformation
- Route map encoding
- GPS and navigation-specific measurements
- Frustum pooling
- Automotive-specific outputs

### Added Components
- Language instruction encoder
- Joint state encoder
- Multimodal feature fusion
- Robot-specific policy network

### Modified Components
- Action space: From throttle/steering to joint angles
- Input modalities: From driving sensors to manipulation sensors
- State representation: From vehicle dynamics to joint dynamics

## Limitations and Future Work

### Current Limitations
- Mock dataset in examples (requires real robot data)
- Limited to position/velocity control
- No force/torque feedback
- No grasp planning integration

### Future Improvements
- Integration with grasp planning
- Force/torque control
- Multi-task learning
- Sim-to-real transfer
- Safety constraints

## Citation

If you use this work, please cite the original MILE paper:

```bibtex
@article{mile2022,
  title={Model-Based Imitation Learning for Urban Driving},
  author={...},
  journal={...},
  year={2022}
}
```

## License

This project follows the same license as the original MILE implementation.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Contact

For questions about the robot adaptation, please open an issue in this repository. 