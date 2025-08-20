import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from mile.utils.network_utils import pack_sequence_dim, unpack_sequence_dim, remove_past
from mile.models.common import Decoder, Policy
from mile.layers.layers import BasicBlock
from mile.models.transition import RSSM


class LanguageEncoder(nn.Module):
    """Encodes language instructions using a SentenceTransformer model.

    Supports freezing for feature extraction and unfreezing for fine-tuning.
    """
    def __init__(self, model_name='all-mpnet-base-v2', hidden_dim=256, freeze=True, normalize=True):
        super().__init__()
        self.language_model = SentenceTransformer(model_name)
        self.freeze = freeze
        self.normalize = normalize
        # Freeze or unfreeze parameters
        for p in self.language_model.parameters():
            p.requires_grad = not self.freeze

        embedding_dim = self.language_model.get_sentence_embedding_dimension()

        self.language_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.hidden_dim = hidden_dim

    def forward(self, text_instructions):
        """
        Args:
            text_instructions: List[str]

        Returns:
            language_features: Tensor (batch_size, hidden_dim)
        """
        device = next(self.parameters()).device

        if self.freeze:
            # Feature extraction path (no grad)
            with torch.no_grad():
                embeddings = self.language_model.encode(
                    text_instructions,
                    convert_to_tensor=True,
                    normalize_embeddings=self.normalize,
                ).to(device)
            embeddings = embeddings.clone()
        else:
            # Fine-tuning path with gradients
            tokens = self.language_model.tokenize(text_instructions)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            outputs = self.language_model(tokens)
            embeddings = outputs['sentence_embedding']
            if self.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)

        # Project to desired dimension
        language_features = self.language_projection(embeddings)
        return language_features


class JointEncoder(nn.Module):
    """Encodes joint states (positions, velocities, etc.)"""
    def __init__(self, joint_dim, hidden_dim=128):
        super().__init__()
        self.joint_encoder = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
        )
        self.hidden_dim = hidden_dim
        
    def forward(self, joint_states):
        """
        Args:
            joint_states: (batch_size, sequence_length, joint_dim)
        Returns:
            joint_features: (batch_size, sequence_length, hidden_dim)
        """
        return self.joint_encoder(joint_states)


class RobotPolicy(nn.Module):
    """Policy network for robot joint control with per-dimension ranges"""
    def __init__(self, in_channels, num_joints, action_range=(-1, 1), action_ranges=None):
        super().__init__()
        self.num_joints = num_joints
        self.action_range = action_range
        self.action_ranges = action_ranges if (action_ranges and len(action_ranges) == num_joints) else None
        # Backbone that outputs pre-activation logits
        self.policy_head = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, num_joints),
        )
        # Choose activation(s) by action range(s)
        if self.action_ranges is None:
            action_range_tuple = tuple(action_range) if isinstance(action_range, (list, tuple)) else action_range
            if action_range_tuple == (-1, 1):
                self.output_activation = 'tanh'
                self.apply_scale = False
            elif action_range_tuple == (0, 1):
                self.output_activation = 'sigmoid'
                self.apply_scale = False
            else:
                self.output_activation = 'tanh'
                self.apply_scale = True
        else:
            # Per-dim activation tags: 'tanh', 'sigmoid', or 'scale'
            self.per_dim_mode = []
            for r in self.action_ranges:
                r = tuple(r)
                if r == (-1.0, 1.0):
                    self.per_dim_mode.append('tanh')
                elif r == (0.0, 1.0):
                    self.per_dim_mode.append('sigmoid')
                else:
                    self.per_dim_mode.append('scale')
        
    def forward(self, state):
        """
        Args:
            state: (batch_size * sequence_length, state_dim)
        Returns:
            actions: (batch_size * sequence_length, num_joints)
        """
        logits = self.policy_head(state)
        if self.action_ranges is None:
            if self.output_activation == 'tanh':
                actions = torch.tanh(logits)
            elif self.output_activation == 'sigmoid':
                actions = torch.sigmoid(logits)
            else:
                actions = torch.tanh(logits)
            if getattr(self, 'apply_scale', False):
                min_val, max_val = self.action_range
                actions = (actions + 1) * (max_val - min_val) / 2 + min_val
            return actions
        else:
            # Per-dim activation & scaling
            actions = logits
            out_list = []
            for j in range(self.num_joints):
                mode = self.per_dim_mode[j]
                xj = actions[:, j]
                if mode == 'tanh':
                    yj = torch.tanh(xj)
                elif mode == 'sigmoid':
                    yj = torch.sigmoid(xj)
                else:
                    # generic scale from [-1,1]
                    yj = torch.tanh(xj)
                    mn, mx = self.action_ranges[j]
                    yj = (yj + 1) * (mx - mn) / 2 + mn
                out_list.append(yj.unsqueeze(-1))
            return torch.cat(out_list, dim=-1)


class RobotMile(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.receptive_field = cfg.RECEPTIVE_FIELD
        self.num_joints = cfg.MODEL.NUM_JOINTS
        
        # Image feature encoder
        if self.cfg.MODEL.ENCODER.NAME == 'resnet18':
            self.encoder = timm.create_model(
                cfg.MODEL.ENCODER.NAME, 
                pretrained=True, 
                features_only=True, 
                out_indices=[2, 3, 4],
            )
            feature_info = self.encoder.feature_info.get_dicts(keys=['num_chs', 'reduction'])

        # Image feature decoder
        self.feat_decoder = Decoder(feature_info, self.cfg.MODEL.ENCODER.OUT_CHANNELS)
        
        # Language encoder
        self.language_encoder = LanguageEncoder(
            model_name=cfg.MODEL.LANGUAGE.MODEL_NAME,
            hidden_dim=cfg.MODEL.LANGUAGE.HIDDEN_DIM,
            freeze=cfg.MODEL.LANGUAGE.FREEZE,
            normalize=cfg.MODEL.LANGUAGE.NORMALIZE,
        )
        
        # Joint state encoder
        self.joint_encoder = JointEncoder(
            joint_dim=cfg.MODEL.JOINT.INPUT_DIM,
            hidden_dim=cfg.MODEL.JOINT.HIDDEN_DIM
        )
        
        # Feature fusion
        visual_dim = self.cfg.MODEL.ENCODER.OUT_CHANNELS
        language_dim = cfg.MODEL.LANGUAGE.HIDDEN_DIM
        joint_dim = cfg.MODEL.JOINT.HIDDEN_DIM
        
        # Global average pooling for visual features
        self.visual_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fusion network
        fusion_input_dim = visual_dim + language_dim + joint_dim
        embedding_dim = self.cfg.MODEL.EMBEDDING_DIM
        
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, embedding_dim),
            nn.ReLU(True),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(True),
        )

        # Recurrent model (RSSM)
        if self.cfg.MODEL.TRANSITION.ENABLED:
            self.rssm = RSSM(
                embedding_dim=embedding_dim,
                action_dim=self.num_joints,  # Joint actions
                hidden_state_dim=self.cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM,
                state_dim=self.cfg.MODEL.TRANSITION.STATE_DIM,
                action_latent_dim=self.cfg.MODEL.TRANSITION.ACTION_LATENT_DIM,
                receptive_field=self.receptive_field,
                use_dropout=self.cfg.MODEL.TRANSITION.USE_DROPOUT,
                dropout_probability=self.cfg.MODEL.TRANSITION.DROPOUT_PROBABILITY,
            )

        # Policy for robot control
        if self.cfg.MODEL.TRANSITION.ENABLED:
            state_dim = (self.cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM + 
                        self.cfg.MODEL.TRANSITION.STATE_DIM)
        else:
            state_dim = embedding_dim
            
        self.policy = RobotPolicy(
            in_channels=state_dim,
            num_joints=self.num_joints,
            action_range=cfg.MODEL.ACTION_RANGE,
            action_ranges=getattr(cfg.MODEL, 'ACTION_RANGES', None),
        )

        # Image reconstruction head (always enabled for better representation learning)
        from mile.models.common import RGBHead
        self.rgb_decoder = RGBHead(
            in_channels=state_dim,
            n_classes=3,  # RGB channels
            downsample_factor=1
        )

        # Used during deployment to save last state
        self.last_h = None
        self.last_sample = None
        self.count = 0

    def forward(self, batch, deployment=False):
        """
        Parameters
        ----------
            batch: dict of torch.Tensor
                keys:
                    image: (b, s, 3, h, w) - observation images
                    text_instructions: List of strings or tokenized text - language instructions
                    joint_states: (b, s, joint_dim) - current joint positions/velocities
                    joint_actions: (b, s, num_joints) - target joint actions (for training)
        """
        b, s = batch['image'].shape[:2]
        
        # Encode multimodal inputs
        embedding = self.encode(batch)  # (b, s, embedding_dim)

        output = dict()
        
        if self.cfg.MODEL.TRANSITION.ENABLED:
            # Use joint actions for training, or previous actions for deployment
            if deployment:
                actions = batch['action']  # Previous actions
            else:
                actions = batch['joint_actions']  # Target actions
                
            state_dict = self.rssm(embedding, actions, use_sample=not deployment, policy=self.policy)

            if deployment:
                state_dict = remove_past(state_dict, s)
                s = 1

            output = {**output, **state_dict}
            state = torch.cat([
                state_dict['posterior']['hidden_state'], 
                state_dict['posterior']['sample']
            ], dim=-1)
        else:
            state = embedding

        # Generate joint actions
        state_packed = pack_sequence_dim(state)
        joint_actions = self.policy(state_packed)
        output['joint_actions'] = unpack_sequence_dim(joint_actions, b, s)

        # Always perform image reconstruction for better representation learning
        # Convert state vector to spatial feature map for RGB decoder
        spatial_size = 8  # Small spatial size for efficiency
        state_spatial = state_packed.view(state_packed.shape[0], state_packed.shape[1], 1, 1)
        state_spatial = state_spatial.expand(-1, -1, spatial_size, spatial_size)
        
        rgb_output = self.rgb_decoder(state_spatial)
        rgb_output = unpack_sequence_dim(rgb_output, b, s)
        output = {**output, **rgb_output}

        return output

    def encode(self, batch):
        """
        Encode multimodal inputs (images, language, joint states) into embeddings
        """
        b, s = batch['image'].shape[:2]
        
        # Encode images
        image = pack_sequence_dim(batch['image'])  # (b*s, 3, h, w)
        visual_features = self.encoder(image)
        visual_features = self.feat_decoder(visual_features)  # (b*s, channels, h, w)
        
        # Global average pooling for visual features
        visual_features = self.visual_pool(visual_features).flatten(1)  # (b*s, channels)
        visual_features = unpack_sequence_dim(visual_features, b, s)  # (b, s, channels)
        
        # Encode language instructions (same for all timesteps in sequence)
        language_features = self.language_encoder(batch['text_instructions'])  # (b, lang_dim)
        language_features = language_features.unsqueeze(1).expand(-1, s, -1)  # (b, s, lang_dim)
        
        # Encode joint states
        joint_features = self.joint_encoder(batch['joint_states'])  # (b, s, joint_dim)
        
        # Concatenate all features
        combined_features = torch.cat([
            visual_features,
            language_features,
            joint_features
        ], dim=-1)  # (b, s, total_dim)
        
        # Fuse features
        combined_features = pack_sequence_dim(combined_features)
        embedding = self.fusion_network(combined_features)
        embedding = unpack_sequence_dim(embedding, b, s)
        
        return embedding

    def deployment_forward(self, batch, is_dreaming=False):
        """
        Deployment forward pass with state memory for fast inference
        """
        assert self.cfg.MODEL.TRANSITION.ENABLED
        b = batch['image'].shape[0]

        if self.count == 0:
            # Encode current observations
            s = batch['image'].shape[1]
            # Prefer explicitly provided previous action when available; otherwise, fall back to sequence-based access
            if 'prev_action' in batch and isinstance(batch['prev_action'], torch.Tensor):
                action_t = batch['prev_action']  # shape: (b, A)
            else:
                # Fallback: use previous step in the sequence if available; otherwise use the last one
                actions_seq = batch['action']  # shape: (b, s, A)
                action_t = actions_seq[:, -2] if actions_seq.shape[1] >= 2 else actions_seq[:, -1]
            batch = remove_past(batch, s)
            embedding_t = self.encode(batch)[:, -1]  # Current embedding

            # Initialize or use previous recurrent state
            if self.last_h is None:
                h_t = action_t.new_zeros(b, self.cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM)
                sample_t = action_t.new_zeros(b, self.cfg.MODEL.TRANSITION.STATE_DIM)
            else:
                h_t = self.last_h
                sample_t = self.last_sample

            # Update recurrent state
            if is_dreaming:
                rssm_output = self.rssm.imagine_step(
                    h_t, sample_t, action_t, use_sample=False, policy=self.policy,
                )
            else:
                rssm_output = self.rssm.observe_step(
                    h_t, sample_t, action_t, embedding_t, use_sample=False, policy=self.policy,
                )['posterior']
                
            sample_t = rssm_output['sample']
            h_t = rssm_output['hidden_state']

            self.last_h = h_t
            self.last_sample = sample_t
            
            # Reset counter based on control frequency with robust fallbacks across configs
            control_frequency = getattr(self.cfg, 'CONTROL_FREQUENCY', 1)
            # Prefer STRIDE_SEC if available; otherwise fall back to discrete STRIDE or default 1
            model_stride_sec = getattr(getattr(self.cfg, 'DATASET', object()), 'STRIDE_SEC', None)
            if model_stride_sec is not None:
                n_steps_per_stride = int(max(1, round(control_frequency * float(model_stride_sec))))
            else:
                stride_steps = getattr(getattr(self.cfg, 'DATASET', object()), 'STRIDE', 1)
                # If STRIDE is defined as steps, use it directly; ensure at least 1
                n_steps_per_stride = int(max(1, int(stride_steps)))
            self.count = n_steps_per_stride - 1
        else:
            self.count -= 1

        # Generate action from current state
        s = 1
        state = torch.cat([self.last_h, self.last_sample], dim=-1)
        joint_actions = self.policy(state)
        
        output = dict()
        output['joint_actions'] = unpack_sequence_dim(joint_actions, b, s)
        output['hidden_state'] = self.last_h
        output['sample'] = self.last_sample

        return output

    def imagine(self, batch, predict_action=True, future_horizon=None):
        """
        Imagine future states and actions
        """
        assert self.cfg.MODEL.TRANSITION.ENABLED
        if future_horizon is None:
            future_horizon = self.cfg.FUTURE_HORIZON

        # Imagine future states
        output_imagine = {
            'action': [],
            'state': [],
            'hidden': [],
            'sample': [],
        }
        
        h_t = batch['hidden_state']  # (b, hidden_dim)
        sample_t = batch['sample']   # (b, state_dim)
        b = h_t.shape[0]
        
        for t in range(future_horizon):
            if predict_action:
                action_t = self.policy(torch.cat([h_t, sample_t], dim=-1))
            else:
                action_t = batch['joint_actions'][:, t]
                
            prior_t = self.rssm.imagine_step(
                h_t, sample_t, action_t, use_sample=True, policy=self.policy,
            )
            
            sample_t = prior_t['sample']
            h_t = prior_t['hidden_state']
            
            output_imagine['action'].append(action_t)
            output_imagine['state'].append(torch.cat([h_t, sample_t], dim=-1))
            output_imagine['hidden'].append(h_t)
            output_imagine['sample'].append(sample_t)

        for k, v in output_imagine.items():
            output_imagine[k] = torch.stack(v, dim=1)

        return output_imagine 