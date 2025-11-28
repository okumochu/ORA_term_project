"""
Shared configuration for Graph-Based RL experiments (HGT)
Focused on graph neural network approaches for FJSP

Simple unified data generation system using problem_config with min/max parameters.
"""

class ExperimentConfig:
    """Centralized configuration container for graph-based methods (HGT)."""
    
    def __init__(self):
        
        # Problem configuration for data generation
        # All parameters are fixed throughout training, using uniform distribution for sampling
        # Within each parallel rollout batch: all envs have same operation count (auto-enforced)
        # Between batches: operation count can vary based on [min, max] range
        machine_num = 5
        self.problem_config = {
            'job_num': 10,
            'machine_num': machine_num,
            'operation_per_job_min': machine_num,  # Min operations per job
            'operation_per_job_max': machine_num,  # Max operations per job
            'machine_per_operation_min': 1,
            'machine_per_operation_max': machine_num,
            'process_time_min': 1,
            'process_time_max': 99,
            'fixed_parameter': {
                'pt': False,           # Processing times vary across environments
                'compatibility': False  # Machine compatibility varies across environments
            }
        }
        
        # Test configuration
        self.test_num = -1  # Number of files to use for testing from test_data
                       # -1 means use all available files, 0 means skip test_data

        # Common parameters (shared across DANIEL and HGT)
        self.common_params = {
            'device': 'cuda', 
            'wandb_project': "FJSP_oversmoothing",
            'model_name': f"HGT_{self.problem_config['job_num']}x{self.problem_config['machine_num']}_SD2"
        }
        
        # Method-specific parameters (DANIEL and HGT only)
        self.method_params = {
            'daniel': {
                # DANIEL network structure (Dual Attention Network)
                'num_layers': 2,  # Number of dual attention layers
                'test_episode': 1000,  # Run test every N episodes
                'max_updates': 1000,  # Maximum training updates


                # fixed
                'neighbor_pooling_type': 'average',  # Pooling type for neighbor aggregation
                'num_mlp_layers_feature_extract': 2,  # MLP layers in feature extraction
                'num_mlp_layers_actor': 3,  # MLP layers in actor network
                'num_mlp_layers_critic': 3,  # MLP layers in critic network
                'hidden_dim_actor': 32,  # Hidden dimension for actor MLP
                'hidden_dim_critic': 32,  # Hidden dimension for critic MLP
                'num_envs': 20,  # Number of parallel environments
                'lr': 3e-4,  # Learning rate
                'gamma': 1,  # Discount factor
                'k_epochs': 4,  # PPO epochs per update
                'eps_clip': 0.2,  # PPO clipping parameter
                'vloss_coef': 0.5,  # Value loss coefficient
                'ploss_coef': 1,  # Policy loss coefficient
                'tau': 0,  # Target network update rate (0 = no target network)
                'gae_lambda': 0.98,  # GAE lambda for advantage estimation
                'entloss_coef': 0.01,  # Entropy loss coefficient
                'minibatch_size': 1024,  # Minibatch size for PPO updates
                'max_grad_norm': 0.5,  # Gradient clipping threshold
                'reset_env_episode': 20,  # Resample environments every N episodes
                
                # DANIEL-specific attention network parameters
                'fea_j_input_dim': 10,  # Operation input feature dimension (actual: 10 features)
                'fea_m_input_dim': 8,  # Machine input feature dimension (actual: 8 features)
                'layer_fea_output_dim': [64, 64],  # Output dim for each DAN layer (2 layers)
                'num_heads_OAB': [4, 4],  # Number of heads for operation attention blocks per layer
                'num_heads_MAB': [4, 4],  # Number of heads for machine attention blocks per layer
                'dropout_prob': 0.0,  # Dropout probability
            }
        }

# Single global instance for consumers that still import it
config = ExperimentConfig()

