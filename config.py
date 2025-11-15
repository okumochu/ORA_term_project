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
        machine_num = 10
        self.problem_config = {
            'job_num': 20,
            'machine_num': machine_num,
            'operation_per_job_min': machine_num,
            'operation_per_job_max': machine_num,
            'machine_per_operation_min': 1,
            'machine_per_operation_max': machine_num,
            'process_time_min': 1,
            'process_time_max': 99,
            'fixed_parameter': {
                'pt': False,           # Processing times vary across environments
                'op_num': True,       # Operation counts vary across environments
                'compatibility': True  # Machine compatibility varies across environments
            }
        }
        
        # Test configuration
        self.test_num = -1  # Number of files to use for testing from test_data
                       # -1 means use all available files, 0 means skip test_data

        # Common parameters (shared across DANIEL and HGT)
        self.common_params = {
            'device': 'cuda', 
            'wandb_project': "FJSP_oversmoothing",
            'model_name': f"{self.problem_config['job_num']}x{self.problem_config['machine_num']}_SD2"
        }
        
        # Method-specific parameters (DANIEL and HGT only)
        self.method_params = {
            'hgt': {
                # HGT network structure
                'op_feature_dim': 10, 
                'machine_feature_dim': 8,
                'hidden_dim': 64, 
                'num_hgt_layers': 2, 
                'num_heads': 2, 
                'dropout': 0.1,
                'num_mlp_layers_actor': 3, 
                'hidden_dim_actor': 32,
                'num_mlp_layers_critic': 3, 
                'hidden_dim_critic': 32,
                'num_envs': 20, 
                'max_updates': 1000, 
                'lr': 1e-4, 
                'gamma': 1,
                'k_epochs': 4, 
                'eps_clip': 0.2, 
                'vloss_coef': 0.5, 
                'ploss_coef': 1,
                'tau': 0, 
                'gae_lambda': 0.98, 
                'entloss_coef': 0.01,
                'minibatch_size': 1024,
                'max_grad_norm': 2.5,  # HGT needs lower gradient clipping for stability
                'test_episode': 200,  # Run test every N episodes
                'reset_env_episode': 20,  # Keep same batch of environments for N episodes (resample every N episodes)
            }
        }

# Single global instance for consumers that still import it
config = ExperimentConfig()
