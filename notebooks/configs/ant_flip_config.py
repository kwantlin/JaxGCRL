"""
Configuration file for training PPO on AntFlip environment.
Based on hyperparameters that worked well for ant_walk in simple_ppo.py
"""

# Environment configuration
ENV_CONFIG = {
    'env_name': 'antflip',
    'min_flip_velocity': 1.0,  # Minimum angular velocity for flip reward
}

# PPO Training hyperparameters
TRAINING_CONFIG = {
    # Training duration
    'total_env_steps': 90_000_000,  # Total environment steps to train for
    
    # Environment settings
    'episode_length': 1000,  # Maximum episode length
    'num_envs': 2048,  # Number of parallel environments
    
    # Learning parameters
    'learning_rate': 3e-4,  # Learning rate for PPO
    'entropy_cost': 3e-4,  # Entropy cost coefficient for exploration
    'discounting': 0.97,  # Discount factor for future rewards
    
    # PPO-specific parameters
    'unroll_length': 10,  # Number of timesteps to unroll in each environment
    'batch_size': 1024,  # Batch size for PPO updates
    'num_minibatches': 8,  # Number of minibatches for PPO updates
    'num_updates_per_batch': 8,  # Number of updates per batch
    
    # Evaluation settings
    'num_evals': 20,  # Number of evaluations during training
    'num_eval_episodes': 5,  # Number of episodes for final evaluation
    
    # Random seed for reproducibility
    'seed': 0,
    
    # Network settings
    'normalize_observations': True,  # Whether to normalize observations
}

# Model saving configuration
SAVE_CONFIG = {
    'model_save_path': 'ppo_antflip_model.pkl',
    'video_save_prefix': 'ppo_antflip_video',
}

# Optional: Environment-specific hyperparameter tuning suggestions
# These can be adjusted based on training performance
TUNING_SUGGESTIONS = {
    'learning_rate': {
        'description': 'If training is unstable, try reducing to 1e-4',
        'range': [1e-4, 5e-4]
    },
    'entropy_cost': {
        'description': 'If exploration is insufficient, try increasing to 5e-4',
        'range': [1e-4, 5e-4]
    },
    'min_flip_velocity': {
        'description': 'Adjust based on desired flip speed. Higher values = more challenging',
        'range': [0.5, 3.0]
    },
    'episode_length': {
        'description': 'May need to increase for more complex flip sequences',
        'range': [500, 2000]
    },
    'num_envs': {
        'description': 'Increase for faster training if compute allows',
        'range': [1024, 4096]
    }
}

# Example usage:
# from configs.ant_flip_config import TRAINING_CONFIG, ENV_CONFIG
# 
# # Use in training script
# make_policy, params, _ = ppo_train.train(
#     environment=env,
#     num_timesteps=TRAINING_CONFIG['total_env_steps'],
#     episode_length=TRAINING_CONFIG['episode_length'],
#     num_envs=TRAINING_CONFIG['num_envs'],
#     learning_rate=TRAINING_CONFIG['learning_rate'],
#     entropy_cost=TRAINING_CONFIG['entropy_cost'],
#     discounting=TRAINING_CONFIG['discounting'],
#     seed=TRAINING_CONFIG['seed'],
#     unroll_length=TRAINING_CONFIG['unroll_length'],
#     batch_size=TRAINING_CONFIG['batch_size'],
#     num_minibatches=TRAINING_CONFIG['num_minibatches'],
#     num_updates_per_batch=TRAINING_CONFIG['num_updates_per_batch'],
#     num_evals=TRAINING_CONFIG['num_evals'],
#     normalize_observations=TRAINING_CONFIG['normalize_observations'],
#     network_factory=network_factory,
#     progress_fn=progress,
# )
