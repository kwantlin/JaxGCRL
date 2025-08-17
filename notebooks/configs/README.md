# PPO Training Configuration Files

This directory contains configuration files for training PPO policies on different ant environments. The configurations are based on hyperparameters that worked well for ant_walk in the `simple_ppo.py` script.

## Available Configurations

### 1. `ant_forward_config.py`
- **Environment**: AntForward
- **Task**: Move forward at a minimum velocity
- **Key Parameter**: `min_forward_velocity = 0.5`
- **State Space**: 27 dimensions
- **Action Space**: 8 dimensions (joint torques)

### 2. `ant_jump_config.py`
- **Environment**: AntJump
- **Task**: Jump to a target height
- **Key Parameter**: `target_jump_height = 1.0`
- **State Space**: 27 dimensions
- **Action Space**: 8 dimensions (joint torques)

### 3. `ant_flip_config.py`
- **Environment**: AntFlip
- **Task**: Perform flips with minimum angular velocity
- **Key Parameter**: `min_flip_velocity = 1.0`
- **State Space**: 27 dimensions
- **Action Space**: 8 dimensions (joint torques)

## Hyperparameters

All configurations use the following base hyperparameters that worked well for ant_walk:

```python
TRAINING_CONFIG = {
    'total_env_steps': 10_000_000,  # 10M environment steps
    'episode_length': 1000,         # Max episode length
    'num_envs': 2048,              # Parallel environments
    'learning_rate': 3e-4,         # Learning rate
    'entropy_cost': 3e-4,          # Entropy cost for exploration
    'discounting': 0.97,           # Discount factor
    'unroll_length': 10,           # PPO unroll length
    'batch_size': 1024,            # Batch size
    'num_minibatches': 8,          # Number of minibatches
    'num_updates_per_batch': 8,    # Updates per batch
    'num_evals': 20,               # Evaluation frequency
    'num_eval_episodes': 5,        # Episodes for final evaluation
    'seed': 0,                     # Random seed
    'normalize_observations': True # Observation normalization
}
```

## Usage

### Using the Training Script

```bash
# Train ant_forward with default config
python notebooks/train_with_config.py --env antforward

# Train ant_jump with default config
python notebooks/train_with_config.py --env antjump

# Train ant_flip with default config
python notebooks/train_with_config.py --env antflip
```

### Overriding Configuration Values

```bash
# Override specific hyperparameters
python notebooks/train_with_config.py --env antforward --lr 1e-4 --num_envs 4096

# Override environment-specific parameters
python notebooks/train_with_config.py --env antjump --episode_length 2000
```

### Using Configurations in Custom Scripts

```python
from configs.ant_forward_config import TRAINING_CONFIG, ENV_CONFIG

# Use the configuration
env = AntForward(min_forward_velocity=ENV_CONFIG['min_forward_velocity'])

make_policy, params, _ = ppo_train.train(
    environment=env,
    num_timesteps=TRAINING_CONFIG['total_env_steps'],
    episode_length=TRAINING_CONFIG['episode_length'],
    # ... other parameters
)
```

## Environment-Specific Tuning Suggestions

### AntForward
- **Learning Rate**: If training is unstable, try reducing to 1e-4
- **Min Forward Velocity**: Adjust between 0.3-1.0 based on desired speed
- **Num Environments**: Increase to 4096 for faster training if compute allows

### AntJump
- **Target Jump Height**: Adjust between 0.5-2.0 based on desired height
- **Episode Length**: May need to increase for higher jump targets
- **Entropy Cost**: Increase to 5e-4 if exploration is insufficient

### AntFlip
- **Min Flip Velocity**: Adjust between 0.5-3.0 based on desired flip speed
- **Episode Length**: May need to increase for complex flip sequences
- **Learning Rate**: May need to reduce for more stable training

## Model Outputs

Training will generate:
- **Model file**: `ppo_{env}_{param}_model.pkl`
- **Videos**: `ppo_{env}_{param}_video_{i}.html` (i=0,1,2,3,4)

Examples:
- `ppo_antforward_vel0.5_model.pkl`
- `ppo_antjump_h1.0_model.pkl`
- `ppo_antflip_flipvel1.0_model.pkl`

## Evaluation

After training, you can evaluate the trained policies using the `eval_ppo.py` script:

```bash
python notebooks/eval_ppo.py --model_path ppo_antforward_vel0.5_model.pkl --num_episodes 10
```

## Notes

- All environments use the same base hyperparameters for consistency
- Environment-specific parameters are clearly documented in each config file
- The configurations are designed to be easily modifiable for experimentation
- Training typically takes several hours depending on your hardware
