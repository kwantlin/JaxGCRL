# PPO Policy Evaluation Script

This script (`eval_ppo.py`) loads a trained PPO policy and collects state-action trajectories for evaluation and analysis.

## Usage

### Basic Usage
```bash
# Activate the JAX environment
conda activate jaxgcrl

# Run evaluation with default settings
python notebooks/eval_ppo.py

# Run with custom parameters
python notebooks/eval_ppo.py --num_episodes 10 --episode_length 1000 --save_trajectories
```

### Command Line Arguments

- `--model_path`: Path to the trained PPO model file (default: `/home/kw2960/JaxGCRL/notebooks/ppo_antjump_h1.0_model.pkl`)
- `--num_episodes`: Number of episodes to collect (default: 5)
- `--episode_length`: Maximum length of each episode (default: 1000)
- `--seed`: Random seed for reproducibility (default: 0)
- `--save_trajectories`: Flag to save collected trajectories to a pickle file

### Example Commands

```bash
# Quick evaluation with 2 episodes
python notebooks/eval_ppo.py --num_episodes 2

# Comprehensive evaluation with trajectory saving
python notebooks/eval_ppo.py --num_episodes 20 --episode_length 2000 --save_trajectories

# Evaluate with different random seed
python notebooks/eval_ppo.py --num_episodes 5 --seed 42
```

## Output

The script provides:

1. **Console Output**: Real-time progress and summary statistics
2. **Trajectory Data** (if `--save_trajectories` is used): A pickle file containing:
   - `observations`: State observations (shape: [episode_length, observation_dim])
   - `actions`: Actions taken (shape: [episode_length, action_dim])
   - `rewards`: Rewards received (shape: [episode_length])
   - `dones`: Episode termination flags (shape: [episode_length])
   - `infos`: Additional environment metrics

## Trajectory Data Structure

Each trajectory is a dictionary with the following structure:
```python
{
    'observations': np.array,  # Shape: (episode_length, 27) for AntJump
    'actions': np.array,       # Shape: (episode_length, 8) for AntJump
    'rewards': np.array,       # Shape: (episode_length,)
    'dones': np.array,         # Shape: (episode_length,)
    'infos': list              # List of dictionaries with metrics
}
```

## Loading Trajectories

```python
import pickle

# Load trajectories
with open('ppo_antjump_h1.0_model_trajectories.pkl', 'rb') as f:
    trajectories = pickle.load(f)

# Access first trajectory
first_traj = trajectories[0]
observations = first_traj['observations']
actions = first_traj['actions']
rewards = first_traj['rewards']

print(f"Episode length: {len(observations)}")
print(f"Total reward: {np.sum(rewards):.2f}")
```

## Supported Environments

Currently supports:
- `AntJump`: Ant jumping to target height
- (Can be extended to support other environments like `AntForward`, `AntFlip`, etc.)

## Environment Detection

The script automatically detects the environment type from the model filename:
- `antjump` in filename → `AntJump` environment
- Target height extracted from filename (e.g., `h1.0` → target_height=1.0)

## Performance Notes

- Uses JIT compilation for efficient execution
- Supports deterministic policy evaluation
- Handles episode termination gracefully
- Provides comprehensive metrics and statistics
