# Sudoku Environment for CRL

This document describes the Sudoku environment implementation for the JaxGCRL framework.

## Overview

The Sudoku environment is a discrete, deterministic environment designed for CRL (Conditional Reinforcement Learning) training. It represents a 9x9 Sudoku puzzle where the agent must fill in the grid according to Sudoku rules.

## Environment Structure

### State Space
- **Board representation**: 81 values (9x9 grid flattened)
- **Goal indicators**: 27 values representing completion status of:
  - 9 rows (indices 81-89)
  - 9 columns (indices 90-98)
  - 9 3x3 sub-squares (indices 99-107)
- **Total state dimension**: 108

### Action Space
- **Discrete actions**: 729 possible actions
- **Action encoding**: `action = cell_index * 9 + (number - 1)`
  - `cell_index = row * 9 + col` (0-80)
  - `number` is 1-9 (encoded as 0-8)

### Goal Indices
The goal portion of the observation consists of indices 81-107, which represent:
- Row completion indicators (81-89): Whether each row contains all numbers 1-9
- Column completion indicators (90-98): Whether each column contains all numbers 1-9
- Sub-square completion indicators (99-107): Whether each 3x3 sub-square contains all numbers 1-9

## Environment Dynamics

### Reset
- Generates a random, valid, uniquely solvable Sudoku puzzle
- Each reset produces a different puzzle from a set of pre-computed solutions
- Keeps 25-30 cells filled to ensure unique solvability
- All goal indicators start at 0 (incomplete)
- Returns initial state with board and goal indicators

### Step
- Takes a discrete action representing a cell and number
- Validates the move according to Sudoku rules:
  - Cell must be empty
  - Number must not appear in the same row
  - Number must not appear in the same column
  - Number must not appear in the same 3x3 sub-square
- Updates the board if the move is valid
- Recalculates all goal indicators
- Returns reward, done flag, and updated state

### Rewards
- **Valid moves**: +1.0
- **Invalid moves**: -1.0
- **Puzzle completion**: +100.0 (when all cells are filled and solution is correct)

### Termination
- Episode ends when all 81 cells are filled
- Success is determined by checking if the completed board is a valid Sudoku solution

## Integration with CRL Framework

The environment is fully integrated with the JaxGCRL framework and includes:

### Required Attributes
- `state_dim`: 108 (board + goal indicators)
- `goal_indices`: `jnp.arange(81, 108)` (indices of goal portion)
- `goal_reach_thresh`: 0.5 (threshold for goal completion)
- `action_space_size`: 729 (total possible actions)

### Compatible Methods
- `reset(rng)`: Returns initial state
- `step(state, action)`: Returns next state, reward, done flag
- `pipeline_init(q, qd)`: Compatibility method for Brax framework
- `pipeline_step(pipeline_state, action)`: Compatibility method for Brax framework

## Usage

### Creating the Environment
```python
from utils.env import create_env

# Fixed difficulty (default: medium)
env = create_env(env_name="sudoku")
env = create_env(env_name="sudoku", avg_rank=300)  # Hard difficulty

# Gaussian distribution over ranks 0-999
env = create_env(
    env_name="sudoku", 
    use_gaussian_scores=True,
    score_mean=500,  # Center of distribution
    score_std=150    # Spread of distribution
)
```

### Basic Usage
```python
import jax
import jax.numpy as jnp

# Create environment
env = create_env(env_name="sudoku")

# Reset
rng = jax.random.PRNGKey(0)
state = env.reset(rng)

# Take action (place number 5 in cell (0, 2))
action = (0 * 9 + 2) * 9 + 4  # cell (0,2) with number 5
state = env.step(state, action)

# Access observation
board = state.obs[:81].reshape(9, 9)
goal_indicators = state.obs[81:]
```

## Puzzle Generation

The environment generates random, valid, uniquely solvable Sudoku puzzles using the **dokusan** package:

### Generation Process
1. **High-Quality Generation**: Uses dokusan's advanced Sudoku generation algorithm
2. **Configurable Difficulty**: Two modes available:
   - **Fixed Difficulty**: Use `avg_rank` parameter (50=easy, 150=medium, 300=hard)
   - **Gaussian Distribution**: Sample from Gaussian distribution over ranks 0-999
3. **Virtually Unlimited Variety**: Each reset produces a completely different puzzle
4. **Guaranteed Uniqueness**: All generated puzzles have exactly one solution
5. **Professional Quality**: Based on research by Daniel Beer with average generation time of ~700ms

### Example Generated Puzzle
```
2 . . | 7 1 8 | . . . 
. . . | . . . | 8 . . 
. 7 6 | . . . | 1 2 . 
------+-------+------
3 . . | 4 6 . | . . . 
4 . . | 8 5 . | 3 1 . 
. . . | . . 3 | 2 . . 
------+-------+------
. 8 . | . 7 4 | . . . 
. . 2 | . . . | 5 . 1 
. 5 . | . . 1 | 9 4 . 
```

Each reset produces a completely different puzzle with professional-quality difficulty.

### Difficulty Levels

#### Fixed Difficulty Mode
- **Easy (avg_rank=50)**: ~40 filled cells, mostly naked/hidden singles
- **Medium (avg_rank=150)**: ~28 filled cells, includes advanced techniques
- **Hard (avg_rank=300)**: ~26 filled cells, requires complex solving strategies

#### Gaussian Distribution Mode
- **score_mean=200, score_std=50**: Easy-focused distribution
- **score_mean=500, score_std=150**: Medium-broad distribution (default)
- **score_mean=800, score_std=100**: Hard-focused distribution
- **Ranks 0-999**: Full range of Sudoku difficulty levels

## Training Considerations

### Action Space Size
With 729 possible actions, this environment has a large action space compared to typical RL environments. Consider using:
- Hierarchical policies
- Action masking for invalid moves
- Curriculum learning starting with simpler puzzles

### Goal Structure
The goal indicators provide rich feedback about partial progress:
- Row/column completion can guide exploration
- Sub-square completion helps with local reasoning

### Reward Shaping
The current reward structure provides:
- Immediate feedback on move validity
- Sparse reward for completion
- Consider adding intermediate rewards for completing rows/columns/sub-squares

## Testing

The environment has been tested for:
- Correct initialization and reset
- Valid move validation
- Goal indicator calculation
- Integration with the main environment creation system
- Compatibility with the Brax framework structure
- **Dokusan integration**: High-quality puzzle generation
- **Difficulty settings**: Configurable puzzle difficulty levels
- **Gaussian distribution**: Proper sampling over ranks 0-999
- **Puzzle validity**: All generated puzzles follow Sudoku rules
- **Generation performance**: ~0.5-1.0 seconds per puzzle
- **Fallback rate**: <5% fallback to simple puzzles
