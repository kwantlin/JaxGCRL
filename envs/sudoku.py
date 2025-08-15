import jax
import jax.numpy as jnp
from brax.envs.base import State
from brax import base
from dokusan import generators, stats
import numpy as np


class Sudoku:
    """
    A discrete, deterministic Sudoku environment for CRL training.
    
    The environment represents a 9x9 Sudoku grid where:
    - State: 9x9 grid (81 values) + goal indicators (27 values)
    - Actions: Discrete actions to place numbers 1-9 in cells
    - Goal: Complete the Sudoku puzzle correctly
    
    Goal indicators include:
    - 9 row completion indicators
    - 9 column completion indicators  
    - 9 3x3 sub-square completion indicators
    
    Puzzle Generation:
    - Uses dokusan package for high-quality Sudoku generation
    - Each reset generates a random, valid, uniquely solvable Sudoku puzzle
    - Two difficulty modes:
      - Fixed difficulty via avg_rank parameter
      - Gaussian distribution over ranks 0-999
    - Virtually unlimited puzzle variety
    """
    
    def __init__(self, **kwargs):
        self.board_size = 9
        self.action_space_size = 81 * 9  # 81 cells * 9 possible numbers
        self.state_dim = 81 + 27  # board + goal indicators (no diagonals)
        self.goal_indices = jnp.arange(81, 108)  # indices 81-107 are goal indicators
        self.goal_reach_thresh = 0.5
        
        # Initialize empty board (0 represents empty cell)
        self.empty_board = jnp.zeros((9, 9), dtype=jnp.int32)
        
        # Difficulty settings for puzzle generation
        self.use_gaussian_scores = kwargs.get('use_gaussian_scores', False)
        if self.use_gaussian_scores:
            self.score_mean = kwargs.get('score_mean', 500)
            self.score_std = kwargs.get('score_std', 150)
            self.min_rank = kwargs.get('min_rank', 0)
            self.max_rank = kwargs.get('max_rank', 999)
        else:
            self.avg_rank = kwargs.get('avg_rank', 150)  # Default to medium difficulty
        
    def reset(self, rng: jax.Array) -> State:
        """Reset the environment to an initial state with a partially filled Sudoku board."""
        rng, rng1 = jax.random.split(rng, 2)
        
        # Generate a random valid Sudoku puzzle using dokusan
        board = self._generate_puzzle(rng1)
        
        # Flatten the board for the state
        flat_board = board.flatten()
        
        # Calculate goal indicators
        goal_indicators = self._calculate_goal_indicators(board)
        
        # Combine board and goal indicators
        obs = jnp.concatenate([flat_board, goal_indicators])
        
        # Initialize metrics
        reward, done = jnp.zeros(2)
        metrics = {
            "reward": reward,
            "success": jnp.array(0.0),
            "success_easy": jnp.array(0.0),
            "cells_filled": jnp.array(0.0),
            "rows_complete": jnp.array(0.0),
            "cols_complete": jnp.array(0.0),
            "squares_complete": jnp.array(0.0),
        }
        
        # Create a dummy pipeline_state (not used in this discrete environment)
        # We need to create a minimal pipeline state for compatibility
        dummy_q = jnp.zeros(1)
        dummy_qd = jnp.zeros(1)
        pipeline_state = base.State(
            q=dummy_q,
            qd=dummy_qd,
            x=base.Transform.zero((1,)),
            xd=base.Motion.zero((1,)),
            contact=None,
        )
        
        return State(pipeline_state, obs, reward, done, metrics)
    
    def step(self, state: State, action: jax.Array) -> State:
        """Take a step in the environment."""
        # Decode action: action is an integer representing cell and number
        cell_idx = action // 9  # Which cell (0-80)
        number = (action % 9) + 1  # Which number (1-9)
        
        # Convert cell index to row, col
        row = cell_idx // 9
        col = cell_idx % 9
        
        # Get current board from observation
        board_flat = state.obs[:81]
        board = board_flat.reshape(9, 9)
        
        # Check if the move is valid
        is_valid = self._is_valid_move(board, row, col, number)
        
        # Apply the move if valid
        if is_valid:
            board = board.at[row, col].set(number)
        
        # Flatten the board
        new_board_flat = board.flatten()
        
        # Calculate new goal indicators
        goal_indicators = self._calculate_goal_indicators(board)
        
        # Create new observation
        obs = jnp.concatenate([new_board_flat, goal_indicators])
        
        # Calculate reward and success
        cells_filled = jnp.sum(board != 0)
        total_cells = 81
        completion_ratio = cells_filled / total_cells
        
        # Check if puzzle is complete and correct
        is_complete = cells_filled == total_cells
        is_correct = self._is_solution_correct(board) if is_complete else False
        success = jnp.array(is_complete and is_correct, dtype=float)
        
        # Success easy: when 50% of cells are filled
        success_easy = jnp.array(completion_ratio >= 0.5, dtype=float)
        
        # Reward: positive for valid moves, negative for invalid moves
        reward = jnp.where(is_valid, 1.0, -1.0)
        
        # Additional reward for completion
        reward = jnp.where(success, 100.0, reward)
        
        # Done when puzzle is complete
        done = jnp.array(is_complete, dtype=float)
        
        # Update metrics
        metrics = {
            "reward": reward,
            "success": success,
            "success_easy": success_easy,
            "cells_filled": jnp.array(cells_filled, dtype=float),
            "rows_complete": jnp.sum(goal_indicators[:9]),
            "cols_complete": jnp.sum(goal_indicators[9:18]),
            "squares_complete": jnp.sum(goal_indicators[18:27]),
        }
        
        # Keep the same pipeline_state since this is a discrete environment
        return state.replace(obs=obs, reward=reward, done=done, metrics=metrics)
    
    def _is_valid_move(self, board: jax.Array, row: int, col: int, number: int) -> bool:
        """Check if placing a number in a cell is valid according to Sudoku rules."""
        # Check if cell is empty
        if board[row, col] != 0:
            return False
        
        # Check row
        if jnp.any(board[row, :] == number):
            return False
        
        # Check column
        if jnp.any(board[:, col] == number):
            return False
        
        # Check 3x3 sub-square
        start_row = 3 * (row // 3)
        start_col = 3 * (col // 3)
        sub_square = board[start_row:start_row+3, start_col:start_col+3]
        if jnp.any(sub_square == number):
            return False
        
        return True
    
    def _is_solution_correct(self, board: jax.Array) -> bool:
        """Check if the completed board is a valid Sudoku solution."""
        # Check rows
        for row in range(9):
            if not jnp.array_equal(jnp.sort(board[row]), jnp.arange(1, 10)):
                return False
        
        # Check columns
        for col in range(9):
            if not jnp.array_equal(jnp.sort(board[:, col]), jnp.arange(1, 10)):
                return False
        
        # Check 3x3 sub-squares
        for i in range(3):
            for j in range(3):
                sub_square = board[3*i:3*i+3, 3*j:3*j+3].flatten()
                if not jnp.array_equal(jnp.sort(sub_square), jnp.arange(1, 10)):
                    return False
        
        return True
    
    def _calculate_goal_indicators(self, board: jax.Array) -> jax.Array:
        """Calculate goal indicators for rows, columns, and sub-squares."""
        indicators = []
        
        # Row completion indicators (9 values)
        for row in range(9):
            row_values = board[row, :]
            is_complete = jnp.logical_and(
                jnp.all(row_values != 0),  # All cells filled
                jnp.array_equal(jnp.sort(row_values), jnp.arange(1, 10))  # Valid Sudoku row
            )
            indicators.append(jnp.array(is_complete, dtype=float))
        
        # Column completion indicators (9 values)
        for col in range(9):
            col_values = board[:, col]
            is_complete = jnp.logical_and(
                jnp.all(col_values != 0),  # All cells filled
                jnp.array_equal(jnp.sort(col_values), jnp.arange(1, 10))  # Valid Sudoku column
            )
            indicators.append(jnp.array(is_complete, dtype=float))
        
        # 3x3 sub-square completion indicators (9 values)
        for i in range(3):
            for j in range(3):
                sub_square = board[3*i:3*i+3, 3*j:3*j+3].flatten()
                is_complete = jnp.logical_and(
                    jnp.all(sub_square != 0),  # All cells filled
                    jnp.array_equal(jnp.sort(sub_square), jnp.arange(1, 10))  # Valid Sudoku sub-square
                )
                indicators.append(jnp.array(is_complete, dtype=float))
        
        return jnp.array(indicators)
    
    def _generate_puzzle(self, rng):
        """Generate a valid Sudoku puzzle using dokusan."""
        # Use the random seed to ensure reproducibility
        seed = int(jax.random.randint(rng, (1,), 0, 1000000)[0])  # Use smaller range to avoid overflow
        
        # Determine the target rank for this puzzle
        if self.use_gaussian_scores:
            # Sample from Gaussian distribution and clamp to valid range
            rng, rng_rank = jax.random.split(rng, 2)
            rank_sample = jax.random.normal(rng_rank, (1,))[0] * self.score_std + self.score_mean
            target_rank = int(jnp.clip(rank_sample, self.min_rank, self.max_rank))
        else:
            target_rank = self.avg_rank
        
        # Generate a puzzle with the specified difficulty
        # We'll try to generate a puzzle with the desired rank
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                # Generate a random Sudoku puzzle
                sudoku = generators.random_sudoku(avg_rank=target_rank)
                
                # Convert dokusan Sudoku to numpy array
                puzzle = np.zeros((9, 9), dtype=np.int32)
                for cell in sudoku.cells():
                    row, col = cell.position.row, cell.position.column
                    value = cell.value if cell.value is not None else 0
                    puzzle[row, col] = value
                
                # Convert to JAX array
                puzzle = jnp.array(puzzle)
                
                return puzzle
                
            except Exception as e:
                # If generation fails, try again with a different seed
                if attempt == max_attempts - 1:
                    # Fallback to a simple puzzle if all attempts fail
                    print(f"Warning: Failed to generate puzzle with dokusan, using fallback. Error: {e}")
                    return self._generate_fallback_puzzle(rng)
        
        # This should never be reached, but just in case
        return self._generate_fallback_puzzle(rng)
    
    def _generate_fallback_puzzle(self, rng):
        """Generate a simple fallback puzzle if dokusan fails."""
        # A simple valid Sudoku puzzle as fallback
        fallback_puzzle = jnp.array([
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9]
        ], dtype=jnp.int32)
        
        return fallback_puzzle
    
    def get_obs(self, state: State) -> jax.Array:
        """Get the observation from the state."""
        return state.obs
    
    def pipeline_init(self, q: jax.Array, qd: jax.Array) -> base.State:
        """Initialize pipeline state (not used in discrete environment)."""
        return base.State(
            q=q,
            qd=qd,
            x=base.Transform.zero((1,)),
            xd=base.Motion.zero((1,)),
            contact=None,
        )
    
    def pipeline_step(self, pipeline_state: base.State, action: jax.Array) -> base.State:
        """Step the pipeline (not used in discrete environment)."""
        return pipeline_state
