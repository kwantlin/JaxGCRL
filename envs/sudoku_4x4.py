import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.base import State
from brax import base
import os


class Sudoku4x4:
    """
    A discrete, deterministic 4x4 Sudoku environment for CRL training.
    
    The environment represents a 4x4 Sudoku grid where:
    - State: 4x4 grid (16 values) + goal indicators (16 values)
    - Actions: Discrete actions to place numbers 1-4 in cells
    - Goal: Complete the Sudoku puzzle correctly
    
    Goal indicators include:
    - 4 row completion indicators
    - 4 column completion indicators  
    - 4 2x2 sub-square completion indicators
    """
    
    def __init__(self, **kwargs):
        # Environment parameters for 4x4 Sudoku
        self.board_size = 4
        # Action space: 16 cells * 5 actions (4 numbers + 1 erase)
        self.action_space_size = 16 * 5  # 16 cells * 5 possible actions (1-4 + erase)
        self.action_size = 16 * 5  # 16 cells * 5 possible actions (1-4 + erase)
        self.state_dim = 16  # just the Sudoku board
        self.observation_size = 32  # state (16) + goal (16)
        self.goal_indices = jnp.arange(0, 16)  # indices 0-15 point to the board portion of the state
        self.goal_reach_thresh = 0.5
        
        # Load the 4x4 Sudoku dataset
        self.dataset_path = "envs/4x4_sudoku_unique_puzzles.csv"
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"4x4 Sudoku dataset not found at {self.dataset_path}")
        
        # Load dataset into memory
        print("Loading 4x4 Sudoku dataset...")
        import pandas as pd
        self.df = pd.read_csv(self.dataset_path, dtype={'Puzzle': str, 'Solution': str})
        self.num_puzzles = len(self.df)
        print(f"Loaded {self.num_puzzles:,} 4x4 Sudoku puzzles")
        
        # Pre-load a subset of puzzles into JAX arrays for efficiency
        # This gives us JAX compatibility without loading all 1M puzzles
        subset_size = min(10000, self.num_puzzles)  # Use first 10K puzzles
        print(f"Pre-loading {subset_size:,} puzzles into JAX arrays...")
        
        self.puzzle_subset = []
        self.solution_subset = []
        
        for i in range(subset_size):
            puzzle_str = self.df.iloc[i]['Puzzle']
            solution_str = self.df.iloc[i]['Solution']
            
            puzzle_grid = self._string_to_grid(puzzle_str)
            solution_grid = self._string_to_grid(solution_str)
            
            self.puzzle_subset.append(puzzle_grid)
            self.solution_subset.append(solution_grid)
        
        # Convert to JAX arrays
        self.puzzle_subset = jnp.array(self.puzzle_subset)  # Shape: (subset_size, 4, 4)
        self.solution_subset = jnp.array(self.solution_subset)  # Shape: (subset_size, 4, 4)
        self.subset_size = subset_size
        
        print("4x4 Sudoku environment initialized")
    
    def _string_to_grid(self, puzzle_str: str) -> jnp.ndarray:
        """
        Convert a 16-character string to a 4x4 grid.
        
        Args:
            puzzle_str: 16-character string representing the puzzle
            
        Returns:
            4x4 JAX array where 0=empty, 1-4=filled numbers
        """
        if len(puzzle_str) != 16:
            raise ValueError(f"Puzzle string must be 16 characters long, got {len(puzzle_str)}")
        
        # Convert string to list of integers
        grid_flat = [int(c) for c in puzzle_str]
        
        # Reshape to 4x4
        grid = jnp.array(grid_flat, dtype=jnp.int32).reshape(4, 4)
        
        return grid
    
    def _generate_puzzle_from_dataset(self, rng):
        """
        Generate a puzzle by sampling from the pre-loaded subset efficiently.
        
        Returns:
            board: 4x4 JAX array with the initial puzzle state
            solution: 4x4 JAX array with the complete solution
        """
        # Sample a random puzzle index from the pre-loaded subset
        puzzle_idx = jax.random.randint(rng, (1,), 0, self.subset_size)[0]
        
        # Get the puzzle and solution from the pre-loaded JAX arrays
        board = self.puzzle_subset[puzzle_idx]
        solution = self.solution_subset[puzzle_idx]
        
        return board, solution
    
    def reset(self, rng: jax.Array) -> State:
        """Reset the environment to an initial state with a partially filled 4x4 Sudoku board."""
        rng1, rng2 = jax.random.split(rng, 2)
        
        # Generate a puzzle from the dataset
        board, solution = self._generate_puzzle_from_dataset(rng1)
        
        # Store the solution and original board in the state info for validation
        info = {"solution": solution, "original_board": board}
        
        # Create pipeline_state with board in q and solution in qd (following ant.py pattern)
        q = board.flatten()  # 16 values: just the board
        qd = solution.flatten()  # 16 values: the complete solution (stored in qd for convenience)
        pipeline_state = self.pipeline_init(q, qd)
        
        # Get observation using _get_obs method (following ant.py pattern)
        obs = self._get_obs(pipeline_state)
        
        # Calculate initial metrics
        cells_filled = jnp.sum(board != 0)
        total_cells = 16
        completion_ratio = cells_filled / total_cells
        
        # Initial puzzle characteristics
        initial_clues = cells_filled
        puzzle_difficulty = 1.0 - (initial_clues / total_cells)  # Higher = harder (fewer clues)
        
        # Distance to solution at start
        cells_different = jnp.sum(board != solution)
        initial_distance_to_solution = cells_different / total_cells
        
        metrics = {
            # Initialize step metrics to 0 (these will be accumulated over the episode)
            "valid_move": jnp.array(0.0, dtype=float),
            "correct_move": jnp.array(0.0, dtype=float),
            "incorrect_move": jnp.array(0.0, dtype=float),
            "erase_action": jnp.array(0.0, dtype=float),
            "moves_made": jnp.array(0.0, dtype=float),
            
            # Initialize progress metrics (these will be averaged over the episode)
            "completion_percentage": jnp.array(completion_ratio * 100.0, dtype=float),
            "distance_to_solution": jnp.array(initial_distance_to_solution, dtype=float),
            "success": jnp.array(0.0, dtype=float),  # No success at start
        }
        
        # Initialize reward and done for reset
        reward = jnp.array(0.0, dtype=float)
        done = jnp.array(0.0, dtype=float)
        
        state = State(pipeline_state, obs, reward, done, metrics, info=info)
        return state
    
    def step(self, state: State, action: jax.Array) -> State:
        """Take a step in the environment."""
        # Decode action: action is an integer representing cell and action type
        cell_idx = action // 5  # Which cell (0-15)
        action_type = action % 5  # Which action (0-4: 0=erase, 1-4=numbers)
        
        # Convert cell index to row, col (ensure they are integers)
        row = jnp.array(cell_idx // 4, dtype=jnp.int32)
        col = jnp.array(cell_idx % 4, dtype=jnp.int32)
        
        # Get current board from pipeline_state.q (following ant.py pattern)
        board_flat = state.pipeline_state.q  # Shape: (16,) or (batch_size, 16)
        board = board_flat.reshape(-1, 4, 4)  # Shape: (4, 4) or (batch_size, 4, 4)
        
        # Get the original puzzle (initial state) to check if cell was originally filled
        original_board = state.info.get("original_board", board)  # Fallback to current board if not stored
        
        # Ensure original_board has the same shape as board
        if len(board.shape) == 3 and len(original_board.shape) == 2:
            original_board = original_board.reshape(1, 4, 4)
        elif len(board.shape) == 2 and len(original_board.shape) == 2:
            original_board = original_board.reshape(1, 4, 4)
            board = board.reshape(1, 4, 4)
        
        # Determine if this is an erase action or a number placement
        is_erase = (action_type == 0)
        number = jnp.where(is_erase, 0, action_type)  # 0 for erase, 1-4 for numbers
        
        # Ensure number is a scalar for board update
        number_scalar = jnp.array(number, dtype=jnp.int32)
        
        # Get solution for validation
        solution = state.info["solution"]
        
        # Check if the move is valid (handle both erase and number placement)
        # For number placement: use existing validation logic
        number_valid = self._is_valid_move(board, row, col, number)
        
        # For erase: use a separate validation method
        erase_valid = self._is_valid_erase(board, original_board, row, col)
        
        # Combine validation based on action type
        is_valid = jnp.where(is_erase, erase_valid, number_valid)
        
        # Check if this is the correct move according to the solution
        # Erase actions are never "correct" in solution sense
        is_correct = jnp.where(is_erase, 
                              jnp.array(False, dtype=bool), 
                              jnp.logical_and(is_valid, (solution[row, col] == number)))
        
        # Apply the move if valid (handle batched inputs properly)
        # Convert 2D indices to 1D indices for batched indexing
        batch_indices = jnp.arange(board.shape[0]) if len(board.shape) == 3 else jnp.array([0])
        flat_indices = row * 4 + col
        
        # Reshape board to 2D for easier indexing
        board_flat = board.reshape(-1, 16)
        
        # Update the board using scatter
        board_flat = board_flat.at[batch_indices, flat_indices].set(
            jnp.where(is_valid, number_scalar, board_flat[batch_indices, flat_indices])
        )
        
        # Reshape back to original shape
        new_board = board_flat.reshape(board.shape)
        new_board_flat = new_board.flatten()
        
        # Get the solution from the current state (stored in qd)
        solution_flat = state.pipeline_state.qd  # Shape: (16,) or (batch_size, 16)
        
        # Update pipeline_state with new board and keep solution (following ant.py pattern)
        new_q = new_board_flat
        new_qd = solution_flat  # Keep the solution in qd
        pipeline_state = state.pipeline_state.replace(q=new_q, qd=new_qd)
        
        # Get new observation using _get_obs method (following ant.py pattern)
        obs = self._get_obs(pipeline_state)
        
        # Calculate reward and success
        cells_filled = jnp.sum(new_board != 0)
        total_cells = 16
        completion_ratio = cells_filled / total_cells
        
        # Check if puzzle is complete and correct
        is_complete = cells_filled == total_cells
        is_correct_solution = self._is_solution_correct(new_board)
        success = jnp.array(jnp.logical_and(is_complete, is_correct_solution), dtype=float)
        
        # Success easy: when 50% of cells are filled
        success_easy = jnp.array(completion_ratio >= 0.5, dtype=float)
        
        # Reward: positive for valid moves, negative for invalid moves
        # Bonus for correct moves according to solution
        base_reward = jnp.where(is_valid, 1.0, -1.0)
        correct_bonus = jnp.where(is_correct, 0.5, 0.0)
        reward = base_reward + correct_bonus
        
        # Additional reward for completion
        reward = jnp.where(success, 100.0, reward)
        
        # Ensure reward is scalar by taking mean if it's a vector
        reward = jnp.mean(reward) if reward.ndim > 0 else reward
        
        # Done when puzzle is complete
        done = jnp.array(is_complete, dtype=float)
        done = jnp.mean(done) if done.ndim > 0 else done  # Ensure done is scalar
        
        # Calculate additional metrics
        # Move quality metrics - ensure they are scalars
        valid_move = jnp.array(is_valid, dtype=float)
        correct_move = jnp.array(is_correct, dtype=float)
        incorrect_move = jnp.array(jnp.logical_and(is_valid, jnp.logical_not(is_correct)), dtype=float)
        erase_action = jnp.array(is_erase, dtype=float)
        
        # Ensure all metrics are scalars for JAX compatibility
        valid_move = jnp.mean(valid_move) if valid_move.ndim > 0 else valid_move
        correct_move = jnp.mean(correct_move) if correct_move.ndim > 0 else correct_move
        incorrect_move = jnp.mean(incorrect_move) if incorrect_move.ndim > 0 else incorrect_move
        erase_action = jnp.mean(erase_action) if erase_action.ndim > 0 else erase_action
        
        # Progress metrics
        cells_remaining = total_cells - cells_filled
        completion_percentage = completion_ratio * 100.0
        
        # Distance to solution (how many cells differ from solution)
        cells_different = jnp.sum(new_board != solution)
        distance_to_solution = cells_different / total_cells
        
        # Efficiency metrics
        # These will be accumulated over the episode
        moves_made = jnp.array(1.0, dtype=float)  # This step counts as 1 move
        
        # Puzzle-specific metrics
        initial_cells = jnp.sum(state.info["solution"] != 0)  # Should be 16, but let's be safe
        cells_to_fill = total_cells - initial_cells
        
        # Milestone metrics
        quarter_complete = jnp.array(completion_ratio >= 0.25, dtype=float)
        half_complete = jnp.array(completion_ratio >= 0.5, dtype=float)
        three_quarters_complete = jnp.array(completion_ratio >= 0.75, dtype=float)
        almost_complete = jnp.array(completion_ratio >= 0.9, dtype=float)
        
        # Update metrics (following ant.py pattern)
        # For metrics that should be accumulated over the episode (like moves_made, valid_move, etc.)
        # For metrics that should be the final value (like completion_percentage, success, etc.)
        
        # Update metrics that work well when averaged over episode steps
        state.metrics.update(
            # Move quality metrics (these accumulate meaningfully over the episode)
            valid_move=valid_move,
            correct_move=correct_move,
            incorrect_move=incorrect_move,
            erase_action=erase_action,
            
            # Efficiency metrics (these accumulate over the episode)
            moves_made=moves_made,
            
            # Progress metrics (comparing current state to goal at each step)
            completion_percentage=completion_percentage,  # Current completion vs goal
            distance_to_solution=distance_to_solution,    # How far from solution
            success=success,                              # Whether puzzle is solved
        )
        
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )
    
    def _is_valid_move(self, board, row, col, number):
        """Check if a move is valid for the given board."""
        # Handle both scalar and vectorized inputs
        if row.ndim == 0:  # Scalar input (from step method)
            return self._is_valid_move_scalar(board, row, col, number)
        else:  # Vectorized input (from vmap)
            return self._is_valid_move_vectorized(board, row, col, number)
    
    def _is_valid_move_scalar(self, board, row, col, number):
        """Check if a single move is valid (scalar version)."""
        # Handle different board shapes
        if len(board.shape) == 1:  # Flattened board
            # Convert to 2D for easier indexing
            board = board.reshape(4, 4)
        elif len(board.shape) == 3 and board.shape[0] == 1:
            board = board[0]  # Shape: (4, 4)
        
        # Check if cell is already filled
        cell_filled = (board[row, col] != 0)
        
        # Check row
        row_values = board[row, :]
        row_has_number = jnp.any(row_values == number)
        
        # Check column
        col_values = board[:, col]
        col_has_number = jnp.any(col_values == number)
        
        # Check 2x2 sub-square
        sub_row_start = 2 * (row // 2)
        sub_col_start = 2 * (col // 2)
        # Use dynamic slice for JAX compatibility
        from jax import lax
        sub_square = lax.dynamic_slice(board, (sub_row_start, sub_col_start), (2, 2))
        square_has_number = jnp.any(sub_square == number)
        
        # Combine all checks
        is_valid = jnp.logical_not(jnp.logical_or(jnp.logical_or(cell_filled, row_has_number), jnp.logical_or(col_has_number, square_has_number)))
        
        return is_valid
    
    def _is_valid_erase(self, board, original_board, row, col):
        """Check if an erase action is valid."""
        # Handle both scalar and vectorized inputs
        if row.ndim == 0:  # Scalar input
            return self._is_valid_erase_scalar(board, original_board, row, col)
        else:  # Vectorized input
            return self._is_valid_erase_vectorized(board, original_board, row, col)
    
    def _is_valid_erase_scalar(self, board, original_board, row, col):
        """Check if a single erase action is valid (scalar version)."""
        # Handle different board shapes
        if len(board.shape) == 1:  # Flattened board
            # Convert to 2D for easier indexing
            board = board.reshape(4, 4)
            original_board = original_board.reshape(4, 4)
        elif len(board.shape) == 3 and board.shape[0] == 1:
            board = board[0]  # Shape: (4, 4)
            original_board = original_board[0]  # Shape: (4, 4)
        
        # Check if cell is filled AND not part of the original puzzle
        cell_is_filled = (board[row, col] != 0)
        cell_was_original = (original_board[row, col] != 0)
        
        return jnp.logical_and(cell_is_filled, jnp.logical_not(cell_was_original))
    
    def _is_valid_erase_vectorized(self, board, original_board, row, col):
        """Check if erase actions are valid (vectorized version)."""
        # Handle different board shapes
        if len(board.shape) == 1:  # Flattened board
            # Already flattened, use directly
            board_flat = board
            original_flat = original_board
        elif len(board.shape) == 3 and board.shape[0] == 1:
            board = board[0]  # Shape: (4, 4)
            original_board = original_board[0]  # Shape: (4, 4)
            board_flat = board.flatten()  # Shape: (16,)
            original_flat = original_board.flatten()  # Shape: (16,)
        else:
            board_flat = board.flatten()  # Shape: (16,)
            original_flat = original_board.flatten()  # Shape: (16,)
        
        # Convert 2D indices to 1D for easier indexing
        flat_indices = row * 4 + col
        
        # Check if cells are filled AND not part of the original puzzle
        cell_is_filled = board_flat[flat_indices] != 0  # Shape: (64,)
        cell_was_original = original_flat[flat_indices] != 0  # Shape: (64,)
        
        return jnp.logical_and(cell_is_filled, jnp.logical_not(cell_was_original))
    
    def _is_valid_move_vectorized(self, board, row, col, number):
        """Check if moves are valid for the given board (vectorized version)."""
        # Understanding: This is called with a single board and vectorized row/col/number
        # The vmap is over the action space (64 possible actions)
        
        # Remove the batch dimension from board if it exists
        if len(board.shape) == 3 and board.shape[0] == 1:
            board = board[0]  # Shape: (4, 4)
        
        # Now board is (4, 4) and row/col/number are vectors of length 64
        # We need to handle this by broadcasting
        
        # Step 1: Check if cell is already filled
        # Convert 2D indices to 1D for easier indexing
        flat_indices = row * 4 + col
        board_flat = board.flatten()  # Shape: (16,)
        cell_values = board_flat[flat_indices]  # Shape: (64,)
        cell_filled = cell_values != 0  # Shape: (64,)
        
        # Step 2: Check row
        # For each action, check if the number exists in the corresponding row
        row_start_indices = row * 4  # Shape: (64,)
        row_indices = row_start_indices[:, None] + jnp.arange(4)  # Shape: (64, 4)
        row_values = board_flat[row_indices]  # Shape: (64, 4)
        row_has_number = jnp.any(row_values == number[:, None], axis=1)  # Shape: (64,)
        
        # Step 3: Check column
        # For each action, check if the number exists in the corresponding column
        col_indices = col[:, None] + jnp.arange(4) * 4  # Shape: (64, 4)
        col_values = board_flat[col_indices]  # Shape: (64, 4)
        col_has_number = jnp.any(col_values == number[:, None], axis=1)  # Shape: (64,)
        
        # Step 4: Check 2x2 sub-square
        # For each action, check if the number exists in the corresponding 2x2 square
        sub_row_start = 2 * (row // 2)  # Shape: (64,)
        sub_col_start = 2 * (col // 2)  # Shape: (64,)
        
        # Generate indices for the 2x2 sub-square for each action
        square_indices = []
        for i in range(2):
            for j in range(2):
                square_indices.append((sub_row_start + i) * 4 + (sub_col_start + j))
        
        square_indices = jnp.array(square_indices)  # Shape: (4, 64)
        square_values = board_flat[square_indices.T]  # Shape: (64, 4)
        square_has_number = jnp.any(square_values == number[:, None], axis=1)  # Shape: (64,)
        
        # Step 5: Combine results
        is_valid = jnp.logical_not(jnp.logical_or(jnp.logical_or(cell_filled, row_has_number), jnp.logical_or(col_has_number, square_has_number)))
        
        return is_valid
    
    def _is_solution_correct(self, board):
        """Check if the board is a valid complete 4x4 Sudoku solution."""
        # Understanding: This is called with a single board
        # The vmap is over the action space, not over multiple environments
        
        # Remove the batch dimension from board if it exists
        if len(board.shape) == 3 and board.shape[0] == 1:
            board = board[0]  # Shape: (4, 4)
        
        # Now board is (4, 4) - we need to check if it's a valid Sudoku solution
        
        # Step 1: Check all rows
        # Sort each row and check if it equals [1,2,3,4]
        expected = jnp.arange(1, 5)  # [1,2,3,4]
        row_sorted = jnp.sort(board, axis=1)  # Sort each row
        rows_valid = jnp.all(row_sorted == expected, axis=1)  # Check each row
        all_rows_valid = jnp.all(rows_valid)  # All rows must be valid
        
        # Step 2: Check all columns
        # Sort each column and check if it equals [1,2,3,4]
        cols_valid = jnp.array(True, dtype=jnp.bool_)
        for col in range(4):
            col_values = board[:, col]
            col_sorted = jnp.sort(col_values)
            col_valid = jnp.array_equal(col_sorted, expected)
            cols_valid = jnp.logical_and(cols_valid, col_valid)
        all_cols_valid = cols_valid
        
        # Step 3: Check all 2x2 sub-squares
        # Check each 2x2 sub-square
        squares_valid = jnp.array(True, dtype=jnp.bool_)
        for i in range(2):
            for j in range(2):
                # Extract 2x2 sub-square
                sub_square = board[2*i:2*i+2, 2*j:2*j+2]
                sub_square_flat = sub_square.flatten()
                sub_square_sorted = jnp.sort(sub_square_flat)
                square_valid = jnp.array_equal(sub_square_sorted, expected)
                squares_valid = jnp.logical_and(squares_valid, square_valid)
        
        # Step 4: Combine all checks
        is_valid = jnp.logical_and(jnp.logical_and(all_rows_valid, all_cols_valid), squares_valid)
        
        return is_valid
    
    def _calculate_goal_indicators(self, board):
        """Calculate completion indicators for rows, columns, and 2x2 squares."""
        # Handle batched input by removing batch dimension if present
        if len(board.shape) == 3:
            board = board[0]  # Take first batch element
        
        indicators = []
        
        # Row completion indicators (4 values)
        for row in range(4):
            row_values = board[row, :]
            is_complete = jnp.logical_and(
                jnp.all(row_values != 0),  # All cells filled
                jnp.array_equal(jnp.sort(row_values), jnp.arange(1, 5))  # Valid Sudoku row
            )
            indicators.append(jnp.array(is_complete, dtype=float))
        
        # Column completion indicators (4 values)
        for col in range(4):
            col_values = board[:, col]
            is_complete = jnp.logical_and(
                jnp.all(col_values != 0),  # All cells filled
                jnp.array_equal(jnp.sort(col_values), jnp.arange(1, 5))  # Valid Sudoku column
            )
            indicators.append(jnp.array(is_complete, dtype=float))
        
        # 2x2 sub-square completion indicators (4 values)
        for i in range(2):
            for j in range(2):
                sub_square = board[2*i:2*i+2, 2*j:2*j+2].flatten()
                is_complete = jnp.logical_and(
                    jnp.all(sub_square != 0),  # All cells filled
                    jnp.array_equal(jnp.sort(sub_square), jnp.arange(1, 5))  # Valid Sudoku sub-square
                )
                indicators.append(jnp.array(is_complete, dtype=float))
        
        return jnp.array(indicators)
    
    def _count_complete_rows(self, board):
        """Count the number of complete rows."""
        # Check each row
        row_complete = jnp.zeros(4, dtype=jnp.bool_)
        for row in range(4):
            row_values = board[row, :]
            is_filled = jnp.all(row_values != 0)
            is_valid = jnp.array_equal(jnp.sort(row_values), jnp.arange(1, 5))
            row_complete = row_complete.at[row].set(
                jnp.logical_and(is_filled, is_valid)
            )
        return jnp.sum(row_complete)
    
    def _count_complete_cols(self, board):
        """Count the number of complete columns."""
        # Check each column
        col_complete = jnp.zeros(4, dtype=jnp.bool_)
        for col in range(4):
            col_values = board[:, col]
            is_filled = jnp.all(col_values != 0)
            is_valid = jnp.array_equal(jnp.sort(col_values), jnp.arange(1, 5))
            col_complete = col_complete.at[col].set(
                jnp.logical_and(is_filled, is_valid)
            )
        return jnp.sum(col_complete)
    
    def _count_complete_squares(self, board):
        """Count the number of complete 2x2 squares."""
        # Check each 2x2 square
        square_complete = jnp.zeros(4, dtype=jnp.bool_)
        square_idx = 0
        for i in range(2):
            for j in range(2):
                sub_square = board[2*i:2*i+2, 2*j:2*j+2].flatten()
                is_filled = jnp.all(sub_square != 0)
                is_valid = jnp.array_equal(jnp.sort(sub_square), jnp.arange(1, 5))
                square_complete = square_complete.at[square_idx].set(
                    jnp.logical_and(is_filled, is_valid)
                )
                square_idx += 1
        return jnp.sum(square_complete)
    
    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Get observation from pipeline state (following ant.py pattern)."""
        # Extract board from pipeline_state.q (16 dimensions)
        board_flat = pipeline_state.q  # Shape: (16,) or (batch_size, 16)
        
        # Extract solution from pipeline_state.qd (16 dimensions)
        solution_flat = pipeline_state.qd  # Shape: (16,) or (batch_size, 16)
        
        # Combine state and goal: state (16) + goal (16) = 32
        obs = jnp.concatenate([board_flat, solution_flat])
        return obs
    
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
