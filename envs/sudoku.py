import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.base import State
from brax import base
import os


class Sudoku:
    """
    A discrete, deterministic Sudoku environment for CRL training.
    
    Uses a dataset of 1.8M Sudoku puzzles with complete solution paths.
    Each puzzle is encoded as: [start_index, move1_row, move1_col, move1_val, move1_strategy, ...]
    
    The environment represents a 9x9 Sudoku grid where:
    - State: 9x9 grid (81 values) + goal indicators (27 values)
    - Actions: Discrete actions to place numbers 1-9 in cells
    - Goal: Complete the Sudoku puzzle correctly
    
    Goal indicators include:
    - 9 row completion indicators
    - 9 column completion indicators  
    - 9 3x3 sub-square completion indicators
    """
    
    def __init__(self, **kwargs):
        # Load the Sudoku dataset
        self.dataset_path = "/scratch/gpfs/kw2960/sudoku/sudoku-train-data.npy"
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Sudoku dataset not found at {self.dataset_path}")
        
        # Load dataset into memory
        print("Loading Sudoku dataset...")
        dataset = np.load(self.dataset_path)
        print(f"Loaded {dataset.shape[0]:,} puzzles with shape {dataset.shape}")
        
        # Pre-process dataset into JAX arrays for JIT compatibility
        print("Pre-processing dataset...")
        self.initial_boards = []
        self.solutions = []
        
        # Process a subset for now to avoid memory issues
        max_puzzles = min(10000, dataset.shape[0])  # Use first 10k puzzles
        for i in range(max_puzzles):
            puzzle_data = dataset[i]
            initial_board, solution = self._decode_puzzle_from_dataset(puzzle_data)
            self.initial_boards.append(initial_board)
            self.solutions.append(solution)
        
        # Convert to JAX arrays
        self.initial_boards = jnp.array(self.initial_boards, dtype=jnp.int32)
        self.solutions = jnp.array(self.solutions, dtype=jnp.int32)
        
        # Environment parameters
        self.board_size = 9
        self.action_space_size = 81 * 9  # 81 cells * 9 possible numbers
        self.action_size = 81 * 9  # 81 cells * 9 possible numbers
        self.state_dim = 81  # just the Sudoku board
        self.observation_size = 162  # state (81) + goal (81)
        self.goal_indices = jnp.arange(0, 81)  # indices 0-80 point to the board portion of the state
        self.goal_reach_thresh = 0.5
        
        # Dataset parameters
        self.num_puzzles = self.initial_boards.shape[0]
        
        print(f"Sudoku environment initialized with {self.num_puzzles:,} puzzles")
    
    def _decode_puzzle_from_dataset(self, puzzle_data):
        """
        Decode a puzzle from the dataset format.
        
        Args:
            puzzle_data: Array of shape (325,) containing [start_index, move1_row, move1_col, move1_val, move1_strategy, ...]
        
        Returns:
            board: 9x9 numpy array with the initial puzzle state
            solution: 9x9 numpy array with the complete solution
        """
        start_index = int(puzzle_data[0])
        moves = puzzle_data[1:]  # 324 values: 81 moves × 4 values each
        
        # Initialize empty board
        board = np.zeros((9, 9), dtype=np.int32)
        solution = np.zeros((9, 9), dtype=np.int32)
        
        # Apply all 81 moves to get the complete solution
        for i in range(81):
            move_idx = i * 4
            row = int(moves[move_idx])
            col = int(moves[move_idx + 1])
            val = int(moves[move_idx + 2])
            # strategy = int(moves[move_idx + 3])  # Not used
            
            solution[row, col] = val
            
            # Only apply the first 'start_index' moves to create the initial puzzle
            if i < start_index:
                board[row, col] = val
        
        return board, solution
    
    def _generate_puzzle(self, rng):
        """Generate a puzzle by sampling from the pre-processed dataset."""
        # Sample a random puzzle index
        puzzle_idx = jax.random.randint(rng, (1,), 0, self.num_puzzles)[0]
        
        # Get the pre-processed puzzle
        initial_board = self.initial_boards[puzzle_idx]
        solution = self.solutions[puzzle_idx]
        
        return initial_board, solution
    
    def reset(self, rng: jax.Array) -> State:
        """Reset the environment to an initial state with a partially filled Sudoku board."""
        rng1, rng2 = jax.random.split(rng, 2)
        
        # Generate a puzzle from the dataset
        board, solution = self._generate_puzzle(rng1)
        
        # Store the solution in the state info for validation
        info = {"solution": solution}
        
        # Create pipeline_state with board in q and solution in qd (following ant.py pattern)
        q = board.flatten()  # 81 values: just the board
        qd = solution.flatten()  # 81 values: the complete solution (stored in qd for convenience)
        pipeline_state = self.pipeline_init(q, qd)
        
        # Get observation using _get_obs method (following ant.py pattern)
        obs = self._get_obs(pipeline_state)
        
        # Calculate initial metrics
        cells_filled = jnp.sum(board != 0)
        total_cells = 81
        completion_ratio = cells_filled / total_cells
        
        # Initial puzzle characteristics
        initial_clues = cells_filled
        puzzle_difficulty = 1.0 - (initial_clues / total_cells)  # Higher = harder (fewer clues)
        
        # Distance to solution at start
        cells_different = jnp.sum(board != solution)
        initial_distance_to_solution = cells_different / total_cells
        
        metrics = {
            # Existing metrics
            "cells_filled": jnp.array(completion_ratio, dtype=float),
            "rows_complete": jnp.array(self._count_complete_rows(board) / 9.0, dtype=float),
            "cols_complete": jnp.array(self._count_complete_cols(board) / 9.0, dtype=float),
            "squares_complete": jnp.array(self._count_complete_squares(board) / 9.0, dtype=float),
            "success_easy": jnp.array(completion_ratio >= 0.5, dtype=float),
            "success": jnp.array(0.0, dtype=float),  # No success at start
            
            # Initial puzzle characteristics
            "initial_clues": jnp.array(initial_clues / total_cells, dtype=float),  # Normalize to [0,1]
            "puzzle_difficulty": jnp.array(puzzle_difficulty, dtype=float),
            "initial_distance_to_solution": jnp.array(initial_distance_to_solution, dtype=float),
            
            # Initialize step metrics to 0
            "valid_move": jnp.array(0.0, dtype=float),
            "correct_move": jnp.array(0.0, dtype=float),
            "incorrect_move": jnp.array(0.0, dtype=float),
            "cells_remaining": jnp.array(1.0 - completion_ratio, dtype=float),
            "completion_percentage": jnp.array(completion_ratio * 100.0, dtype=float),
            "distance_to_solution": jnp.array(initial_distance_to_solution, dtype=float),
            "moves_made": jnp.array(0.0, dtype=float),
            "quarter_complete": jnp.array(completion_ratio >= 0.25, dtype=float),
            "half_complete": jnp.array(completion_ratio >= 0.5, dtype=float),
            "three_quarters_complete": jnp.array(completion_ratio >= 0.75, dtype=float),
            "almost_complete": jnp.array(completion_ratio >= 0.9, dtype=float),
        }
        
        # Initialize reward and done for reset
        reward = jnp.array(0.0, dtype=float)
        done = jnp.array(0.0, dtype=float)
        
        state = State(pipeline_state, obs, reward, done, metrics, info=info)
        return state
    
    def step(self, state: State, action: jax.Array) -> State:
        """Take a step in the environment."""
        # Decode action: action is an integer representing cell and number
        cell_idx = action // 9  # Which cell (0-80)
        number = (action % 9) + 1  # Which number (1-9)
        
        # Convert cell index to row, col (ensure they are integers)
        row = jnp.array(cell_idx // 9, dtype=jnp.int32)
        col = jnp.array(cell_idx % 9, dtype=jnp.int32)
        
        # Get current board from pipeline_state.q (following ant.py pattern)
        board_flat = state.pipeline_state.q  # Shape: (81,) or (batch_size, 81)
        board = board_flat.reshape(-1, 9, 9)  # Shape: (9, 9) or (batch_size, 9, 9)
        
        # Check if the move is valid (with debugging)
        is_valid = self._is_valid_move(board, row, col, number)
        
        # Check if this is the correct move according to the solution
        solution = state.info["solution"]
        is_correct = (solution[row, col] == number)
        
        # Apply the move if valid (handle batched inputs)
        # Convert 2D indices to 1D indices for batched indexing
        batch_indices = jnp.arange(board.shape[0]) if len(board.shape) == 3 else jnp.array([0])
        flat_indices = row * 9 + col
        
        # Reshape board to 2D for easier indexing
        board_flat = board.reshape(-1, 81)
        
        # Update the board using scatter
        board_flat = board_flat.at[batch_indices, flat_indices].set(
            jnp.where(is_valid, number, board_flat[batch_indices, flat_indices])
        )
        
        # Reshape back to original shape
        board = board_flat.reshape(board.shape)
        
        # Flatten the board back
        new_board_flat = board.flatten()
        
        # Get the solution from the current state (stored in qd)
        solution_flat = state.pipeline_state.qd  # Shape: (81,) or (batch_size, 81)
        
        # Update pipeline_state with new board and keep solution (following ant.py pattern)
        new_q = new_board_flat
        new_qd = solution_flat  # Keep the solution in qd
        pipeline_state = state.pipeline_state.replace(q=new_q, qd=new_qd)
        
        # Get new observation using _get_obs method (following ant.py pattern)
        obs = self._get_obs(pipeline_state)
        
        # Calculate reward and success
        cells_filled = jnp.sum(board != 0)
        total_cells = 81
        completion_ratio = cells_filled / total_cells
        
        # Check if puzzle is complete and correct
        is_complete = cells_filled == total_cells
        is_correct_solution = self._is_solution_correct(board)
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
        valid_move = jnp.mean(jnp.array(is_valid, dtype=float)) if is_valid.ndim > 0 else jnp.array(is_valid, dtype=float)
        correct_move = jnp.mean(jnp.array(is_correct, dtype=float)) if is_correct.ndim > 0 else jnp.array(is_correct, dtype=float)
        incorrect_move = jnp.mean(jnp.array(jnp.logical_and(is_valid, jnp.logical_not(is_correct)), dtype=float)) if is_valid.ndim > 0 else jnp.array(jnp.logical_and(is_valid, jnp.logical_not(is_correct)), dtype=float)
        
        # Progress metrics
        cells_remaining = total_cells - cells_filled
        completion_percentage = completion_ratio * 100.0
        
        # Distance to solution (how many cells differ from solution)
        cells_different = jnp.sum(board != solution)
        distance_to_solution = cells_different / total_cells
        
        # Efficiency metrics
        # These will be accumulated over the episode
        moves_made = jnp.array(1.0, dtype=float)  # This step counts as 1 move
        
        # Puzzle-specific metrics
        initial_cells = jnp.sum(state.info["solution"] != 0)  # Should be 81, but let's be safe
        cells_to_fill = total_cells - initial_cells
        
        # Milestone metrics
        quarter_complete = jnp.array(completion_ratio >= 0.25, dtype=float)
        half_complete = jnp.array(completion_ratio >= 0.5, dtype=float)
        three_quarters_complete = jnp.array(completion_ratio >= 0.75, dtype=float)
        almost_complete = jnp.array(completion_ratio >= 0.9, dtype=float)
        
        # Update metrics (following ant.py pattern)
        state.metrics.update(
            # Existing metrics
            cells_filled=jnp.array(completion_ratio, dtype=float),
            rows_complete=jnp.array(self._count_complete_rows(board) / 9.0, dtype=float),
            cols_complete=jnp.array(self._count_complete_cols(board) / 9.0, dtype=float),
            squares_complete=jnp.array(self._count_complete_squares(board) / 9.0, dtype=float),
            success=success,
            success_easy=success_easy,
            
            # Move quality metrics
            valid_move=valid_move,
            correct_move=correct_move,
            incorrect_move=incorrect_move,
            
            # Progress metrics
            cells_remaining=cells_remaining / total_cells,  # Normalize to [0,1]
            completion_percentage=completion_percentage,
            distance_to_solution=distance_to_solution,
            
            # Efficiency metrics
            moves_made=moves_made,
            
            # Milestone metrics
            quarter_complete=quarter_complete,
            half_complete=half_complete,
            three_quarters_complete=three_quarters_complete,
            almost_complete=almost_complete,
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
        # Remove the batch dimension from board if it exists
        if len(board.shape) == 3 and board.shape[0] == 1:
            board = board[0]  # Shape: (9, 9)
        
        # Check if cell is already filled
        if board[row, col] != 0:
            return False
        
        # Check row
        row_values = board[row, :]
        if jnp.any(row_values == number):
            return False
        
        # Check column
        col_values = board[:, col]
        if jnp.any(col_values == number):
            return False
        
        # Check 3x3 sub-square
        sub_row_start = 3 * (row // 3)
        sub_col_start = 3 * (col // 3)
        sub_square = board[sub_row_start:sub_row_start+3, sub_col_start:sub_col_start+3]
        if jnp.any(sub_square == number):
            return False
        
        return True
    
    def _is_valid_move_vectorized(self, board, row, col, number):
        """Check if moves are valid for the given board (vectorized version)."""
        # Understanding: This is called with a single board and vectorized row/col/number
        # The vmap is over the action space (729 possible actions)
        
        # Remove the batch dimension from board if it exists
        if len(board.shape) == 3 and board.shape[0] == 1:
            board = board[0]  # Shape: (9, 9)
        
        # Now board is (9, 9) and row/col/number are vectors of length 729
        # We need to handle this by broadcasting
        
        # Step 1: Check if cell is already filled
        # Convert 2D indices to 1D for easier indexing
        flat_indices = row * 9 + col
        board_flat = board.flatten()  # Shape: (81,)
        cell_values = board_flat[flat_indices]  # Shape: (729,)
        cell_filled = cell_values != 0  # Shape: (729,)
        
        # Step 2: Check row
        # For each action, check if the number exists in the corresponding row
        row_start_indices = row * 9  # Shape: (729,)
        row_indices = row_start_indices[:, None] + jnp.arange(9)  # Shape: (729, 9)
        row_values = board_flat[row_indices]  # Shape: (729, 9)
        row_has_number = jnp.any(row_values == number[:, None], axis=1)  # Shape: (729,)
        
        # Step 3: Check column
        # For each action, check if the number exists in the corresponding column
        col_indices = col[:, None] + jnp.arange(9) * 9  # Shape: (729, 9)
        col_values = board_flat[col_indices]  # Shape: (729, 9)
        col_has_number = jnp.any(col_values == number[:, None], axis=1)  # Shape: (729,)
        
        # Step 4: Check 3x3 sub-square
        # For each action, check if the number exists in the corresponding 3x3 square
        sub_row_start = 3 * (row // 3)  # Shape: (729,)
        sub_col_start = 3 * (col // 3)  # Shape: (729,)
        
        # Generate indices for the 3x3 sub-square for each action
        square_indices = []
        for i in range(3):
            for j in range(3):
                square_indices.append((sub_row_start + i) * 9 + (sub_col_start + j))
        
        square_indices = jnp.array(square_indices)  # Shape: (9, 729)
        square_values = board_flat[square_indices.T]  # Shape: (729, 9)
        square_has_number = jnp.any(square_values == number[:, None], axis=1)  # Shape: (729,)
        
        # Step 5: Combine results
        is_valid = jnp.logical_not(jnp.logical_or(jnp.logical_or(cell_filled, row_has_number), jnp.logical_or(col_has_number, square_has_number)))
        
        return is_valid
    

    
    def _is_solution_correct(self, board):
        """Check if the board is a valid complete Sudoku solution."""
        # Understanding: This is called with a single board
        # The vmap is over the action space, not over multiple environments
        
        # Remove the batch dimension from board if it exists
        if len(board.shape) == 3 and board.shape[0] == 1:
            board = board[0]  # Shape: (9, 9)
        
        # Now board is (9, 9) - we need to check if it's a valid Sudoku solution
        
        # Step 1: Check all rows
        # Sort each row and check if it equals [1,2,3,4,5,6,7,8,9]
        expected = jnp.arange(1, 10)  # [1,2,3,4,5,6,7,8,9]
        row_sorted = jnp.sort(board, axis=1)  # Sort each row
        rows_valid = jnp.all(row_sorted == expected, axis=1)  # Check each row
        all_rows_valid = jnp.all(rows_valid)  # All rows must be valid
        
        # Step 2: Check all columns
        # Sort each column and check if it equals [1,2,3,4,5,6,7,8,9]
        cols_valid = jnp.array(True, dtype=jnp.bool_)
        for col in range(9):
            col_values = board[:, col]
            col_sorted = jnp.sort(col_values)
            col_valid = jnp.array_equal(col_sorted, expected)
            cols_valid = jnp.logical_and(cols_valid, col_valid)
        all_cols_valid = cols_valid
        
        # Step 3: Check all 3x3 sub-squares
        # Check each 3x3 sub-square
        squares_valid = jnp.array(True, dtype=jnp.bool_)
        for i in range(3):
            for j in range(3):
                # Extract 3x3 sub-square
                sub_square = board[3*i:3*i+3, 3*j:3*j+3]
                sub_square_flat = sub_square.flatten()
                sub_square_sorted = jnp.sort(sub_square_flat)
                square_valid = jnp.array_equal(sub_square_sorted, expected)
                squares_valid = jnp.logical_and(squares_valid, square_valid)
        
        # Step 4: Combine all checks
        is_valid = jnp.logical_and(jnp.logical_and(all_rows_valid, all_cols_valid), squares_valid)
        
        return is_valid
    

    
    def _calculate_goal_indicators(self, board):
        """Calculate completion indicators for rows, columns, and 3x3 squares."""
        # Handle batched input by removing batch dimension if present
        if len(board.shape) == 3:
            board = board[0]  # Take first batch element
        
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
    
    def _count_complete_rows(self, board):
        """Count the number of complete rows."""
        # Check each row
        row_complete = jnp.zeros(9, dtype=jnp.bool_)
        for row in range(9):
            row_values = board[row, :]
            is_filled = jnp.all(row_values != 0)
            is_valid = jnp.array_equal(jnp.sort(row_values), jnp.arange(1, 10))
            row_complete = row_complete.at[row].set(
                jnp.logical_and(is_filled, is_valid)
            )
        return jnp.sum(row_complete)
    
    def _count_complete_cols(self, board):
        """Count the number of complete columns."""
        # Check each column
        col_complete = jnp.zeros(9, dtype=jnp.bool_)
        for col in range(9):
            col_values = board[:, col]
            is_filled = jnp.all(col_values != 0)
            is_valid = jnp.array_equal(jnp.sort(col_values), jnp.arange(1, 10))
            col_complete = col_complete.at[col].set(
                jnp.logical_and(is_filled, is_valid)
            )
        return jnp.sum(col_complete)
    
    def _count_complete_squares(self, board):
        """Count the number of complete 3x3 squares."""
        # Check each 3x3 square
        square_complete = jnp.zeros(9, dtype=jnp.bool_)
        square_idx = 0
        for i in range(3):
            for j in range(3):
                sub_square = board[3*i:3*i+3, 3*j:3*j+3].flatten()
                is_filled = jnp.all(sub_square != 0)
                is_valid = jnp.array_equal(jnp.sort(sub_square), jnp.arange(1, 10))
                square_complete = square_complete.at[square_idx].set(
                    jnp.logical_and(is_filled, is_valid)
                )
                square_idx += 1
        return jnp.sum(square_complete)
    
    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Get observation from pipeline state (following ant.py pattern)."""
        # Extract board from pipeline_state.q (81 dimensions)
        board_flat = pipeline_state.q  # Shape: (81,) or (batch_size, 81)
        
        # Extract solution from pipeline_state.qd (81 dimensions)
        solution_flat = pipeline_state.qd  # Shape: (81,) or (batch_size, 81)
        
        # Combine state and goal: state (81) + goal (81) = 162
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
