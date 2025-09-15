import jax
import jax.numpy as jnp
from brax.envs.base import State
from brax import base


class NQueens4x4:
    """
    A discrete, deterministic 4x4 N-Queens environment.

    Board: 4x4 grid, values: 0 empty, 1 queen.
    Actions: 16 cells with 2 options each -> place queen (1) or erase (0): 16 * 2 = 32 actions.
      - action_type 0 = erase, 1 = place queen
    Goal: place 4 queens so that none attack each other.
    Observation: board (16) + goal-indicator vector (constraint satisfaction indicators)
    """

    def __init__(self, **kwargs):
        self.board_size = 4
        self.num_queens = 4
        self.action_space_size = 16 * 2
        self.action_size = self.action_space_size
        self.state_dim = 16
        # Constraint indicators length: rows(4)+cols(4)+main-diags(7)+anti-diags(7)+queens-count(1) = 23
        self.num_constraint_indicators = 23
        self.observation_size = 16 + self.num_constraint_indicators
        # Goal lives in the second segment of the observation (indices 16..38)
        self.goal_indices = jnp.arange(16, 16 + self.num_constraint_indicators)
        self.goal_reach_thresh = 0.5

    def reset(self, rng: jax.Array) -> State:
        # Start with empty board
        board = jnp.zeros((4, 4), dtype=jnp.int32)
        info = {"initial_board": board}

        q = board.flatten()
        # Goal is the constraint indicator vector for the current board
        qd = self._compute_constraint_indicators(board)
        pipeline_state = self.pipeline_init(q, qd)

        obs = self._get_obs(pipeline_state)

        metrics = {
            "valid_move": jnp.array(0.0, dtype=float),
            "invalid_move": jnp.array(0.0, dtype=float),
            "moves_made": jnp.array(0.0, dtype=float),
            "queens_placed": jnp.array(0.0, dtype=float),
            "success": jnp.array(0.0, dtype=float),
        }

        reward = jnp.array(0.0, dtype=float)
        done = jnp.array(0.0, dtype=float)
        return State(pipeline_state, obs, reward, done, metrics, info=info)

    def step(self, state: State, action: jax.Array) -> State:
        # Decode action
        cell_idx = action // 2
        action_type = action % 2  # 0 erase, 1 place queen
        row = jnp.array(cell_idx // 4, dtype=jnp.int32)
        col = jnp.array(cell_idx % 4, dtype=jnp.int32)

        board_flat = state.pipeline_state.q
        board = board_flat.reshape(4, 4)

        is_erase = (action_type == 0)
        target_value = jnp.where(is_erase, 0, 1)

        # Validate move
        can_place = jnp.logical_and(board[row, col] == 0, self._is_safe(board, row, col))
        can_erase = board[row, col] == 1
        is_valid = jnp.where(is_erase, can_erase, can_place)

        # Apply move if valid
        new_board = board.at[row, col].set(jnp.where(is_valid, target_value, board[row, col]))
        new_board_flat = new_board.flatten()

        # Update pipeline state (also update goal/indicators in qd)
        new_qd = self._compute_constraint_indicators(new_board)
        pipeline_state = state.pipeline_state.replace(q=new_board_flat, qd=new_qd)
        obs = self._get_obs(pipeline_state)

        # Rewards and termination
        queens_placed = jnp.sum(new_board)
        success = jnp.array(jnp.logical_and(queens_placed == self.num_queens, self._all_safe(new_board)), dtype=float)
        base_reward = jnp.where(is_valid, 1.0, -1.0)
        complete_bonus = jnp.where(success == 1.0, 100.0, 0.0)
        reward = base_reward + complete_bonus

        done = jnp.array(success == 1.0, dtype=float)

        valid_move = jnp.array(is_valid, dtype=float)
        invalid_move = 1.0 - valid_move

        state.metrics.update(
            valid_move=valid_move,
            invalid_move=invalid_move,
            moves_made=jnp.array(1.0, dtype=float),
            queens_placed=jnp.array(queens_placed, dtype=float),
            success=success,
        )

        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)

    def _is_safe(self, board: jax.Array, row: jax.Array, col: jax.Array) -> jax.Array:
        # No queen in same row or column
        row_clear = jnp.all(board[row, :] == 0)
        col_clear = jnp.all(board[:, col] == 0)

        # Diagonals
        def diag_clear(b, r, c, dr, dc):
            vals = []
            rr, cc = r + dr, c + dc
            for _ in range(4):
                vals.append(jnp.where((rr >= 0) & (rr < 4) & (cc >= 0) & (cc < 4), b[rr, cc], 0))
                rr, cc = rr + dr, cc + dc
            return jnp.all(jnp.array(vals) == 0)

        d1 = diag_clear(board, row, col, -1, -1)  # up-left
        d2 = diag_clear(board, row, col, -1, 1)   # up-right
        d3 = diag_clear(board, row, col, 1, -1)   # down-left
        d4 = diag_clear(board, row, col, 1, 1)    # down-right

        return jnp.logical_and(jnp.logical_and(row_clear, col_clear), jnp.logical_and(jnp.logical_and(d1, d2), jnp.logical_and(d3, d4)))

    def _all_safe(self, board: jax.Array) -> jax.Array:
        """Check that no two queens attack each other using fixed-size checks."""
        # Rows and columns must have at most one queen
        rows_ok = jnp.all(jnp.sum(board, axis=1) <= 1)
        cols_ok = jnp.all(jnp.sum(board, axis=0) <= 1)

        # Main diagonals (i-j = const): 7 diagonals for 4x4
        main_diag_sums = jnp.array([
            board[3, 0],
            board[2, 0] + board[3, 1],
            board[1, 0] + board[2, 1] + board[3, 2],
            board[0, 0] + board[1, 1] + board[2, 2] + board[3, 3],
            board[0, 1] + board[1, 2] + board[2, 3],
            board[0, 2] + board[1, 3],
            board[0, 3],
        ])
        mains_ok = jnp.all(main_diag_sums <= 1)

        # Anti-diagonals (i+j = const): 7 diagonals for 4x4
        anti_diag_sums = jnp.array([
            board[0, 0],
            board[0, 1] + board[1, 0],
            board[0, 2] + board[1, 1] + board[2, 0],
            board[0, 3] + board[1, 2] + board[2, 1] + board[3, 0],
            board[1, 3] + board[2, 2] + board[3, 1],
            board[2, 3] + board[3, 2],
            board[3, 3],
        ])
        antis_ok = jnp.all(anti_diag_sums <= 1)

        return jnp.logical_and(jnp.logical_and(rows_ok, cols_ok), jnp.logical_and(mains_ok, antis_ok))

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        board_flat = pipeline_state.q
        goal_flat = pipeline_state.qd if hasattr(pipeline_state, 'qd') else jnp.zeros((self.num_constraint_indicators,), dtype=board_flat.dtype)
        return jnp.concatenate([board_flat, goal_flat])

    def pipeline_init(self, q: jax.Array, qd: jax.Array = None) -> base.State:
        if qd is None:
            qd = jnp.zeros((self.num_constraint_indicators,), dtype=q.dtype)
        return base.State(
            q=q,
            qd=qd,
            x=base.Transform.zero((1,)),
            xd=base.Motion.zero((1,)),
            contact=None,
        )

    def pipeline_step(self, pipeline_state: base.State, action: jax.Array) -> base.State:
        return pipeline_state

    def _compute_constraint_indicators(self, board: jax.Array) -> jax.Array:
        """Returns a vector of constraint satisfaction indicators:
        [rows_ok(4), cols_ok(4), main_diags_ok(7), anti_diags_ok(7), queens_count_ok(1)].
        Each entry is 1.0 if the specific constraint is satisfied, else 0.0.
        """
        # Ensure board is (4,4)
        if len(board.shape) == 1:
            board = board.reshape(4, 4)

        # Rows: at most one queen
        row_sums = jnp.sum(board, axis=1)
        rows_ok = (row_sums <= 1).astype(jnp.float32)

        # Cols: at most one queen
        col_sums = jnp.sum(board, axis=0)
        cols_ok = (col_sums <= 1).astype(jnp.float32)

        # Main diagonals i-j const (7 diags)
        main_diag_sums = jnp.array([
            board[3, 0],
            board[2, 0] + board[3, 1],
            board[1, 0] + board[2, 1] + board[3, 2],
            board[0, 0] + board[1, 1] + board[2, 2] + board[3, 3],
            board[0, 1] + board[1, 2] + board[2, 3],
            board[0, 2] + board[1, 3],
            board[0, 3],
        ])
        mains_ok = (main_diag_sums <= 1).astype(jnp.float32)

        # Anti-diagonals i+j const (7 diags)
        anti_diag_sums = jnp.array([
            board[0, 0],
            board[0, 1] + board[1, 0],
            board[0, 2] + board[1, 1] + board[2, 0],
            board[0, 3] + board[1, 2] + board[2, 1] + board[3, 0],
            board[1, 3] + board[2, 2] + board[3, 1],
            board[2, 3] + board[3, 2],
            board[3, 3],
        ])
        antis_ok = (anti_diag_sums <= 1).astype(jnp.float32)

        # Queen count must equal 4
        queens_count_ok = jnp.array([1.0], dtype=jnp.float32) * ((jnp.sum(board) == self.num_queens).astype(jnp.float32))

        return jnp.concatenate([rows_ok, cols_ok, mains_ok, antis_ok, queens_count_ok])


