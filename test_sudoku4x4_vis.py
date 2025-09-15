import os
import jax
import jax.numpy as jnp

from envs.sudoku_4x4 import Sudoku4x4
from utils.env import create_sudoku_4x4_visualization


def generate_rollout(env: Sudoku4x4, num_steps: int = 100):
    """
    Runs the 4x4 Sudoku environment with random actions and collects pipeline states.

    Args:
        env: Initialized Sudoku4x4 environment
        num_steps: Number of steps to roll out

    Returns:
        Tuple[List[pipeline_state], List[int]]: rollout states and actions
    """
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)

    rollout = []
    actions_taken = []
    key = jax.random.PRNGKey(seed=0)
    key, subkey = jax.random.split(key)

    state = jit_env_reset(rng=subkey)
    rollout.append(state.pipeline_state)

    for t in range(num_steps):
        key, subkey = jax.random.split(key)
        # Action space size: 16 cells * 5 actions (0=erase, 1-4 numbers) = 80
        action = jax.random.randint(subkey, shape=(), minval=0, maxval=env.action_space_size)
        # Convert to Python int for JSON and debugging
        try:
            action_int = int(jax.device_get(action))
        except Exception:
            action_int = int(action)
        actions_taken.append(action_int)
        if t < 5:
            print(f"[DEBUG] step={t} action={action_int}")
        state = jit_env_step(state, action)
        rollout.append(state.pipeline_state)

    print(f"[DEBUG] total_rollout_states={len(rollout)} total_actions={len(actions_taken)}")
    return rollout, actions_taken


def main():
    # Initialize environment
    env = Sudoku4x4()

    # Generate a rollout with random actions
    rollout, actions = generate_rollout(env, num_steps=100)

    # Create HTML visualization string
    print(f"[DEBUG] passing actions to visualization. len(actions)={len(actions)} first5={actions[:5]}")
    html_string = create_sudoku_4x4_visualization(rollout, env, actions)

    # Save to file inside notebooks directory
    output_path = os.path.join(os.path.dirname(__file__), "sudoku4x4_vis_test.html")
    with open(output_path, "w") as f:
        f.write(html_string)

    print(f"Saved 4x4 Sudoku visualization to: {output_path}")


if __name__ == "__main__":
    main()


