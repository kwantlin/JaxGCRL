import os
import jax

from envs.nqueens_4x4 import NQueens4x4
from utils.env import create_nqueens_4x4_visualization


def generate_rollout(env: NQueens4x4, num_steps: int = 100):
    """
    Runs the 4x4 N-Queens environment with random actions and collects pipeline states and actions.

    Returns:
        Tuple[List[pipeline_state], List[int]]
    """
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)

    rollout = []
    actions_taken = []
    key = jax.random.PRNGKey(seed=42)
    key, subkey = jax.random.split(key)

    state = jit_env_reset(rng=subkey)
    rollout.append(state.pipeline_state)

    for t in range(num_steps):
        key, subkey = jax.random.split(key)
        action = jax.random.randint(subkey, shape=(), minval=0, maxval=env.action_space_size)
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
    env = NQueens4x4()
    rollout, actions = generate_rollout(env, num_steps=100)
    print(f"[DEBUG] passing actions to visualization. len(actions)={len(actions)} first5={actions[:5]}")
    html_string = create_nqueens_4x4_visualization(rollout, env, actions)

    output_path = os.path.join(os.path.dirname(__file__), "nqueens4x4_vis_test.html")
    with open(output_path, "w") as f:
        f.write(html_string)
    print(f"Saved 4x4 N-Queens visualization to: {output_path}")


if __name__ == "__main__":
    main()


