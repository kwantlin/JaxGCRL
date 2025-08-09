import sys
sys.path.append('../')
import os
import jax
import jax.numpy as jnp
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
import flax.linen as nn
from brax.training import distribution
from brax.training.agents.ppo import train as ppo_train
import pickle
from functools import partial
from utils import create_env

# Add project root to the Python path to allow importing from 'envs'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.ant_base import AntJump

# --- Constants ---
MODEL_PATH = 'ppo_antjump_h1.0_model.pkl'
NUM_EVAL_EPISODES = 100
EPISODE_LENGTH = 1000
TARGET_JUMP_HEIGHT = 1.0  # Inferred from the model filename

def main():
    """Loads a pre-trained policy, evaluates it on the antjump environment, and reports the results."""

    # 1. Load environment
    env = AntJump(target_jump_height=TARGET_JUMP_HEIGHT)

    # 2. Define PPO network architecture.
    # This must match the architecture used during training.
    ppo_network = ppo_networks.make_ppo_networks(
        env.observation_size,
        env.action_size,
    )

    # 3. Load the saved model parameters
    try:
        params = model.load_params(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please ensure the model file exists and the path is correct.")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return
    network_factory = ppo_networks.make_ppo_networks
    
    # 4. Get the make_policy function from the PPO train function
    make_policy, _, _ = ppo_train.train(
        environment=env,
        num_timesteps=0,  # No training steps needed
        episode_length=EPISODE_LENGTH,
        action_repeat=1,
        num_envs=1,
        num_evals=1,
        learning_rate=0,
        entropy_cost=0,
        discounting=0,
        seed=0,
        normalize_observations=True,
        network_factory=network_factory,
    )

    policy = make_policy(params, deterministic=True)
    
    # 5. JIT functions for performance
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_policy = jax.jit(policy)

    # 6. Define a function to collect a single trajectory
    def collect_trajectory(rng):
        """Rolls out a trajectory for one episode and returns the total reward."""
        rng, reset_rng = jax.random.split(rng)
        state = jit_env_reset(rng=reset_rng)

        def scan_step(carry, _):
            state, rng = carry
            policy_rng, next_rng = jax.random.split(rng)
            # The deterministic policy ignores the key, but we pass it for API consistency
            action_tuple = jit_policy(state.obs, policy_rng)
            action = action_tuple[0]  # Extract the action array from the tuple
            next_state = jit_env_step(state, action)
            return (next_state, next_rng), (state.obs, action, next_state.reward)

        init_carry = (state, rng)
        _, (observations, actions, rewards) = jax.lax.scan(
            scan_step, init_carry, None, length=EPISODE_LENGTH
        )
        return observations, actions, rewards

    # 7. Vmap to run multiple evaluations in parallel
    print(f"Running {NUM_EVAL_EPISODES} evaluation episodes...")
    eval_rngs = jax.random.split(jax.random.PRNGKey(0), NUM_EVAL_EPISODES)
    observations, actions, rewards = jax.vmap(collect_trajectory)(eval_rngs)
    total_rewards = jnp.sum(rewards, axis=1)
    total_rewards.block_until_ready()  # Ensure computation is finished before reporting

    # 8. Process trajectories
    states = observations[:, :, :env.observation_size]
    print("States shape:", states.shape)
    print("Actions shape:", actions.shape)
    sa_pairs = jnp.reshape(jnp.concatenate((states, actions), axis=-1), (NUM_EVAL_EPISODES, -1))
    print("SA pairs shape:", sa_pairs.shape)
    

    # --- Start of ant_fullobs evaluation ---
    print("\n--- Evaluating ant_fullobs policy with inferred goals ---")

    # 1. Load ant_fullobs environment and model
    fullobs_env_name = 'ant_fullobs'
    # This path is assumed based on the structure in eval-ant.py
    fullobs_run_folder_path = f'/home/kw2960/JaxGCRL/runs/run_{fullobs_env_name}-main-standard-della-maxent-gaussianmlp_s_1'
    
    try:
        fullobs_ckpt_path = fullobs_run_folder_path + '/ckpt/best.pkl'
        fullobs_params = model.load_params(fullobs_ckpt_path)
        fullobs_policy_params, _, fullobs_context_params = fullobs_params

        args_path = fullobs_run_folder_path + '/args.pkl'
        with open(args_path, "rb") as f:
            fullobs_args = pickle.load(f)
        
        fullobs_env_loaded = True
    except (FileNotFoundError, IsADirectoryError):
        print("ant_fullobs checkpoint or args not found, skipping evaluation.")
        fullobs_env_loaded = False

    if fullobs_env_loaded:
        fullobs_env = create_env(env_name=fullobs_env_name, backend='mjx')
        
        # 2. Define networks (structure from eval-ant.py)
        class Net(nn.Module):
            output_size: int
            width: int = 1024
            num_blocks: int = 4
            block_size: int = 2
            use_ln: bool = True
            @nn.compact
            def __call__(self, x):
                lecun_uniform = nn.initializers.variance_scaling(1/3, "fan_in", "uniform")
                normalize = nn.LayerNorm() if self.use_ln else (lambda x: x)
                residual_stream = jnp.zeros((x.shape[0], self.width))
                for _ in range(self.num_blocks):
                    for _ in range(self.block_size):
                        x = nn.swish(normalize(nn.Dense(self.width, kernel_init=lecun_uniform)(x)))
                    x += residual_stream
                    residual_stream = x
                x = nn.Dense(self.output_size, kernel_init=lecun_uniform)(x)
                return x

        block_size = 2
        num_blocks = max(1, fullobs_args.n_hidden // block_size)
        fullobs_action_size = fullobs_env.action_size
        fullobs_goal_size = fullobs_env.observation_size - fullobs_env.state_dim

        fullobs_actor = Net(fullobs_action_size * 2, fullobs_args.h_dim, num_blocks, block_size, fullobs_args.use_ln)
        fullobs_context_net = Net(fullobs_goal_size * 2, fullobs_args.h_dim, num_blocks, block_size, fullobs_args.use_ln)
        fullobs_parametric_action_distribution = distribution.NormalTanhDistribution(event_size=fullobs_action_size)

        def make_crl_policy(actor, parametric_action_distribution, params, deterministic=False):
            def policy(obs, key_sample):
                obs = jnp.expand_dims(obs, 0)
                logits = actor.apply(params, obs)
                if deterministic:
                    action = parametric_action_distribution.mode(logits)
                else:
                    action = parametric_action_distribution.sample(logits, key_sample)[0]
                return action, {}
            return policy

        fullobs_inference_fn = make_crl_policy(fullobs_actor, fullobs_parametric_action_distribution, fullobs_policy_params, deterministic=True)
        fullobs_context_encoder = lambda traj: fullobs_context_net.apply(fullobs_context_params, traj)

        # 3. Infer goals from antjump trajectories
        context_output = fullobs_context_encoder(sa_pairs)
        context_mean, context_log_std = jnp.split(context_output, 2, axis=-1)

        def sample_from_gaussian(rng, mean, log_std):
            noise = jax.random.normal(rng, shape=mean.shape)
            return mean + noise * jnp.exp(log_std)

        sample_rngs = jax.random.split(jax.random.PRNGKey(1), NUM_EVAL_EPISODES)
        inferred_goals = jax.vmap(sample_from_gaussian)(sample_rngs, context_mean, context_log_std)
        print("Inferred goals shape:", inferred_goals.shape)

        # 4. Rollout ant_fullobs with inferred goals and evaluate with antjump reward
        jit_fullobs_env_reset = jax.jit(fullobs_env.reset)
        jit_fullobs_env_step = jax.jit(fullobs_env.step)
        jit_fullobs_inference_fn = jax.jit(fullobs_inference_fn)

        def collect_fullobs_trajectory(rng, goal):
            state = jit_fullobs_env_reset(rng=rng)

            def scan_step(carry, _):
                state, rng = carry
                act_rng, next_rng = jax.random.split(rng)
                obs = jnp.concatenate((state.obs[:fullobs_env.state_dim], goal))
                act, _ = jit_fullobs_inference_fn(obs, act_rng)
                next_state = jit_fullobs_env_step(state, act)
                
                # Use antjump reward function
                z = next_state.pipeline_state.x.pos[0, 2]
                reward = jnp.where(z > TARGET_JUMP_HEIGHT, 1.0, 0.0)
                
                return (next_state, next_rng), reward
            
            _, rewards = jax.lax.scan(scan_step, (state, rng), None, length=EPISODE_LENGTH)
            return jnp.sum(rewards)

        print("Running ant_fullobs rollouts with inferred goals...")
        rollout_rngs = jax.random.split(jax.random.PRNGKey(2), NUM_EVAL_EPISODES)
        fullobs_rewards = jax.vmap(collect_fullobs_trajectory)(rollout_rngs, inferred_goals)
        fullobs_rewards.block_until_ready()

        # 5. Report results for the new evaluation
        fullobs_mean_reward = jnp.mean(fullobs_rewards)
        fullobs_std_reward = jnp.std(fullobs_rewards)
        fullobs_std_err = fullobs_std_reward / jnp.sqrt(NUM_EVAL_EPISODES)

        print("\n--- ant_fullobs with Inferred Goals ---")
        print(f"Mean total reward: {fullobs_mean_reward:.4f}")
        print(f"Standard deviation: {fullobs_std_reward:.4f}")
        print(f"Standard error of the mean: {fullobs_std_err:.4f}")

    # 10. Report results
    mean_reward = jnp.mean(total_rewards)
    std_reward = jnp.std(total_rewards)
    std_err = std_reward / jnp.sqrt(NUM_EVAL_EPISODES)

    print("\n--- Evaluation Results ---")
    print(f"Environment: antjump (target_height={TARGET_JUMP_HEIGHT})")
    print(f"Episodes: {NUM_EVAL_EPISODES}")
    print(f"Mean total reward: {mean_reward:.4f}")
    print(f"Standard deviation: {std_reward:.4f}")
    print(f"Standard error of the mean: {std_err:.4f}")

if __name__ == '__main__':
    main()

