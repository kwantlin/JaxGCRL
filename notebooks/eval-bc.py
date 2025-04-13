import sys
sys.path.append('../')

import jax
from jax import numpy as jp
import matplotlib.pyplot as plt
from IPython.display import HTML
from brax.io import model, html
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from src import networks
from utils import get_env_config, create_env
import pickle
import numpy as np

import flax.linen as nn
from brax.training import gradients, distribution, types, pmap
import functools
from jax import numpy as jnp
import seaborn as sns
import pandas as pd


RUN_FOLDER_PATH = '/home/kwantlin/JaxGCRL/runs/run_ant-main_s_1'
CKPT_NAME = '/step_11427840.pkl'

params = model.load_params(RUN_FOLDER_PATH + '/ckpt' + CKPT_NAME)
policy_params, encoders_params, context_params = params
sa_encoder_params, g_encoder_params = encoders_params['sa_encoder'], encoders_params['g_encoder']

args_path = RUN_FOLDER_PATH + '/args.pkl'

with open(args_path, "rb") as f:
    args = pickle.load(f)

config = get_env_config(args)

env = create_env(env_name=args.env_name, backend=args.backend)
obs_size = env.observation_size
action_size = env.action_size
goal_size = env.observation_size - env.state_dim

BC_RUN_FOLDER_PATH = '/home/kwantlin/JaxGCRL/runs/run_ant-bc_s_1'
BC_CKPT_NAME = '/step_11427840.pkl'

bc_params = model.load_params(BC_RUN_FOLDER_PATH + '/ckpt' + BC_CKPT_NAME)
bc_policy_params, bc_context_params = bc_params

bc_args_path = BC_RUN_FOLDER_PATH + '/args.pkl'

with open(bc_args_path, "rb") as f:
    bc_args = pickle.load(f)

bc_config = get_env_config(bc_args)


class Net(nn.Module):
    """
    MLP with residual connections: residual blocks have $block_size layers. Uses swish activation, optionally uses layernorm.
    """
    output_size: int
    width: int = 1024
    num_blocks: int = 4
    block_size: int = 2
    use_ln: bool = True
    @nn.compact
    def __call__(self, x):
        lecun_uniform = nn.initializers.variance_scaling(1/3, "fan_in", "uniform")
        normalize = nn.LayerNorm() if self.use_ln else (lambda x: x)
        
        # Start of net
        residual_stream = jnp.zeros((x.shape[0], self.width))
        
        # Main body
        for i in range(self.num_blocks):
            for j in range(self.block_size):
                x = nn.swish(normalize(nn.Dense(self.width, kernel_init=lecun_uniform)(x)))
            x += residual_stream
            residual_stream = x
                
        # Last layer mapping to representation dimension
        x = nn.Dense(self.output_size, kernel_init=lecun_uniform)(x)
        return x


def make_policy(actor, parametric_action_distribution, params, deterministic=False):
    def policy(obs, key_sample):
        obs = jnp.expand_dims(obs, 0)
        logits = actor.apply(params, obs)
        if deterministic:
            action = parametric_action_distribution.mode(logits)
        else:
            action = parametric_action_distribution.sample(logits, key_sample)
            action = action[0]
        extras = {}
        return action, extras
    return policy


# Network functions
block_size = 2 # Maybe make this a hyperparameter
num_blocks = max(1, args.n_hidden // block_size)
actor = Net(action_size * 2, args.h_dim, num_blocks, block_size, args.use_ln)
bc_actor = Net(action_size * 2, args.h_dim, num_blocks, block_size, args.use_ln)
sa_net = Net(args.repr_dim, args.h_dim, num_blocks, block_size, args.use_ln)
g_net = Net(args.repr_dim, args.h_dim, num_blocks, block_size, args.use_ln)
context_net = Net(goal_size * 2, args.h_dim, num_blocks, block_size, args.use_ln)
parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size) # Would like to replace this but it's annoying to.

inference_fn = make_policy(actor, parametric_action_distribution, policy_params)
bc_inference_fn = make_policy(bc_actor, parametric_action_distribution, bc_policy_params)

# inference_fn = inference_fn(params[:1])

# crl_networks = networks.make_crl_networks(config, env, obs_size, action_size)

# inference_fn = networks.make_inference_fn(crl_networks)
# inference_fn = inference_fn(params[:2])

sa_encoder = lambda obs: sa_net.apply(sa_encoder_params, obs)
g_encoder = lambda obs: g_net.apply(g_encoder_params, obs)
context_encoder = lambda traj: context_net.apply(context_params, traj)


NUM_ENVS = 100

jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)
jit_inference_fn = jax.jit(inference_fn)
bc_jit_inference_fn = jax.jit(bc_inference_fn)

def collect_trajectory(rng):
    def step_fn(carry, _):
        state, rng = carry
        act_rng, next_rng = jax.random.split(rng)
        act, _ = jit_inference_fn(state.obs, act_rng)
        next_state = jit_env_step(state, act)
        return (next_state, next_rng), (state, act, state.reward)
    
    init_state = jit_env_reset(rng=rng)
    (final_state, _), (states, actions, rewards) = jax.lax.scan(
        step_fn, 
        (init_state, rng), 
        None, 
        length=1024
    )
    return states.obs, actions, rewards

episode_rngs = jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)
observations, actions, rewards = jax.vmap(collect_trajectory)(episode_rngs)
states = observations[:, :, :env.state_dim]
goals = observations[:, 0, env.state_dim:]
print(states.shape, actions.shape, goals.shape)
last_states = observations[:, -1, env.goal_indices]
print(last_states.shape)
# Calculate total reward per rollout
total_rewards = jnp.sum(rewards, axis=1)  # Sum rewards along trajectory dimension
print("Total rewards per rollout (mean and std):", jnp.mean(total_rewards), jnp.std(total_rewards))

sa_pairs = jnp.reshape(jnp.concatenate((states, actions), axis=-1), (NUM_ENVS, -1))
print(sa_pairs.shape)

context_output = context_encoder(sa_pairs)
context_mean, context_log_std = jnp.split(context_output, 2, axis=-1)
# print(context_mean, context_log_std)

# Sample NUM_SAMPLES times from each episode's context distribution
NUM_SAMPLES = 100
sample_rng = jax.random.PRNGKey(0)
sample_rngs = jax.random.split(sample_rng, NUM_ENVS)

def sample_from_gaussian(rng, mean, log_std):
    noise = jax.random.normal(rng, shape=(NUM_SAMPLES, mean.shape[0]))
    std = jnp.exp(log_std)
    samples = mean + noise * std
    return samples

# Generate samples for each episode
inferred_goals = jax.vmap(sample_from_gaussian)(
    sample_rngs,
    context_mean,
    context_log_std
)

# print("Context samples shape:", inferred_goals.shape)  # (NUM_ENVS, num_samples, context_dim)
# print("goals", goals)
# print("context_samples", context_samples)

jit_env_reset_with_target = jax.jit(env.reset_with_target)

def collect_trajectory_with_target(rng, target, true_goal):
    def step_fn(carry, _):
        state, rng = carry
        act_rng, next_rng = jax.random.split(rng)
        act, _ = jit_inference_fn(state.obs, act_rng)
        next_state = jit_env_step(state, act)
        
        # Compute distance-based reward
        current_pos = next_state.obs[env.goal_indices]
        dist_to_goal = jnp.linalg.norm(current_pos - true_goal)
        reward = jnp.where(dist_to_goal < env.goal_reach_thresh, 1.0, 0.0)
        
        return (next_state, next_rng), reward
    
    init_state = jit_env_reset_with_target(rng=rng, target=target)
    (final_state, _), rewards = jax.lax.scan(
        step_fn, 
        (init_state, rng), 
        None, 
        length=1024
    )
    return rewards

# Collect trajectories using last states as targets
last_state_rngs = jax.random.split(jax.random.PRNGKey(1), NUM_ENVS)
last_state_rews = jax.vmap(collect_trajectory_with_target)(
    last_state_rngs,
    last_states,
    goals
)

def bc_collect_trajectory_with_target(rng, target, true_goal):
    def step_fn(carry, _):
        state, rng = carry
        act_rng, next_rng = jax.random.split(rng)
        act, _ = bc_jit_inference_fn(state.obs, act_rng)
        next_state = jit_env_step(state, act)
        
        # Compute distance-based reward
        current_pos = next_state.obs[env.goal_indices]
        dist_to_goal = jnp.linalg.norm(current_pos - true_goal)
        reward = jnp.where(dist_to_goal < env.goal_reach_thresh, 1.0, 0.0)
        
        return (next_state, next_rng), reward
    
    init_state = jit_env_reset_with_target(rng=rng, target=target)
    (final_state, _), rewards = jax.lax.scan(
        step_fn, 
        (init_state, rng), 
        None, 
        length=1024
    )
    return rewards

bc_last_state_rews = jax.vmap(bc_collect_trajectory_with_target)(
    last_state_rngs,
    last_states,
    goals
)

# Compute euclidean distances between goals and last states
goal_distances = jnp.linalg.norm(last_states - goals, axis=1)
# print("Goal distances with last states as goal:", goal_distances)

total_rewards_last_state = jnp.sum(last_state_rews, axis=1)  # Sum rewards along trajectory dimension
total_rewards_bc_last_state = jnp.sum(bc_last_state_rews, axis=1)  # Sum rewards along trajectory dimension
# print("Total rewards per rollout with last states as goal:", total_rewards_last_state)

# Collect trajectories using inferred goals as targets
# Split RNG for each env and each sample
inferred_goal_rngs = jax.random.split(jax.random.PRNGKey(1), NUM_ENVS * NUM_SAMPLES)
inferred_goal_rngs = inferred_goal_rngs.reshape(NUM_ENVS, NUM_SAMPLES, -1)

# For each env, collect trajectories for all inferred goals
inferred_goal_results = []
for env_idx in range(NUM_ENVS):
    env_rews = jax.vmap(collect_trajectory_with_target, in_axes=(0, 0, None))(
        inferred_goal_rngs[env_idx], 
        inferred_goals[env_idx],
        goals[env_idx]
    )
    inferred_goal_results.append((env_rews))

# Stack results back into arrays with same shape as before
inferred_goal_rews = jax.tree_map(
    lambda *x: jnp.stack(x), 
    *inferred_goal_results
)

# BC: For each env, collect trajectories for all inferred goals
bc_inferred_goal_results = []
for env_idx in range(NUM_ENVS):
    env_rews = jax.vmap(bc_collect_trajectory_with_target, in_axes=(0, 0, None))(
        inferred_goal_rngs[env_idx], 
        inferred_goals[env_idx],
        goals[env_idx]
    )
    bc_inferred_goal_results.append((env_rews))

# Stack results back into arrays with same shape as before
bc_inferred_goal_rews = jax.tree_map(
    lambda *x: jnp.stack(x), 
    *bc_inferred_goal_results
)

print("inferred_goal_rews shape:", inferred_goal_rews.shape)
total_rewards_inferred_goal_mean = jnp.mean(jnp.sum(inferred_goal_rews, axis=2), axis=1)
total_rewards_bc_inferred_goal_mean = jnp.mean(jnp.sum(bc_inferred_goal_rews, axis=2), axis=1)
# print("(Mean) Total rewards per rollout with inferred goals:", total_rewards_inferred_goal_mean)
total_rewards_inferred_goal_std = jnp.std(jnp.sum(inferred_goal_rews, axis=2), axis=1)
total_rewards_bc_inferred_goal_std = jnp.std(jnp.sum(bc_inferred_goal_rews, axis=2), axis=1)
# print("(Std) Total rewards per rollout with inferred goals:", total_rewards_inferred_goal_std)
# Compute euclidean distances between goals and last states
goal_distances = jnp.linalg.norm(inferred_goals - goals[:, None], axis=-1)
# print("Goal distances with inferred goals:", goal_distances)

# Compute differences and their statistics for total rewards vs last state rewards
reward_diff_last_state = total_rewards_last_state - total_rewards
reward_diff_bc_last_state = total_rewards_bc_last_state - total_rewards

reward_diff_last_state_mean = jnp.mean(reward_diff_last_state)
reward_diff_bc_last_state_mean = jnp.mean(reward_diff_bc_last_state)

reward_diff_last_state_std = jnp.std(reward_diff_last_state)
reward_diff_bc_last_state_std = jnp.std(reward_diff_bc_last_state)

print("Mean difference between total rewards and last state rewards:", reward_diff_last_state_mean)
print("Std of difference between total rewards and last state rewards:", reward_diff_last_state_std)

print("Mean difference between total rewards and last state rewards:", reward_diff_bc_last_state_mean)
print("Std of difference between total rewards and last state rewards:", reward_diff_bc_last_state_std)


# Compute differences and their statistics for total rewards vs inferred goal rewards
reward_diff_inferred = total_rewards_inferred_goal_mean - total_rewards
reward_diff_bc_inferred = total_rewards_bc_inferred_goal_mean - total_rewards

reward_diff_inferred_mean = jnp.mean(reward_diff_inferred)
reward_diff_bc_inferred_mean = jnp.mean(reward_diff_bc_inferred)

reward_diff_inferred_std = jnp.std(reward_diff_inferred)
reward_diff_bc_inferred_std = jnp.std(reward_diff_bc_inferred)
print("Mean difference between total rewards and inferred goal rewards:", reward_diff_inferred_mean)
print("Std of difference between total rewards and inferred goal rewards:", reward_diff_inferred_std)

print("Mean difference between total rewards and inferred goal rewards:", reward_diff_bc_inferred_mean)
print("Std of difference between total rewards and inferred goal rewards:", reward_diff_bc_inferred_std)

