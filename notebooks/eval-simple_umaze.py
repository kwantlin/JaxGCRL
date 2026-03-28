import sys
import os
sys.path.append('../')

import jax
from jax import numpy as jp
import matplotlib.pyplot as plt
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
from functools import partial

# note: ant: step_11427840
# note: reacher: step_20490752
# note: simple_u_maze: step_20490752
# note: pusher_easy: step_30823424

env_name = 'simple_u_maze'
# Load standard CRL checkpoint. For expert demos!
RUN_FOLDER_PATH = f'/home/kw2960/JaxGCRL/runs/run_{env_name}-main-standard-della-maxent-gaussianmlp_s_1'
CKPT_NAME = '/best.pkl'
params = model.load_params(RUN_FOLDER_PATH + '/ckpt' + CKPT_NAME)
policy_params, encoders_params, context_params = params

# CRL Mean field checkpoint
MEAN_FIELD_RUN_FOLDER_PATH = f'/scratch/gpfs/EYSENBACH/kw2960/JaxGCRL/runs/run_simple_u_maze-main-meanfield-numenvs256-numtimesteps40000000-batchsize1024-1-della-maxent-gaussianmlp-1e-4_s_3'
MEAN_FIELD_CKPT_NAME = '/best.pkl'
mean_field_params = model.load_params(MEAN_FIELD_RUN_FOLDER_PATH + '/ckpt' + MEAN_FIELD_CKPT_NAME)
_, _, mean_field_context_params = mean_field_params

# # CRL Mean field encoded checkpoint
# MEAN_FIELD_ENCODED_RUN_FOLDER_PATH = f'/home/kwantlin/JaxGCRL/runs/run_{env_name}-main-meanfield-encoded_s_1'
# MEAN_FIELD_ENCODED_CKPT_NAME = '/step_11427840.pkl'
# mean_field_encoded_params = model.load_params(MEAN_FIELD_ENCODED_RUN_FOLDER_PATH + '/ckpt' + MEAN_FIELD_ENCODED_CKPT_NAME)
# mean_field_encoded_policy_params, mean_field_encoded_encoder_params, mean_field_encoded_context_params = mean_field_encoded_params
# mean_field_encoded_sa_encoder_params, _ = mean_field_encoded_encoder_params['sa_encoder'], mean_field_encoded_encoder_params['g_encoder']

# GoalKDE + CRL
GOALKDE_RUN_FOLDER_PATH = f'/home/kw2960/JaxGCRL/runs/run_{env_name}-goalkde-standard-della-maxent-gaussianmlp_s_1'
GOALKDE_CKPT_NAME = '/best.pkl'
goalkde_params = model.load_params(GOALKDE_RUN_FOLDER_PATH + '/ckpt' + GOALKDE_CKPT_NAME)
goalkde_policy_params, goalkde_encoder_params, goalkde_context_params = goalkde_params

# GoalKDE + CRL Mean field
GOALKDE_MEAN_FIELD_RUN_FOLDER_PATH = f'/scratch/gpfs/EYSENBACH/kw2960/JaxGCRL/runs/run_simple_u_maze-goalkde-meanfield-1x-40000000-1024-256-1e-4-1e-4-1e-4-2-della-maxent-gaussianmlp-1e-4_s_2'
GOALKDE_MEAN_FIELD_CKPT_NAME = '/best.pkl'
goalkde_mean_field_params = model.load_params(GOALKDE_MEAN_FIELD_RUN_FOLDER_PATH + '/ckpt' + GOALKDE_MEAN_FIELD_CKPT_NAME)
_, _, goalkde_mean_field_context_params = goalkde_mean_field_params


print("Loaded all models")

# Common code
args_path = RUN_FOLDER_PATH + '/args.pkl'

with open(args_path, "rb") as f:
    args = pickle.load(f)

config = get_env_config(args)

env = create_env(env_name=args.env_name, backend=args.backend)
obs_size = env.observation_size
action_size = env.action_size
goal_size = env.observation_size - env.state_dim
NUM_STEPS = 1024


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
# sa_net = Net(args.repr_dim, args.h_dim, num_blocks, block_size, args.use_ln)
# g_net = Net(args.repr_dim, args.h_dim, num_blocks, block_size, args.use_ln)
context_net = Net(goal_size * 2, args.h_dim, num_blocks, block_size, args.use_ln)
backward_repr = Net(args.repr_dim, args.h_dim, num_blocks, block_size, args.use_ln)

parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size) # Would like to replace this but it's annoying to.

inference_fn = make_policy(actor, parametric_action_distribution, policy_params)

# sa_encoder = lambda obs: sa_net.apply(sa_encoder_params, obs)
# g_encoder = lambda obs: g_net.apply(g_encoder_params, obs)
context_encoder = lambda traj: context_net.apply(context_params, traj)
mean_field_context_encoder = lambda traj: context_net.apply(mean_field_context_params, traj)

# Mean field encoded context encoder, which uses the sa_net to encode the state-action pairs
# mean_field_encoded_sa_net = Net(args.repr_dim, args.h_dim, num_blocks, block_size, args.use_ln)
# mean_field_encoded_sa_encoder = lambda obs: mean_field_encoded_sa_net.apply(mean_field_encoded_sa_encoder_params, obs)
# mean_field_encoded_context_encoder = lambda traj: context_net.apply(mean_field_encoded_context_params, traj)

goalkde_inference_fn = make_policy(actor, parametric_action_distribution, goalkde_policy_params)
goalkde_context_encoder = lambda traj: context_net.apply(goalkde_context_params, traj)
goalkde_mean_field_context_encoder = lambda traj: context_net.apply(goalkde_mean_field_context_params, traj)


NUM_ENVS = 2000

jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)
jit_inference_fn = jax.jit(inference_fn)
jit_goalkde_inference_fn = jax.jit(goalkde_inference_fn)

def collect_trajectory(rng):
    def step_fn(carry, _):
        state, rng = carry
        act_rng, next_rng = jax.random.split(rng)
        act, _ = jit_inference_fn(state.obs, act_rng)
        next_state = jit_env_step(state, act)
        # Use 0/1 sparse success reward instead of env reward
        true_goal = state.obs[env.state_dim:]
        current_pos = next_state.obs[env.goal_indices]
        dist_to_goal = jnp.linalg.norm(current_pos - true_goal)
        reward = jnp.where(dist_to_goal < env.goal_reach_thresh, 1.0, 0.0)
        return (next_state, next_rng), (state, act, reward)
    
    init_state = jit_env_reset(rng=rng)
    (final_state, _), (states, actions, rewards) = jax.lax.scan(
        step_fn, 
        (init_state, rng), 
        None, 
        length=NUM_STEPS
    )
    return states.obs, actions, rewards

# Collect trajectories across NUM_ENVS
episode_rngs = jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)
observations, actions, rewards = jax.vmap(collect_trajectory)(episode_rngs)
print(observations.shape, actions.shape, rewards.shape)
states = observations[:, :, :env.state_dim]
goals = observations[:, 0, env.state_dim:]
print(states.shape, actions.shape, goals.shape)

# Calculate mean true goal
mean_true_goal = jnp.mean(goals, axis=0)
print("Mean true goal:", mean_true_goal)
print("Mean true goal shape:", mean_true_goal.shape)

# Also calculate standard deviation of goals
std_true_goal = jnp.std(goals, axis=0)
print("Standard deviation of true goals:", std_true_goal)

# Calculate the range of goals (min and max)
min_goals = jnp.min(goals, axis=0)
max_goals = jnp.max(goals, axis=0)
print("Min goals:", min_goals)
print("Max goals:", max_goals)

last_states = observations[:, -1, env.goal_indices]
print(last_states.shape)
# Calculate total reward per rollout
total_rewards = jnp.sum(rewards, axis=1)  # Sum rewards along trajectory dimension
print("Total rewards per rollout (mean and stderr):", jnp.mean(total_rewards), jnp.std(total_rewards) / jnp.sqrt(NUM_ENVS))


# Process with standard context encoder
sa_pairs = jnp.reshape(jnp.concatenate((states, actions), axis=-1), (NUM_ENVS, -1))
print("sa pairs shape", sa_pairs.shape)
context_output = context_encoder(sa_pairs)
context_mean, context_log_std = jnp.split(context_output, 2, axis=-1)
print("context mean shape", context_mean.shape)
print("context log std shape", context_log_std.shape)

# Process with mean field context encoder
sa_pairs_mf = jnp.reshape(jnp.concatenate((states, actions), axis=-1), (NUM_ENVS * NUM_STEPS, -1))
print("mean field sa pairs shape", sa_pairs_mf.shape)
mf_context_output = mean_field_context_encoder(sa_pairs_mf)
mf_context_mean, mf_context_log_std = jnp.split(mf_context_output, 2, axis=-1)
mf_context_mean = jnp.reshape(mf_context_mean, (NUM_ENVS, NUM_STEPS, -1))
mf_context_log_std = jnp.reshape(mf_context_log_std, (NUM_ENVS, NUM_STEPS, -1))
print("mean field context mean shape", mf_context_mean.shape)
print("mean field context log std shape", mf_context_log_std.shape)


# Process with mean field context encoder
# mf_encoded_encoder_output = mean_field_encoded_sa_encoder(sa_pairs_mf)
# print("mean field encoded encoder output shape", mf_encoded_encoder_output.shape)
# mf_encoded_context_output = mean_field_encoded_context_encoder(mf_encoded_encoder_output)
# print("mean field encoded context output shape", mf_encoded_context_output.shape)
# mf_encoded_context_mean, mf_encoded_context_log_std = jnp.split(mf_encoded_context_output, 2, axis=-1)
# mf_encoded_context_mean = jnp.reshape(mf_encoded_context_mean, (NUM_ENVS, NUM_STEPS, -1))
# mf_encoded_context_log_std = jnp.reshape(mf_encoded_context_log_std, (NUM_ENVS, NUM_STEPS, -1))
# print("mean field encoded context mean shape", mf_encoded_context_mean.shape)
# print("mean field encoded context log std shape", mf_encoded_context_log_std.shape)

# Sample NUM_SAMPLES times from each episode's context distribution
NUM_SAMPLES = 1
sample_rng = jax.random.PRNGKey(0)
sample_rngs = jax.random.split(sample_rng, NUM_ENVS)

def sample_from_gaussian(rng, mean, log_std):
    """Sample from a single Gaussian distribution."""
    noise = jax.random.normal(rng, shape=(NUM_SAMPLES, mean.shape[0]))
    std = jnp.exp(log_std)
    samples = mean + noise * std
    return samples

def sample_from_mean_field_gaussian(rng, means, log_stds):
    """
    Sample from a product of Gaussians distribution.
    
    For mean field context encoders that process state-action pairs individually,
    this combines multiple Gaussian predictions into a single distribution using
    the product of Gaussians approach.
    
    Args:
        rng: JAX random key
        means: Shape [seq_len, goal_dim] or [NUM_SAMPLES, seq_len, goal_dim]
        log_stds: Shape [seq_len, goal_dim] or [NUM_SAMPLES, seq_len, goal_dim]
        
    Returns:
        Samples from the combined Gaussian distribution
    """
    # Handle batched or non-batched inputs
    if means.ndim == 3:  # [NUM_SAMPLES, seq_len, goal_dim]
        # Convert to precision (inverse variance) space
        precisions = 1.0 / jnp.exp(2 * log_stds)  # shape: [NUM_SAMPLES, seq_len, goal_dim]
        
        # Compute combined precision and variance
        combined_precision = jnp.sum(precisions, axis=1)  # shape: [NUM_SAMPLES, goal_dim]
        combined_variance = 1.0 / combined_precision  # shape: [NUM_SAMPLES, goal_dim]
        
        # Weighted mean (weighted by precision)
        weighted_means = means * precisions  # shape: [NUM_SAMPLES, seq_len, goal_dim]
        combined_mean = jnp.sum(weighted_means, axis=1) / combined_precision  # shape: [NUM_SAMPLES, goal_dim]
        
        # Sample from the combined distribution
        noise = jax.random.normal(rng, shape=combined_mean.shape)
        combined_std = jnp.sqrt(combined_variance)
        samples = combined_mean + noise * combined_std
    else:  # [seq_len, goal_dim]
        # Convert to precision space
        precisions = 1.0 / jnp.exp(2 * log_stds)  # shape: [seq_len, goal_dim]
        
        # Compute combined precision and variance
        combined_precision = jnp.sum(precisions, axis=0)  # shape: [goal_dim]
        combined_variance = 1.0 / combined_precision  # shape: [goal_dim]
        
        # Weighted mean (weighted by precision)
        weighted_means = means * precisions  # shape: [seq_len, goal_dim]
        combined_mean = jnp.sum(weighted_means, axis=0) / combined_precision  # shape: [goal_dim]
        
        # Sample from the combined distribution
        noise = jax.random.normal(rng, shape=(NUM_SAMPLES, combined_mean.shape[0]))
        combined_std = jnp.sqrt(combined_variance)
        samples = combined_mean + noise * combined_std
        
    return samples

# Generate samples for each episode using standard context encoder
inferred_goals = jax.vmap(sample_from_gaussian)(
    sample_rngs,
    context_mean,
    context_log_std
)

print("inferred_goals shape:", inferred_goals.shape)
print("goal shape:", goals.shape)
goal_to_inferred_goal_distances = jnp.linalg.norm(goals - jnp.squeeze(inferred_goals, axis=1), axis=1)
print("mean goal to inferred goal distance:", jnp.mean(goal_to_inferred_goal_distances))

# Generate samples for each episode using mean field context encoder
mf_inferred_goals = jax.vmap(sample_from_mean_field_gaussian)(
    sample_rngs,
    mf_context_mean,
    mf_context_log_std
)
print("mean field inferred_goals shape:", mf_inferred_goals.shape)

goal_to_mf_inferred_goal_distances = jnp.linalg.norm(goals - jnp.squeeze(mf_inferred_goals, axis=1), axis=1)
print("mean goal to mf inferred goal distance:", jnp.mean(goal_to_mf_inferred_goal_distances))

# # Generate samples for each episode using mean field context encoder
# mf_encoded_inferred_goals = jax.vmap(sample_from_mean_field_gaussian)(
#     sample_rngs,
#     mf_encoded_context_mean,
#     mf_encoded_context_log_std
# )
# print("mean field encoded inferred_goals shape:", mf_encoded_inferred_goals.shape)

def collect_trajectory_with_target(rng, target, true_goal):
    def step_fn(carry, _):
        state, rng = carry
        act_rng, next_rng = jax.random.split(rng)
        obs = jnp.concatenate((state.obs[:env.state_dim], target), axis=-1)
        act, _ = jit_inference_fn(obs, act_rng)
        next_state = jit_env_step(state, act)
        
        # Compute distance-based reward
        current_pos = next_state.obs[env.goal_indices]
        dist_to_goal = jnp.linalg.norm(current_pos - true_goal)
        reward = jnp.where(dist_to_goal < env.goal_reach_thresh, 1.0, 0.0)
        
        return (next_state, next_rng), reward
    
    init_state = jit_env_reset(rng=rng)
    (final_state, _), rewards = jax.lax.scan(
        step_fn, 
        (init_state, rng), 
        None, 
        length=NUM_STEPS
    )
    return rewards

# Collect trajectories using last states as targets
last_state_rngs = jax.random.split(jax.random.PRNGKey(1), NUM_ENVS)
last_state_rews = jax.vmap(collect_trajectory_with_target)(
    last_state_rngs,
    last_states,
    goals
)

# Compute euclidean distances between goals and last states
goal_to_last_state_distances = jnp.linalg.norm(last_states - goals, axis=1)
print("goal_to_last_state_distances shape:", goal_to_last_state_distances.shape)
print("mean goal to last state distance:", jnp.mean(goal_to_last_state_distances))

total_rewards_last_state = jnp.sum(last_state_rews, axis=1)  # Sum rewards along trajectory dimension

# Collect trajectories using inferred goals as targets from standard context encoder
inferred_goal_rngs = jax.random.split(jax.random.PRNGKey(1), NUM_ENVS * NUM_SAMPLES)
inferred_goal_rngs = inferred_goal_rngs.reshape(NUM_ENVS, NUM_SAMPLES, -1)

inferred_goal_rews = jax.vmap(
    jax.vmap(collect_trajectory_with_target, in_axes=(0, 0, None)),
    in_axes=(0, 0, 0)
)(
    inferred_goal_rngs,
    inferred_goals,
    goals
)

print("inferred_goal_rews shape:", inferred_goal_rews.shape)
print(jnp.sum(inferred_goal_rews, axis=2).shape)
print(jnp.mean(jnp.sum(inferred_goal_rews, axis=2), axis=1).shape)
total_rewards_inferred_goal_mean = jnp.mean(jnp.sum(inferred_goal_rews, axis=2), axis=1)
total_rewards_inferred_goal_std = jnp.std(jnp.sum(inferred_goal_rews, axis=2), axis=1)

# Collect trajectories using mean field inferred goals
mf_inferred_goal_rngs = jax.random.split(jax.random.PRNGKey(2), NUM_ENVS * NUM_SAMPLES)
mf_inferred_goal_rngs = mf_inferred_goal_rngs.reshape(NUM_ENVS, NUM_SAMPLES, -1)

mf_inferred_goal_rews = jax.vmap(
    jax.vmap(collect_trajectory_with_target, in_axes=(0, 0, None)),
    in_axes=(0, 0, 0)
)(
    mf_inferred_goal_rngs,
    mf_inferred_goals,
    goals
)

print("mean field inferred_goal_rews shape:", mf_inferred_goal_rews.shape)
mf_total_rewards_inferred_goal_mean = jnp.mean(jnp.sum(mf_inferred_goal_rews, axis=2), axis=1)
mf_total_rewards_inferred_goal_std = jnp.std(jnp.sum(mf_inferred_goal_rews, axis=2), axis=1)

# # Collect trajectories using mean field encoded inferred goals
# mf_encoded_inferred_goal_rngs = jax.random.split(jax.random.PRNGKey(2), NUM_ENVS * NUM_SAMPLES)
# mf_encoded_inferred_goal_rngs = mf_encoded_inferred_goal_rngs.reshape(NUM_ENVS, NUM_SAMPLES, -1)

# mf_encoded_inferred_goal_rews = jax.vmap(
#     jax.vmap(collect_trajectory_with_target, in_axes=(0, 0, None)),
#     in_axes=(0, 0, 0)
# )(
#     mf_encoded_inferred_goal_rngs,
#     mf_encoded_inferred_goals,
#     goals
# )

# print("mean field encoded inferred_goal_rews shape:", mf_encoded_inferred_goal_rews.shape)
# mf_encoded_total_rewards_inferred_goal_mean = jnp.mean(jnp.sum(mf_encoded_inferred_goal_rews, axis=2), axis=1)
# mf_encoded_total_rewards_inferred_goal_std = jnp.std(jnp.sum(mf_encoded_inferred_goal_rews, axis=2), axis=1)

# Compute differences and their statistics for total rewards vs last state rewards
reward_diff_last_state = total_rewards - total_rewards_last_state
reward_diff_last_state_mean = jnp.mean(reward_diff_last_state)
reward_diff_last_state_stderror = jnp.std(reward_diff_last_state) / jnp.sqrt(NUM_ENVS)

print("Mean difference between total rewards and last state rewards:", reward_diff_last_state_mean)
print("Standard error of difference between total rewards and last state rewards:", reward_diff_last_state_stderror)

# Compute differences and their statistics for total rewards vs inferred goal rewards (standard context encoder)
reward_diff_inferred = total_rewards - total_rewards_inferred_goal_mean
reward_diff_inferred_mean = jnp.mean(reward_diff_inferred)
reward_diff_inferred_stderror = jnp.std(reward_diff_inferred) / jnp.sqrt(NUM_ENVS)

print("total_rewards shape:", total_rewards.shape)
print("total_rewards_inferred_goal_mean shape:", total_rewards_inferred_goal_mean.shape)
epsilon = 1e-8
reward_diff_inferred_pct = total_rewards_inferred_goal_mean / (total_rewards + epsilon)
reward_diff_inferred_pct_mean = jnp.mean(reward_diff_inferred_pct)
reward_diff_inferred_pct_stderror = jnp.std(reward_diff_inferred_pct) / jnp.sqrt(NUM_ENVS)

print("Mean difference between total rewards and inferred goal rewards (standard):", reward_diff_inferred_mean)
print("Standard error of difference between total rewards and inferred goal rewards (standard):", reward_diff_inferred_stderror)

print("Mean difference between total rewards and inferred goal rewards (standard) (percentage):", reward_diff_inferred_pct_mean)
print("Standard error of difference between total rewards and inferred goal rewards (standard) (percentage):", reward_diff_inferred_pct_stderror)

# Compute differences and their statistics for total rewards vs mean field inferred goal rewards
mf_reward_diff_inferred = total_rewards - mf_total_rewards_inferred_goal_mean
mf_reward_diff_inferred_mean = jnp.mean(mf_reward_diff_inferred)
mf_reward_diff_inferred_stderror = jnp.std(mf_reward_diff_inferred) / jnp.sqrt(NUM_ENVS)

mf_reward_diff_inferred_pct = mf_total_rewards_inferred_goal_mean / (total_rewards + epsilon)
mf_reward_diff_inferred_pct_mean = jnp.mean(mf_reward_diff_inferred_pct)
mf_reward_diff_inferred_pct_stderror = jnp.std(mf_reward_diff_inferred_pct) / jnp.sqrt(NUM_ENVS)

print("Mean difference between total rewards and inferred goal rewards (mean field):", mf_reward_diff_inferred_mean)
print("Standard error of difference between total rewards and inferred goal rewards (mean field):", mf_reward_diff_inferred_stderror)

print("Mean difference between total rewards and inferred goal rewards (mean field) (percentage):", mf_reward_diff_inferred_pct_mean)
print("Standard error of difference between total rewards and inferred goal rewards (mean field) (percentage):", mf_reward_diff_inferred_pct_stderror)

# # Compute differences and their statistics for total rewards vs mean field encoded inferred goal rewards
# mf_encoded_reward_diff_inferred = mf_encoded_total_rewards_inferred_goal_mean - total_rewards
# mf_encoded_reward_diff_inferred_mean = jnp.mean(mf_encoded_reward_diff_inferred)
# mf_encoded_reward_diff_inferred_stderror = jnp.std(mf_encoded_reward_diff_inferred) / jnp.sqrt(NUM_ENVS)

# print("Mean difference between total rewards and inferred goal rewards (mean field encoded):", mf_encoded_reward_diff_inferred_mean)
# print("Standard error of difference between total rewards and inferred goal rewards (mean field encoded):", mf_encoded_reward_diff_inferred_stderror)




### GoalKDE ###

# Process with standard context encoder
# sa_pairs = jnp.reshape(jnp.concatenate((states, actions), axis=-1), (NUM_ENVS, -1))
# print("sa pairs shape", sa_pairs.shape)
goalkde_context_output = goalkde_context_encoder(sa_pairs)
goalkde_context_mean, goalkde_context_log_std = jnp.split(goalkde_context_output, 2, axis=-1)
print("goalkde context mean shape", goalkde_context_mean.shape)
print("goalkde context log std shape", goalkde_context_log_std.shape)

# Process with mean field context encoder
# sa_pairs_mf = jnp.reshape(jnp.concatenate((states, actions), axis=-1), (NUM_ENVS * NUM_STEPS, -1))
# print("mean field sa pairs shape", sa_pairs_mf.shape)
goalkde_mf_context_output = goalkde_mean_field_context_encoder(sa_pairs_mf)
goalkde_mf_context_mean, goalkde_mf_context_log_std = jnp.split(goalkde_mf_context_output, 2, axis=-1)
goalkde_mf_context_mean = jnp.reshape(goalkde_mf_context_mean, (NUM_ENVS, NUM_STEPS, -1))
goalkde_mf_context_log_std = jnp.reshape(goalkde_mf_context_log_std, (NUM_ENVS, NUM_STEPS, -1))
print("goalkde mean field context mean shape", goalkde_mf_context_mean.shape)
print("goalkde mean field context log std shape", goalkde_mf_context_log_std.shape)

# Sample NUM_SAMPLES times from each episode's context distribution
sample_rng = jax.random.PRNGKey(0)
sample_rngs = jax.random.split(sample_rng, NUM_ENVS)

# Generate samples for each episode using standard context encoder
goalkde_inferred_goals = jax.vmap(sample_from_gaussian)(
    sample_rngs,
    goalkde_context_mean,
    goalkde_context_log_std
)
print("goalkde inferred_goals shape:", goalkde_inferred_goals.shape)

goal_to_goalkde_inferred_goal_distances = jnp.linalg.norm(goals - jnp.squeeze(goalkde_inferred_goals, axis=1), axis=1)
print("mean goal to goalkde inferred goal distance:", jnp.mean(goal_to_goalkde_inferred_goal_distances))

# Generate samples for each episode using mean field context encoder
goalkde_mf_inferred_goals = jax.vmap(sample_from_mean_field_gaussian)(
    sample_rngs,
    goalkde_mf_context_mean,
    goalkde_mf_context_log_std
)
print("goalkde mean field inferred_goals shape:", goalkde_mf_inferred_goals.shape)

goal_to_goalkde_mf_inferred_goal_distances = jnp.linalg.norm(goals - jnp.squeeze(goalkde_mf_inferred_goals, axis=1), axis=1)
print("mean goal to goalkde mf inferred goal distance:", jnp.mean(goal_to_goalkde_mf_inferred_goal_distances))

def goalkde_collect_trajectory_with_target(rng, target, true_goal):
    def step_fn(carry, _):
        state, rng = carry
        act_rng, next_rng = jax.random.split(rng)
        obs = jnp.concatenate((state.obs[:env.state_dim], target), axis=-1)
        act, _ = jit_goalkde_inference_fn(obs, act_rng)
        next_state = jit_env_step(state, act)
        
        # Compute distance-based reward
        current_pos = next_state.obs[env.goal_indices]
        dist_to_goal = jnp.linalg.norm(current_pos - true_goal)
        reward = jnp.where(dist_to_goal < env.goal_reach_thresh, 1.0, 0.0)
        
        return (next_state, next_rng), reward
    
    init_state = jit_env_reset(rng=rng)
    (final_state, _), rewards = jax.lax.scan(
        step_fn, 
        (init_state, rng), 
        None, 
        length=NUM_STEPS
    )
    return rewards


# Collect trajectories using true goals as targets
last_state_rngs = jax.random.split(jax.random.PRNGKey(1), NUM_ENVS)
goalkde_true_goal_rews = jax.vmap(goalkde_collect_trajectory_with_target)(
    last_state_rngs,
    goals,
    goals
)


# Collect trajectories using last states as targets
last_state_rngs = jax.random.split(jax.random.PRNGKey(1), NUM_ENVS)
goalkde_last_state_rews = jax.vmap(goalkde_collect_trajectory_with_target)(
    last_state_rngs,
    last_states,
    goals
)

# Compute euclidean distances between goals and last states
# goalkde_goal_distances = jnp.linalg.norm(last_states - goals, axis=2)
# print("mean goal to goal distance:", jnp.mean(goalkde_goal_distances))

print(goalkde_last_state_rews.shape)

goalkde_total_rewards_true_goal = jnp.sum(goalkde_true_goal_rews, axis=1)

goalkde_total_rewards_last_state = jnp.sum(goalkde_last_state_rews, axis=1)  # Sum rewards along trajectory dimension


# Collect trajectories using inferred goals as targets from standard context encoder
goalkde_inferred_goal_rngs = jax.random.split(jax.random.PRNGKey(1), NUM_ENVS * NUM_SAMPLES)
goalkde_inferred_goal_rngs = goalkde_inferred_goal_rngs.reshape(NUM_ENVS, NUM_SAMPLES, -1)

goalkde_inferred_goal_rews = jax.vmap(
    jax.vmap(goalkde_collect_trajectory_with_target, in_axes=(0, 0, None)),
    in_axes=(0, 0, 0)
)(
    goalkde_inferred_goal_rngs,
    goalkde_inferred_goals,
    goals
)

print("goalkde inferred_goal_rews shape:", goalkde_inferred_goal_rews.shape)
goalkde_total_rewards_inferred_goal_mean = jnp.mean(jnp.sum(goalkde_inferred_goal_rews, axis=2), axis=1)
goalkde_total_rewards_inferred_goal_std = jnp.std(jnp.sum(goalkde_inferred_goal_rews, axis=2), axis=1)

# Collect trajectories using mean field inferred goals
goalkde_mf_inferred_goal_rngs = jax.random.split(jax.random.PRNGKey(2), NUM_ENVS * NUM_SAMPLES)
goalkde_mf_inferred_goal_rngs = goalkde_mf_inferred_goal_rngs.reshape(NUM_ENVS, NUM_SAMPLES, -1)

goalkde_mf_inferred_goal_rews = jax.vmap(
    jax.vmap(goalkde_collect_trajectory_with_target, in_axes=(0, 0, None)),
    in_axes=(0, 0, 0)
)(
    goalkde_mf_inferred_goal_rngs,
    goalkde_mf_inferred_goals,
    goals
)

print("goalkde mean field inferred_goal_rews shape:", goalkde_mf_inferred_goal_rews.shape)
goalkde_mf_total_rewards_inferred_goal_mean = jnp.mean(jnp.sum(goalkde_mf_inferred_goal_rews, axis=2), axis=1)
goalkde_mf_total_rewards_inferred_goal_std = jnp.std(jnp.sum(goalkde_mf_inferred_goal_rews, axis=2), axis=1)

# Compute differences and their statistics for total rewards vs true goal rewards
goalkde_reward_diff_true_goal = total_rewards - goalkde_total_rewards_true_goal
goalkde_reward_diff_true_goal_mean = jnp.mean(goalkde_reward_diff_true_goal)
goalkde_reward_diff_true_goal_stderror = jnp.std(goalkde_reward_diff_true_goal) / jnp.sqrt(NUM_ENVS)

goalkde_reward_diff_true_goal_pct = goalkde_total_rewards_true_goal / (total_rewards + epsilon)
goalkde_reward_diff_true_goal_pct_mean = jnp.mean(goalkde_reward_diff_true_goal_pct)
goalkde_reward_diff_true_goal_pct_stderror = jnp.std(goalkde_reward_diff_true_goal_pct) / jnp.sqrt(NUM_ENVS)

# Compute differences and their statistics for total rewards vs last state rewards
goalkde_reward_diff_last_state = total_rewards - goalkde_total_rewards_last_state
goalkde_reward_diff_last_state_mean = jnp.mean(goalkde_reward_diff_last_state)
goalkde_reward_diff_last_state_stderror = jnp.std(goalkde_reward_diff_last_state) / jnp.sqrt(NUM_ENVS)

goalkde_reward_diff_last_state_pct = goalkde_total_rewards_last_state / (total_rewards + epsilon)
goalkde_reward_diff_last_state_pct_mean = jnp.mean(goalkde_reward_diff_last_state_pct)
goalkde_reward_diff_last_state_pct_stderror = jnp.std(goalkde_reward_diff_last_state_pct) / jnp.sqrt(NUM_ENVS)

print("Mean difference between total rewards and GoalKDE last state rewards:", goalkde_reward_diff_last_state_mean)
print("Standard error of difference between total rewards and GoalKDE last state rewards:", goalkde_reward_diff_last_state_stderror)

# Compute differences and their statistics for total rewards vs inferred goal rewards (standard context encoder)
goalkde_reward_diff_inferred = total_rewards - goalkde_total_rewards_inferred_goal_mean
goalkde_reward_diff_inferred_mean = jnp.mean(goalkde_reward_diff_inferred)
goalkde_reward_diff_inferred_stderror = jnp.std(goalkde_reward_diff_inferred) / jnp.sqrt(NUM_ENVS)

goalkde_reward_diff_inferred_pct = goalkde_total_rewards_inferred_goal_mean / (total_rewards + epsilon)
goalkde_reward_diff_inferred_pct_mean = jnp.mean(goalkde_reward_diff_inferred_pct)
goalkde_reward_diff_inferred_pct_stderror = jnp.std(goalkde_reward_diff_inferred_pct) / jnp.sqrt(NUM_ENVS)

print("Mean difference between total rewards and GoalKDE inferred goal rewards (standard):", goalkde_reward_diff_inferred_mean)
print("Standard error of difference between total rewards and GoalKDE inferred goal rewards (standard):", goalkde_reward_diff_inferred_stderror)

# Compute differences and their statistics for total rewards vs mean field inferred goal rewards
goalkde_mf_reward_diff_inferred = total_rewards - goalkde_mf_total_rewards_inferred_goal_mean
goalkde_mf_reward_diff_inferred_mean = jnp.mean(goalkde_mf_reward_diff_inferred)
goalkde_mf_reward_diff_inferred_stderror = jnp.std(goalkde_mf_reward_diff_inferred) / jnp.sqrt(NUM_ENVS)

goalkde_mf_reward_diff_inferred_pct = goalkde_mf_total_rewards_inferred_goal_mean / (total_rewards + epsilon)
goalkde_mf_reward_diff_inferred_pct_mean = jnp.mean(goalkde_mf_reward_diff_inferred_pct)
goalkde_mf_reward_diff_inferred_pct_stderror = jnp.std(goalkde_mf_reward_diff_inferred_pct) / jnp.sqrt(NUM_ENVS)

print("Mean difference between total rewards and GoalKDE inferred goal rewards (mean field):", goalkde_mf_reward_diff_inferred_mean)
print("Standard error of difference between total rewards and GoalKDE inferred goal rewards (mean field):", goalkde_mf_reward_diff_inferred_stderror)





# Create a new directory for the environment's results
output_dir = f"results_{env_name}"
os.makedirs(output_dir, exist_ok=True)

# Create a visualization of the performance differences
# Prepare data for plotting
methods = [
    'CRL + Oracle + Last State', 'CRL + Oracle + Full Tau', 'CRL + Oracle + Mean Field',
    'CRL + GoalKDE + True Goal', 'CRL + GoalKDE + Last State', 'CRL + GoalKDE + Full Tau', 'CRL + GoalKDE + Mean Field',    
]

mean_diffs = [
    float(reward_diff_last_state_mean),
    float(reward_diff_inferred_mean),
    float(mf_reward_diff_inferred_mean),
    float(goalkde_reward_diff_true_goal_mean),
    float(goalkde_reward_diff_last_state_mean),
    float(goalkde_reward_diff_inferred_mean),
    float(goalkde_mf_reward_diff_inferred_mean),
]

std_errors = [
    float(reward_diff_last_state_stderror),
    float(reward_diff_inferred_stderror),
    float(mf_reward_diff_inferred_stderror),
    float(goalkde_reward_diff_true_goal_stderror),
    float(goalkde_reward_diff_last_state_stderror),
    float(goalkde_reward_diff_inferred_stderror),
    float(goalkde_mf_reward_diff_inferred_stderror),
    
]

method_types = ['CRL']*3 + ['GoalKDE']*4 

df = pd.DataFrame({
    'Method': methods,
    'Mean Difference': mean_diffs,
    'Std Error': std_errors,
    'Method Type': method_types
})

# Set up the figure
plt.figure(figsize=(14, 8))

# Create the bar plot with error bars
ax = sns.barplot(
    x='Method', 
    y='Mean Difference', 
    hue='Method Type',
    data=df,
    palette=['#1f77b4', '#ff7f0e']  # CRL, GoalKDE
)

# Add error bars
for i, (_, row) in enumerate(df.iterrows()):
    ax.errorbar(
        i, row['Mean Difference'], 
        yerr=row['Std Error'], 
        fmt='none', 
        color='black', 
        capsize=5
    )

# Add a horizontal line at y=avg rew for reference (zero regret = matching expert performance)
plt.axhline(y=jnp.mean(total_rewards), color='green', linestyle='-', alpha=0.7, label=f'Mean Expert Reward: {float(jnp.mean(total_rewards)):.3f}')

# Add horizontal lines for standard error bands
# expert_stderr = float(jnp.std(total_rewards) / jnp.sqrt(NUM_ENVS))
# plt.axhline(y=jnp.mean(total_rewards)+expert_stderr, color='green', linestyle=':', alpha=0.5, label=f'+1 StdErr: {expert_stderr:.3f}')
# plt.axhline(y=jnp.mean(total_rewards)-expert_stderr, color='green', linestyle=':', alpha=0.5, label=f'-1 StdErr: {-expert_stderr:.3f}')

# Add a note about expert performance in the legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, loc='best')

# Customize the plot
plt.title(f'Regret Compared to Expert Demonstrations ({env_name})', fontsize=16)
plt.ylabel('Mean Regret', fontsize=14)
plt.xlabel('Method', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Add a note explaining the interpretation
# plt.figtext(0.5, 0.01, 
#             "Note: Higher values indicate better performance compared to expert demonstrations.\n"
#             "Error bars represent standard error of the mean.", 
#             ha='center', fontsize=10)

# Save the figure
plt.savefig(f'{output_dir}/performance_comparison_{env_name}.png', dpi=300, bbox_inches='tight')

# Save the performance comparison data to CSV
performance_df = df
performance_df.to_csv(f'{output_dir}/performance_comparison_{env_name}.csv', index=False)
print(f"Performance comparison data saved to {output_dir}/performance_comparison_{env_name}.csv")





# Create a new figure for goal distance comparison
plt.figure(figsize=(12, 6))

# Prepare data for the distance comparison plot
distance_data = {
    'Method': [
        'Last State', 
        'CRL + Oracle + Full Tau', 'CRL + Oracle + Mean Field',
        'CRL + GoalKDE + Full Tau', 'CRL + GoalKDE + Mean Field',
    ],
    'Mean Distance': [
        float(jnp.mean(goal_to_last_state_distances)),
        float(jnp.mean(goal_to_inferred_goal_distances)), 
        float(jnp.mean(goal_to_mf_inferred_goal_distances)),
        float(jnp.mean(goal_to_goalkde_inferred_goal_distances)),
        float(jnp.mean(goal_to_goalkde_mf_inferred_goal_distances)),
    ],
    'Std Error': [
        float(jnp.std(goal_to_last_state_distances) / jnp.sqrt(NUM_ENVS)),
        float(jnp.std(goal_to_inferred_goal_distances) / jnp.sqrt(NUM_ENVS)),
        float(jnp.std(goal_to_mf_inferred_goal_distances) / jnp.sqrt(NUM_ENVS)),
        float(jnp.std(goal_to_goalkde_inferred_goal_distances) / jnp.sqrt(NUM_ENVS)),
        float(jnp.std(goal_to_goalkde_mf_inferred_goal_distances) / jnp.sqrt(NUM_ENVS)),
    ],
    'Method Type': [
        'Baseline',
        'CRL', 'CRL (Mean Field)',
        'GoalKDE', 'GoalKDE (Mean Field)',
    ]
}

# Create DataFrame for the distance plot
distance_df = pd.DataFrame(distance_data)

# Create the bar plot for distances
ax = sns.barplot(
    x='Method', 
    y='Mean Distance', 
    hue='Method Type',
    data=distance_df,
    palette=['gray', '#1f77b4', '#1f77b4', '#ff7f0e', '#ff7f0e']
)

# Add error bars
for i, (_, row) in enumerate(distance_df.iterrows()):
    ax.errorbar(
        i, row['Mean Distance'], 
        yerr=row['Std Error'], 
        fmt='none', 
        color='black', 
        capsize=5
    )

# Customize the plot
plt.title(f'Goal Distance Comparison ({env_name})', fontsize=16)
plt.ylabel('Mean Distance to True Goal', fontsize=14)
plt.xlabel('Method', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Add legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, loc='best')

# Save the figure
plt.savefig(f'{output_dir}/goal_distance_comparison_{env_name}.png', dpi=300, bbox_inches='tight')

# Save the goal distance comparison data to CSV
distance_df.to_csv(f'{output_dir}/goal_distance_comparison_{env_name}.csv', index=False)
print(f"Goal distance comparison data saved to {output_dir}/goal_distance_comparison_{env_name}.csv")

# Show the plot
# plt.show()



# Create a new figure for showing whether time matters for inferring behavior (full trajectory vs. mean field), using percentages
methods = [
    'CRL + GoalKDE + Full Tau', 'CRL + GoalKDE + Mean Field',
    
]

_base_mean = float(jnp.mean(total_rewards))
_base_denom = abs(_base_mean) if _base_mean != 0.0 else 1e-8

mean_diffs = [
    float(1.0 - goalkde_reward_diff_inferred_mean/_base_denom),
    float(1.0 - goalkde_mf_reward_diff_inferred_mean/_base_denom),
]

std_errors = [
    float(abs(goalkde_reward_diff_inferred_stderror)/_base_denom),
    float(abs(goalkde_mf_reward_diff_inferred_stderror)/_base_denom),
]

method_types = ['GoalKDE']*2 

df = pd.DataFrame({
    'Method': methods,
    'Mean Difference': mean_diffs,
    'Std Error': std_errors,
    'Method Type': method_types
})

# Set up the figure
plt.figure(figsize=(14, 8))

# Create the bar plot with error bars
ax = sns.barplot(
    x='Method', 
    y='Mean Difference', 
    hue='Method Type',
    data=df,
    palette=sns.color_palette(n_colors=len(df['Method Type'].unique()))
)

# Add error bars
for i, (_, row) in enumerate(df.iterrows()):
    yerr = float(row['Std Error'])
    if not np.isfinite(yerr) or yerr < 0:
        yerr = abs(yerr)
    ax.errorbar(i, row['Mean Difference'], yerr=yerr, fmt='none', color='black', capsize=5)

# Add a horizontal line at y=avg rew for reference (zero regret = matching expert performance)
# plt.axhline(y=jnp.mean(total_rewards), color='green', linestyle='-', alpha=0.7, label=f'Mean Expert Reward: {float(jnp.mean(total_rewards)):.3f}')

# Add horizontal lines for standard error bands
# expert_stderr = float(jnp.std(total_rewards) / jnp.sqrt(NUM_ENVS))
# plt.axhline(y=jnp.mean(total_rewards)+expert_stderr, color='green', linestyle=':', alpha=0.5, label=f'+1 StdErr: {expert_stderr:.3f}')
# plt.axhline(y=jnp.mean(total_rewards)-expert_stderr, color='green', linestyle=':', alpha=0.5, label=f'-1 StdErr: {-expert_stderr:.3f}')

# Add a note about expert performance in the legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, loc='best')

# Customize the plot
plt.title(f'Imitation Score ({env_name})', fontsize=16)
plt.ylabel('Imitation Score (%)', fontsize=14)
plt.xlabel('Method', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Add a note explaining the interpretation
# plt.figtext(0.5, 0.01, 
#             "Note: Higher values indicate better performance compared to expert demonstrations.\n"
#             "Error bars represent standard error of the mean.", 
#             ha='center', fontsize=10)

# Save the figure
plt.savefig(f'{output_dir}/full_trajectory_vs_mean_field_{env_name}.png', dpi=300, bbox_inches='tight')

# Save the performance comparison data to CSV
performance_df = df
performance_df.to_csv(f'{output_dir}/full_trajectory_vs_mean_field_{env_name}.csv', index=False)
print(f"Performance comparison data saved to {output_dir}/full_trajectory_vs_mean_field_{env_name}.csv")




# Create a visualization of crl + oracle vs crl + goalkde
methods = [
    'CRL + Oracle + Mean Field', 'CRL + GoalKDE + Mean Field',
    
]

mean_diffs = [
    
    float(1.0 - mf_reward_diff_inferred_mean/jnp.mean(total_rewards)),
    float(1.0 - goalkde_mf_reward_diff_inferred_mean/jnp.mean(total_rewards)),
]

_err_base_cmp = float(jnp.mean(total_rewards))
_err_denom_cmp = abs(_err_base_cmp) if _err_base_cmp != 0.0 else 1e-8
std_errors = [
    float(abs(mf_reward_diff_inferred_stderror) / _err_denom_cmp),
    float(abs(goalkde_mf_reward_diff_inferred_stderror) / _err_denom_cmp),
]

method_types = ['CRL']*1 + ['GoalKDE']*1 

df = pd.DataFrame({
    'Method': methods,
    'Mean Difference': mean_diffs,
    'Std Error': std_errors,
    'Method Type': method_types
})

# Set up the figure
plt.figure(figsize=(14, 8))

# Create the bar plot with error bars
ax = sns.barplot(
    x='Method', 
    y='Mean Difference', 
    hue='Method Type',
    data=df,
    palette=['#1f77b4', '#ff7f0e']  # Blue for CRL, Orange for GoalKDE, Purple for NN, Green for BC, Red for FB
)

# Add error bars
for i, (_, row) in enumerate(df.iterrows()):
    yerr = float(row['Std Error'])
    if not np.isfinite(yerr) or yerr < 0:
        yerr = abs(yerr)
    ax.errorbar(i, row['Mean Difference'], yerr=yerr, fmt='none', color='black', capsize=5)

# Add a horizontal line at y=avg rew for reference (zero regret = matching expert performance)
# plt.axhline(y=jnp.mean(total_rewards), color='green', linestyle='-', alpha=0.7, label=f'Mean Expert Reward: {float(jnp.mean(total_rewards)):.3f}')

# Add horizontal lines for standard error bands
# expert_stderr = float(jnp.std(total_rewards) / jnp.sqrt(NUM_ENVS))
# plt.axhline(y=jnp.mean(total_rewards)+expert_stderr, color='green', linestyle=':', alpha=0.5, label=f'+1 StdErr: {expert_stderr:.3f}')
# plt.axhline(y=jnp.mean(total_rewards)-expert_stderr, color='green', linestyle=':', alpha=0.5, label=f'-1 StdErr: {-expert_stderr:.3f}')

# Add a note about expert performance in the legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, loc='best')

# Customize the plot
plt.title(f'Imitation Score ({env_name})', fontsize=16)
plt.ylabel('Imitation Score (%)', fontsize=14)
plt.xlabel('Method', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Add a note explaining the interpretation
# plt.figtext(0.5, 0.01, 
#             "Note: Higher values indicate better performance compared to expert demonstrations.\n"
#             "Error bars represent standard error of the mean.", 
#             ha='center', fontsize=10)

# Save the figure
plt.savefig(f'{output_dir}/crl_oracle_vs_crl_goalkde_{env_name}.png', dpi=300, bbox_inches='tight')

# Save the performance comparison data to CSV
performance_df = df
performance_df.to_csv(f'{output_dir}/crl_oracle_vs_crl_goalkde_{env_name}.csv', index=False)
print(f"Performance comparison data saved to {output_dir}/crl_oracle_vs_crl_goalkde_{env_name}.csv")


# Create a visualization for CRL + GoalKDE with error analysis
methods = [
    'True Goal', 'Last State', 'MF Inferred Goal',
]

mean_diffs = [
    float(1.0 - goalkde_reward_diff_true_goal_mean/jnp.mean(total_rewards)),
    float(1.0 - goalkde_reward_diff_last_state_mean/jnp.mean(total_rewards)),
    float(1.0 - goalkde_mf_reward_diff_inferred_mean/jnp.mean(total_rewards)),
]

_err_base = float(jnp.mean(total_rewards))
_err_denom = abs(_err_base) if _err_base != 0.0 else 1e-8
std_errors = [
    float(abs(goalkde_reward_diff_true_goal_stderror) / _err_denom),
    float(abs(goalkde_reward_diff_last_state_stderror) / _err_denom),
    float(abs(goalkde_mf_reward_diff_inferred_stderror) / _err_denom),
]

method_types = ['CRL + GoalKDE']*3

df = pd.DataFrame({
    'Method': methods,
    'Mean Difference': mean_diffs,
    'Std Error': std_errors,
    'Method Type': method_types
})

# Set up the figure
plt.figure(figsize=(12, 8))

# Create the bar plot with error bars
ax = sns.barplot(
    x='Method', 
    y='Mean Difference', 
    hue='Method Type',
    data=df,
    palette=['#ff7f0e']  # Orange for CRL + GoalKDE
)

# Add error bars (ensure non-negative, finite yerr)
for i, (_, row) in enumerate(df.iterrows()):
    yerr = float(row['Std Error'])
    if not np.isfinite(yerr) or yerr < 0:
        yerr = abs(yerr)
    ax.errorbar(i, row['Mean Difference'], yerr=yerr, fmt='none', color='black', capsize=5)

# Add dashed line from true goal to mean field
true_goal_height = mean_diffs[0]
mean_field_height = mean_diffs[2]
plt.plot([0.4, 1.6], [true_goal_height, true_goal_height], 'k--', linewidth=2)

# Add stacked bars on top of mean field bar
# First bar: from mean field height to true goal height (goal inference error)
goal_inference_bar_height = true_goal_height - mean_field_height
plt.bar(2, goal_inference_bar_height, bottom=mean_field_height, color='orange', alpha=0.6, width=0.8)

# Second bar: from true goal height to 1.0 (distribution shift error)
distribution_shift_bar_height = 1.0 - true_goal_height
plt.bar(2, distribution_shift_bar_height, bottom=true_goal_height, color='lightblue', alpha=0.6, width=0.8)

# Create custom legend with stacked bars
from matplotlib.patches import Rectangle
goal_inference_legend = Rectangle((0, 0), 1, 1, color='orange', alpha=0.6)
distribution_shift_legend = Rectangle((0, 0), 1, 1, color='lightblue', alpha=0.6)
ax.legend([goal_inference_legend, distribution_shift_legend], 
          ['Goal Inference Error', 'Distribution Shift Error'], 
          loc='best')

# Customize the plot
plt.title(f'CRL + GoalKDE Error Analysis ({env_name})', fontsize=16)
plt.ylabel('Imitation Score (%)', fontsize=14)
plt.xlabel('Method', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, 1.0)  # Ensure y-axis extends to 1.0 so the blue arrow is visible
plt.tight_layout()

# Save the figure
plt.savefig(f'{output_dir}/crl_goalkde_error_analysis_{env_name}.png', dpi=300, bbox_inches='tight')

# Save the performance comparison data to CSV
performance_df = df
performance_df.to_csv(f'{output_dir}/crl_goalkde_error_analysis_{env_name}.csv', index=False)
print(f"Performance comparison data saved to {output_dir}/crl_goalkde_error_analysis_{env_name}.csv")

