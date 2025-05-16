import sys
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

env_name = 'reacher'
# Load standard CRL checkpoint. For expert demos!
RUN_FOLDER_PATH = f'/n/fs/klips/JaxGCRL/runs/run_{env_name}-main-standard_s_1'
CKPT_NAME = '/step_20490752.pkl'
params = model.load_params(RUN_FOLDER_PATH + '/ckpt' + CKPT_NAME)
policy_params, encoders_params, context_params = params

# CRL Mean field checkpoint
MEAN_FIELD_RUN_FOLDER_PATH = f'/n/fs/klips/JaxGCRL/runs/run_{env_name}-main-meanfield_s_1'
MEAN_FIELD_CKPT_NAME = '/step_20490752.pkl'
mean_field_params = model.load_params(MEAN_FIELD_RUN_FOLDER_PATH + '/ckpt' + MEAN_FIELD_CKPT_NAME)
_, _, mean_field_context_params = mean_field_params

# # CRL Mean field encoded checkpoint
# MEAN_FIELD_ENCODED_RUN_FOLDER_PATH = f'/home/kwantlin/JaxGCRL/runs/run_{env_name}-main-meanfield-encoded_s_1'
# MEAN_FIELD_ENCODED_CKPT_NAME = '/step_11427840.pkl'
# mean_field_encoded_params = model.load_params(MEAN_FIELD_ENCODED_RUN_FOLDER_PATH + '/ckpt' + MEAN_FIELD_ENCODED_CKPT_NAME)
# mean_field_encoded_policy_params, mean_field_encoded_encoder_params, mean_field_encoded_context_params = mean_field_encoded_params
# mean_field_encoded_sa_encoder_params, _ = mean_field_encoded_encoder_params['sa_encoder'], mean_field_encoded_encoder_params['g_encoder']

# GoalKDE + CRL
GOALKDE_RUN_FOLDER_PATH = f'/n/fs/klips/JaxGCRL/runs/run_{env_name}-goalkde-standard_s_1'
GOALKDE_CKPT_NAME = '/step_20490752.pkl'
goalkde_params = model.load_params(GOALKDE_RUN_FOLDER_PATH + '/ckpt' + GOALKDE_CKPT_NAME)
goalkde_policy_params, goalkde_encoder_params, goalkde_context_params = goalkde_params

# GoalKDE + CRL Mean field
GOALKDE_MEAN_FIELD_RUN_FOLDER_PATH = f'/n/fs/klips/JaxGCRL/runs/run_{env_name}-goalkde-meanfield_s_1'
GOALKDE_MEAN_FIELD_CKPT_NAME = '/step_20490752.pkl'
goalkde_mean_field_params = model.load_params(GOALKDE_MEAN_FIELD_RUN_FOLDER_PATH + '/ckpt' + GOALKDE_MEAN_FIELD_CKPT_NAME)
_, _, goalkde_mean_field_context_params = goalkde_mean_field_params

# FB
FB_RUN_FOLDER_PATH = f'/n/fs/klips/JaxGCRL/runs/run_{env_name}-fb_s_1'
FB_CKPT_NAME = '/step_20490752.pkl'
fb_params = model.load_params(FB_RUN_FOLDER_PATH + '/ckpt' + FB_CKPT_NAME)
fb_policy_params, fb_repr_params, fb_target_forward_params, fb_target_backward_params = fb_params

# BC
BC_RUN_FOLDER_PATH = f'/n/fs/klips/JaxGCRL/runs/run_{env_name}-bc-standard_s_1'
BC_CKPT_NAME = '/step_20490752.pkl'
bc_params = model.load_params(BC_RUN_FOLDER_PATH + '/ckpt' + BC_CKPT_NAME)
bc_policy_params, bc_context_params = bc_params


# BC MEAN FIELD
BC_MEAN_FIELD_RUN_FOLDER_PATH = f'/n/fs/klips/JaxGCRL/runs/run_{env_name}-bc-meanfield_s_1'
BC_MEAN_FIELD_CKPT_NAME = '/step_20490752.pkl'
bc_mean_field_params = model.load_params(BC_MEAN_FIELD_RUN_FOLDER_PATH + '/ckpt' + BC_MEAN_FIELD_CKPT_NAME)
_, bc_mean_field_context_params = bc_mean_field_params

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
backward_repr = Net(goal_size, args.h_dim, num_blocks, block_size, args.use_ln)

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

fb_inference_fn = make_policy(actor, parametric_action_distribution, fb_policy_params)

bc_inference_fn = make_policy(actor, parametric_action_distribution, bc_policy_params)
bc_context_encoder = lambda traj: context_net.apply(bc_context_params, traj)
bc_mean_field_context_encoder = lambda traj: context_net.apply(bc_mean_field_context_params, traj)

NUM_ENVS = 2000

jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)
jit_inference_fn = jax.jit(inference_fn)
jit_goalkde_inference_fn = jax.jit(goalkde_inference_fn)
jit_fb_inference_fn = jax.jit(fb_inference_fn)
jit_bc_inference_fn = jax.jit(bc_inference_fn)
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

print("Mean difference between total rewards and inferred goal rewards (standard):", reward_diff_inferred_mean)
print("Standard error of difference between total rewards and inferred goal rewards (standard):", reward_diff_inferred_stderror)

# Compute differences and their statistics for total rewards vs mean field inferred goal rewards
mf_reward_diff_inferred = total_rewards - mf_total_rewards_inferred_goal_mean
mf_reward_diff_inferred_mean = jnp.mean(mf_reward_diff_inferred)
mf_reward_diff_inferred_stderror = jnp.std(mf_reward_diff_inferred) / jnp.sqrt(NUM_ENVS)

print("Mean difference between total rewards and inferred goal rewards (mean field):", mf_reward_diff_inferred_mean)
print("Standard error of difference between total rewards and inferred goal rewards (mean field):", mf_reward_diff_inferred_stderror)

# # Compute differences and their statistics for total rewards vs mean field encoded inferred goal rewards
# mf_encoded_reward_diff_inferred = mf_encoded_total_rewards_inferred_goal_mean - total_rewards
# mf_encoded_reward_diff_inferred_mean = jnp.mean(mf_encoded_reward_diff_inferred)
# mf_encoded_reward_diff_inferred_stderror = jnp.std(mf_encoded_reward_diff_inferred) / jnp.sqrt(NUM_ENVS)

# print("Mean difference between total rewards and inferred goal rewards (mean field encoded):", mf_encoded_reward_diff_inferred_mean)
# print("Standard error of difference between total rewards and inferred goal rewards (mean field encoded):", mf_encoded_reward_diff_inferred_stderror)




### Nearest Neighbor Policy ###

# For each demonstration, run a nearest neighbor policy rollout
# For each step, select the action from the demo whose state is closest to the current state

def nearest_neighbor_policy_rollout(demo_states, demo_actions, env, num_steps, rng, true_goal):
    def step_fn(carry, _):
        state, rng = carry
        act_rng, next_rng = jax.random.split(rng)
        # Find nearest neighbor in demo_states
        dists = jnp.linalg.norm(demo_states - state.obs[None, :env.state_dim], axis=1)
        nn_idx = jnp.argmin(dists)
        nn_action = demo_actions[nn_idx]
        next_state = jit_env_step(state, nn_action)
        # Track the L2 distance to the nearest neighbor
        nn_dist = dists[nn_idx]
        # Compute reward based on distance to true goal
        current_pos = next_state.obs[env.goal_indices]
        dist_to_goal = jnp.linalg.norm(current_pos - true_goal)
        reward = jnp.where(dist_to_goal < env.goal_reach_thresh, 1.0, 0.0)
        return (next_state, next_rng), (state, nn_action, reward, nn_dist)
    init_state = jit_env_reset(rng=rng)
    (final_state, _), (states, actions, rewards, nn_dists) = jax.lax.scan(
        step_fn,
        (init_state, rng),
        None,
        length=num_steps
    )
    return states.obs, actions, rewards, nn_dists

# Run nearest neighbor policy for each demonstration
nn_rngs = jax.random.split(jax.random.PRNGKey(42), NUM_ENVS)
nn_obs, nn_actions, nn_rewards, nn_dists = jax.vmap(nearest_neighbor_policy_rollout, in_axes=(0, 0, None, None, 0, 0))(
    states, actions, env, NUM_STEPS, nn_rngs, goals
)

# nn_obs: [NUM_ENVS, NUM_STEPS, obs_dim]
# nn_actions: [NUM_ENVS, NUM_STEPS, action_dim]
# nn_rewards: [NUM_ENVS, NUM_STEPS]
# nn_dists: [NUM_ENVS, NUM_STEPS]

nn_total_rewards = jnp.sum(nn_rewards, axis=1)
print("Nearest Neighbor Policy: total rewards per rollout (mean and stderr):", jnp.mean(nn_total_rewards), jnp.std(nn_total_rewards) / jnp.sqrt(NUM_ENVS))

# Compute the difference between nearest neighbor policy rewards and expert policy rewards
nn_expert_reward_diff = total_rewards - nn_total_rewards
nn_expert_reward_diff_mean = jnp.mean(nn_expert_reward_diff)
nn_expert_reward_diff_stderror = jnp.std(nn_expert_reward_diff) / jnp.sqrt(NUM_ENVS)

print("Mean difference between nearest neighbor and expert policy rewards:", nn_expert_reward_diff_mean)
print("Standard error of difference between nearest neighbor and expert policy rewards:", nn_expert_reward_diff_stderror)



### FB Representation ###


def fb_infer_latent(backward_repr, backward_params, states):
    backward_reprs = backward_repr.apply(backward_params, states)
    backward_reprs = backward_reprs / jnp.linalg.norm(backward_reprs, axis=-1, keepdims=True) * jnp.sqrt(goal_size)
    avg_backward_repr = jnp.mean(backward_reprs, axis=0)
    latent = avg_backward_repr / jnp.linalg.norm(avg_backward_repr) * jnp.sqrt(goal_size)
    return latent

fb_infer_latent = jax.jit(fb_infer_latent, static_argnums=0)

# Sample NUM_SAMPLES times from each episode's context distribution
sample_rng = jax.random.PRNGKey(0)
sample_rngs = jax.random.split(sample_rng, NUM_ENVS)

fb_infer_latent_partial = partial(fb_infer_latent, backward_repr)
fb_inferred_goals = jax.vmap(fb_infer_latent_partial, in_axes=(None, 0))(fb_target_backward_params, states)

print("FB inferred goals shape:", fb_inferred_goals.shape)
# Add an extra dimension at axis 1 to fb_inferred_goals
# This transforms the shape from [NUM_ENVS, goal_size] to [NUM_ENVS, 1, goal_size]
fb_inferred_goals = fb_inferred_goals[:, None, :]
print("FB inferred goals shape after adding dimension:", fb_inferred_goals.shape)

# # Remove the extra dimension for comparison
# fb_inferred_goals_flat = jnp.squeeze(fb_inferred_goals, axis=1)  # shape: [NUM_ENVS, goal_size]

# Check if all elements are (almost) equal
are_equal = jnp.allclose(fb_inferred_goals, last_states, atol=1e-5)
print("Are FB inferred goals the same as last states?", are_equal)

# Optionally, print the mean absolute difference
mean_abs_diff = jnp.mean(jnp.abs(fb_inferred_goals - last_states))
print("Mean absolute difference between FB inferred goals and last states:", mean_abs_diff)

# Calculate distances between true goals and inferred latents
goal_to_fb_inferred_goal_distances = jnp.linalg.norm(goals - fb_inferred_goals, axis=1)
print("Mean goal to FB inferred goal distance:", jnp.mean(goal_to_fb_inferred_goal_distances))

print("FB inferred goals shape:", fb_inferred_goals.shape)
print("last states shape:", last_states.shape)
print("true goals shape:", goals.shape)

def fb_collect_trajectory_with_target(rng, target, true_goal):
    def step_fn(carry, _):
        state, rng = carry
        act_rng, next_rng = jax.random.split(rng)
        obs = jnp.concatenate((state.obs[:env.state_dim], target), axis=-1)
        act, _ = jit_fb_inference_fn(obs, act_rng)
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

# # Collect trajectories using true goals as targets
# last_state_rngs = jax.random.split(jax.random.PRNGKey(1), NUM_ENVS)
# fb_true_goal_rews = jax.vmap(fb_collect_trajectory_with_target)(
#     last_state_rngs,
#     goals,
#     goals
# )


# # Collect trajectories using last states as targets
# last_state_rngs = jax.random.split(jax.random.PRNGKey(1), NUM_ENVS)
# fb_last_state_rews = jax.vmap(fb_collect_trajectory_with_target)(
#     last_state_rngs,
#     last_states,
#     goals
# )

# Compute euclidean distances between goals and last states
# goalkde_goal_distances = jnp.linalg.norm(last_states - goals, axis=2)
# print("mean goal to goal distance:", jnp.mean(goalkde_goal_distances))

# print(fb_last_state_rews.shape)

# fb_total_rewards_true_goal = jnp.sum(fb_true_goal_rews, axis=1)

# fb_total_rewards_last_state = jnp.sum(fb_last_state_rews, axis=1)  # Sum rewards along trajectory dimension

# Collect trajectories using inferred goals as targets from standard context encoder
fb_inferred_goal_rngs = jax.random.split(jax.random.PRNGKey(1), NUM_ENVS * NUM_SAMPLES)
fb_inferred_goal_rngs = fb_inferred_goal_rngs.reshape(NUM_ENVS, NUM_SAMPLES, -1)

fb_inferred_goal_rews = jax.vmap(
    jax.vmap(fb_collect_trajectory_with_target, in_axes=(0, 0, None)),
    in_axes=(0, 0, 0)
)(
    fb_inferred_goal_rngs,
    fb_inferred_goals,
    goals
)

print("fb inferred_goal_rews shape:", fb_inferred_goal_rews.shape)
fb_total_rewards_inferred_goal_mean = jnp.mean(jnp.sum(fb_inferred_goal_rews, axis=2), axis=1)
fb_total_rewards_inferred_goal_std = jnp.std(jnp.sum(fb_inferred_goal_rews, axis=2), axis=1)


# # Compute differences and their statistics for total rewards vs true goal rewards
# fb_reward_diff_true_goal = total_rewards - fb_total_rewards_true_goal
# fb_reward_diff_true_goal_mean = jnp.mean(fb_reward_diff_true_goal)
# fb_reward_diff_true_goal_stderror = jnp.std(fb_reward_diff_true_goal) / jnp.sqrt(NUM_ENVS)

# Compute differences and their statistics for total rewards vs last state rewards
fb_reward_diff_inferred = total_rewards - fb_total_rewards_inferred_goal_mean
fb_reward_diff_inferred_mean = jnp.mean(fb_reward_diff_inferred)
fb_reward_diff_inferred_stderror = jnp.std(fb_reward_diff_inferred) / jnp.sqrt(NUM_ENVS)

print("Mean difference between total rewards and FB inferred goal rewards (standard):", fb_reward_diff_inferred_mean)
print("Standard error of difference between total rewards and FB inferred goal rewards (standard):", fb_reward_diff_inferred_stderror)








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

# Compute differences and their statistics for total rewards vs last state rewards
goalkde_reward_diff_last_state = total_rewards - goalkde_total_rewards_last_state
goalkde_reward_diff_last_state_mean = jnp.mean(goalkde_reward_diff_last_state)
goalkde_reward_diff_last_state_stderror = jnp.std(goalkde_reward_diff_last_state) / jnp.sqrt(NUM_ENVS)

print("Mean difference between total rewards and GoalKDE last state rewards:", goalkde_reward_diff_last_state_mean)
print("Standard error of difference between total rewards and GoalKDE last state rewards:", goalkde_reward_diff_last_state_stderror)

# Compute differences and their statistics for total rewards vs inferred goal rewards (standard context encoder)
goalkde_reward_diff_inferred = total_rewards - goalkde_total_rewards_inferred_goal_mean
goalkde_reward_diff_inferred_mean = jnp.mean(goalkde_reward_diff_inferred)
goalkde_reward_diff_inferred_stderror = jnp.std(goalkde_reward_diff_inferred) / jnp.sqrt(NUM_ENVS)

print("Mean difference between total rewards and GoalKDE inferred goal rewards (standard):", goalkde_reward_diff_inferred_mean)
print("Standard error of difference between total rewards and GoalKDE inferred goal rewards (standard):", goalkde_reward_diff_inferred_stderror)

# Compute differences and their statistics for total rewards vs mean field inferred goal rewards
goalkde_mf_reward_diff_inferred = total_rewards - goalkde_mf_total_rewards_inferred_goal_mean
goalkde_mf_reward_diff_inferred_mean = jnp.mean(goalkde_mf_reward_diff_inferred)
goalkde_mf_reward_diff_inferred_stderror = jnp.std(goalkde_mf_reward_diff_inferred) / jnp.sqrt(NUM_ENVS)

print("Mean difference between total rewards and GoalKDE inferred goal rewards (mean field):", goalkde_mf_reward_diff_inferred_mean)
print("Standard error of difference between total rewards and GoalKDE inferred goal rewards (mean field):", goalkde_mf_reward_diff_inferred_stderror)






### BC ###

# Process with standard context encoder
# sa_pairs = jnp.reshape(jnp.concatenate((states, actions), axis=-1), (NUM_ENVS, -1))
# print("sa pairs shape", sa_pairs.shape)
bc_context_output = bc_context_encoder(sa_pairs)
bc_context_mean, bc_context_log_std = jnp.split(bc_context_output, 2, axis=-1)
print("bc context mean shape", bc_context_mean.shape)
print("bc context log std shape", bc_context_log_std.shape)

# Process with mean field context encoder
# sa_pairs_mf = jnp.reshape(jnp.concatenate((states, actions), axis=-1), (NUM_ENVS * NUM_STEPS, -1))
# print("mean field sa pairs shape", sa_pairs_mf.shape)
bc_mf_context_output = bc_mean_field_context_encoder(sa_pairs_mf)
bc_mf_context_mean, bc_mf_context_log_std = jnp.split(bc_mf_context_output, 2, axis=-1)
bc_mf_context_mean = jnp.reshape(bc_mf_context_mean, (NUM_ENVS, NUM_STEPS, -1))
bc_mf_context_log_std = jnp.reshape(bc_mf_context_log_std, (NUM_ENVS, NUM_STEPS, -1))
print("bc mean field context mean shape", bc_mf_context_mean.shape)
print("bc mean field context log std shape", bc_mf_context_log_std.shape)

# Sample NUM_SAMPLES times from each episode's context distribution
sample_rng = jax.random.PRNGKey(0)
sample_rngs = jax.random.split(sample_rng, NUM_ENVS)

# Generate samples for each episode using standard context encoder
bc_inferred_goals = jax.vmap(sample_from_gaussian)(
    sample_rngs,
    bc_context_mean,
    bc_context_log_std
)
print("bc inferred_goals shape:", bc_inferred_goals.shape)

goal_to_bc_inferred_goal_distances = jnp.linalg.norm(goals - jnp.squeeze(bc_inferred_goals, axis=1), axis=1)
print("mean goal to bc inferred goal distance:", jnp.mean(goal_to_bc_inferred_goal_distances))

# Generate samples for each episode using mean field context encoder
bc_mf_inferred_goals = jax.vmap(sample_from_mean_field_gaussian)(
    sample_rngs,
    bc_mf_context_mean,
    bc_mf_context_log_std
)
print("bc mean field inferred_goals shape:", bc_mf_inferred_goals.shape)

goal_to_bc_mf_inferred_goal_distances = jnp.linalg.norm(goals - jnp.squeeze(bc_mf_inferred_goals, axis=1), axis=1)
print("mean goal to bc mf inferred goal distance:", jnp.mean(goal_to_bc_mf_inferred_goal_distances))

def bc_collect_trajectory_with_target(rng, target, true_goal):
    def step_fn(carry, _):
        state, rng = carry
        act_rng, next_rng = jax.random.split(rng)
        obs = jnp.concatenate((state.obs[:env.state_dim], target), axis=-1)
        act, _ = jit_bc_inference_fn(obs, act_rng)
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
true_goal_rngs = jax.random.split(jax.random.PRNGKey(1), NUM_ENVS)
bc_true_goal_rews = jax.vmap(bc_collect_trajectory_with_target)(
    true_goal_rngs,
    goals,
    goals
)

# Collect trajectories using last states as targets
last_state_rngs = jax.random.split(jax.random.PRNGKey(1), NUM_ENVS)
bc_last_state_rews = jax.vmap(bc_collect_trajectory_with_target)(
    last_state_rngs,
    last_states,
    goals
)

# Compute euclidean distances between goals and last states
# bc_goal_distances = jnp.linalg.norm(last_states - goals, axis=1)
# print("mean goal to goal distance:", jnp.mean(bc_goal_distances))

bc_total_rewards_true_goal = jnp.sum(bc_true_goal_rews, axis=1)

bc_total_rewards_last_state = jnp.sum(bc_last_state_rews, axis=1)  # Sum rewards along trajectory dimension


# Collect trajectories using inferred goals as targets from standard context encoder
bc_inferred_goal_rngs = jax.random.split(jax.random.PRNGKey(1), NUM_ENVS * NUM_SAMPLES)
bc_inferred_goal_rngs = bc_inferred_goal_rngs.reshape(NUM_ENVS, NUM_SAMPLES, -1)

bc_inferred_goal_rews = jax.vmap(
    jax.vmap(bc_collect_trajectory_with_target, in_axes=(0, 0, None)),
    in_axes=(0, 0, 0)
)(
    bc_inferred_goal_rngs,
    bc_inferred_goals,
    goals
)

print("bc inferred_goal_rews shape:", bc_inferred_goal_rews.shape)
bc_total_rewards_inferred_goal_mean = jnp.mean(jnp.sum(bc_inferred_goal_rews, axis=2), axis=1)
bc_total_rewards_inferred_goal_std = jnp.std(jnp.sum(bc_inferred_goal_rews, axis=2), axis=1)

# Collect trajectories using mean field inferred goals
bc_mf_inferred_goal_rngs = jax.random.split(jax.random.PRNGKey(2), NUM_ENVS * NUM_SAMPLES)
bc_mf_inferred_goal_rngs = bc_mf_inferred_goal_rngs.reshape(NUM_ENVS, NUM_SAMPLES, -1)

bc_mf_inferred_goal_rews = jax.vmap(
    jax.vmap(bc_collect_trajectory_with_target, in_axes=(0, 0, None)),
    in_axes=(0, 0, 0)
)(
    bc_mf_inferred_goal_rngs,
    bc_mf_inferred_goals,
    goals
)

print("bc mean field inferred_goal_rews shape:", bc_mf_inferred_goal_rews.shape)
bc_mf_total_rewards_inferred_goal_mean = jnp.mean(jnp.sum(bc_mf_inferred_goal_rews, axis=2), axis=1)
bc_mf_total_rewards_inferred_goal_std = jnp.std(jnp.sum(bc_mf_inferred_goal_rews, axis=2), axis=1)

# Compute differences and their statistics for total rewards vs true goal rewards
bc_reward_diff_true_goal = total_rewards - bc_total_rewards_true_goal
bc_reward_diff_true_goal_mean = jnp.mean(bc_reward_diff_true_goal)
bc_reward_diff_true_goal_stderror = jnp.std(bc_reward_diff_true_goal) / jnp.sqrt(NUM_ENVS)

# Compute differences and their statistics for total rewards vs last state rewards
bc_reward_diff_last_state = total_rewards - bc_total_rewards_last_state
bc_reward_diff_last_state_mean = jnp.mean(bc_reward_diff_last_state)
bc_reward_diff_last_state_stderror = jnp.std(bc_reward_diff_last_state) / jnp.sqrt(NUM_ENVS)

print("Mean difference between total rewards and BC last state rewards:", bc_reward_diff_last_state_mean)
print("Standard error of difference between total rewards and BC last state rewards:", bc_reward_diff_last_state_stderror)

# Compute differences and their statistics for total rewards vs inferred goal rewards (standard context encoder)
bc_reward_diff_inferred = total_rewards - bc_total_rewards_inferred_goal_mean
bc_reward_diff_inferred_mean = jnp.mean(bc_reward_diff_inferred)
bc_reward_diff_inferred_stderror = jnp.std(bc_reward_diff_inferred) / jnp.sqrt(NUM_ENVS)

print("Mean difference between total rewards and BC inferred goal rewards (standard):", bc_reward_diff_inferred_mean)
print("Standard error of difference between total rewards and BC inferred goal rewards (standard):", bc_reward_diff_inferred_stderror)

# Compute differences and their statistics for total rewards vs mean field inferred goal rewards
bc_mf_reward_diff_inferred = total_rewards - bc_mf_total_rewards_inferred_goal_mean
bc_mf_reward_diff_inferred_mean = jnp.mean(bc_mf_reward_diff_inferred)
bc_mf_reward_diff_inferred_stderror = jnp.std(bc_mf_reward_diff_inferred) / jnp.sqrt(NUM_ENVS)

print("Mean difference between total rewards and BC inferred goal rewards (mean field):", bc_mf_reward_diff_inferred_mean)
print("Standard error of difference between total rewards and BC inferred goal rewards (mean field):", bc_mf_reward_diff_inferred_stderror)


# Create a visualization of the performance differences
# Prepare data for plotting
methods = [
    'CRL Last State', 'CRL Inferred Goal', 'CRL Mean Field',
    'GoalKDE True Goal', 'GoalKDE Last State', 'GoalKDE Inferred Goal', 'GoalKDE Mean Field',
    'Nearest Neighbor',
    'FB Inferred Goal',
    'BC True Goal', 'BC Last State', 'BC Inferred Goal', 'BC Mean Field',
    
]

mean_diffs = [
    float(reward_diff_last_state_mean),
    float(reward_diff_inferred_mean),
    float(mf_reward_diff_inferred_mean),
    float(goalkde_reward_diff_true_goal_mean),
    float(goalkde_reward_diff_last_state_mean),
    float(goalkde_reward_diff_inferred_mean),
    float(goalkde_mf_reward_diff_inferred_mean),
    float(nn_expert_reward_diff_mean),
    float(fb_reward_diff_inferred_mean),
    float(bc_reward_diff_true_goal_mean),
    float(bc_reward_diff_last_state_mean),
    float(bc_reward_diff_inferred_mean),
    float(bc_mf_reward_diff_inferred_mean),
]

std_errors = [
    float(reward_diff_last_state_stderror),
    float(reward_diff_inferred_stderror),
    float(mf_reward_diff_inferred_stderror),
    float(goalkde_reward_diff_true_goal_stderror),
    float(goalkde_reward_diff_last_state_stderror),
    float(goalkde_reward_diff_inferred_stderror),
    float(goalkde_mf_reward_diff_inferred_stderror),
    float(nn_expert_reward_diff_stderror),
    float(fb_reward_diff_inferred_stderror),
    float(bc_reward_diff_true_goal_stderror),
    float(bc_reward_diff_last_state_stderror),
    float(bc_reward_diff_inferred_stderror),
    float(bc_mf_reward_diff_inferred_stderror),
    
]

method_types = ['CRL']*3 + ['GoalKDE']*4 + ['NN']*1 + ['FB']*1 + ['BC']*4 

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
    palette=['#1f77b4', '#ff7f0e', 'purple', '#2ca02c', '#d62728']  # Blue for CRL, Orange for GoalKDE, Purple for NN, Green for BC, Red for FB
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
plt.savefig(f'performance_comparison_{env_name}.png', dpi=300, bbox_inches='tight')

# Save the performance comparison data to CSV
performance_df = df
performance_df.to_csv(f'performance_comparison_{env_name}.csv', index=False)
print(f"Performance comparison data saved to performance_comparison_{env_name}.csv")

# Create a new figure for goal distance comparison
plt.figure(figsize=(12, 6))

# Prepare data for the distance comparison plot
distance_data = {
    'Method': [
        'Last State', 
        'CRL Inferred Goal', 'CRL MF Inferred Goal',
        'GoalKDE Inferred Goal', 'GoalKDE MF Inferred Goal',
        'BC Inferred Goal', 'BC MF Inferred Goal',
    ],
    'Mean Distance': [
        float(jnp.mean(goal_to_last_state_distances)),
        float(jnp.mean(goal_to_inferred_goal_distances)), 
        float(jnp.mean(goal_to_mf_inferred_goal_distances)),
        float(jnp.mean(goal_to_goalkde_inferred_goal_distances)),
        float(jnp.mean(goal_to_goalkde_mf_inferred_goal_distances)),
        float(jnp.mean(goal_to_bc_inferred_goal_distances)),
        float(jnp.mean(goal_to_bc_mf_inferred_goal_distances)),
    ],
    'Std Error': [
        float(jnp.std(goal_to_last_state_distances) / jnp.sqrt(NUM_ENVS)),
        float(jnp.std(goal_to_inferred_goal_distances) / jnp.sqrt(NUM_ENVS)),
        float(jnp.std(goal_to_mf_inferred_goal_distances) / jnp.sqrt(NUM_ENVS)),
        float(jnp.std(goal_to_goalkde_inferred_goal_distances) / jnp.sqrt(NUM_ENVS)),
        float(jnp.std(goal_to_goalkde_mf_inferred_goal_distances) / jnp.sqrt(NUM_ENVS)),
        float(jnp.std(goal_to_bc_inferred_goal_distances) / jnp.sqrt(NUM_ENVS)),
        float(jnp.std(goal_to_bc_mf_inferred_goal_distances) / jnp.sqrt(NUM_ENVS)),
    ],
    'Method Type': [
        'Baseline',
        'CRL', 'CRL (Mean Field)',
        'GoalKDE', 'GoalKDE (Mean Field)',
        'BC', 'BC (Mean Field)'
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
    palette=['gray', '#1f77b4', '#1f77b4', '#ff7f0e', '#ff7f0e', '#d62728', '#d62728']
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
plt.savefig(f'goal_distance_comparison_{env_name}.png', dpi=300, bbox_inches='tight')

# Save the goal distance comparison data to CSV
distance_df.to_csv(f'goal_distance_comparison_{env_name}.csv', index=False)
print(f"Goal distance comparison data saved to goal_distance_comparison_{env_name}.csv")

# Show the plot
plt.show()
