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


# Load standard CRL checkpoint. For expert demos!
RUN_FOLDER_PATH = '/home/kwantlin/JaxGCRL/runs/run_ant-main_s_1'
CKPT_NAME = '/step_11427840.pkl'
params = model.load_params(RUN_FOLDER_PATH + '/ckpt' + CKPT_NAME)
policy_params, encoders_params, context_params = params

# CRL Mean field checkpoint
MEAN_FIELD_RUN_FOLDER_PATH = '/home/kwantlin/JaxGCRL/runs/run_ant-main-meanfield_s_1'
MEAN_FIELD_CKPT_NAME = '/step_11427840.pkl'
mean_field_params = model.load_params(MEAN_FIELD_RUN_FOLDER_PATH + '/ckpt' + MEAN_FIELD_CKPT_NAME)
_, _, mean_field_context_params = mean_field_params

# CRL Mean field encoded checkpoint
MEAN_FIELD_ENCODED_RUN_FOLDER_PATH = '/home/kwantlin/JaxGCRL/runs/run_ant-main-meanfield-encoded_s_1'
MEAN_FIELD_ENCODED_CKPT_NAME = '/step_11427840.pkl'
mean_field_encoded_params = model.load_params(MEAN_FIELD_ENCODED_RUN_FOLDER_PATH + '/ckpt' + MEAN_FIELD_ENCODED_CKPT_NAME)
mean_field_encoded_policy_params, mean_field_encoded_encoder_params, mean_field_encoded_context_params = mean_field_encoded_params
mean_field_encoded_sa_encoder_params, _ = mean_field_encoded_encoder_params['sa_encoder'], mean_field_encoded_encoder_params['g_encoder']

# BC
BC_RUN_FOLDER_PATH = '/home/kwantlin/JaxGCRL/runs/run_ant-bc_s_1'
BC_CKPT_NAME = '/step_11427840.pkl'
bc_params = model.load_params(BC_RUN_FOLDER_PATH + '/ckpt' + BC_CKPT_NAME)
bc_policy_params, bc_context_params = bc_params


# BC MEAN FIELD
BC_MEAN_FIELD_RUN_FOLDER_PATH = '/home/kwantlin/JaxGCRL/runs/run_ant-bc-meanfield_s_1'
BC_MEAN_FIELD_CKPT_NAME = '/step_11427840.pkl'
bc_mean_field_params = model.load_params(BC_MEAN_FIELD_RUN_FOLDER_PATH + '/ckpt' + BC_MEAN_FIELD_CKPT_NAME)
_, bc_mean_field_context_params = bc_mean_field_params

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
parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size) # Would like to replace this but it's annoying to.

inference_fn = make_policy(actor, parametric_action_distribution, policy_params)

# sa_encoder = lambda obs: sa_net.apply(sa_encoder_params, obs)
# g_encoder = lambda obs: g_net.apply(g_encoder_params, obs)
context_encoder = lambda traj: context_net.apply(context_params, traj)
mean_field_context_encoder = lambda traj: context_net.apply(mean_field_context_params, traj)
# Mean field encoded context encoder, which uses the sa_net to encode the state-action pairs
mean_field_encoded_sa_net = Net(args.repr_dim, args.h_dim, num_blocks, block_size, args.use_ln)
mean_field_encoded_sa_encoder = lambda obs: mean_field_encoded_sa_net.apply(mean_field_encoded_sa_encoder_params, obs)
mean_field_encoded_context_encoder = lambda traj: context_net.apply(mean_field_encoded_context_params, traj)

bc_inference_fn = make_policy(actor, parametric_action_distribution, bc_policy_params)
bc_context_encoder = lambda traj: context_net.apply(bc_context_params, traj)
bc_mean_field_context_encoder = lambda traj: context_net.apply(bc_mean_field_context_params, traj)

NUM_ENVS = 1000

jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)
jit_inference_fn = jax.jit(inference_fn)
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
print("Total rewards per rollout (mean and std):", jnp.mean(total_rewards), jnp.std(total_rewards))


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
mf_encoded_encoder_output = mean_field_encoded_sa_encoder(sa_pairs_mf)
print("mean field encoded encoder output shape", mf_encoded_encoder_output.shape)
mf_encoded_context_output = mean_field_encoded_context_encoder(mf_encoded_encoder_output)
print("mean field encoded context output shape", mf_encoded_context_output.shape)
mf_encoded_context_mean, mf_encoded_context_log_std = jnp.split(mf_encoded_context_output, 2, axis=-1)
mf_encoded_context_mean = jnp.reshape(mf_encoded_context_mean, (NUM_ENVS, NUM_STEPS, -1))
mf_encoded_context_log_std = jnp.reshape(mf_encoded_context_log_std, (NUM_ENVS, NUM_STEPS, -1))
print("mean field encoded context mean shape", mf_encoded_context_mean.shape)
print("mean field encoded context log std shape", mf_encoded_context_log_std.shape)

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

# Generate samples for each episode using mean field context encoder
mf_inferred_goals = jax.vmap(sample_from_mean_field_gaussian)(
    sample_rngs,
    mf_context_mean,
    mf_context_log_std
)
print("mean field inferred_goals shape:", mf_inferred_goals.shape)


# Generate samples for each episode using mean field context encoder
mf_encoded_inferred_goals = jax.vmap(sample_from_mean_field_gaussian)(
    sample_rngs,
    mf_encoded_context_mean,
    mf_encoded_context_log_std
)
print("mean field encoded inferred_goals shape:", mf_encoded_inferred_goals.shape)

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
goal_distances = jnp.linalg.norm(last_states - goals, axis=1)

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

# Collect trajectories using mean field encoded inferred goals
mf_encoded_inferred_goal_rngs = jax.random.split(jax.random.PRNGKey(2), NUM_ENVS * NUM_SAMPLES)
mf_encoded_inferred_goal_rngs = mf_encoded_inferred_goal_rngs.reshape(NUM_ENVS, NUM_SAMPLES, -1)

mf_encoded_inferred_goal_rews = jax.vmap(
    jax.vmap(collect_trajectory_with_target, in_axes=(0, 0, None)),
    in_axes=(0, 0, 0)
)(
    mf_encoded_inferred_goal_rngs,
    mf_encoded_inferred_goals,
    goals
)

print("mean field encoded inferred_goal_rews shape:", mf_encoded_inferred_goal_rews.shape)
mf_encoded_total_rewards_inferred_goal_mean = jnp.mean(jnp.sum(mf_encoded_inferred_goal_rews, axis=2), axis=1)
mf_encoded_total_rewards_inferred_goal_std = jnp.std(jnp.sum(mf_encoded_inferred_goal_rews, axis=2), axis=1)

# Compute differences and their statistics for total rewards vs last state rewards
reward_diff_last_state = total_rewards_last_state - total_rewards
reward_diff_last_state_mean = jnp.mean(reward_diff_last_state)
reward_diff_last_state_stderror = jnp.std(reward_diff_last_state) / jnp.sqrt(NUM_ENVS)

print("Mean difference between total rewards and last state rewards:", reward_diff_last_state_mean)
print("Standard error of difference between total rewards and last state rewards:", reward_diff_last_state_stderror)

# Compute differences and their statistics for total rewards vs inferred goal rewards (standard context encoder)
reward_diff_inferred = total_rewards_inferred_goal_mean - total_rewards
reward_diff_inferred_mean = jnp.mean(reward_diff_inferred)
reward_diff_inferred_stderror = jnp.std(reward_diff_inferred) / jnp.sqrt(NUM_ENVS)

print("Mean difference between total rewards and inferred goal rewards (standard):", reward_diff_inferred_mean)
print("Standard error of difference between total rewards and inferred goal rewards (standard):", reward_diff_inferred_stderror)

# Compute differences and their statistics for total rewards vs mean field inferred goal rewards
mf_reward_diff_inferred = mf_total_rewards_inferred_goal_mean - total_rewards
mf_reward_diff_inferred_mean = jnp.mean(mf_reward_diff_inferred)
mf_reward_diff_inferred_stderror = jnp.std(mf_reward_diff_inferred) / jnp.sqrt(NUM_ENVS)

print("Mean difference between total rewards and inferred goal rewards (mean field):", mf_reward_diff_inferred_mean)
print("Standard error of difference between total rewards and inferred goal rewards (mean field):", mf_reward_diff_inferred_stderror)

# Compute differences and their statistics for total rewards vs mean field encoded inferred goal rewards
mf_encoded_reward_diff_inferred = mf_encoded_total_rewards_inferred_goal_mean - total_rewards
mf_encoded_reward_diff_inferred_mean = jnp.mean(mf_encoded_reward_diff_inferred)
mf_encoded_reward_diff_inferred_stderror = jnp.std(mf_encoded_reward_diff_inferred) / jnp.sqrt(NUM_ENVS)

print("Mean difference between total rewards and inferred goal rewards (mean field encoded):", mf_encoded_reward_diff_inferred_mean)
print("Standard error of difference between total rewards and inferred goal rewards (mean field encoded):", mf_encoded_reward_diff_inferred_stderror)






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

# Generate samples for each episode using mean field context encoder
bc_mf_inferred_goals = jax.vmap(sample_from_mean_field_gaussian)(
    sample_rngs,
    bc_mf_context_mean,
    bc_mf_context_log_std
)
print("bc mean field inferred_goals shape:", bc_mf_inferred_goals.shape)

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

# Collect trajectories using last states as targets
last_state_rngs = jax.random.split(jax.random.PRNGKey(1), NUM_ENVS)
bc_last_state_rews = jax.vmap(bc_collect_trajectory_with_target)(
    last_state_rngs,
    last_states,
    goals
)

# Compute euclidean distances between goals and last states
bc_goal_distances = jnp.linalg.norm(last_states - goals, axis=1)

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

# Compute differences and their statistics for total rewards vs last state rewards
bc_reward_diff_last_state = bc_total_rewards_last_state - total_rewards
bc_reward_diff_last_state_mean = jnp.mean(bc_reward_diff_last_state)
bc_reward_diff_last_state_stderror = jnp.std(bc_reward_diff_last_state) / jnp.sqrt(NUM_ENVS)

print("Mean difference between total rewards and BC last state rewards:", bc_reward_diff_last_state_mean)
print("Standard error of difference between total rewards and BC last state rewards:", bc_reward_diff_last_state_stderror)

# Compute differences and their statistics for total rewards vs inferred goal rewards (standard context encoder)
bc_reward_diff_inferred = bc_total_rewards_inferred_goal_mean - total_rewards
bc_reward_diff_inferred_mean = jnp.mean(bc_reward_diff_inferred)
bc_reward_diff_inferred_stderror = jnp.std(bc_reward_diff_inferred) / jnp.sqrt(NUM_ENVS)

print("Mean difference between total rewards and BC inferred goal rewards (standard):", bc_reward_diff_inferred_mean)
print("Standard error of difference between total rewards and BC inferred goal rewards (standard):", bc_reward_diff_inferred_stderror)

# Compute differences and their statistics for total rewards vs mean field inferred goal rewards
bc_mf_reward_diff_inferred = bc_mf_total_rewards_inferred_goal_mean - total_rewards
bc_mf_reward_diff_inferred_mean = jnp.mean(bc_mf_reward_diff_inferred)
bc_mf_reward_diff_inferred_stderror = jnp.std(bc_mf_reward_diff_inferred) / jnp.sqrt(NUM_ENVS)

print("Mean difference between total rewards and BC inferred goal rewards (mean field):", bc_mf_reward_diff_inferred_mean)
print("Standard error of difference between total rewards and BC inferred goal rewards (mean field):", bc_mf_reward_diff_inferred_stderror)


# Create a visualization of the performance differences
# Prepare data for plotting
methods = ['CRL Last State', 'CRL Inferred Goal', 'CRL Mean Field', 'CRL Mean Field Encoded',
           'BC Last State', 'BC Inferred Goal', 'BC Mean Field']

# Collect all the mean differences and convert from JAX arrays to numpy arrays
mean_diffs = [
    float(reward_diff_last_state_mean),
    float(reward_diff_inferred_mean),
    float(mf_reward_diff_inferred_mean),
    float(mf_encoded_reward_diff_inferred_mean),
    float(bc_reward_diff_last_state_mean),
    float(bc_reward_diff_inferred_mean),
    float(bc_mf_reward_diff_inferred_mean)
]

# Collect all the standard errors and convert from JAX arrays to numpy arrays
std_errors = [
    float(reward_diff_last_state_stderror),
    float(reward_diff_inferred_stderror),
    float(mf_reward_diff_inferred_stderror),
    float(mf_encoded_reward_diff_inferred_stderror),
    float(bc_reward_diff_last_state_stderror),
    float(bc_reward_diff_inferred_stderror),
    float(bc_mf_reward_diff_inferred_stderror)
]

# Create a DataFrame for easier plotting with seaborn
df = pd.DataFrame({
    'Method': methods,
    'Mean Difference': mean_diffs,
    'Std Error': std_errors,
    'Method Type': ['CRL']*4 + ['BC']*3
})

# Set up the figure
plt.figure(figsize=(12, 8))

# Create the bar plot with error bars
ax = sns.barplot(
    x='Method', 
    y='Mean Difference', 
    hue='Method Type',
    data=df,
    palette=['#1f77b4', '#ff7f0e']  # Blue for CRL, Orange for BC
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

# Add a horizontal line at y=0 for reference
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

# Customize the plot
plt.title('Performance Difference Compared to Expert Demonstrations', fontsize=16)
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
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


