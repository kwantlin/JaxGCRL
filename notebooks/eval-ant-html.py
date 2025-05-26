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

env_name = 'ant'
# Load standard CRL checkpoint. For expert demos!
RUN_FOLDER_PATH = f'/n/fs/klips/JaxGCRL/runs/run_{env_name}-main-standard_s_1'
CKPT_NAME = '/step_19522560.pkl'
params = model.load_params(RUN_FOLDER_PATH + '/ckpt' + CKPT_NAME)
policy_params, encoders_params, context_params = params

# CRL Mean field checkpoint
MEAN_FIELD_RUN_FOLDER_PATH = f'/n/fs/klips/JaxGCRL/runs/run_{env_name}-main-meanfield_s_1'
MEAN_FIELD_CKPT_NAME = '/step_19522560.pkl'
mean_field_params = model.load_params(MEAN_FIELD_RUN_FOLDER_PATH + '/ckpt' + MEAN_FIELD_CKPT_NAME)
_, _, mean_field_context_params = mean_field_params

'''
# CRL Mean field encoded checkpoint
MEAN_FIELD_ENCODED_RUN_FOLDER_PATH = f'/home/kwantlin/JaxGCRL/runs/run_{env_name}-main-meanfield-encoded_s_1'
MEAN_FIELD_ENCODED_CKPT_NAME = '/step_11427840.pkl'
mean_field_encoded_params = model.load_params(MEAN_FIELD_ENCODED_RUN_FOLDER_PATH + '/ckpt' + MEAN_FIELD_ENCODED_CKPT_NAME)
mean_field_encoded_policy_params, mean_field_encoded_encoder_params, mean_field_encoded_context_params = mean_field_encoded_params
mean_field_encoded_sa_encoder_params, _ = mean_field_encoded_encoder_params['sa_encoder'], mean_field_encoded_encoder_params['g_encoder']

# GoalKDE + CRL
GOALKDE_RUN_FOLDER_PATH = f'/n/fs/klips/JaxGCRL/runs/run_{env_name}-goalkde-standard_s_1'
GOALKDE_CKPT_NAME = '/step_15808512.pkl'
goalkde_params = model.load_params(GOALKDE_RUN_FOLDER_PATH + '/ckpt' + GOALKDE_CKPT_NAME)
goalkde_policy_params, goalkde_encoder_params, goalkde_context_params = goalkde_params

# GoalKDE + CRL Mean field
GOALKDE_MEAN_FIELD_RUN_FOLDER_PATH = f'/n/fs/klips/JaxGCRL/runs/run_{env_name}-goalkde-meanfield_s_1'
GOALKDE_MEAN_FIELD_CKPT_NAME = '/step_15808512.pkl'
goalkde_mean_field_params = model.load_params(GOALKDE_MEAN_FIELD_RUN_FOLDER_PATH + '/ckpt' + GOALKDE_MEAN_FIELD_CKPT_NAME)
_, _, goalkde_mean_field_context_params = goalkde_mean_field_params

# FB
FB_RUN_FOLDER_PATH = f'/n/fs/klips/JaxGCRL/runs/run_{env_name}-fb_s_1'
FB_CKPT_NAME = '/step_41521152.pkl'
fb_params = model.load_params(FB_RUN_FOLDER_PATH + '/ckpt' + FB_CKPT_NAME)
fb_policy_params, fb_repr_params, fb_target_forward_params, fb_target_backward_params = fb_params

# BC
BC_RUN_FOLDER_PATH = f'/n/fs/klips/JaxGCRL/runs/run_{env_name}-bc-standard_s_1'
BC_CKPT_NAME = '/step_11427840.pkl'
bc_params = model.load_params(BC_RUN_FOLDER_PATH + '/ckpt' + BC_CKPT_NAME)
bc_policy_params, bc_context_params = bc_params


# BC MEAN FIELD
BC_MEAN_FIELD_RUN_FOLDER_PATH = f'/n/fs/klips/JaxGCRL/runs/run_{env_name}-bc-meanfield_s_1'
BC_MEAN_FIELD_CKPT_NAME = '/step_11427840.pkl'
bc_mean_field_params = model.load_params(BC_MEAN_FIELD_RUN_FOLDER_PATH + '/ckpt' + BC_MEAN_FIELD_CKPT_NAME)
_, bc_mean_field_context_params = bc_mean_field_params
'''
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

'''
goalkde_inference_fn = make_policy(actor, parametric_action_distribution, goalkde_policy_params)
goalkde_context_encoder = lambda traj: context_net.apply(goalkde_context_params, traj)
goalkde_mean_field_context_encoder = lambda traj: context_net.apply(goalkde_mean_field_context_params, traj)

fb_inference_fn = make_policy(actor, parametric_action_distribution, fb_policy_params)

bc_inference_fn = make_policy(actor, parametric_action_distribution, bc_policy_params)
bc_context_encoder = lambda traj: context_net.apply(bc_context_params, traj)
bc_mean_field_context_encoder = lambda traj: context_net.apply(bc_mean_field_context_params, traj)
'''
NUM_ENVS = 5

jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)
jit_inference_fn = jax.jit(inference_fn)
'''
jit_goalkde_inference_fn = jax.jit(goalkde_inference_fn)
jit_fb_inference_fn = jax.jit(fb_inference_fn)
jit_bc_inference_fn = jax.jit(bc_inference_fn)
'''
def collect_trajectory(rng):
    def step_fn(carry, _):
        state, rng = carry
        act_rng, next_rng = jax.random.split(rng)
        act, _ = jit_inference_fn(state.obs, act_rng)
        next_state = jit_env_step(state, act)
        return (next_state, next_rng), (state, act, state.reward, state.pipeline_state)
    
    init_state = jit_env_reset(rng=rng)
    (final_state, _), (states, actions, rewards, pipeline_states) = jax.lax.scan(
        step_fn, 
        (init_state, rng), 
        None, 
        length=NUM_STEPS
    )
    return states.obs, actions, rewards, pipeline_states

# Collect trajectories across NUM_ENVS
episode_rngs = jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)
observations, actions, rewards, pipeline_states = jax.vmap(collect_trajectory)(episode_rngs)
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
        
        return (next_state, next_rng), (reward, next_state.pipeline_state)
    
    init_state = jit_env_reset(rng=rng)
    (final_state, _), (rewards, pipeline_states) = jax.lax.scan(
        step_fn, 
        (init_state, rng), 
        None, 
        length=NUM_STEPS
    )
    return rewards, pipeline_states

# Collect trajectories using last states as targets
last_state_rngs = jax.random.split(jax.random.PRNGKey(1), NUM_ENVS)
last_state_rews, last_state_pipeline_states = jax.vmap(collect_trajectory_with_target)(
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

inferred_goal_rews, inferred_goal_pipeline_states = jax.vmap(
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

mf_inferred_goal_rews, mf_inferred_goal_pipeline_states = jax.vmap(
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

# Create visualization directories
import os
viz_dir = "visualizations"
os.makedirs(viz_dir, exist_ok=True)

# Create subdirectories for each type of visualization
original_dir = os.path.join(viz_dir, "original")
last_state_dir = os.path.join(viz_dir, "last_state")
standard_inferred_dir = os.path.join(viz_dir, "standard_inferred")
mean_field_inferred_dir = os.path.join(viz_dir, "mean_field_inferred")

for d in [original_dir, last_state_dir, standard_inferred_dir, mean_field_inferred_dir]:
    os.makedirs(d, exist_ok=True)

# Helper function to convert pipeline states to list
def pipeline_states_to_list(pipeline_states):
    return [jax.tree_util.tree_map(lambda x: x[i], pipeline_states) for i in range(pipeline_states.x.pos.shape[0])]

# Visualize original trajectories
print("Rendering original trajectories...")
for i in range(NUM_ENVS):
    env_states = pipeline_states_to_list(jax.tree_util.tree_map(lambda x: x[i], pipeline_states))
    html.save(
        os.path.join(original_dir, f"trajectory_{i}.html"),
        env.sys.tree_replace({'opt.timestep': env.dt}),
        env_states
    )

# Visualize trajectories with last states as targets
print("Rendering trajectories with last states as targets...")
for i in range(NUM_ENVS):
    env_states = pipeline_states_to_list(jax.tree_util.tree_map(lambda x: x[i], last_state_pipeline_states))
    html.save(
        os.path.join(last_state_dir, f"trajectory_{i}.html"),
        env.sys.tree_replace({'opt.timestep': env.dt}),
        env_states
    )

# Visualize trajectories with standard inferred goals as targets
print("Rendering trajectories with standard inferred goals as targets...")
for i in range(NUM_ENVS):
    # For each environment, we have NUM_SAMPLES trajectories
    for j in range(NUM_SAMPLES):
        # Get the pipeline states for this environment and sample
        env_sample_states = jax.tree_util.tree_map(lambda x: x[i, j], inferred_goal_pipeline_states)
        env_states = pipeline_states_to_list(env_sample_states)
        html.save(
            os.path.join(standard_inferred_dir, f"trajectory_{i}_sample_{j}.html"),
            env.sys.tree_replace({'opt.timestep': env.dt}),
            env_states
        )

# Visualize trajectories with mean field inferred goals as targets
print("Rendering trajectories with mean field inferred goals as targets...")
for i in range(NUM_ENVS):
    # For each environment, we have NUM_SAMPLES trajectories
    for j in range(NUM_SAMPLES):
        # Get the pipeline states for this environment and sample
        env_sample_states = jax.tree_util.tree_map(lambda x: x[i, j], mf_inferred_goal_pipeline_states)
        env_states = pipeline_states_to_list(env_sample_states)
        html.save(
            os.path.join(mean_field_inferred_dir, f"trajectory_{i}_sample_{j}.html"),
            env.sys.tree_replace({'opt.timestep': env.dt}),
            env_states
        )

print(f"Visualizations saved in {viz_dir}/")
print(f"Original trajectories: {original_dir}/")
print(f"Last state trajectories: {last_state_dir}/")
print(f"Standard inferred goal trajectories: {standard_inferred_dir}/")
print(f"Mean field inferred goal trajectories: {mean_field_inferred_dir}/")




