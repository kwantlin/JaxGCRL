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
        # Return current state's obs, action, reward, current state's pipeline_state, and next_state's obs
        return (next_state, next_rng), (state.obs, act, state.reward, state.pipeline_state, next_state.obs)
    
    init_state = jit_env_reset(rng=rng)
    # Scan over the steps
    (final_state, _), (obs_at_step, actions_at_step, rewards_at_step, pipeline_states_at_step, next_obs_at_step) = jax.lax.scan(
        step_fn, 
        (init_state, rng), 
        None, 
        length=NUM_STEPS
    )
    # For matplotlib plotting, we want the sequence of achieved observations by the policy
    # obs_at_step contains the observation *before* the action was taken.
    # next_obs_at_step contains the observation *after* the action was taken.
    # We typically want to plot what was achieved, so next_obs_at_step is more relevant for trajectory path.
    # However, the first observation is from init_state, so we prepend that.
    achieved_observations = jnp.concatenate([jnp.expand_dims(init_state.obs, axis=0), next_obs_at_step[:-1]], axis=0)
    return obs_at_step, actions_at_step, rewards_at_step, pipeline_states_at_step, achieved_observations

# Collect trajectories across NUM_ENVS
episode_rngs = jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)
# original_observations here are the observations *before* taking an action at each step.
original_observations, actions, rewards, pipeline_states, original_achieved_observations_all_steps = jax.vmap(collect_trajectory)(episode_rngs)

print("original_observations (before action) shape:", original_observations.shape)
print("original_achieved_observations_all_steps (for plotting) shape:", original_achieved_observations_all_steps.shape)
print("actions shape:", actions.shape) 
print("rewards shape:", rewards.shape)
print("pipeline_states shape:", jax.tree_util.tree_map(lambda x: x.shape, pipeline_states))

states = original_observations[:, :, :env.state_dim] # This uses obs before action
goals = original_observations[:, 0, env.state_dim:] # This is the goal part of the initial obs for each env
print("states shape (from obs before action):", states.shape) 
print("goals shape (initial goal from obs):", goals.shape)

# last_states are the goal_indices from the *final achieved observation* of the original trajectories
last_states = original_achieved_observations_all_steps[:, -1, env.goal_indices]
print("last_states (achieved, for commanding) shape:", last_states.shape)

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
        # Construct observation for policy: current state + commanded target
        obs_for_policy = jnp.concatenate((state.obs[:env.state_dim], target), axis=-1)
        act, _ = jit_inference_fn(obs_for_policy, act_rng)
        next_state = jit_env_step(state, act)
        
        # Compute distance-based reward based on current achieved position and the true_goal
        current_achieved_pos = next_state.obs[env.goal_indices]
        dist_to_goal = jnp.linalg.norm(current_achieved_pos - true_goal)
        reward = jnp.where(dist_to_goal < env.goal_reach_thresh, 1.0, 0.0)
        
        # Return reward and the full observation of the next state
        return (next_state, next_rng), (reward, next_state.obs)
    
    init_state = jit_env_reset(rng=rng)
    # Accumulate rewards and observations over time
    (final_state, _), (rewards, obs_over_time) = jax.lax.scan(
        step_fn, 
        (init_state, rng), 
        None, 
        length=NUM_STEPS
    )
    return rewards, obs_over_time

# Collect trajectories using last states as targets
last_state_rngs = jax.random.split(jax.random.PRNGKey(1), NUM_ENVS)
last_state_rews, last_state_obs_all_steps = jax.vmap(collect_trajectory_with_target)(
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

inferred_goal_rews, inferred_goal_obs_all_steps = jax.vmap(
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

mf_inferred_goal_rews, mf_inferred_goal_obs_all_steps = jax.vmap(
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

# Helper function to convert pipeline states to list (still used for original trajectories)
def pipeline_states_to_list(pipeline_states):
    return [jax.tree_util.tree_map(lambda x: x[i], pipeline_states) for i in range(pipeline_states.x.pos.shape[0])]

# Matplotlib plotting function
def plot_trajectory_matplotlib(filepath, commanded_trajectory_obs_single_rollout, original_expert_obs_single_rollout, true_goal_single_env, commanded_goal_single_rollout, title):
    # Ensure data is NumPy array for Matplotlib
    commanded_trajectory_obs_np = np.array(commanded_trajectory_obs_single_rollout)
    original_expert_obs_np = np.array(original_expert_obs_single_rollout) # Added for original trajectory
    true_goal_np = np.array(true_goal_single_env)
    commanded_goal_np = np.array(commanded_goal_single_rollout)

    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Extract (x,y) positions for commanded trajectory
    commanded_positions_over_time = commanded_trajectory_obs_np[:, env.goal_indices]
    num_steps_commanded = commanded_positions_over_time.shape[0]
    # Use 'Blues' colormap for commanded policy rollout
    colors_commanded = plt.cm.Blues(np.linspace(0.3, 1, num_steps_commanded)) # Start from a slightly darker blue
    
    # Plot commanded trajectory's (x,y) position over time
    scatter_commanded = ax.scatter(commanded_positions_over_time[:, 0], commanded_positions_over_time[:, 1], c=colors_commanded, label='Inferred Goal Policy Rollout', s=25, alpha=0.8, zorder=3, cmap='Blues')
    ax.plot(commanded_positions_over_time[:, 0], commanded_positions_over_time[:, 1], alpha=0.6, linewidth=0.8, color=colors_commanded[0] if num_steps_commanded > 0 else 'blue', zorder=2) # Connect points

    # Extract (x,y) positions for original/expert trajectory
    original_positions_over_time = original_expert_obs_np[:, env.goal_indices]
    num_steps_original = original_positions_over_time.shape[0]
    # Use 'Reds' colormap for original demonstrated trajectory
    colors_original = plt.cm.Reds(np.linspace(0.3, 1, num_steps_original)) # Start from a slightly darker red

    # Plot original/expert trajectory's (x,y) position over time
    scatter_original = ax.scatter(original_positions_over_time[:, 0], original_positions_over_time[:, 1], c=colors_original, label='Original Demonstrated Trajectory', s=25, alpha=0.8, marker='^', zorder=3, cmap='Reds')
    ax.plot(original_positions_over_time[:, 0], original_positions_over_time[:, 1], alpha=0.6, linewidth=0.8, color=colors_original[0] if num_steps_original > 0 else 'red', linestyle='--', zorder=2) # Connect points, dashed
    
    # Plot true goal
    ax.scatter(true_goal_np[0], true_goal_np[1], marker='X', color='limegreen', s=300, label=f'True Goal ({true_goal_np[0]:.2f}, {true_goal_np[1]:.2f})', zorder=5, edgecolors='black', linewidth=1)
    
    # Plot commanded goal
    ax.scatter(commanded_goal_np[0], commanded_goal_np[1], marker='P', color='gold', s=300, label=f'Commanded Goal ({commanded_goal_np[0]:.2f}, {commanded_goal_np[1]:.2f})', zorder=5, edgecolors='black', linewidth=1)
    
    ax.set_xlabel("X Position", fontsize=12)
    ax.set_ylabel("Y Position", fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=10)
    ax.axis('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a colorbar for commanded trajectory
    cbar_commanded = fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=1)), 
                                ax=ax, fraction=0.040, pad=0.04)
    cbar_commanded.set_ticks([0, 0.5, 1])
    cbar_commanded.set_ticklabels(['Start', 'Mid', 'End'])
    cbar_commanded.set_label('Inferred Goal Policy Rollout Time', labelpad=-40, fontsize=9)

    # Add a colorbar for original trajectory
    cbar_original = fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1)), 
                               ax=ax, fraction=0.040, pad=0.12)
    cbar_original.set_ticks([0, 0.5, 1])
    cbar_original.set_ticklabels(['Start', 'Mid', 'End'])
    cbar_original.set_label('Original Trajectory Time', labelpad=-40, fontsize=9)

    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)

# Visualize original trajectories (HTML)
print("Rendering original trajectories (HTML)...")
for i in range(NUM_ENVS):
    env_states = pipeline_states_to_list(jax.tree_util.tree_map(lambda x: x[i], pipeline_states))
    html.save(
        os.path.join(original_dir, f"trajectory_{i}.html"),
        env.sys.tree_replace({'opt.timestep': env.dt}),
        env_states
    )

# Visualize trajectories with last states as targets (Matplotlib)
print("Plotting trajectories with last states as targets (Matplotlib)...")
for i in range(NUM_ENVS):
    commanded_traj_obs = last_state_obs_all_steps[i] 
    original_expert_obs = original_achieved_observations_all_steps[i]
    true_g = goals[i]
    commanded_g = last_states[i]
    filepath = os.path.join(last_state_dir, f"trajectory_{i}.png")
    title = f"Env {i}: Last State as Target vs Original Trajectory"
    plot_trajectory_matplotlib(filepath, commanded_traj_obs, original_expert_obs, true_g, commanded_g, title)

# Visualize trajectories with standard inferred goals as targets (Matplotlib)
print("Plotting trajectories with standard inferred goals as targets (Matplotlib)...")
for i in range(NUM_ENVS):
    original_expert_obs = original_achieved_observations_all_steps[i]
    true_g = goals[i]
    for j in range(NUM_SAMPLES):
        commanded_traj_obs = inferred_goal_obs_all_steps[i, j]
        commanded_g = inferred_goals[i, j]
        filepath = os.path.join(standard_inferred_dir, f"trajectory_{i}_sample_{j}.png")
        title = f"Env {i}, Sample {j}: Standard Inferred Goal vs Original Trajectory"
        plot_trajectory_matplotlib(filepath, commanded_traj_obs, original_expert_obs, true_g, commanded_g, title)

# Visualize trajectories with mean field inferred goals as targets (Matplotlib)
print("Plotting trajectories with mean field inferred goals as targets (Matplotlib)...")
for i in range(NUM_ENVS):
    original_expert_obs = original_achieved_observations_all_steps[i]
    true_g = goals[i]
    for j in range(NUM_SAMPLES):
        commanded_traj_obs = mf_inferred_goal_obs_all_steps[i, j]
        commanded_g = mf_inferred_goals[i, j]
        filepath = os.path.join(mean_field_inferred_dir, f"trajectory_{i}_sample_{j}.png")
        title = f"Env {i}, Sample {j}: Mean Field Inferred Goal vs Original Trajectory"
        plot_trajectory_matplotlib(filepath, commanded_traj_obs, original_expert_obs, true_g, commanded_g, title)

print(f"Visualizations saved in {viz_dir}/")
print(f"Original trajectories (HTML): {original_dir}/")
print(f"Last state trajectories (PNGs): {last_state_dir}/")
print(f"Standard inferred goal trajectories (PNGs): {standard_inferred_dir}/")
print(f"Mean field inferred goal trajectories (PNGs): {mean_field_inferred_dir}/")



