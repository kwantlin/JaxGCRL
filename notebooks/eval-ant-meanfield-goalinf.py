import sys
sys.path.append('../')
import os
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
from scipy.stats import multivariate_normal
import shutil
import subprocess

# note: ant: step_11427840
# note: reacher: step_20490752
# note: simple_u_maze: step_20490752
# note: pusher_easy: step_30823424

env_name = 'ant'
# Load standard CRL checkpoint. For expert demos!
RUN_FOLDER_PATH = f'/n/fs/klips/JaxGCRL/runs/archive-beforemaxentedit/run_{env_name}-main-standard_s_1'
CKPT_NAME = '/step_11427840.pkl'
params = model.load_params(RUN_FOLDER_PATH + '/ckpt' + CKPT_NAME)
policy_params, encoders_params, context_params = params

# CRL Mean field checkpoint
MEAN_FIELD_RUN_FOLDER_PATH = f'/n/fs/klips/JaxGCRL/runs/archive-beforemaxentedit/run_{env_name}-main-meanfield_s_1'
MEAN_FIELD_CKPT_NAME = '/step_11427840.pkl'
mean_field_params = model.load_params(MEAN_FIELD_RUN_FOLDER_PATH + '/ckpt' + MEAN_FIELD_CKPT_NAME)
_, _, mean_field_context_params = mean_field_params

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


def get_mean_field_distribution_params(means, log_stds):
    """
    Computes the parameters of the combined Gaussian distribution from a
    product of Gaussians (mean field).
    """
    precisions = 1.0 / jnp.exp(2 * log_stds)
    combined_precision = jnp.sum(precisions, axis=0)
    combined_variance = 1.0 / combined_precision
    weighted_means = means * precisions
    combined_mean = jnp.sum(weighted_means, axis=0) / combined_precision
    return combined_mean, combined_variance

# Get combined distribution parameters for each environment
mf_dist_means, mf_dist_variances = jax.vmap(get_mean_field_distribution_params)(
    mf_context_mean,
    mf_context_log_std
)


# Plotting
def plot_trajectory_with_goals(filepath, expert_trajectory_obs, true_goal, inferred_goal, title, env, dist_mean, dist_variance):
    # Ensure data is NumPy array for Matplotlib
    expert_trajectory_obs_np = np.array(expert_trajectory_obs)
    true_goal_np = np.array(true_goal)
    inferred_goal_np = np.array(inferred_goal)
    dist_mean_np = np.array(dist_mean)
    dist_variance_np = np.array(dist_variance)
    dist_std_np = np.sqrt(dist_variance_np)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(title, fontsize=16)
    
    # --- Subplot 1: Trajectory Plot ---
    
    # Extract (x,y) positions for expert trajectory
    expert_positions_over_time = expert_trajectory_obs_np[:, env.goal_indices]
    num_steps = expert_positions_over_time.shape[0]
    colors = plt.cm.Reds(np.linspace(0.3, 1, num_steps))

    # Plot expert trajectory's (x,y) position over time
    ax1.scatter(expert_positions_over_time[:, 0], expert_positions_over_time[:, 1], c=colors, label='Expert Trajectory', s=25, alpha=0.8, marker='^', zorder=3)
    ax1.plot(expert_positions_over_time[:, 0], expert_positions_over_time[:, 1], alpha=0.6, linewidth=0.8, color=colors[0] if num_steps > 0 else 'red', linestyle='--', zorder=2)
    
    # Plot true goal
    ax1.scatter(true_goal_np[0], true_goal_np[1], marker='X', color='limegreen', s=300, label=f'True Goal ({true_goal_np[0]:.2f}, {true_goal_np[1]:.2f})', zorder=5, edgecolors='black', linewidth=1)
    
    # Plot inferred goal
    ax1.scatter(inferred_goal_np[0], inferred_goal_np[1], marker='P', color='gold', s=300, label=f'Inferred Goal ({inferred_goal_np[0]:.2f}, {inferred_goal_np[1]:.2f})', zorder=5, edgecolors='black', linewidth=1)
    
    ax1.set_xlabel("X Position", fontsize=12)
    ax1.set_ylabel("Y Position", fontsize=12)
    ax1.set_title("Expert Trajectory and Goals", fontsize=14)
    ax1.legend(loc='best', fontsize=10)
    ax1.axis('equal')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add a colorbar for trajectory
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1)), 
                                ax=ax1, fraction=0.040, pad=0.04)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Start', 'Mid', 'End'])
    cbar.set_label('Expert Trajectory Time', labelpad=-40, fontsize=9)

    # --- Subplot 2: Heatmap of Inferred Goal Distribution ---

    # Create a grid of points
    x_lim = ax1.get_xlim()
    y_lim = ax1.get_ylim()
    x = np.linspace(x_lim[0], x_lim[1], 100)
    y = np.linspace(y_lim[0], y_lim[1], 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    # Calculate PDF of the 2D Gaussian
    covariance_matrix = np.diag(dist_variance_np)
    rv = multivariate_normal(dist_mean_np, covariance_matrix)
    Z = rv.pdf(pos)

    # Plot the heatmap
    ax2.contourf(X, Y, Z, levels=20, cmap='viridis')
    ax2.scatter(true_goal_np[0], true_goal_np[1], marker='X', color='limegreen', s=200, label='True Goal', edgecolors='black')
    ax2.scatter(inferred_goal_np[0], inferred_goal_np[1], marker='P', color='gold', s=200, label='Inferred Goal (Sample)', edgecolors='black')

    ax2.set_xlabel("X Position", fontsize=12)
    ax2.set_ylabel("Y Position", fontsize=12)
    ax2.set_title("Inferred Goal Distribution", fontsize=14)
    ax2.legend(loc='best', fontsize=10)
    ax2.axis('equal')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Add text for mean and std
    textstr = '\n'.join((
        r'$\mu_x=%.2f, \mu_y=%.2f$' % (dist_mean_np[0], dist_mean_np[1], ),
        r'$\sigma_x=%.2f, \sigma_y=%.2f$' % (dist_std_np[0], dist_std_np[1], )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)

# Create visualization directories
viz_dir = "visualizations_meanfield_goalinf"
os.makedirs(viz_dir, exist_ok=True)

# Visualize trajectories with mean field inferred goals as targets (Matplotlib)
print("Plotting expert trajectories with mean field inferred goals (Matplotlib)...")
for i in range(NUM_ENVS):
    expert_obs = original_achieved_observations_all_steps[i]
    true_g = goals[i]
    dist_mean = mf_dist_means[i]
    dist_var = mf_dist_variances[i]
    for j in range(NUM_SAMPLES): # This loop will run once if NUM_SAMPLES=1
        inferred_g = mf_inferred_goals[i, j]
        filepath = os.path.join(viz_dir, f"trajectory_{i}_sample_{j}.png")
        title = f"Env {i}, Sample {j}: Expert Trajectory with Mean Field Inferred Goal"
        plot_trajectory_with_goals(filepath, expert_obs, true_g, inferred_g, title, env, dist_mean, dist_var)

print(f"Visualizations saved in {viz_dir}/")


# --- Video Generation ---

def plot_posterior_frame(filepath, full_expert_obs, observed_expert_obs, true_goal, dist_mean, dist_variance, env, timestep):
    full_expert_obs_np = np.array(full_expert_obs)
    observed_expert_obs_np = np.array(observed_expert_obs)
    true_goal_np = np.array(true_goal)
    dist_mean_np = np.array(dist_mean)
    dist_variance_np = np.array(dist_variance)
    dist_std_np = np.sqrt(dist_variance_np)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(f"Posterior Distribution at Timestep {timestep}", fontsize=16)

    # --- Determine plot bounds from the full trajectory ---
    full_positions = full_expert_obs_np[:, env.goal_indices]
    x_min, y_min = full_positions.min(axis=0)
    x_max, y_max = full_positions.max(axis=0)
    x_margin = (x_max - x_min) * 0.2
    y_margin = (y_max - y_min) * 0.2
    x_lim = (x_min - x_margin, x_max + x_margin)
    y_lim = (y_min - y_margin, y_max + y_margin)

    # --- Heatmap (plotted first as background) ---
    x = np.linspace(x_lim[0], x_lim[1], 100)
    y = np.linspace(y_lim[0], y_lim[1], 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    covariance_matrix = np.diag(dist_variance_np)
    rv = multivariate_normal(dist_mean_np, covariance_matrix)
    Z = rv.pdf(pos)
    
    ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)

    # --- Trajectory (plotted on top) ---
    observed_positions = observed_expert_obs_np[:, env.goal_indices]

    ax.plot(full_positions[:, 0], full_positions[:, 1], color='grey', alpha=0.7, label='Full Expert Trajectory', zorder=1)
    
    num_observed_steps = observed_positions.shape[0]
    if num_observed_steps > 0:
        colors = plt.cm.Reds(np.linspace(0.3, 1, num_observed_steps))
        ax.scatter(observed_positions[:, 0], observed_positions[:, 1], c=colors, s=35, alpha=1.0, marker='^', zorder=3, label='Observed Trajectory')
    
    ax.scatter(true_goal_np[0], true_goal_np[1], marker='X', color='limegreen', s=400, label='True Goal', zorder=5, edgecolors='black')
    
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()
    ax.axis('equal')
    ax.grid(True, linestyle='--', alpha=0.3)

    textstr = '\n'.join((
        f'μ = [{dist_mean_np[0]:.2f}, {dist_mean_np[1]:.2f}]',
        f'σ = [{dist_std_np[0]:.2f}, {dist_std_np[1]:.2f}]'))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close(fig)

# --- Main Video Generation Loop ---
VIDEO_VIZ_DIR = "visualizations_posterior_videos"
os.makedirs(VIDEO_VIZ_DIR, exist_ok=True)
FRAME_STRIDE = 20
ENVS_TO_RENDER = 5 # Render for a smaller number of envs to be faster

print("\n--- Generating Posterior Videos ---")

jit_mean_field_context_encoder = jax.jit(mean_field_context_encoder)
jit_get_mean_field_distribution_params = jax.jit(get_mean_field_distribution_params)

for i in range(ENVS_TO_RENDER):
    print(f"Generating video for trajectory {i}...")
    
    frames_dir = os.path.join(VIDEO_VIZ_DIR, f"env_{i}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    full_expert_obs = original_achieved_observations_all_steps[i]
    true_g = goals[i]
    
    frame_idx = 0
    for t in range(1, NUM_STEPS + 1, FRAME_STRIDE):
        partial_states = states[i, :t]
        partial_actions = actions[i, :t]
        
        # This part needs to run on device
        sa_pairs_partial = jnp.concatenate((partial_states, partial_actions), axis=-1)
        
        mf_context_output_partial = jit_mean_field_context_encoder(sa_pairs_partial)
        mf_context_mean_partial, mf_context_log_std_partial = jnp.split(mf_context_output_partial, 2, axis=-1)
        
        dist_mean, dist_var = jit_get_mean_field_distribution_params(mf_context_mean_partial, mf_context_log_std_partial)
        
        # Plotting happens on host
        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}.png")
        observed_expert_obs_for_plot = original_achieved_observations_all_steps[i, :t]
        
        plot_posterior_frame(
            frame_path,
            full_expert_obs,
            observed_expert_obs_for_plot,
            true_g,
            dist_mean,
            dist_var,
            env,
            t
        )
        frame_idx += 1

    # After generating frames, create video
    video_path = os.path.join(VIDEO_VIZ_DIR, f"posterior_env_{i}.mp4")
    # Use -framerate for input and start_number to handle non-zero based sequences if needed.
    ffmpeg_cmd = (
        f"ffmpeg -framerate 10 -i {frames_dir}/frame_%04d.png -c:v libx264 "
        f"-crf 25 -pix_fmt yuv420p -y {video_path}"
    )
    print(f"Running ffmpeg for trajectory {i}...")
    
    # Use subprocess.run for better error handling
    result = subprocess.run(ffmpeg_cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Video saved successfully: {video_path}")
    else:
        print(f"Error running ffmpeg for trajectory {i}:")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    # Clean up frames
    shutil.rmtree(frames_dir)

print(f"Posterior videos saved in {VIDEO_VIZ_DIR}/")