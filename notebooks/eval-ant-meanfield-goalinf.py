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
from matplotlib.ticker import MaxNLocator

# Increase global font sizes to improve readability across all visuals
plt.rcParams.update({
    'font.size': 32,
    'axes.titlesize': 44,
    'axes.labelsize': 42,
    'xtick.labelsize': 38,
    'ytick.labelsize': 38,
    'legend.fontsize': 38,
    'figure.titlesize': 46
})
try:
    from palettable.colorbrewer.qualitative import Set2_7  # noqa: F401
    PALETTE_COLORS = Set2_7.mpl_colors
except Exception:
    PALETTE_COLORS = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494']

# Standardized colors aligned with create_combined_plots
COLOR_IMITATION = PALETTE_COLORS[0]      # teal
COLOR_EXPERT = PALETTE_COLORS[1]         # orange
COLOR_TRUE_GOAL = PALETTE_COLORS[4]      # green
COLOR_INFERRED_GOAL = PALETTE_COLORS[6]  # tan

# note: ant: step_11427840
# note: reacher: step_20490752
# note: simple_u_maze: step_20490752
# note: pusher_easy: step_30823424

env_name = 'ant'
# Load standard CRL checkpoint. For expert demos!
RUN_FOLDER_PATH = f'/home/kw2960/JaxGCRL/runs/run_{env_name}-main-standard-della-maxent-gaussianmlp_s_1'
CKPT_NAME = '/best.pkl'
params = model.load_params(RUN_FOLDER_PATH + '/ckpt' + CKPT_NAME)
policy_params, encoders_params, context_params = params

# # OPTION 1: CRL Mean field
# CRL Mean field checkpoint
# MEAN_FIELD_RUN_FOLDER_PATH = f'/home/kw2960/JaxGCRL/runs/run_{env_name}-main-meanfield-della-maxent-gaussianmlp_s_1'
# MEAN_FIELD_CKPT_NAME = '/best.pkl'
# mean_field_params = model.load_params(MEAN_FIELD_RUN_FOLDER_PATH + '/ckpt' + MEAN_FIELD_CKPT_NAME)
# _, _, mean_field_context_params = mean_field_params

# # OPTION 2: CRL + GoalKDE + Mean field
# # CRL + GoalKDE + Mean field checkpoint
MEAN_FIELD_RUN_FOLDER_PATH = f'/home/kw2960/JaxGCRL/runs/run_{env_name}-goalkde-meanfield-della-maxent-gaussianmlp_s_1'
MEAN_FIELD_CKPT_NAME = '/best.pkl'
mean_field_params = model.load_params(MEAN_FIELD_RUN_FOLDER_PATH + '/ckpt' + MEAN_FIELD_CKPT_NAME)
mean_field_policy_params, _, mean_field_context_params = mean_field_params

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
mean_field_inference_fn = make_policy(actor, parametric_action_distribution, mean_field_policy_params)

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
jit_mean_field_inference_fn = jax.jit(mean_field_inference_fn)
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

def collect_trajectory_with_goal(rng, goal):
    """Collect a trajectory with a specific goal."""
    def step_fn(carry, _):
        state, rng = carry
        act_rng, next_rng = jax.random.split(rng)
        act, _ = jit_inference_fn(state.obs, act_rng)
        next_state = jit_env_step(state, act)
        # Return current state's obs, action, reward, current state's pipeline_state, and next_state's obs
        return (next_state, next_rng), (state.obs, act, state.reward, state.pipeline_state, next_state.obs)
    
    # Reset environment with the specific goal
    init_state = env.reset_with_target(rng=rng, target=goal)
    # Scan over the steps
    (final_state, _), (obs_at_step, actions_at_step, rewards_at_step, pipeline_states_at_step, next_obs_at_step) = jax.lax.scan(
        step_fn, 
        (init_state, rng), 
        None, 
        length=NUM_STEPS
    )
    # For matplotlib plotting, we want the sequence of achieved observations by the policy
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
    fig.suptitle(title, fontsize=44)
    
    # --- Subplot 1: Trajectory Plot ---
    
    # Extract (x,y) positions for expert trajectory
    expert_positions_over_time = expert_trajectory_obs_np[:, env.goal_indices]

    # Plot expert trajectory's (x,y) position over time using Set2 palette color
    ax1.scatter(
        expert_positions_over_time[:, 0],
        expert_positions_over_time[:, 1],
        color=COLOR_EXPERT,
        label='Expert Trajectory',
        s=25,
        alpha=0.9,
        marker='^',
        zorder=3
    )
    ax1.plot(
        expert_positions_over_time[:, 0],
        expert_positions_over_time[:, 1],
        alpha=0.6,
        linewidth=0.8,
        color=COLOR_EXPERT,
        linestyle='--',
        zorder=2
    )
    
    # Plot true goal
    ax1.scatter(true_goal_np[0], true_goal_np[1], marker='X', color=COLOR_TRUE_GOAL, s=300, label=f'True Goal ({true_goal_np[0]:.2f}, {true_goal_np[1]:.2f})', zorder=5, edgecolors='black', linewidth=1)
    
    # Plot inferred goal
    ax1.scatter(inferred_goal_np[0], inferred_goal_np[1], marker='P', color=COLOR_INFERRED_GOAL, s=300, label=f'Inferred Goal ({inferred_goal_np[0]:.2f}, {inferred_goal_np[1]:.2f})', zorder=5, edgecolors='black', linewidth=1)
    
    ax1.set_xlabel("X Position", fontsize=42)
    ax1.set_ylabel("Y Position", fontsize=42)
    ax1.set_title("Expert Trajectory and Goals", fontsize=44)
    ax1.legend(loc='best', fontsize=34)
    ax1.xaxis.set_major_locator(MaxNLocator(3))
    ax1.yaxis.set_major_locator(MaxNLocator(3))
    ax1.axis('equal')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Removed time colorbar to maintain consistent Set2 palette usage

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

    # Plot the heatmap using a neutral, perceptually-uniform colormap to avoid clashing with Set2
    ax2.contourf(X, Y, Z, levels=20, cmap='Greys')
    ax2.scatter(true_goal_np[0], true_goal_np[1], marker='X', color=COLOR_TRUE_GOAL, s=200, label='True Goal', edgecolors='black')
    ax2.scatter(inferred_goal_np[0], inferred_goal_np[1], marker='P', color=COLOR_INFERRED_GOAL, s=200, label='Inferred Goal (Sample)', edgecolors='black')

    ax2.set_xlabel("X Position", fontsize=42)
    ax2.set_ylabel("Y Position", fontsize=42)
    ax2.set_title("Inferred Goal Distribution", fontsize=44)
    ax2.legend(loc='best', fontsize=34)
    ax2.xaxis.set_major_locator(MaxNLocator(3))
    ax2.yaxis.set_major_locator(MaxNLocator(3))
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
# Save in the current working directory (where the script is called from)
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

# Add function for imitation policy unrolling with target construction
def collect_imitation_trajectory_with_target(rng, target):
    """Collect trajectory using the mean field (CRL+GoalKDE) policy toward a target goal."""
    def step_fn(carry, _):
        state, rng = carry
        act_rng, next_rng = jax.random.split(rng)
        # Construct observation for policy: current state + commanded target
        obs_for_policy = jnp.concatenate((state.obs[:env.state_dim], target), axis=-1)
        act, _ = jit_mean_field_inference_fn(obs_for_policy, act_rng)
        next_state = jit_env_step(state, act)
        return (next_state, next_rng), next_state.obs
    
    init_state = jit_env_reset(rng=rng)
    # Scan over the steps to get observations over time
    (final_state, _), obs_over_time = jax.lax.scan(
        step_fn, 
        (init_state, rng), 
        None, 
        length=NUM_STEPS
    )
    return obs_over_time

# --- Video Generation ---

def create_summary_plot(env_idx, saved_frames_dir, frames_to_save, full_expert_obs, imitation_obs, true_g, final_dist_mean, env, fixed_x_lim, fixed_y_lim):
    """Create a summary plot with all saved frames in a row and legend at the bottom."""
    # Get the saved frame files
    saved_frame_files = []
    all_files = os.listdir(saved_frames_dir)
    print(f"  Looking for saved frames in {saved_frames_dir}")
    print(f"  Available files: {all_files}")
    
    # Get all frame files and sort them by frame index
    frame_files = []
    for filename in all_files:
        if filename.startswith("frame_") and filename.endswith('.png'):
            # Extract frame index from filename
            try:
                frame_idx = int(filename.split('_')[1])
                frame_files.append((frame_idx, os.path.join(saved_frames_dir, filename)))
            except (IndexError, ValueError):
                continue
    
    # Sort by frame index and take the first frames_to_save
    frame_files.sort(key=lambda x: x[0])
    saved_frame_files = [file_path for _, file_path in frame_files[:frames_to_save]]
    
    print(f"  Found {len(saved_frame_files)} frame files: {[os.path.basename(f) for f in saved_frame_files]}")
    
    if len(saved_frame_files) != frames_to_save:
        print(f"Warning: Expected {frames_to_save} saved frames, found {len(saved_frame_files)}")
        print(f"Found files: {saved_frame_files}")
        return
    
    # Create the summary plot with reduced spacing
    fig, axes = plt.subplots(1, frames_to_save, figsize=(5*frames_to_save, 5))
    if frames_to_save == 1:
        axes = [axes]
    
    # Reduce spacing between subplots
    plt.subplots_adjust(wspace=0.05)
    
    # Plot each saved frame
    for idx, (ax, frame_file) in enumerate(zip(axes, saved_frame_files)):
        # Load the saved frame image
        img = plt.imread(frame_file)
        ax.imshow(img)
        # Parse timestep from filename pattern: frame_XXXX_timestep_T.png
        base = os.path.basename(frame_file)
        timestep_label = None
        if '_timestep_' in base:
            try:
                timestep_str = base.split('_timestep_')[1].split('.png')[0]
                timestep_label = int(timestep_str)
            except (IndexError, ValueError):
                timestep_label = None
        if timestep_label is not None:
            ax.set_title(f"Timestep {timestep_label}", fontsize=24)
        else:
            ax.set_title("")
        ax.axis('off')
    
    # Add legend at the bottom (removed full expert trajectory)
    legend_elements = [
        plt.Line2D([0], [0], marker='^', color=COLOR_EXPERT, alpha=1.0, linewidth=0, markersize=8, label='Observed Expert Trajectory'),
        plt.Line2D([0], [0], marker='o', color=COLOR_IMITATION, alpha=1.0, linewidth=0, markersize=8, label='Imitation Policy Rollout'),
        plt.Line2D([0], [0], marker='X', color=COLOR_TRUE_GOAL, markersize=12, linewidth=0, label='True Goal'),
        plt.Line2D([0], [0], marker='P', color=COLOR_INFERRED_GOAL, markersize=12, linewidth=0, label='Inferred Goal')
    ]
    
    # Create a separate axis for the legend with reduced height
    legend_ax = fig.add_axes([0.1, 0.01, 0.8, 0.04])
    legend_ax.axis('off')
    legend = legend_ax.legend(handles=legend_elements, loc='center', ncol=4, 
                             handlelength=2.0, handleheight=1.5, labelspacing=0.3, columnspacing=0.8, fontsize=20)
    
    plt.tight_layout()
    summary_path = os.path.join(saved_frames_dir, f"summary_plot_env_{env_idx}.png")
    plt.savefig(summary_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Summary plot saved to {summary_path}")

def plot_posterior_frame(filepath, full_expert_obs, observed_expert_obs, imitation_obs, true_goal, inferred_goal, dist_mean, dist_variance, env, timestep, fixed_x_lim=None, fixed_y_lim=None, show_legend=True):
    full_expert_obs_np = np.array(full_expert_obs)
    observed_expert_obs_np = np.array(observed_expert_obs)
    imitation_obs_np = np.array(imitation_obs)
    true_goal_np = np.array(true_goal)
    inferred_goal_np = np.array(inferred_goal)
    dist_mean_np = np.array(dist_mean)
    dist_variance_np = np.array(dist_variance)
    dist_std_np = np.sqrt(dist_variance_np)

    fig, ax = plt.subplots(figsize=(12, 10))

    # --- Trajectory (plotted first to determine axis limits) ---
    full_positions = full_expert_obs_np[:, env.goal_indices]
    observed_positions = observed_expert_obs_np[:, env.goal_indices]
    imitation_positions = imitation_obs_np[:, env.goal_indices]

    # Removed full expert trajectory line per request
    
    num_observed_steps = observed_positions.shape[0]
    if num_observed_steps > 0:
        ax.scatter(
            observed_positions[:, 0], observed_positions[:, 1],
            color=COLOR_EXPERT,
            s=35,
            alpha=0.9,
            marker='^',
            zorder=3,
            label='Observed Expert Trajectory'
        )
    
    # Add imitation trajectory (policy rollout toward inferred goal)
    num_imitation_steps = imitation_positions.shape[0]
    if num_imitation_steps > 0:
        ax.scatter(
            imitation_positions[:, 0], imitation_positions[:, 1],
            color=COLOR_IMITATION,
            s=35,
            alpha=0.9,
            marker='o',
            zorder=3,
            label='Imitation Policy Rollout'
        )
        ax.plot(
            imitation_positions[:, 0], imitation_positions[:, 1],
            color=COLOR_IMITATION,
            alpha=0.7,
            linewidth=1.5,
            zorder=2
        )
    
    ax.scatter(true_goal_np[0], true_goal_np[1], marker='X', color=COLOR_TRUE_GOAL, s=400, label='True Goal', zorder=5, edgecolors='black')
    
    # Add inferred goal
    ax.scatter(inferred_goal_np[0], inferred_goal_np[1], marker='P', color=COLOR_INFERRED_GOAL, s=400, label='Inferred Goal (Full Traj)', zorder=5, edgecolors='black')
    
    # Set axis labels and properties
    ax.set_xlabel("X Position", fontsize=42)
    ax.set_ylabel("Y Position", fontsize=42)
    ax.tick_params(axis='both', labelsize=38)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    if show_legend:
        ax.legend(handlelength=3.0, handleheight=2.0, labelspacing=0.6, columnspacing=1.5, fontsize=16)
    ax.axis('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Use fixed axis limits if provided, otherwise let matplotlib calculate them
    if fixed_x_lim is not None and fixed_y_lim is not None:
        ax.set_xlim(fixed_x_lim)
        ax.set_ylim(fixed_y_lim)
        x_lim = fixed_x_lim
        y_lim = fixed_y_lim
    else:
        # Force matplotlib to calculate the axis limits
        ax.autoscale_view()
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
    
    plt.tight_layout()

    # --- Heatmap (plotted over the actual axis range) ---
    x = np.linspace(x_lim[0], x_lim[1], 100)
    y = np.linspace(y_lim[0], y_lim[1], 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    covariance_matrix = np.diag(dist_variance_np)
    rv = multivariate_normal(dist_mean_np, covariance_matrix)
    Z = rv.pdf(pos)
    
    # Plot heatmap as background (behind everything else) using a neutral grayscale
    ax.contourf(X, Y, Z, levels=20, cmap='Greys', alpha=0.6, zorder=0)

    textstr = '\n'.join((
        f'μ = [{dist_mean_np[0]:.2f}, {dist_mean_np[1]:.2f}]',
        f'σ = [{dist_std_np[0]:.2f}, {dist_std_np[1]:.2f}]'))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=38, verticalalignment='top', bbox=props)
    
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
jit_collect_imitation_trajectory_with_target = jax.jit(collect_imitation_trajectory_with_target)

for i in range(ENVS_TO_RENDER):
    print(f"Generating video for trajectory {i}...")
    
    frames_dir = os.path.join(VIDEO_VIZ_DIR, f"env_{i}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Create subdirectory for saved frames as images
    saved_frames_dir = os.path.join(VIDEO_VIZ_DIR, f"env_{i}_saved_frames")
    os.makedirs(saved_frames_dir, exist_ok=True)
    
    full_expert_obs = original_achieved_observations_all_steps[i]
    true_g = goals[i]
    
    # Get final inferred goal from the full trajectory for imitation policy rollout
    full_states = states[i]
    full_actions = actions[i]
    full_sa_pairs = jnp.concatenate((full_states, full_actions), axis=-1)
    full_mf_context_output = jit_mean_field_context_encoder(full_sa_pairs)
    full_mf_context_mean, full_mf_context_log_std = jnp.split(full_mf_context_output, 2, axis=-1)
    final_dist_mean, final_dist_var = jit_get_mean_field_distribution_params(full_mf_context_mean, full_mf_context_log_std)
    
    # Generate imitation policy trajectory toward the final inferred goal
    imitation_rng = jax.random.PRNGKey(i + 100)  # Use different seed for each environment
    imitation_obs = jit_collect_imitation_trajectory_with_target(imitation_rng, final_dist_mean)
    
    # Calculate fixed axis limits based on all data (expert + imitation + goals)
    full_expert_positions = full_expert_obs[:, env.goal_indices]
    imitation_positions = imitation_obs[:, env.goal_indices]
    
    # Get bounds from expert trajectory
    expert_x_min, expert_y_min = full_expert_positions.min(axis=0)
    expert_x_max, expert_y_max = full_expert_positions.max(axis=0)
    
    # Get bounds from imitation trajectory
    imitation_x_min, imitation_y_min = imitation_positions.min(axis=0)
    imitation_x_max, imitation_y_max = imitation_positions.max(axis=0)
    
    # Get bounds from goals
    goal_x_min = min(true_g[0], final_dist_mean[0])
    goal_x_max = max(true_g[0], final_dist_mean[0])
    goal_y_min = min(true_g[1], final_dist_mean[1])
    goal_y_max = max(true_g[1], final_dist_mean[1])
    
    # Combine all bounds
    x_min = min(expert_x_min, imitation_x_min, goal_x_min)
    x_max = max(expert_x_max, imitation_x_max, goal_x_max)
    y_min = min(expert_y_min, imitation_y_min, goal_y_min)
    y_max = max(expert_y_max, imitation_y_max, goal_y_max)
    
    # Add margin
    x_range = x_max - x_min
    y_range = y_max - y_min
    margin = max(x_range, y_range) * 0.3  # Use 30% margin based on the larger range
    fixed_x_lim = (x_min - margin, x_max + margin)
    fixed_y_lim = (y_min - margin, y_max + margin)
    
    frame_idx = 0
    # Calculate which frames to save as images (from first half of video)
    total_frames = len(range(1, NUM_STEPS + 1, FRAME_STRIDE))
    frames_to_save = 4  # Save 4 frames from first half
    if total_frames > 0:
        # Calculate frames from first half (0 to total_frames//2)
        first_half_frames = total_frames // 2
        save_indices = [int(i * (first_half_frames - 1) / (frames_to_save - 1)) for i in range(frames_to_save)]
        save_indices = [min(idx, first_half_frames - 1) for idx in save_indices]  # Ensure we don't exceed bounds
    else:
        save_indices = []
    
    print(f"  Total frames: {total_frames}, First half frames: {total_frames // 2}, Saving frames at indices: {save_indices}")
    
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
        # Get partial imitation trajectory up to current timestep
        imitation_obs_partial = imitation_obs[:t]
        
        plot_posterior_frame(
            frame_path,
            full_expert_obs,
            observed_expert_obs_for_plot,
            imitation_obs_partial,
            true_g,
            final_dist_mean,
            dist_mean,
            dist_var,
            env,
            t,
            fixed_x_lim,
            fixed_y_lim
        )
        
        # Save frame as image if it's one of the selected frames (without legend)
        if frame_idx in save_indices:
            saved_frame_path = os.path.join(saved_frames_dir, f"frame_{frame_idx:04d}_timestep_{t}.png")
            print(f"    Saving frame {frame_idx} (timestep {t}) to {saved_frame_path}")
            # Create a separate plot without legend for saved frames
            plot_posterior_frame(
                saved_frame_path,
                full_expert_obs,
                observed_expert_obs_for_plot,
                imitation_obs_partial,
                true_g,
                final_dist_mean,
                dist_mean,
                dist_var,
                env,
                t,
                fixed_x_lim,
                fixed_y_lim,
                show_legend=False
            )
            print(f"    Successfully saved frame {frame_idx}")
        
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
    
    # Create summary plot with all saved frames in a row
    print(f"Creating summary plot for environment {i}...")
    create_summary_plot(i, saved_frames_dir, frames_to_save, full_expert_obs, imitation_obs, true_g, final_dist_mean, env, fixed_x_lim, fixed_y_lim)

print(f"Posterior videos saved in {VIDEO_VIZ_DIR}/")

# --- Policy Unrolling Toward Inferred Goals ---

print("\n--- Generating Policy Unrolling Visualizations ---")

# Create a new function to collect trajectories with inferred goals
def collect_trajectory_with_inferred_goal(rng, inferred_goal):
    """Collect a trajectory with a specific inferred goal."""
    def step_fn(carry, _):
        state, rng = carry
        act_rng, next_rng = jax.random.split(rng)
        act, _ = jit_inference_fn(state.obs, act_rng)
        next_state = jit_env_step(state, act)
        return (next_state, next_rng), (state.obs, act, state.reward, state.pipeline_state, next_state.obs)
    
    # Reset environment with the inferred goal
    init_state = env.reset_with_target(rng=rng, target=inferred_goal)
    # Scan over the steps
    (final_state, _), (obs_at_step, actions_at_step, rewards_at_step, pipeline_states_at_step, next_obs_at_step) = jax.lax.scan(
        step_fn, 
        (init_state, rng), 
        None, 
        length=NUM_STEPS
    )
    # For matplotlib plotting, we want the sequence of achieved observations by the policy
    achieved_observations = jnp.concatenate([jnp.expand_dims(init_state.obs, axis=0), next_obs_at_step[:-1]], axis=0)
    return obs_at_step, actions_at_step, rewards_at_step, pipeline_states_at_step, achieved_observations



# Add function similar to eval-ant-html.py for policy unrolling with target construction
def collect_trajectory_with_target(rng, target, true_goal):
    """Collect trajectory by constructing observations with target goal (like eval-ant-html.py)."""
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

# JIT the new functions
jit_collect_trajectory_with_inferred_goal = jax.jit(collect_trajectory_with_inferred_goal)
jit_collect_trajectory_with_target = jax.jit(collect_trajectory_with_target)

# Collect trajectories with inferred goals (original method)
print("Collecting trajectories with inferred goals (original method)...")
inferred_goal_rngs = jax.random.split(jax.random.PRNGKey(1), NUM_ENVS * NUM_SAMPLES)
inferred_goal_rngs = inferred_goal_rngs.reshape(NUM_ENVS, NUM_SAMPLES, -1)

# Collect trajectories for each environment and sample
inferred_trajectories = jax.vmap(jax.vmap(jit_collect_trajectory_with_inferred_goal))(
    inferred_goal_rngs,
    mf_inferred_goals
)

# Extract achieved observations from inferred trajectories
inferred_achieved_observations = inferred_trajectories[4]  # Index 4 is achieved_observations
inferred_pipeline_states = inferred_trajectories[3]  # Index 3 is pipeline_states
print("Inferred trajectory achieved observations shape:", inferred_achieved_observations.shape)

# Collect trajectories using target construction method (like eval-ant-html.py)
print("Collecting trajectories with target construction method...")
target_rngs = jax.random.split(jax.random.PRNGKey(2), NUM_ENVS * NUM_SAMPLES)
target_rngs = target_rngs.reshape(NUM_ENVS, NUM_SAMPLES, -1)

target_trajectories = jax.vmap(
    jax.vmap(collect_trajectory_with_target, in_axes=(0, 0, None)),
    in_axes=(0, 0, 0)
)(
    target_rngs,
    mf_inferred_goals,
    goals
)

target_rewards, target_observations = target_trajectories
print("Target trajectory observations shape:", target_observations.shape)

# Calculate distances to inferred goals for both original and inferred trajectories
def calculate_goal_distances(trajectories, goals):
    """Calculate distances from trajectory positions to goals over time."""
    positions = trajectories[:, :, env.goal_indices]  # Shape: [num_envs, num_steps, 2]
    goal_distances = jnp.linalg.norm(positions - goals[:, None, :], axis=-1)  # Shape: [num_envs, num_steps]
    return goal_distances

# Calculate distances for original trajectories to true goals
original_distances_to_true_goals = calculate_goal_distances(original_achieved_observations_all_steps, goals)

# Calculate distances for inferred trajectories to inferred goals
# Squeeze out the NUM_SAMPLES dimension since we only have 1 sample
inferred_achieved_observations_squeezed = jnp.squeeze(inferred_achieved_observations, axis=1)  # Shape: [num_envs, num_steps, obs_dim]
inferred_distances_to_inferred_goals = calculate_goal_distances(inferred_achieved_observations_squeezed, jnp.squeeze(mf_inferred_goals, axis=1))

# Calculate distances for inferred trajectories to true goals (to see if they're actually reaching the true goal)
inferred_distances_to_true_goals = calculate_goal_distances(inferred_achieved_observations_squeezed, goals)

# Calculate distances for target trajectories to true goals
target_observations_squeezed = jnp.squeeze(target_observations, axis=1)  # Shape: [num_envs, num_steps, obs_dim]
target_distances_to_true_goals = calculate_goal_distances(target_observations_squeezed, goals)

print("Original trajectories - mean final distance to true goal:", jnp.mean(original_distances_to_true_goals[:, -1]))
print("Inferred trajectories - mean final distance to inferred goal:", jnp.mean(inferred_distances_to_inferred_goals[:, -1]))
print("Inferred trajectories - mean final distance to true goal:", jnp.mean(inferred_distances_to_true_goals[:, -1]))
print("Target trajectories - mean final distance to true goal:", jnp.mean(target_distances_to_true_goals[:, -1]))

# Create comparison visualization function
def plot_trajectory_comparison(filepath, original_traj, inferred_traj, true_goal, inferred_goal, 
                             original_distances, inferred_distances, title, env):
    """Plot comparison between original and inferred goal trajectories."""
    original_traj_np = np.array(original_traj)
    inferred_traj_np = np.array(inferred_traj)
    true_goal_np = np.array(true_goal)
    inferred_goal_np = np.array(inferred_goal)
    original_distances_np = np.array(original_distances)
    inferred_distances_np = np.array(inferred_distances)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(title, fontsize=16)
    
    # --- Subplot 1: Trajectory Comparison ---
    original_positions = original_traj_np[:, env.goal_indices]
    inferred_positions = inferred_traj_np[:, env.goal_indices]
    
    # Plot original trajectory (Observed Expert) with Set2 color
    ax1.plot(original_positions[:, 0], original_positions[:, 1], color=COLOR_EXPERT, linewidth=2, label='Original Trajectory', alpha=0.8)
    ax1.scatter(original_positions[:, 0], original_positions[:, 1], 
                color=COLOR_EXPERT, s=30, alpha=0.9, marker='o', zorder=3)
    
    # Plot inferred trajectory (Imitation rollout) with Set2 color
    ax1.plot(inferred_positions[:, 0], inferred_positions[:, 1], color=COLOR_IMITATION, linewidth=2, label='Inferred Goal Trajectory', alpha=0.8)
    ax1.scatter(inferred_positions[:, 0], inferred_positions[:, 1], 
                color=COLOR_IMITATION, s=30, alpha=0.9, marker='s', zorder=3)
    
    # Plot goals
    ax1.scatter(true_goal_np[0], true_goal_np[1], marker='X', color=COLOR_TRUE_GOAL, s=400, 
                label=f'True Goal ({true_goal_np[0]:.2f}, {true_goal_np[1]:.2f})', zorder=5, edgecolors='black')
    ax1.scatter(inferred_goal_np[0], inferred_goal_np[1], marker='P', color=COLOR_INFERRED_GOAL, s=400, 
                label=f'Inferred Goal ({inferred_goal_np[0]:.2f}, {inferred_goal_np[1]:.2f})', zorder=5, edgecolors='black')
    
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Y Position")
    ax1.set_title("Trajectory Comparison")
    ax1.legend()
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # --- Subplot 2: Distance to Goals Over Time ---
    timesteps = np.arange(len(original_distances_np))
    ax2.plot(timesteps, original_distances_np, 'b-', linewidth=2, label='Original to True Goal')
    ax2.plot(timesteps, inferred_distances_np, 'r-', linewidth=2, label='Inferred to Inferred Goal')
    
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Distance to Goal")
    ax2.set_title("Distance to Goals Over Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # --- Subplot 3: Final Positions Heatmap ---
    # Create a grid around the goals
    x_min = min(original_positions[:, 0].min(), inferred_positions[:, 0].min(), true_goal_np[0], inferred_goal_np[0]) - 1
    x_max = max(original_positions[:, 0].max(), inferred_positions[:, 0].max(), true_goal_np[0], inferred_goal_np[0]) + 1
    y_min = min(original_positions[:, 1].min(), inferred_positions[:, 1].min(), true_goal_np[1], inferred_goal_np[1]) - 1
    y_max = max(original_positions[:, 1].max(), inferred_positions[:, 1].max(), true_goal_np[1], inferred_goal_np[1]) + 1
    
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    
    # Create heatmap based on distance to true goal
    Z = np.sqrt((X - true_goal_np[0])**2 + (Y - true_goal_np[1])**2)
    
    ax3.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    ax3.scatter(original_positions[-1, 0], original_positions[-1, 1], marker='o', color='blue', s=200, 
                label='Original Final Position', edgecolors='black', zorder=5)
    ax3.scatter(inferred_positions[-1, 0], inferred_positions[-1, 1], marker='s', color='red', s=200, 
                label='Inferred Final Position', edgecolors='black', zorder=5)
    ax3.scatter(true_goal_np[0], true_goal_np[1], marker='X', color='limegreen', s=300, 
                label='True Goal', edgecolors='black', zorder=5)
    ax3.scatter(inferred_goal_np[0], inferred_goal_np[1], marker='P', color='gold', s=300, 
                label='Inferred Goal', edgecolors='black', zorder=5)
    
    ax3.set_xlabel("X Position")
    ax3.set_ylabel("Y Position")
    ax3.set_title("Final Positions (Distance Heatmap)")
    ax3.legend()
    ax3.axis('equal')
    ax3.grid(True, alpha=0.3)
    
    # --- Subplot 4: Success Metrics ---
    metrics_data = {
        'Original to True Goal': original_distances_np[-1],
        'Inferred to Inferred Goal': inferred_distances_np[-1],
        'Inferred to True Goal': inferred_distances_to_true_goals[0, -1] if len(inferred_distances_to_true_goals.shape) > 1 else inferred_distances_to_true_goals[-1]
    }
    
    metrics_names = list(metrics_data.keys())
    metrics_values = list(metrics_data.values())
    colors = [COLOR_EXPERT, COLOR_IMITATION, PALETTE_COLORS[1]]
    
    bars = ax4.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
    ax4.set_ylabel("Final Distance")
    ax4.set_title("Final Distance Metrics")
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close(fig)

# Create enhanced policy unrolling visualization function (like eval-ant-html.py)
def plot_policy_unrolling_comparison(filepath, original_traj, target_traj, true_goal, inferred_goal, title, env):
    """Plot comparison between original trajectory and policy unrolling toward inferred goal."""
    original_traj_np = np.array(original_traj)
    target_traj_np = np.array(target_traj)
    true_goal_np = np.array(true_goal)
    inferred_goal_np = np.array(inferred_goal)

    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Extract (x,y) positions for target trajectory (policy unrolling)
    target_positions_over_time = target_traj_np[:, env.goal_indices]
    num_steps_target = target_positions_over_time.shape[0]
    # Use 'Blues' colormap for target policy rollout
    colors_target = plt.cm.Blues(np.linspace(0.3, 1, num_steps_target))
    
    # Plot target trajectory's (x,y) position over time
    scatter_target = ax.scatter(target_positions_over_time[:, 0], target_positions_over_time[:, 1], 
                               c=colors_target, label='Policy Unrolling Toward Inferred Goal', s=25, alpha=0.8, zorder=3, cmap='Blues')
    ax.plot(target_positions_over_time[:, 0], target_positions_over_time[:, 1], 
            alpha=0.6, linewidth=0.8, color=colors_target[0] if num_steps_target > 0 else 'blue', zorder=2)

    # Extract (x,y) positions for original/expert trajectory
    original_positions_over_time = original_traj_np[:, env.goal_indices]
    num_steps_original = original_positions_over_time.shape[0]
    # Use 'Reds' colormap for original demonstrated trajectory
    colors_original = plt.cm.Reds(np.linspace(0.3, 1, num_steps_original))

    # Plot original/expert trajectory's (x,y) position over time
    scatter_original = ax.scatter(original_positions_over_time[:, 0], original_positions_over_time[:, 1], 
                                 c=colors_original, label='Original Demonstrated Trajectory', s=25, alpha=0.8, marker='^', zorder=3, cmap='Reds')
    ax.plot(original_positions_over_time[:, 0], original_positions_over_time[:, 1], 
            alpha=0.6, linewidth=0.8, color=colors_original[0] if num_steps_original > 0 else 'red', linestyle='--', zorder=2)
    
    # Plot true goal
    ax.scatter(true_goal_np[0], true_goal_np[1], marker='X', color='limegreen', s=300, 
               label=f'True Goal ({true_goal_np[0]:.2f}, {true_goal_np[1]:.2f})', zorder=5, edgecolors='black', linewidth=1)
    
    # Plot inferred goal
    ax.scatter(inferred_goal_np[0], inferred_goal_np[1], marker='P', color='gold', s=300, 
               label=f'Inferred Goal ({inferred_goal_np[0]:.2f}, {inferred_goal_np[1]:.2f})', zorder=5, edgecolors='black', linewidth=1)
    
    ax.set_xlabel("X Position", fontsize=12)
    ax.set_ylabel("Y Position", fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=10)
    ax.axis('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a colorbar for target trajectory
    cbar_target = fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=1)), 
                              ax=ax, fraction=0.040, pad=0.04)
    cbar_target.set_ticks([0, 0.5, 1])
    cbar_target.set_ticklabels(['Start', 'Mid', 'End'])
    cbar_target.set_label('Policy Unrolling Time', labelpad=-40, fontsize=9)

    # Add a colorbar for original trajectory
    cbar_original = fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1)), 
                                ax=ax, fraction=0.040, pad=0.12)
    cbar_original.set_ticks([0, 0.5, 1])
    cbar_original.set_ticklabels(['Start', 'Mid', 'End'])
    cbar_original.set_label('Original Trajectory Time', labelpad=-40, fontsize=9)

    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)

# Create visualization directories
unrolling_viz_dir = "visualizations_policy_unrolling"
enhanced_unrolling_viz_dir = "visualizations_enhanced_policy_unrolling"
html_viz_dir = "visualizations_html"

for d in [unrolling_viz_dir, enhanced_unrolling_viz_dir, html_viz_dir]:
    os.makedirs(d, exist_ok=True)

# Helper function to convert pipeline states to list (for HTML visualization)
def pipeline_states_to_list(pipeline_states):
    return [jax.tree_util.tree_map(lambda x: x[i], pipeline_states) for i in range(pipeline_states.x.pos.shape[0])]

# Generate original comparison visualizations
print("Generating policy unrolling comparison visualizations...")
for i in range(NUM_ENVS):
    for j in range(NUM_SAMPLES):
        original_traj = original_achieved_observations_all_steps[i]
        inferred_traj = inferred_achieved_observations_squeezed[i]
        true_g = goals[i]
        inferred_g = mf_inferred_goals[i, j]
        original_distances = original_distances_to_true_goals[i]
        inferred_distances = inferred_distances_to_inferred_goals[i]
        
        filepath = os.path.join(unrolling_viz_dir, f"trajectory_comparison_env_{i}_sample_{j}.png")
        title = f"Policy Unrolling Comparison - Env {i}, Sample {j}"
        
        plot_trajectory_comparison(
            filepath, original_traj, inferred_traj, true_g, inferred_g,
            original_distances, inferred_distances, title, env
        )

# Generate enhanced policy unrolling visualizations (like eval-ant-html.py)
print("Generating enhanced policy unrolling visualizations...")
for i in range(NUM_ENVS):
    for j in range(NUM_SAMPLES):
        original_traj = original_achieved_observations_all_steps[i]
        target_traj = target_observations_squeezed[i]
        true_g = goals[i]
        inferred_g = mf_inferred_goals[i, j]
        
        filepath = os.path.join(enhanced_unrolling_viz_dir, f"policy_unrolling_env_{i}_sample_{j}.png")
        title = f"Policy Unrolling Toward Inferred Goal - Env {i}, Sample {j}"
        
        plot_policy_unrolling_comparison(
            filepath, original_traj, target_traj, true_g, inferred_g, title, env
        )

# Generate HTML visualizations for original trajectories
print("Rendering original trajectories (HTML)...")
for i in range(NUM_ENVS):
    env_states = pipeline_states_to_list(jax.tree_util.tree_map(lambda x: x[i], pipeline_states))
    html.save(
        os.path.join(html_viz_dir, f"original_trajectory_{i}.html"),
        env.sys.tree_replace({'opt.timestep': env.dt}),
        env_states
    )

# Generate HTML visualizations for inferred goal trajectories
print("Rendering inferred goal trajectories (HTML)...")
for i in range(NUM_ENVS):
    for j in range(NUM_SAMPLES):
        env_states = pipeline_states_to_list(jax.tree_util.tree_map(lambda x: x[i, 0], inferred_pipeline_states))
        html.save(
            os.path.join(html_viz_dir, f"inferred_goal_trajectory_{i}_sample_{j}.html"),
            env.sys.tree_replace({'opt.timestep': env.dt}),
            env_states
        )

print(f"Policy unrolling visualizations saved in {unrolling_viz_dir}/")
print(f"Enhanced policy unrolling visualizations saved in {enhanced_unrolling_viz_dir}/")
print(f"HTML visualizations saved in {html_viz_dir}/")

# Create summary statistics
print("\n--- Summary Statistics ---")
print(f"Original trajectories - mean final distance to true goal: {jnp.mean(original_distances_to_true_goals[:, -1]):.4f}")
print(f"Inferred trajectories - mean final distance to inferred goal: {jnp.mean(inferred_distances_to_inferred_goals[:, -1]):.4f}")
print(f"Inferred trajectories - mean final distance to true goal: {jnp.mean(inferred_distances_to_true_goals[:, -1]):.4f}")
print(f"Target trajectories - mean final distance to true goal: {jnp.mean(target_distances_to_true_goals[:, -1]):.4f}")

# Calculate success rates (within 0.5 distance threshold)
success_threshold = 0.5
original_success_rate = jnp.mean(original_distances_to_true_goals[:, -1] < success_threshold)
inferred_success_rate = jnp.mean(inferred_distances_to_inferred_goals[:, -1] < success_threshold)
inferred_to_true_success_rate = jnp.mean(inferred_distances_to_true_goals[:, -1] < success_threshold)
target_success_rate = jnp.mean(target_distances_to_true_goals[:, -1] < success_threshold)

print(f"Original trajectories - success rate (distance < {success_threshold}): {original_success_rate:.3f}")
print(f"Inferred trajectories - success rate to inferred goal (distance < {success_threshold}): {inferred_success_rate:.3f}")
print(f"Inferred trajectories - success rate to true goal (distance < {success_threshold}): {inferred_to_true_success_rate:.3f}")
print(f"Target trajectories - success rate to true goal (distance < {success_threshold}): {target_success_rate:.3f}")

print("\nPolicy unrolling visualization complete!")