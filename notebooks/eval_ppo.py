import argparse
import os
import sys
import pickle
import numpy as np
import jax
import jax.numpy as jp
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.ant_base import AntJump
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
from brax import math
import flax.linen as nn
from brax.training import distribution


def quat_to_3x3(quat):
    """Convert quaternion to 3x3 rotation matrix."""
    # Add safeguards against invalid quaternions using JAX operations
    quat_norm = jp.linalg.norm(quat)
    
    # Use jp.where for conditional operations
    quat_normalized = jp.where(
        quat_norm < 1e-6,
        jp.array([1.0, 0.0, 0.0, 0.0]),  # Unit quaternion
        quat / quat_norm
    )
    
    return math.quat_to_3x3(quat_normalized)


def get_pos_quat_from_obs(obs: jp.ndarray, env) -> tuple[jp.ndarray, jp.ndarray]:
    """Returns (pos[3], quat[4]) from obs for Ant variants.

    Heuristics:
    - For full-obs/posvel style envs (has goal_indices or looks like state_dim 29), assume
      obs layout starts with qpos: [x, y, z, qw, qx, qy, qz, ...].
    - Fallback: same assumption; Brax Ant commonly exposes qpos first.
    """
    # Prefer explicit full-obs signal
    if hasattr(env, 'goal_indices') or getattr(env, 'state_dim', None) in (29,):
        pos = obs[0:3]
        quat = obs[3:7]
        return pos, quat
    # Fallback: use first 7 entries as pose
    pos = obs[0:3]
    quat = obs[3:7]
    return pos, quat


def calculate_antforward_reward(obs, action, min_forward_velocity=0.5, ctrl_cost_weight=0.5, healthy_reward=1.0, healthy_z_range=(0.2, 2.5)):
    """Calculate AntForward reward from observations."""
    # Extract torso position and velocity from observations
    # obs[0] = z-coordinate of torso (height)
    # obs[1:5] = torso orientation quaternion [w, x, y, z]
    # obs[13:16] = torso velocity [vx, vy, vz]
    
    # Transform to local frame using torso orientation
    torso_quat = obs[1:5]  # [w, x, y, z] quaternion
    torso_vel = obs[13:16]  # [vx, vy, vz] world velocity
    torso_rot = quat_to_3x3(torso_quat)
    local_velocity = torso_rot.T @ torso_vel
    
    # Calculate forward reward
    local_x_velocity = local_velocity[0]
    # The environment uses the opposite sign for local x velocity
    forward_reward = jp.where(-local_x_velocity > min_forward_velocity, -local_x_velocity, 0.0)
    
    # Calculate healthy reward - match environment behavior
    # The environment seems to always give healthy reward regardless of health status
    # when terminate_when_unhealthy=True (which is the default)
    healthy_reward_component = healthy_reward
    
    # Control cost
    ctrl_cost = ctrl_cost_weight * jp.sum(jp.square(action))
    
    return forward_reward + healthy_reward_component - ctrl_cost


def calculate_antforward_reward_exact(obs, next_obs, action, dt, min_forward_velocity=0.5, ctrl_cost_weight=0.5, healthy_reward=1.0, env=None):
    """Reproduce AntForward-style reward using (s, a, s') with pose-based velocity.

    Generalized to AntForward and AntFullObs/AntPosVel: always compute world velocity
    as pos_after - pos_before over dt, then rotate by R(q_after) to local frame.
    """
    pos_before, _ = get_pos_quat_from_obs(obs, env)
    pos_after, torso_quat_after = get_pos_quat_from_obs(next_obs, env)
    world_velocity = (pos_after - pos_before) / dt

    torso_rot_after = quat_to_3x3(torso_quat_after)
    local_velocity = torso_rot_after.T @ world_velocity
    local_x_velocity = local_velocity[0]

    forward_reward = jp.where(local_x_velocity > min_forward_velocity, local_x_velocity, 0.0)
    ctrl_cost = ctrl_cost_weight * jp.sum(jp.square(action))
    return forward_reward + healthy_reward - ctrl_cost


def calculate_antjump_reward(obs, action, target_height=1.0, ctrl_cost_weight=0.5, healthy_reward=1.0, healthy_z_range=(0.2, 2.5)):
    """Calculate AntJump reward from observations."""
    # Extract torso height from observations
    z_position = obs[0]  # Torso z-coordinate (height)
    
    # Calculate jump reward
    jump_reward = jp.where(z_position > target_height, z_position, 0.0)
    
    # Calculate healthy reward - always give healthy reward like environment
    healthy_reward_component = healthy_reward
    
    # Control cost
    ctrl_cost = ctrl_cost_weight * jp.sum(jp.square(action))
    
    return jump_reward + healthy_reward_component - ctrl_cost


def calculate_antjump_reward_exact(obs, next_obs, action, dt, target_height=1.0, ctrl_cost_weight=0.5, healthy_reward=1.0):
    """Reproduce AntJump env reward exactly using (s, a, s')."""
    z_after = next_obs[2]
    jump_reward = jp.where(z_after > target_height, z_after, 0.0)
    healthy = healthy_reward
    ctrl_cost = ctrl_cost_weight * jp.sum(jp.square(action))
    return jump_reward + healthy - ctrl_cost


def calculate_antflip_reward(obs, action, min_flip_velocity=1.0, ctrl_cost_weight=0.5):
    """Calculate AntFlip reward from observations."""
    # Extract torso position and orientation
    torso_height = obs[0]  # z-coordinate
    torso_quat = obs[1:5]  # quaternion
    torso_ang_vel = obs[16:19]  # angular velocity
    
    # Calculate stand reward
    torso_rot = quat_to_3x3(torso_quat)
    upright = torso_rot[2, 2]  # Projection of torso z-axis on world z-axis
    # The environment uses the opposite sign for upright calculation
    upright = -upright
    
    standing = jp.where(torso_height > 0.7, 1.0, 0.0)  # stand_height = 0.7
    stand_reward = (standing * 3 + upright) / 4
    
    # Calculate flip reward
    flip_speed_y = torso_ang_vel[1]  # Angular velocity around y-axis
    move_reward = jp.clip(flip_speed_y / min_flip_velocity, 0.0, 1.0)
    
    # Control cost
    ctrl_cost = ctrl_cost_weight * jp.sum(jp.square(action))
    
    # AntFlip does NOT include healthy reward in the final reward calculation!
    # The environment only uses: stand_reward * (5 * move_reward + 1) / 6 - ctrl_cost
    return stand_reward * (5 * move_reward + 1) / 6 - ctrl_cost


def calculate_antflip_reward_exact(obs, next_obs, action, dt, min_flip_velocity=1.0, ctrl_cost_weight=0.5):
    """Reproduce AntFlip env reward exactly using (s, a, s').

    Uses AFTER-step quantities for height, orientation, and angular velocity.
    Matches env: upright = R(q)[2,2]; flip_speed from qd[4] (y axis) as in obs[18:21].
    """
    torso_height = next_obs[2]
    torso_quat = next_obs[3:7]
    torso_rot = quat_to_3x3(torso_quat)
    torso_ang_vel = next_obs[18:21]

    upright = torso_rot[2, 2]

    standing = jp.where(torso_height > 0.5, 1.0, 0.0)
    stand_reward = (standing * 3 + upright) / 4

    flip_speed_y = torso_ang_vel[1]
    move_reward = jp.clip(flip_speed_y / min_flip_velocity, 0.0, 1.0)
    ctrl_cost = ctrl_cost_weight * jp.sum(jp.square(action))
    return stand_reward * (5 * move_reward + 1) / 6 - ctrl_cost


def get_reward_function(env_type, **kwargs):
    """Get the appropriate reward function based on environment type."""
    ctrl_cost_weight = kwargs.get('ctrl_cost_weight', 0.5)
    healthy_reward = kwargs.get('healthy_reward', 1.0)
    
    if 'antforward' in env_type:
        min_forward_velocity = kwargs.get('min_forward_velocity', 0.5)
        dt = kwargs['dt']
        return lambda obs, next_obs, action, env=None: calculate_antforward_reward_exact(
            obs, next_obs, action, dt, min_forward_velocity, ctrl_cost_weight, healthy_reward, env=env
        )
    elif 'antjump' in env_type:
        target_height = kwargs.get('target_height', 1.0)
        dt = kwargs['dt']
        return lambda obs, next_obs, action, env=None: calculate_antjump_reward_exact(
            obs, next_obs, action, dt, target_height, ctrl_cost_weight, healthy_reward
        )
    elif 'antflip' in env_type:
        min_flip_velocity = kwargs.get('min_flip_velocity', 1.0)
        dt = kwargs['dt']
        return lambda obs, next_obs, action, env=None: calculate_antflip_reward_exact(
            obs, next_obs, action, dt, min_flip_velocity, ctrl_cost_weight
        )
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


def find_latest_model(env_name):
    """Find the latest model file for a given environment."""
    # Try multiple possible locations
    patterns = [
        f"simple_ppo/{env_name}/ppo_{env_name}_*_model.pkl",
        f"notebooks/simple_ppo/{env_name}/ppo_{env_name}_*_model.pkl",
        f"notebooks/ppo_{env_name}_*_model.pkl"
    ]
    
    model_files = []
    for pattern in patterns:
        model_files.extend(glob.glob(pattern))
    
    if not model_files:
        raise FileNotFoundError(f"No model files found for {env_name} in any of the expected locations")
    
    # Return the most recently modified file
    latest_model = max(model_files, key=os.path.getmtime)
    print(f"Found latest model: {latest_model}")
    return latest_model


def load_policy(model_path, env):
    """Load a trained PPO policy from a saved model file."""
    # Load the trained parameters
    params = model.load_params(model_path)
    
    # Create the network factory (same as used in training)
    network_factory = ppo_networks.make_ppo_networks
    
    # Create the PPO networks
    ppo_network = network_factory(env.observation_size, env.action_size)
    
    # Create the policy function using make_inference_fn
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    policy = make_policy(params, deterministic=True)
    
    return policy


def collect_trajectory(env, policy, episode_length=1000, seed=0, reward_function=None):
    """Collect a single trajectory of state-action pairs."""
    rng = jax.random.PRNGKey(seed)
    
    # JIT the environment and policy functions for efficiency
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_policy = jax.jit(policy)
    
    # JIT the reward function (always provided)
    jit_reward_function = jax.jit(reward_function)
    
    # Reset environment
    rng, reset_rng = jax.random.split(rng)
    state = jit_env_reset(rng=reset_rng)
    
    # Initialize trajectory storage
    observations = []
    actions = []
    rewards = []
    env_rewards = []
    custom_rewards = []
    dones = []
    infos = []
    
    # Collect trajectory
    for step in range(episode_length):
        # Get action from policy
        act_rng, rng = jax.random.split(rng)
        action, _ = jit_policy(state.obs, act_rng)
        
        # Store current state and action
        observations.append(np.array(state.obs))
        actions.append(np.array(action))
        
        # Step environment
        next_state = jit_env_step(state, action)
        
        # Calculate rewards on the SAME transition (s, a, s')
        # - env reward comes from next_state.reward
        # - custom reward uses (obs_before, obs_after, action) to reproduce env exactly
        env_rew = float(next_state.reward)
        custom_rew = float(jit_reward_function(state.obs, next_state.obs, action))
        
        # Primary rewards vector: always custom
        rewards.append(float(custom_rew))
        
        # Store both for reference
        env_rewards.append(env_rew)
        custom_rewards.append(float(custom_rew))
        
        state = next_state
        
        dones.append(bool(state.done))
        infos.append(dict(state.metrics))
        
        # Check if episode is done
        if state.done.all():
            break
    
    # Convert to numpy arrays
    trajectory = {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'env_rewards': np.array(env_rewards),
        'custom_rewards': np.array(custom_rewards),
        'dones': np.array(dones),
        'infos': infos
    }
    
    return trajectory


def extract_torso_positions(trajectory):
    """Extract xyz positions of the torso from a trajectory."""
    # The first 3 elements of the observation are typically the torso position
    # For ant environments: [z, w, x, y, ...] where z is height, x,y are horizontal
    observations = trajectory['observations']
    
    # Extract torso positions (first 3 elements: z, x, y)
    # Note: ant observations are [z, w, x, y, ...] where w is quaternion
    torso_positions = []
    for obs in observations:
        # For ant environments, the torso position is in the first few elements
        # Format: [z, w, x, y, ...] where z is height
        z = obs[0]  # Height
        x = obs[2]  # X position (after quaternion w)
        y = obs[3]  # Y position
        torso_positions.append([x, y, z])
    
    return np.array(torso_positions)


def plot_torso_trajectory_3d(trajectory, title="Torso Trajectory", save_path=None):
    """Create a 3D plot of the torso trajectory."""
    torso_positions = extract_torso_positions(trajectory)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the trajectory
    x, y, z = torso_positions[:, 0], torso_positions[:, 1], torso_positions[:, 2]
    ax.plot(x, y, z, 'b-', linewidth=2, label='Torso Path')
    
    # Mark start and end points
    ax.scatter(x[0], y[0], z[0], c='green', s=100, marker='o', label='Start')
    ax.scatter(x[-1], y[-1], z[-1], c='red', s=100, marker='s', label='End')
    
    # Add intermediate points for better visualization
    if len(x) > 10:
        step = len(x) // 10
        ax.scatter(x[::step], y[::step], z[::step], c='orange', s=50, alpha=0.7)
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title(title)
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D trajectory plot saved to: {save_path}")
    
    plt.show()


def plot_multiple_trajectories(trajectories, title="Multiple Torso Trajectories", save_path=None):
    """Create a 3D plot showing multiple trajectories."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, trajectory in enumerate(trajectories):
        torso_positions = extract_torso_positions(trajectory)
        x, y, z = torso_positions[:, 0], torso_positions[:, 1], torso_positions[:, 2]
        
        color = colors[i % len(colors)]
        ax.plot(x, y, z, color=color, linewidth=2, label=f'Episode {i+1}')
        
        # Mark start and end points
        ax.scatter(x[0], y[0], z[0], c=color, s=50, marker='o')
        ax.scatter(x[-1], y[-1], z[-1], c=color, s=50, marker='s')
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title(title)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multiple trajectories plot saved to: {save_path}")
    
    plt.show()


def main(args):
    """Main evaluation function."""
    # Auto-find model if env is specified
    if args.env:
        model_path = find_latest_model(args.env)
    else:
        model_path = args.model_path
    
    print(f"Loading model from: {model_path}")
    
    # Create environment (same as used during training)
    env_params = {}
    if 'antjump' in model_path:
        # Extract target jump height from filename if possible
        target_height = 1.0  # default
        if 'h1.0' in model_path:
            target_height = 1.0
        elif 'h1.3' in model_path:
            target_height = 1.3
        # Add more cases as needed
        
        env = AntJump(target_jump_height=target_height)
        env_params['target_height'] = target_height
        print(f"Created AntJump environment with target height: {target_height}")
    elif 'antforward' in model_path:
        # Extract forward velocity from filename if possible
        min_forward_velocity = 0.5  # default
        if 'vel0.5' in model_path:
            min_forward_velocity = 0.5
        elif 'vel1.0' in model_path:
            min_forward_velocity = 1.0
        # Add more cases as needed
        
        from envs.ant_base import AntForward
        env = AntForward(min_forward_velocity=min_forward_velocity)
        env_params['min_forward_velocity'] = min_forward_velocity
        print(f"Created AntForward environment with min_forward_velocity: {min_forward_velocity}")
    elif 'antflip' in model_path:
        # Extract flip velocity from filename if possible
        min_flip_velocity = 1.0  # default
        if 'flipvel1.0' in model_path:
            min_flip_velocity = 1.0
        elif 'flipvel2.0' in model_path:
            min_flip_velocity = 2.0
        # Add more cases as needed
        
        from envs.ant_base import AntFlip
        env = AntFlip(min_flip_velocity=min_flip_velocity)
        env_params['min_flip_velocity'] = min_flip_velocity
        print(f"Created AntFlip environment with min_flip_velocity: {min_flip_velocity}")
    else:
        raise ValueError(f"Unsupported environment type in model path: {model_path}")
    
    # Always use custom reward calculations; read parameters directly from env
    if 'antjump' in model_path:
        env_params['target_height'] = getattr(env, '_target_jump_height', args.target_height)
    elif 'antforward' in model_path:
        env_params['min_forward_velocity'] = getattr(env, '_min_forward_velocity', args.min_forward_velocity)
    elif 'antflip' in model_path:
        # AntFlip stores spin speed as the threshold parameter
        env_params['min_flip_velocity'] = getattr(env, '_spin_speed', args.min_flip_velocity)

    # Control weight and healthy reward
    env_params['ctrl_cost_weight'] = getattr(env, '_ctrl_cost_weight', args.ctrl_cost_weight)
    env_params['healthy_reward'] = getattr(env, '_healthy_reward', args.healthy_reward)

    # Time step for velocity discretization
    env_params['dt'] = getattr(env, 'dt', 0.05)

    reward_function = get_reward_function(model_path, **env_params)
    print("Using custom reward calculations")
    print(f"Reward parameters: {env_params}")
    
    # Load the trained policy
    policy = load_policy(model_path, env)
    print("Policy loaded successfully")
    
    # # Collect trajectories
    # all_trajectories = []
    # total_rewards = []
    
    # for episode in range(args.num_episodes):
    #     print(f"Collecting episode {episode + 1}/{args.num_episodes}")
        
    #     trajectory = collect_trajectory(
    #         env, 
    #         policy, 
    #         episode_length=args.episode_length,
    #         seed=args.seed + episode,
    #         reward_function=reward_function
    #     )
        
    #     all_trajectories.append(trajectory)
    #     total_reward = np.sum(trajectory['rewards'])
    #     total_rewards.append(total_reward)
        
    #     print(f"  Episode {episode + 1} total reward: {total_reward:.2f}")
    #     print(f"  Episode {episode + 1} length: {len(trajectory['observations'])}")
        
    #     # Print env vs custom sums for reference
    #     if 'env_rewards' in trajectory:
    #         env_sum = float(np.nansum(trajectory['env_rewards']))
    #         custom_sum = float(np.nansum(trajectory['custom_rewards']))
    #         diff_sum = custom_sum - env_sum
    #         print(f"    Env reward sum:    {env_sum:.4f}")
    #         print(f"    Custom reward sum: {custom_sum:.4f}")
    #         print(f"    Difference (cust - env): {diff_sum:.4f}")
    
    # # Print summary statistics
    # print("\n" + "="*50)
    # print("EVALUATION SUMMARY")
    # print("="*50)
    # print(f"Number of episodes: {args.num_episodes}")
    # print(f"Mean total reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    # print(f"Min total reward: {np.min(total_rewards):.2f}")
    # print(f"Max total reward: {np.max(total_rewards):.2f}")
    # print(f"Mean episode length: {np.mean([len(t['observations']) for t in all_trajectories]):.1f}")
    
    # print(f"Reward calculation: Custom reward functions")
    # print(f"Reward parameters: {env_params}")

    # ================== GoalKDE comparison ==================
    print("\nRunning GoalKDE goal inference and imitation rollout...")
    # Policy/action dims come from PPO env; GoalKDE uses fullobs (state_dim=29, goal_dim inferred)
    obs_dim = env.observation_size
    act_dim = env.action_size
    state_dim = 29  # Ant state dimension per requirement

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
            residual_stream = jp.zeros((x.shape[0], self.width))
            for i in range(self.num_blocks):
                for j in range(self.block_size):
                    x = nn.swish(normalize(nn.Dense(self.width, kernel_init=lecun_uniform)(x)))
                x = x + residual_stream
                residual_stream = x
            x = nn.Dense(self.output_size, kernel_init=lecun_uniform)(x)
            return x

    parametric_action_distribution = distribution.NormalTanhDistribution(event_size=act_dim)

    def make_policy(actor, params, deterministic=True):
        def policy_fn(obs, key_sample):
            obs = jp.expand_dims(obs, 0)
            logits = actor.apply(params, obs)
            if deterministic:
                action = parametric_action_distribution.mode(logits)
            else:
                action = parametric_action_distribution.sample(logits, key_sample)
            action = action[0]
            return action, {}
        return policy_fn

    # Load GoalKDE checkpoint
    # goalkde_dir = '/scratch/gpfs/kw2960/JaxGCRL/runs/run_ant_fullobs-goalkde-meanfield-della-maxent-gaussianmlp-_s_1'
    goalkde_dir = '/home/kw2960/JaxGCRL/runs/run_ant_posvel-goalkde-meanfield-della-maxent-gaussianmlp-_s_1'
    goalkde_ckpt = os.path.join(goalkde_dir, 'ckpt', 'best.pkl')
    print(f"Loading GoalKDE checkpoint: {goalkde_ckpt}")
    goalkde_params = model.load_params(goalkde_ckpt)
    try:
        goalkde_policy_params, _, goalkde_context_params = goalkde_params
    except Exception:
        if isinstance(goalkde_params, dict):
            goalkde_policy_params = goalkde_params.get('policy') or goalkde_params.get('policy_params')
            goalkde_context_params = goalkde_params.get('context') or goalkde_params.get('context_params')
            if goalkde_policy_params is None or goalkde_context_params is None:
                raise ValueError('Unsupported GoalKDE checkpoint format')
        else:
            raise

    # Recreate networks
    # Read network hyperparameters and goal dim from GoalKDE run (args.pkl)
    h_dim = 1024
    n_hidden = 8
    use_ln = True
    gargs_path = os.path.join(goalkde_dir, 'args.pkl')
    try:
        with open(gargs_path, 'rb') as f:
            gargs = pickle.load(f)
        h_dim = int(getattr(gargs, 'h_dim', h_dim))
        n_hidden = int(getattr(gargs, 'n_hidden', n_hidden))
        use_ln = bool(getattr(gargs, 'use_ln', use_ln))
        env_name_g = str(getattr(gargs, 'env_name', ''))
        # For ant_fullobs runs the goal dimension is known to be 9
        if 'fullobs' in env_name_g.lower():
            goal_dim = 9
        if 'posvel' in env_name_g.lower():
            goal_dim = 6
    except Exception:
        pass

    block_size = 2
    num_blocks = max(1, n_hidden // block_size)

    # Set context output dim deterministically from goal_dim (mean and log_std)
    context_out_dim = goal_dim * 2

    actor = Net(act_dim * 2, h_dim, num_blocks, block_size, use_ln)
    context_net = Net(context_out_dim, h_dim, num_blocks, block_size, use_ln)
    goalkde_policy = make_policy(actor, goalkde_policy_params, deterministic=True)

    print(f"Dims: PPO obs_dim={obs_dim}, state_dim={state_dim}, inferred goal_dim={goal_dim}")

    # JIT fns
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_expert_policy = jax.jit(policy)
    jit_goalkde_policy = jax.jit(goalkde_policy)
    # Pass env into generalized reward function
    jit_reward_fn = jax.jit(lambda o, n, a: reward_function(o, n, a, env=env))

    NUM_ENVS = args.num_envs
    NUM_STEPS = args.episode_length

    # Collect expert trajectories (batched)
    def collect_expert_trajectory(rng):
        def step_fn(carry, _):
            state, rng = carry
            act_rng, next_rng = jax.random.split(rng)
            act, _ = jit_expert_policy(state.obs, act_rng)
            next_state = jit_env_step(state, act)
            rew = jit_reward_fn(state.obs, next_state.obs, act)
            done = next_state.done
            return (next_state, next_rng), (state.obs, act, rew, done)
        init_state = jit_env_reset(rng=rng)
        (final_state, _), (states, actions, rewards, dones) = jax.lax.scan(step_fn, (init_state, rng), None, length=NUM_STEPS)
        return states, actions, rewards, dones

    rng = jax.random.PRNGKey(args.seed)
    episode_rngs = jax.random.split(rng, NUM_ENVS)
    expert_states, expert_actions, expert_rewards, expert_dones = jax.vmap(collect_expert_trajectory)(episode_rngs)

    # Infer goals via mean-field product of Gaussians using only the first 29 dims as state
    expert_sa_states = expert_states[..., :state_dim]
    sa_pairs_mf = jp.reshape(jp.concatenate((expert_sa_states, expert_actions), axis=-1), (NUM_ENVS * NUM_STEPS, -1))
    def context_apply(x):
        xb = jp.expand_dims(x, 0)
        yb = context_net.apply(goalkde_context_params, xb)
        return jp.squeeze(yb, axis=0)
    context_out = jax.vmap(context_apply)(sa_pairs_mf)
    context_mean, context_log_std = jp.split(context_out, 2, axis=-1)
    context_mean = jp.reshape(context_mean, (NUM_ENVS, NUM_STEPS, -1))
    context_log_std = jp.reshape(context_log_std, (NUM_ENVS, NUM_STEPS, -1))
    precisions = 1.0 / jp.exp(2.0 * context_log_std)
    combined_precision = jp.sum(precisions, axis=1)
    combined_variance = 1.0 / combined_precision
    weighted_means = context_mean * precisions
    combined_mean = jp.sum(weighted_means, axis=1) / combined_precision
    # Sample one goal per env using per-env RNG (stable w.r.t. NUM_ENVS)
    base_key = jax.random.PRNGKey(args.seed + 1)
    env_ids = jp.arange(NUM_ENVS, dtype=jp.int32)
    env_keys = jax.vmap(lambda i: jax.random.fold_in(base_key, i))(env_ids)
    combined_std = jp.sqrt(combined_variance)
    def sample_goal(key, mean, std):
        eps = jax.random.normal(key, mean.shape)
        return mean + eps * std
    inferred_goals = jax.vmap(sample_goal)(env_keys, combined_mean, combined_std)
    # # Log per-env inferred goal breakdown and forward speed vs threshold (AntForward / ant_fullobs semantics)
    # if 'antforward' in model_path:
    #     goals_np = np.array(inferred_goals)
    #     print("Inferred goal breakdown per env (pos[0:3], vel[3:6], ang_vel[6:9]):")
    #     min_fv = float(env_params.get('min_forward_velocity', 0.5))
    #     fwd_speeds = []
    #     for i in range(goals_np.shape[0]):
    #         g = goals_np[i]
    #         pos = g[0:3]
    #         vel = g[3:6]
    #         ang = g[6:9]
    #         vx, vy = float(vel[0]), float(vel[1])
    #         speed = float(np.sqrt(vx * vx + vy * vy))
    #         fwd_speeds.append(speed)
    #         print(f"  Env {i}: pos={pos}, vel={vel}, ang_vel={ang}")
    #         print(f"          forward_speed_from_goal=sqrt(vx^2+vy^2)={speed:.4f} vs min_forward_velocity={min_fv:.4f} -> {'OK' if speed>=min_fv else 'LOW'}")
    #     print(f"  Mean forward speed from goals: {np.mean(fwd_speeds):.4f}")

    # Rollout GoalKDE policy toward inferred goals
    def rollout_goalkde(rng, target):
        def step_fn(carry, t):
            state, rng, counts, first_idxs = carry
            act_rng, next_rng = jax.random.split(rng)
            state_part = state.obs[:state_dim]
            # If target carries an extra leading dim, squeeze it
            goal_part = jp.reshape(target, (-1,))[:goal_dim]
            obs_for_policy = jp.concatenate((state_part, goal_part), axis=0)
            # Ensure 1D of expected length before policy
            obs_for_policy = jp.reshape(obs_for_policy, (state_dim + goal_dim,))
            act, _ = jit_goalkde_policy(obs_for_policy, act_rng)
            next_state = jit_env_step(state, act)
            rew = jit_reward_fn(state.obs, next_state.obs, act)

            # Diagnostics for NaNs/Infs
            bad_in = ~jp.all(jp.isfinite(obs_for_policy))
            bad_act = ~jp.all(jp.isfinite(act))
            bad_obs = ~jp.all(jp.isfinite(next_state.obs))
            bad_rew = ~jp.isfinite(rew)
            bads = jp.array([bad_in, bad_act, bad_obs, bad_rew], dtype=jp.int32)
            counts = counts.at[:4].add(bads)
            tvec = jp.array([t, t, t, t], dtype=jp.int32)
            first_idxs = first_idxs.at[:4].set(jp.where((first_idxs[:4] < 0) & (bads > 0), tvec, first_idxs[:4]))

            # Additional debugging: check for NaNs in quaternion
            torso_quat = next_state.obs[3:7]
            quat_norm = jp.linalg.norm(torso_quat)
            bad_quat = ~jp.isfinite(quat_norm) | (quat_norm < 1e-6)
            counts = counts.at[4].add(bad_quat.astype(jp.int32))
            first_idxs = jp.where((first_idxs[4] < 0) & bad_quat, t, first_idxs.at[4].set(t))
            
            # Check for NaNs in the environment state before the step
            bad_state_before = ~jp.all(jp.isfinite(state.obs))
            counts = counts.at[5].add(bad_state_before.astype(jp.int32))
            first_idxs = jp.where((first_idxs[5] < 0) & bad_state_before, t, first_idxs.at[5].set(t))

            return (next_state, next_rng, counts, first_idxs), (state.obs, rew)

        init_state = jit_env_reset(rng=rng)
        init_counts = jp.zeros((6,), dtype=jp.int32)  # Added state_before check
        init_first = -jp.ones((6,), dtype=jp.int32)   # Added state_before check
        (final_state, _, counts, first_idxs), (obs_seq, rews) = jax.lax.scan(
            step_fn, (init_state, rng, init_counts, init_first), jp.arange(NUM_STEPS), length=NUM_STEPS
        )
        return obs_seq, rews, counts, first_idxs

    rollout_rngs = jax.random.split(jax.random.PRNGKey(args.seed + 2), NUM_ENVS)
    goalkde_out = jax.vmap(rollout_goalkde)(rollout_rngs, inferred_goals)
    goalkde_states, goalkde_rewards, diag_counts, first_idxs = goalkde_out
    
    # Truncate trajectories at first NaN and compute total rewards
    def compute_truncated_total_rewards(rewards, first_nan_steps, expert_length):
        """Compute total rewards up to the first NaN OR expert length, whichever comes first."""
        # Create a mask: True for valid steps (before first NaN), False after
        step_indices = jp.arange(rewards.shape[0])
        
        # Truncate at either first NaN or expert length
        truncation_step = jp.where(
            first_nan_steps < 0,  # No NaNs found
            expert_length,  # Truncate at expert length
            jp.minimum(first_nan_steps, expert_length)  # Truncate at whichever comes first
        )
        
        valid_mask = step_indices < truncation_step
        
        # Apply mask and sum
        masked_rewards = jp.where(valid_mask, rewards, 0.0)
        total_reward = jp.sum(masked_rewards)
        
        # Return total reward and whether trajectory was truncated due to NaNs
        was_truncated_due_to_nan = first_nan_steps >= 0
        return total_reward, was_truncated_due_to_nan, truncation_step
    
    # Compute expert trajectory lengths from done flags (first True, else NUM_STEPS)
    def done_to_length(dones_row):
        idxs = jp.arange(dones_row.shape[0])
        first_done = jp.min(jp.where(dones_row, idxs, dones_row.shape[0]))
        return first_done
    expert_lengths = jax.vmap(done_to_length)(expert_dones)
    
    # Apply truncation to each environment for imitation trajectories
    goalkde_total_rewards, truncation_flags, truncation_steps = jax.vmap(compute_truncated_total_rewards)(
        goalkde_rewards, first_idxs[:, 2], expert_lengths  # Use next_obs NaN step (index 2) and expert lengths
    )
    
    # (suppress NaN diagnostics output; lengths and totals suffice)

    # Compare
    # Compute expert totals truncated to expert_lengths for fair step alignment
    def sum_to_length(rews_row, length):
        idxs = jp.arange(rews_row.shape[0])
        mask = idxs < length
        return jp.sum(jp.where(mask, rews_row, 0.0))
    expert_total_trunc = jax.vmap(sum_to_length)(expert_rewards, expert_lengths)
    
    print("GoalKDE imitation comparison:")
    print(f"  Expert mean total reward (to expert length): {float(jp.mean(expert_total_trunc)):.4f}")
    print(f"  Imitation mean total reward (truncated):     {float(jp.mean(goalkde_total_rewards)):.4f}")
    
    # Report truncation statistics
    truncation_flags_np = np.array(truncation_flags)
    truncation_steps_np = np.array(truncation_steps)
    expert_lengths_np = np.array(expert_lengths)
    
    # Verify lengths match per env (imitation truncated to min(first_nan, expert_len))
    lengths_match = np.all(truncation_steps_np == expert_lengths_np)
    print(f"  Lengths match (imitation vs expert): {bool(lengths_match)}")
    print(f"  Expert lengths:    {expert_lengths_np}")
    print(f"  Imitation lengths: {truncation_steps_np}")

    # AntForward velocity diagnostics: compare torso velocity vs intended velocity
    if 'antforward' in model_path:
        # Helper to compute local forward speed from obs row
        def local_forward_speed_from_obs(obs_row):
            # world velocity from indices 13:16, quat from 3:7
            v_world = obs_row[13:16]
            quat = obs_row[3:7]
            R = quat_to_3x3(quat)
            v_local = R.T @ v_world
            return v_local[0]
        v_local_expert = jax.vmap(jax.vmap(local_forward_speed_from_obs))(expert_states)
        v_local_imit = jax.vmap(jax.vmap(local_forward_speed_from_obs))(goalkde_states)
        # Align by expert length per env
        def mean_over_len(arr_row, length):
            idxs = jp.arange(arr_row.shape[0])
            mask = idxs < length
            vals = jp.where(mask, arr_row, 0.0)
            denom = jp.maximum(1, jp.sum(mask.astype(jp.int32)))
            return jp.sum(vals) / denom
        expert_fwd_mean = jax.vmap(mean_over_len)(v_local_expert, expert_lengths)
        imit_fwd_mean = jax.vmap(mean_over_len)(v_local_imit, expert_lengths)
        min_vel = float(env_params.get('min_forward_velocity', 0.5))
        # Fraction of steps above target
        def frac_above(arr_row, length, thr):
            idxs = jp.arange(arr_row.shape[0])
            mask = idxs < length
            hits = jp.where(mask, arr_row > thr, False)
            denom = jp.maximum(1, jp.sum(mask.astype(jp.int32)))
            return jp.sum(hits.astype(jp.int32)) / denom
        expert_frac = jax.vmap(frac_above)(v_local_expert, expert_lengths, min_vel * jp.ones_like(expert_lengths))
        imit_frac = jax.vmap(frac_above)(v_local_imit, expert_lengths, min_vel * jp.ones_like(expert_lengths))
        # print("  Forward velocity vs target (aligned to expert lengths):")
        # print(f"    Target min_forward_velocity: {min_vel:.3f}")
        # print(f"    Expert mean local vx per env:    {np.array(expert_fwd_mean)}")
        # print(f"    Imitation mean local vx per env: {np.array(imit_fwd_mean)}")
        # print(f"    Expert frac>target per env:      {np.array(expert_frac)}")
        # print(f"    Imitation frac>target per env:   {np.array(imit_frac)}")
        # print(f"    Expert mean vx: {float(jp.mean(expert_fwd_mean)):.4f}, Imitation mean vx: {float(jp.mean(imit_fwd_mean)):.4f}")

    # Compute regret using aligned (equal-length) totals
    regret = expert_total_trunc - goalkde_total_rewards
    regret_mean = jp.mean(regret)
    regret_var = jp.var(regret)
    # Standard error (unbiased): std(ddof=1) / sqrt(N)
    n_envs = regret.shape[0]
    regret_std_unbiased = jp.std(regret, ddof=1)
    regret_stderr = regret_std_unbiased / jp.sqrt(n_envs)
    print(f"  Mean regret (aligned lengths): {float(regret_mean):.4f}")
    # print(f"  Variance of regret: {float(regret_var):.4f}")
    print(f"  Std. error of mean regret: {float(regret_stderr):.4f}")
    # Stash for plotting later
    goalkde_regret_mean = float(regret_mean)
    goalkde_regret_stderr = float(regret_stderr)
    
    # Minimal per-env totals and aggregate
    goalkde_total_rewards_np = np.array(goalkde_total_rewards)
    # print(f"  Imitation totals (per env, truncated): {goalkde_total_rewards_np}")
    # print(f"  Expert totals (per env, aligned):     {np.array(expert_total_trunc)}")
    # print(f"  Regret: {np.array(regret)}")

    # ================= FB goal inference and imitation comparison =================
    try:
        fb_dir = '/home/kw2960/JaxGCRL/runs/run_ant_posvel-fb-della_2__1200000000_512_2048_1000_s_2'
        fb_ckpt = os.path.join(fb_dir, 'ckpt', 'best.pkl')
        print(f"\nLoading FB checkpoint: {fb_ckpt}")
        fb_params = model.load_params(fb_ckpt)

        # Unpack possible formats
        fb_policy_params, fb_backward_params = None, None
        if isinstance(fb_params, tuple) and len(fb_params) >= 4:
            fb_policy_params = fb_params[0]
            fb_backward_params = fb_params[3]
        elif isinstance(fb_params, dict):
            fb_policy_params = fb_params.get('policy') or fb_params.get('policy_params')
            fb_backward_params = (fb_params.get('target_backward') or
                                   fb_params.get('target_backward_params') or
                                   fb_params.get('backward') or fb_params.get('backward_params'))
        if fb_policy_params is None or fb_backward_params is None:
            raise ValueError('Unsupported FB checkpoint format; missing policy or backward params')

        # FB policy uses state + backward goal representation (repr_dim)
        repr_dim = 64
        backward_repr = Net(repr_dim, h_dim, num_blocks, block_size, use_ln)

        def slice_goal_obs(obs):
            return obs[env.goal_indices] if hasattr(env, 'goal_indices') else obs[-goal_dim:]

        def infer_fb_goal_for_env(states_row):
            goal_obs_seq = jax.vmap(slice_goal_obs)(states_row)
            reps = backward_repr.apply(fb_backward_params, goal_obs_seq)
            # Normalize per-step reps to match training convention (scale to sqrt(goal_dim))
            norms = jp.linalg.norm(reps, axis=-1, keepdims=True) + 1e-8
            reps_norm = reps / norms * jp.sqrt(goal_dim)
            return jp.mean(reps_norm, axis=0)

        fb_goals = jax.vmap(infer_fb_goal_for_env)(expert_states)
        print(f"FB inferred goals shape: {tuple(fb_goals.shape)}")

        def rollout_fb(rng, target):
            def step_fn(carry, t):
                state, rng, counts, first_idxs = carry
                act_rng, next_rng = jax.random.split(rng)
                state_part = state.obs[:state_dim]
                goal_part = jp.reshape(target, (-1,))[:goal_dim]
                # Map goal to representation via backward repr
                goal_b = jp.expand_dims(goal_part, 0)
                goal_repr_b = backward_repr.apply(fb_backward_params, goal_b)
                goal_repr = jp.squeeze(goal_repr_b, axis=0)
                actor_obs = jp.concatenate((state_part, goal_repr), axis=0)
                actor_obs = jp.reshape(actor_obs, (state_dim + repr_dim,))
                # Actor forward and deterministic action
                actor_obs_b = jp.expand_dims(actor_obs, 0)
                logits = actor.apply(fb_policy_params, actor_obs_b)
                act = parametric_action_distribution.mode(logits)[0]
                next_state = jit_env_step(state, act)
                rew = jit_reward_fn(state.obs, next_state.obs, act)

                bad_in = ~jp.all(jp.isfinite(actor_obs))
                bad_act = ~jp.all(jp.isfinite(act))
                bad_obs = ~jp.all(jp.isfinite(next_state.obs))
                bad_rew = ~jp.isfinite(rew)
                bads = jp.array([bad_in, bad_act, bad_obs, bad_rew], dtype=jp.int32)
                counts = counts.at[:4].add(bads)
                tvec = jp.array([t, t, t, t], dtype=jp.int32)
                first_idxs = first_idxs.at[:4].set(jp.where((first_idxs[:4] < 0) & (bads > 0), tvec, first_idxs[:4]))

                return (next_state, next_rng, counts, first_idxs), rew

            init_state = jit_env_reset(rng=rng)
            init_counts = jp.zeros((6,), dtype=jp.int32)
            init_first = -jp.ones((6,), dtype=jp.int32)
            (final_state, _, counts, first_idxs), rews = jax.lax.scan(
                step_fn, (init_state, rng, init_counts, init_first), jp.arange(NUM_STEPS), length=NUM_STEPS
            )
            return rews, counts, first_idxs

        fb_rollout_rngs = jax.random.split(jax.random.PRNGKey(args.seed + 3), NUM_ENVS)
        fb_rewards, fb_counts, fb_first_idxs = jax.vmap(rollout_fb)(fb_rollout_rngs, fb_goals)

        # FB: robust truncation = min(next_obs_bad, reward_bad), and treat step-0 as no-trunc
        fb_first_bad = jp.minimum(fb_first_idxs[:, 2], fb_first_idxs[:, 3])
        fb_total_rewards, fb_trunc_flags, fb_trunc_steps = jax.vmap(compute_truncated_total_rewards)(
            fb_rewards, fb_first_bad, expert_lengths
        )

        # Debug: check immediate truncations
        fb_trunc_steps_np = np.array(fb_trunc_steps)
        print(f"FB truncation at step 0 count: {int((fb_trunc_steps_np == 0).sum())}")

        fb_regret = expert_total_trunc - fb_total_rewards
        fb_regret_mean = jp.mean(fb_regret)
        fb_regret_stderr = jp.std(fb_regret, ddof=1) / jp.sqrt(fb_regret.shape[0])

        print("\nFB imitation comparison:")
        print(f"  Mean regret (aligned lengths): {float(fb_regret_mean):.4f}")
        print(f"  Std. error of mean regret:    {float(fb_regret_stderr):.4f}")

        print("\nSummary (lower is better):")
        print(f"  GoalKDE mean regret: {float(regret_mean):.4f} +/- {float(regret_stderr):.4f}")
        print(f"  FB      mean regret: {float(fb_regret_mean):.4f} +/- {float(fb_regret_stderr):.4f}")

    except Exception as e:
        print(f"FB comparison skipped due to error: {e}")

    # ================= Additional mean-field checkpoint goal inference (like GoalKDE) =================
    try:
        mainmf_dir = '/home/kw2960/JaxGCRL/runs/run_ant_posvel-main-meanfield-test_s_1'
        mainmf_ckpt = os.path.join(mainmf_dir, 'ckpt', 'best.pkl')
        print(f"\nLoading MainMF checkpoint: {mainmf_ckpt}")
        mainmf_params = model.load_params(mainmf_ckpt)
        # Expect similar structure: (_, _, context_params) or dict with 'context'
        try:
            _, _, mainmf_context_params = mainmf_params
        except Exception:
            if isinstance(mainmf_params, dict):
                mainmf_context_params = mainmf_params.get('context') or mainmf_params.get('context_params')
                if mainmf_context_params is None:
                    raise ValueError('Unsupported MainMF checkpoint format')
            else:
                raise

        # Try to infer goal_dim from this run's args.pkl if present
        mm_args_path = os.path.join(mainmf_dir, 'args.pkl')
        try:
            with open(mm_args_path, 'rb') as f:
                mmargs = pickle.load(f)
            if hasattr(mmargs, 'env_name') and 'fullobs' in str(mmargs.env_name).lower():
                goal_dim_mm = 9
            elif hasattr(mmargs, 'env_name') and 'posvel' in str(mmargs.env_name).lower():
                goal_dim_mm = 6
            else:
                goal_dim_mm = goal_dim
        except Exception:
            goal_dim_mm = goal_dim

        # Reuse context_net architecture but with current params
        def mainmf_context_apply(x):
            xb = jp.expand_dims(x, 0)
            yb = context_net.apply(mainmf_context_params, xb)
            return jp.squeeze(yb, axis=0)

        mm_context_out = jax.vmap(mainmf_context_apply)(sa_pairs_mf)
        mm_context_mean, mm_context_log_std = jp.split(mm_context_out, 2, axis=-1)
        mm_context_mean = jp.reshape(mm_context_mean, (NUM_ENVS, NUM_STEPS, -1))
        mm_context_log_std = jp.reshape(mm_context_log_std, (NUM_ENVS, NUM_STEPS, -1))
        mm_precisions = 1.0 / jp.exp(2.0 * mm_context_log_std)
        mm_combined_precision = jp.sum(mm_precisions, axis=1)
        mm_combined_variance = 1.0 / mm_combined_precision
        mm_weighted_means = mm_context_mean * mm_precisions
        mm_combined_mean = jp.sum(mm_weighted_means, axis=1) / mm_combined_precision
        # Sample one goal per env using per-env RNG (stable w.r.t. NUM_ENVS)
        base_key_mm = jax.random.PRNGKey(args.seed + 101)
        env_keys_mm = jax.vmap(lambda i: jax.random.fold_in(base_key_mm, i))(env_ids)
        mm_combined_std = jp.sqrt(mm_combined_variance)
        def sample_goal_mm(key, mean, std):
            eps = jax.random.normal(key, mean.shape)
            return mean + eps * std
        mainmf_goals = jax.vmap(sample_goal_mm)(env_keys_mm, mm_combined_mean, mm_combined_std)

        # Rollout using the same GoalKDE policy toward mainmf_goals
        mainmf_out = jax.vmap(rollout_goalkde)(rollout_rngs, mainmf_goals)
        _, mainmf_rewards, mainmf_counts, mainmf_first_idxs = mainmf_out

        # Truncate and compute regret
        mainmf_total_rewards, _, mainmf_trunc_steps = jax.vmap(compute_truncated_total_rewards)(
            mainmf_rewards, mainmf_first_idxs[:, 2], expert_lengths
        )
        mainmf_regret = expert_total_trunc - mainmf_total_rewards
        mainmf_regret_mean = jp.mean(mainmf_regret)
        mainmf_regret_stderr = jp.std(mainmf_regret, ddof=1) / jp.sqrt(mainmf_regret.shape[0])
        print("\nMainMF imitation comparison:")
        print(f"  Mean regret (aligned lengths): {float(mainmf_regret_mean):.4f}")
        print(f"  Std. error of mean regret:    {float(mainmf_regret_stderr):.4f}")
    except Exception as e:
        print(f"MainMF comparison skipped due to error: {e}")

    # ===================== Final plotting for AntForward =====================
    try:
        # Extract expert policy name and inference method for file naming
        expert_policy = 'antforward' if 'antforward' in model_path else 'antjump' if 'antjump' in model_path else 'antflip'
        inference_method = 'ant_fullobs' if 'fullobs' in goalkde_dir else 'ant_posvel' if 'posvel' in goalkde_dir else 'unknown'
        
        if expert_policy in ['antforward', 'antjump'] and ('fb_regret_mean' in locals() and 'fb_regret_stderr' in locals()):
            labels = []
            means = []
            stderrs = []
            colors = []
            # MainMF (blue) if available
            if 'mainmf_regret_mean' in locals() and 'mainmf_regret_stderr' in locals():
                labels.append('MainMF')
                means.append(float(mainmf_regret_mean))
                stderrs.append(float(mainmf_regret_stderr))
                colors.append('#1f77b4')  # blue
            # GoalKDE (orange)
            labels.append('CRL + GoalKDE')
            means.append(goalkde_regret_mean)
            stderrs.append(goalkde_regret_stderr)
            colors.append('#ff7f0e')
            # FB (green)
            labels.append('FB')
            means.append(float(fb_regret_mean))
            stderrs.append(float(fb_regret_stderr))
            colors.append('#2ca02c')

            # PNG plot
            fig, ax = plt.subplots(figsize=(6, 5))
            x = np.arange(len(labels))
            ax.bar(x, means, yerr=stderrs, capsize=6, color=colors)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=0)
            ax.set_ylabel('Regret')
            ax.grid(axis='y', linestyle='--', alpha=0.5)

            out_dir = os.path.join('.', f'results_{inference_method}')
            os.makedirs(out_dir, exist_ok=True)
            out_png = os.path.join(out_dir, f'{expert_policy}_regret_goalkde_vs_fb.png')
            plt.tight_layout()
            plt.savefig(out_png, dpi=300, bbox_inches='tight')
            print(f"Saved regret comparison plot to: {out_png}")
            plt.close(fig)

            # CSV
            import csv
            out_csv = os.path.join(out_dir, f'{expert_policy}_regret_goalkde_vs_fb.csv')
            with open(out_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Method', 'MeanRegret', 'StdError'])
                for lbl, m, s in zip(labels, means, stderrs):
                    writer.writerow([lbl, f"{m:.6f}", f"{s:.6f}"])
            print(f"Saved regret comparison data to: {out_csv}")
    except Exception as pe:
        print(f"Final plotting skipped due to error: {pe}")
    
    # ===================== Plot expert vs imitation rewards (GoalKDE & FB) =====================
    try:
        # Extract expert policy name and inference method for file naming
        expert_policy = 'antforward' if 'antforward' in model_path else 'antjump' if 'antjump' in model_path else 'antflip'
        inference_method = 'ant_fullobs' if 'fullobs' in goalkde_dir else 'ant_posvel' if 'posvel' in goalkde_dir else 'unknown'
        
        # Compute means and stderr for expert and imitation (GoalKDE)
        exp_mean = float(jp.mean(expert_total_trunc))
        exp_stderr = float(jp.std(expert_total_trunc, ddof=1) / jp.sqrt(expert_total_trunc.shape[0]))
        gkde_mean = float(jp.mean(goalkde_total_rewards))
        gkde_stderr = float(jp.std(goalkde_total_rewards, ddof=1) / jp.sqrt(goalkde_total_rewards.shape[0]))

        have_fb = ('fb_regret_mean' in locals()) and ('fb_total_rewards' in locals())
        if have_fb:
            fb_mean = float(jp.mean(fb_total_rewards))
            fb_stderr = float(jp.std(fb_total_rewards, ddof=1) / jp.sqrt(fb_total_rewards.shape[0]))

        if expert_policy in ['antforward', 'antjump']:
            # Build grouped bars: [GoalKDE Expert, GoalKDE Imit, FB Expert, FB Imit]
            labels = ['GoalKDE Expert', 'GoalKDE Imit']
            means = [exp_mean, gkde_mean]
            stderrs = [exp_stderr, gkde_stderr]
            colors = ['#1f77b4', '#ff7f0e']  # blue for expert, orange for GoalKDE imitation
            if have_fb:
                labels += ['FB Expert', 'FB Imit']
                means += [exp_mean, fb_mean]
                stderrs += [exp_stderr, fb_stderr]
                colors += ['#1f77b4', '#2ca02c']  # blue for expert, green for FB imitation

            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(labels))
            ax.bar(x, means, yerr=stderrs, capsize=6, color=colors)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20)
            ax.set_ylabel('Total Reward (± StdErr)')
            ax.grid(axis='y', linestyle='--', alpha=0.5)

            out_dir = os.path.join('.', f'results_{inference_method}')
            os.makedirs(out_dir, exist_ok=True)
            out_png = os.path.join(out_dir, f'{expert_policy}_expert_vs_imitation_rewards.png')
            plt.tight_layout()
            plt.savefig(out_png, dpi=300, bbox_inches='tight')
            print(f"Saved expert vs imitation rewards plot to: {out_png}")
            plt.close(fig)

            # CSV dump
            import csv
            out_csv = os.path.join(out_dir, f'{expert_policy}_expert_vs_imitation_rewards.csv')
            with open(out_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Series', 'MeanTotalReward', 'StdError'])
                for lbl, m, s in zip(labels, means, stderrs):
                    writer.writerow([lbl, f"{m:.6f}", f"{s:.6f}"])
            print(f"Saved expert vs imitation rewards data to: {out_csv}")
    except Exception as pe:
        print(f"Expert vs imitation plotting skipped due to error: {pe}")
    
    # Save trajectories if requested
    if args.save_trajectories:
        output_path = model_path.replace('.pkl', '_trajectories.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(all_trajectories, f)
        print(f"\nTrajectories saved to: {output_path}")
    
    # Print some trajectory statistics (disabled when per-episode eval is commented)
    # print("\nTRAJECTORY STATISTICS:")
    # print(f"Observation shape: {all_trajectories[0]['observations'].shape}")
    # print(f"Action shape: {all_trajectories[0]['actions'].shape}")
    # print(f"Total state-action pairs collected: {sum(len(t['observations']) for t in all_trajectories)}")
    
    # 3D Visualization
    # if args.visualize_3d:
    #     print("\nCreating 3D visualizations...")
    #     plot_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else '.'
    #     plot_dir = os.path.join(plot_dir, 'plots')
    #     os.makedirs(plot_dir, exist_ok=True)
    #     if args.plot_single:
    #         for i, trajectory in enumerate(all_trajectories):
    #             plot_path = os.path.join(plot_dir, f'torso_trajectory_episode_{i+1}.png')
    #             plot_torso_trajectory_3d(trajectory, title=f"Torso Trajectory - Episode {i+1}", save_path=plot_path)
    #     else:
    #         plot_path = os.path.join(plot_dir, 'torso_trajectories_all.png')
    #         plot_multiple_trajectories(all_trajectories, title=f"Torso Trajectories - {args.num_episodes} Episodes", save_path=plot_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained PPO policy and collect trajectories.')
    
    parser.add_argument(
        '--model_path',
        type=str,
        default='simple_ppo/antforward/ppo_antforward_vel0.5_model.pkl',
        help='Path to the trained PPO model file.'
    )
    parser.add_argument(
        '--env',
        type=str,
        choices=['antforward', 'antjump', 'antflip'],
        help='Environment name. If provided, will automatically find the latest model in simple_ppo/{env}/'
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=5,
        help='Number of episodes to collect.'
    )
    parser.add_argument(
        '--episode_length',
        type=int,
        default=1024,
        help='Maximum length of each episode.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducibility.'
    )
    parser.add_argument(
        '--num_envs',
        type=int,
        default=100,
        help='Number of parallel rollouts for GoalKDE comparison (vmapped).'
    )
    parser.add_argument(
        '--save_trajectories',
        action='store_true',
        help='Save collected trajectories to a pickle file.'
    )
    parser.add_argument(
        '--visualize_3d',
        action='store_true',
        help='Create 3D visualization of torso trajectories.'
    )
    parser.add_argument(
        '--plot_single',
        action='store_true',
        help='Plot individual trajectories separately.'
    )
    parser.add_argument(
        '--target_height',
        type=float,
        default=1.0,
        help='Target jump height for AntJump environment (when using custom rewards).'
    )
    parser.add_argument(
        '--min_forward_velocity',
        type=float,
        default=0.5,
        help='Minimum forward velocity for AntForward environment (when using custom rewards).'
    )
    parser.add_argument(
        '--min_flip_velocity',
        type=float,
        default=1.5,
        help='Minimum flip velocity for AntFlip environment (when using custom rewards).'
    )
    parser.add_argument(
        '--ctrl_cost_weight',
        type=float,
        default=0.5,
        help='Control cost weight for reward calculations.'
    )
    parser.add_argument(
        '--healthy_reward',
        type=float,
        default=1.0,
        help='Healthy reward per timestep when ant is within healthy height range.'
    )
    
    args = parser.parse_args()
    main(args)
