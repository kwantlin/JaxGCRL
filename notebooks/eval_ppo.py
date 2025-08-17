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


def find_latest_model(env_name):
    """Find the latest model file for a given environment."""
    pattern = f"simple_ppo/{env_name}/ppo_{env_name}_*_model.pkl"
    model_files = glob.glob(pattern)
    
    if not model_files:
        raise FileNotFoundError(f"No model files found for {env_name} in simple_ppo/{env_name}/")
    
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


def collect_trajectory(env, policy, episode_length=1000, seed=0):
    """Collect a single trajectory of state-action pairs."""
    rng = jax.random.PRNGKey(seed)
    
    # JIT the environment and policy functions for efficiency
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_policy = jax.jit(policy)
    
    # Reset environment
    rng, reset_rng = jax.random.split(rng)
    state = jit_env_reset(rng=reset_rng)
    
    # Initialize trajectory storage
    observations = []
    actions = []
    rewards = []
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
        state = jit_env_step(state, action)
        
        # Store reward and done
        rewards.append(float(state.reward))
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
    if 'antjump' in model_path:
        # Extract target jump height from filename if possible
        target_height = 1.0  # default
        if 'h1.0' in model_path:
            target_height = 1.0
        elif 'h1.3' in model_path:
            target_height = 1.3
        # Add more cases as needed
        
        env = AntJump(target_jump_height=target_height)
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
        print(f"Created AntFlip environment with min_flip_velocity: {min_flip_velocity}")
    else:
        raise ValueError(f"Unsupported environment type in model path: {model_path}")
    
    # Load the trained policy
    policy = load_policy(model_path, env)
    print("Policy loaded successfully")
    
    # Collect trajectories
    all_trajectories = []
    total_rewards = []
    
    for episode in range(args.num_episodes):
        print(f"Collecting episode {episode + 1}/{args.num_episodes}")
        
        trajectory = collect_trajectory(
            env, 
            policy, 
            episode_length=args.episode_length,
            seed=args.seed + episode
        )
        
        all_trajectories.append(trajectory)
        total_reward = np.sum(trajectory['rewards'])
        total_rewards.append(total_reward)
        
        print(f"  Episode {episode + 1} total reward: {total_reward:.2f}")
        print(f"  Episode {episode + 1} length: {len(trajectory['observations'])}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Mean total reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Min total reward: {np.min(total_rewards):.2f}")
    print(f"Max total reward: {np.max(total_rewards):.2f}")
    print(f"Mean episode length: {np.mean([len(t['observations']) for t in all_trajectories]):.1f}")
    
    # Save trajectories if requested
    if args.save_trajectories:
        output_path = model_path.replace('.pkl', '_trajectories.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(all_trajectories, f)
        print(f"\nTrajectories saved to: {output_path}")
    
    # Print some trajectory statistics
    print("\nTRAJECTORY STATISTICS:")
    print(f"Observation shape: {all_trajectories[0]['observations'].shape}")
    print(f"Action shape: {all_trajectories[0]['actions'].shape}")
    print(f"Total state-action pairs collected: {sum(len(t['observations']) for t in all_trajectories)}")
    
    # 3D Visualization
    if args.visualize_3d:
        print("\nCreating 3D visualizations...")
        
        # Create output directory for plots
        plot_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else '.'
        plot_dir = os.path.join(plot_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        if args.plot_single:
            # Plot each trajectory separately
            for i, trajectory in enumerate(all_trajectories):
                plot_path = os.path.join(plot_dir, f'torso_trajectory_episode_{i+1}.png')
                plot_torso_trajectory_3d(
                    trajectory, 
                    title=f"Torso Trajectory - Episode {i+1}",
                    save_path=plot_path
                )
        else:
            # Plot all trajectories together
            plot_path = os.path.join(plot_dir, 'torso_trajectories_all.png')
            plot_multiple_trajectories(
                all_trajectories,
                title=f"Torso Trajectories - {args.num_episodes} Episodes",
                save_path=plot_path
            )


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
        default=1000,
        help='Maximum length of each episode.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducibility.'
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
    
    args = parser.parse_args()
    main(args)
