"""
Training script that uses configuration files for different ant environments.
"""

import argparse
from datetime import datetime
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), '..'))

from envs.ant_base import AntForward, AntJump, AntFlip
from envs.walker2d import WalkerForward, WalkerJump, WalkerFlip

import brax
from brax import envs
from brax.io import html
from brax.io import model
from brax.training.agents.ppo import train as ppo_train
from brax.training.agents.ppo import networks as ppo_networks

import jax

# Register custom environments
envs.register_environment('antforward', AntForward)
envs.register_environment('antjump', AntJump)
envs.register_environment('antflip', AntFlip)
envs.register_environment('walkerforward', WalkerForward)
envs.register_environment('walkerjump', WalkerJump)
envs.register_environment('walkerflip', WalkerFlip)


def load_config(env_name):
    """Load configuration for the specified environment."""
    print(f"Loading config for environment: {env_name}")
    if env_name == 'antforward':
        from configs.ant_forward_config import TRAINING_CONFIG, ENV_CONFIG
        print(f"Loaded antforward config: {ENV_CONFIG}")
    elif env_name == 'antjump':
        from configs.ant_jump_config import TRAINING_CONFIG, ENV_CONFIG
        print(f"Loaded antjump config: {ENV_CONFIG}")
    elif env_name == 'antflip':
        from configs.ant_flip_config import TRAINING_CONFIG, ENV_CONFIG
        print(f"Loaded antflip config: {ENV_CONFIG}")
    else:
        raise ValueError(f"Unsupported environment: {env_name}")
    
    return TRAINING_CONFIG, ENV_CONFIG


def create_environment(env_name, env_config):
    """Create environment based on configuration."""
    print(f"Creating environment: {env_name} with config: {env_config}")
    if env_name == 'antforward':
        env = AntForward(min_forward_velocity=env_config['min_forward_velocity'])
        print(f"Created AntForward with min_forward_velocity: {env_config['min_forward_velocity']}")
        return env
    elif env_name == 'antjump':
        env = AntJump(target_jump_height=env_config['target_jump_height'])
        print(f"Created AntJump with target_jump_height: {env_config['target_jump_height']}")
        return env
    elif env_name == 'antflip':
        env = AntFlip(min_flip_velocity=env_config['min_flip_velocity'])
        print(f"Created AntFlip with min_flip_velocity: {env_config['min_flip_velocity']}")
        return env
    else:
        raise ValueError(f"Unsupported environment: {env_name}")


def main(args):
    """Main training function."""
    # Load configuration
    training_config, env_config = load_config(args.env)
    
    # Override config with command line arguments if provided
    if args.total_env_steps is not None:
        training_config['total_env_steps'] = args.total_env_steps
    if args.episode_length is not None:
        training_config['episode_length'] = args.episode_length
    if args.num_envs is not None:
        training_config['num_envs'] = args.num_envs
    if args.lr is not None:
        training_config['learning_rate'] = args.lr
    if args.entropy_cost is not None:
        training_config['entropy_cost'] = args.entropy_cost
    if args.seed is not None:
        training_config['seed'] = args.seed
    
    # Create output directory for this training run
    output_dir = f"simple_ppo/{args.env}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Create environment
    env = create_environment(args.env, env_config)
    print(f"Created {args.env} environment with config: {env_config}")
    
    # PPO network factory
    network_factory = ppo_networks.make_ppo_networks

    times = [datetime.now()]

    def progress(num_steps, metrics):
        times.append(datetime.now())
        print(
            '  Steps: {:,}, Time: {}, Eval Mean Reward: {:,.4f}'.format(
                num_steps, times[-1] - times[0], metrics['eval/episode_reward']
            )
        )

    print(f'Training {args.env} with PPO...')
    print(f'Training config: {training_config}')
    
    # PPO train function
    make_policy, params, _ = ppo_train.train(
        environment=env,
        num_timesteps=training_config['total_env_steps'],
        episode_length=training_config['episode_length'],
        num_envs=training_config['num_envs'],
        learning_rate=training_config['learning_rate'],
        entropy_cost=training_config['entropy_cost'],
        discounting=training_config['discounting'],
        seed=training_config['seed'],
        unroll_length=training_config['unroll_length'],
        batch_size=training_config['batch_size'],
        num_minibatches=training_config['num_minibatches'],
        num_updates_per_batch=training_config['num_updates_per_batch'],
        num_evals=training_config['num_evals'],
        normalize_observations=training_config['normalize_observations'],
        network_factory=network_factory,
        progress_fn=progress,
    )
    print('Training finished.')

    # Create parameter string for filename
    param_str = ''
    if 'forward' in args.env:
        param_str = f'_vel{env_config["min_forward_velocity"]}'
    elif 'jump' in args.env:
        param_str = f'_h{env_config["target_jump_height"]}'
    elif 'flip' in args.env:
        param_str = f'_flipvel{env_config["min_flip_velocity"]}'

    # Save model
    model_path = os.path.join(output_dir, f'ppo_{args.env}{param_str}_model.pkl')
    model.save_params(model_path, params)
    print(f'Model saved to {model_path}')
    
    # Save training configuration
    import json
    config_save_path = os.path.join(output_dir, 'training_config.json')
    with open(config_save_path, 'w') as f:
        json.dump({
            'env_config': env_config,
            'training_config': training_config
        }, f, indent=2)
    print(f'Training configuration saved to {config_save_path}')

    # Visualize
    print('Creating video...')
    policy = make_policy(params, deterministic=True)

    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_policy = jax.jit(policy)

    rollouts = []
    rng = jax.random.PRNGKey(training_config['seed'])

    for i in range(training_config['num_eval_episodes']):
        print(f'Visualizing episode {i+1}/{training_config["num_eval_episodes"]}')
        rng, reset_rng = jax.random.split(rng)
        state = jit_env_reset(rng=reset_rng)
        rollout = [state.pipeline_state]
        for _ in range(training_config['episode_length']):
            act_rng, rng = jax.random.split(rng)
            act, _ = jit_policy(state.obs, act_rng)
            state = jit_env_step(state, act)
            rollout.append(state.pipeline_state)
            if state.done.all():
                break

        html_path = os.path.join(output_dir, f'ppo_{args.env}{param_str}_video_{i}.html')
        html.save(html_path, env.sys.tree_replace({'opt.timestep': env.dt}), rollout)
        print(f'Video saved to {html_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PPO on Ant environments using config files.')

    parser.add_argument(
        '--env',
        type=str,
        required=True,
        choices=['antforward', 'antjump', 'antflip'],
        help='Environment to train.',
    )
    
    # Optional overrides for config values
    parser.add_argument(
        '--total_env_steps',
        type=int,
        help='Override total number of environment steps to train for.',
    )
    parser.add_argument(
        '--episode_length',
        type=int,
        help='Override episode length.',
    )
    parser.add_argument(
        '--num_envs',
        type=int,
        help='Override number of parallel environments.',
    )
    parser.add_argument(
        '--lr',
        type=float,
        help='Override learning rate.',
    )
    parser.add_argument(
        '--entropy_cost',
        type=float,
        help='Override entropy cost coefficient.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Override random seed.',
    )

    args = parser.parse_args()
    main(args)
