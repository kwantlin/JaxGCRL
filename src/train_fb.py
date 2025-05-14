import functools
import time
from typing import Callable, Optional, NamedTuple

import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import jax
from jax import numpy as jnp
import optax
from absl import logging
import brax
from brax import envs
from brax.training import gradients, distribution, types, pmap
from brax.training.replay_buffers_test import jit_wrap

from envs.wrappers import TrajectoryIdWrapper
from src.evaluator import CrlEvaluator
from src.replay_buffer import ReplayBufferState, Transition, TrajectoryUniformSamplingQueue

Metrics = types.Metrics
Env = envs.Env
State = envs.State
_PMAP_AXIS_NAME = "i"

# The SAEncoder, GoalEncoder, and Actor all use the same function. Output size for SA/Goal encoders should be representation size, and for Actor should be 2 * action_size.
# To keep parity with the existing architecture, by default we only use one residual block of depth 2, hence effectively not using the residual connections.
class Net(nn.Module):
    """
    MLP with residual connections: residual blocks have $block_size layers. Uses swish activation, optionally uses layernorm.
    """
    output_size: int
    width: int = 1024
    num_blocks: int = 1
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

# The brax version of this does not take in the actor and action_distribution arguments; before we pass it to brax evaluator or return it from train(), we do a partial application.
def make_policy(actor, parametric_action_distribution, params, deterministic=False):
    def policy(obs, key_sample):
        print("obs shape", obs.shape)
        logits = actor.apply(params, obs)
        print("LOGITS", logits.shape)
        if deterministic:
            action = parametric_action_distribution.mode(logits)
        else:
            action = parametric_action_distribution.sample(logits, key_sample)
            print("ACTION SHAPE", action.shape)
        extras = {}
        return action, extras
    return policy

@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    gradient_steps: jnp.ndarray
    env_steps: jnp.ndarray
    actor_state: TrainState
    # critic_state: TrainState
    # value_state: TrainState
    fb_repr_state: TrainState
    # target_critic_params: flax.core.FrozenDict
    target_forward_repr_params: flax.core.FrozenDict
    target_backward_repr_params: flax.core.FrozenDict

def _init_training_state(key, actor, forward_repr, backward_repr, state_dim, goal_dim, action_dim, repr_dim, episode_length, actor_lr, repr_lr, num_local_devices_to_use):
    """
    Initializes the training state for a forward-backward representation learning model.
    """
    actor_key, critic_key, value_key, forward_key, backward_key = jax.random.split(key, 5)
    
    # Actor
    actor_params = actor.init(actor_key, jnp.ones([1, state_dim + goal_dim]))
    actor_state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=optax.adam(learning_rate=actor_lr))

    # Critic and Value
    # critic_params = critic.init(critic_key, jnp.ones([1, state_dim + action_dim + goal_dim]))
    # value_params = value.init(value_key, jnp.ones([1, state_dim + goal_dim]))
    # critic_state = TrainState.create(apply_fn=critic.apply, params=critic_params, tx=optax.adam(learning_rate=critic_lr))
    # value_state = TrainState.create(apply_fn=value.apply, params=value_params, tx=optax.adam(learning_rate=critic_lr))

    # Forward and backward representation networks
    forward_repr_params = forward_repr.init(forward_key, jnp.ones([1, state_dim + action_dim + goal_dim]))
    backward_repr_params = backward_repr.init(backward_key, jnp.ones([1, state_dim]))
    
    # Single optimizer for both forward and backward repr networks
    repr_optimizer = optax.adam(learning_rate=repr_lr)
    fb_repr_state = TrainState.create(apply_fn=None, params=(forward_repr_params, backward_repr_params), tx=repr_optimizer)

    # Target networks: just store params
    # target_critic_params = jax.tree_util.tree_map(lambda x: x.copy(), critic_params)
    target_forward_repr_params = jax.tree_util.tree_map(lambda x: x.copy(), forward_repr_params)
    target_backward_repr_params = jax.tree_util.tree_map(lambda x: x.copy(), backward_repr_params)

    training_state = TrainingState(
        env_steps=jnp.zeros(()), 
        gradient_steps=jnp.zeros(()), 
        actor_state=actor_state,
        # critic_state=critic_state,
        # value_state=value_state,
        fb_repr_state=fb_repr_state,
        # target_critic_params=target_critic_params,
        target_forward_repr_params=target_forward_repr_params,
        target_backward_repr_params=target_backward_repr_params
    )
    
    training_state = jax.device_put_replicated(training_state, jax.local_devices()[:num_local_devices_to_use])
    return training_state


def forward_backward_repr_loss(
    forward_params, backward_params, forward_repr, backward_repr, transitions, state_dim, goal_dim, repr_dim, key,
    training_state, actor,const_std=True, repr_agg='mean', orthonorm_coef=1.0, discount=0.99
):
    """
    Compute the forward-backward representation loss, following the structure of ogbench/impls/agents/fb_repr.py.
    """
    states = transitions.observation[:, :state_dim]
    actions = transitions.action
    next_states = transitions.extras["future_state"][:, :state_dim]
    goals = transitions.observation[:, state_dim:]
    batch_size = states.shape[0]
    latent_dim = goal_dim

    # Sample latents - we already did this in the actor step
    latents = goals

    # Compute next actions using the actor
    next_dist = actor.apply(jax.lax.stop_gradient(training_state.actor_state.params), jnp.concatenate([next_states, latents], axis=-1))
    if const_std:
        next_actions = jnp.clip(next_dist[..., :next_dist.shape[-1] // 2], -1, 1)  # mode
    else:
        next_actions = jnp.clip(next_dist[..., :next_dist.shape[-1] // 2], -1, 1)  # fallback to mode if sampling not available

    # Compute target forward and backward representations using target params
    if training_state.target_forward_repr_params is not None:
        next_forward_reprs = forward_repr.apply(training_state.target_forward_repr_params, jnp.concatenate([next_states, next_actions, latents], axis=-1))
    else:
        next_forward_reprs = forward_repr.apply(forward_params, jnp.concatenate([next_states, next_actions, latents], axis=-1))
    if training_state.target_backward_repr_params is not None:
        next_backward_reprs = backward_repr.apply(training_state.target_backward_repr_params, next_states)
    else:
        next_backward_reprs = backward_repr.apply(backward_params, next_states)
    next_backward_reprs = next_backward_reprs / jnp.linalg.norm(next_backward_reprs, axis=-1, keepdims=True) * jnp.sqrt(repr_dim)
    target_occ_measures = jnp.einsum('bd,td->bt', next_forward_reprs, next_backward_reprs)

    # Aggregate target occupancy measures
    if repr_agg == 'mean':
        target_occ_measures = jnp.mean(target_occ_measures, axis=0)
    else:
        target_occ_measures = jnp.min(target_occ_measures, axis=0)

    # Compute current forward and backward representations
    forward_reprs = forward_repr.apply(forward_params, jnp.concatenate([states, actions, latents], axis=-1))
    backward_reprs = backward_repr.apply(backward_params, next_states)
    backward_reprs = backward_reprs / jnp.linalg.norm(backward_reprs, axis=-1, keepdims=True) * jnp.sqrt(repr_dim)
    occ_measures = jnp.einsum('bd,td->bt', forward_reprs, backward_reprs)

    I = jnp.eye(occ_measures.shape[0])
    repr_off_diag_loss = jax.vmap(
        lambda x: (x * (1 - I)) ** 2,
        0, 0
    )(occ_measures - discount * target_occ_measures)

    repr_diag_loss = jax.vmap(jnp.diag, 0, 0)(occ_measures)

    repr_loss = jnp.mean(
        repr_diag_loss + jnp.sum(repr_off_diag_loss, axis=-1) / (occ_measures.shape[0] - 1)
    )

    # Orthonormalization loss
    covariance = jnp.matmul(backward_reprs, backward_reprs.T)
    ortho_diag_loss = -2 * jnp.diag(covariance)
    ortho_off_diag_loss = (covariance * (1 - I)) ** 2
    ortho_loss = orthonorm_coef * jnp.mean(
        ortho_diag_loss + jnp.sum(ortho_off_diag_loss, axis=-1) / (occ_measures.shape[0] - 1)
    )

    total_loss = repr_loss + ortho_loss

    metrics = {
        'repr_loss': repr_loss,
        'repr_diag_loss': jnp.mean(jnp.sum(repr_diag_loss, axis=-1) / (occ_measures.shape[0] - 1)),
        'repr_off_diag_loss': jnp.mean(repr_off_diag_loss),
        'ortho_loss': ortho_loss,
        'ortho_diag_loss': jnp.sum(ortho_diag_loss, axis=-1) / (occ_measures.shape[0] - 1),
        'ortho_off_diag_loss': jnp.mean(ortho_off_diag_loss),
        'occ_measure_mean': occ_measures.mean(),
        'occ_measure_max': occ_measures.max(),
        'occ_measure_min': occ_measures.min(),
    }

    return total_loss, metrics

def actor_loss(actor_params, training_state, actor, forward_repr, parametric_action_distribution, transitions, state_dim, goal_dim, repr_dim, key, entropy_coef=0.1):
    """Compute the FB-style actor loss (not IQL)."""
    states = transitions.observation[:, :state_dim]
    goals = transitions.observation[:, state_dim:]

    # Sample actions from the actor
    action_mean_and_SD = actor.apply(actor_params, jnp.concatenate([states, goals], axis=-1))
    actions = parametric_action_distribution.sample(action_mean_and_SD, key)

    # Compute forward representations for these actions
    # Assume forward_repr returns (F1, F2)
    F1= forward_repr.apply(training_state.fb_repr_state.params[0], jnp.concatenate([states, actions, goals], axis=-1))

    # Q-values as dot product between F and z (goals)
    Q1 = jnp.einsum('sd,sd->s', F1, goals)
    # Q2 = jnp.einsum('sd,sd->s', F2, goals)
    # Q = jnp.minimum(Q1, Q2)
    Q = Q1

    # Actor loss: negative minimum Q-value
    actor_loss = -Q

    # Entropy regularization if Gaussian
    log_prob = parametric_action_distribution.log_prob(action_mean_and_SD, actions)
    actor_loss = actor_loss + entropy_coef * log_prob
    mean_log_prob = log_prob.mean()

    actor_loss = actor_loss.mean()

    metrics = {
        'actor_loss': actor_loss,
        'actor_Q': Q.mean(),
        'actor_log_prob': mean_log_prob,
    }
    return actor_loss, metrics

def critic_loss(critic_params, value_params, training_state, critic, value, forward_repr, backward_repr, parametric_action_distribution, transitions, state_dim, goal_dim, repr_dim, key, discount=0.99):
    """Compute the IQL critic loss (matching fb_repr.py logic, using constant discount)."""
    states = transitions.observation[:, :state_dim]
    actions = transitions.action
    goals = transitions.observation[:, state_dim:]
    next_states = transitions.extras["next_state"][:, :state_dim]
    rewards = transitions.reward

    # Compute next_v using value network
    next_v = value.apply(value_params, jnp.concatenate([next_states, goals], axis=-1))

    # Get q1, q2 from critic
    q1 = critic.apply(critic_params, jnp.concatenate([states, actions, goals], axis=-1))

    # Compute target q using the provided discount constant
    q = rewards + discount * next_v

    # Compute critic loss as mean squared error for both q1 and q2
    critic_loss = ((q1 - q) ** 2).mean()

    metrics = {
        'critic_loss': critic_loss,
        'q_mean': q.mean(),
        'q_max': q.max(),
        'q_min': q.min(),
    }
    return critic_loss, metrics

def value_loss(value_params, training_state, value, critic, parametric_action_distribution, transitions, state_dim, goal_dim, repr_dim, key, expectile=0.9):
    """Compute the IQL value loss (matching fb_repr.py logic)."""
    # Unpack states, actions, goals
    states = transitions.observation[:, :state_dim]
    goals = transitions.observation[:, state_dim:]
    actions = transitions.action

    # Compute Q-values from target critic (q1, q2)
    q= critic.apply(training_state.target_critic_params, jnp.concatenate([states, actions, goals], axis=-1))

    # Compute value estimates
    v = value.apply(value_params, jnp.concatenate([states, goals], axis=-1))

    # Expectile loss (as in fb_repr.py)
    diff = q - v
    weight = jnp.where(diff >= 0, expectile, 1 - expectile)
    value_loss = jnp.mean(weight * (diff ** 2))

    metrics = {
        'value_loss': value_loss,
        'v_mean': v.mean(),
        'v_max': v.max(),
        'v_min': v.min(),
    }
    return value_loss, metrics


def actor_step_latent(env, env_state, actor, parametric_action_distribution, actor_params, explore_goal, key, extra_fields=()):
    """
    Executes one step of an actor in the environment by selecting an action based on the
    policy, stepping the environment, and returning the updated state and transition data.

    Parameters
    ----------
    env : Env
        The environment in which the actor operates.
    env_state : State
        The current state of the environment.
    actor : brax.training.types.Policy
        The policy used to select the action.
    parametric_action_distribution : brax.training.distribution.ParametricDistribution
        A tanh normal distribution, used to map the actor's output to an action vector with elements between [-1, 1].
    actor_params : Any
        Parameters for the actor network.
    key : PRNGKey
        A random key for stochastic policy decisions.
    extra_fields : Sequence[str], optional
        A sequence of extra fields to be extracted from the environment state.

    Returns
    -------
    Tuple[State, Transition]
        A tuple containing the new state after taking the action and the transition data
        encompassing observation, action, reward, discount, and extra information.

    """
    new_state = jnp.concatenate([env_state.obs[:, :env.state_dim], explore_goal], axis=1)
    action_mean_and_SD = actor.apply(actor_params, new_state)
    action = parametric_action_distribution.sample(action_mean_and_SD, key)
    # env_state.obs = new_state  # This doesn't work with JAX immutability
    env_state = env_state.replace(obs=new_state)
    nstate = env.step(env_state, action)
    # print(f"state.pipeline_state shape: {jax.tree_map(lambda x: x.shape, env_state.pipeline_state)}")
    # print(f"action shape: {jax.tree_map(lambda x: x.shape, action)}")
    state_extras = {x: nstate.info[x] for x in extra_fields}
    return nstate, Transition(
        observation=new_state,
        action=action,
        reward=nstate.reward,
        discount=1 - nstate.done,
        extras={"policy_extras": {}, "state_extras": state_extras},
    )
    
def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)

def sample_latents(batch_size, latent_dim, key):
    """Sample latents by generating random latents only (no mixing with backward representations)."""
    latent_rng = key
    # Random latents
    latents = jax.random.normal(latent_rng, shape=(batch_size, latent_dim))
    latents = latents / jnp.linalg.norm(latents, axis=-1, keepdims=True) * jnp.sqrt(latent_dim)
    return latents

def fb_repr_loss_fn(
    params, forward_repr, backward_repr, transitions, state_dim, goal_dim, repr_dim, key,
    training_state, actor, const_std, repr_agg, orthonorm_coef, discount
):
    forward_params, backward_params = params
    loss, metrics = forward_backward_repr_loss(
        forward_params, backward_params, forward_repr, backward_repr, transitions, state_dim, goal_dim, repr_dim, key, 
        training_state, actor=actor,
        const_std=const_std,
        repr_agg=repr_agg,
        orthonorm_coef=orthonorm_coef,
        discount=discount
    )
    return loss, metrics

def train(
    environment: envs.Env,
    num_timesteps,
    episode_length: int,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 128,
    policy_lr: float = 1e-4,
    repr_lr: float = 1e-4,
    seed: int = 0,
    batch_size: int = 256,
    num_evals: int = 1,
    min_replay_size: int = 0,
    max_replay_size: Optional[int] = None,
    deterministic_eval: bool = False,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    checkpoint_logdir: Optional[str] = None,
    eval_env: Optional[envs.Env] = None,
    unroll_length: int = 50,
    train_step_multiplier: int = 1,
    config: NamedTuple = None,
    use_ln: bool = False,
    h_dim: int = 256,
    n_hidden: int = 2,
    repr_dim: int = 64,
    tau: float = 0.005,
):
    """
    Trains a forward-backward representation learning agent.
    """

    # Reproducibility preparation for (optional) multi-GPU training
    process_id = jax.process_index()
    num_local_devices_to_use = jax.local_device_count()
    device_count = num_local_devices_to_use * jax.process_count()
    logging.info(
        "local_device_count: %s; total_device_count: %s",
        num_local_devices_to_use,
        device_count,
    )

    # Sanity checks
    if min_replay_size >= num_timesteps:
        raise ValueError("No training will happen because min_replay_size >= num_timesteps")

    if ((episode_length - 1) * num_envs) % batch_size != 0:
        raise ValueError("(episode_length - 1) * num_envs must be divisible by batch_size")

    if max_replay_size is None:
        max_replay_size = num_timesteps

    # The number of environment steps executed for every `actor_step()` call.
    env_steps_per_actor_step = action_repeat * num_envs * unroll_length
    num_prefill_actor_steps = min_replay_size // unroll_length + 1
    num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
    assert num_timesteps - min_replay_size >= 0
    num_evals_after_init = max(num_evals - 1, 1)
    num_training_steps_per_epoch = -(
        -(num_timesteps - num_prefill_env_steps) // (num_evals_after_init * env_steps_per_actor_step)
    )

    assert num_envs % device_count == 0
    env = environment
    wrap_for_training = envs.training.wrap

    rng = jax.random.PRNGKey(seed)
    rng, key = jax.random.split(rng)
    env = TrajectoryIdWrapper(env)
    env = wrap_for_training(env, episode_length=episode_length, action_repeat=action_repeat)
    unwrapped_env = environment
    env_train_context = wrap_for_training(environment, episode_length=episode_length, action_repeat=action_repeat)

    obs_size = env.observation_size
    action_size = env.action_size
    state_dim = env.state_dim
    goal_dim = obs_size - state_dim
    
    dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))
    dummy_extras = {"state_extras": {"truncation": 0.0, "traj_id": 0.0}, "policy_extras": {}}
    dummy_transition = Transition(observation=dummy_obs, action=dummy_action, reward=0.0, discount=0.0, extras=dummy_extras)
    
    replay_buffer = TrajectoryUniformSamplingQueue(
        max_replay_size=max_replay_size // device_count,
        dummy_data_sample=dummy_transition,
        sample_batch_size=batch_size // device_count,
        num_envs=num_envs,
        episode_length=episode_length,
    )
    replay_buffer = jit_wrap(replay_buffer)
    
    # Network functions
    block_size = 2
    num_blocks = max(1, n_hidden // block_size)
    actor = Net(action_size * 2, h_dim, num_blocks, block_size, use_ln)
    # critic = Net(1, h_dim, num_blocks, block_size, use_ln)  # Outputs a single Q-value
    # value = Net(1, h_dim, num_blocks, block_size, use_ln)   # Outputs a single V-value
    forward_repr = Net(goal_dim, h_dim, num_blocks, block_size, use_ln)
    backward_repr = Net(goal_dim, h_dim, num_blocks, block_size, use_ln)
    parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size)

    # Initialize training state
    global_key, local_key = jax.random.split(rng)
    local_key = jax.random.fold_in(local_key, process_id)    
    training_state = _init_training_state(global_key, actor, forward_repr, backward_repr, state_dim, len(env.goal_indices), env.action_size, repr_dim, episode_length, policy_lr, repr_lr, num_local_devices_to_use)
    del global_key
    
    # Update functions
    
    actor_update = gradients.gradient_update_fn(actor_loss, training_state.actor_state.tx, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)
    # critic_update = gradients.gradient_update_fn(critic_loss, training_state.critic_state.tx, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)
    # value_update = gradients.gradient_update_fn(value_loss, training_state.value_state.tx, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)
    # Joint gradient update for forward and backward representations
    fb_repr_update = gradients.gradient_update_fn(fb_repr_loss_fn, training_state.fb_repr_state.tx, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)
    
    def update_step(carry, transitions):
        training_state, key = carry
        key, key_fb, key_actor, key_critic, key_value = jax.random.split(key, 5)
        
        # Joint update for forward and backward representations
        (fb_loss, fb_metrics), fb_repr_params, fb_repr_opt_state = fb_repr_update(
            training_state.fb_repr_state.params,
            forward_repr,
            backward_repr,
            transitions,
            env.state_dim,
            len(env.goal_indices),
            repr_dim,
            key_fb,
            training_state,
            actor,
            True,
            'mean',
            1.0,
            0.99,
            optimizer_state=training_state.fb_repr_state.opt_state,
        )
        
        # # Update value function
        # (value_loss, value_metrics), value_params, value_optimizer_state = value_update(
        #     training_state.value_state.params,
        #     training_state,
        #     value,
        #     critic,
        #     parametric_action_distribution,
        #     transitions,
        #     env.state_dim,
        #     len(env.goal_indices),
        #     repr_dim,
        #     key_value,
        #     0.9,
        #     optimizer_state=training_state.value_state.opt_state
        # )
        
        # # Update critic
        # (critic_loss, critic_metrics), critic_params, critic_optimizer_state = critic_update(
        #     training_state.critic_state.params,
        #     value_params,  # Pass current value params for target computation
        #     training_state,
        #     critic,
        #     value,
        #     forward_repr,
        #     backward_repr,
        #     parametric_action_distribution,
        #     transitions,
        #     env.state_dim,
        #     len(env.goal_indices),
        #     repr_dim,
        #     key_critic,
        #     0.99,
        #     optimizer_state=training_state.critic_state.opt_state
        # )
        
        # Update actor
        (actor_loss, actor_metrics), actor_params, actor_optimizer_state = actor_update(
            training_state.actor_state.params,
            training_state,
            actor,
            forward_repr,
            parametric_action_distribution,
            transitions,
            env.state_dim,
            len(env.goal_indices),
            repr_dim,
            key_actor,
            0.1,
            optimizer_state=training_state.actor_state.opt_state
        )
        
        # Update target networks
        # new_target_critic_params = jax.tree_util.tree_map(
        #     lambda p, tp: p * tau + tp * (1 - tau),
        #     critic_params,
        #     training_state.target_critic_params
        # )
        new_target_forward_repr_params = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            fb_repr_params[0],
            training_state.target_forward_repr_params
        )
        new_target_backward_repr_params = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            fb_repr_params[1],
            training_state.target_backward_repr_params
        )

        metrics = {
            'fb_loss': fb_loss,
            'actor_loss': actor_loss,
            # 'critic_loss': critic_loss,
            # 'value_loss': value_loss,
        }
        metrics.update(fb_metrics)
        metrics.update(actor_metrics)
        # metrics.update(critic_metrics)
        # metrics.update(value_metrics)

        new_training_state = TrainingState(
            env_steps=training_state.env_steps,
            gradient_steps=training_state.gradient_steps + 1,
            actor_state=training_state.actor_state.replace(params=actor_params, opt_state=actor_optimizer_state),
            # critic_state=training_state.critic_state.replace(params=critic_params, opt_state=critic_optimizer_state),
            # value_state=training_state.value_state.replace(params=value_params, opt_state=value_optimizer_state),
            fb_repr_state=training_state.fb_repr_state.replace(params=fb_repr_params, opt_state=fb_repr_opt_state),
            # target_critic_params=new_target_critic_params,
            target_forward_repr_params=new_target_forward_repr_params,
            target_backward_repr_params=new_target_backward_repr_params
        )
        
        return (new_training_state, key), metrics

    def get_experience(actor_params, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused_t):
            env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            env_state, transition = actor_step_latent(env, env_state, actor, parametric_action_distribution, actor_params, exploration_goals, current_key, extra_fields=("truncation", "traj_id"))
            return (env_state, next_key), transition
        
       # Split the key to create a batch of keys matching env_state.obs.shape[0]
        print("fb: env_state.obs.shape", env_state.obs.shape)
        batch_size = env_state.obs.shape[0]
        subkey, sampling_key = jax.random.split(key)
        exploration_goals = sample_latents(batch_size, goal_dim, sampling_key)
        print("fb: exploration_goals shape", exploration_goals.shape)
        (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=episode_length)
        buffer_state = replay_buffer.insert(buffer_state, data)
        return env_state, buffer_state

    def training_step(training_state, env_state, buffer_state, key):
        # Collect experience
        experience_key, training_key = jax.random.split(key, 2)
        env_state, buffer_state = get_experience(training_state.actor_state.params, env_state, buffer_state, experience_key)
        training_state = training_state.replace(env_steps=training_state.env_steps + env_steps_per_actor_step)
        
        # Train
        training_state, buffer_state, metrics = train_steps(training_state, buffer_state, training_key)
        return training_state, env_state, buffer_state, metrics

    def prefill_replay_buffer(training_state, env_state, buffer_state, key):
        def f(carry, unused):
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            env_state, buffer_state = get_experience(training_state.actor_state.params, env_state, buffer_state, key)
            new_training_state = training_state.replace(env_steps=training_state.env_steps + env_steps_per_actor_step)
            return (new_training_state, env_state, buffer_state, new_key), ()
        return jax.lax.scan(f, (training_state, env_state, buffer_state, key), (), length=num_prefill_actor_steps)[0]
    
    prefill_replay_buffer = jax.pmap(prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME)

    def train_steps(training_state, buffer_state, key):
        # Sample from buffer
        experience_key, training_key, sampling_key = jax.random.split(key, 3)
        buffer_state, transitions = replay_buffer.sample(buffer_state)
        
        # Process transitions using flatten_crl_fn (vmap)
        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        vmap_flatten_crl_fn = jax.vmap(TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, None, 0, 0))
        transitions = vmap_flatten_crl_fn(config, env, transitions, batch_keys)
        
        # Shuffle and reshape transitions
        transitions = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions)
        permutation = jax.random.permutation(experience_key, len(transitions.observation))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
        transitions = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1, batch_size) + x.shape[1:]), transitions)
        
        # Train
        (training_state, _), metrics = jax.lax.scan(update_step, (training_state, training_key), transitions)
        return training_state, buffer_state, metrics

    def training_epoch(training_state, env_state, buffer_state, key):
        def f(carry, unused_t):
            ts, es, bs, k = carry
            k, new_key = jax.random.split(k)
            ts, es, bs, metrics = training_step(ts, es, bs, k)
            return (ts, es, bs, new_key), metrics
        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(f, (training_state, env_state, buffer_state, key), (), length=num_training_steps_per_epoch)
        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, env_state, buffer_state, metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

    def training_epoch_with_timing(training_state, env_state, buffer_state, key):
        nonlocal training_walltime
        t = time.time()
        (training_state, env_state, buffer_state, metrics) = training_epoch(training_state, env_state, buffer_state, key)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return (training_state, env_state, buffer_state, metrics)

    # Initialization and setup
    local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)
    env_keys = jax.random.split(env_key, num_envs // jax.process_count())
    env_keys = jnp.reshape(env_keys, (num_local_devices_to_use, -1) + env_keys.shape[1:])
    env_state = jax.pmap(env.reset)(env_keys)

    # Replay buffer init and prefill
    buffer_state = jax.pmap(replay_buffer.init)(jax.random.split(rb_key, num_local_devices_to_use))
    t = time.time()
    prefill_key, local_key = jax.random.split(local_key)
    prefill_keys = jax.random.split(prefill_key, num_local_devices_to_use)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(training_state, env_state, buffer_state, prefill_keys)
    replay_size = jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
    assert replay_size >= min_replay_size
    training_walltime = time.time() - t

    # Eval init
    global make_policy
    make_policy = functools.partial(make_policy, actor, parametric_action_distribution)
    if not eval_env:
        eval_env = environment
    eval_env = wrap_for_training(eval_env, episode_length=episode_length, action_repeat=action_repeat)
    evaluator = CrlEvaluator(eval_env, functools.partial(make_policy, deterministic=deterministic_eval), num_eval_envs=num_eval_envs,
                             episode_length=episode_length, action_repeat=action_repeat, key=eval_key)

    # Run initial eval
    metrics = {}
    if process_id == 0 and num_evals > 1:
        metrics = evaluator.run_evaluation(_unpmap(training_state.actor_state.params), training_metrics={})
        logging.info(metrics)
        progress_fn(0, metrics, make_policy, _unpmap(training_state.actor_state.params), unwrapped_env)

    # Collect/train/eval loop
    current_step = 0
    for eval_epoch_num in range(num_evals_after_init):
        logging.info("step %s", current_step)

        # Collect data and train
        epoch_key, local_key = jax.random.split(local_key)
        epoch_keys = jax.random.split(epoch_key, num_local_devices_to_use)
        (training_state, env_state, buffer_state, training_metrics) = training_epoch_with_timing(training_state, env_state, buffer_state, epoch_keys)
        current_step = int(_unpmap(training_state.env_steps))

        # Logging and evals
        if process_id == 0:
            ## Save policy and representation params
            if checkpoint_logdir:
                params = _unpmap((
                    training_state.actor_state.params,
                    # training_state.value_state.params,
                    # training_state.critic_state.params,
                    training_state.fb_repr_state.params,
                    # training_state.target_critic_params,
                    training_state.target_forward_repr_params,
                    training_state.target_backward_repr_params
                ))
                path = f"{checkpoint_logdir}/step_{current_step}.pkl"
                # Log all params
                logging.info(f"Saving checkpoint at {path} with actor, fb_repr, and target params.")
                brax.io.model.save_params(path, params)

            ## Run evals
            metrics = evaluator.run_evaluation(_unpmap(training_state.actor_state.params), training_metrics)
            logging.info(metrics)
            progress_fn(current_step, metrics, make_policy, _unpmap(training_state.actor_state.params), unwrapped_env)

    # Final validity checks
    total_steps = current_step
    logging.info("total steps: %s", total_steps)
    assert total_steps >= num_timesteps

    pmap.assert_is_replicated(training_state)
    pmap.synchronize_hosts()
    
    params = _unpmap((
        training_state.actor_state.params,
        # training_state.value_state.params,
        # training_state.critic_state.params,
        training_state.fb_repr_state.params,
        # training_state.target_critic_params,
        training_state.target_forward_repr_params,
        training_state.target_backward_repr_params
    ))
    # Log all params at the end as well
    logging.info("Returning actor, fb_repr, and target params.")
    return (make_policy, params, metrics)