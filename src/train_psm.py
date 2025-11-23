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
def make_policy(actor, parametric_action_distribution, params, state_dim, deterministic=False):
    actor_params = params
    def policy(obs, key_sample):
        # For PSM, actor takes (state, goal) directly
        logits = actor.apply(actor_params, obs)
        if deterministic:
            action = parametric_action_distribution.mode(logits)
        else:
            action = parametric_action_distribution.sample(logits, key_sample)
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
    psm_repr_state: TrainState  # Contains (psm_params, w_params)
    # target_critic_params: flax.core.FrozenDict
    target_psm_params: flax.core.FrozenDict
    target_w_params: flax.core.FrozenDict
    w_inf: jnp.ndarray  # Optimized w for goal inference (d_dim,)
    lmult_state: TrainState  # Lagrange multiplier network state

def _init_training_state(key, actor, psm_repr, w_net, lmult_net, state_dim, goal_dim, action_dim, d_dim, z_dim, episode_length, actor_lr, repr_lr, inf_lr, num_local_devices_to_use):
    """
    Initializes the training state for a PSM representation learning model.
    """
    actor_key, psm_key, w_key, lmult_key = jax.random.split(key, 4)
    
    # Actor
    actor_params = actor.init(actor_key, jnp.ones([1, state_dim + goal_dim]))
    actor_state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=optax.adam(learning_rate=actor_lr))

    # PSM network: takes (obs, goal, action) -> (phi: d_dim, b: scalar)
    psm_params = psm_repr.init(psm_key, jnp.ones([1, state_dim + goal_dim + action_dim]))
    # W network: takes z (z_dim binary) -> d_dim
    w_params = w_net.init(w_key, jnp.ones([1, z_dim]))
    
    # Single optimizer for both PSM and w networks
    repr_optimizer = optax.adam(learning_rate=repr_lr)
    psm_repr_state = TrainState.create(apply_fn=None, params=(psm_params, w_params), tx=repr_optimizer)

    # Target networks: just store params
    target_psm_params = jax.tree_util.tree_map(lambda x: x.copy(), psm_params)
    target_w_params = jax.tree_util.tree_map(lambda x: x.copy(), w_params)
    
    # Initialize w_inf randomly (will be optimized per goal)
    w_inf = jax.random.normal(lmult_key, shape=(d_dim,)) * 0.01
    w_inf = w_inf / jnp.linalg.norm(w_inf) * jnp.sqrt(d_dim)
    
    # Lagrange multiplier network: takes (obs, perm_obs) -> action_dim (softmax)
    lmult_params = lmult_net.init(lmult_key, jnp.ones([1, state_dim + goal_dim + state_dim + goal_dim]))
    lmult_state = TrainState.create(apply_fn=lmult_net.apply, params=lmult_params, tx=optax.adam(learning_rate=inf_lr))

    training_state = TrainingState(
        env_steps=jnp.zeros(()), 
        gradient_steps=jnp.zeros(()), 
        actor_state=actor_state,
        psm_repr_state=psm_repr_state,
        target_psm_params=target_psm_params,
        target_w_params=target_w_params,
        w_inf=w_inf,
        lmult_state=lmult_state
    )
    
    training_state = jax.device_put_replicated(training_state, jax.local_devices()[:num_local_devices_to_use])
    return training_state


def psm_repr_loss(
    psm_params, w_params, psm_repr, w_net, transitions, state_dim, goal_dim, goal_indices, d_dim, z_dim, key,
    training_state, actor, parametric_action_distribution
):
    """
    Compute the PSM representation loss, following reference_psm.py structure.
    """
    states = transitions.observation[:, :state_dim]
    actions = transitions.action
    next_states = transitions.extras["next_state"][:, :state_dim]
    goals = transitions.observation[:, state_dim:]
    discounts = transitions.discount
    batch_size = states.shape[0]
    
    # Sample binary z latents
    key, key_z = jax.random.split(key)
    z = sample_latents(batch_size, z_dim, key_z)
    # Repeat z for mesh grid: each z gets paired with batch_size goals
    z = jnp.repeat(z, batch_size, axis=0)  # (batch_size^2, z_dim)
    
    # Create mesh grid of all (obs_i, goal_j) pairs
    idx = jnp.arange(batch_size)
    mesh_i, mesh_j = jnp.meshgrid(idx, idx, indexing='ij')
    mesh_i = mesh_i.reshape(-1)  # (batch_size^2,)
    mesh_j = mesh_j.reshape(-1)  # (batch_size^2,)
    
    m_obs = states[mesh_i]  # (batch_size^2, state_dim)
    m_next_obs = next_states[mesh_i]  # (batch_size^2, state_dim)
    m_action = actions[mesh_i]  # (batch_size^2, action_dim)
    m_next_goal = goals[mesh_j]  # (batch_size^2, goal_dim)
    m_discount = discounts[mesh_i]  # (batch_size^2,)
    
    # Compute target successor measure
    # Sample next actions using actor
    next_dist = actor.apply(jax.lax.stop_gradient(training_state.actor_state.params), 
                            jnp.concatenate([m_next_obs, m_next_goal], axis=-1))
    key, subkey = jax.random.split(key)
    next_actions = parametric_action_distribution.sample(next_dist, subkey)
    
    # Compute target PSM: phi(next_obs, next_goal, next_action) and b
    target_psm_input = jnp.concatenate([m_next_obs, m_next_goal, next_actions], axis=-1)
    target_psm_output = psm_repr.apply(training_state.target_psm_params, target_psm_input)
    target_phi, target_b = jnp.split(target_psm_output, [d_dim], axis=-1)
    target_b = target_b.squeeze(-1)  # (batch_size^2,)
    
    # Compute target w
    target_w = w_net.apply(training_state.target_w_params, z)  # (batch_size^2, d_dim)
    
    # Target M = phi @ w + b
    target_M = jnp.einsum('sd,sd->s', target_phi, target_w) + target_b  # (batch_size^2,)
    target_M = target_M.reshape(batch_size, batch_size)
    
    # Compute current PSM: phi(obs, goal, action) and b
    psm_input = jnp.concatenate([m_obs, m_next_goal, m_action], axis=-1)
    psm_output = psm_repr.apply(psm_params, psm_input)
    phi, b = jnp.split(psm_output, [d_dim], axis=-1)
    b = b.squeeze(-1)  # (batch_size^2,)
    
    # Compute current w
    w = w_net.apply(w_params, z)  # (batch_size^2, d_dim)
    
    # Current M = phi @ w + b
    M = jnp.einsum('sd,sd->s', phi, w) + b  # (batch_size^2,)
    M = M.reshape(batch_size, batch_size)
    
    # PSM loss
    I = jnp.eye(batch_size)
    off_diag_mask = 1 - I
    # Reshape discount for mesh grid - use discount from the observation (mesh_i)
    m_discount_reshaped = m_discount.reshape(batch_size, batch_size)
    psm_offdiag = 0.5 * jnp.mean(((M - m_discount_reshaped * target_M) * off_diag_mask) ** 2)
    psm_diag = -jnp.mean((1 - jnp.diag(m_discount_reshaped))) * jnp.mean(jnp.diag(M))
    psm_loss = psm_offdiag + psm_diag
    
    metrics = {
        'psm_loss': psm_loss,
        'psm_diag': psm_diag,
        'psm_offdiag': psm_offdiag,
        'M_mean': jnp.mean(M),
        'M_max': jnp.max(M),
        'M_min': jnp.min(M),
    }
    
    return psm_loss, metrics

def _infer_step(psm_params, w_inf, lmult_params, psm_repr, lmult_net, actor_params, actor, parametric_action_distribution, obs, goal, perm_obs, state_dim, goal_dim, action_dim, d_dim, key):
    """
    Single step of w_inf optimization using Dual Gradient Descent.
    For continuous actions, we sample actions from the actor for both target and permuted goals.
    """
    # Normalize w_inf
    w_inf_norm = w_inf / jnp.linalg.norm(w_inf) * jnp.sqrt(d_dim)
    
    # Sample actions for target goal using actor
    key, key_g = jax.random.split(key)
    obs_goal_g = jnp.concatenate([obs, goal], axis=-1)
    action_dist_g = actor.apply(actor_params, obs_goal_g)
    actions_g = parametric_action_distribution.sample(action_dist_g, key_g)
    
    # Compute phi for target goal
    psm_input_g = jnp.concatenate([obs, goal, actions_g], axis=-1)
    psm_output_g = psm_repr.apply(psm_params, psm_input_g)
    phi_g, b_g = jnp.split(psm_output_g, [d_dim], axis=-1)
    b_g = b_g.squeeze(-1)  # (batch_size,)
    
    # Objective: maximize mean Q for target goal (minimize negative)
    Q_g = jnp.einsum('sd,d->s', phi_g, w_inf_norm) + b_g
    obj = -jnp.mean(Q_g)
    
    # Sample actions for permuted goals
    key, key_perm = jax.random.split(key)
    obs_goal_perm = jnp.concatenate([obs, perm_obs], axis=-1)
    action_dist_perm = actor.apply(actor_params, obs_goal_perm)
    actions_perm = parametric_action_distribution.sample(action_dist_perm, key_perm)
    
    # Compute phi for permuted goals (constraints)
    psm_input_perm = jnp.concatenate([obs, perm_obs, actions_perm], axis=-1)
    psm_output_perm = psm_repr.apply(psm_params, psm_input_perm)
    phi_perm, b_perm = jnp.split(psm_output_perm, [d_dim], axis=-1)
    b_perm = b_perm.squeeze(-1)  # (batch_size,)
    
    # Compute constraint: phi_perm @ w_inf + b_perm >= 0
    constraint_values = jnp.einsum('sd,d->s', phi_perm, w_inf_norm) + b_perm
    
    # Lagrange multipliers: lmult takes (obs, perm_obs) -> action_dim (softmax)
    lmult_input = jnp.concatenate([obs, perm_obs], axis=-1)
    lmult_logits = lmult_net.apply(lmult_params, lmult_input)  # (batch_size, action_dim)
    lmult = nn.softmax(lmult_logits, axis=-1)  # (batch_size, action_dim)
    # Aggregate over actions (mean) to get per-sample multiplier
    lmult_agg = jnp.mean(lmult, axis=-1)  # (batch_size,)
    
    # Constraint term: penalize negative constraint values
    constraints = -jnp.mean(constraint_values * lmult_agg)
    
    # Total loss for w_inf
    loss_w = obj + constraints
    
    # Loss for Lagrange multipliers: maximize constraint violation (dual ascent)
    loss_lmult = jnp.mean(constraint_values * lmult_agg)
    
    metrics = {
        'inf_obj': obj,
        'inf_constraints': constraints,
        'inf_lamb': jnp.mean(lmult_agg),
        'inf_constraint_mean': jnp.mean(constraint_values),
    }
    
    return loss_w, loss_lmult, metrics

def infer_w_step(carry, unused_step):
    """Single step of w_inf inference optimization."""
    (w_inf, lmult_params, w_inf_opt_state, lmult_opt_state, w_inf_opt, lmult_opt, 
     psm_params, actor_params, psm_repr, lmult_net, actor, parametric_action_distribution, 
     replay_buffer, buffer_state, goal, state_dim, goal_dim, action_dim, d_dim, batch_size, key) = carry
    
    # Sample batch from replay buffer
    buffer_state, transitions = replay_buffer.sample(buffer_state)
    obs = transitions.observation[:, :state_dim]
    actual_batch_size = obs.shape[0]
    
    # Tile goal to match batch size
    goal_tiled = jnp.tile(goal[None, :], (actual_batch_size, 1))
    
    # Permute observations for constraints
    key, perm_key = jax.random.split(key)
    perm = jax.random.permutation(perm_key, actual_batch_size)
    perm_obs = transitions.observation[perm, :state_dim]
    
    # Compute gradients
    (loss_w, loss_lmult, metrics), (w_inf_grad, lmult_grad) = jax.value_and_grad(
        lambda w, l: _infer_step(psm_params, w, l, psm_repr, lmult_net, actor_params, actor, parametric_action_distribution, obs, goal_tiled, perm_obs, state_dim, goal_dim, action_dim, d_dim, key),
        argnums=(0, 1), has_aux=True
    )(w_inf, lmult_params)
    
    # Update w_inf
    w_inf_updates, w_inf_opt_state = w_inf_opt.update(w_inf_grad, w_inf_opt_state, w_inf)
    w_inf = optax.apply_updates(w_inf, w_inf_updates)
    
    # Update Lagrange multipliers (dual ascent - maximize)
    lmult_grad_neg = jax.tree_util.tree_map(lambda x: -x, lmult_grad)  # Negate for maximization
    lmult_updates, lmult_opt_state = lmult_opt.update(lmult_grad_neg, lmult_opt_state, lmult_params)
    lmult_params = optax.apply_updates(lmult_params, lmult_updates)
    
    key, _ = jax.random.split(key)
    return (w_inf, lmult_params, w_inf_opt_state, lmult_opt_state, w_inf_opt, lmult_opt,
            psm_params, actor_params, psm_repr, lmult_net, actor, parametric_action_distribution,
            replay_buffer, buffer_state, goal, state_dim, goal_dim, action_dim, d_dim, batch_size, key), metrics

def infer_w(psm_params, w_inf_init, lmult_params, actor_params, psm_repr, lmult_net, actor, parametric_action_distribution, replay_buffer, buffer_state, goal, state_dim, goal_dim, action_dim, d_dim, num_inference_steps, batch_size, key, inf_lr=1e-4):
    """
    Optimize w_inf for a specific goal using Dual Gradient Descent (DGD).
    
    This function should be called before evaluation or when a new goal is set.
    It optimizes w_inf to maximize Q-values for the target goal while ensuring
    the successor measure is non-negative for random goals (constraints).
    
    Args:
        psm_params: Parameters of the PSM network
        w_inf_init: Initial w_inf value (will be optimized)
        lmult_params: Parameters of the Lagrange multiplier network
        actor_params: Parameters of the actor network (for action sampling)
        psm_repr: PSM network module
        lmult_net: Lagrange multiplier network module
        actor: Actor network module
        parametric_action_distribution: Action distribution
        replay_buffer: Replay buffer for sampling states
        buffer_state: Current buffer state
        goal: Target goal to optimize for (goal_dim,)
        state_dim: Dimension of state
        goal_dim: Dimension of goal
        action_dim: Dimension of action
        d_dim: Dimension of d (same as repr_dim)
        num_inference_steps: Number of optimization steps
        batch_size: Batch size for sampling
        key: Random key
        inf_lr: Learning rate for inference optimization
        
    Returns:
        w_inf: Optimized w_inf for the goal
        metrics: Dictionary of inference metrics
        buffer_state: Updated buffer state
    """
    w_inf = w_inf_init
    lmult_params_current = lmult_params
    
    # Create optimizers
    w_inf_opt = optax.adam(learning_rate=inf_lr)
    lmult_opt = optax.adam(learning_rate=inf_lr)
    w_inf_opt_state = w_inf_opt.init(w_inf)
    lmult_opt_state = lmult_opt.init(lmult_params)
    
    # Run inference steps
    carry = (w_inf, lmult_params_current, w_inf_opt_state, lmult_opt_state, w_inf_opt, lmult_opt,
             psm_params, actor_params, psm_repr, lmult_net, actor, parametric_action_distribution,
             replay_buffer, buffer_state, goal, state_dim, goal_dim, action_dim, d_dim, batch_size, key)
    (w_inf, lmult_params, _, _, _, _, _, _, _, _, _, _, _, buffer_state, _, _, _, _, _, _), all_metrics = jax.lax.scan(
        infer_w_step, carry, None, length=num_inference_steps
    )
    
    # Final normalization
    w_inf = w_inf / jnp.linalg.norm(w_inf) * jnp.sqrt(d_dim)
    
    # Aggregate metrics
    metrics = {k: jnp.mean(v) for k, v in all_metrics.items()}
    
    return w_inf, metrics, buffer_state

def actor_loss(actor_params, training_state, actor, psm_repr, w_net, parametric_action_distribution, transitions, state_dim, goal_dim, d_dim, z_dim, key, entropy_coef=0.1):
    """Compute the PSM-style actor loss using optimized w_inf."""
    states = transitions.observation[:, :state_dim]
    goals = transitions.observation[:, state_dim:]
    
    # Sample actions from the actor
    action_mean_and_SD = actor.apply(actor_params, jnp.concatenate([states, goals], axis=-1))
    actions = parametric_action_distribution.sample(action_mean_and_SD, key)
    
    # Use optimized w_inf from training state
    w_inf = training_state.w_inf
    w_inf_norm = w_inf / jnp.linalg.norm(w_inf) * jnp.sqrt(d_dim)
    
    # Compute Q-function: phi(obs, goal, action) @ w_inf + b
    psm_input = jnp.concatenate([states, goals, actions], axis=-1)
    psm_output = psm_repr.apply(training_state.psm_repr_state.params[0], psm_input)
    phi, b = jnp.split(psm_output, [d_dim], axis=-1)
    b = b.squeeze(-1)  # (batch_size,)
    
    Q = jnp.einsum('sd,d->s', phi, w_inf_norm) + b
    
    # Actor loss: negative mean Q-value
    actor_loss = -Q.mean()
    
    # Entropy regularization
    log_prob = parametric_action_distribution.log_prob(action_mean_and_SD, actions)
    actor_loss = actor_loss + entropy_coef * log_prob.mean()
    
    metrics = {
        'actor_loss': actor_loss,
        'actor_Q': Q.mean(),
        'actor_log_prob': log_prob.mean(),
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


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)

def sample_latents(batch_size, z_dim, key):
    """Sample binary latents for PSM (z_dim bits)."""
    # Sample random integers and convert to binary
    z_ints = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=2**z_dim)
    # Convert to binary array: check each bit position
    bit_positions = jnp.arange(z_dim)
    binary_array = ((z_ints[:, None] & (1 << bit_positions)) > 0).astype(jnp.float32)
    return binary_array

def psm_repr_loss_fn(
    params, psm_repr, w_net, transitions, state_dim, goal_dim, goal_indices, d_dim, z_dim, key,
    training_state, actor, parametric_action_distribution
):
    psm_params, w_params = params
    loss, metrics = psm_repr_loss(
        psm_params, w_params, psm_repr, w_net, transitions, state_dim, goal_dim, goal_indices, d_dim, z_dim, key,
        training_state, actor=actor,
        parametric_action_distribution=parametric_action_distribution
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
    num_inference_steps: int = 100,  # Number of steps to optimize w_inf per goal
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
    
    dummy_obs = jnp.zeros((obs_size,))  # state_dim + goal_dim
    dummy_action = jnp.zeros((action_size,))
    dummy_extras = {"state_extras": {"truncation": 0.0, "traj_id": 0.0}, "policy_extras": {}, "next_state": jnp.zeros((state_dim,))}
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
    # PSM network: outputs phi (d_dim) and b (1) concatenated -> (d_dim + 1)
    psm_repr = Net(repr_dim + 1, h_dim, num_blocks, block_size, use_ln)  # d_dim = repr_dim
    # W network: maps z (z_dim) -> d_dim
    z_dim = 16  # Binary z dimension
    w_net = Net(repr_dim, h_dim, num_blocks, block_size, use_ln)  # d_dim = repr_dim
    # Lagrange multiplier network: takes (obs, perm_obs) -> action_dim (softmax output)
    lmult_net = Net(action_size, h_dim, num_blocks, block_size, use_ln)
    parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size)

    # Initialize training state
    global_key, local_key = jax.random.split(rng)
    local_key = jax.random.fold_in(local_key, process_id)
    inf_lr = repr_lr  # Use same learning rate for inference
    training_state = _init_training_state(global_key, actor, psm_repr, w_net, lmult_net, state_dim, len(env.goal_indices), env.action_size, repr_dim, z_dim, episode_length, policy_lr, repr_lr, inf_lr, num_local_devices_to_use)
    del global_key
    
    # Print parameter shapes
    psm_params = _unpmap(training_state.psm_repr_state.params)[0]
    w_params = _unpmap(training_state.psm_repr_state.params)[1]
    print("PSM network parameter shapes:", jax.tree_util.tree_map(lambda x: x.shape, psm_params))
    print("W network parameter shapes:", jax.tree_util.tree_map(lambda x: x.shape, w_params))
    
    # Update functions
    actor_update = gradients.gradient_update_fn(actor_loss, training_state.actor_state.tx, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)
    # Joint gradient update for PSM and w networks
    psm_repr_update = gradients.gradient_update_fn(psm_repr_loss_fn, training_state.psm_repr_state.tx, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)
    
    def update_step(carry, transitions):
        training_state, key = carry
        key, key_psm, key_actor = jax.random.split(key, 3)
        
        # Joint update for PSM and w networks
        (psm_loss, psm_metrics), psm_repr_params, psm_repr_opt_state = psm_repr_update(
            training_state.psm_repr_state.params,
            psm_repr,
            w_net,
            transitions,
            env.state_dim,
            len(env.goal_indices),
            env.goal_indices,
            repr_dim,  # d_dim
            z_dim,
            key_psm,
            training_state,
            actor,
            parametric_action_distribution,
            optimizer_state=training_state.psm_repr_state.opt_state,
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
        (actor_loss_val, actor_metrics), actor_params, actor_optimizer_state = actor_update(
            training_state.actor_state.params,
            training_state,
            actor,
            psm_repr,
            w_net,
            parametric_action_distribution,
            transitions,
            env.state_dim,
            len(env.goal_indices),
            repr_dim,  # d_dim
            z_dim,
            key_actor,
            optimizer_state=training_state.actor_state.opt_state
        )
        
        # Update target networks
        new_target_psm_params = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            psm_repr_params[0],
            training_state.target_psm_params
        )
        new_target_w_params = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            psm_repr_params[1],
            training_state.target_w_params
        )

        metrics = {
            'psm_loss': psm_loss,
            'actor_loss': actor_loss_val,
        }
        metrics.update(psm_metrics)
        metrics.update(actor_metrics)

        new_training_state = TrainingState(
            env_steps=training_state.env_steps,
            gradient_steps=training_state.gradient_steps + 1,
            actor_state=training_state.actor_state.replace(params=actor_params, opt_state=actor_optimizer_state),
            psm_repr_state=training_state.psm_repr_state.replace(params=psm_repr_params, opt_state=psm_repr_opt_state),
            target_psm_params=new_target_psm_params,
            target_w_params=new_target_w_params,
            w_inf=training_state.w_inf,  # Preserve w_inf (only updated during inference)
            lmult_state=training_state.lmult_state  # Preserve lmult_state (only updated during inference)
        )
        
        return (new_training_state, key), metrics

    def get_experience(actor_params, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused_t):
            env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            # For PSM, actor takes (state, goal) directly (no backward repr needed)
            policy_obs = env_state.obs
            action_mean_and_SD = actor.apply(actor_params, policy_obs)
            action = parametric_action_distribution.sample(action_mean_and_SD, current_key)
            nstate = env.step(env_state, action)
            state_extras = {"truncation": nstate.info.get("truncation", 0.0), "traj_id": nstate.info.get("traj_id", 0.0)}
            transition = Transition(
                observation=policy_obs,
                action=action,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"policy_extras": {}, "state_extras": state_extras, "next_state": nstate.obs[:, :state_dim]},
            )
            return (nstate, next_key), transition
        
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
    if not eval_env:
        eval_env = environment
    eval_env = TrajectoryIdWrapper(eval_env)
    eval_env = wrap_for_training(eval_env, episode_length=episode_length, action_repeat=action_repeat)
    global make_policy
    make_policy = functools.partial(
        make_policy,
        actor,
        parametric_action_distribution,
        state_dim=env.state_dim,
    )
    evaluator = CrlEvaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )

    # Run initial eval
    metrics = {}
    if process_id == 0 and num_evals > 1:
        # We pass in the actor params to the evaluator
        eval_params = _unpmap(training_state.actor_state.params)
        metrics = evaluator.run_evaluation(eval_params, training_metrics={})
        logging.info(metrics)
        progress_fn(0, metrics, make_policy, eval_params, unwrapped_env)

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
                    training_state.psm_repr_state.params,
                    training_state.target_psm_params,
                    training_state.target_w_params
                ))
                path = f"{checkpoint_logdir}/step_{current_step}.pkl"
                # Log all params
                logging.info(f"Saving checkpoint at {path} with actor, psm_repr, and target params.")
                brax.io.model.save_params(path, params)
            ## Run evals
            eval_params = _unpmap(training_state.actor_state.params)
            metrics = evaluator.run_evaluation(eval_params, training_metrics)
            logging.info(metrics)
            progress_fn(current_step, metrics, make_policy, eval_params, unwrapped_env)

    # Final validity checks
    total_steps = current_step
    logging.info("total steps: %s", total_steps)
    assert total_steps >= num_timesteps

    pmap.assert_is_replicated(training_state)
    pmap.synchronize_hosts()
    
    params = _unpmap((
        training_state.actor_state.params,
        training_state.psm_repr_state.params,
        training_state.target_psm_params,
        training_state.target_w_params
    ))
    # Log all params at the end as well
    logging.info("Returning actor, psm_repr, and target params.")
    return (make_policy, params, metrics)
