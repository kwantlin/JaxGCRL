from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import mujoco
import jax
from jax import numpy as jnp


# Observation space: 28 dims for the actual observation, 3 dims for goal position. See _get_obs() for details.
# Action space:      7 dimensional, [-1, 1] each, corresponding to joint target angles (each joint has a different range, see convert_action).
class ArmReach(PipelineEnv):
    def __init__(self, backend="mjx", **kwargs):
        # Load XML and manage simulation parameters
        if backend == "mjx":
            sys = mjcf.load("envs/assets/panda_reach_mjx.xml")
            sys = sys.tree_replace({
                "opt.timestep": 0.01,
                "opt.iterations": 4,
                "opt.ls_iterations": 8,
            })
            self.n_frames = 4
            self.episode_length = 100 # Merely a recommendation + used for postexplore timestep
        elif backend == "positional":
            sys = mjcf.load("envs/assets/panda_reach_positional.xml")
            sys = sys.tree_replace({
                "opt.timestep": 0.0003,
                "opt.iterations": 50,
                "opt.ls_iterations": 100,
            })
            self.n_frames = 10
            self.episode_length = 1000 # Merely a recommendation + used for postexplore timestep
        else:
            raise Exception("Please use mjx or positional backends for better speed/stability tradeoffs.")
            
        kwargs["n_frames"] = kwargs.get("n_frames", self.n_frames)
        super().__init__(sys=sys, backend=backend, **kwargs)
        
        # Set additional configuration information
        self.goal_indices = jnp.array([15, 16, 17]) # For data collection/goal-conditioning
        self.completion_goal_indices = jnp.array([15, 16, 17]) # For reward/checking completion of goal (identical in this case)
        self.obs_dim = 28
        
        self.arm_noise_scale = 1
        self.goal_noise_scale = 0.5
        
    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        # Variance in starting arm state and goal state to avoid overfitting/closed loop control
        rng, subkey = jax.random.split(rng)
        q = self.sys.init_q + self.arm_noise_scale * jax.random.uniform(subkey, [self.sys.q_size()])
        qd = jnp.zeros([self.sys.qd_size()])

        # Initialize state
        pipeline_state = self.pipeline_init(q, qd)
        timestep = 0.0

        # Define goal distribution
        rng, subkey = jax.random.split(rng)
        goal = jnp.array([0, 1, 0.5]) + self.goal_noise_scale * jax.random.uniform(subkey, [3])
                              
        # Get other components for state (obs, reward, etc.)
        obs = self._get_obs(pipeline_state, goal, timestep)
        reward, done, zero = jnp.zeros(3)
        metrics = {"success": zero, "success_easy": zero, "success_hard": zero}
        
        # Fill info
        rng, subkey = jax.random.split(rng)
        info = {
            "seed": 0, # Seed is required, but fill it with a dummy value
            "goal": goal, 
            "timestep": 0.0, 
            "postexplore_timestep": self.episode_length * jax.random.uniform(subkey)
        } 
        
        return State(pipeline_state, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, self.convert_action(action))
        timestep = state.info["timestep"] + 1
        
        obs = self._get_obs(pipeline_state, state.info["goal"], timestep)
        done = 0.0
        
        eef_pos = obs[self.completion_goal_indices]
        goal_info = state.info["goal"]
        goal_eef_pos = goal_info[:3]
        dist = jnp.linalg.norm(eef_pos - goal_eef_pos)
        reward = jnp.array(dist < 0.1, dtype=float)
        info = {**state.info, "timestep": timestep}
        
        state.metrics.update(success=reward, success_easy=jnp.array(dist < 0.25, dtype=float), success_hard=jnp.array(dist < 0.03, dtype=float))
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, info=info)
    
    # We deviate from the existing interface by allowing the goal to be modified directly,
    # since this is useful for exploration during trajectory collection.
    #
    # Additionally, the goal is now virtual, and is no longer a physical object in simulation. While 
    # a physical representation better for visualization of positional goals, many goals are not only
    # positions. It could include orientation, velocity, timestep, gripper open-ness, etc. which cannot be 
    # represented naturally with an object.
    def update_goal(self, state: State, goal: jax.Array) -> State:
        updated_info = {**info, "goal": goal} # Dictionary unpacking to return updated dict
        return state.replace(info=updated_info)
    
    # Converts the [-1, 1] actions to the corresponding target angle or strength.
    # We use the exact numbers for radians specified in the XML, even if they might be cleaner in terms of pi.
    def convert_action(self, action: jax.Array) -> jax.Array:
        # Flip the gripper action values to make -1 open, 1 closed.
        min_value = jnp.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        max_value = jnp.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        
        # This offset and multiplier yields f(-1) = min_value, f(1) = max_value
        offset = (min_value + max_value) / 2
        multiplier = (max_value - min_value) / 2
        converted_action = offset + action * multiplier 
        return converted_action

    # 28 dimensional observation space (NOT including goal -- hence, config.obs_dim = 28, but the returned observation has dim 28 + goal dims):
    # - 7 for q (7 joint angles)
    # - 7 for qd (7 joint angular velocities)
    # - 1 for timestep
    # - 7 for position and quaternion of end-effector
    # - 6 for velocity and angular velocity of end-effector
    #
    # Goal space: position of link 7/end-effector
    def _get_obs(self, pipeline_state: base.State, goal: jax.Array, timestep) -> jax.Array:
        q = pipeline_state.q
        qd = pipeline_state.qd
        t = jnp.array([timestep])
        
        eef_index = 6 # Links 1-7 are indices 0-6. The end-effector (eef) base is merged with link 7, so we say link 7 index = eef index.
        eef_x_pos = pipeline_state.x.pos[eef_index]
        eef_x_rot = pipeline_state.x.rot[eef_index]
        eef_xd_vel = pipeline_state.xd.vel[eef_index]
        eef_xd_angvel = pipeline_state.xd.ang[eef_index]
        
        return jnp.concatenate([q] + [qd] + [t] + [eef_x_pos] + [eef_x_rot] + [eef_xd_vel] + [eef_xd_angvel] + [goal])