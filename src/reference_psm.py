import pdb
import copy
import math
import logging
import dataclasses
from collections import OrderedDict
import typing as tp

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
import omegaconf

from url_benchmark import utils
# from url_benchmark import replay_buffer as rb
from url_benchmark.in_memory_replay_buffer import ReplayBuffer
from url_benchmark.dmc import TimeStep
from url_benchmark import goals as _goals
from .ddpg import MetaDict
from .fb_modules import IdentityMap
from .ddpg import Encoder
from .fb_modules import mlp
from url_benchmark import utils
import time
import xxhash
logger = logging.getLogger(__name__)

@dataclasses.dataclass
class PSMConfig:
    # @package agent
    _target_: str = "url_benchmark.agent.psm.PSMAgent"
    name: str = "psm"
    # reward_free: ${reward_free}
    obs_type: str = omegaconf.MISSING  # to be specified later
    obs_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    action_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    device: str = omegaconf.II("device")  # ${device}
    lr: float = 1e-4
    lr_w: float = 1e-4
    lr_coef: float = 1
    fb_target_tau: float = 0.01  # 0.001-0.01
    update_every_steps: int = 1
    use_tb: bool = omegaconf.II("use_tb")  # ${use_tb}
    use_wandb: bool = omegaconf.II("use_wandb")  # ${use_wandb}
    use_hiplog: bool = omegaconf.II("use_hiplog")  # ${use_wandb}
    num_expl_steps: int = omegaconf.MISSING  # ???  # to be specified later
    num_inference_steps: int = 30000
    hidden_dim: int = 1024   # 128, 2048
    backward_hidden_dim: int = 526   # 512
    feature_dim: int = 512   # 128, 1024
    z_dim: int = 16  # 100
    d_dim: int = 256  # 100
    stddev_schedule: str = "0.2"  # "linear(1,0.2,200000)" #
    stddev_clip: float = 0.3  # 1
    update_z_every_step: int = 300
    update_z_proba: float = 1.0
    nstep: int = 1
    batch_size: int = 16 # 512
    num_neg_samples = 512
    init_fb: bool = True
    update_encoder: bool = omegaconf.II("update_encoder")  # ${update_encoder}
    goal_space: tp.Optional[str] = omegaconf.II("goal_space")
    ortho_coef: float = 0.0  # 0.01-10
    cons_coef: float = 0.01  # 0.01-10
    log_std_bounds: tp.Tuple[float, float] = (-5, 2)  # param for DiagGaussianActor
    temp: float = 1  # temperature for DiagGaussianActor
    boltzmann: bool = False  # set to true for DiagGaussianActor
    debug: bool = False
    future_ratio: float = 0.0
    mix_ratio: float = 0.5  # 0-1
    rand_weight: bool = False  # True, False
    preprocess: bool = True
    norm_z: bool = True
    q_loss: bool = False
    q_loss_coef: float = 0.01
    additional_metric: bool = False
    add_trunk: bool = False
    use_dgd: bool = True
    softmax: bool = True
    div_eps: float = 1000.0
    div_coef: float = 0.01
    inf_coeff: float = 5.0



cs = ConfigStore.instance()
cs.store(group="agent", name="psm", node=PSMConfig)

class SamplingActor(nn.Module):
    def __init__(self, obs_dim, z_dim, action_dim, feature_dim, hidden_dim) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.action_dim = action_dim

        self.mlp = mlp(self.obs_dim + self.z_dim, hidden_dim, "ntanh",
                             hidden_dim, "irelu")
        feature_dim = hidden_dim

        self.policy = mlp(feature_dim, hidden_dim, "irelu", self.action_dim)
        self.apply(utils.weight_init)

    def forward(self, obs, z):
        assert z.shape[-1] == self.z_dim
        h = self.mlp(torch.cat([obs, z], dim=-1))
        logits = self.policy(h)

        dist = torch.distributions.Categorical(logits=logits) 
        return dist
    
class SamplingSeedActor(nn.Module):
    def __init__(self, action_dim, z_dim, batch_size):
        super().__init__()
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.powers = torch.tensor([2**i for i in range(self.z_dim)][::-1]).to('cuda').repeat(batch_size,1)
        self.max_seed = 2**z_dim+20000
        self.seed_to_action = []
        
        for i in range(self.max_seed):
            torch.random.manual_seed(i)
            action = torch.randint(0, self.action_dim, (1,)).unsqueeze(0).numpy()
            self.seed_to_action.append(action)
        self.seed_to_action = np.array(self.seed_to_action)
        self.seed_to_action = torch.tensor(self.seed_to_action).to('cuda')
    
    def forward(self, obs_hash, z):
        # import ipdb;ipdb.set_trace()
        actions = []
        z_seed_time = time.time()
        seed_long = (z*self.powers).sum(1)
        # print("Time to compute z seed: ", time.time()-z_seed_time)
        final_seed_computation_time = time.time()
        final_seed = seed_long+obs_hash.reshape(-1)
        # print("Time to compute final seed: ", time.time()-final_seed_computation_time)
        # import ipdb;ipdb.set_trace()
        actions_computation_time = time.time()
        actions = self.seed_to_action[final_seed.long()]
        # print("Time to compute actions: ", time.time()-actions_computation_time)
        return torch.tensor(actions.reshape(-1,1)).to('cuda')

class PSM(nn.Module):
    def __init__(self, obs_dim, goal_dim, d_dim, action_dim, feature_dim, hidden_dim) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.d_dim = d_dim
        self.action_dim = action_dim
        self.mlp_phi = mlp(self.obs_dim + self.goal_dim, hidden_dim, "ntanh",
                            hidden_dim, "irelu",
                            hidden_dim, "irelu")
        
        self.mlp_b = mlp(self.obs_dim + self.goal_dim, hidden_dim, "ntanh",
                            hidden_dim, "irelu",
                            hidden_dim, "irelu")
        feature_dim = hidden_dim

        seq_phi = [feature_dim, hidden_dim, "irelu", self.d_dim * self.action_dim]
        seq_b = [feature_dim, hidden_dim, "irelu", self.action_dim]

        self.phi_fc = mlp(*seq_phi)
        self.b_fc = mlp(*seq_b)

        self.apply(utils.weight_init)

    def forward(self, obs, goal):
        phi = self.phi_fc(self.mlp_phi(torch.cat([obs, goal], dim=-1)))
        b = self.b_fc(self.mlp_b(torch.cat([obs, goal], dim=-1)))
        phi = phi.reshape(-1, self.action_dim, self.d_dim)
        b = b.reshape(-1, self.action_dim)
        return phi, b

class PSMAgent:

    # pylint: disable=unused-argument
    def __init__(self,
                 **kwargs: tp.Any
                 ):
        cfg = PSMConfig(**kwargs)
        self.cfg = cfg
        assert len(cfg.action_shape) == 1
        self.action_dim = cfg.action_shape[0]
        # self.solved_meta: tp.Any = None
        self.obs_dim = cfg.obs_shape[0]
        if cfg.feature_dim < self.obs_dim:
            logger.warning(f"feature_dim {cfg.feature_dim} should not be smaller that obs_dim {self.obs_dim}")
        goal_dim = self.obs_dim
        if cfg.goal_space is not None:
            goal_dim = _goals.get_goal_space_dim(cfg.goal_space)
        if cfg.d_dim < goal_dim:
            logger.warning(f"d_dim {cfg.d_dim} should not be smaller that goal_dim {goal_dim}")

        self.goal_dim = goal_dim

        self.psm = PSM(self.obs_dim, goal_dim, cfg.d_dim, self.action_dim, cfg.feature_dim, cfg.hidden_dim).to(cfg.device)

        self.psm_target = PSM(self.obs_dim, goal_dim, cfg.d_dim, self.action_dim, cfg.feature_dim, cfg.hidden_dim).to(cfg.device)

        self.w = mlp(cfg.z_dim, cfg.hidden_dim, "irelu", 
                     cfg.hidden_dim, "irelu",
                     cfg.hidden_dim, "irelu", cfg.d_dim,"L2").to(cfg.device)
        
        self.w.apply(utils.weight_init)
        
        self.w_target = mlp(cfg.z_dim, cfg.hidden_dim, "irelu",
                            cfg.hidden_dim, "irelu",
                            cfg.hidden_dim, "irelu", cfg.d_dim,"L2").to(cfg.device)
        
        self.w_inf = torch.zeros((cfg.d_dim), requires_grad=True, device=cfg.device)

        # self.sampling_actor = SamplingActor(self.obs_dim, cfg.z_dim, self.action_dim, cfg.feature_dim, cfg.hidden_dim).to(cfg.device)
        self.sampling_actor = SamplingSeedActor(self.action_dim, cfg.z_dim, cfg.batch_size * cfg.batch_size).to(cfg.device)
        
        # load the weights into the target networks
        self.psm_target.load_state_dict(self.psm.state_dict())
        self.w_target.load_state_dict(self.w.state_dict())

        # optimizer
        self.opt = torch.optim.Adam([{'params': self.psm.parameters()},  # type: ignore
                                        {'params': self.w.parameters(), 'lr': cfg.lr_coef * cfg.lr}],
                                       lr=cfg.lr)
        
        self.inf_opt = torch.optim.Adam([self.w_inf], lr=cfg.lr_w)

        # self.actor_opt = torch.optim.Adam([{'params': self.sampling_actor.parameters()}], lr=cfg.lr)

        self.train()
        self.psm.train()
        self.w.train()

    def train(self, training: bool = True) -> None:
        self.training = training
        for net in [self.psm, self.w]:
            net.train(training)

    def init_from(self, other) -> None:
        # copy parameters over
        names = []
        if self.cfg.init_fb:
            names += ["psm", "w", "psm_target", "w_target"]
        for name in names:
            utils.hard_update_params(getattr(other, name), getattr(self, name))
        for key, val in self.__dict__.items():
            if isinstance(val, torch.optim.Optimizer):
                val.load_state_dict(copy.deepcopy(getattr(other, key).state_dict()))

    def int_to_binary_array(self, int_vector, num_bits=None):
        if num_bits is None:
            num_bits = int_vector.max().bit_length()
        
        binary_array = ((int_vector[:, None] & (1 << np.arange(num_bits))) > 0).astype(int)
        return binary_array
    
    def sample_z(self, size, device: str = "cpu"):
        z_np = np.random.randint(0, 2**self.cfg.z_dim, (size,))
        binary_array = self.int_to_binary_array(z_np, self.cfg.z_dim)
        return torch.FloatTensor(binary_array).to(device)
    
    def update_psm(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        next_obs_hash: torch.Tensor,
        next_goal: torch.Tensor,
        z: torch.Tensor,
    ) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        #==========================================================
        
        mesh_making_time= time.time()

        idx = torch.arange(obs.shape[0]).to(obs.device)
        mesh = torch.stack(torch.meshgrid(idx, idx, indexing='xy')).T.reshape(-1, 2)
        
        
        # print(mesh[:, 0].shape)
        # print(obs[mesh[:, 0]].shape)
        m_obs = obs[mesh[:, 0]]
        m_next_obs = next_obs[mesh[:, 0]]
        m_next_obs_hash = next_obs_hash[mesh[:, 0]]
        m_action = action[mesh[:, 0]]
        m_next_goal = next_goal[mesh[:, 1]]
        # print("Time to make mesh: ", time.time()-mesh_making_time)
        # compute target successor measure
        # perm = torch.randperm(obs.shape[0])
        # perm_goals = next_goal[perm]
        with torch.no_grad():
            # compute greedy action
            target_phi, target_b = self.psm_target(m_next_obs, m_next_goal)
            target_w = self.w_target(z)
            sampling_time = time.time()
            next_actions = self.sampling_actor(m_next_obs_hash, z)
            target_phi = target_phi[torch.arange(target_phi.shape[0]), next_actions.squeeze(1)]
            target_b = target_b[torch.arange(target_b.shape[0]), next_actions.squeeze(1)]

            
            # print(target_phi.shape, target_w.shape, target_b.shape)
            target_M = torch.einsum("sd, sd -> s", target_phi, target_w) + target_b

        start_t = time.time()
        # compute PSM loss
        phi, b = self.psm(m_obs, m_next_goal)
        # print(m_obs.device)
        # print(phi.shape, m_action.shape)
        phi = phi[torch.arange(phi.shape[0]), m_action.squeeze(1)]
        # print(phi.shape)
        b = b[torch.arange(b.shape[0]), m_action.squeeze(1)]
        # print(phi.shape, self.w(z).shape, b.shape, target_M.shape)
        M = torch.einsum("sd, sd -> s", phi, self.w(z)) + b
        M = M.reshape(obs.shape[0], obs.shape[0])
        target_M = target_M.reshape(obs.shape[0], obs.shape[0])
        I = torch.eye(*M.size(), device=M.device)
        off_diag = ~I.bool()
        psm_offdiag: tp.Any = 0.5 * (M - discount * target_M)[off_diag].pow(2).mean()
        psm_diag: tp.Any = -(1-discount).mean()*(M.diag()).mean()
        psm_loss = psm_offdiag + psm_diag
        
        #==========================================================

        if self.cfg.use_tb or self.cfg.use_wandb or self.cfg.use_hiplog:
            metrics['psm_loss'] = psm_loss.item()
            metrics['psm_diag'] = psm_diag.item()
            metrics['psm_offdiag'] = psm_offdiag.item()
 
            if isinstance(self.opt, torch.optim.Adam):
                metrics["opt_lr"] = self.opt.param_groups[0]["lr"]

        # optimize PSM
        start_t = time.time()
        self.opt.zero_grad(set_to_none=True)
        # self.actor_opt.zero_grad(set_to_none=True)
        psm_loss.backward()
        self.opt.step()
        return metrics
    
    def update(self, replay_loader: ReplayBuffer, step: int) -> tp.Dict[str, float]:
        # print("Update started")
        metrics: tp.Dict[str, float] = {}
        # print(step)
        if step % self.cfg.update_every_steps != 0:
            return metrics
        
        start_t = time.time()
        batch = replay_loader.sample(self.cfg.batch_size)
        batch = batch.to(self.cfg.device)
        # batch2 = replay_loader.sample(self.cfg.batch_size ** 2)
        # batch2 = batch2.to(self.cfg.device)
        replay_buffer_sampling_time = time.time() - start_t
        start_t = time.time()
        # print("Batch sampled")
        # pdb.set_trace()
        obs = batch.obs
        action = batch.action.type(torch.int64)
        discount = batch.discount
        next_obs = batch.next_obs
        next_goal = batch.next_goal
        next_obs_hash = batch.next_obs_hash
        rewards = batch.reward
        
        batch_time = time.time() - start_t
        start_t = time.time()
        z = self.sample_z(self.cfg.batch_size, device=self.cfg.device)
        z = torch.repeat_interleave(z, self.cfg.batch_size, 0)
        if not z.shape[-1] == self.cfg.z_dim:
            raise RuntimeError("There's something wrong with the logic here")

        # rz = self.sample_z(self.cfg.batch_size ** 2, device=self.cfg.device)

        z_sampling_time = time.time() - start_t
        start_t = time.time()
        robs = None
        raction = None
        rnext_obs = None
        rz = None
        metrics.update(self.update_psm(obs=obs, action=action, discount=discount,
                                      next_obs=next_obs,next_obs_hash=next_obs_hash, next_goal=next_goal, 
                                      z=z))
        # print("Time to update: {}".format(time.time()-update_start))
        # update critic target
        psm_update_time = time.time() - start_t
        start_t = time.time()

        # print(f"Replay buffer sampling time: {replay_buffer_sampling_time}, Batch time: {batch_time}, Z sampling time: {z_sampling_time}, PSM update time: {psm_update_time}")
        utils.soft_update_params(self.psm, self.psm_target,
                                 self.cfg.fb_target_tau)
        utils.soft_update_params(self.w, self.w_target,
                                 self.cfg.fb_target_tau)

        return metrics
    
    def init_inference(self):
        '''
        Initialize the w_inf parameter
        Initialize the lagrange variables, optimizers etc
        '''

        # initialize the w_inf parameter
        self.w_inf = torch.randn((self.cfg.d_dim), requires_grad=True, device=self.cfg.device)
        self.inf_opt = torch.optim.Adam([{'params': self.w_inf}], lr=self.cfg.lr_w)

        if self.cfg.use_dgd:
            self.lmult = mlp(self.obs_dim + self.goal_dim, self.cfg.hidden_dim, "irelu", 
                            self.cfg.hidden_dim, "irelu", self.action_dim, "soft").to(self.cfg.device)
            
            self.lmult.apply(utils.weight_init)
            
            self.lmult_opt = torch.optim.Adam([{'params': self.lmult.parameters()}], lr=self.cfg.lr_w)

    def infer_w(self, replay_loader: ReplayBuffer, inf_logger, goal):
        metrics: tp.Dict[str, float] = {}
        self.init_inference()
        goal = torch.tensor(goal).unsqueeze(0).to(self.cfg.device)
        # print(goal.shape)
        for step in range(self.cfg.num_inference_steps):
            batch = replay_loader.sample(self.cfg.batch_size)
            obs = batch.obs
            # print(goal.shape)
            goal_rep = goal.repeat(obs.shape[0], 1)
            # print(goal.shape)
            obs = torch.tensor(obs).to(self.cfg.device)
            # print(obs.shape, goal.shape)
            metrics.update(self._infer_step(obs, goal_rep, step))
            inf_logger.log_metrics(metrics)
            inf_logger.dump()
        self.w_inf = (F.normalize(self.w_inf.reshape(1,-1))*math.sqrt(self.w_inf.shape[0])).reshape(-1)
        return metrics


    def _infer_step(self, obs, goal, step):
        perm = torch.randperm(obs.shape[0])
        perm_obs = obs[perm]

        metrics = {}

        with torch.no_grad():
            phi_g, b_g = self.psm(obs, goal)
            phi_perm, b_perm = self.psm(obs, perm_obs)
        obj = -torch.einsum("sad, d -> sa", phi_g, (F.normalize(self.w_inf.reshape(1,-1))*math.sqrt(self.w_inf.shape[0])).reshape(-1)).mean()
        if self.cfg.use_dgd:
            with torch.no_grad():
                l_mult = self.lmult(torch.cat([obs, perm_obs], dim=-1))
            constraints = - ((torch.einsum("sad, d -> sa", phi_perm, (F.normalize(self.w_inf.reshape(1,-1))*math.sqrt(self.w_inf.shape[0])).reshape(-1)) + b_perm) * l_mult).mean()
        else:
            constraints = - (torch.min(torch.einsum("sad, d -> sa", phi_perm, (F.normalize(self.w_inf.reshape(1,-1))*math.sqrt(self.w_inf.shape[0])).reshape(-1)) + b_perm, torch.tensor(0.0)) * self.cfg.inf_coeff).mean()
        
        self.inf_opt.zero_grad(set_to_none=True)
        loss = obj + constraints
        loss.backward()
        self.inf_opt.step()

        metrics['obj'] = obj.item()
        metrics['constraints'] = constraints.item()
        metrics['lamb'] = l_mult.mean().item() if self.cfg.use_dgd else self.cfg.inf_coeff
        # metrics['step'] = step

        if self.cfg.use_dgd:        
            constraints = ((torch.einsum("sad, d -> sa", phi_perm, (F.normalize(self.w_inf.reshape(1,-1))*math.sqrt(self.w_inf.shape[0])).reshape(-1)) + b_perm) * self.lmult(torch.cat([obs, perm_obs], dim=-1))).mean()

            self.lmult_opt.zero_grad(set_to_none=True)
            loss = constraints
            loss.backward()
            self.lmult_opt.step()
        
        # print('Step: ', step, ' | Objective: ', metrics["obj"], ' | Constraints: ', metrics["constraints"])
        return metrics

    
    def q_function(self, obs, goal):
        obs = torch.tensor(obs).to(self.cfg.device)
        goal = torch.tensor(goal).to(self.cfg.device)
        with torch.no_grad():
            phi, b = self.psm(obs, goal)
            q = torch.einsum("sad, d -> sa", phi, self.w_inf) + b

        return q
    
    def act(self, obs, goal):
        q_function = self.q_function(obs, goal)

        # pi = torch.distributions.Categorical(logits=q_function)
        # action = pi.sample()
        action = torch.argmax(q_function, dim=1)

        return action
    
    def visualize_sampling_policy(self, work_dir, step, env, num_z, comp_log_prob=False):
        z = self.sample_z(num_z)
        state_list = env.get_state_list()
        obs_list = [env.get_obs_from_state(state) for state in state_list] # implement this function
        for i in range(num_z):
            obs = torch.cat(obs_list, dim=0).to(self.cfg.device)
            z = z[i].unsqueeze(0).repeat(obs.shape[0], 1)
            act_list = self.sampling_actor(obs, z)
            diversity = 0.0
            if comp_log_prob:
                if i == 0:
                    log_pi_first = self.sampling_actor.log_prob(act_list, obs_list, z)
                else:
                    log_pi = self.sampling_actor.log_prob(act_list, obs_list, z)
                    diversity = torch.abs(log_pi - log_pi_first).mean()
                
            env.plot_policy_from_list(work_dir, obs_list, act_list, diversity, f"training_step_{step}_sampling_policy_{i}") #implement this function

    def plot_q_function(self, work_dir, step, env, goal):
        state_list = env.get_state_list()
        print('in plot_q_function')
        # print(state_list)
        obs_list = [torch.tensor(env.get_obs_from_state(state)).unsqueeze(0) for state in state_list] # implement this function
        # print(obs_list)
        # print(len(state_list))
        obs_list = torch.cat(obs_list, dim=0).to(self.cfg.device)
        goal = torch.tensor(goal).unsqueeze(0).repeat(obs_list.shape[0], 1).to(self.cfg.device)
        # print(obs_list.shape, goal.shape)
        q_list = self.q_function(obs_list, goal)
        # print(q_list)
        v_list = torch.max(q_list, dim=1)[0]
        a_list = torch.argmax(q_list, dim=1)
        # print(v_list, a_list)
        env.plot_v_function(work_dir, obs_list.cpu(), v_list, a_list, f"training_step_{step}_v_function") # write this function


    


    