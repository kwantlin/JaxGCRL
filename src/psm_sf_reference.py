

import os

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
from tensordict.tensordict import TensorDict
import omegaconf

from ..models.psm_models import PhiMap, PsiMap, Actor, SimpleActor
from ..models.parallel_modules import weight_init
from .. import utils
from ..utils import COMPILE_OPTIONS
from ..agent.base_agent import Agent, class_register, register

from torchrl.data import ReplayBuffer
import time


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
            action = (torch.rand(size=(self.action_dim,)).unsqueeze(0)-1)*2
            self.seed_to_action.append(action)
        # import ipdb;ipdb.set_trace()
        self.seed_to_action = np.array(self.seed_to_action).squeeze()
    
    def forward(self, obs_hash, z):
        # import ipdb;ipdb.set_trace()
        actions = []
        # import ipdb;ipdb.set_trace()
        seed_long = (z*self.powers).sum(1)
        final_seed = (seed_long+ obs_hash.reshape(-1))%self.max_seed
        # import ipdb;ipdb.set_trace()
        actions = self.seed_to_action[final_seed.cpu().numpy().astype(np.int32)]
        return torch.FloatTensor(actions).to('cuda')


@dataclasses.dataclass
class OptimConfig:
    lr_sf: float = 1e-4
    lr_phi: float = 1e-4
    lr_actor: float = 1e-4
    weight_decay: float = 0
    clip_grad_norm: float = 0
    target_tau: float = 0.01  # 0.001-0.01
    ortho_coef: float = 1.0  # 0.01-10
    mix_ratio: float = 0.5  # 0-1
    pessimism_penalty: float = 0
    actor_pessimism_penalty: float = 0.5

@dataclasses.dataclass
class ActorArchiConfig:
    hidden_dim: int = 1024
    model: str = 'simple'  # TODO not used at the moment
    hidden_layers: int = 1
    embedding_layers: int = 2

@dataclasses.dataclass
class SFArchiConfig:
    hidden_dim: int = 1024
    model: str = 'simple'  # TODO not used at the moment
    hidden_layers: int = 1
    embedding_layers: int = 2
    num_parallel: int = 2

@dataclasses.dataclass
class PhiArchiConfig:
    hidden_dim: int = 256
    model: str = 'simple'  # TODO not used at the moment
    hidden_layers: int = 2
    norm: bool = True
    batch_norm: bool = False


@dataclasses.dataclass
class ArchiConfig:
    z_dim: int = 128
    norm_z: bool = True
    sf: SFArchiConfig = dataclasses.field(default_factory=SFArchiConfig)
    phi: PhiArchiConfig = dataclasses.field(default_factory=PhiArchiConfig)
    actor: ActorArchiConfig = dataclasses.field(default_factory=ActorArchiConfig)

@dataclasses.dataclass
class AwacConfig:
    beta: float = 1.0
    num_value_samples: int = 1
    max_clip: tp.Optional[float] = None
    score_processing: str = "softmax"


@dataclasses.dataclass
class PSM_AgentConfig:
    # @package agent
    _target_: str = "psm_sf.agent.psm.PSMAgent"
    name: str = "psm"
    # batch size is interpolated from here in base config
    batch_size: int = 1024
    update_every_steps: int = 1
    obs_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    action_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    device: str = omegaconf.II("device")
    phi_input: str = "s"  # ["s", "as", "sas", "ss"]
    stddev_schedule: str = "0.2"  # "linear(1,0.2,200000)"  # TODO remove?
    stddev_clip: float = 0.3
    update_z_every_step: int = 250
    num_inference_steps: int = 5120
    eval_actor_samples: int = 1
    eval_gpi_samples: int = 1
    archi: ArchiConfig = dataclasses.field(default_factory=ArchiConfig)
    optim: OptimConfig = dataclasses.field(default_factory=OptimConfig)
    discount: float = omegaconf.II("discount")
    use_awac: bool = False
    awac: AwacConfig = dataclasses.field(default_factory=AwacConfig)
    log_grad_norm: bool = False
    use_mix_rollout: bool = False
    norm_obs: bool = False
    max_log_seed:int = 16


try:
    # on mac we cannot install hydra
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore.instance()
    cs.store(group="agent", name="psm", node=PSM_AgentConfig)
except:
    import warnings
    from pathlib import Path
    warnings.warn(f"Hydra has failed to setup the configuration in {Path(__file__).name}")


@class_register
class PSMAgent:

    # pylint: disable=unused-argument
    def __init__(self,
                 **kwargs: tp.Any
                 ):
        cfg_fields = {field.name for field in dataclasses.fields(PSM_AgentConfig)}
        kwargs = {x: y for x, y in kwargs.items() if x in cfg_fields}
        cfg = PSM_AgentConfig(**kwargs)
        self.cfg = cfg
        assert len(cfg.action_shape) == 1
        self.action_dim = cfg.action_shape[0]
        self.obs_dim = cfg.obs_shape[0]
        self.goal_keys: tp.List = []

        # If defined and an int, this will tell how big are the batch sizes the agent will request from
        # expert buffer will be. This allows us to prefetch batches to speed up training.
        self.expert_buffer_prefetch_batch_size = None
        self.train_buffer_prefetch_batch_size = self.cfg.batch_size

        arch = cfg.archi

        if self.cfg.phi_input == "s":
            goal_dim = self.obs_dim
            goal_keys = [("next_observation")]
        elif self.cfg.phi_input == "as":
            goal_dim = self.obs_dim + self.action_dim
            goal_keys = ["action", ("next_observation")]
        elif self.cfg.phi_input == "sas":
            goal_dim = 2*self.obs_dim + self.action_dim
            goal_keys = ["observation", "action", ("next_observation")]
        elif self.cfg.phi_input == "ss":
            goal_dim = 2*self.obs_dim
            goal_keys = ["observation", ("next_observation")]
        else:
            raise ValueError(f"Unsupported config for phi_input: {self.cfg.phi_input}")
        self.goal_keys = goal_keys
        self.goal_dim = goal_dim
        self.sampling_actor = SamplingSeedActor(self.action_dim, self.cfg.max_log_seed,self.cfg.batch_size).to(cfg.device)
        # create networks
        self.phi = PhiMap(goal_dim, arch.z_dim, arch.phi.hidden_dim, arch.phi.hidden_layers, arch.phi.norm, arch.phi.batch_norm)
        self.sf_psi = PsiMap(self.obs_dim, arch.z_dim, self.action_dim, arch.sf.hidden_dim, 
                                     arch.sf.hidden_layers, arch.sf.embedding_layers, arch.sf.num_parallel)
        self.psm_psi = PsiMap(self.obs_dim,self.cfg.max_log_seed , self.action_dim, arch.sf.hidden_dim, 
                                     arch.sf.hidden_layers, arch.sf.embedding_layers, arch.sf.num_parallel,output_dim=arch.z_dim)
        self.actor = Actor(self.obs_dim, arch.z_dim, self.action_dim, arch.actor.hidden_dim, 
                      arch.actor.hidden_layers, arch.actor.embedding_layers)
        self.nets: tp.Dict[str, nn.Module] = dict(actor=self.actor, sf_psi=self.sf_psi, psm_psi=self.psm_psi, phi=self.phi)

        # initialize networks and targets
        for net in self.nets.values():
            net.apply(weight_init)
        self.nets = {x: y.to(cfg.device) for x, y in self.nets.items()}
        self.targets = {x: copy.deepcopy(self.nets[x]) for x in ["sf_psi","psm_psi", "phi"]}
        self.target_phi = self.targets["phi"]
        self.target_sf_psi = self.targets["sf_psi"]
        self.target_psm_psi = self.targets["psm_psi"]



        # optimizers
        self.optim_actor=torch.optim.Adam(self.nets["actor"].parameters(), lr=cfg.optim.lr_actor, weight_decay=cfg.optim.weight_decay)
        self.optim_sf_psi=torch.optim.Adam(self.nets["sf_psi"].parameters(), lr=self.cfg.optim.lr_sf, weight_decay=cfg.optim.weight_decay)
        self.optim_psm_psi=torch.optim.Adam(self.nets["psm_psi"].parameters(), lr=self.cfg.optim.lr_sf, weight_decay=cfg.optim.weight_decay)
        self.optim_phi=torch.optim.Adam(self.nets["phi"].parameters(), lr=self.cfg.optim.lr_phi, weight_decay=cfg.optim.weight_decay)

        self.optims = dict(
            actor=self.optim_actor,
            sf_psi=self.optim_sf_psi,
            psm_psi=self.optim_psm_psi,
            phi=self.optim_phi
        )

        self.norm_obs = nn.BatchNorm1d(self.cfg.obs_shape[0], affine=False, momentum=0.01) if self.cfg.norm_obs else nn.Identity()
        self.norm_obs = self.norm_obs.to(self.cfg.device)

        self.train()
        for net in self.targets.values():
            net.train()

        # precompute some useful variables
        self.num_parallel_scaling = arch.sf.num_parallel**2 - arch.sf.num_parallel
        self.off_diag = 1 - torch.eye(cfg.batch_size, cfg.batch_size, device=cfg.device)
        self.off_diag_sum = self.off_diag.sum()


        self.replay_buffer = None
        # precompute parameters number
        self.param_num = {
            k: sum(p.numel() for p in model.parameters()) for (k, model) in self.nets.items()
        }


    def initialize_replay_buffer(self, torch_rb):
        if self.replay_buffer is None:
            replay_buffer = {}
            replay_buffer_size= torch_rb["train"]['observation'].shape[0]
            size = int(replay_buffer_size * 0.1)
            replay_buffer['observation'] = torch_rb["train"]['observation'][:size].cpu()
            replay_buffer['action'] = torch_rb["train"]['action'][:size].cpu()
            replay_buffer['next_observation'] = torch_rb["train"]['next']['observation'][:size].cpu()
            replay_buffer['terminated'] = torch_rb["train"]['next']['terminated'][:size].cpu()
            print('Replay buffer size:', size)
            self.replay_buffer = replay_buffer

            self.replay_buffer['next_observation_hash'] = np.arange(0, self.replay_buffer['next_observation'].shape[0])
            # np.zeros(self.replay_buffer['next_observation'].shape[0])
            
            
            # next_obs_np = self.replay_buffer['next_observation'].cpu().numpy()
            
            # for i in range(self.replay_buffer['next_observation'].shape[0]):
            #     self.replay_buffer['next_observation_hash'][i] = hash(str(next_obs_np[i]))%self.sampling_actor.max_seed 
            self.replay_buffer['next_observation_hash'] = torch.tensor(self.replay_buffer['next_observation_hash'])


    def int_to_binary_array(self, int_vector, num_bits=None):
        if num_bits is None:
            num_bits = int_vector.max().bit_length()
        
        binary_array = ((int_vector[:, None] & (1 << np.arange(num_bits))) > 0).astype(int)
        return binary_array
    
    def sample_z_psm(self, size, device: str = "cpu"):
        z_np = np.random.randint(0, 2**self.cfg.max_log_seed, (size,))
        binary_array = self.int_to_binary_array(z_np, self.cfg.max_log_seed)
        return torch.FloatTensor(binary_array).to(device)

    def train(self, training: bool = True) -> None:
        self.training = training
        for net in self.nets.values():
            net.train(training)
        if hasattr(self, "norm_obs"):
            self.norm_obs.train(training)

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        self.cfg.device = device
        for net in self.nets.values():
            net.to(*args, **kwargs)
        for net in self.targets.values():
            net.to(*args, **kwargs)

    def init_from(self, other) -> None:
        # TODO reimplement this function differently? (eg by taking a path)
        if hasattr(other, "norm_obs"):
            self.norm_obs.load_state_dict(copy.deepcopy(other.norm_obs.state_dict()))
        names = ["actor", "psm_psi", "sf_psi", "phi"]
        for dico in ["optims", "nets", "targets"]:
            ds = [getattr(x, dico) for x in (self, other)]
            for k, val in ds[0].items():
                if k not in names:
                    continue
                if isinstance(val, torch.optim.Optimizer):
                    val.load_state_dict(copy.deepcopy(ds[1][k].state_dict()))
                else:
                    utils.hard_update_params(ds[1][k], val)

    def init_from_state_dicts(self, state_dict_paths):
        for name, path in state_dict_paths.items():
            state_dict = torch.load(path, map_location=self.cfg.device)
            if name == "actor":
                self.actor.load_state_dict(state_dict)
                self.nets[name] = self.actor
            elif name == "sf_psi":
                self.sf_psi.load_state_dict(state_dict)
                self.nets[name] = self.sf_psi
            elif name == "phi":
                self.phi.load_state_dict(state_dict)
                self.nets[name] = self.phi
            elif name == "psm_psi":
                self.psm_psi.load_state_dict(state_dict)
                self.nets[name] = self.psm_psi


    def get_phi_input(self, td: TensorDict, normalize: bool=True) -> torch.Tensor:
        if not hasattr(self, "norm_obs") or not normalize:
            return torch.concat([td.get(k).to(self.cfg.device) for k in self.goal_keys], dim=-1)
        cat_tensors = []
        with utils.eval_mode(self):
            for k in self.goal_keys:
                cat_tensors.append(self.norm_obs(td.get(k).to(self.cfg.device)) if k in ["observation", ("next_observation")] else td.get(k))
        return torch.concat(cat_tensors, dim=-1)

    def sample_z(self, size: int, device: str = "cpu") -> torch.Tensor:
        arch = self.cfg.archi
        z = torch.randn((size, arch.z_dim), dtype=torch.float32, device=device)
        if arch.norm_z:
            z = math.sqrt(z.shape[1]) * F.normalize(z, dim=1)
        return z

    def maybe_update_rollout_context(self, td: TensorDict, replay_buffer: tp.Union[ReplayBuffer, tp.Mapping[str, ReplayBuffer]]) -> TensorDict:
        # get mask for environmets where we need to change z
        opt = self.cfg.optim
        if "z" in td.keys():
            z = td["z"].clone().to(self.cfg.device)
            mask_reset_z = (td.get("step_count") % self.cfg.update_z_every_step == 0)
            num_new_z = torch.sum(mask_reset_z).item()
            if num_new_z > 0:
                new_z = self.sample_z(num_new_z, device=self.cfg.device)
                if self.cfg.use_mix_rollout and(opt.mix_ratio > 0) and len(replay_buffer["train"]) > 0:
                    mix_idxs: tp.Any = np.where(np.random.uniform(size=num_new_z) < opt.mix_ratio)[0]
                    if len(mix_idxs) > 0:
                        batch = replay_buffer["train"].sample(len(mix_idxs))
                        phi_input = self.get_phi_input(batch)
                        with torch.no_grad(), utils.eval_mode(self):
                            mix_z = self.nets["phi"](phi_input.to(self.cfg.device)).detach()
                        if self.cfg.archi.norm_z:
                            mix_z = math.sqrt(mix_z.shape[1]) * F.normalize(mix_z, dim=1)
                        new_z[mix_idxs] = mix_z
                z[mask_reset_z.ravel()] = new_z
        else:
            z = self.sample_z(td.batch_size[0], device=self.cfg.device)
            
        return TensorDict({"z": z}, batch_size=td.batch_size)

    def act(self, td: TensorDict, step: int, eval_mode: bool) -> tp.Any:
        num_es, num_gpi = self.cfg.eval_actor_samples, self.cfg.eval_gpi_samples
        obs = td["observation"].to(self.cfg.device)  # batch x obs_dim
        z = td["z"].to(self.cfg.device)  # batch x z_dim
        with torch.no_grad(), utils.eval_mode(self):
            if hasattr(self, "norm_obs"):
                obs = self.norm_obs(obs)
            stddev = utils.schedule(self.cfg.stddev_schedule, step)
            dist = self.nets["actor"](obs, z, stddev)
            if eval_mode:
                if num_es > 1 or num_gpi > 1:
                    batch_size = td["observation"].shape[0]
                    noise = torch.randn((num_gpi, batch_size, z.shape[-1]), dtype=torch.float32, device=self.cfg.device)  # num_gpi x batch x z_dim
                    noise[0] = 0  # make sure we include the original z into those used for gpi
                    zs = z.expand(num_gpi, -1, -1) + noise  # num_gpi x batch x z_dim
                    if self.cfg.archi.norm_z:
                        zs = math.sqrt(zs.shape[-1]) * F.normalize(zs, dim=-1)
                    obs_exp = obs.expand(num_gpi, -1, -1)  # num_gpi x batch x obs_dim
                    dist = self.nets["actor"](obs_exp, zs, stddev)  # num_gpi x batch x action_dim
                    if num_es == 1:
                        sampled_actions = dist.mean.expand(num_es, -1, -1, -1) # num_es x num_gpi x batch x action_dim
                    else:
                        sampled_actions = dist.sample(sample_shape=(num_es,))  # num_es x num_gpi x batch x action_dim
                    obs_exp = obs_exp.expand(num_es, -1, -1, -1)  # num_es x num_gpi x batch x obs_dim
                    z_exp = zs.expand(num_es, -1, -1, -1)  # num_es x num_gpi x batch x z_dim
                    # flatten
                    obs_flat = torch.flatten(obs_exp, start_dim=0, end_dim=2) # (num_es * num_gpi * batch) x obs_dim
                    z_flat = torch.flatten(z_exp, start_dim=0, end_dim=2) # (num_es * num_gpi * batch) x z_dim
                    actions_flat = sampled_actions.view(num_es * num_gpi * batch_size, sampled_actions.shape[-1]) # (num_es * num_gpi * batch) x action_dim
                    psis = self.nets["sf_psi"](obs_flat, z_flat, actions_flat)  # num_parallel x (num_es * num_gpi * batch) x z_dim
                    z_r = z.expand(num_es, num_gpi, -1, -1)  # num_es x num_gpi x batch x z_dim
                    z_r = torch.flatten(z_r, start_dim=0, end_dim=2) # (num_es * num_gpi * batch) x z_dim
                    Qs = (psis*z_r).sum(-1)  # num_parallel x (num_es * num_gpi * batch)
                    Q_mean, Q_unc = self.get_targets_uncertainty(Qs)  # (num_es * num_gpi * batch)
                    Q_flat = Q_mean - self.cfg.optim.actor_pessimism_penalty * Q_unc  # (num_es * num_gpi * batch)
                    Q = Q_flat.view(num_es * num_gpi, batch_size, 1) # (num_es * num_gpi) x batch x 1
                    actions_view = sampled_actions.view(num_es * num_gpi, batch_size, sampled_actions.shape[-1]) # (num_es * num_gpi) x batch x action_dim
                    Qs_argmax = torch.argmax(Q, dim=0, keepdim=True).expand_as(actions_view)
                    action = torch.gather(actions_view, dim=0, index=Qs_argmax)[0]
                    return action
                else:
                    action = dist.mean
            else:
                action = dist.sample()
        return action

    def get_targets_uncertainty(self, preds, dim=0):
        preds_mean = preds.mean(dim=dim)
        preds_uns = preds.unsqueeze(dim=dim) # 1 x n_parallel x ...
        preds_uns2 = preds.unsqueeze(dim=dim+1) # n_parallel x 1 x ...
        preds_diffs = torch.abs(preds_uns - preds_uns2) # n_parallel x n_parallel x ...
        preds_unc = preds_diffs.sum(dim=(dim, dim+1),) / self.num_parallel_scaling
        return preds_mean, preds_unc


    # @torch.compile(**COMPILE_OPTIONS)
    def _update_psm(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        next_obs_hash,
        goal: torch.Tensor,
        z: torch.Tensor,
        pessimism_penalty: float,
        stddev: float,
        stddev_clip: float,
        ortho_coef: float,
        clip_grad_norm: float,
    ) -> tp.Any:
        # compute target successor measure
        with torch.no_grad():
            next_action = self.sampling_actor(next_obs_hash,z)
            # import ipdb;ipdb.set_trace()
            # dist = self.actor(next_obs, z, stddev)
            # next_action = dist.sample(clip=stddev_clip)
            target_psm_psis = self.target_psm_psi(next_obs, z, next_action)  # num_parallel x batch x z_dim
            target_phi = self.target_phi(goal)  # batch x z_dim
            target_Ms = torch.matmul(target_psm_psis, target_phi.T)  # num_parallel x batch x batch
            target_M_mean, target_M_unc = self.get_targets_uncertainty(target_Ms)  # batch x batch
            target_M = target_M_mean - pessimism_penalty * target_M_unc  # batch x batch

        # compute SF loss
        psis = self.psm_psi(obs, z, action)  # num_parallel x batch x z_dim
        phi = self.phi(goal)  # batch x z_dim
        Ms = torch.matmul(psis, phi.T)  # num_parallel x batch x batch

        diff : tp.Any = Ms - discount * target_M  # num_parallel x batch x batch
        psm_offdiag: tp.Any = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
        psm_diag: tp.Any = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
        psm_loss = psm_offdiag + psm_diag

        # compute orthonormality loss for phi embedding
        Cov = torch.matmul(phi, phi.T)
        orth_loss_diag = - Cov.diag().mean()
        orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
        orth_loss = orth_loss_offdiag + orth_loss_diag
        psm_loss += 1 * orth_loss

        # optimize PSM
        self.optim_psm_psi.zero_grad(set_to_none=True)
        self.optim_phi.zero_grad(set_to_none=True)

        psm_loss.backward()
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.psm_psi.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.phi.parameters(), clip_grad_norm)
        self.optim_psm_psi.step()
        self.optim_phi.step()

        return target_M, Ms, psis, phi, z, psm_loss, psm_diag, psm_offdiag, orth_loss, orth_loss_diag, orth_loss_offdiag

    # @torch.compile(**COMPILE_OPTIONS)
    def _update_sf(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        goal: torch.Tensor,
        z: torch.Tensor,
        pessimism_penalty: float,
        stddev: float,
        stddev_clip: float,
        ortho_coef: float,
        clip_grad_norm: float,
    ) -> tp.Any:
        # compute target successor measure
        with torch.no_grad():
            dist = self.actor(next_obs, z, stddev)
            next_action = dist.sample(clip=stddev_clip)
            target_psis = self.target_sf_psi(next_obs, z, next_action)  # num_parallel x batch x z_dim
            target_phi = self.phi(goal)  # batch x z_dim
            target_Ms = torch.matmul(target_psis, target_phi.T)  # num_parallel x batch x batch
            target_M_mean, target_M_unc = self.get_targets_uncertainty(target_Ms)  # batch x batch
            target_M = target_M_mean - pessimism_penalty * target_M_unc  # batch x batch

        # compute SF loss
        psis = self.sf_psi(obs, z, action)  # num_parallel x batch x z_dim
        phi = self.phi(goal).detach()  # batch x z_dim
        Ms = torch.matmul(psis, phi.T)  # num_parallel x batch x batch

        diff : tp.Any = Ms - discount * target_M  # num_parallel x batch x batch
        sf_offdiag: tp.Any = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
        sf_diag: tp.Any = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
        sf_loss = sf_offdiag + sf_diag

        # compute orthonormality loss for phi embedding
        Cov = torch.matmul(phi, phi.T)
        orth_loss_diag = - Cov.diag().mean()
        orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
        orth_loss = orth_loss_offdiag + orth_loss_diag
        sf_loss += 0*ortho_coef * orth_loss

        # optimize SF
        self.optim_sf_psi.zero_grad(set_to_none=True)
        # self.optim_phi.zero_grad(set_to_none=True)

        sf_loss.backward()
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.sf_psi.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.phi.parameters(), clip_grad_norm)
        self.optim_sf_psi.step()
        # self.optim_phi.step()

        return target_M, Ms, psis, phi, z, sf_loss, sf_diag, sf_offdiag, orth_loss, orth_loss_diag, orth_loss_offdiag

    def update_psm(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        next_obs_hash: torch.Tensor,
        goal: torch.Tensor,
        z: torch.Tensor,
        step: int,
    ) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        opt = self.cfg.optim

        target_M, Ms, psis, phi, z, psm_loss, psm_diag, psm_offdiag, orth_loss, orth_loss_diag, orth_loss_offdiag = self._update_psm(
            obs,
            action,
            discount,
            next_obs,
            next_obs_hash,
            goal,
            z,
            opt.pessimism_penalty,
            utils.schedule(self.cfg.stddev_schedule, step),
            self.cfg.stddev_clip,
            opt.ortho_coef,
            opt.clip_grad_norm,
        )
    
        # This is still bit nasty
        with torch.no_grad():
            metrics['target_M'] = target_M.mean().to("cpu", non_blocking=True)
            metrics['M1'] = Ms[0].mean().to("cpu", non_blocking=True)
            metrics['psm_psi1'] = psis[0].mean().to("cpu", non_blocking=True)
            metrics['phi'] = phi.mean().to("cpu", non_blocking=True)
            metrics['phi_norm'] = torch.norm(phi, dim=-1).mean().to("cpu", non_blocking=True)
            metrics['z_norm'] = torch.norm(z, dim=-1).mean().to("cpu", non_blocking=True)
            metrics['psm_loss'] = psm_loss.to("cpu", non_blocking=True)
            metrics['psm_diag'] = psm_diag.to("cpu", non_blocking=True)
            metrics['psm_offdiag'] = psm_offdiag.to("cpu", non_blocking=True)
            metrics['orth_loss'] = orth_loss.to("cpu", non_blocking=True)
            metrics['orth_loss_diag'] = orth_loss_diag.to("cpu", non_blocking=True)
            metrics['orth_loss_offdiag'] = orth_loss_offdiag.to("cpu", non_blocking=True)

            # collect gradient norm
            if self.cfg.log_grad_norm:
                for k in  ["phi", "psm_psi"]:
                    metrics[f'{k}_grad_norm'] = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach())
                    for p in self.nets[k].parameters() ])).to("cpu", non_blocking=True) / self.param_num[k]

        return metrics


    def update_sf(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        goal: torch.Tensor,
        z: torch.Tensor,
        step: int,
    ) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        opt = self.cfg.optim

        target_M, Ms, sf_psis, phi, z, sf_loss, sf_diag, sf_offdiag, orth_loss, orth_loss_diag, orth_loss_offdiag = self._update_sf(
                obs,
                action,
                discount,
                next_obs,
                goal,
                z,
                opt.pessimism_penalty,
                utils.schedule(self.cfg.stddev_schedule, step),
                self.cfg.stddev_clip,
                opt.ortho_coef,
                opt.clip_grad_norm,
            )

        # This is still bit nasty
        with torch.no_grad():
            metrics['target_M'] = target_M.mean().to("cpu", non_blocking=True)
            metrics['M1'] = Ms[0].mean().to("cpu", non_blocking=True)
            metrics['sf_psi1'] = sf_psis[0].mean().to("cpu", non_blocking=True)
            metrics['phi'] = phi.mean().to("cpu", non_blocking=True)
            metrics['phi_norm'] = torch.norm(phi, dim=-1).mean().to("cpu", non_blocking=True)
            metrics['z_norm'] = torch.norm(z, dim=-1).mean().to("cpu", non_blocking=True)
            metrics['sf_loss'] = sf_loss.to("cpu", non_blocking=True)
            metrics['sf_diag'] = sf_diag.to("cpu", non_blocking=True)
            metrics['sf_offdiag'] = sf_offdiag.to("cpu", non_blocking=True)
            metrics['orth_loss'] = orth_loss.to("cpu", non_blocking=True)
            metrics['orth_loss_diag'] = orth_loss_diag.to("cpu", non_blocking=True)
            metrics['orth_loss_offdiag'] = orth_loss_offdiag.to("cpu", non_blocking=True)

            # collect gradient norm
            if self.cfg.log_grad_norm:
                for k in  ["phi", "sf_psi"]:
                    metrics[f'{k}_grad_norm'] = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach())
                    for p in self.nets[k].parameters() ])).to("cpu", non_blocking=True) / self.param_num[k]

        return metrics

    ##################################
    # Deterministic policy gradient update
    ##################################

    # @torch.compile(**COMPILE_OPTIONS)
    def _update_td3_actor(
        self,
        obs: torch.Tensor,
        z: torch.Tensor,
        stddev: float,
        stddev_clip: float,
        actor_pessimism_penalty: float,
        clip_grad_norm: float
    ) -> tp.Any:
        dist = self.actor(obs, z, stddev)
        action = dist.sample(clip=stddev_clip)
        psis = self.sf_psi(obs, z, action)  # num_parallel x batch x z_dim
        Qs = (psis*z).sum(-1)  # num_parallel x batch
        Q_mean, Q_unc = self.get_targets_uncertainty(Qs)  # batch
        Q = Q_mean - actor_pessimism_penalty * Q_unc  # batch
        actor_loss = -Q.mean()

        # optimize actor
        self.optim_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), clip_grad_norm)

        self.optim_actor.step()

        return actor_loss, Q

    def update_td3_actor(self, obs: torch.Tensor, action: torch.Tensor, z: torch.Tensor, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        opt = self.cfg.optim
        stddev = utils.schedule(self.cfg.stddev_schedule, step)

        actor_loss, Q = self._update_td3_actor(
            obs,
            z,
            stddev,
            self.cfg.stddev_clip,
            opt.actor_pessimism_penalty,
            opt.clip_grad_norm,
        )

        with torch.no_grad():
            metrics['actor_loss'] = actor_loss.to("cpu", non_blocking=True)
            metrics['q'] = Q.mean().to("cpu", non_blocking=True)
            if self.cfg.log_grad_norm:
                metrics["actor_grad_norm"] = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach())
                 for p in self.nets["actor"].parameters() ])).to("cpu", non_blocking=True) / self.param_num["actor"]
        return metrics

    def update_actor(self, obs: torch.Tensor, action: torch.Tensor, z: torch.Tensor, step: int) -> tp.Dict[str, float]:
        return self.update_td3_actor(obs=obs, action=action, z=z, step=step)
            

    def update(self, replay_buffer: tp.Union[ReplayBuffer, tp.Mapping[str, ReplayBuffer]], step: int) -> tp.Dict[str, float]:
        self.initialize_replay_buffer(replay_buffer)
        torch.compiler.cudagraph_mark_step_begin()
        metrics: tp.Dict[str, float] = {}
        opt = self.cfg.optim

        if step % self.cfg.update_every_steps != 0:
            return metrics

        start_t = time.time()
        idx = np.random.randint(0, len(self.replay_buffer['observation']),size=self.cfg.batch_size)
        batch = {'observation':self.replay_buffer['observation'][idx], 'action': self.replay_buffer['action'][idx],'next_obs_hash':self.replay_buffer['next_observation_hash'][idx], 'next_observation':self.replay_buffer['next_observation'][idx], 'terminated': self.replay_buffer['terminated'][idx]}
        batch = {k: torch.tensor(v, device=self.cfg.device) for k, v in batch.items()}
        
        # if isinstance(replay_buffer, dict):
        #     batch = replay_buffer["train"].sample(self.cfg.batch_size)
        # else:
        #     batch = replay_buffer.sample(self.cfg.batch_size)
        obs, action, next_obs, terminated = batch['observation'], batch['action'], batch['next_observation'], batch['terminated']
        next_obs_hash =  batch['next_obs_hash']
        discount = self.cfg.discount * ~terminated
        if hasattr(self, "norm_obs"):
            self.norm_obs(obs)
            self.norm_obs(next_obs)
            with utils.eval_mode(self):
                obs, next_obs = self.norm_obs(obs), self.norm_obs(next_obs)
        # if isinstance(replay_buffer, dict):
        #     batch = replay_buffer["train"].sample(self.cfg.batch_size)
        # else:
        #     batch = replay_buffer.sample(self.cfg.batch_size)

        # obs, action, next_obs, terminated = batch.get("observation"), batch.get("action"), batch.get(("next", "observation")), batch.get(("next", "terminated"))
        discount = self.cfg.discount * ~terminated
        if hasattr(self, "norm_obs"):
            self.norm_obs(obs)
            self.norm_obs(next_obs)
            with utils.eval_mode(self):
                obs, next_obs = self.norm_obs(obs), self.norm_obs(next_obs)
        
        goal = self.get_phi_input(batch)
        # print(step)
        if step < 250000:
            z = self.sample_z_psm(self.cfg.batch_size, device=self.cfg.device)
            metrics.update(self.update_psm(obs=obs, action=action, discount=discount,
                                    next_obs=next_obs,next_obs_hash=next_obs_hash, goal=goal, z=z, step=step))
            for name in ["psm_psi", "phi"]:
                utils.soft_update_params(self.nets[name], self.targets[name], opt.target_tau)
        else:

            z = self.sample_z(self.cfg.batch_size, device=self.cfg.device)

            phi_input = goal
            perm = torch.randperm(self.cfg.batch_size)
            phi_input = phi_input[perm]

            if opt.mix_ratio > 0:
                mix_idxs: tp.Any = np.where(np.random.uniform(size=self.cfg.batch_size) < opt.mix_ratio)[0]
                with torch.no_grad(), utils.eval_mode(self):
                    mix_z = self.nets["phi"](phi_input[mix_idxs]).detach()
                if self.cfg.archi.norm_z:
                    mix_z = math.sqrt(mix_z.shape[1]) * F.normalize(mix_z, dim=1)
                z[mix_idxs] = mix_z

            metrics.update(self.update_sf(obs=obs, action=action, discount=discount,
                                        next_obs=next_obs, goal=goal, z=z, step=step))
            metrics.update(self.update_actor(obs=obs, action=action, z=z, step=step))

            for name in ["sf_psi"]:
                utils.soft_update_params(self.nets[name], self.targets[name], opt.target_tau)
            
        metrics.update({"update_time": time.time() - start_t})

        # Get all metrics as normal numbers
        # TODO this still slows down training to have to wait these results
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                metrics[k] = v.item()

        return metrics

    ##################################
    # Reward-based inference methods
    ##################################

    @register("reward", "reward")
    def infer_context_from_reward(self, td: tp.Optional[TensorDict] = None, **kwargs: tp.Any) -> TensorDict:
        # batch must contain obs, action, and next_obs, each being a Nxd tensor (N: num samples, d: state or action dimension)
        if td is None:
            return TensorDict({"z": self.sample_z(1, device=self.cfg.device)}, batch_size=[1])
        td['next_observation'] = td["next"]["observation"]
        goal = self.get_phi_input(td)
        with torch.no_grad(), utils.eval_mode(self):
            phi = self.nets["phi"](goal)
        z = torch.matmul(td["reward"].T, phi) / phi.shape[0]
        if self.cfg.archi.norm_z:
            z = math.sqrt(z.shape[1]) * F.normalize(z, dim=1)
        return TensorDict({"z": z}, batch_size=[z.shape[0]])

    ##################################
    # Goal-based inference methods
    ##################################

    @register("goal-based", "goal-based")
    def infer_context_from_goal(self, td: TensorDict, **kwargs: tp.Any) -> TensorDict:
        # goal must contain obs, action, and next_obs, each being a Nxd tensor (N: number of goals, d: state or action dimension)
        goal = self.get_phi_input(td)
        with torch.no_grad(), utils.eval_mode(self):
            z = self.nets["phi"](goal)
        if self.cfg.archi.norm_z:
            z = math.sqrt(z.shape[1]) * F.normalize(z, dim=1)
        return TensorDict({"z": z}, batch_size=[z.shape[0]])

    ##################################
    # Imitation inference methods
    ##################################

    # This function implements average-state matching. TODO implement the other methods
    @register("imitation", "imitation")
    def infer_context_from_demo(self, trajectory: TensorDict, **kwargs: tp.Any) -> TensorDict:
        # trajectory must contain obs, action, and next_obs, each being a Nxd tensor (N: num samples, d: state or action dimension)
        goal = self.get_phi_input(trajectory)
        with torch.no_grad(), utils.eval_mode(self):
            phi = self.nets["phi"](goal)
        z = phi.mean(dim=0, keepdim=True)
        if self.cfg.archi.norm_z:
            z = math.sqrt(z.shape[1]) * F.normalize(z, dim=1)
        return TensorDict({"z": z}, batch_size=[z.shape[0]])

    ##################################
    # Tracking inference methods
    ##################################

    @register("tracking", "tracking")
    def infer_context_from_tracking(self, trajectory: TensorDict, step: int, lookahead: int, **kwargs: tp.Any) -> TensorDict:
        # trajectory must contain obs, action, and next_obs, each being a Txd tensor (T: horizon, d: state or action dimension)
        goal = self.get_phi_input(trajectory)
        with torch.no_grad(), utils.eval_mode(self):
            z = self.nets["phi"](goal)
        if self.cfg.archi.norm_z:
            z = math.sqrt(z.shape[1]) * F.normalize(z, dim=1)
        for step in range(0, z.shape[0]):
            z[step] = z[min(step + lookahead, z.shape[0]-1)]
        return TensorDict({"z": z}, batch_size=[z.shape[0]])
