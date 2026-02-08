import os, pickle, sys
# sys.path.append(os.path.dirname(__file__) + '/..')
import random
import time
from dataclasses import dataclass, field, asdict
from typing import List

import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from rl_tasks.repeat_first import RepeatFirstEnv, RepeatFirstPenEnv
from rl_tasks.repeat_first_popgym import RepeatFirstPopgymEnv
from rl_tasks.repeat_previous_popgym import RepeatPreviousPopgymEnv
from rl_tasks.copy_task import CopyTaskEnv
from recurrent_units.mapping import LRU_MAPPING
from recurrent_wrappers.create_models import WRAPPERS
from ppo_discrete_agents import create_agent
from utils import get_env_type
torch.set_num_threads(1)

from stream_components.normalization_wrappers import NormalizeObservation, ScaleReward
from aux_wrappers.previous_action_wrapper import PreviousAction
from aux_wrappers.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
gym.register_envs(ale_py)

@dataclass
class Args:
    exp_id: int = 0
    """identification of this experiment"""
    exp_name: str = ""
    """the name of this experiment"""
    notes: str = ""
    """notes for this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "Acrobot-v1"
    """the id of the environment"""
    corridor_length: int = 10
    """corridor length for TMaze"""
    total_timesteps: int = 2000000
    """total timesteps of the experiments"""
    hidden_size: int = 64
    """hidden size of the agent"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 200
    """the number of steps to run in each environment per policy rollout"""
    num_sequences: int = 5
    """the number of sequences an environment's rollout is split into for training per epoch"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.1
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""

    # bookeeping
    params_to_track: List[str] = field(default_factory=list)
    """params to track into folder tree to be compatible with plotting code"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    grad_log_interval: int = 0
    """log LRU grad norms every N optimization steps (0 disables)"""
    seq_grad_log_interval: int = 0
    """log per-timestep grad norms through LRU input every N optimization steps (0 disables)"""
    dormancy_log_interval: int = 0
    """log dormant neuron stats every N iterations (0 disables)"""
    dormancy_threshold: float = 0.025
    """normalized activity threshold for dormant neurons"""
    auto_track: bool = False
    """track dormancy and grad stats every iteration at a fixed point in the update"""
    use_default_hyperparams: bool = False
    """set hyperparameters to preset values, mostly to reproduce baselines' performance"""

    # architecture
    arch: str = "bestnet"
    """deep neural architecture"""
    rnn_type: str = "glru"
    """memory type of the agent"""
    rnn_hidden_size: int = 64
    """memory hidden units size"""
    rnn_mode: str = "sequential"  # "sequential", "convergence", "pondernet", "act"
    """computation mode of rnn"""
    rnn_init: str = "stored" # "zero", "stored"
    """rnn states initialization strategy after sampling from buffer"""
    rnn_burnin: float = 0.2
    """portion of training sequence to produce an rnn start state without weight updates"""
    ponder_eps: float = 0.5
    """threshold for halting"""
    ponder_n: int = 8
    """max pondering steps (safety limit)"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def create_env(env_id, seq_len=10):
    if env_id == "RepeatFirst":
        env = RepeatFirstEnv(seq_len=seq_len)
        episode_len = env.get_episode_length()
    elif env_id == "RepeatFirstPen":
        env = RepeatFirstPenEnv(seq_len=seq_len)
        episode_len = env.get_episode_length()
    elif env_id == "RepeatFirstPopgym":
        env = RepeatFirstPopgymEnv(seq_len=seq_len)
        episode_len = env.get_episode_length()
    elif env_id == "RepeatPreviousPopgym":
        env = RepeatPreviousPopgymEnv(seq_len=seq_len)
        episode_len = env.get_episode_length()
    elif env_id == "CopyTask":
        env = CopyTaskEnv(seq_len=seq_len)
        episode_len = env.get_episode_length()
    else:
        env = gym.make(env_id)
        episode_len = None
    return env, episode_len

def wrap_env(env, env_type):
    if env_type == 'ale_py':
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
    else:
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = ScaleReward(env, gamma=0.99)
        env = NormalizeObservation(env)
        env = PreviousAction(env, mode="concat")
    return env

def make_env(env_id, seq_len=10, env_type='custom'):
    def thunk():
        env, _ = create_env(env_id, seq_len=seq_len)
        env = wrap_env(env, env_type)
        return env
    return thunk

def _serialize_obs(obs):
    if isinstance(obs, (int, float, np.integer, np.floating)):
        return float(obs) if isinstance(obs, (np.floating, float)) else int(obs)
    try:
        return np.array(obs).tolist()
    except Exception:
        return obs

def get_unwrapped_state(env, fallback_obs):
    unwrapped_state = getattr(env.unwrapped, "state", None)
    if unwrapped_state is None:
        return _serialize_obs(fallback_obs)
    return _serialize_obs(unwrapped_state)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def lru_grad_norms(agent: nn.Module):
    norms = {}
    for mod_name, mod in agent.named_modules():
        if not isinstance(mod, WRAPPERS):
            continue
        submods = []
        if hasattr(mod, "ln1"):
            submods.append(("ln1", mod.ln1))
        submods.append(("rnn", mod.rnn))
        if hasattr(mod, "ln2"):
            submods.append(("ln2", mod.ln2))
        submods.append(("mlp", mod.mlp))
        if hasattr(mod, "ln3"):
            submods.append(("ln3", mod.ln3))
        for sub_name, sub in submods:
            total = 0.0
            has_grad = False
            for p in sub.parameters(recurse=True):
                if p.grad is None:
                    continue
                g = p.grad.detach()
                total += g.norm(2).item() ** 2
                has_grad = True
            if has_grad:
                norms[f"{mod_name}.{sub_name}"] = total ** 0.5
    return norms

def lru_seq_grad_norms(agent: nn.Module):
    norms = {}
    for mod_name, mod in agent.named_modules():
        if not isinstance(mod, WRAPPERS):
            continue
        if mod._last_seq_grad is not None:
            norms[mod_name] = mod._last_seq_grad
    return norms

def _compute_dormancy_stats(
    embedder: nn.Module,
    rnn: nn.Module,
    rnn_type: str,
    rnn_hidden_size: int,
    obs: torch.Tensor,
    dones: torch.Tensor,
    episode_len: int,
    threshold: float,
):
    if episode_len is None:
        return None
    total_steps, batch_size = obs.shape[0], obs.shape[1]
    usable_steps = (total_steps // episode_len) * episode_len
    if usable_steps == 0:
        return None

    obs = obs[:usable_steps]
    dones = dones[:usable_steps]
    flat_obs = obs.reshape(-1, obs.shape[-1])
    x = embedder(flat_obs)
    if rnn_type in ["gru", "rnn"]:
        x = F.relu(x)
    x = x.reshape(usable_steps, batch_size, -1)
    done_seq = dones.reshape(usable_steps, batch_size)
    h0 = torch.zeros(batch_size, rnn_hidden_size, device=obs.device)

    try:
        _, _, aux = rnn(x, h0, done=done_seq, track_activity=True)
    except TypeError:
        return None

    if not isinstance(aux, dict):
        return None
    rnn_out = aux.get("rnn_out")
    mlp_out = aux.get("mlp_out")
    if rnn_out is None or mlp_out is None:
        return None

    num_sequences = usable_steps // episode_len
    rnn_out = rnn_out.reshape(num_sequences, episode_len, batch_size, -1)
    rnn_activity = rnn_out.abs().mean(dim=(0, 2))
    rnn_norm = rnn_activity / (rnn_activity.mean(dim=1, keepdim=True) + 1e-8)
    rnn_dormant = (rnn_norm < threshold).float().mean(dim=1)

    mlp_out = mlp_out.reshape(num_sequences, episode_len, batch_size, -1)
    mlp_activity = mlp_out.abs().mean(dim=(1, 2))
    mlp_norm = mlp_activity / (mlp_activity.mean(dim=1, keepdim=True) + 1e-8)
    mlp_dormant = (mlp_norm < threshold).float().mean(dim=1)
    mlp_dormant_mean = mlp_dormant.mean()

    return rnn_dormant, mlp_dormant_mean


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    assert (args.num_envs * args.num_sequences) % args.num_minibatches == 0
    seqsperminibatch = (args.num_envs * args.num_sequences) // args.num_minibatches
    seqlength = args.num_steps // args.num_sequences
    seqinds = np.arange(args.num_envs * args.num_sequences)
    flatinds = (
        np.arange(args.batch_size)
        .reshape(args.num_steps, args.num_envs)
        .transpose(1, 0)
        .reshape(args.num_envs, args.num_sequences, seqlength)
        .transpose(2, 0, 1)
        .reshape(seqlength, args.num_envs * args.num_sequences)
    )
    # flatinds = np.arange(args.batch_size).reshape(seqlength, args.num_envs * args.num_sequences)

    print(' '.join(sys.argv))
    script_name = os.path.basename(sys.argv[0])
    run_name = f"{script_name}__exp_id_{args.exp_id}"
    print(run_name)
    if len(args.exp_name) > 0:
        print(args.exp_name, end='\n\n')
    print(args, end='\n\n')
    # enforce rnn combination contraints
    if args.rnn_type not in LRU_MAPPING:
        args.rnn_mode = 'sequential'
    if args.rnn_type == 'mlp':
        args.rnn_init = 'zero'

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_type = get_env_type(args.env_id)
    env, episode_len = create_env(args.env_id, args.corridor_length)
    # env = wrap_env(env, env_type)
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.corridor_length, env_type) for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
    )
    # envs = wrap_env(envs, env_type)
    
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = create_agent(
        envs=envs,
        env_type=env_type,
        rnn_unit=args.rnn_type, 
        arch=args.arch,
        d_model=args.hidden_size, 
        d_state=args.rnn_hidden_size, 
        rnn_mode=args.rnn_mode,
        ponder_eps=args.ponder_eps,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    num_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print("Number of parameters:", num_params)

    

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    next_obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    obs_curr, _ = envs.reset(seed=args.seed)
    obs_curr = torch.Tensor(obs_curr).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    # Initialize RNN state
    if args.rnn_type == "mlp":
        actor_rnn_state_env_lv = None
        critic_rnn_state_env_lv = None
        default_rnn_seq_lv = None
    else:
        actor_rnn_state_env_lv = torch.zeros(args.num_envs, args.rnn_hidden_size).to(device)
        critic_rnn_state_env_lv = torch.zeros(args.num_envs, args.rnn_hidden_size).to(device)
        default_rnn_seq_lv = torch.zeros((args.num_envs, args.num_sequences, args.rnn_hidden_size)).to(device)

    iteration_returns = []
    iteration_timesteps = []
    episode_returns = []
    episode_count = 0
    episode_trajectory = []
    env0_episode_count = 0
    rollout_log_interval = 1000

    timing_stats = {
        'forward': 0.0,
        'loss_computation': 0.0,
        'backward_and_step': 0.0,
    }

    last_log_time = start_time
    last_step_count = 0
    episodes_since_log = 0
    update_step = 0
    tracking_history = {
        "iterations": [],
        "actor_rnn_dormancy": [],
        "actor_mlp_dormancy": [],
        "grad_norms": [],
        "seq_grad_norms": [],
    }

    for iteration in range(1, args.num_iterations + 1):
        iter_grad_norms = None
        iter_seq_grad_norms = None
        iter_rnn_dormancy = None
        iter_mlp_dormancy = None
        mid_epoch_idx = args.update_epochs // 2
        mid_minibatch_idx = args.num_minibatches // 2
        if args.rnn_type != 'mlp': # same for lstm and lru
            # used for training, so we index by seq level, not env level
            initial_actor_rnn_state = default_rnn_seq_lv.clone()
            initial_critic_rnn_state = default_rnn_seq_lv.clone()
            # will be filled with rnn_state_lv that stores persistent data if stored
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = obs_curr
            dones[step] = next_done
            if args.rnn_init == 'stored' and step % seqlength == 0:
                # storing rnn states if needed
                if actor_rnn_state_env_lv is not None:
                    initial_actor_rnn_state[:, step // seqlength, :] = actor_rnn_state_env_lv
                if critic_rnn_state_env_lv is not None:
                    initial_critic_rnn_state[:, step // seqlength, :] = critic_rnn_state_env_lv

            # ALGO LOGIC: action logic
            agent.eval()
            t0 = time.time()
            with torch.no_grad():
                action, logprob, _, actor_rnn_state_env_lv, aux_data_rollout = agent.get_action(
                    x=obs_curr,
                    rnn_state=actor_rnn_state_env_lv,
                    done=next_done,
                    rnn_burnin=0.0,
                    batch_size=args.num_envs,
                )
                value, critic_rnn_state_env_lv = agent.get_value(
                    x=obs_curr,
                    rnn_state=critic_rnn_state_env_lv,
                    done=next_done,
                    rnn_burnin=0.0,
                    batch_size=args.num_envs,
                )
                values[step] = value.flatten()
            timing_stats['forward'] += time.time() - t0
            agent.train()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            prev_state = get_unwrapped_state(envs.envs[0], obs_curr[0].cpu().numpy())
            obs_curr, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            next_state = get_unwrapped_state(envs.envs[0], obs_curr[0])
            if env_type in ['custom']:
                episode_trajectory.append(
                    {
                        "s": prev_state,
                        "a": int(action[0].item()) if len(action.shape) == 1 else [f"{v:.2f}" for v in action[0].tolist()],
                        "r": float(reward[0]),
                        "s_prime": next_state,
                        "done": bool(next_done[0]),
                    }
                )
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            actual_next_obs = np.copy(obs_curr)

            assert next_done.any() == ("final_obs" in infos)
            if "final_obs" in infos and '_episode' in infos['final_info']:
                for env_idx in range(len(infos["final_obs"])):
                    if infos['final_info']["_episode"][env_idx]:
                        actual_next_obs[env_idx] = infos["final_obs"][env_idx]
                        episode_count += 1
                        episodes_since_log += 1
                        total_return = infos['final_info']["episode"]["r"][env_idx]
                        episode_returns.append(total_return)
                        if env_idx == 0:
                            env0_episode_count += 1
                            if rollout_log_interval > 0 and episode_trajectory and env0_episode_count % rollout_log_interval == 0:
                                print(f"\nEnv0 episode {env0_episode_count} rollout (return {total_return:.2f}):")
                                for t, transition in enumerate(episode_trajectory):
                                    print(f"  t={t}: {transition}")
                                print()
                            episode_trajectory = []

            obs_curr, next_done = torch.Tensor(obs_curr).to(device), torch.Tensor(next_done).to(device)
            next_obs[step] = torch.Tensor(actual_next_obs).to(device)

        track_dormancy = (
            args.rnn_type != "mlp"
            and (
                args.auto_track
                or (args.dormancy_log_interval > 0 and iteration % args.dormancy_log_interval == 0)
            )
        )
        if track_dormancy:
            agent.eval()
            with torch.no_grad():
                actor_stats = _compute_dormancy_stats(
                    agent.actor_embedder,
                    agent.actor,
                    agent.rnn_type,
                    args.rnn_hidden_size,
                    obs,
                    dones,
                    episode_len,
                    args.dormancy_threshold,
                )
                critic_stats = _compute_dormancy_stats(
                    agent.critic_embedder,
                    agent.critic,
                    agent.rnn_type,
                    args.rnn_hidden_size,
                    obs,
                    dones,
                    episode_len,
                    args.dormancy_threshold,
                )
            agent.train()
            if actor_stats is not None:
                rnn_dormant, mlp_dormant = actor_stats
                iter_rnn_dormancy = rnn_dormant.detach().cpu()
                iter_mlp_dormancy = mlp_dormant.item()
                rnn_items = ", ".join([f"{v:.3f}" for v in rnn_dormant.tolist()])
                if iteration % 20 == 0:
                    print(
                        f"Dormancy (actor) rnn: [{rnn_items}], "
                        f"mlp: {mlp_dormant:.3f}"
                    )
            if critic_stats is not None:
                rnn_dormant, mlp_dormant = critic_stats
                rnn_items = ", ".join([f"{v:.3f}" for v in rnn_dormant.tolist()])
                if iteration % 20 == 0:
                    print(
                        f"Dormancy (critic) rnn: [{rnn_items}], "
                        f"mlp: {mlp_dormant:.3f}"
                    )

        # compute advantage
        agent.eval()
        with torch.no_grad():
            next_value, _= agent.get_value(
                x=next_obs[-1],
                rnn_state=critic_rnn_state_env_lv,
                done=next_done,
                rnn_burnin=0.0,
                batch_size=args.num_envs,
            )
            next_value = next_value.reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                # truncation and termination are treated the same
                # to distinguish, needs to store next_obs_values separate from values
                # use nextvalues = next_obs_values[t] instead of values[t+1]
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        agent.train()

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        if args.rnn_type != 'mlp':
            initial_actor_rnn_state = initial_actor_rnn_state.reshape(-1, args.rnn_hidden_size)
            initial_critic_rnn_state = initial_critic_rnn_state.reshape(-1, args.rnn_hidden_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(seqinds)
            # sample a minibatch from collected batch, without replacement
            for mb_idx, start in enumerate(range(0, args.num_envs * args.num_sequences, seqsperminibatch)):
                end = start + seqsperminibatch
                mbseqinds = seqinds[start:end]
                mb_inds = flatinds[:, mbseqinds].ravel()
                if args.rnn_type != 'mlp':
                    mb_actor_rnn_state = initial_actor_rnn_state[mbseqinds]
                    mb_critic_rnn_state = initial_critic_rnn_state[mbseqinds]
                else:
                    mb_actor_rnn_state = None
                    mb_critic_rnn_state = None
                t0 = time.time()
                is_midpoint = (
                    args.auto_track
                    and epoch == mid_epoch_idx
                    and mb_idx == mid_minibatch_idx
                )
                track_seq_grad = (
                    is_midpoint
                    if args.auto_track
                    else (
                        args.seq_grad_log_interval > 0
                        and (update_step + 1) % args.seq_grad_log_interval == 0
                    )
                )
                _, newlogprob, entropy, _, aux_data = agent.get_action(
                    x=b_obs[mb_inds],
                    rnn_state=mb_actor_rnn_state,
                    done=b_dones[mb_inds],
                    action=b_actions[mb_inds],
                    rnn_burnin=args.rnn_burnin,
                    batch_size=seqsperminibatch,
                    track_seq_grad=track_seq_grad,
                )
                newvalue, _ = agent.get_value(
                    x=b_obs[mb_inds],
                    rnn_state=mb_critic_rnn_state,
                    done=b_dones[mb_inds],
                    rnn_burnin=args.rnn_burnin,
                    batch_size=seqsperminibatch,
                    track_seq_grad=track_seq_grad,
                )
                timing_stats['forward'] += time.time() - t0
                t0 = time.time()
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                timing_stats['loss_computation'] += time.time() - t0

                t0 = time.time()
                optimizer.zero_grad()
                loss.backward()
                update_step += 1
                log_grad_norms = (
                    is_midpoint
                    if args.auto_track
                    else (args.grad_log_interval > 0 and update_step % args.grad_log_interval == 0)
                )
                if log_grad_norms:
                    grad_norms = lru_grad_norms(agent)
                    if grad_norms:
                        iter_grad_norms = grad_norms
                        items = ", ".join(
                            [f"{k}={v:.2e}" for k, v in sorted(grad_norms.items())]
                        )
                        if iteration % 20 == 0:
                            print(f"LRU grad norms: {items}")
                log_seq_grad_norms = (
                    is_midpoint
                    if args.auto_track
                    else (args.seq_grad_log_interval > 0 and update_step % args.seq_grad_log_interval == 0)
                )
                if log_seq_grad_norms:
                    seq_grad_norms = lru_seq_grad_norms(agent)
                    if seq_grad_norms:
                        iter_seq_grad_norms = {
                            name: {
                                "min": g.min().item(),
                                "mean": g.mean().item(),
                                "max": g.max().item(),
                            }
                            for name, g in seq_grad_norms.items()
                        }
                        items = []
                        for name, g in sorted(seq_grad_norms.items()):
                            stats = f"min={g.min():.2e},mean={g.mean():.2e},max={g.max():.2e}"
                            items.append(f"{name}({stats})")
                        if iteration % 20 == 0:
                            print(f"LRU seq grad norms: {', '.join(items)}")
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                timing_stats['backward_and_step'] += time.time() - t0

        # Log at every iteration
        elapsed_time = time.time() - last_log_time
        steps_since_last_log = global_step - last_step_count
        
        sps = steps_since_last_log / elapsed_time if elapsed_time > 0 else 0
        avg_reward = sum(episode_returns) / episodes_since_log if episodes_since_log > 0 else np.nan
        iteration_returns.append(avg_reward)
        iteration_timesteps.append(global_step)
        tracking_history["iterations"].append(iteration)
        tracking_history["actor_rnn_dormancy"].append(
            None if iter_rnn_dormancy is None else iter_rnn_dormancy.tolist()
        )
        tracking_history["actor_mlp_dormancy"].append(iter_mlp_dormancy)
        tracking_history["grad_norms"].append(iter_grad_norms)
        tracking_history["seq_grad_norms"].append(iter_seq_grad_norms)

        # if iteration % 20 == 0:
        print(f"{iteration}/{args.num_iterations} (Episode {episode_count}, Step {global_step}): SPS={int(sps)}, avg_return={avg_reward:.2f}\n")
        # timing_breakdown = ", ".join([f"{k}={v/elapsed_time*100:.1f}%" for k, v in timing_stats.items()])
        # print(f"Timing breakdown: {timing_breakdown}")
        
        episode_returns = []
        episodes_since_log = 0
        
        for key in timing_stats:
            timing_stats[key] = 0.0     
       
        last_log_time = time.time()
        last_step_count = global_step

    envs.close()

    total_time = time.time() - start_time
    print()
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Total episodes: {episode_count}, Total steps: {global_step}")

    exp_dir = f"data_{run_name}"
    if not os.path.exists(exp_dir):
        try:
            os.makedirs(exp_dir)
        except FileExistsError:
            pass

    notes_path = os.path.join(exp_dir, f"notes_{args.exp_id}.txt")
    try:
        fd = os.open(notes_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            command = f"python utils/plot.py --dir {exp_dir} --int_space 0 --total_steps {args.total_timesteps} "
            if args.exp_name != "":
                command += f" --exp_name {args.exp_name}"
            if args.notes != "":
                command += f" --notes {args.notes}"
            f.write(f"{command}\n\n")
            for key, value in asdict(args).items():
                f.write(f"{key}: {value}\n")
            f.write(f"\n{' '.join(sys.argv)}\n")
    except FileExistsError:
        pass

    run_dir = os.path.join(exp_dir, f"{args.env_id}_{args.corridor_length}")
    if not os.path.exists(run_dir):
        try:
            os.makedirs(run_dir)
        except FileExistsError:
            pass

    for param in args.params_to_track:
        if not hasattr(args, param):
            print(f'{param} not in args')
            continue
        run_dir = os.path.join(run_dir, f"{param}__{getattr(args, param)}")
        if not os.path.exists(run_dir):
            try:
                os.makedirs(run_dir)
            except FileExistsError:
                pass

    with open(os.path.join(run_dir, f"seed_{args.seed}.pkl"), "wb") as f:
        pickle.dump(
            {
                "returns": iteration_returns,
                "timesteps": iteration_timesteps,
                "tracking": tracking_history,
            },
            f,
        )

    print(f"Data saved to {run_dir}, seed {args.seed}")
    # evaluate_agent(args.env_id, agent, run_name, num_episodes=3, seed=args.seed)
