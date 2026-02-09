import os, pickle, sys
# sys.path.append(os.path.dirname(__file__) + '/..')
import random
import time
from dataclasses import dataclass, field, asdict
from typing import List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from rl_tasks.repeat_first import RepeatFirstBlankLastEnv
from rl_tasks.copy_task import CopyTaskEnv
from recurrent_units.mapping import LRU_MAPPING
from recurrent_wrappers.create_models import create_model
torch.set_num_threads(1)

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
    params_to_track: List[str] = field(default_factory=list)

    # Algorithm specific arguments
    env_id: str = "CopyTask"
    """the id of the environment"""
    corridor_length: int = 5
    """corridor length for TMaze"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    hidden_size: int = 64
    """hidden size of the agent"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_sequences: int = 1
    """the number of sequences an environment's rollout is split into for training per epoch"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
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
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""

    # RNN architecture
    rnn_type: str = "lru"
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


def make_env(env_id, seq_len=10):
    def thunk():
        if env_id == "RepeatFirst":
            env = RepeatFirstBlankLastEnv(seq_len=seq_len)
        elif env_id == "CopyTask":
            env = CopyTaskEnv(seq_len=seq_len)
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def init_weights(m, std=1.0, bias_const=0.0):
    if isinstance(m, nn.LayerNorm):
        if m.weight is not None: nn.init.constant_(m.weight, 1.0)
        if m.bias   is not None: nn.init.constant_(m.bias,   0.0)
        return
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, std)
        if m.bias is not None:
            nn.init.constant_(m.bias, bias_const)
        return
    for _, p in m.named_parameters(recurse=False):
        if p is None: continue
        if p.ndim < 2:
            nn.init.constant_(p, bias_const)
        else:
            nn.init.orthogonal_(p, std)

class Agent(nn.Module):
    def __init__(self, n_obs, n_actions, rnn_unit="lstm", d_model=64, d_state=64, rnn_mode="sequential", ponder_eps=0.01):
        super().__init__()
        self.rnn_type = rnn_unit
        self.rnn_mode = rnn_mode

        self.actor_embedder = layer_init(nn.Linear(n_obs, d_model), std=1)
        self.actor, self.outdim = create_model(
            name=rnn_unit,
            embed_dim=d_model,
            hidden_dim=d_state,
            rnn_mode=rnn_mode,
            use_layernorm=True)
        self.actor_readout = layer_init(nn.Linear(self.outdim, n_actions), std=0.01)
        self.actor.apply(init_weights)

        self.critic_embedder = layer_init(nn.Linear(n_obs, d_model), std=1)
        self.critic, self.outdim = create_model(
            name=rnn_unit,
            embed_dim=d_model,
            hidden_dim=d_state,
            rnn_mode=rnn_mode,
            use_layernorm=True)
        self.critic_readout = layer_init(nn.Linear(self.outdim, 1), std=0.01)
        self.critic.apply(init_weights)
        
    def get_states(self, x, rnn_state, done, network='actor', rnn_burnin=0.0):
        if network == 'actor':
            embedder = self.actor_embedder
            rnn = self.actor
        elif network == 'critic':
            embedder = self.critic_embedder
            rnn = self.critic
        x = embedder(x)
        if self.rnn_type in ["gru", "rnn"]:
            x = F.relu(x)
        if self.rnn_type == 'mlp':
            return rnn(x)
        batch_size = rnn_state.shape[0]
        seq_len = x.shape[0] // batch_size
        x = x.reshape((seq_len, batch_size, -1))
        done_reshaped = done.reshape((seq_len, batch_size))
        x, rnn_state, aux_data = rnn(x, rnn_state, done=done_reshaped)
        x = x.reshape(-1, self.outdim)
        return x, rnn_state, aux_data

    def get_value(self, x, rnn_state, done, rnn_burnin=0.0):
        x, critic_rnn_state, _ = self.get_states(x, rnn_state, done, network='critic', rnn_burnin=rnn_burnin)
        return self.critic_readout(x), critic_rnn_state

    def get_action(self, x, rnn_state, done, action=None, rnn_burnin=0.0):
        x, rnn_state, aux_data = self.get_states(x, rnn_state, done, network='actor', rnn_burnin=rnn_burnin)
        logits = self.actor_readout(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        # don't return critic value here. Instead, needs to call get_value because we need to unroll critic rnn
        return action, probs.log_prob(action), probs.entropy(), rnn_state, aux_data

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    assert (args.num_envs * args.num_sequences) % args.num_minibatches == 0
    # concerns of bugs for num_sequences > 1
    # claim: currently mixes timesteps across sequences, so RNN unroll sees non‑contiguous time
    # TODO: needs to check if this claim is true
    seqsperminibatch = (args.num_envs * args.num_sequences) // args.num_minibatches
    seqlength = args.num_steps // args.num_sequences
    seqinds = np.arange(args.num_envs * args.num_sequences)
    flatinds = np.arange(args.batch_size).reshape(seqlength, args.num_envs * args.num_sequences)

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
    env = make_env(args.env_id)()
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.corridor_length) for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(
        n_obs=np.prod(envs.single_observation_space.shape), 
        n_actions=envs.single_action_space.n,
        rnn_unit=args.rnn_type, 
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

    timing_stats = {
        'forward': 0.0,
        'loss_computation': 0.0,
        'backward_and_step': 0.0,
    }

    last_log_time = start_time
    last_step_count = 0
    episodes_since_log = 0

    for iteration in range(1, args.num_iterations + 1):
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
                initial_actor_rnn_state[:, step // seqlength, :] = actor_rnn_state_env_lv
                initial_critic_rnn_state[:, step // seqlength, :] = critic_rnn_state_env_lv

            # ALGO LOGIC: action logic
            agent.eval()
            t0 = time.time()
            with torch.no_grad():
                action, logprob, _, actor_rnn_state_env_lv, aux_data_rollout = agent.get_action(obs_curr, actor_rnn_state_env_lv, next_done, rnn_burnin=0.0)
                value, critic_rnn_state_env_lv = agent.get_value(obs_curr, critic_rnn_state_env_lv, next_done, rnn_burnin=0.0)
                values[step] = value.flatten()
            timing_stats['forward'] += time.time() - t0
            agent.train()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            obs_curr, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            actual_next_obs = np.copy(obs_curr)

            assert next_done.any() == ("final_obs" in infos)
            if "final_obs" in infos:
                for env_idx in range(len(infos["final_obs"])):
                    if infos['final_info']["_episode"][env_idx]:
                        actual_next_obs[env_idx] = infos["final_obs"][env_idx]
                        episode_count += 1
                        episodes_since_log += 1
                        episode_returns.append(infos['final_info']["episode"]["r"][env_idx])

            obs_curr, next_done = torch.Tensor(obs_curr).to(device), torch.Tensor(next_done).to(device)
            next_obs[step] = torch.Tensor(actual_next_obs).to(device)

        # compute advantage
        agent.eval()
        with torch.no_grad():
            next_value, _= agent.get_value(next_obs[-1], critic_rnn_state_env_lv, next_done, rnn_burnin=0.0)
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
            for start in range(0, args.num_envs * args.num_sequences, seqsperminibatch):
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
                _, newlogprob, entropy, _, aux_data = agent.get_action(
                    b_obs[mb_inds],
                    mb_actor_rnn_state,
                    b_dones[mb_inds],
                    b_actions[mb_inds],
                    rnn_burnin=args.rnn_burnin
                )
                newvalue, _ = agent.get_value(b_obs[mb_inds], mb_critic_rnn_state, b_dones[mb_inds], rnn_burnin=args.rnn_burnin)
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
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                timing_stats['backward_and_step'] += time.time() - t0

        # Log at every iteration
        elapsed_time = time.time() - last_log_time
        steps_since_last_log = global_step - last_step_count
        
        sps = steps_since_last_log / elapsed_time if elapsed_time > 0 else 0
        avg_reward = sum(episode_returns) / episodes_since_log if episodes_since_log > 0 else 0
        iteration_returns.append(avg_reward)
        iteration_timesteps.append(global_step)

        print(f"\n{iteration}/{args.num_iterations} (Episode {episode_count}, Step {global_step}): SPS={int(sps)}, avg_return={avg_reward:.2f}")

        timing_breakdown = ", ".join([f"{k}={v/elapsed_time*100:.1f}%" for k, v in timing_stats.items()])
        print(f"Timing breakdown: {timing_breakdown}")
        
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

    exp_dir = f"data_rnn_variants_{run_name}"
    if not os.path.exists(exp_dir):
        try:
            os.makedirs(exp_dir)
        except FileExistsError:
            pass

    notes_path = os.path.join(exp_dir, f"notes_{args.exp_id}.txt")
    try:
        fd = os.open(notes_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            command = f"python utils/plot.py --dir {exp_dir} --total_steps {args.total_timesteps}"
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
        pickle.dump((iteration_returns, iteration_timesteps), f)

    print(f"Data saved to {run_dir}, seed {args.seed}")
    # evaluate_agent(args.env_id, agent, run_name, num_episodes=3, seed=args.seed)
