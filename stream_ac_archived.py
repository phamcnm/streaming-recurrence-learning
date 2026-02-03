import os, pickle, sys, argparse, time
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
from torch.distributions import Categorical
from stream_components.optim import ObGD as Optimizer
from stream_components.time_wrapper import AddTimeInfo
from stream_components.normalization_wrappers import NormalizeObservation, ScaleReward
from stream_components.sparse_init import sparse_init
from rl_tasks.repeat_first import RepeatFirstBlankLastEnv
from rl_tasks.copy_task import CopyTaskEnv
from recurrent_units.mapping import LRU_MAPPING
from recurrent_wrappers.create_models import create_model
import matplotlib.pyplot as plt
torch.set_num_threads(1)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        number_of_inputs = 8
        _, fan_in = m.weight.shape
        sparsity_factor = number_of_inputs / fan_in
        sparse_init(m.weight, sparsity=1-sparsity_factor)
        m.bias.data.fill_(0.0)

class Actor(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, d_model=32, d_state=32, recurrent_unit='lru', ponder_mode='sequential', ponder_eps=0.1):
        super(Actor, self).__init__()
        self.fc_layer = nn.Linear(n_obs, d_model)
        self.recurrent_unit = recurrent_unit
        self.rnn, rnn_dim = create_model(
            name=recurrent_unit,
            embed_dim=d_model,
            hidden_dim=d_state,
            rnn_mode=ponder_mode,
            layernorm=True
        )
        self.fc_pi = nn.Linear(rnn_dim, n_actions)
        self.apply(initialize_weights)

    def forward(self, x, hidden_state, apply_change=True, inner_loops: int = 1):
        x = self.fc_layer(x)
        if self.recurrent_unit in LRU_MAPPING:
            x, hidden_state, aux_data = self.rnn(x, hidden_state, inner_loops=inner_loops)
        else:
            x, hidden_state, aux_data = self.rnn(x)
        pref = self.fc_pi(x)
        return pref, hidden_state, aux_data

class Critic(nn.Module):
    def __init__(self, n_obs=11, d_model=32, d_state=32, recurrent_unit='lru', ponder_mode='sequential', repeat_mode='none', ponder_eps=0.1):
        super(Critic, self).__init__()
        self.fc_layer   = nn.Linear(n_obs, d_model)
        self.recurrent_unit = recurrent_unit
        self.use_q_head = repeat_mode=='ds_adaptive'
        self.rnn, rnn_dim = create_model(
            name=recurrent_unit,
            embed_dim=d_model,
            hidden_dim=d_state,
            rnn_mode=ponder_mode,
            layernorm=True
        )
        self.x = None # penultimate output
        if self.use_q_head:
            self.q_head = nn.Linear(rnn_dim, 1)
        self.linear_layer  = nn.Linear(rnn_dim, 1)
        self.apply(initialize_weights)

    def forward(self, x, hidden_state, apply_change=True, inner_loops: int = 1):
        x = self.fc_layer(x)
        if self.recurrent_unit in LRU_MAPPING:
            x, hidden_state, aux_data = self.rnn(x, hidden_state, inner_loops=inner_loops)
        else:
            x, hidden_state, aux_data = self.rnn(x)
        self.x = x.detach()
        val = self.linear_layer(x)
        return val, hidden_state, aux_data
    
    def get_penultimate_output(self):
        return self.x
    
    def compute_q(self, x):
        assert self.use_q_head
        q = self.q_head(x)
        return q

class StreamAC(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, d_model=32, d_state=32, 
                 lr=1.0, gamma=0.99, lamda=0.8, kappa_policy=3.0, kappa_value=2.0,
                 recurrent_unit='lru', ponder_mode='sequential', 
                 repeat_mode='none', ponder_eps=0.1, ponder_n=8):
        super(StreamAC, self).__init__()
        self.gamma = gamma
        self.policy_net = Actor(n_obs=n_obs, n_actions=n_actions, d_model=d_model, d_state=d_state, 
                                recurrent_unit=recurrent_unit, ponder_mode=ponder_mode, ponder_eps=ponder_eps)
        self.value_net = Critic(n_obs=n_obs, d_model=d_model, d_state=d_state, 
                                recurrent_unit=recurrent_unit, ponder_mode=ponder_mode, repeat_mode=repeat_mode, ponder_eps=ponder_eps)

        self.optimizer_policy = Optimizer(self.policy_net.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_policy)
        self.optimizer_value = Optimizer(self.value_net.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value)
        if repeat_mode == 'ds_adaptive':
            self.optimizer_q = Optimizer(self.value_net.q_head.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value)

        self.overshooting_global_counter = 0
        self.overshooting_local_counter = 0
        
        self.timing_stats = {
            'forward': 0.0,
            'loss_computation': 0.0,
            'backward_and_step': 0.0,
        }

    def reset_local_counters(self):
        self.overshooting_local_counter = 0
        
    def reset_timing_stats(self):
        for key in self.timing_stats:
            self.timing_stats[key] = 0.0

    def pi(self, x, hidden_state, apply_change=True, inner_loops: int = 1):
        x = x.unsqueeze(0).unsqueeze(0)
        preferences, next_hidden_state, aux_data = self.policy_net(x, hidden_state, apply_change=apply_change, inner_loops=inner_loops)
        probs = F.softmax(preferences, dim=-1)
        return probs, next_hidden_state, aux_data

    def v(self, x, hidden_state, apply_change=True, inner_loops: int = 1):
        x = x.unsqueeze(0).unsqueeze(0)
        v, next_hidden_state, aux_data = self.value_net(x, hidden_state, apply_change=apply_change, inner_loops=inner_loops)
        return v, next_hidden_state, aux_data

    def sample_action(self, s, hidden_state=None, apply_change=False, inner_loops: int = 1):
        x = torch.from_numpy(s).float()
        probs, next_hidden_state, _ = self.pi(x, hidden_state, apply_change=apply_change, inner_loops=inner_loops)
        dist = Categorical(probs)
        action = dist.sample().squeeze(0).squeeze(0).numpy()
        return action, next_hidden_state

    def update_params(self, s, a, r, s_prime, done, policy_hidden, value_hidden, entropy_coeff, inner_loops, critical_step, repeat_mode, last_repeat, overshooting_info=False):
        done_mask = 0 if done else 1
        s, a, r, s_prime, done_mask = torch.tensor(np.array(s), dtype=torch.float), torch.tensor(np.array(a)), \
                                         torch.tensor(np.array(r)), torch.tensor(np.array(s_prime), dtype=torch.float), \
                                         torch.tensor(np.array(done_mask), dtype=torch.float)
        v_s, next_value_hidden, aux_data = self.v(s, value_hidden, apply_change=True, inner_loops=inner_loops+critical_step)

        if repeat_mode == 'ds_adaptive':
            x = self.value_net.get_penultimate_output()
        
        v_prime, _, _ = self.v(s_prime, next_value_hidden, apply_change=False, inner_loops=inner_loops)
            
        td_target = r + self.gamma * v_prime * done_mask
        delta = td_target - v_s
        probs, next_policy_hidden, _ = self.pi(s, policy_hidden, apply_change=True, inner_loops=inner_loops+critical_step)
        dist = Categorical(probs)

        log_prob_pi = -(dist.log_prob(a)).sum()
        value_output = -v_s
        entropy_pi = -entropy_coeff * dist.entropy().sum() * torch.sign(delta).item()

        self.optimizer_value.zero_grad()
        self.optimizer_policy.zero_grad()
        value_output.backward()
        (log_prob_pi + entropy_pi).backward()
        
        # old_last_repeat = last_repeat
        if repeat_mode == 'ds_adaptive':
            with torch.no_grad():
                q_halt = self.value_net.compute_q(x)
                act_done = (F.sigmoid(q_halt)>= 0.5).item()
                last_repeat = last_repeat or act_done
        reset_e = done and last_repeat

        self.optimizer_policy.step(delta.item(), reset=reset_e)
        self.optimizer_value.step(delta.item(), reset=reset_e)

        if repeat_mode == 'ds_adaptive':
            q_halt = self.value_net.compute_q(x)
            aux_data = (aux_data, act_done)
            q_halt_target = torch.tensor((abs(delta.item()) < 0.15), dtype=torch.float)
            self.optimizer_q.zero_grad()
            q_halt.backward()
            delta_q = F.sigmoid(q_halt) - q_halt_target
            self.optimizer_q.step(delta_q.item(), reset=reset_e)

        if overshooting_info:
            # deprecated
            with torch.no_grad():
                v_s_current, next_value_hidden_after = self.v(s, value_hidden, apply_change=False, inner_loops=inner_loops)
                v_s_next, _ = self.v(s_prime, next_value_hidden_after, apply_change=False, inner_loops=inner_loops)
            td_target = r + self.gamma * v_s_next * done_mask
            delta_bar = td_target - v_s_current
            if torch.sign(delta_bar * delta).item() == -1:
                self.overshooting_global_counter += 1
                self.overshooting_local_counter += 1
        return next_policy_hidden, next_value_hidden, aux_data
    
def create_env(env_id, seq_len=10, render=False):
    if env_id == "RepeatFirst":
        env = RepeatFirstBlankLastEnv(seq_len=seq_len)
        episode_len = env.get_episode_length()
    elif env_id == "CopyTask":
        env = CopyTaskEnv(seq_len=seq_len)
        episode_len = env.get_episode_length()
    else:
        env, _ = gym.make(env_id)
        episode_len = None
    return env, episode_len

def main(exp_id, exp_name, seed, env_id, seq_len, total_timesteps, total_episodes, log_interval,
        lr, gamma, lamda,  entropy_coeff, d_model, d_state, kappa_policy, kappa_value,
        recurrent_unit, ponder_mode, ponder_eps, ponder_n, inner_loops, repeat_mode, 
        args, overshooting_info, debug=False, render=False):
    if debug:
        showfig = True
        if showfig:
            plt.ion()
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        ims = None
    if len(exp_name) > 0:
        print(exp_name, end='\n\n')
    if recurrent_unit not in LRU_MAPPING:
        ponder_mode = "sequential"
    success_counter = 0
    torch.manual_seed(seed); np.random.seed(seed)
    env, episode_len = create_env(env_id, seq_len, render)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = ScaleReward(env, gamma=gamma)
    # env = NormalizeObservation(env)
    # env = AddTimeInfo(env)
    policy_hidden, value_hidden = None, None
    agent = StreamAC(
        n_obs=env.observation_space.shape[0], n_actions=env.action_space.n, d_model=d_model, d_state=d_state, 
        lr=lr, gamma=gamma, lamda=lamda, kappa_policy=kappa_policy, kappa_value=kappa_value, repeat_mode=repeat_mode,
        recurrent_unit=recurrent_unit, ponder_mode=ponder_mode, ponder_eps=ponder_eps)
    num_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print("Number of parameters:", num_params)

    if total_episodes is None or total_episodes == 0:
        total_episodes = total_timesteps//(episode_len)
    iteration_returns = np.zeros(total_episodes//log_interval)
    critic_accuracy = np.zeros(total_episodes//log_interval)
    iteration_timesteps = (np.arange(total_episodes//log_interval)+1) * log_interval * (episode_len)

    repetitions = 3 * (repeat_mode in ['ds_adaptive', 'ds_fixed']) + 1
    num_deep_supervision_steps = np.zeros((total_episodes, episode_len))
    num_ponder_steps = np.full((total_episodes, episode_len), inner_loops)

    episode_count, step_count, last_step_count = 0, 0, 0
    episode_rewards = []
    start_time = time.time()
    last_log_time = start_time
    
    s, _ = env.reset(seed=seed)
    critical_step = 3

    while episode_count < total_episodes:
        initial_policy_hidden = None if policy_hidden is None else policy_hidden.clone()
        initial_value_hidden = None if value_hidden is None else value_hidden.clone()

        # sample action
        with torch.no_grad():
            for _ in range(repetitions+critical_step*(repeat_mode=='ds_hardcoded')):
                a, policy_hidden = agent.sample_action(s, policy_hidden, inner_loops=inner_loops+critical_step*(ponder_mode=='ponder_hardcoded'))
                if repeat_mode == 'ds_adaptive':
                    _, value_hidden, _ = agent.v(torch.tensor(np.array(s), dtype=torch.float), value_hidden)
                    q_halt = agent.value_net.compute_q(agent.value_net.get_penultimate_output())
                    if F.sigmoid(q_halt) >= 0.5:
                        break
        
        # step
        s_prime, r, terminated, truncated, info = env.step(a)

        # training
        policy_hidden = initial_policy_hidden
        value_hidden = initial_value_hidden
        for repeat in range(repetitions+critical_step*(repeat_mode=='ds_hardcoded')):
            next_policy_hidden, next_value_hidden, aux_data = agent.update_params(
                s, a, r, s_prime, terminated or truncated, policy_hidden, value_hidden, entropy_coeff, 
                inner_loops, critical_step*(ponder_mode=='ponder_hardcoded'), repeat_mode, repeat==repetitions-1, overshooting_info)
            policy_hidden = next_policy_hidden.detach() if next_policy_hidden is not None else None
            value_hidden = next_value_hidden.detach() if next_value_hidden is not None else None
            if repeat_mode == 'ds_adaptive':
                (aux_data, act_done) = aux_data
                if act_done:
                    break

        num_deep_supervision_steps[episode_count, step_count%(episode_len)] = repeat+1
        if aux_data is not None:
            ponder_steps = aux_data if isinstance(aux_data, int) else aux_data['summary'][0]
            num_ponder_steps[episode_count, step_count%(episode_len)] = ponder_steps
        
        s = s_prime
        critical_step = 0
        step_count += 1

        if terminated or truncated:
            total_return = info['episode']['r']
            episode_rewards.append(total_return)
            episode_count += 1
            
            if episode_count % log_interval == 0: # logging
                elapsed_time = time.time() - last_log_time
                steps_since_last_log = step_count - last_step_count
                
                sps = steps_since_last_log / elapsed_time if elapsed_time > 0 else 0
                avg_reward = sum(episode_rewards) / log_interval
                idx = episode_count//log_interval-1
                iteration_returns[idx] = avg_reward
                episode_rewards = []
                
                print(f"Avg {avg_reward:.2f}, SPS {int(sps)}, Episode {episode_count}/{total_episodes} (step {iteration_timesteps[idx]})")
                ponder_steps = np.mean(num_ponder_steps[(episode_count-log_interval):episode_count,], axis=0).tolist()
                repeat_steps = np.mean(num_deep_supervision_steps[(episode_count-log_interval):episode_count,], axis=0).tolist()
                print(f"    Ponder {ponder_steps if episode_len<10 else ponder_steps[:6] + ['...'] + ponder_steps[-5:]}")
                print(f"    Repeat {repeat_steps if episode_len<10 else repeat_steps[:6] + ['...'] + repeat_steps[-5:]}\n")
                last_log_time = time.time()
                last_step_count = step_count

                if debug:
                    env_stats = {'gamma': gamma, 'obs_stats': env.env.obs_stats.get_stats(), 'rew_stats': env.env.env.reward_stats.get_stats()} # merge
                    # avg_returns, critic_mse, ims = evaluate_critic(env_id, env_stats, seq_len, agent, ponder_mode, inner_loops, repeat_mode, fig, axs, ims, showfig=showfig)
                    # critic_accuracy[idx] = critic_mse
                
                if avg_reward >= 0.95: # early stopping based on success rate
                    success_counter += 1
                else:
                    success_counter = 0
                if success_counter >= 5:
                    iteration_returns[idx+1:] = iteration_returns[idx-4 : idx+1].mean()
                    critic_accuracy[idx+1:] = critic_accuracy[idx-4 : idx+1].mean()
                    print(f"\nEarly stopping at episode {episode_count} - Success rate >= 95% for 5 consecutive logging intervals")
                    break

            s, _ = env.reset()
            critical_step = 3
            terminated, truncated = False, False
            policy_hidden, value_hidden = None, None

    env.close()

    total_time = time.time() - start_time
    print()
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Total episodes: {episode_count}, Total steps: {step_count}")
    
    exp_dir = f"data_stream_ac__exp_id_{exp_id}"
    if not os.path.exists(exp_dir):
        try:
            os.makedirs(exp_dir)
        except FileExistsError:
            pass
    
    notes_path = os.path.join(exp_dir, f"notes_{exp_id}.txt")
    try:
        fd = os.open(notes_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            command = f"python utils/plot.py --dir {exp_dir} --int_space 0"
            if exp_name != "":
                command += f" --exp_name {exp_name}"
            f.write(f"{command}\n\n")
            for key, value in vars(args).items():
                f.write(f"{key}: {value}\n")
            f.write(f"\n{' '.join(sys.argv)}\n")
    except FileExistsError:
        pass

    run_dir = os.path.join(exp_dir, f"{env_id}_{seq_len}")
    if not os.path.exists(run_dir):
        try:
            os.makedirs(run_dir)
        except FileExistsError:
            pass

    run_dir = os.path.join(run_dir, f"ponder__{ponder_mode}")
    if not os.path.exists(run_dir):
        try:
            os.makedirs(run_dir)
        except FileExistsError:
            pass

    run_dir = os.path.join(run_dir, f"repeat__{repeat_mode}")
    if not os.path.exists(run_dir):
        try:
            os.makedirs(run_dir)
        except FileExistsError:
            pass

    with open(os.path.join(run_dir, f"seed_{seed}.pkl"), "wb") as f:
        pickle.dump((iteration_returns, iteration_timesteps, critic_accuracy), f)
    
    print(f"Data saved to {run_dir}, seed {seed}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream AC(λ)')

    # experiment and environment args
    parser.add_argument('--exp_id', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env_id', type=str, default='CopyTask')
    parser.add_argument('--seq_len', type=int, default=5, help='Length of seq_len parameter for environment')
    parser.add_argument('--total_timesteps', type=int, default=2000000, help='Total timesteps to run, only used if total episodes is None/0')
    parser.add_argument('--total_episodes', type=int, default=10000, help='Total number of episodes to run')
    parser.add_argument('--log_interval', type=int, default=100, help='Log performance every n episodes')

    # stream-x args
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--entropy_coeff', type=float, default=0.1)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--d_state', type=int, default=40)
    parser.add_argument('--kappa_policy', type=float, default=3.0)
    parser.add_argument('--kappa_value', type=float, default=2.0)

    # rnn args
    parser.add_argument('--rnn_type', type=str, default='glru')
    parser.add_argument('--ponder_mode', type=str, default='sequential', choices=["sequential", "convergence", "act", "ponder_hardcoded"])
    parser.add_argument('--ponder_eps', type=float, default=0.5) # threshold to halt pondering
    parser.add_argument('--ponder_n', type=int, default=4) # max ponder steps, 
    parser.add_argument('--inner_loops', type=int, default=1, help='Number of inner recurrent updates per step')
    parser.add_argument('--repeat_mode', type=str, default='none', choices=['none', 'ds_adaptive', 'ds_fixed', 'ds_hardcoded']) # ds for deep_supervision

    # operational args
    parser.add_argument('--overshooting_info', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    print(args, end='\n\n')

    main(args.exp_id, args.exp_name, args.seed, args.env_id, args.seq_len, args.total_timesteps, args.total_episodes, args.log_interval,
         args.lr, args.gamma, args.lamda, args.entropy_coeff, args.d_model, args.d_state, args.kappa_policy, args.kappa_value, 
         args.rnn_type, args.ponder_mode, args.ponder_eps, args.ponder_n, args.inner_loops, args.repeat_mode,
         args, args.overshooting_info, args.debug, args.render)
