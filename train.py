"""
MASAC-Discrete training for SheriffEnv with auto-launched TensorBoard.

Run:
    python train_masac.py

TensorBoard starts automatically and the browser opens to the dashboard.
"""
import os
import sys
import time
import atexit
import socket
import random
import subprocess
import webbrowser
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from sheriff_env import SheriffEnv


# ============================================================
# 0. Auto-launch TensorBoard
# ============================================================
def _find_free_port(start_port=6006, max_tries=20):
    """Find the next free TCP port starting from start_port."""
    for p in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError(f"No free port found between {start_port} and {start_port + max_tries}")


def start_tensorboard(log_dir: str, open_browser: bool = True) -> subprocess.Popen:
    """
    Launch TensorBoard as a background subprocess and open the browser.
    Kills the subprocess automatically on Python exit.
    """
    port = _find_free_port(6006)
    cmd = [
        sys.executable, "-m", "tensorboard.main",
        "--logdir", log_dir,
        "--port", str(port),
        "--reload_interval", "5",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("[TensorBoard] Not installed. Run: pip install tensorboard")
        return None

    # Give it a moment to bind the port
    time.sleep(2.0)

    url = f"http://localhost:{port}"
    print(f"[TensorBoard] Running at {url}")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            print(f"[TensorBoard] Open manually: {url}")

    # Kill on Python exit
    atexit.register(lambda: _safe_kill(proc))
    return proc


def _safe_kill(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


# ============================================================
# 1. Replay Buffer
# ============================================================
class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dims):
        self.capacity = capacity
        self.obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act      = np.zeros((capacity, len(act_dims)), dtype=np.int64)
        self.rew      = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done     = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def push(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr]      = obs
        self.act[self.ptr]      = act
        self.rew[self.ptr]      = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr]     = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx]),
            torch.as_tensor(self.act[idx]),
            torch.as_tensor(self.rew[idx]).unsqueeze(-1),
            torch.as_tensor(self.next_obs[idx]),
            torch.as_tensor(self.done[idx]).unsqueeze(-1),
        )

    def __len__(self):
        return self.size


# ============================================================
# 2. Actor
# ============================================================
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dims, hidden=256):
        super().__init__()
        self.act_dims = act_dims
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, d) for d in act_dims])

    def forward(self, obs):
        h = self.net(obs)
        return [head(h) for head in self.heads]

    def sample(self, obs):
        logits = self.forward(obs)
        actions, log_probs = [], []
        for logit in logits:
            dist = torch.distributions.Categorical(logits=logit)
            a = dist.sample()
            actions.append(a)
            log_probs.append(dist.log_prob(a))
        actions = torch.stack(actions, dim=-1)
        log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1, keepdim=True)
        return actions, log_prob

    def entropy(self, obs):
        logits = self.forward(obs)
        ent = 0.0
        for logit in logits:
            ent = ent + torch.distributions.Categorical(logits=logit).entropy()
        return ent.unsqueeze(-1)


# ============================================================
# 3. Twin Critic
# ============================================================
class TwinCritic(nn.Module):
    def __init__(self, obs_dim, act_dims, hidden=256):
        super().__init__()
        self.act_dims = act_dims
        in_dim = obs_dim + sum(act_dims)
        self.q1 = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _one_hot(self, act):
        parts = [F.one_hot(act[:, i], num_classes=d).float()
                 for i, d in enumerate(self.act_dims)]
        return torch.cat(parts, dim=-1)

    def forward(self, obs, act):
        x = torch.cat([obs, self._one_hot(act)], dim=-1)
        return self.q1(x), self.q2(x)


# ============================================================
# 4. MASAC Agent
# ============================================================
class MASACAgent:
    def __init__(self, obs_dim, act_dims, lr=3e-4, gamma=0.99,
                 tau=0.005, alpha=0.2, target_entropy=None, device="cpu"):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.act_dims = act_dims

        self.actor         = Actor(obs_dim, act_dims).to(device)
        self.critic        = TwinCritic(obs_dim, act_dims).to(device)
        self.critic_target = TwinCritic(obs_dim, act_dims).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt  = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        if target_entropy is None:
            target_entropy = -0.5 * sum(act_dims)
        self.target_entropy = target_entropy
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)
        self.alpha = self.log_alpha.exp().item()

    def select_action(self, obs, eval_mode=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(obs_t)
            out = []
            for logit in logits:
                out.append((logit.argmax(-1) if eval_mode
                            else torch.distributions.Categorical(logits=logit).sample()).item())
        return np.array(out, dtype=np.int64)

    def update(self, batch):
        obs, act, rew, next_obs, done = batch

        with torch.no_grad():
            next_act, next_logp = self.actor.sample(next_obs)
            q1_t, q2_t = self.critic_target(next_obs, next_act)
            q_t = torch.min(q1_t, q2_t) - self.alpha * next_logp
            target_q = rew + self.gamma * (1.0 - done) * q_t

        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        new_act, new_logp = self.actor.sample(obs)
        q1_new, q2_new = self.critic(obs, new_act)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * new_logp - q_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        entropy = self.actor.entropy(obs).mean()
        alpha_loss = (self.log_alpha * (entropy - self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()

        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha":       self.alpha,
            "entropy":     entropy.item(),
            "mean_q":      q_new.mean().item(),
        }


# ============================================================
# 5. Training loop
# ============================================================
def train_masac(
    config_path="config.yaml",
    total_episodes=5000,
    warmup_episodes=50,
    batch_size=256,
    buffer_capacity=500_000,
    lr=3e-4,
    gamma=0.99,
    tau=0.005,
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=42,
    log_dir="runs",
    log_every_updates=10,
    auto_tensorboard=True,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    run_name = f"masac_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_path = os.path.join(log_dir, run_name)
    writer = SummaryWriter(run_path)
    print(f"[Logs] {run_path}")

    tb_proc = None
    if auto_tensorboard:
        tb_proc = start_tensorboard(log_dir, open_browser=True)

    env = SheriffEnv(config_path)
    env.reset(seed=seed)

    agents_ids = env.possible_agents
    obs_dim    = env.observation_spaces[agents_ids[0]].shape[0]
    act_dims   = [int(x) for x in env.action_spaces[agents_ids[0]].nvec]

    buffers = {a: ReplayBuffer(buffer_capacity, obs_dim, act_dims) for a in agents_ids}
    agents  = {
        a: MASACAgent(obs_dim, act_dims, lr=lr, gamma=gamma, tau=tau, device=device)
        for a in agents_ids
    }

    writer.add_hparams(
        {
            "lr": lr, "gamma": gamma, "tau": tau,
            "batch_size": batch_size, "buffer_cap": buffer_capacity,
            "total_eps": total_episodes, "warmup_eps": warmup_episodes,
            "target_entropy": -0.5 * sum(act_dims),
            "device": device, "seed": seed,
        },
        {},
    )

    print(f"Device: {device} | Agents: {agents_ids} | Obs: {obs_dim} | Act: {act_dims}")

    episode_returns = {a: deque(maxlen=100) for a in agents_ids}
    episode_lengths = deque(maxlen=100)
    total_steps = 0
    global_update_step = 0
    train_start = time.time()

    for ep in range(total_episodes):
        obs_dict, _ = env.reset()
        done = False
        ep_rew = {a: 0.0 for a in agents_ids}
        ep_steps = 0

        while not done:
            actions = {}
            for a in env.agents:
                if total_steps < warmup_episodes * 300:
                    actions[a] = env.action_spaces[a].sample()
                else:
                    actions[a] = agents[a].select_action(obs_dict[a])

            next_obs_dict, rew_dict, term_dict, trunc_dict, _ = env.step(actions)
            done = all(term_dict.values()) or all(trunc_dict.values())

            for a in env.agents:
                if a in obs_dict and a in next_obs_dict:
                    buffers[a].push(obs_dict[a], actions[a], rew_dict[a],
                                    next_obs_dict[a], float(done))
                    ep_rew[a] += rew_dict[a]

            obs_dict = next_obs_dict
            ep_steps += 1
            total_steps += 1

            if total_steps >= warmup_episodes * 300:
                for a in agents_ids:
                    if len(buffers[a]) >= batch_size:
                        batch = tuple(t.to(device) for t in buffers[a].sample(batch_size))
                        info = agents[a].update(batch)

                        if global_update_step % log_every_updates == 0:
                            tag = f"agent/{a}"
                            writer.add_scalar(f"{tag}/actor_loss",  info["actor_loss"],  global_update_step)
                            writer.add_scalar(f"{tag}/critic_loss", info["critic_loss"], global_update_step)
                            writer.add_scalar(f"{tag}/alpha",       info["alpha"],       global_update_step)
                            writer.add_scalar(f"{tag}/entropy",     info["entropy"],     global_update_step)
                            writer.add_scalar(f"{tag}/mean_q",      info["mean_q"],      global_update_step)
                            writer.add_scalar(f"{tag}/target_entropy",
                                              agents[a].target_entropy, global_update_step)

                        global_update_step += 1

        for a in agents_ids:
            episode_returns[a].append(ep_rew[a])
        episode_lengths.append(ep_steps)

        for a in agents_ids:
            writer.add_scalar(f"return/episode/{a}",    ep_rew[a], ep)
            writer.add_scalar(f"return/moving_avg/{a}", np.mean(episode_returns[a]), ep)
        writer.add_scalar("episode/length",      ep_steps,                  ep)
        writer.add_scalar("episode/length_avg",  np.mean(episode_lengths),  ep)
        writer.add_scalar("episode/total_steps", total_steps,               ep)
        writer.add_scalar("episode/wall_time_s", time.time() - train_start, ep)

        if (ep + 1) % 50 == 0:
            avg = {a: np.mean(episode_returns[a]) for a in agents_ids}
            print(f"Ep {ep+1:5d} | " +
                  " | ".join([f"{a}: {avg[a]:7.2f}" for a in agents_ids]) +
                  f" | ep_len: {np.mean(episode_lengths):6.1f}")
            if total_steps >= warmup_episodes * 300:
                print(f"        actor_loss={info['actor_loss']:.4f} "
                      f"critic_loss={info['critic_loss']:.4f} "
                      f"alpha={info['alpha']:.4f} "
                      f"entropy={info['entropy']:.3f} "
                      f"mean_q={info['mean_q']:.3f}")

    writer.add_hparams(
        {
            "lr": lr, "gamma": gamma, "tau": tau,
            "batch_size": batch_size, "buffer_cap": buffer_capacity,
            "total_eps": total_episodes, "warmup_eps": warmup_episodes,
            "target_entropy": -0.5 * sum(act_dims),
            "device": device, "seed": seed,
        },
        {f"final_return/{a}": float(np.mean(episode_returns[a])) for a in agents_ids},
    )

    os.makedirs("masac_checkpoints", exist_ok=True)
    for a in agents_ids:
        torch.save({
            "actor":         agents[a].actor.state_dict(),
            "critic":        agents[a].critic.state_dict(),
            "critic_target": agents[a].critic_target.state_dict(),
        }, f"masac_checkpoints/{a}.pt")
    print("Saved → masac_checkpoints/")

    writer.flush()
    writer.close()
    _safe_kill(tb_proc)
    return agents


# ============================================================
# 6. Entry point
# ============================================================
if __name__ == "__main__":
    train_masac(
        config_path="config.yaml",
        total_episodes=5000,
        warmup_episodes=50,
        batch_size=256,
        device="cuda" if torch.cuda.is_available() else "cpu",
        log_dir="runs",
        auto_tensorboard=True,
    )
