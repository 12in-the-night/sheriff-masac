"""
MASAC-Discrete for SheriffEnv — 2× T4 (32 GB), maxed out.
============================================================

Profile:
  - 2× NVIDIA T4 via DDP (DistributedDataParallel)
  - hidden = 2048
  - batch_size = 32768 (total, DDP splits to 16384 per GPU)
  - replay buffer = 2,000,000 transitions (in RAM)
  - update_every = 4 env steps
  - FP16 autocast + GradScaler (Tensor Cores)
  - pinned memory + non_blocking transfers
  - auto-launched TensorBoard

VRAM: ~11–13 GB per T4, ~22–26 GB total of 32 GB.

Run:
  # Linux / Colab / cloud
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \\
    python -m torch.distributed.run --nproc_per_node=2 \\
    train_masac_t4x2.py --lr 1e-3

  # Windows
  set MASTER_ADDR=127.0.0.1
  set MASTER_PORT=29500
  python -m torch.distributed.run --nproc_per_node=2 train_masac_t4x2.py --lr 1e-3
"""
import os
import sys
import time
import atexit
import socket
import random
import argparse
import subprocess
import webbrowser
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from sheriff_env import SheriffEnv


# ============================================================
# FIXED PROFILE — do not edit unless you know what you're doing
# ============================================================
HIDDEN          = 2048
BATCH_SIZE      = 32768        # total; DDP splits to 16384 per GPU
BUFFER_CAPACITY = 2_000_000    # in RAM, not VRAM
UPDATE_EVERY    = 4            # env steps between updates
USE_AMP         = True


# ============================================================
# TensorBoard auto-launch
# ============================================================
def _find_free_port(start_port=6006, max_tries=20):
    for p in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError(f"No free port {start_port}..{start_port + max_tries}")


def start_tensorboard(log_dir, open_browser=True):
    port = _find_free_port()
    cmd = [sys.executable, "-m", "tensorboard.main",
           "--logdir", log_dir, "--port", str(port), "--reload_interval", "5"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("[TensorBoard] pip install tensorboard")
        return None
    time.sleep(2.0)
    url = f"http://localhost:{port}"
    print(f"[TensorBoard] {url}")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
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
# Replay Buffer (RAM, pinned)
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

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        obs      = torch.as_tensor(self.obs[idx],      dtype=torch.float32).to(device, non_blocking=True)
        act      = torch.as_tensor(self.act[idx],      dtype=torch.int64  ).to(device, non_blocking=True)
        rew      = torch.as_tensor(self.rew[idx],      dtype=torch.float32).to(device, non_blocking=True).unsqueeze(-1)
        next_obs = torch.as_tensor(self.next_obs[idx], dtype=torch.float32).to(device, non_blocking=True)
        done     = torch.as_tensor(self.done[idx],     dtype=torch.float32).to(device, non_blocking=True).unsqueeze(-1)
        return obs, act, rew, next_obs, done

    def __len__(self):
        return self.size


# ============================================================
# Actor — Multi-Discrete categorical
# ============================================================
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dims, hidden):
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
# Twin Critic
# ============================================================
class TwinCritic(nn.Module):
    def __init__(self, obs_dim, act_dims, hidden):
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
# MASAC Agent
# ============================================================
class MASACAgent(nn.Module):
    def __init__(self, obs_dim, act_dims, hidden, lr, gamma, tau,
                 alpha=0.2, target_entropy=None, use_amp=True):
        super().__init__()
        self.gamma = gamma
        self.tau = tau
        self.act_dims = act_dims
        self.use_amp = use_amp

        self.actor         = Actor(obs_dim, act_dims, hidden)
        self.critic        = TwinCritic(obs_dim, act_dims, hidden)
        self.critic_target = TwinCritic(obs_dim, act_dims, hidden)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        if target_entropy is None:
            target_entropy = -0.5 * sum(act_dims)
        self.target_entropy = target_entropy
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)
        self.alpha = self.log_alpha.exp().item()

        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def select_action(self, obs, device, eval_mode=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(obs_t)
            out = []
            for logit in logits:
                out.append((logit.argmax(-1) if eval_mode
                            else torch.distributions.Categorical(logits=logit).sample()).item())
        return np.array(out, dtype=np.int64)

    def update(self, batch):
        obs, act, rew, next_obs, done = batch
        ac = torch.cuda.amp.autocast(enabled=self.use_amp)

        # ----- Critic -----
        with torch.no_grad(), ac:
            next_act, next_logp = self.actor.sample(next_obs)
            q1_t, q2_t = self.critic_target(next_obs, next_act)
            q_t = torch.min(q1_t, q2_t) - self.alpha * next_logp
            target_q = rew + self.gamma * (1.0 - done) * q_t

        with ac:
            q1, q2 = self.critic(obs, act)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad(set_to_none=True)
        self.scaler.scale(critic_loss).backward()
        self.scaler.step(self.critic_opt)
        self.scaler.update()

        # ----- Actor -----
        with ac:
            new_act, new_logp = self.actor.sample(obs)
            q1_new, q2_new = self.critic(obs, new_act)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (self.alpha * new_logp - q_new).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        self.scaler.scale(actor_loss).backward()
        self.scaler.step(self.actor_opt)
        self.scaler.update()

        # ----- Alpha -----
        with torch.no_grad(), ac:
            entropy = self.actor.entropy(obs).mean()
        alpha_loss = -(self.log_alpha * (entropy - self.target_entropy)).mean()

        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()

        # ----- Soft target -----
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha":       self.alpha,
            "entropy":     entropy.item(),
            "mean_q":      q_new.mean().item(),
        }


# ============================================================
# Training loop — DDP, 2× T4
# ============================================================
def train(config_path="config.yaml",
          total_episodes=5000,
          warmup_episodes=50,
          lr=1e-3,
          gamma=0.99,
          tau=0.005,
          seed=42,
          log_dir="runs",
          log_every_updates=20,
          auto_tensorboard=True):

    # ----- DDP init -----
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "Run with: python -m torch.distributed.run --nproc_per_node=2 train_masac_t4x2.py"
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main = (local_rank == 0)

    # ----- Seeding -----
    random.seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    torch.manual_seed(seed + local_rank)
    torch.backends.cudnn.benchmark = True

    # ----- TensorBoard -----
    run_name = f"masac_t4x2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_path = os.path.join(log_dir, run_name)
    writer   = SummaryWriter(run_path) if is_main else None
    tb_proc  = start_tensorboard(log_dir) if (auto_tensorboard and is_main) else None

    if is_main:
        print(f"[Profile] hidden={HIDDEN}, batch={BATCH_SIZE} (DDP → {BATCH_SIZE // 2}/GPU)")
        print(f"[Profile] buffer={BUFFER_CAPACITY}, update_every={UPDATE_EVERY}, amp={USE_AMP}")
        print(f"[Device]  {device}, world_size={dist.get_world_size()}")
        print(f"[Logs]    {run_path}")

    # ----- Env -----
    env = SheriffEnv(config_path)
    env.reset(seed=seed)
    agents_ids = env.possible_agents
    obs_dim    = env.observation_spaces[agents_ids[0]].shape[0]
    act_dims   = [int(x) for x in env.action_spaces[agents_ids[0]].nvec]

    # ----- Buffers & agents -----
    buffers = {a: ReplayBuffer(BUFFER_CAPACITY, obs_dim, act_dims) for a in agents_ids}

    agents = {}
    for a in agents_ids:
        agent = MASACAgent(obs_dim, act_dims, HIDDEN, lr, gamma, tau,
                           use_amp=USE_AMP).to(device)
        agents[a] = DDP(agent, device_ids=[local_rank], output_device=local_rank)

    if is_main:
        n_params = sum(p.numel() for p in agents[agents_ids[0]].parameters())
        print(f"[Model]   {n_params:,} params per agent × {len(agents_ids)} agents")

    # ----- Hparams -----
    if is_main:
        writer.add_hparams(
            {"hidden": HIDDEN, "batch_size": BATCH_SIZE, "buffer_cap": BUFFER_CAPACITY,
             "update_every": UPDATE_EVERY, "use_amp": USE_AMP,
             "lr": lr, "gamma": gamma, "tau": tau,
             "total_eps": total_episodes, "world_size": dist.get_world_size()},
            {},
        )

    # ----- State -----
    episode_returns = {a: deque(maxlen=100) for a in agents_ids}
    episode_lengths = deque(maxlen=100)
    total_steps = 0
    update_count = 0
    train_start = time.time()
    info = {"actor_loss": 0.0, "critic_loss": 0.0, "alpha": 0.0,
            "entropy": 0.0, "mean_q": 0.0}

    # ----- Loop -----
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
                    actions[a] = agents[a].module.select_action(obs_dict[a], device)

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

            if (total_steps >= warmup_episodes * 300
                and total_steps % UPDATE_EVERY == 0):
                for a in agents_ids:
                    if len(buffers[a]) >= BATCH_SIZE:
                        batch = buffers[a].sample(BATCH_SIZE, device)
                        info = agents[a].update(batch)

                        if is_main and update_count % log_every_updates == 0:
                            tag = f"agent/{a}"
                            writer.add_scalar(f"{tag}/actor_loss",  info["actor_loss"],  update_count)
                            writer.add_scalar(f"{tag}/critic_loss", info["critic_loss"], update_count)
                            writer.add_scalar(f"{tag}/alpha",       info["alpha"],       update_count)
                            writer.add_scalar(f"{tag}/entropy",     info["entropy"],     update_count)
                            writer.add_scalar(f"{tag}/mean_q",      info["mean_q"],      update_count)
                        update_count += 1

        for a in agents_ids:
            episode_returns[a].append(ep_rew[a])
        episode_lengths.append(ep_steps)

        if is_main:
            for a in agents_ids:
                writer.add_scalar(f"return/episode/{a}",    ep_rew[a], ep)
                writer.add_scalar(f"return/moving_avg/{a}", np.mean(episode_returns[a]), ep)
            writer.add_scalar("episode/length",      ep_steps,                  ep)
            writer.add_scalar("episode/length_avg",  np.mean(episode_lengths),  ep)
            writer.add_scalar("episode/total_steps", total_steps,               ep)
            writer.add_scalar("episode/wall_time_s", time.time() - train_start, ep)

            if (ep + 1) % 50 == 0:
                avg = {a: np.mean(episode_returns[a]) for a in agents_ids}
                print(f"Ep {ep+1:5d} | "
                      + " | ".join([f"{a}: {avg[a]:7.2f}" for a in agents_ids])
                      + f" | ep_len: {np.mean(episode_lengths):6.1f}")
                print(f"        actor_loss={info['actor_loss']:.4f} "
                      f"critic_loss={info['critic_loss']:.4f} "
                      f"alpha={info['alpha']:.4f} entropy={info['entropy']:.3f}")

    # ----- Save -----
    if is_main:
        os.makedirs("masac_checkpoints", exist_ok=True)
        for a in agents_ids:
            inner = agents[a].module
            torch.save({
                "actor":         inner.actor.state_dict(),
                "critic":        inner.critic.state_dict(),
                "critic_target": inner.critic_target.state_dict(),
            }, f"masac_checkpoints/{a}.pt")
        print("Saved → masac_checkpoints/")
        writer.flush()
        writer.close()
        _safe_kill(tb_proc)

    dist.destroy_process_group()


# ============================================================
# Entry
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int,   default=5000)
    parser.add_argument("--warmup",   type=int,   default=50)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--no-tb",    action="store_true")
    args = parser.parse_args()

    train(
        config_path="config.yaml",
        total_episodes=args.episodes,
        warmup_episodes=args.warmup,
        lr=args.lr,
        seed=args.seed,
        auto_tensorboard=(not args.no_tb),
    )


if __name__ == "__main__":
    main()
