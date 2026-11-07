"""
MASAC on 2× T4 — 10M params, 8× vectorized env, gradient all-reduce.
Batch 32768 fills VRAM (~12 GB per T4). LR 2e-3 (scaled with batch).
"""
import os, sys, time, atexit, socket, random, argparse, subprocess, webbrowser
from collections import deque
from datetime import datetime

_N_THREADS = 2
os.environ.setdefault("OMP_NUM_THREADS",       str(_N_THREADS))
os.environ.setdefault("MKL_NUM_THREADS",       str(_N_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS",  str(_N_THREADS))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

torch.set_num_threads(_N_THREADS)
torch.set_num_interop_threads(1)

from env import SheriffEnv
from vec_env import VectorizedSheriff


# ============================================================
# Config — 10M params per agent, batch 32768
# ============================================================
HIDDEN          = 1147           # 9,996,597 params per agent (~10M)
BATCH_SIZE      = 32768          # total; 16384 per rank → ~12 GB VRAM/T4
BUFFER_CAPACITY = 200_000        # per rank per agent (5 agents → 1M total)
UPDATE_EVERY    = 4              # env steps between gradient updates
WARMUP_STEPS    = 70             # real episode length (7 phases × 10 rounds)
N_VEC           = 8              # parallel envs per rank
USE_AMP         = True


def _find_free_port(start_port=6006, max_tries=20):
    for p in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("No free port")


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
        proc.terminate(); proc.wait(timeout=3)
    except Exception:
        try: proc.kill()
        except Exception: pass


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
            d = torch.distributions.Categorical(logits=logit)
            a = d.sample()
            actions.append(a)
            log_probs.append(d.log_prob(a))
        actions = torch.stack(actions, dim=-1)
        log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1, keepdim=True)
        return actions, log_prob

    def entropy(self, obs):
        logits = self.forward(obs)
        ent = 0.0
        for logit in logits:
            ent = ent + torch.distributions.Categorical(logits=logit).entropy()
        return ent.unsqueeze(-1)


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

        # nn.Parameter so it moves with .to(device)
        self.log_alpha = nn.Parameter(torch.tensor(np.log(alpha)))
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)
        self.alpha = self.log_alpha.exp().item()

        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def _allreduce_grads(self, module):
        """Average gradients across all ranks."""
        if not dist.is_initialized():
            return
        world = dist.get_world_size()
        for p in module.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= world

    def _allreduce_param_grad(self, param):
        if not dist.is_initialized() or param.grad is None:
            return
        world = dist.get_world_size()
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad /= world

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
        ac = torch.amp.autocast("cuda", enabled=self.use_amp)

        # ---- Critic ----
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
        self.scaler.unscale_(self.critic_opt)
        self._allreduce_grads(self.critic)
        self.scaler.step(self.critic_opt)
        self.scaler.update()

        # ---- Actor ----
        with ac:
            new_act, new_logp = self.actor.sample(obs)
            q1_new, q2_new = self.critic(obs, new_act)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (self.alpha * new_logp - q_new).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.actor_opt)
        self._allreduce_grads(self.actor)
        self.scaler.step(self.actor_opt)
        self.scaler.update()

        # ---- Alpha ----
        with torch.no_grad(), ac:
            entropy = self.actor.entropy(obs).mean()
        alpha_loss = -(self.log_alpha * (entropy - self.target_entropy)).mean()

        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self._allreduce_param_grad(self.log_alpha)
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()

        # ---- Soft target ----
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


def train(config_path="config.yaml", total_episodes=12000, warmup_episodes=50,
          lr=2e-3, gamma=0.99, tau=0.005, seed=42, log_dir="runs",
          log_every_updates=20, auto_tensorboard=True):

    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Run via torch.distributed.run --nproc_per_node=2")

    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    global_rank = dist.get_rank()
    world_size  = dist.get_world_size()
    is_main     = (global_rank == 0)

    # Same init for both ranks → identical starting weights
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    run_name = f"masac_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_path = os.path.join(log_dir, run_name)
    writer   = SummaryWriter(run_path) if is_main else None
    tb_proc  = start_tensorboard(log_dir) if (auto_tensorboard and is_main) else None

    if is_main:
        print(f"[Mode]    GRADIENT ALL-REDUCE + {N_VEC}× vectorized env")
        print(f"[Profile] hidden={HIDDEN}, batch={BATCH_SIZE} ({BATCH_SIZE//world_size}/GPU)")
        print(f"[Profile] buffer={BUFFER_CAPACITY}/rank, update_every={UPDATE_EVERY}, amp={USE_AMP}")
        print(f"[Device]  {device}, rank={global_rank}, world={world_size}")

    # Reference env (for action/obs spaces)
    env = SheriffEnv(config_path)
    env.reset(seed=seed)
    agents_ids = env.possible_agents
    obs_dim    = env.observation_spaces[agents_ids[0]].shape[0]
    act_dims   = [int(x) for x in env.action_spaces[agents_ids[0]].nvec]

    # Per-rank seed for env workers (different episodes per rank)
    np.random.seed(seed + local_rank)
    torch.manual_seed(seed + local_rank)

    buffers = {a: ReplayBuffer(BUFFER_CAPACITY, obs_dim, act_dims) for a in agents_ids}

    agents = {a: MASACAgent(obs_dim, act_dims, HIDDEN, lr, gamma, tau,
                            use_amp=USE_AMP).to(device)
              for a in agents_ids}

    # Broadcast rank-0 weights to all ranks (identical starting point)
    for a in agents_ids:
        for p in agents[a].parameters():
            dist.broadcast(p.data, src=0)

    if is_main:
        n_params = sum(p.numel() for p in agents[agents_ids[0]].parameters())
        print(f"[Model]   {n_params:,} params/agent × {len(agents_ids)} agents "
              f"= {n_params * len(agents_ids) / 1e6:.1f}M total")
        print(f"[Model]   all_reduce sync enabled (world={world_size})")

    if is_main:
        writer.add_hparams(
            {"hidden": HIDDEN, "batch_size": BATCH_SIZE, "buffer_cap": BUFFER_CAPACITY,
             "update_every": UPDATE_EVERY, "lr": lr, "world_size": world_size,
             "n_vec": N_VEC},
            {},
        )

    # Vectorized env
    vec = VectorizedSheriff(n_envs=N_VEC, config_path=config_path,
                            base_seed=seed + local_rank * 100_000)

    per_rank_batch = BATCH_SIZE // world_size

    episode_returns = {a: deque(maxlen=100) for a in agents_ids}
    episode_lengths = deque(maxlen=100)
    total_steps   = 0
    update_count  = 0
    episodes_done = 0
    train_start   = time.time()
    info = {"actor_loss": 0.0, "critic_loss": 0.0, "alpha": 0.0,
            "entropy": 0.0, "mean_q": 0.0}

    # Initial reset
    obs_list      = vec.reset()
    done_list     = [False] * N_VEC
    ep_rew_list   = [{a: 0.0 for a in agents_ids} for _ in range(N_VEC)]
    ep_steps_list = [0] * N_VEC

    # ============================================================
    # Main loop
    # ============================================================
    while episodes_done < total_episodes:
        # ---- 1. Build actions for all live envs ----
        live_idxs = [i for i in range(N_VEC) if not done_list[i]]
        live_actions = []
        for i in live_idxs:
            if total_steps < warmup_episodes * WARMUP_STEPS:
                acts = {a: env.action_spaces[a].sample() for a in agents_ids}
            else:
                acts = {a: agents[a].select_action(obs_list[i][a], device)
                        for a in agents_ids}
            live_actions.append(acts)

        # ---- 2. Step all live envs in parallel ----
        results = vec.step_all(live_actions)

        # ---- 3. Ingest results ----
        for j, i in enumerate(live_idxs):
            obs, rew, done = results[j]

            for a in agents_ids:
                if a in obs_list[i] and a in obs:
                    buffers[a].push(obs_list[i][a], live_actions[j][a],
                                    rew[a], obs[a], float(done))
                    ep_rew_list[i][a] += rew[a]

            obs_list[i]      = obs
            ep_steps_list[i] += 1
            total_steps      += 1

            if done:
                episodes_done += 1
                for a in agents_ids:
                    episode_returns[a].append(ep_rew_list[i][a])
                episode_lengths.append(ep_steps_list[i])

                # Reset only this env
                obs_list[i]      = vec.reset_single(i)
                done_list[i]     = False
                ep_rew_list[i]   = {a: 0.0 for a in agents_ids}
                ep_steps_list[i] = 0

                if is_main and episodes_done % 50 == 0:
                    avg = {a: np.mean(episode_returns[a]) for a in agents_ids}
                    print(f"Ep {episodes_done:5d} | " +
                          " | ".join([f"{a}: {avg[a]:7.2f}" for a in agents_ids]) +
                          f" | ep_len: {np.mean(episode_lengths):6.1f}")
                    print(f"        actor={info['actor_loss']:.3f} "
                          f"critic={info['critic_loss']:.3f} "
                          f"alpha={info['alpha']:.3f} "
                          f"entropy={info['entropy']:.1f}")

                if is_main:
                    for a in agents_ids:
                        writer.add_scalar(f"return/episode/{a}",    ep_rew_list[i][a], episodes_done)
                        writer.add_scalar(f"return/moving_avg/{a}", np.mean(episode_returns[a]), episodes_done)
                    writer.add_scalar("episode/length",      ep_steps_list[i],  episodes_done)
                    writer.add_scalar("episode/length_avg",  np.mean(episode_lengths), episodes_done)
                    writer.add_scalar("episode/total_steps", total_steps,        episodes_done)
                    writer.add_scalar("episode/wall_time_s", time.time() - train_start, episodes_done)

        # ---- 4. Gradient updates ----
        if (total_steps >= warmup_episodes * WARMUP_STEPS
            and total_steps % UPDATE_EVERY == 0):
            for a in agents_ids:
                if len(buffers[a]) >= per_rank_batch:
                    batch = buffers[a].sample(per_rank_batch, device)
                    info = agents[a].update(batch)

                    if is_main and update_count % log_every_updates == 0:
                        tag = f"agent/{a}"
                        for k in ["actor_loss", "critic_loss", "alpha", "entropy", "mean_q"]:
                            writer.add_scalar(f"{tag}/{k}", info[k], update_count)
                    update_count += 1

    # ---- Cleanup ----
    vec.close()

    if is_main:
        os.makedirs("masac_checkpoints", exist_ok=True)
        for a in agents_ids:
            torch.save({
                "actor":         agents[a].actor.state_dict(),
                "critic":        agents[a].critic.state_dict(),
                "critic_target": agents[a].critic_target.state_dict(),
            }, f"masac_checkpoints/{a}.pt")
        print("Saved → masac_checkpoints/")

    if dist.is_initialized():
        dist.barrier()

    if is_main:
        writer.flush()
        writer.close()
        _safe_kill(tb_proc)

    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int,   default=12000)
    p.add_argument("--warmup",   type=int,   default=50)
    p.add_argument("--lr",       type=float, default=2e-3)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--no-tb",    action="store_true")
    args = p.parse_args()
    train(config_path="config.yaml",
          total_episodes=args.episodes,
          warmup_episodes=args.warmup,
          lr=args.lr,
          seed=args.seed,
          auto_tensorboard=(not args.no_tb))


if __name__ == "__main__":
    main()
