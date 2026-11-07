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

# ---------------- Standard library imports ----------------
import os           # Path manipulation, env vars for DDP
import sys          # sys.executable — needed to launch TensorBoard subprocess
import time         # Wall-clock timing (sleep, elapsed)
import atexit       # Register cleanup on Python exit (kill TensorBoard)
import socket       # Find a free TCP port for TensorBoard
import random       # Python RNG seeding
import argparse     # CLI argument parsing
import subprocess   # Launch TensorBoard as a child process
import webbrowser   # Auto-open browser to TensorBoard
from collections import deque  # Rolling average for episode returns
from datetime import datetime  # Timestamp in run folder name

# ---------------- Third-party imports ----------------
import numpy as np                  # Array storage in replay buffer
import torch                        # Core PyTorch tensors + autograd
import torch.nn as nn               # Neural network modules
import torch.nn.functional as F     # Functional ops (one_hot, mse_loss)
import torch.optim as optim         # Adam optimizer
import torch.distributed as dist    # DDP: init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP  # Per-GPU model wrapper
from torch.utils.tensorboard import SummaryWriter  # Scalar + hparam logging

# ---------------- Local imports ----------------
from env import SheriffEnv          # Our PettingZoo environment


# ============================================================
# FIXED PROFILE — do not edit unless you know what you're doing
# ============================================================
HIDDEN          = 2048         # Width of every hidden layer in Actor + Critic
BATCH_SIZE      = 32768        # Total across DDP; splits to 16384 per GPU
BUFFER_CAPACITY = 2_000_000    # Replay buffer size, lives in RAM not VRAM
UPDATE_EVERY    = 4            # Env steps between gradient updates
USE_AMP         = True         # FP16 autocast + GradScaler (Tensor Cores)


# ============================================================
# TensorBoard auto-launch
# ============================================================
def _find_free_port(start_port=6006, max_tries=20):
    """Probe successive ports until we find one the OS will bind.

    TensorBoard defaults to 6006, but if another TensorBoard (or any
    process) is already using it, TB silently fails to start. This
    walks up to 6007, 6008, … up to start_port+20 so we always get a
    working URL printed to the console.
    """
    for p in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))       # Attempt to claim the port
                return p                        # Success — use it
            except OSError:
                continue                        # Taken — try the next port
    raise RuntimeError(f"No free port {start_port}..{start_port + max_tries}")


def start_tensorboard(log_dir, open_browser=True):
    """Spawn TensorBoard as a child process and open the browser to it.

    Returns the Popen handle so we can kill it at exit. Returns None if
    TensorBoard isn't installed (missing module).
    """
    port = _find_free_port()                    # Get a port TB can actually bind

    # Launch via `python -m tensorboard.main` so it uses the same interpreter
    cmd = [sys.executable, "-m", "tensorboard.main",
           "--logdir", log_dir,                 # Where the event files live
           "--port", str(port),                 # Free port from above
           "--reload_interval", "5"]            # Refresh event files every 5 s

    try:
        # Quiet child process: no stdout/stderr spam in our training log
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("[TensorBoard] pip install tensorboard")
        return None                             # Let training continue without TB

    time.sleep(2.0)                             # Give TB a moment to bind the port

    url = f"http://localhost:{port}"
    print(f"[TensorBoard] {url}")               # User-visible URL

    if open_browser:                            # If the platform supports it
        try:
            webbrowser.open(url)                # Auto-open default browser
        except Exception:
            pass                                # Headless env: user clicks URL manually

    atexit.register(lambda: _safe_kill(proc))   # Kill TB even if training crashes
    return proc


def _safe_kill(proc):
    """Terminate a subprocess gracefully, fall back to SIGKILL if it ignores us."""
    if proc is None or proc.poll() is not None:
        return                                  # Already dead — nothing to do
    try:
        proc.terminate()                        # SIGTERM — polite shutdown
        proc.wait(timeout=3)                    # Wait up to 3 s
    except Exception:
        try:
            proc.kill()                         # SIGKILL — force it
        except Exception:
            pass                                # Process gone; nothing else we can do


# ============================================================
# Replay Buffer (RAM, pinned)
# ============================================================
class ReplayBuffer:
    """Pre-allocated ring buffer of transitions.

    One buffer per agent. All arrays are pre-allocated at fixed size so
    `push()` is O(1) — no list growth, no GC pressure. Only `ptr` advances
    (mod capacity), which is the standard trick for unlimited history in
    constant memory.
    """

    def __init__(self, capacity, obs_dim, act_dims):
        self.capacity = capacity                                  # Max transitions
        # Preallocate every array once — never re-allocated after this
        self.obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act      = np.zeros((capacity, len(act_dims)), dtype=np.int64)
        self.rew      = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done     = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0                                              # Write cursor
        self.size = 0                                             # How full we are

    def push(self, obs, act, rew, next_obs, done):
        """Store one transition at ptr, then advance ptr with wraparound."""
        self.obs[self.ptr]      = obs                             # Copy observation
        self.act[self.ptr]      = act                             # Copy action vector
        self.rew[self.ptr]      = rew                             # Scalar reward
        self.next_obs[self.ptr] = next_obs                        # Next observation
        self.done[self.ptr]     = done                            # Episode-terminal flag
        self.ptr = (self.ptr + 1) % self.capacity                 # Wrap
        self.size = min(self.size + 1, self.capacity)             # Cap at capacity

    def sample(self, batch_size, device):
        """Return a random minibatch as GPU tensors.

        Uses `non_blocking=True` so the CPU→GPU copy is asynchronous.
        This only gives a speedup if the source numpy arrays are pinned
        in host memory — see how the buffer is built in `train()`.
        """
        idx = np.random.randint(0, self.size, size=batch_size)    # Uniform indices
        # Transfer each field as a GPU tensor (async when pinned)
        obs      = torch.as_tensor(self.obs[idx],      dtype=torch.float32).to(device, non_blocking=True)
        act      = torch.as_tensor(self.act[idx],      dtype=torch.int64  ).to(device, non_blocking=True)
        rew      = torch.as_tensor(self.rew[idx],      dtype=torch.float32).to(device, non_blocking=True).unsqueeze(-1)
        next_obs = torch.as_tensor(self.next_obs[idx], dtype=torch.float32).to(device, non_blocking=True)
        done     = torch.as_tensor(self.done[idx],     dtype=torch.float32).to(device, non_blocking=True).unsqueeze(-1)
        return obs, act, rew, next_obs, done                      # Tuple for unpacking

    def __len__(self):
        """Number of stored transitions (for warmup gating)."""
        return self.size


# ============================================================
# Actor — Multi-Discrete categorical
# ============================================================
class Actor(nn.Module):
    """Multi-discrete policy.

    Two shared hidden layers feed 12 independent linear "heads", one per
    dimension of the env's MultiDiscrete action space. Each head emits
    categorical logits, so we sample one action per dimension. The joint
    policy is factorized: π(a | s) = ∏_i π_i(a_i | s). This keeps output
    size linear in the action space instead of exponential.
    """

    def __init__(self, obs_dim, act_dims, hidden):
        super().__init__()
        self.act_dims = act_dims                                  # Per-dimension sizes
        self.net = nn.Sequential(                                 # Shared trunk
            nn.Linear(obs_dim, hidden), nn.ReLU(),                #   Input → hidden
            nn.Linear(hidden, hidden), nn.ReLU(),                 #   Hidden → hidden
        )
        # One linear head per action dimension
        self.heads = nn.ModuleList([nn.Linear(hidden, d) for d in act_dims])

    def forward(self, obs):
        """Return a list of logit tensors, one per action dimension."""
        h = self.net(obs)                                         # Shared features
        return [head(h) for head in self.heads]                   # One logits block per dim

    def sample(self, obs):
        """Sample an action vector and return (actions, summed log-prob).

        Log-probs sum because the joint distribution is the product of
        independent per-dimension categoricals.
        """
        logits = self.forward(obs)
        actions, log_probs = [], []
        for logit in logits:                                      # For each dimension
            dist = torch.distributions.Categorical(logits=logit)  # Categorical over actions
            a = dist.sample()                                     # Sample one action
            actions.append(a)                                     # Store action
            log_probs.append(dist.log_prob(a))                    # Store log π(a_i | s)
        actions = torch.stack(actions, dim=-1)                    # (batch, n_dims)
        log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1, keepdim=True)  # Sum over dims
        return actions, log_prob

    def entropy(self, obs):
        """Sum the entropy of each per-dimension categorical."""
        logits = self.forward(obs)
        ent = 0.0                                                 # Accumulator
        for logit in logits:
            ent = ent + torch.distributions.Categorical(logits=logit).entropy()
        return ent.unsqueeze(-1)                                  # Shape (batch, 1)


# ============================================================
# Twin Critic
# ============================================================
class TwinCritic(nn.Module):
    """Twin Q-networks for SAC's clipped-double-Q trick.

    Q(s, a) input is the observation concatenated with the one-hot-encoded
    action. Two independent Q-heads each predict a scalar. Taking the
    minimum of the two prevents the systematic overestimation bias of a
    single Q-network.
    """

    def __init__(self, obs_dim, act_dims, hidden):
        super().__init__()
        self.act_dims = act_dims                                  # For one-hot encoding
        in_dim = obs_dim + sum(act_dims)                          # obs ⊕ one-hot action
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),                 # Input → hidden
            nn.Linear(hidden, hidden), nn.ReLU(),                 # Hidden → hidden
            nn.Linear(hidden, 1),                                 # Hidden → scalar Q
        )
        # Q2 network — identical shape, independent init
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _one_hot(self, act):
        """Concatenate per-dimension one-hots into a single vector."""
        parts = [F.one_hot(act[:, i], num_classes=d).float()
                 for i, d in enumerate(self.act_dims)]            # One block per dim
        return torch.cat(parts, dim=-1)                           # (batch, sum_dims)

    def forward(self, obs, act):
        """Return (q1, q2) each with shape (batch, 1)."""
        x = torch.cat([obs, self._one_hot(act)], dim=-1)          # (batch, obs+onehot)
        return self.q1(x), self.q2(x)                             # Two independent Qs


# ============================================================
# MASAC Agent
# ============================================================
class MASACAgent(nn.Module):
    """One independent MASAC learner. Owns its own actor + critic.

    Every agent (player) in the env has one of these. There is no shared
    network, no centralized critic, no reward averaging. Each agent
    maximizes its own return independently — exactly what a competitive
    Free-For-All game needs.
    """

    def __init__(self, obs_dim, act_dims, hidden, lr, gamma, tau,
                 alpha=0.2, target_entropy=None, use_amp=True):
        super().__init__()
        self.gamma = gamma                                        # Discount factor
        self.tau = tau                                            # Polyak-averaging rate for target
        self.act_dims = act_dims                                  # For reference in update
        self.use_amp = use_amp                                    # Whether to use autocast

        # ----- Networks ----------------------------------------
        self.actor         = Actor(obs_dim, act_dims, hidden)     # Policy π
        self.critic        = TwinCritic(obs_dim, act_dims, hidden)  # Twin Q-networks
        self.critic_target = TwinCritic(obs_dim, act_dims, hidden)  # Target Q's (frozen-ish)
        self.critic_target.load_state_dict(self.critic.state_dict())  # Start identical

        # ----- Optimizers --------------------------------------
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr)   # Actor gradient
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)   # Critic gradient

        # ----- Entropy temperature (learned) -------------------
        if target_entropy is None:
            # Heuristic from the SAC paper: -dim(A) / 2
            target_entropy = -0.5 * sum(act_dims)
        self.target_entropy = target_entropy                      # Desired policy entropy
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True)  # log α, trainable
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)      # Optimizer for log α
        self.alpha = self.log_alpha.exp().item()                  # Cache current α value

        # ----- FP16 gradient scaling ---------------------------
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)  # Dynamic loss scale

    def select_action(self, obs, device, eval_mode=False):
        """Single-step action selection (no batch, no gradient).

        eval_mode=True → deterministic argmax (use at eval time).
        eval_mode=False → categorical sample (use at train time for exploration).
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # (1, obs_dim)
        with torch.no_grad():                                     # No grad — pure inference
            logits = self.actor(obs_t)                            # Forward pass
            out = []
            for logit in logits:                                  # Per action dimension
                if eval_mode:
                    out.append(logit.argmax(-1).item())           # Greedy
                else:
                    out.append(torch.distributions.Categorical(logits=logit).sample().item())  # Sample
        return np.array(out, dtype=np.int64)                      # (n_dims,) int array

    def update(self, batch):
        """One SAC gradient step: critic → actor → alpha → soft target.

        This is the heart of MASAC. Returns a dict of metrics for logging.
        """
        obs, act, rew, next_obs, done = batch                     # Unpack transition batch
        ac = torch.cuda.amp.autocast(enabled=self.use_amp)        # Context manager for FP16

        # ---------- CRITIC UPDATE ----------
        # Target Q uses the *target* critic and the *current* actor's action.
        with torch.no_grad(), ac:                                 # No gradient into target
            next_act, next_logp = self.actor.sample(next_obs)     # a' ~ π(·|s')
            q1_t, q2_t = self.critic_target(next_obs, next_act)   # Q'1, Q'2 for (s', a')
            q_t = torch.min(q1_t, q2_t) - self.alpha * next_logp  # SAC soft target
            target_q = rew + self.gamma * (1.0 - done) * q_t      # Bellman backup (mask terminal)

        with ac:                                                  # FP16 forward
            q1, q2 = self.critic(obs, act)                        # Q1, Q2 for (s, a)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)  # TD error

        self.critic_opt.zero_grad(set_to_none=True)               # Clear grads (cheap)
        self.scaler.scale(critic_loss).backward()                 # Scaled backward pass
        self.scaler.step(self.critic_opt)                         # Gradient step (unscales)
        self.scaler.update()                                      # Update loss scale

        # ---------- ACTOR UPDATE ----------
        # Loss = α * log π(a|s) - Q(s,a): minimize, i.e. maximize Q while
        # keeping policy stochastic (entropy term).
        with ac:                                                  # FP16 forward
            new_act, new_logp = self.actor.sample(obs)            # Sample fresh a ~ π(·|s)
            q1_new, q2_new = self.critic(obs, new_act)            # Evaluate Q(s, a_sample)
            q_new = torch.min(q1_new, q2_new)                     # Pessimistic Q
            actor_loss = (self.alpha * new_logp - q_new).mean()   # The SAC actor objective

        self.actor_opt.zero_grad(set_to_none=True)                # Clear grads
        self.scaler.scale(actor_loss).backward()                  # Scaled backward
        self.scaler.step(self.actor_opt)                          # Update actor
        self.scaler.update()                                      # Update scaler

        # ---------- ALPHA UPDATE ----------
        # Auto-tune α so entropy matches target_entropy.
        with torch.no_grad(), ac:                                 # No grad into policy here
            entropy = self.actor.entropy(obs).mean()              # Current policy entropy
        alpha_loss = -(self.log_alpha * (entropy - self.target_entropy)).mean()  # α loss

        self.alpha_opt.zero_grad(set_to_none=True)                # Clear grads
        alpha_loss.backward()                                     # Backward (no scaler needed — FP32)
        self.alpha_opt.step()                                     # Update log α
        self.alpha = self.log_alpha.exp().item()                  # Cache new α

        # ---------- SOFT TARGET UPDATE ----------
        # Polyak average: θ_target ← τ·θ + (1-τ)·θ_target
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)  # In-place: saves a tensor

        # Return metrics for TensorBoard logging
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

    # ---------- DDP initialization ----------
    # Refuse to run without torch.distributed launcher (we need LOCAL_RANK).
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "Run with: python -m torch.distributed.run --nproc_per_node=2 train.py"
        )
    local_rank = int(os.environ["LOCAL_RANK"])                    # 0 or 1 (GPU index)
    dist.init_process_group("nccl", init_method="env://")         # NCCL backend for GPUs
    torch.cuda.set_device(local_rank)                             # Bind to this GPU
    device = torch.device(f"cuda:{local_rank}")                   # Tensor target device
    is_main = (local_rank == 0)                                   # Only rank 0 logs/saves

    # ---------- Seeding ----------
    # Per-rank seeds so the two processes don't generate identical rollouts.
    random.seed(seed + local_rank)                                # Python RNG
    np.random.seed(seed + local_rank)                             # NumPy RNG
    torch.manual_seed(seed + local_rank)                          # Torch RNG
    torch.backends.cudnn.benchmark = True                         # Auto-tune cudnn kernels

    # ---------- TensorBoard ----------
    run_name = f"masac_t4x2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"  # Unique folder
    run_path = os.path.join(log_dir, run_name)                    # Full log path
    writer   = SummaryWriter(run_path) if is_main else None       # Only rank 0 writes
    tb_proc  = start_tensorboard(log_dir) if (auto_tensorboard and is_main) else None  # Auto-launch

    if is_main:                                                   # Only rank 0 prints config
        print(f"[Profile] hidden={HIDDEN}, batch={BATCH_SIZE} (DDP → {BATCH_SIZE // 2}/GPU)")
        print(f"[Profile] buffer={BUFFER_CAPACITY}, update_every={UPDATE_EVERY}, amp={USE_AMP}")
        print(f"[Device]  {device}, world_size={dist.get_world_size()}")
        print(f"[Logs]    {run_path}")

    # ---------- Environment ----------
    env = SheriffEnv(config_path)                                 # Build PettingZoo env
    env.reset(seed=seed)                                          # Prime it once
    agents_ids = env.possible_agents                              # List of player names
    obs_dim    = env.observation_spaces[agents_ids[0]].shape[0]   # Shared obs size
    act_dims   = [int(x) for x in env.action_spaces[agents_ids[0]].nvec]  # Per-dim action sizes

    # ---------- Per-agent replay buffers + learners ----------
    buffers = {a: ReplayBuffer(BUFFER_CAPACITY, obs_dim, act_dims) for a in agents_ids}  # One buffer each

    agents = {}
    for a in agents_ids:                                          # For every player
        agent = MASACAgent(obs_dim, act_dims, HIDDEN, lr, gamma, tau,  # Build learner
                           use_amp=USE_AMP).to(device)            # Move to GPU
        agents[a] = DDP(agent, device_ids=[local_rank], output_device=local_rank)  # Wrap for DDP

    if is_main:                                                   # Print param count
        n_params = sum(p.numel() for p in agents[agents_ids[0]].parameters())
        print(f"[Model]   {n_params:,} params per agent × {len(agents_ids)} agents")

    # ---------- Hparams ----------
    if is_main:                                                   # Log hyperparams once
        writer.add_hparams(
            {"hidden": HIDDEN, "batch_size": BATCH_SIZE, "buffer_cap": BUFFER_CAPACITY,
             "update_every": UPDATE_EVERY, "use_amp": USE_AMP,
             "lr": lr, "gamma": gamma, "tau": tau,
             "total_eps": total_episodes, "world_size": dist.get_world_size()},
            {},
        )

    # ---------- Training state ----------
    episode_returns = {a: deque(maxlen=100) for a in agents_ids}  # 100-ep rolling window
    episode_lengths = deque(maxlen=100)                           # 100-ep rolling window
    total_steps = 0                                               # Global env step counter
    update_count = 0                                              # Global gradient-step counter
    train_start = time.time()                                     # Wall clock start
    info = {"actor_loss": 0.0, "critic_loss": 0.0, "alpha": 0.0,
            "entropy": 0.0, "mean_q": 0.0}                        # Placeholder for logging

    # ---------- Main loop ----------
    for ep in range(total_episodes):
        obs_dict, _ = env.reset()                                 # Fresh episode
        done = False
        ep_rew = {a: 0.0 for a in agents_ids}                     # Per-agent episode return
        ep_steps = 0                                              # Steps in this episode

        while not done:
            # ---- Action selection ----
            actions = {}
            for a in env.agents:                                  # Only currently-active agents
                if total_steps < warmup_episodes * 300:           # Warmup phase
                    actions[a] = env.action_spaces[a].sample()    # Uniform random
                else:
                    # Use the underlying module (DDP wraps it) so select_action works
                    actions[a] = agents[a].module.select_action(obs_dict[a], device)

            # ---- Step env ----
            next_obs_dict, rew_dict, term_dict, trunc_dict, _ = env.step(actions)
            done = all(term_dict.values()) or all(trunc_dict.values())  # Episode over?

            # ---- Store transitions ----
            for a in env.agents:                                  # For each agent active this step
                if a in obs_dict and a in next_obs_dict:          # Must have transitioned
                    buffers[a].push(obs_dict[a], actions[a], rew_dict[a],
                                    next_obs_dict[a], float(done))  # Store
                    ep_rew[a] += rew_dict[a]                      # Accumulate return

            obs_dict = next_obs_dict                              # Advance observation
            ep_steps += 1                                         # Count this step
            total_steps += 1                                      # Count global step

            # ---- Gradient updates ----
            # Only every UPDATE_EVERY steps to keep GPU busy, not waiting on env.
            if (total_steps >= warmup_episodes * 300
                and total_steps % UPDATE_EVERY == 0):
                for a in agents_ids:                              # Update each agent
                    if len(buffers[a]) >= BATCH_SIZE:             # Need full batch
                        batch = buffers[a].sample(BATCH_SIZE, device)  # Sample
                        info = agents[a].update(batch)            # One SAC step

                        # ---- TensorBoard scalar logging ----
                        if is_main and update_count % log_every_updates == 0:
                            tag = f"agent/{a}"                    # Group by agent
                            writer.add_scalar(f"{tag}/actor_loss",  info["actor_loss"],  update_count)
                            writer.add_scalar(f"{tag}/critic_loss", info["critic_loss"], update_count)
                            writer.add_scalar(f"{tag}/alpha",       info["alpha"],       update_count)
                            writer.add_scalar(f"{tag}/entropy",     info["entropy"],     update_count)
                            writer.add_scalar(f"{tag}/mean_q",      info["mean_q"],      update_count)
                        update_count += 1                         # Advance counter

        # ---- Episode-level bookkeeping ----
        for a in agents_ids:
            episode_returns[a].append(ep_rew[a])                  # Append to rolling window
        episode_lengths.append(ep_steps)                          # Append length

        # ---- Episode-level logging ----
        if is_main:
            for a in agents_ids:
                writer.add_scalar(f"return/episode/{a}",    ep_rew[a], ep)                 # Raw return
                writer.add_scalar(f"return/moving_avg/{a}", np.mean(episode_returns[a]), ep)  # 100-ep avg
            writer.add_scalar("episode/length",      ep_steps,                  ep)        # Steps
            writer.add_scalar("episode/length_avg",  np.mean(episode_lengths),  ep)        # 100-ep avg
            writer.add_scalar("episode/total_steps", total_steps,               ep)        # Global progress
            writer.add_scalar("episode/wall_time_s", time.time() - train_start, ep)        # Wall clock

            # ---- Console print every 50 episodes ----
            if (ep + 1) % 50 == 0:
                avg = {a: np.mean(episode_returns[a]) for a in agents_ids}  # Short alias
                print(f"Ep {ep+1:5d} | "
                      + " | ".join([f"{a}: {avg[a]:7.2f}" for a in agents_ids])
                      + f" | ep_len: {np.mean(episode_lengths):6.1f}")
                print(f"        actor_loss={info['actor_loss']:.4f} "
                      f"critic_loss={info['critic_loss']:.4f} "
                      f"alpha={info['alpha']:.4f} entropy={info['entropy']:.3f}")

    # ---------- Save final checkpoints ----------
    if is_main:                                                   # Only rank 0 writes
        os.makedirs("masac_checkpoints", exist_ok=True)           # Ensure folder exists
        for a in agents_ids:
            inner = agents[a].module                              # Unwrap DDP → raw network
            torch.save({
                "actor":         inner.actor.state_dict(),        # Policy weights
                "critic":        inner.critic.state_dict(),       # Q-networks
                "critic_target": inner.critic_target.state_dict(),# Target Q-networks
            }, f"masac_checkpoints/{a}.pt")                       # One file per agent
        print("Saved → masac_checkpoints/")
        writer.flush()                                            # Flush pending scalars
        writer.close()                                            # Close TB writer
        _safe_kill(tb_proc)                                       # Kill TB child process

    dist.destroy_process_group()                                  # Clean DDP shutdown


# ============================================================
# Entry
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="MASAC training on SheriffEnv (2× T4)")
    parser.add_argument("--episodes", type=int,   default=5000)    # Total training episodes
    parser.add_argument("--warmup",   type=int,   default=50)      # Random-action warmup
    parser.add_argument("--lr",       type=float, default=1e-3)    # Learning rate
    parser.add_argument("--seed",     type=int,   default=42)      # Base random seed
    parser.add_argument("--no-tb",    action="store_true")         # Skip auto TensorBoard
    args = parser.parse_args()

    train(
        config_path="config.yaml",                                # Config file next to script
        total_episodes=args.episodes,                             # Pass CLI override
        warmup_episodes=args.warmup,                              # Pass CLI override
        lr=args.lr,                                               # Pass CLI override
        seed=args.seed,                                           # Pass CLI override
        auto_tensorboard=(not args.no_tb),                        # CLI flag inverted
    )


if __name__ == "__main__":
    main()                                                        # Standard script entry
