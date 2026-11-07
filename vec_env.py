"""N parallel SheriffEnv instances running in subprocesses.

Each worker owns its own SheriffEnv. step_all() sends N actions at once
and receives N results — the N env.step() calls run in parallel.
"""
from multiprocessing import Process, Pipe
from env import SheriffEnv


def _worker(conn, config_path, seed):
    """Subprocess main loop. Receives commands, replies with env output."""
    env = SheriffEnv(config_path)
    env.reset(seed=seed)
    while True:
        cmd, data = conn.recv()
        if cmd == "reset":
            obs, _ = env.reset(seed=data)
            conn.send(obs)
        elif cmd == "step":
            obs, rew, term, trunc, _ = env.step(data)
            done = all(term.values()) or all(trunc.values())
            conn.send((obs, rew, done))
        elif cmd == "stop":
            conn.send(None)
            break


class VectorizedSheriff:
    """N envs in N subprocesses. step_all() runs them in parallel."""

    def __init__(self, n_envs=8, config_path="config.yaml", base_seed=0):
        self.n_envs = n_envs
        self.conns, self.procs = [], []

        # Probe env just for space metadata (deleted after)
        probe = SheriffEnv(config_path)
        self.action_spaces      = probe.action_spaces
        self.observation_spaces = probe.observation_spaces
        self.possible_agents    = probe.possible_agents
        del probe

        for i in range(n_envs):
            parent, child = Pipe()
            p = Process(
                target=_worker,
                args=(child, config_path, base_seed + i * 7919),
                daemon=True,
            )
            p.start()
            self.conns.append(parent)
            self.procs.append(p)

    def reset(self):
        """Reset all N envs. Returns list[N] of obs_dict."""
        for c in self.conns:
            c.send(("reset", None))
        return [c.recv() for c in self.conns]

    def reset_single(self, idx):
        """Reset one env mid-loop when its episode ends."""
        self.conns[idx].send(("reset", None))
        return self.conns[idx].recv()

    def step_all(self, actions_list):
        """Send N actions, receive N results. Parallel.
        Returns list[N] of (obs_dict, rew_dict, done_bool).
        """
        for c, a in zip(self.conns, actions_list):
            c.send(("step", a))
        return [c.recv() for c in self.conns]

    def close(self):
        for c in self.conns:
            try:
                c.send(("stop", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=1)
