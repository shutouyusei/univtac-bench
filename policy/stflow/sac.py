"""Soft actor-critic on the fast side of a slow-fast SFP checkpoint (``tactile_adapter`` + ``tactile_prediction``).

The action of a control step is the adapters' residual on the slow velocity: the slow side (no tactile) gives
``v_slow``, the full expert (adapters reading the block's fingertips) ``v_full``, and the actor's mean is
``mu = v_full - v_slow``. Exploration adds ``sigma * eps`` (one learnt, state-independent std per action dim), and
the executed velocity is ``v_slow + a``; without exploration this is exactly the SFP controller. Only the adapters
and ``log_std`` are trained; the slow side, the tokenizers and the tactile read-out stay frozen.

Reward (no task term): at every block start inside a chunk, ``-RMS(p - A)`` where ``A`` is the fingertips'
pooled, layer-normed tokens and ``p`` the slow side's read-out for that query, computed on a second integration
``x_slow`` that follows the slow velocity alone. The forecast therefore depends on the chunk start's observation
and the slow plan only, so the fast side cannot move its own target. The reward belongs to the step before the
reading. Every episode end is treated as a truncation (no terminal state): with negative rewards a terminal end
would pay the policy for ending early, e.g. by dropping the peg.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

LOG_2PI = math.log(2 * math.pi)


@dataclass
class SACConfig:
    """``scale``: per action dim size of the IL residual (normalised velocity); the exploration std is
    ``sigma_frac * scale`` at the start and the entropy target is that of a Gaussian with this std."""

    scale: list[float] = field(default_factory=list)
    sigma_frac: float = 0.2
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    utd: float = 1.0
    warmup_episodes: int = 10
    actor_lr: float = 1e-5
    critic_lr: float = 3e-4
    log_std_lr: float = 1e-3
    alpha_lr: float = 1e-3
    init_alpha: float = 0.01
    hidden: int = 512
    grad_clip: float = 1.0


def pooled_reading(policy, tokens: Tensor) -> Tensor:
    """``(B, n_tips, d)``: tactile tokens mean-pooled per fingertip and layer-normed (the read-out's space)."""
    b, d = tokens.shape[0], tokens.shape[-1]
    return F.layer_norm(tokens.view(b, len(policy.tactile.names), -1, d).mean(2).float(), (d,))


def gaussian_log_prob(a: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
    z = (a - mean) / log_std.exp()
    return (-0.5 * z.pow(2) - log_std - 0.5 * LOG_2PI).sum(-1)


def query_mask(n_obs: int, horizon: int, device) -> Tensor:
    from stflow.methods.sfp import independent_query_mask

    return independent_query_mask(n_obs, horizon, device=device)


def _np(t: Tensor, dtype=np.float32) -> np.ndarray:
    return t.detach().float().cpu().numpy().astype(dtype)


class SACController:
    """The SFP controller (open loop over ``n_action_steps``, fingertips re-read every ``tactile_refresh`` steps)
    with the fast residual as a stochastic action. ``act`` returns the joint target and a record for the replay
    buffer (numpy): ``q`` (query = step in the chunk), ``x`` (normalised action before the step), ``v_slow``,
    ``mean``, ``action`` (the executed residual), ``hidden`` (slow-side token of the query, critic input),
    ``reading`` / ``tactile_tokens`` at a block start, ``obs_tokens`` at a chunk start, and ``reward`` (for the
    previous step) at a block start inside a chunk."""

    def __init__(self, policy, learner: "SAC", n_action_steps: int, tactile_refresh: int, explore: bool = True,
                 seed: int = 0):
        self.policy = policy
        self.learner = learner
        self.n_action_steps = n_action_steps
        self.tactile_refresh = tactile_refresh
        self.explore = explore
        self.generator = torch.Generator().manual_seed(seed)
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.obs_tokens = None
        self.tactile_tokens = None
        self.reading = None
        self.x = None
        self.x_slow = None

    def act(self, observation: dict) -> tuple[Tensor, dict]:
        policy = self.policy
        horizon = policy.cfg.chunk_size
        new_chunk = self.obs_tokens is None or self.step == self.n_action_steps
        if new_chunk:
            self.obs_tokens = policy.encode(observation)
            self.x = policy.normalizer.action(observation["state"][:, : policy.cfg.action_dim])
            self.x_slow = self.x.clone()
            self.step = 0
        new_block = new_chunk or self.step % self.tactile_refresh == 0
        if new_block:
            self.tactile_tokens = policy.tactile_tokens(observation["tactile"])
            self.reading = pooled_reading(policy, self.tactile_tokens)
        obs, q = self.obs_tokens, self.step
        mask = query_mask(obs.shape[1], horizon, obs.device)
        t = torch.full((1, horizon), q / horizon, device=obs.device, dtype=obs.dtype)
        v_plan, h_plan = policy.expert(obs, self.x_slow[:, None].expand(-1, horizon, -1).to(obs.dtype), t,
                                       mask=mask, return_hidden=True)
        reward = None
        if new_block and not new_chunk:
            names, d = policy.tactile.names, h_plan.shape[-1]
            forecast = policy.tactile_predictor(h_plan[:, q]).float().view(1, len(names), d)
            reward = -float((forecast - self.reading).pow(2).mean().sqrt())
        xx = self.x[:, None].expand(-1, horizon, -1).to(obs.dtype)
        v_s, hidden = policy.expert(obs, xx, t, mask=mask, return_hidden=True)
        v_f = policy.expert(obs, xx, t, mask=mask, tactile=self.tactile_tokens, tactile_scale=1.0)
        v_slow, mean = v_s[:, q], v_f[:, q] - v_s[:, q]
        action = mean
        if self.explore:
            eps = torch.randn(mean.shape, generator=self.generator).to(mean)
            action = mean + self.learner.log_std.detach().exp().to(mean) * eps
        record = {
            "q": q, "new_chunk": new_chunk, "new_block": new_block, "reward": reward,
            "x": _np(self.x[0]), "v_slow": _np(v_slow[0]), "mean": _np(mean[0]), "action": _np(action[0]),
            "hidden": _np(hidden[0, q], np.float16),
            "reading": _np(self.reading[0], np.float16) if new_block else None,
            "tactile_tokens": _np(self.tactile_tokens[0], np.float16) if new_block else None,
            "obs_tokens": _np(obs[0], np.float16) if new_chunk else None,
        }
        self.x = self.x + (v_slow + action).to(self.x.dtype) / horizon
        self.x_slow = self.x_slow + v_plan[:, q].to(self.x_slow.dtype) / horizon
        self.step += 1
        return policy.normalizer.unaction(self.x.float())[0].cpu(), record


STEP_KEYS = ("chunk", "block", "q", "x", "v_slow", "hidden", "action", "r", "has_next")
TABLE_KEYS = ("obs", "tac", "reading")


class ReplayBuffer:
    """Episodes of records. Per episode: tables ``obs`` (chunks), ``tac`` and ``reading`` (blocks), and per step
    ``chunk`` / ``block`` (table rows), ``q``, ``x``, ``v_slow``, ``hidden``, ``action``, ``r`` and ``has_next``
    (False on the last step: every end is a truncation, so the last step has no next state and is never
    sampled)."""

    def __init__(self):
        self.episodes: list[dict] = []
        self._open: dict | None = None
        self._valid: list[np.ndarray] = []

    def start_episode(self) -> None:
        self._open = {k: [] for k in STEP_KEYS + TABLE_KEYS if k != "has_next"}

    def add(self, record: dict) -> None:
        ep = self._open
        if record["new_chunk"]:
            ep["obs"].append(record["obs_tokens"])
        if record["new_block"]:
            ep["tac"].append(record["tactile_tokens"])
            ep["reading"].append(record["reading"])
        if record["reward"] is not None and ep["r"]:
            ep["r"][-1] += record["reward"]
        ep["chunk"].append(len(ep["obs"]) - 1)
        ep["block"].append(len(ep["tac"]) - 1)
        for k in ("q", "x", "v_slow", "hidden", "action"):
            ep[k].append(record[k])
        ep["r"].append(0.0)

    def end_episode(self) -> dict | None:
        ep, self._open = self._open, None
        if not ep or not ep["q"]:
            return None
        n = len(ep["q"])
        out = {k: np.stack(ep[k]) for k in TABLE_KEYS}
        out.update({k: np.asarray(ep[k], dtype=np.int32) for k in ("chunk", "block", "q")})
        out.update({k: np.stack(ep[k]) for k in ("x", "v_slow", "hidden", "action")})
        out["r"] = np.asarray(ep["r"], dtype=np.float32)
        out["has_next"] = np.arange(n) < n - 1
        self._append(out)
        return out

    def _append(self, ep: dict) -> None:
        self.episodes.append(ep)
        self._valid.append(np.flatnonzero(ep["has_next"]))

    def __len__(self) -> int:
        return int(sum(len(v) for v in self._valid))

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        sizes = np.array([len(v) for v in self._valid], dtype=float)
        eps = rng.choice(len(self._valid), size=n, p=sizes / sizes.sum())
        steps = np.array([self._valid[e][rng.integers(len(self._valid[e]))] for e in eps])
        return eps, steps

    def gather(self, eps: np.ndarray, steps: np.ndarray, device, offset: int = 0) -> dict:
        """Tensors of the steps ``steps + offset`` (offset 1: the next states)."""
        rows = [(self.episodes[e], k + offset) for e, k in zip(eps, steps)]

        def stack(fn, dtype=torch.float32):
            return torch.from_numpy(np.stack([fn(ep, k) for ep, k in rows])).to(device=device, dtype=dtype)

        return {
            "obs": stack(lambda ep, k: ep["obs"][ep["chunk"][k]]),
            "tac": stack(lambda ep, k: ep["tac"][ep["block"][k]]),
            "reading": stack(lambda ep, k: ep["reading"][ep["block"][k]]),
            "q": stack(lambda ep, k: ep["q"][k], torch.long),
            "x": stack(lambda ep, k: ep["x"][k]),
            "v_slow": stack(lambda ep, k: ep["v_slow"][k]),
            "hidden": stack(lambda ep, k: ep["hidden"][k]),
            "action": stack(lambda ep, k: ep["action"][k]),
            "r": stack(lambda ep, k: ep["r"][k]),
        }

    def save_episode(self, index: int, path: Path) -> None:
        np.savez(path, **self.episodes[index])

    def load_dir(self, directory: Path) -> None:
        for path in sorted(Path(directory).glob("*.npz")):
            with np.load(path) as f:
                self._append({k: f[k] for k in f.files})


class TwinQ(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int):
        super().__init__()

        def mlp():
            return nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Linear(hidden, 1),
            )

        self.q1, self.q2 = mlp(), mlp()

    def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)


class SAC:
    """Learner: the adapters of ``policy.expert`` and ``log_std`` (actor), a twin critic on
    ``[slow hidden, pooled reading, x, q / H]`` with the residual divided by ``scale``, and the temperature."""

    def __init__(self, policy, cfg: SACConfig):
        self.policy, self.cfg = policy, cfg
        self.device = next(policy.parameters()).device
        for name, p in policy.named_parameters():
            p.requires_grad_(name.startswith("expert.adapters."))
        action_dim, d = policy.cfg.action_dim, policy.cfg.d_model
        if len(cfg.scale) != action_dim:
            raise ValueError(f"SACConfig.scale needs {action_dim} values, got {len(cfg.scale)}")
        self.scale = torch.tensor(cfg.scale, dtype=torch.float32, device=self.device)
        sigma = cfg.sigma_frac * self.scale
        self.log_std = nn.Parameter(sigma.log())
        self.target_entropy = float((sigma.log() + 0.5 * (LOG_2PI + 1)).sum())
        state_dim = d + len(policy.tactile.names) * d + action_dim + 1
        self.critic = TwinQ(state_dim, action_dim, cfg.hidden).to(self.device)
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_alpha = torch.tensor(math.log(cfg.init_alpha), device=self.device, requires_grad=True)
        adapters = [p for p in policy.expert.adapters.parameters()]
        self.actor_opt = torch.optim.Adam([
            {"params": adapters, "lr": cfg.actor_lr}, {"params": [self.log_std], "lr": cfg.log_std_lr},
        ])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.updates = 0

    def actor_mean(self, b: dict) -> Tensor:
        policy, horizon = self.policy, self.policy.cfg.chunk_size
        n = b["x"].shape[0]
        xx = b["x"][:, None].expand(-1, horizon, -1)
        t = (b["q"].float() / horizon)[:, None].expand(-1, horizon)
        mask = query_mask(b["obs"].shape[1], horizon, self.device)
        v_full = policy.expert(b["obs"], xx, t, mask=mask, tactile=b["tac"], tactile_scale=1.0)
        return v_full[torch.arange(n, device=self.device), b["q"]].float() - b["v_slow"]

    def critic_state(self, b: dict) -> Tensor:
        horizon = self.policy.cfg.chunk_size
        return torch.cat([b["hidden"], b["reading"].flatten(1), b["x"], (b["q"].float() / horizon)[:, None]], -1)

    def update(self, buffer: ReplayBuffer, n_updates: int, rng: np.random.Generator) -> dict:
        cfg = self.cfg
        totals: dict[str, float] = {}
        for _ in range(n_updates):
            eps, steps = buffer.sample(cfg.batch_size, rng)
            b = buffer.gather(eps, steps, self.device)
            nb = buffer.gather(eps, steps, self.device, offset=1)
            state, next_state = self.critic_state(b), self.critic_state(nb)
            alpha = self.log_alpha.exp().detach()
            std = self.log_std.exp()
            with torch.no_grad():
                next_mean = self.actor_mean(nb)
                next_a = next_mean + std.detach() * torch.randn_like(next_mean)
                next_logp = gaussian_log_prob(next_a, next_mean, self.log_std.detach())
                tq1, tq2 = self.target(next_state, next_a / self.scale)
                y = b["r"] + cfg.gamma * (torch.min(tq1, tq2) - alpha * next_logp)
            q1, q2 = self.critic(state, b["action"] / self.scale)
            critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
            self.critic_opt.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
            self.critic_opt.step()

            mean = self.actor_mean(b)
            a = mean + std * torch.randn_like(mean)
            logp = gaussian_log_prob(a, mean, self.log_std)
            nq1, nq2 = self.critic(state, a / self.scale)
            actor_loss = (alpha * logp - torch.min(nq1, nq2)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            params = [p for g in self.actor_opt.param_groups for p in g["params"]]
            nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            self.actor_opt.step()
            self.critic_opt.zero_grad()  # the actor loss left gradients on the critic

            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

            with torch.no_grad():
                for p, tp in zip(self.critic.parameters(), self.target.parameters()):
                    tp.mul_(1 - cfg.tau).add_(cfg.tau * p)
            self.updates += 1
            step_stats = {
                "critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(), "alpha": alpha.item(),
                "q_mean": q1.mean().item(), "target_mean": y.mean().item(), "entropy": -logp.mean().item(),
                "mean_abs": mean.abs().mean().item(), "std_mean": std.mean().item(),
            }
            for k, v in step_stats.items():
                totals[k] = totals.get(k, 0.0) + v / n_updates
        return totals

    def save(self, path: Path) -> None:
        path = Path(path)
        tmp = path.with_suffix(".tmp")
        torch.save({
            "adapters": self.policy.expert.adapters.state_dict(), "log_std": self.log_std.detach(),
            "critic": self.critic.state_dict(), "target": self.target.state_dict(),
            "log_alpha": self.log_alpha.detach(), "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(), "alpha_opt": self.alpha_opt.state_dict(),
            "updates": self.updates,
        }, tmp)
        tmp.replace(path)

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.policy.expert.adapters.load_state_dict(state["adapters"])
        with torch.no_grad():
            self.log_std.copy_(state["log_std"])
            self.log_alpha.copy_(state["log_alpha"])
        self.critic.load_state_dict(state["critic"])
        self.target.load_state_dict(state["target"])
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])
        self.alpha_opt.load_state_dict(state["alpha_opt"])
        self.updates = state["updates"]
