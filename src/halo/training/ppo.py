"""Minimal PPO actor-critic for the 3-action attention policy.

Deliberately small: a 2x64 MLP over the FEATURE_DIM world-state vector.
PPO chosen for stability and recognizability, not because the task needs it.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .features import ACTIONS, FEATURE_DIM, reward


class ActorCritic(nn.Module):
    def __init__(self, dim: int = FEATURE_DIM, n_actions: int = len(ACTIONS)):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.actor = nn.Linear(64, n_actions)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        h = self.body(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    @torch.no_grad()
    def act(self, state: np.ndarray):
        x = torch.as_tensor(state, dtype=torch.float32, device=next(self.parameters()).device)
        logits, v = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return int(a), float(dist.log_prob(a)), float(v), torch.softmax(logits, -1)


class PPOTrainer:
    def __init__(self, lr: float = 1e-3, gamma: float = 0.0, lam: float = 0.95,
                 clip: float = 0.2, ent_coef: float = 0.005, device: str = "cpu"):
        self.net = ActorCritic().to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma, self.lam, self.clip, self.ent_coef = gamma, lam, clip, ent_coef
        self.device = device
        self.updates = 0

    def collect(self, episodes) -> dict:
        """Roll out the policy over sim episodes; returns buffers + metrics."""
        buf = {"s": [], "a": [], "logp": [], "r": [], "v": [], "done": []}
        ep_rewards, actions_taken, ep_stats = [], [], []
        total = 0
        for ep in episodes:
            ep_r = 0.0
            ep_correct = 0
            prev_over = False
            for i, st in enumerate(ep.steps):
                a, logp, v, _ = self.net.act(st.state)
                # early-warning bonus: correct WARN >= 0.8s before closest approach
                early = (a == 2 and st.label == 2
                         and st.min_future_dist > 1.6)
                r = reward(a, st.label, early=early,
                           repeated_bother=prev_over)
                prev_over = a > st.label
                buf["s"].append(st.state); buf["a"].append(a)
                buf["logp"].append(logp)
                buf["r"].append(r * 0.25)   # scale for value-fn stability
                buf["v"].append(v); buf["done"].append(0.0)
                ep_r += r; actions_taken.append(a)
                ep_correct += (a == st.label); total += 1
            buf["done"][-1] = 1.0
            ep_rewards.append(ep_r)
            ep_stats.append({"scenario": ep.scenario, "reward": ep_r,
                             "acc": ep_correct / max(len(ep.steps), 1)})
        return {"buf": buf, "ep_rewards": ep_rewards, "ep_stats": ep_stats,
                "action_dist": [actions_taken.count(i) / max(total, 1)
                                for i in range(3)]}

    def update(self, buf: dict, epochs: int = 10) -> None:
        s = torch.as_tensor(np.array(buf["s"]), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(buf["a"], device=self.device)
        old_logp = torch.as_tensor(buf["logp"], dtype=torch.float32, device=self.device)
        v = torch.as_tensor(buf["v"], dtype=torch.float32)
        r = buf["r"]; done = buf["done"]

        # GAE. Reward here is immediate per-step (attention is a per-frame
        # decision), so gamma=0 reduces this to a contextual-bandit advantage:
        # r - V(s). The critic learns E[reward|state], not a return.
        adv = np.zeros(len(r), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(r))):
            next_v = 0.0 if t == len(r) - 1 else float(v[t + 1])
            delta = r[t] + self.gamma * next_v * (1 - done[t]) - float(v[t])
            gae = delta + self.gamma * self.lam * (1 - done[t]) * gae
            adv[t] = gae
        ret_t = torch.as_tensor(adv, device=self.device) + torch.as_tensor(v, device=self.device)
        adv_t = torch.as_tensor(adv, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        pi_l = v_l = ent = 0.0
        for _ in range(epochs):
            logits, vals = self.net(s)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(a)
            ratio = torch.exp(logp - old_logp)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t
            pi_loss = -torch.min(surr1, surr2).mean()
            v_loss = (ret_t - vals).pow(2).mean()
            entropy = dist.entropy().mean()
            loss = pi_loss + 0.5 * v_loss - self.ent_coef * entropy
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.opt.step()
            pi_l, v_l, ent = float(pi_loss), float(v_loss), float(entropy)
        self.updates += 1
        return {"pi_loss": round(pi_l, 4), "v_loss": round(v_l, 4),
                "entropy": round(ent, 4)}

    def policy_probs(self, state: np.ndarray) -> list[float]:
        with torch.no_grad():
            logits, _ = self.net(torch.as_tensor(state, dtype=torch.float32,
                                                 device=self.device))
            return torch.softmax(logits, -1).tolist()

    def save(self, path) -> None:
        torch.save({"model": self.net.state_dict(), "updates": self.updates}, path)

    def load(self, path) -> bool:
        try:
            ck = torch.load(path, map_location=self.device, weights_only=True)
            self.net.load_state_dict(ck["model"])
            self.updates = ck.get("updates", 0)
            return True
        except Exception:
            return False
