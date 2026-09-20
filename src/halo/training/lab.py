"""TrainingLab — orchestrates PPO runs, evaluation, and status for /training.

Runs training on a background thread so the API can stream progress.
Evaluation runs on (a) a fixed sim holdout set and (b) recorded live
runs (*.jsonl) where HALO's own heuristic action is the teacher label.
"""

from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path

import numpy as np

from .features import ACTIONS, reward
from .ppo import PPOTrainer
from .simulator import Episode, holdout_set, sample_episode
from .features import featurize


class TrainingLab:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.trainer = PPOTrainer()
        self.rng = random.Random(7)
        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()

        self.episode_no = 0
        self.curve: list[dict] = []          # {ep, reward, acc}
        self.loss_curve: list[dict] = []     # {update, pi_loss, v_loss, entropy}
        self.scenario_stats: dict[str, dict] = {}   # name -> {n, r_sum, acc_sum}
        self.last_batch: dict | None = None
        self.last_episode: Episode | None = None
        self.last_actions: list[int] = []
        self.last_rewards: list[float] = []
        self.action_dist = [1 / 3, 1 / 3, 1 / 3]
        self.accuracy = 0.0
        self.eval_before: dict | None = None
        self.eval_after: dict | None = None
        self.training = False
        self._holdout = holdout_set(random.Random(99), n=30)

        ck = self.out_dir / "halo_ppo_latest.pt"
        if ck.exists():
            self.trainer.load(ck)

    # ---------------- training ----------------

    def train_async(self, episodes: int = 50, real_runs: bool = False) -> None:
        if self.training:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._train_loop, args=(episodes,), daemon=True)
        self.training = True
        self._thread.start()

    def _train_loop(self, n: int) -> None:
        try:
            for _ in range(n):
                if self._stop.is_set():
                    break
                eps = [sample_episode(self.rng) for _ in range(8)]
                res = self.trainer.collect(eps)
                losses = self.trainer.update(res["buf"])
                with self._lock:
                    self.episode_no += len(eps)
                    mean_r = sum(res["ep_rewards"]) / len(res["ep_rewards"])
                    mean_acc = sum(s["acc"] for s in res["ep_stats"]) / len(res["ep_stats"])
                    self.curve.append({"ep": self.episode_no,
                                       "reward": round(mean_r, 2),
                                       "acc": round(mean_acc, 3)})
                    self.curve = self.curve[-400:]
                    losses["update"] = self.trainer.updates
                    self.loss_curve.append(losses)
                    self.loss_curve = self.loss_curve[-400:]
                    for s in res["ep_stats"]:
                        st = self.scenario_stats.setdefault(
                            s["scenario"], {"n": 0, "r_sum": 0.0, "acc_sum": 0.0})
                        st["n"] += 1
                        st["r_sum"] += s["reward"]
                        st["acc_sum"] += s["acc"]
                    self.last_batch = {
                        "episodes": len(eps), "mean_reward": round(mean_r, 2),
                        "mean_acc": round(mean_acc, 3),
                        "scenarios": [s["scenario"] for s in res["ep_stats"]]}
                    self.accuracy = mean_acc
                    self.action_dist = res["action_dist"]
                    # keep the most interesting episode for the animation
                    self.last_episode = max(eps, key=lambda e:
                                            sum(s.label for s in e.steps))
                    self.last_actions = [self.trainer.net.act(s.state)[0]
                                         for s in self.last_episode.steps]
                    self.last_rewards = [
                        reward(a, s.label)
                        for a, s in zip(self.last_actions, self.last_episode.steps)]
                if self.episode_no % 20 == 0:
                    self.trainer.save(self.out_dir / "halo_ppo_latest.pt")
        finally:
            self.trainer.save(self.out_dir / "halo_ppo_latest.pt")
            self.training = False

    def stop(self) -> None:
        self._stop.set()

    # ---------------- evaluation ----------------

    def evaluate(self, run_files: list[Path] | None = None) -> dict:
        """Evaluate current policy on sim holdout + recorded real frames."""
        sim = self._eval_sim()
        real = self._eval_runs(run_files or [])
        out = {"sim": sim, "real": real, "ts": time.time()}
        with self._lock:
            if self.eval_before is None:
                self.eval_before = out
            else:
                self.eval_after = out
        return out

    def _eval_sim(self) -> dict:
        correct = total = false_w = missed_w = 0
        total_r = 0.0
        for ep in self._holdout:
            for st in ep.steps:
                a = int(np.argmax(self.trainer.policy_probs(st.state)))
                r = reward(a, st.label)
                total_r += r
                correct += a == st.label
                false_w += (a == 2 and st.label < 2)
                missed_w += (a < 2 and st.label == 2)
                total += 1
        return {"episodes": len(self._holdout),
                "mean_reward": round(total_r / max(total, 1) * 10, 2),
                "accuracy": round(correct / max(total, 1), 3),
                "false_alerts": false_w, "missed_threats": missed_w}

    def _eval_runs(self, run_files: list[Path]) -> dict:
        correct = total = false_w = missed_w = 0
        total_r = 0.0
        for path in run_files[-3:]:
            try:
                for line in Path(path).read_text().splitlines()[-400:]:
                    rec = json.loads(line)
                    scene = rec.get("context") or {}
                    for tr in rec.get("tracks", []):
                        state = featurize(tr, scene)
                        label = ACTIONS.index(tr.get("halo_action", "OBSERVE")) \
                            if tr.get("halo_action") in ACTIONS else 0
                        a = int(np.argmax(self.trainer.policy_probs(state)))
                        total_r += reward(a, label)
                        correct += a == label
                        false_w += (a == 2 and label < 2)
                        missed_w += (a < 2 and label == 2)
                        total += 1
            except Exception:
                continue
        if not total:
            return {"frames": 0, "note": "no recorded runs yet"}
        return {"frames": total,
                "mean_reward": round(total_r / total * 10, 2),
                "accuracy": round(correct / total, 3),
                "false_alerts": false_w, "missed_threats": missed_w}

    # ---------------- status ----------------

    def status(self) -> dict:
        with self._lock:
            ep = self.last_episode
            steps = []
            if ep:
                for i, s in enumerate(ep.steps):
                    steps.append({
                        "t": s.t, "pos": list(s.subject_pos),
                        "ambient": [list(p) for p in s.ambient],
                        "label": ACTIONS[s.label],
                        "action": ACTIONS[self.last_actions[i]] if i < len(self.last_actions) else "?",
                        "reward": self.last_rewards[i] if i < len(self.last_rewards) else 0.0,
                        "dist": round(float(np.hypot(*s.subject_pos)), 1),
                        "closing": round(s.closing_rate, 2),
                    })
            probs = None
            if ep:
                probs = [round(p, 2) for p in
                         self.trainer.policy_probs(ep.steps[-1].state)]
            scenarios = sorted(
                ({"name": k, "episodes": v["n"],
                  "mean_reward": round(v["r_sum"] / v["n"], 2),
                  "acc": round(v["acc_sum"] / v["n"], 3)}
                 for k, v in self.scenario_stats.items()),
                key=lambda s: s["mean_reward"])
            return {
                "training": self.training,
                "episode_no": self.episode_no,
                "curve": self.curve,
                "loss_curve": self.loss_curve,
                "scenarios": scenarios,
                "last_batch": self.last_batch,
                "mean_reward": self.curve[-1]["reward"] if self.curve else 0.0,
                "accuracy": round(self.accuracy, 3),
                "action_dist": [round(a, 2) for a in self.action_dist],
                "policy_probs": probs,
                "scenario": ep.scenario if ep else None,
                "context": (f"{ep.steps[0].environment}_{ep.steps[0].density}"
                            if ep else None),
                "steps": steps,
                "eval_before": self.eval_before,
                "eval_after": self.eval_after,
                "updates": self.trainer.updates,
            }
