from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
from typing import Callable
import numpy as np

from .config import ConstraintConfig, EnvConfig, TrainingConfig
from .env import AirfoilEnv
from .evaluators import AeroEvaluator, NeuralFoilEvaluator, XFoilEvaluator
from .plotting import save_reward_plot

_EVALUATORS = {"xfoil": XFoilEvaluator, "neuralfoil": NeuralFoilEvaluator}


class _EpisodeReturnCallback:
    def __init__(self):
        from stable_baselines3.common.callbacks import BaseCallback

        class _CB(BaseCallback):
            def __init__(self, outer):
                super().__init__()
                self.outer = outer

            def _on_step(self) -> bool:
                infos = self.locals.get("infos", [])
                if infos and "episode" in infos[0]:
                    self.outer.episode_returns.append(float(infos[0]["episode"]["r"]))
                return True

        self.episode_returns: list[float] = []
        self.callback = _CB(self)


def train_ppo(
    evaluator_factory: Callable[[], AeroEvaluator],
    env_cfg: EnvConfig | None = None,
    constraint_cfg: ConstraintConfig | None = None,
    train_cfg: TrainingConfig | None = None,
    model_out: str | None = None,
    evaluator_name: str | None = None,
):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    except Exception as exc:
        raise ImportError("stable-baselines3 required. Install with: pip install -e .[rl]") from exc

    env_cfg = env_cfg or EnvConfig()
    constraint_cfg = constraint_cfg or ConstraintConfig()
    train_cfg = train_cfg or TrainingConfig()

    def make_env():
        return lambda: AirfoilEnv(evaluator=evaluator_factory(), env_cfg=env_cfg, constraint_cfg=constraint_cfg)

    n_envs = max(1, train_cfg.n_envs)
    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    vec_env = VecMonitor(vec_cls([make_env() for _ in range(n_envs)]))

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=train_cfg.learning_rate, n_steps=train_cfg.n_steps,
        batch_size=train_cfg.batch_size, n_epochs=train_cfg.n_epochs,
        gamma=train_cfg.gamma, gae_lambda=train_cfg.gae_lambda,
        clip_range=train_cfg.clip_range, ent_coef=train_cfg.ent_coef,
        verbose=0, policy_kwargs={"net_arch": [64, 64]},
    )

    cb = _EpisodeReturnCallback()
    t0 = perf_counter()
    try:
        model.learn(total_timesteps=train_cfg.total_timesteps, callback=cb.callback, progress_bar=True)
    except Exception:
        model.learn(total_timesteps=train_cfg.total_timesteps, callback=cb.callback)
    elapsed = perf_counter() - t0

    model_path = meta_path = None
    if model_out:
        out = Path(model_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(out))
        model_path = Path(f"{out}.zip")
        meta_path = out.with_suffix(".json")
        meta_path.write_text(json.dumps({
            "evaluator": evaluator_name,
            "env": asdict(env_cfg),
            "constraints": asdict(constraint_cfg),
            "training": asdict(train_cfg),
        }, indent=2))

    reward_plot_path = Path(f"{Path(model_out)}_rewards.png") if model_out else Path("outputs/training_rewards.png")
    save_reward_plot(cb.episode_returns, reward_plot_path)

    returns = np.asarray(cb.episode_returns, dtype=np.float32)
    print("training_done=1")
    print(f"timesteps={train_cfg.total_timesteps}")
    print(f"n_envs={n_envs}")
    print(f"episodes={returns.size}")
    if returns.size > 0:
        print(f"reward_first={float(returns[0]):.6f}")
        print(f"reward_last={float(returns[-1]):.6f}")
        print(f"reward_best={float(np.max(returns)):.6f}")
        print(f"reward_mean={float(np.mean(returns)):.6f}")
        print(f"reward_mean_last20={float(np.mean(returns[-20:])):.6f}")
        print(f"reward_std={float(np.std(returns)):.6f}")
    print(f"train_seconds={elapsed:.2f}")
    if model_path:
        print(f"model={model_path}")
    if meta_path:
        print(f"meta={meta_path}")
    print(f"reward_plot={reward_plot_path}")

    vec_env.close()
    return model, cb.episode_returns


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Train a DRL airfoil optimizer")
    p.add_argument("--evaluator", choices=_EVALUATORS, default="neuralfoil",
                    help="Aero backend (default: neuralfoil)")
    p.add_argument("--timesteps", type=int, default=20_000, help="Training timesteps (default: 20000)")
    p.add_argument("-j", "--n-envs", type=int, default=4, help="Parallel workers (default: 4)")
    p.add_argument("--model-out", type=str, default=None,
                    help="Save path (default: models/ppo_<evaluator>)")
    p.add_argument("--alpha", type=float, default=0.0, help="Angle of attack in degrees (default: 0)")
    p.add_argument("--re", type=float, default=1e6, help="Reynolds number (default: 1e6)")
    args = p.parse_args()

    train_ppo(
        evaluator_factory=_EVALUATORS[args.evaluator],
        env_cfg=EnvConfig(alpha=args.alpha, reynolds=args.re),
        train_cfg=TrainingConfig(total_timesteps=args.timesteps, n_envs=args.n_envs),
        model_out=args.model_out or f"models/ppo_{args.evaluator}",
        evaluator_name=args.evaluator,
    )


if __name__ == "__main__":
    main()
