from __future__ import annotations

import argparse
import csv
import json
import math
from copy import deepcopy
from pathlib import Path

from .config import (
    ConstraintConfig,
    EnvConfig,
    HardConstraintConfig,
    SoftConstraintConfig,
    _merge_dataclass,
)
from .env import AirfoilEnv
from .evaluators import NeuralFoilEvaluator, XFoilEvaluator
from .geometry import to_surface_coordinates
from .plotting import save_airfoil_plot

_EVALUATORS = {"xfoil": XFoilEvaluator, "neuralfoil": NeuralFoilEvaluator}


def _find_latest_model(directory: str = "models") -> str | None:
    """Find the most recently modified .zip model in a directory."""
    d = Path(directory)
    if not d.is_dir():
        return None
    zips = sorted(d.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        return None
    # PPO.load expects the path without .zip
    return str(zips[0].with_suffix(""))


def _load_metadata(model_path: str | Path) -> dict:
    meta = Path(model_path).with_suffix(".json")
    if not meta.exists():
        return {}
    try:
        return json.loads(meta.read_text())
    except Exception:
        return {}


def _load_training_config(meta: dict) -> tuple[EnvConfig, ConstraintConfig]:
    env = _merge_dataclass(EnvConfig, meta.get("env", {}))
    cd = meta.get("constraints", {})
    hard = _merge_dataclass(HardConstraintConfig, cd.get("hard", {}))
    soft = _merge_dataclass(SoftConstraintConfig, cd.get("soft", {}))
    return env, ConstraintConfig(hard=hard, soft=soft)


def _resolve_evaluator(meta: dict, model_path: str | Path) -> str:
    meta_eval = str(meta.get("evaluator", "")).strip().lower()
    if meta_eval in _EVALUATORS:
        return meta_eval
    stem = Path(model_path).stem.lower()
    for name in _EVALUATORS:
        if name in stem:
            return name
    return "xfoil"


def generate_airfoil(
    model_path: str,
    seed: int = 0,
    out_csv: str = "outputs/generated_airfoil.csv",
    out_png: str = "outputs/generated_airfoil.png",
) -> tuple[Path, Path]:
    try:
        from stable_baselines3 import PPO
    except Exception as exc:
        raise ImportError("stable-baselines3 required. Install with: pip install -e .[rl]") from exc

    meta = _load_metadata(model_path)
    env_cfg, constraint_cfg = _load_training_config(meta)
    eval_name = _resolve_evaluator(meta, model_path)

    rollout_backend = _EVALUATORS[eval_name]()
    final_backend = XFoilEvaluator()

    env = AirfoilEnv(evaluator=rollout_backend, env_cfg=env_cfg, constraint_cfg=constraint_cfg)
    model = PPO.load(model_path)
    obs, _ = env.reset(seed=seed)

    best_state = best_xfoil_res = None
    best_xfoil_ld = float("-inf")
    best_step = -1

    for _ in range(env_cfg.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if env.state is not None:
            res = final_backend.evaluate(
                coords=to_surface_coordinates(env.state, env_cfg.n_surface),
                alpha=env_cfg.alpha, reynolds=env_cfg.reynolds,
            )
            if res.success and math.isfinite(res.ld) and res.ld > best_xfoil_ld:
                best_xfoil_ld = float(res.ld)
                best_step = env.current_step
                best_state = deepcopy(env.state)
                best_xfoil_res = res
        if terminated or truncated:
            break

    if env.state is None:
        raise RuntimeError("Environment produced no state.")

    selected = best_state if best_state is not None else env.state
    coords = to_surface_coordinates(selected, env_cfg.n_surface)

    # Save outputs.
    csv_path = Path(out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y"])
        for x, y in coords:
            w.writerow([float(x), float(y)])

    png_path = save_airfoil_plot(coords, out_png)

    final_res = best_xfoil_res or final_backend.evaluate(
        coords=coords, alpha=env_cfg.alpha, reynolds=env_cfg.reynolds,
    )

    print(f"model:     {model_path}")
    print(f"evaluator: {eval_name}")
    print(f"success:   {bool(final_res.success)}")
    if best_state is not None:
        print(f"best step: {best_step}")
    print(f"L/D:       {float(final_res.ld):.4f}")
    print(f"CL:        {float(final_res.cl):.6f}")
    print(f"CD:        {float(final_res.cd):.6f}")
    print(f"Cm:        {float(final_res.cm):.6f}")
    print(f"csv:       {csv_path}")
    print(f"png:       {png_path}")

    return csv_path, png_path


def main() -> None:
    p = argparse.ArgumentParser(description="Generate an airfoil from a trained model")
    p.add_argument("--model", type=str, default=None,
                    help="Model path (default: latest .zip in models/)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-csv", type=str, default="outputs/generated_airfoil.csv")
    p.add_argument("--out-png", type=str, default="outputs/generated_airfoil.png")
    args = p.parse_args()

    model_path = args.model or _find_latest_model()
    if model_path is None:
        p.error("No model found. Train one first with: python train.py\n"
                "Or specify a path with: python generate.py --model path/to/model")

    generate_airfoil(model_path=model_path, seed=args.seed,
                     out_csv=args.out_csv, out_png=args.out_png)


if __name__ == "__main__":
    main()
