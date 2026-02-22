from __future__ import annotations

import argparse
import math
from time import perf_counter

import numpy as np

from .config import HardConstraintConfig
from .constraints import apply_hard_constraints
from .evaluators import NeuralFoilEvaluator, XFoilEvaluator
from .geometry import dussauge_control_x, make_naca_like_state, to_surface_coordinates


def _make_cases(n_cases: int, n_points: int, n_surface: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    x_ctrl = dussauge_control_x(n_points)
    cfg = HardConstraintConfig()
    cases = []
    for _ in range(n_cases):
        state = apply_hard_constraints(make_naca_like_state(n_points, rng, x_ctrl), cfg)
        cases.append(to_surface_coordinates(state, n_surface))
    return cases


def _run_backend(name: str, evaluator, cases: list[np.ndarray], alpha: float, reynolds: float, warmup: int):
    for c in cases[:warmup]:
        evaluator.evaluate(c, alpha=alpha, reynolds=reynolds)

    durations, cl_vals, cd_vals, ld_vals, ok = [], [], [], [], []
    for coords in cases:
        t0 = perf_counter()
        res = evaluator.evaluate(coords, alpha=alpha, reynolds=reynolds)
        durations.append(perf_counter() - t0)
        ok.append(bool(res.success))
        cl_vals.append(float(res.cl))
        cd_vals.append(float(res.cd))
        ld_vals.append(float(res.ld))

    n = len(cases)
    total = sum(durations)
    arr = np.asarray(durations, dtype=np.float64)
    return {
        "name": name, "n": n, "successes": sum(ok),
        "mean_ms": total / n * 1000 if n else float("nan"),
        "p50_ms": float(np.percentile(arr, 50) * 1000),
        "p95_ms": float(np.percentile(arr, 95) * 1000),
        "evals_per_sec": n / total if total > 0 else float("nan"),
        "cl": cl_vals, "cd": cd_vals, "ld": ld_vals, "ok": ok,
    }


def _pairwise_deltas(a: dict, b: dict) -> dict:
    idx = [i for i, (sa, sb) in enumerate(zip(a["ok"], b["ok"])) if sa and sb]
    if not idx:
        return {"n": 0, "cl": float("nan"), "cd": float("nan"), "ld": float("nan"), "ld_med": float("nan")}

    def delta(key):
        return np.abs(np.float64(a[key])[idx] - np.float64(b[key])[idx])

    dcl, dcd, dld = delta("cl"), delta("cd"), delta("ld")
    return {"n": len(idx), "cl": float(np.mean(dcl)), "cd": float(np.mean(dcd)),
            "ld": float(np.mean(dld)), "ld_med": float(np.median(dld))}


def run_benchmark(n_cases: int, n_points: int, n_surface: int,
                  alpha: float, reynolds: float, seed: int, warmup: int,
                  neuralfoil_model_size: str):
    cases = _make_cases(n_cases, n_points, n_surface, seed)

    xf = _run_backend("xfoil", XFoilEvaluator(), cases, alpha, reynolds, warmup)
    nf = _run_backend("neuralfoil", NeuralFoilEvaluator(model_size=neuralfoil_model_size),
                      cases, alpha, reynolds, warmup)
    d = _pairwise_deltas(xf, nf)

    print(f"cases={n_cases} n_points={n_points} n_surface={n_surface} alpha={alpha} Re={reynolds:.3e} seed={seed}")
    print("backend      success   mean_ms   p50_ms   p95_ms   evals/s   mean_LD_success")
    for row in (xf, nf):
        ld_ok = [v for v, s in zip(row["ld"], row["ok"]) if s and math.isfinite(v)]
        mean_ld = float(np.mean(ld_ok)) if ld_ok else float("nan")
        print(f"{row['name']:<12} {row['successes']:>3}/{row['n']:<3}"
              f"   {row['mean_ms']:>7.3f}  {row['p50_ms']:>7.3f}  {row['p95_ms']:>7.3f}"
              f"  {row['evals_per_sec']:>7.2f}   {mean_ld:>8.3f}")
    print(f"common_success={d['n']} mean_abs_delta(CL)={d['cl']:.5f} "
          f"mean_abs_delta(CD)={d['cd']:.5f} mean_abs_delta(LD)={d['ld']:.5f} "
          f"median_abs_delta(LD)={d['ld_med']:.5f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark xfoil vs neuralfoil on random airfoils")
    p.add_argument("--n-cases", type=int, default=50, help="Number of airfoils to test (default: 50)")
    p.add_argument("--alpha", type=float, default=0.0, help="Angle of attack (default: 0)")
    p.add_argument("--re", type=float, default=1e6, help="Reynolds number (default: 1e6)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_benchmark(
        n_cases=args.n_cases, n_points=12, n_surface=120,
        alpha=args.alpha, reynolds=args.re, seed=args.seed, warmup=3,
        neuralfoil_model_size="medium",
    )


if __name__ == "__main__":
    main()
