from __future__ import annotations

import math
from collections.abc import Callable
import numpy as np

from .config import ConstraintConfig, EnvConfig
from .constraints import apply_hard_constraints, apply_soft_constraints
from .evaluators import AeroEvaluator
from .geometry import (
    AirfoilState,
    apply_action,
    dussauge_control_x,
    make_naca_like_state,
    make_observation,
    to_surface_coordinates,
)

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:
    raise ImportError("gymnasium is required for AirfoilEnv. Install with: pip install -e .[rl]") from exc


class AirfoilEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        evaluator: AeroEvaluator,
        env_cfg: EnvConfig | None = None,
        constraint_cfg: ConstraintConfig | None = None,
        hard_constraint_fn: Callable | None = None,
        soft_constraint_fn: Callable | None = None,
    ) -> None:
        super().__init__()
        self.evaluator = evaluator
        self.cfg = env_cfg or EnvConfig()
        self.constraints = constraint_cfg or ConstraintConfig()
        self.hard_constraint_fn = hard_constraint_fn
        self.soft_constraint_fn = soft_constraint_fn

        n = self.cfg.n_points
        self.action_space = spaces.Box(
            low=np.concatenate([
                np.full(n, -self.cfg.max_rel_thickness_change, dtype=np.float32),
                np.full(n, -self.cfg.max_camber_step, dtype=np.float32),
            ]),
            high=np.concatenate([
                np.full(n, self.cfg.max_rel_thickness_change, dtype=np.float32),
                np.full(n, self.cfg.max_camber_step, dtype=np.float32),
            ]),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-1.5, high=1.5, shape=(n * 3 + 1,), dtype=np.float32)

        self.state: AirfoilState | None = None
        self._x_control = dussauge_control_x(self.cfg.n_points)
        self.current_step = 0
        self.prev_ld = 0.0
        self.consecutive_failures = 0
        self.last_cm = 0.0

    def _apply_hard(self, state: AirfoilState) -> AirfoilState:
        return apply_hard_constraints(state, self.constraints.hard, self.hard_constraint_fn)

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.consecutive_failures = 0

        for _ in range(10):
            candidate = self._apply_hard(
                make_naca_like_state(self.cfg.n_points, self.np_random, x_control=self._x_control)
            )
            result = self._evaluate(candidate)
            if self._valid_result(result):
                self.state = candidate
                self.prev_ld = float(result.ld)
                self.last_cm = float(result.cm)
                return make_observation(self.state, self.prev_ld), {}

        # Fallback if evaluator fails repeatedly.
        self.state = self._apply_hard(
            make_naca_like_state(self.cfg.n_points, np.random.default_rng(0), x_control=self._x_control)
        )
        self.prev_ld = 0.0
        self.last_cm = 0.0
        return make_observation(self.state, self.prev_ld), {}

    def step(self, action: np.ndarray):
        if self.state is None:
            raise RuntimeError("Call reset() first.")

        act = np.asarray(action, dtype=np.float32).ravel()
        n = self.cfg.n_points
        if act.size != 2 * n:
            raise ValueError(f"Expected action length {2 * n}, got {act.size}")

        candidate = self._apply_hard(apply_action(
            state=self.state,
            rel_thickness_change=act[:n],
            delta_camber=act[n:],
            max_rel_thickness_change=self.cfg.max_rel_thickness_change,
            max_camber_step=self.cfg.max_camber_step,
            smooth_updates=self.cfg.smooth_updates,
        ))

        result = self._evaluate(candidate)
        if not self._valid_result(result):
            self.consecutive_failures += 1
            return (
                make_observation(self.state, self.prev_ld),
                float(self.cfg.invalid_reward),
                self.consecutive_failures >= self.cfg.max_consecutive_failures,
                False,
                {"success": False, "LD": self.prev_ld, "CL": 0.0, "CD": 0.0, "Cm": self.last_cm},
            )

        self.consecutive_failures = 0
        ld = float(result.ld)
        delta_ld = ld - self.prev_ld

        action_penalty = self.cfg.change_penalty * float(
            np.mean(act[:n] ** 2) + np.mean((act[n:] / max(self.cfg.max_camber_step, 1e-8)) ** 2)
        )
        soft_penalty, soft_metrics = apply_soft_constraints(
            candidate, self.constraints.soft, self.soft_constraint_fn,
        )
        reward = delta_ld - action_penalty - self.constraints.soft.weight * soft_penalty

        self.state = candidate
        self.prev_ld = ld
        self.last_cm = float(result.cm)
        self.current_step += 1

        info = {
            "success": True,
            "LD": ld, "CL": float(result.cl), "CD": float(result.cd), "Cm": float(result.cm),
            "delta_LD": delta_ld, "action_penalty": action_penalty, "constraint_penalty": float(soft_penalty),
            **soft_metrics,
        }
        return make_observation(self.state, self.prev_ld), float(reward), self.current_step >= self.cfg.max_steps, False, info

    def _evaluate(self, state: AirfoilState):
        coords = to_surface_coordinates(state, self.cfg.n_surface)
        return self.evaluator.evaluate(coords=coords, alpha=self.cfg.alpha, reynolds=self.cfg.reynolds)

    def _valid_result(self, result) -> bool:
        if not result.success:
            return False
        cl, cd, ld = float(result.cl), float(result.cd), float(result.ld)
        if not (math.isfinite(cl) and math.isfinite(cd) and math.isfinite(ld)):
            return False
        if cd < self.cfg.min_cd or cd > self.cfg.max_cd:
            return False
        if abs(cl) > self.cfg.max_abs_cl:
            return False
        if ld < self.cfg.min_ld or ld > self.cfg.max_ld:
            return False
        return True
