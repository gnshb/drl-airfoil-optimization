from __future__ import annotations

from collections.abc import Callable
import numpy as np

from .config import HardConstraintConfig, SoftConstraintConfig
from .geometry import AirfoilState


def dussauge_hard_constraints(state: AirfoilState, cfg: HardConstraintConfig) -> AirfoilState:
    """Non-negative thickness, camber bounded by thickness, TE anchor, LE camber=0."""
    t = np.maximum(state.thickness.copy(), 0.0)
    c = np.clip(state.camber.copy(), -0.49 * t, 0.49 * t)

    te_idx = int(np.argmin(np.abs(state.x_control - 1.0)))
    if np.isclose(state.x_control[te_idx], 1.0):
        t[te_idx] = max(cfg.te_thickness, 0.0)
        c[te_idx] = 0.0

    if c.size > 0:
        c[0] = 0.0

    return AirfoilState(x_control=state.x_control, thickness=t, camber=c)


def dussauge_soft_constraints(state: AirfoilState, cfg: SoftConstraintConfig) -> tuple[float, dict[str, float]]:
    """Penalize max thickness outside allowed global bounds."""
    max_t = float(np.max(state.thickness))
    under = max(0.0, cfg.min_allowed_thickness - max_t)
    over = max(0.0, max_t - cfg.max_allowed_thickness)
    penalty = cfg.thickness_violation_penalty * (under + over)
    return penalty, {"max_thickness": max_t, "thickness_under": under, "thickness_over": over}


def apply_hard_constraints(
    state: AirfoilState,
    cfg: HardConstraintConfig,
    user_fn: Callable[[AirfoilState, HardConstraintConfig], AirfoilState] | None = None,
) -> AirfoilState:
    out = dussauge_hard_constraints(state, cfg)
    if user_fn is not None:
        out = user_fn(out, cfg)
    return out


def apply_soft_constraints(
    state: AirfoilState,
    cfg: SoftConstraintConfig,
    user_fn: Callable | None = None,
) -> tuple[float, dict[str, float]]:
    if not cfg.enabled:
        return 0.0, {}
    penalty, metrics = dussauge_soft_constraints(state, cfg)
    if user_fn is not None:
        p, m = user_fn(state, cfg)
        penalty += float(p)
        metrics.update(m)
    return penalty, metrics
