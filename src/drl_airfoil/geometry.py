from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class AirfoilState:
    x_control: np.ndarray
    thickness: np.ndarray
    camber: np.ndarray


def dussauge_control_x(n_points: int) -> np.ndarray:
    """Sparse control points inspired by Dussauge et al. (no LE control point)."""
    if n_points == 12:
        return np.asarray(
            [0.01, 0.03, 0.07, 0.12, 0.20, 0.30, 0.42, 0.56, 0.70, 0.82, 0.93, 1.00],
            dtype=np.float32,
        )
    theta = np.linspace(0.0, np.pi, n_points + 1, dtype=np.float32)
    return (0.5 * (1.0 - np.cos(theta)))[1:]


def make_naca_like_state(
    n_points: int,
    rng: np.random.Generator,
    x_control: np.ndarray | None = None,
) -> AirfoilState:
    x = (
        np.asarray(x_control, dtype=np.float32)
        if x_control is not None
        else np.linspace(0.0, 1.0, n_points, dtype=np.float32)
    )

    t = float(rng.uniform(0.08, 0.13))
    m = float(rng.uniform(0.0, 0.04))
    p = float(rng.uniform(0.3, 0.6)) if m > 1e-6 else 0.4

    yt = np.maximum(
        5.0 * t * (
            0.2969 * np.sqrt(np.maximum(x, 1e-8))
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4
        ),
        0.0,
    ).astype(np.float32)

    camber = np.zeros_like(x, dtype=np.float32)
    if m > 0.0:
        before = x < p
        after = ~before
        camber[before] = m / (p**2) * (2.0 * p * x[before] - x[before] ** 2)
        camber[after] = m / ((1.0 - p) ** 2) * (
            (1.0 - 2.0 * p) + 2.0 * p * x[after] - x[after] ** 2
        )
    camber = np.maximum(camber, 0.0)

    # Zero TE camber/thickness if x=1 is part of controls.
    te_idx = int(np.argmin(np.abs(x - 1.0)))
    if np.isclose(x[te_idx], 1.0):
        camber[te_idx] = 0.0
        yt[te_idx] = 0.0

    return AirfoilState(x_control=x, thickness=2.0 * yt, camber=camber)


def apply_action(
    state: AirfoilState,
    rel_thickness_change: np.ndarray,
    delta_camber: np.ndarray,
    max_rel_thickness_change: float,
    max_camber_step: float,
    smooth_updates: bool,
) -> AirfoilState:
    rel_t = np.clip(rel_thickness_change, -max_rel_thickness_change, max_rel_thickness_change)
    dc = np.clip(delta_camber, -max_camber_step, max_camber_step)

    new_t = state.thickness * (1.0 + rel_t)
    new_c = state.camber + dc

    if smooth_updates:
        new_t = _smooth(new_t)
        new_c = _smooth(new_c)

    return AirfoilState(
        x_control=state.x_control,
        thickness=new_t.astype(np.float32),
        camber=new_c.astype(np.float32),
    )


def _smooth(arr: np.ndarray) -> np.ndarray:
    if arr.size < 3:
        return arr.copy()
    out = arr.copy()
    out[1:-1] = 0.25 * arr[:-2] + 0.5 * arr[1:-1] + 0.25 * arr[2:]
    return out


# ---------------------------------------------------------------------------
# Surface interpolation
# ---------------------------------------------------------------------------

def _surface_control_points(state: AirfoilState):
    """Build upper/lower control points with LE=0 and TE=1 anchors."""
    x = np.asarray(state.x_control, dtype=np.float32)
    t = np.maximum(np.asarray(state.thickness, dtype=np.float32), 0.0)
    c = np.asarray(state.camber, dtype=np.float32)

    yu = c + 0.5 * t
    yl = c - 0.5 * t

    # Strip existing LE/TE to avoid duplicates before re-adding them.
    keep = np.ones(x.size, dtype=bool)

    le_idx = int(np.argmin(np.abs(x)))
    if np.isclose(x[le_idx], 0.0):
        keep[le_idx] = False

    te_idx = int(np.argmin(np.abs(x - 1.0)))
    has_te = bool(np.isclose(x[te_idx], 1.0))
    y_te_u = float(yu[te_idx]) if has_te else float(yu[-1])
    y_te_l = float(yl[te_idx]) if has_te else float(yl[-1])
    if has_te:
        keep[te_idx] = False

    xi, yui, yli = x[keep], yu[keep], yl[keep]
    z = np.float32(0.0)
    one = np.float32(1.0)

    x_out = np.concatenate([[z], xi, [one]])
    y_u = np.concatenate([[z], yui, [np.float32(y_te_u)]])
    y_l = np.concatenate([[z], yli, [np.float32(y_te_l)]])
    return x_out, y_u, x_out, y_l


def _interp_sqrt_x(x_ctrl: np.ndarray, y_ctrl: np.ndarray, x_query: np.ndarray) -> np.ndarray:
    """Natural cubic spline in sqrt(x) space (C2-continuous)."""
    s = np.sqrt(np.clip(x_ctrl.astype(np.float64), 0.0, 1.0))
    y = y_ctrl.astype(np.float64)
    sq = np.sqrt(np.clip(x_query.astype(np.float64), 0.0, 1.0))

    n = s.size
    if n < 2:
        return np.full_like(sq, y[0] if n == 1 else 0.0, dtype=np.float32)
    if n == 2:
        return np.interp(sq, s, y).astype(np.float32)

    # Tridiagonal system for natural cubic spline second derivatives M[i].
    h = np.diff(s)
    dy = np.diff(y)
    nm = n - 2

    # Coefficients: a*M[i-1] + b*M[i] + c*M[i+1] = r
    a = h[:nm].copy()
    b = 2.0 * (h[:nm] + h[1:nm + 1])
    c = h[1:nm + 1].copy()
    r = 6.0 * (dy[1:nm + 1] / h[1:nm + 1] - dy[:nm] / h[:nm])

    # Thomas algorithm (forward sweep + back substitution).
    for i in range(1, nm):
        w = a[i] / b[i - 1]
        b[i] -= w * c[i - 1]
        r[i] -= w * r[i - 1]

    M = np.zeros(n, dtype=np.float64)
    M[nm] = r[nm - 1] / b[nm - 1]
    for i in range(nm - 2, -1, -1):
        M[i + 1] = (r[i] - c[i] * M[i + 2]) / b[i]

    # Evaluate spline.
    idx = np.clip(np.searchsorted(s, sq, side="right") - 1, 0, n - 2)
    hi = h[idx]
    t = sq - s[idx]
    u = s[idx + 1] - sq

    out = (
        M[idx] * u**3 / (6.0 * hi)
        + M[idx + 1] * t**3 / (6.0 * hi)
        + (y[idx] / hi - M[idx] * hi / 6.0) * u
        + (y[idx + 1] / hi - M[idx + 1] * hi / 6.0) * t
    )
    return out.astype(np.float32)


def to_surface_coordinates(state: AirfoilState, n_surface: int) -> np.ndarray:
    if n_surface < 2:
        raise ValueError("n_surface must be >= 2")

    x_u_ctrl, y_u_ctrl, x_l_ctrl, y_l_ctrl = _surface_control_points(state)

    theta = np.linspace(0.0, np.pi, n_surface, dtype=np.float32)
    x = 0.5 * (1.0 - np.cos(theta))

    y_upper = _interp_sqrt_x(x_u_ctrl, y_u_ctrl, x)
    y_lower = _interp_sqrt_x(x_l_ctrl, y_l_ctrl, x)

    upper = np.column_stack([x[::-1], y_upper[::-1]])
    lower = np.column_stack([x[1:], y_lower[1:]])
    return np.vstack([upper, lower]).astype(np.float32)


def make_observation(state: AirfoilState, prev_ld: float) -> np.ndarray:
    n = state.x_control.size
    half = 0.5 * state.thickness
    return np.concatenate([
        state.x_control,
        state.camber + half,
        state.camber - half,
        [np.clip(prev_ld / 100.0, -1.0, 1.0)],
    ]).astype(np.float32)
