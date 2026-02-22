import numpy as np

from drl_airfoil.config import HardConstraintConfig, SoftConstraintConfig
from drl_airfoil.constraints import apply_hard_constraints, apply_soft_constraints
from drl_airfoil.geometry import AirfoilState, dussauge_control_x


def test_hard_constraints_enforce_te_anchor_and_positive_thickness():
    x_ctrl = dussauge_control_x(12)
    state = AirfoilState(
        x_control=x_ctrl,
        thickness=np.array([0.02] * 11 + [-0.03], dtype=np.float32),
        camber=np.array([0.01] * 11 + [0.04], dtype=np.float32),
    )

    out = apply_hard_constraints(state, HardConstraintConfig())
    assert np.all(out.thickness >= 0.0)
    assert np.all(np.abs(out.camber) <= 0.49 * out.thickness + 1e-7)
    assert out.camber[0] == 0.0
    assert np.isclose(out.thickness[-1], HardConstraintConfig().te_thickness)
    assert out.camber[-1] == 0.0


def test_soft_constraints_disabled():
    x = np.linspace(0.0, 1.0, 12, dtype=np.float32)
    state = AirfoilState(
        x_control=x,
        thickness=np.full(12, 0.02, dtype=np.float32),
        camber=np.zeros(12, dtype=np.float32),
    )
    penalty, metrics = apply_soft_constraints(state, SoftConstraintConfig(enabled=False))
    assert penalty == 0.0
    assert metrics == {}


def test_soft_constraints_penalize_thickness_violation():
    x = np.linspace(0.0, 1.0, 12, dtype=np.float32)
    state = AirfoilState(
        x_control=x,
        thickness=np.full(12, 0.2, dtype=np.float32),
        camber=np.zeros(12, dtype=np.float32),
    )
    cfg = SoftConstraintConfig(enabled=True, weight=1.0, max_allowed_thickness=0.15, thickness_violation_penalty=50.0)
    penalty, metrics = apply_soft_constraints(state, cfg)
    assert penalty > 0.0
    assert metrics["thickness_over"] > 0.0
