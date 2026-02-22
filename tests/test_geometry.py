import numpy as np

from drl_airfoil.geometry import (
    AirfoilState,
    apply_action,
    dussauge_control_x,
    make_observation,
    to_surface_coordinates,
)


def test_surface_coordinates_shape():
    n = 12
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    state = AirfoilState(
        x_control=x,
        thickness=np.linspace(0.0, 0.1, n, dtype=np.float32),
        camber=np.zeros(n, dtype=np.float32),
    )
    coords = to_surface_coordinates(state, n_surface=40)
    assert coords.shape == (79, 2)


def test_apply_action_clips_and_updates():
    n = 6
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    state = AirfoilState(
        x_control=x,
        thickness=np.full(n, 0.1, dtype=np.float32),
        camber=np.zeros(n, dtype=np.float32),
    )

    nxt = apply_action(
        state,
        rel_thickness_change=np.full(n, 1.0, dtype=np.float32),
        delta_camber=np.full(n, 1.0, dtype=np.float32),
        max_rel_thickness_change=0.1,
        max_camber_step=0.005,
        smooth_updates=False,
    )

    assert np.allclose(nxt.thickness, 0.11)
    assert np.allclose(nxt.camber, 0.005)


def test_observation_size():
    n = 10
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    state = AirfoilState(
        x_control=x,
        thickness=np.full(n, 0.1, dtype=np.float32),
        camber=np.zeros(n, dtype=np.float32),
    )
    obs = make_observation(state, prev_ld=42.0)
    assert obs.shape == (n * 3 + 1,)


def test_dussauge_control_points_no_le():
    x = dussauge_control_x(12)
    assert x.size == 12
    assert x[0] > 0.0
    assert np.isclose(x[-1], 1.0)
