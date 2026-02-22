from __future__ import annotations

from dataclasses import dataclass, field

@dataclass(slots=True)
class HardConstraintConfig:
    # Dussauge hard limits on local updates are enforced in action space:
    # EnvConfig.max_rel_thickness_change (default 0.10 -> 90%-110%)
    # EnvConfig.max_camber_step (default 0.005 -> [-0.005, 0.005])
    # Keep a small finite trailing-edge thickness by default to avoid cusp-like TE.
    te_thickness: float = 0.002


@dataclass(slots=True)
class SoftConstraintConfig:
    # Dussauge soft constraint: penalize shapes outside global thickness limits.
    enabled: bool = True
    weight: float = 1.0
    min_allowed_thickness: float = 0.03
    max_allowed_thickness: float = 0.15
    thickness_violation_penalty: float = 50.0


@dataclass(slots=True)
class ConstraintConfig:
    hard: HardConstraintConfig = field(default_factory=HardConstraintConfig)
    soft: SoftConstraintConfig = field(default_factory=SoftConstraintConfig)


@dataclass(slots=True)
class EnvConfig:
    n_points: int = 12
    n_surface: int = 120
    max_steps: int = 50
    alpha: float = 0.0
    reynolds: float = 1_000_000.0
    max_rel_thickness_change: float = 0.10
    max_camber_step: float = 0.005
    change_penalty: float = 0.01
    max_consecutive_failures: int = 5
    invalid_reward: float = -2.0
    min_cd: float = 1e-6
    max_cd: float = 1.0
    max_abs_cl: float = 2.0
    min_ld: float = -50.0
    max_ld: float = 250.0
    smooth_updates: bool = True


@dataclass(slots=True)
class TrainingConfig:
    total_timesteps: int = 20_000
    n_envs: int = 4
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 128
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01


def _merge_dataclass(cls: type, values: dict):
    obj = cls()
    for k, v in values.items():
        if hasattr(obj, k):
            setattr(obj, k, v)
    return obj
