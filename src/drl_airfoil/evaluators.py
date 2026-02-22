from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import numpy as np


@dataclass(slots=True)
class AeroResult:
    success: bool
    cl: float
    cd: float
    cm: float

    @property
    def ld(self) -> float:
        if not self.success or self.cd <= 0.0:
            return float("nan")
        return self.cl / self.cd


_FAIL = dict(success=False, cl=0.0, cd=1.0, cm=0.0)


class AeroEvaluator(ABC):
    @abstractmethod
    def evaluate(self, coords: np.ndarray, alpha: float, reynolds: float) -> AeroResult:
        raise NotImplementedError


class XFoilEvaluator(AeroEvaluator):
    def __init__(self, max_iter: int = 50, n_crit: float = 9.0, xtr: tuple[float, float] = (0.05, 0.05)) -> None:
        try:
            from xfoil import XFoil
            from xfoil.model import Airfoil
        except Exception as exc:
            raise ImportError("xfoil not installed. Install with: pip install -e .[xfoil]") from exc

        self._Airfoil = Airfoil
        self._xf = XFoil()
        self._xf.print = False
        self._xf.max_iter = int(max_iter)
        self._xf.n_crit = float(n_crit)
        self._xf.xtr = tuple(float(v) for v in xtr)

    def evaluate(self, coords: np.ndarray, alpha: float, reynolds: float) -> AeroResult:
        arr = np.asarray(coords, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return AeroResult(**_FAIL)
        try:
            self._xf.airfoil = self._Airfoil(arr[:, 0], arr[:, 1])
            self._xf.Re = float(reynolds)
            cl, cd, cm, _ = self._xf.a(float(alpha))
        except Exception:
            return AeroResult(**_FAIL)

        cl = float(cl) if cl is not None else float("nan")
        cd = float(cd) if cd is not None else float("nan")
        cm = float(cm) if cm is not None else 0.0
        return AeroResult(success=math.isfinite(cl) and math.isfinite(cd) and cd > 0.0, cl=cl, cd=cd, cm=cm)


class NeuralFoilEvaluator(AeroEvaluator):
    def __init__(self, model_size: str = "medium") -> None:
        try:
            import neuralfoil as nf
        except Exception as exc:
            raise ImportError("neuralfoil not installed. Install with: pip install -e .[neuralfoil]") from exc
        self._nf = nf
        self.model_size = model_size

    def evaluate(self, coords: np.ndarray, alpha: float, reynolds: float) -> AeroResult:
        try:
            aero = self._nf.get_aero_from_coordinates(
                coordinates=np.asarray(coords, dtype=np.float32),
                alpha=float(alpha),
                Re=float(reynolds),
                model_size=self.model_size,
            )
        except Exception:
            return AeroResult(**_FAIL)

        cl = _extract(aero, ("CL", "cl"), 0.0)
        cd = _extract(aero, ("CD", "cd"), 1.0)
        cm = _extract(aero, ("CM", "Cm", "cm"), 0.0)
        return AeroResult(success=math.isfinite(cl) and math.isfinite(cd) and cd > 0.0, cl=cl, cd=cd, cm=cm)


def _extract(d: dict, keys: tuple[str, ...], default: float) -> float:
    for k in keys:
        if k in d:
            v = np.asarray(d[k], dtype=np.float32).ravel()
            if v.size > 0:
                return float(v[0])
    return default
