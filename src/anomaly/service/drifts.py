import math
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: max gap between the empirical CDFs."""
    a, b = np.sort(a), np.sort(b)
    values = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, values, side="right") / len(a)
    cdf_b = np.searchsorted(b, values, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def ks_pvalue(statistic: float, n: int, m: int) -> float:
    """Asymptotic p-value for the two-sample KS test (approximate for small samples)."""
    effective = math.sqrt(n * m / (n + m))
    lam = (effective + 0.12 + 0.11 / effective) * statistic
    if lam < 0.2:
        return 1.0
    total = 0.0
    for k in range(1, 101):
        term = (-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam)
        total += term
        if abs(term) < 1e-12:
            break
    return float(min(1.0, max(0.0, 2.0 * total)))


@dataclass(frozen=True)
class DriftSnapshot:
    samples: int
    anomaly_rate: float
    mean_score_ratio: float
    ks_statistic: float | None
    ks_pvalue: float | None
    input_mean_z: float | None


class DriftMonitor:
    """Rolling-window drift signals for one category. Thread-safe."""

    def __init__(
        self,
        reference_ratios,
        reference_input_mean: float | None,
        reference_input_std: float | None,
        window: int = 200,
        min_samples: int = 30,
    ) -> None:
        self._reference = np.asarray(reference_ratios, dtype=np.float64)
        self._ref_mean = reference_input_mean
        self._ref_std = reference_input_std
        self._min_samples = min(min_samples, window)
        self._ratios: deque[float] = deque(maxlen=window)
        self._input_means: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()

    @property
    def has_score_reference(self) -> bool:
        return len(self._reference) >= 2

    def observe(self, score_ratio: float, input_mean: float) -> DriftSnapshot | None:
        """Record one prediction. Returns None until enough samples have accumulated."""
        with self._lock:
            self._ratios.append(score_ratio)
            self._input_means.append(input_mean)
            if len(self._ratios) < self._min_samples:
                return None
            ratios = np.array(self._ratios)
            means = np.array(self._input_means)

        ks = p_value = None
        if self.has_score_reference:
            ks = ks_statistic(ratios, self._reference)
            p_value = ks_pvalue(ks, len(ratios), len(self._reference))

        z = None
        if self._ref_mean is not None and self._ref_std:
            standard_error = self._ref_std / math.sqrt(len(means))
            z = float((means.mean() - self._ref_mean) / standard_error)

        return DriftSnapshot(
            samples=len(ratios),
            anomaly_rate=float(np.mean(ratios >= 1.0)),
            mean_score_ratio=float(ratios.mean()),
            ks_statistic=ks,
            ks_pvalue=p_value,
            input_mean_z=z,
        )