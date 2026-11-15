import numpy as np
import pytest
from scipy.stats import ks_2samp

from anomaly.service.drifts import DriftMonitor, ks_pvalue, ks_statistic


def test_ks_statistic_matches_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 80), rng.normal(0.5, 1, 120)
    assert ks_statistic(a, b) == pytest.approx(ks_2samp(a, b).statistic)


def test_ks_pvalue_limits():
    assert ks_pvalue(0.0, 50, 50) == 1.0
    assert ks_pvalue(1.0, 50, 50) < 1e-6


def test_monitor_waits_for_min_samples():
    monitor = DriftMonitor([0.1, 0.2, 0.3], 0.0, 0.1, window=50, min_samples=5)
    for _ in range(4):
        assert monitor.observe(0.2, 0.0) is None
    assert monitor.observe(0.2, 0.0) is not None


def test_monitor_flags_shifted_scores_and_inputs():
    reference = np.linspace(0.1, 0.5, 50)
    monitor = DriftMonitor(reference, 0.0, 0.1, window=50, min_samples=10)
    snapshot = None
    for _ in range(10):
        snapshot = monitor.observe(2.0, 0.5)
    assert snapshot.anomaly_rate == 1.0
    assert snapshot.ks_statistic == pytest.approx(1.0)
    assert snapshot.ks_pvalue < 0.01
    assert snapshot.input_mean_z > 4


def test_monitor_is_quiet_when_nothing_changes():
    rng = np.random.default_rng(1)
    reference = rng.uniform(0.1, 0.5, 200)
    monitor = DriftMonitor(reference, 0.0, 0.1, window=100, min_samples=30)
    snapshot = None
    for ratio in rng.uniform(0.1, 0.5, 100):
        snapshot = monitor.observe(float(ratio), 0.0)
    assert snapshot.ks_statistic < 0.3
    assert snapshot.anomaly_rate == 0.0
    assert abs(snapshot.input_mean_z) < 1e-9


def test_window_evicts_old_samples():
    monitor = DriftMonitor([0.1, 0.2], None, None, window=5, min_samples=5)
    snapshot = None
    for _ in range(12):
        snapshot = monitor.observe(1.5, 0.0)
    assert snapshot.samples == 5


def test_without_references_only_basic_signals_exist():
    monitor = DriftMonitor([], None, None, window=10, min_samples=3)
    snapshot = None
    for _ in range(3):
        snapshot = monitor.observe(1.2, 0.0)
    assert snapshot.ks_statistic is None
    assert snapshot.input_mean_z is None
    assert snapshot.anomaly_rate == 1.0