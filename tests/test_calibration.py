"""Tests for pure-math confidence calibration utilities (research direction §5)."""

import pytest

from schemaforge.calibration import (
    RiskCoveragePoint,
    TemperatureScaler,
    coverage_at_risk,
    expected_calibration_error,
    fit_temperature,
    reliability_bins,
    risk_coverage_curve,
)

_PERFECT_CONFIDENCES = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]


def _perfect_calibrated_data():
    """20 samples per 0.05..0.95 confidence bin, empirical accuracy == confidence."""
    confidences: list[float] = []
    correct: list[bool] = []
    for c in _PERFECT_CONFIDENCES:
        n_correct = int(round(20 * c))
        confidences.extend([c] * 20)
        correct.extend([True] * n_correct + [False] * (20 - n_correct))
    return confidences, correct


def test_reliability_bins_perfectly_calibrated():
    confidences, correct = _perfect_calibrated_data()
    bins = reliability_bins(confidences, correct, n_bins=10)
    assert len(bins) == 10
    for i, b in enumerate(bins):
        assert b.count == 20
        assert b.lo == pytest.approx(i / 10)
        assert b.hi == pytest.approx((i + 1) / 10)
        assert b.confidence == pytest.approx(_PERFECT_CONFIDENCES[i])
        assert b.accuracy == pytest.approx(_PERFECT_CONFIDENCES[i])


def test_reliability_bins_confidence_one_lands_in_last_bin():
    bins = reliability_bins([1.0], [True], n_bins=10)
    assert bins[-1].count == 1
    assert bins[-1].hi == 1.0
    assert sum(b.count for b in bins) == 1


def test_reliability_bins_empty_bin():
    confidences = [0.95] * 10
    correct = [True] * 10
    bins = reliability_bins(confidences, correct, n_bins=10)
    for i, b in enumerate(bins):
        if i < 9:
            assert b.count == 0
            assert b.confidence == 0.0
            assert b.accuracy == 0.0
        else:
            assert b.count == 10
            assert b.confidence == pytest.approx(0.95)
            assert b.accuracy == pytest.approx(1.0)


def test_reliability_bins_raises_on_invalid_input():
    with pytest.raises(ValueError):
        reliability_bins([0.5], [])
    with pytest.raises(ValueError):
        reliability_bins([0.5], [True, False])
    with pytest.raises(ValueError):
        reliability_bins([], [])
    with pytest.raises(ValueError):
        reliability_bins([1.5], [True])
    with pytest.raises(ValueError):
        reliability_bins([-0.1], [True])
    with pytest.raises(ValueError):
        reliability_bins([0.5], [True], n_bins=0)


def test_ece_zero_for_perfectly_calibrated():
    confidences, correct = _perfect_calibrated_data()
    assert expected_calibration_error(confidences, correct) == pytest.approx(
        0.0, abs=1e-9
    )


def test_ece_positive_for_miscalibrated():
    confidences = [0.99] * 10
    correct = [True] * 5 + [False] * 5
    assert expected_calibration_error(confidences, correct) > 0.0


def test_ece_raises_on_invalid_input():
    with pytest.raises(ValueError):
        expected_calibration_error([0.5], [])
    with pytest.raises(ValueError):
        expected_calibration_error([], [])
    with pytest.raises(ValueError):
        expected_calibration_error([1.5], [True])


def test_risk_coverage_curve_hand_computed():
    confidences = [0.9, 0.8, 0.7, 0.6, 0.5]
    correct = [True, True, False, True, False]
    points = risk_coverage_curve(confidences, correct)
    assert [p.threshold for p in points] == [0.9, 0.8, 0.7, 0.6, 0.5]
    assert [p.coverage for p in points] == [0.2, 0.4, 0.6, 0.8, 1.0]
    assert points[0].risk == pytest.approx(0.0)
    assert points[1].risk == pytest.approx(0.0)
    assert points[2].risk == pytest.approx(1 / 3)
    assert points[3].risk == pytest.approx(0.25)
    assert points[4].risk == pytest.approx(0.4)


def test_risk_coverage_curve_monotonic_and_full_coverage():
    confidences = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.5]
    correct = [True, False, True, True, False, True, False]
    points = risk_coverage_curve(confidences, correct)
    coverages = [p.coverage for p in points]
    assert coverages == sorted(coverages)
    assert all(b > a for a, b in zip(coverages, coverages[1:]))
    assert points[-1].coverage == pytest.approx(1.0)


def test_risk_coverage_curve_raises_on_invalid_input():
    with pytest.raises(ValueError):
        risk_coverage_curve([], [])
    with pytest.raises(ValueError):
        risk_coverage_curve([0.5], [True, False])
    with pytest.raises(ValueError):
        risk_coverage_curve([1.5], [True])


def test_coverage_at_risk_empty_points():
    assert coverage_at_risk([], max_risk=0.5) == 0.0


def test_coverage_at_risk_no_point_meets_bound():
    points = [
        RiskCoveragePoint(coverage=0.4, risk=0.3, threshold=0.8),
        RiskCoveragePoint(coverage=1.0, risk=0.4, threshold=0.5),
    ]
    assert coverage_at_risk(points, max_risk=0.2) == 0.0


def test_coverage_at_risk_returns_max_coverage():
    points = [
        RiskCoveragePoint(coverage=0.2, risk=0.0, threshold=0.9),
        RiskCoveragePoint(coverage=0.4, risk=0.0, threshold=0.8),
        RiskCoveragePoint(coverage=0.6, risk=0.25, threshold=0.7),
        RiskCoveragePoint(coverage=0.8, risk=0.25, threshold=0.6),
        RiskCoveragePoint(coverage=1.0, risk=0.4, threshold=0.5),
    ]
    assert coverage_at_risk(points, max_risk=0.3) == pytest.approx(0.8)
    assert coverage_at_risk(points, max_risk=0.5) == pytest.approx(1.0)


def test_coverage_at_risk_negative_max_risk_raises():
    with pytest.raises(ValueError):
        coverage_at_risk(
            [RiskCoveragePoint(coverage=1.0, risk=0.1, threshold=0.5)],
            max_risk=-0.1,
        )


def test_temperature_scaler_identity_at_temperature_one():
    scaler = TemperatureScaler(1.0)
    assert scaler.calibrate(0.7) == pytest.approx(0.7, abs=1e-12)


def test_temperature_scaler_raises_on_invalid():
    with pytest.raises(ValueError):
        TemperatureScaler(0.0).calibrate(0.7)
    with pytest.raises(ValueError):
        TemperatureScaler(-1.0).calibrate(0.7)
    with pytest.raises(ValueError):
        TemperatureScaler(1.0).calibrate(-0.1)
    with pytest.raises(ValueError):
        TemperatureScaler(1.0).calibrate(1.1)


def test_temperature_scaler_clamps_endpoints():
    scaler = TemperatureScaler(2.0)
    low = scaler.calibrate(0.0)
    high = scaler.calibrate(1.0)
    assert 0.0 < low < 1.0
    assert 0.0 < high < 1.0
    assert low < 0.5 < high
    assert high > 0.5


def test_fit_temperature_reduces_ece_on_overconfident_data():
    confidences = [0.95] * 100
    correct = [True] * 50 + [False] * 50
    raw_ece = expected_calibration_error(confidences, correct)
    scaler = fit_temperature(confidences, correct)
    calibrated = [scaler.calibrate(c) for c in confidences]
    fitted_ece = expected_calibration_error(calibrated, correct)
    assert scaler.temperature > 1.0
    assert fitted_ece <= raw_ece + 1e-9


def test_fit_temperature_raises_on_invalid_input():
    with pytest.raises(ValueError):
        fit_temperature([], [])
    with pytest.raises(ValueError):
        fit_temperature([0.5], [True, False])
    with pytest.raises(ValueError):
        fit_temperature([0.5], [True], lo=0.0)
    with pytest.raises(ValueError):
        fit_temperature([0.5], [True], lo=2.0, hi=1.0)
    with pytest.raises(ValueError):
        fit_temperature([0.5], [True], steps=0)
