"""Pure-math CPU-only confidence-calibration utilities.

Implements the reliability-diagram binning, expected calibration error,
risk-coverage curve and temperature-scaling calibrator behind research
direction §5 ("Confidence-Aware Inference").  No torch, no model loading, no
network, no file I/O: the module consumes plain floats/bools that a separate
GPU-side caller has already produced by running a model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ReliabilityBin:
    """One equal-width confidence bin of a reliability diagram."""

    lo: float  # bin lower edge, inclusive
    hi: float  # bin upper edge, exclusive (except the last bin, which is inclusive of 1.0)
    confidence: float  # mean predicted confidence of samples in this bin; 0.0 if empty
    accuracy: float  # mean correctness (fraction correct) of samples in this bin; 0.0 if empty
    count: int  # number of samples in this bin


def reliability_bins(
    confidences: list[float], correct: list[bool], n_bins: int = 10
) -> list[ReliabilityBin]:
    """Bin (confidence, correct) pairs into n_bins equal-width bins over [0, 1].

    Raises ValueError if `confidences` and `correct` have different lengths, if
    either is empty, if any confidence is outside [0, 1], or if n_bins < 1.  Bin
    edges are `i/n_bins` to `(i+1)/n_bins` for i in range(n_bins); the last
    bin's upper edge is inclusive so a confidence of exactly 1.0 lands in the
    last bin, not off the end.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have equal lengths")
    if not confidences:
        raise ValueError("confidences must be non-empty")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    for c in confidences:
        if not 0.0 <= c <= 1.0:
            raise ValueError("confidences must all lie within [0, 1]")

    counts = [0] * n_bins
    confidence_sum = [0.0] * n_bins
    correct_sum = [0] * n_bins
    for c, ok in zip(confidences, correct):
        idx = min(int(c * n_bins), n_bins - 1)
        counts[idx] += 1
        confidence_sum[idx] += c
        if ok:
            correct_sum[idx] += 1

    return [
        ReliabilityBin(
            lo=i / n_bins,
            hi=(i + 1) / n_bins,
            confidence=confidence_sum[i] / counts[i] if counts[i] else 0.0,
            accuracy=correct_sum[i] / counts[i] if counts[i] else 0.0,
            count=counts[i],
        )
        for i in range(n_bins)
    ]


def expected_calibration_error(
    confidences: list[float], correct: list[bool], n_bins: int = 10
) -> float:
    """ECE: sum over non-empty bins of (count/total) * |accuracy - confidence|.

    Built on top of `reliability_bins` (reuse it, do not reimplement the
    binning).  Raises the same ValueErrors as `reliability_bins` for invalid
    input (let them propagate).
    """
    bins = reliability_bins(confidences, correct, n_bins=n_bins)
    total = len(confidences)
    return sum(
        (b.count / total) * abs(b.accuracy - b.confidence)
        for b in bins
        if b.count > 0
    )


@dataclass(frozen=True)
class RiskCoveragePoint:
    """One operating point on a risk-coverage curve."""

    coverage: float  # fraction of all samples retained at this threshold (0, 1]
    risk: float  # error rate (1 - accuracy) among retained samples
    threshold: float  # the confidence threshold used (samples with confidence >= threshold are retained)


def risk_coverage_curve(
    confidences: list[float], correct: list[bool]
) -> list[RiskCoveragePoint]:
    """Sweep every distinct confidence value (descending) as a retention threshold.

    For each distinct value v in sorted(set(confidences), reverse=True), retain
    every sample with confidence >= v, compute coverage = retained_count / total
    and risk = 1 - (correct_count_among_retained / retained_count).  Returns one
    RiskCoveragePoint per distinct threshold, in descending threshold order (so
    the first point has the smallest coverage / highest-confidence-only subset,
    the last point has coverage 1.0).  Raises ValueError on empty or
    mismatched-length input, same style as `reliability_bins`.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have equal lengths")
    if not confidences:
        raise ValueError("confidences must be non-empty")
    for c in confidences:
        if not 0.0 <= c <= 1.0:
            raise ValueError("confidences must all lie within [0, 1]")

    total = len(confidences)
    points: list[RiskCoveragePoint] = []
    for threshold in sorted(set(confidences), reverse=True):
        retained_correct = sum(
            1 for c, ok in zip(confidences, correct) if c >= threshold and ok
        )
        retained_count = sum(1 for c in confidences if c >= threshold)
        points.append(
            RiskCoveragePoint(
                coverage=retained_count / total,
                risk=1.0 - (retained_correct / retained_count),
                threshold=threshold,
            )
        )
    return points


def coverage_at_risk(points: list[RiskCoveragePoint], max_risk: float) -> float:
    """Return the maximum coverage achievable while keeping risk <= max_risk.

    Scan `points` (assumed already sorted descending by threshold / ascending...
    do not assume an order from the caller -- explicitly find, among all points
    with risk <= max_risk, the one with the greatest coverage, and return its
    coverage).  Returns 0.0 if no point satisfies `risk <= max_risk` or if
    `points` is empty.  Raises ValueError if `max_risk` is negative.
    """
    if max_risk < 0:
        raise ValueError("max_risk must be non-negative")
    best = 0.0
    for point in points:
        if point.risk <= max_risk and point.coverage > best:
            best = point.coverage
    return best


@dataclass(frozen=True)
class TemperatureScaler:
    """A single-scalar temperature-scaling calibrator."""

    temperature: float

    def calibrate(self, raw_confidence: float) -> float:
        """Apply temperature scaling to a raw softmax-like confidence in (0, 1).

        Interprets `raw_confidence` as the max-class probability of a binary
        softmax and rescales it via logit / temperature, then re-applies
        sigmoid, i.e.: `sigmoid(logit(raw_confidence) / temperature)` where
        `logit(p) = ln(p / (1 - p))`.  Clamp `raw_confidence` to the open
        interval (1e-6, 1 - 1e-6) before taking the logit, to avoid
        `inf`/`ZeroDivisionError` for exactly-0 or exactly-1 inputs -- do not
        raise for raw_confidence == 0.0 or == 1.0, clamp instead.  Raises
        ValueError if `raw_confidence` is outside [0, 1] (before clamping) or
        if `self.temperature <= 0`.
        """
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= raw_confidence <= 1.0:
            raise ValueError("raw_confidence must lie within [0, 1]")
        p = min(max(raw_confidence, 1e-6), 1 - 1e-6)
        logit = math.log(p) - math.log(1.0 - p)
        scaled = logit / self.temperature
        if scaled >= 0:
            return 1.0 / (1.0 + math.exp(-scaled))
        z = math.exp(scaled)
        return z / (1.0 + z)


def fit_temperature(
    confidences: list[float],
    correct: list[bool],
    *,
    lo: float = 0.05,
    hi: float = 50.0,
    steps: int = 500,
) -> TemperatureScaler:
    """Fit a single scalar temperature minimizing ECE via grid search over [lo, hi].

    Default ``hi=50``/``steps=500``: a badly overconfident raw signal (e.g. mean
    token log-probability from greedy HF generation) can need a temperature well
    past the naive ``hi=5`` a softmax-confidence prior would suggest -- iteration
    6 (see logs/V2_TRAINING_FAILURES.md) found the ECE-minimizing temperature at
    ~28 for one real checkpoint, with a default of 5 silently landing on the grid
    ceiling instead of the true optimum.

    Grid search (not gradient descent -- keep this dependency-free, no
    scipy/torch) over `steps` evenly spaced temperature values in [lo, hi]
    inclusive; for each candidate temperature, build a `TemperatureScaler
    (candidate)`, apply `.calibrate()` to every confidence, compute
    `expected_calibration_error` on the calibrated confidences against
    `correct`, and return the `TemperatureScaler` for whichever candidate
    temperature achieves the lowest ECE (ties broken by the smallest
    temperature).  Raises ValueError if `confidences`/`correct` are empty or
    mismatched length, or if `lo <= 0`, `hi <= lo`, or `steps < 1`.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have equal lengths")
    if not confidences:
        raise ValueError("confidences must be non-empty")
    if lo <= 0:
        raise ValueError("lo must be positive")
    if hi <= lo:
        raise ValueError("hi must be greater than lo")
    if steps < 1:
        raise ValueError("steps must be >= 1")

    temps = (
        [lo + (hi - lo) * i / (steps - 1) for i in range(steps)]
        if steps > 1
        else [lo]
    )
    candidates: list[tuple[float, float]] = []
    for temp in temps:
        scaler = TemperatureScaler(temp)
        calibrated = [scaler.calibrate(c) for c in confidences]
        candidates.append((expected_calibration_error(calibrated, correct), temp))
    _, best_temp = min(candidates, key=lambda pair: (pair[0], pair[1]))
    return TemperatureScaler(best_temp)
