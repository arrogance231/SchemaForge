"""SchemaForge V2 confidence-calibration package.

Provides reliability-diagram binning and expected calibration error
(:mod:`calibration`), risk-coverage analysis, and a dependency-free temperature
scaling calibrator for research direction §5.
"""

from schemaforge.calibration.calibration import (
    ReliabilityBin,
    RiskCoveragePoint,
    TemperatureScaler,
    coverage_at_risk,
    expected_calibration_error,
    fit_temperature,
    reliability_bins,
    risk_coverage_curve,
)

__all__ = [
    "ReliabilityBin",
    "reliability_bins",
    "expected_calibration_error",
    "RiskCoveragePoint",
    "risk_coverage_curve",
    "coverage_at_risk",
    "TemperatureScaler",
    "fit_temperature",
]
