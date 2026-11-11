"""gbt_peak_filter.filter_peaks keeps exactly the peaks the model accepts."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1] / "src/pv_finder/evaluation/vertex_finding"
    ),
)
from gbt_peak_filter import compute_hist_features, filter_peaks  # noqa: E402


class _HeightModel:
    """Stand-in classifier: probability = feature 0 (peak height), clipped."""

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = np.clip(x[:, 0], 0.0, 1.0)
        return np.stack([1 - p, p], axis=1)


def test_filter_peaks_threshold() -> None:
    hist = np.zeros(12000, dtype=np.float32)
    z = np.array([-10.0, 0.0, 10.0])
    h = np.array([0.2, 0.8, 0.5])
    kept = filter_peaks(z, h, hist, _HeightModel(), 0.5)
    np.testing.assert_array_equal(kept, np.array([0.0, 10.0]))
    assert len(filter_peaks(z[:0], h[:0], hist, _HeightModel(), 0.5)) == 0


def test_features_shape_and_neighbour() -> None:
    hist = np.zeros(12000, dtype=np.float32)
    f = compute_hist_features(
        0.0, 1.0, hist, np.array([0.0, 1.0]), np.array([1.0, 0.5])
    )
    assert f.shape == (8,)
    assert abs(f[6] - 1.0) < 1e-6  # nearest other peak 1 mm away
    assert abs(f[7] - 2.0) < 1e-6  # height ratio 1.0 / 0.5
