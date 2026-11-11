"""Alternative position estimators for a reconstructed vertex.

PV-Finder's production position is the local centroid of the output histogram
(``peak_finding_fast``, +-3 bins of 0.04 mm). These estimators are used to
test whether the hard-scatter (HS) resolution is limited by that read-out:

* :func:`parabolic_peak`: three-point interpolation around the maximum bin,
  on the bin heights or on their logarithm (exact for a Gaussian peak);
* :func:`weighted_z0_mean`: inverse-variance mean of the z0 of the tracks
  assigned to the vertex (the GNN assignment in the PV-Finder chain);
* :func:`robust_weighted_z0_mean`: the same, iteratively dropping tracks more
  than ``n_sigma`` standard deviations from the current mean.

Histogram positions follow the peak finder convention: bin ``i`` is centred
at ``Z_MIN + i * BIN_WIDTH``.
"""

from __future__ import annotations

import numpy as np

from pv_finder.utils.constants import Z_MAX, Z_MIN

BIN_WIDTH = (Z_MAX - Z_MIN) / 12000  # 0.04 mm


def parabolic_peak(hist: np.ndarray, peak_bin: int, log: bool = False) -> float:
    """Vertex z [mm] from a parabola through the maximum bin and its neighbours.

    With ``log=True`` the parabola is fitted to ln(height), which is exact for
    a sampled Gaussian. Falls back to the bin centre at the histogram edge, when
    a neighbour is not positive (log only) or when the three points are not
    concave.
    """
    i = int(peak_bin)
    centre = Z_MIN + i * BIN_WIDTH
    if i <= 0 or i >= len(hist) - 1:
        return float(centre)
    y = np.asarray(hist[i - 1 : i + 2], dtype=np.float64)
    if log:
        if np.any(y <= 0.0):
            return float(centre)
        y = np.log(y)
    denom = y[0] - 2.0 * y[1] + y[2]
    if not denom < 0.0:
        return float(centre)
    delta = 0.5 * (y[0] - y[2]) / denom
    return float(centre + np.clip(delta, -0.5, 0.5) * BIN_WIDTH)


def weighted_z0_mean(z0: np.ndarray, z0_err: np.ndarray) -> float:
    """Inverse-variance weighted mean of ``z0``; NaN for no usable track."""
    z0 = np.asarray(z0, dtype=np.float64)
    err = np.asarray(z0_err, dtype=np.float64)
    ok = np.isfinite(z0) & np.isfinite(err) & (err > 0.0)
    if not ok.any():
        return float("nan")
    w = 1.0 / err[ok] ** 2
    return float(np.sum(w * z0[ok]) / np.sum(w))


def robust_weighted_z0_mean(
    z0: np.ndarray,
    z0_err: np.ndarray,
    n_sigma: float = 3.0,
    max_iter: int = 5,
    seed: float | None = None,
) -> float:
    """:func:`weighted_z0_mean` after iteratively removing outlier tracks.

    A track is removed when ``|z0 - mean| > n_sigma * z0_err``. Iteration stops
    when the kept set does not change, after ``max_iter`` passes, or when no
    track would remain (the previous mean is then returned).

    With ``seed`` (e.g. the histogram peak position), the first selection is
    made around the seed instead of around the plain mean, so a few far
    outliers cannot pull the starting point; if no track is compatible with
    the seed, the seed is returned.
    """
    z0 = np.asarray(z0, dtype=np.float64)
    err = np.asarray(z0_err, dtype=np.float64)
    keep = np.isfinite(z0) & np.isfinite(err) & (err > 0.0)
    if seed is not None:
        keep &= np.abs(z0 - seed) <= n_sigma * err
        if not keep.any():
            return float(seed)
    mean = weighted_z0_mean(z0[keep], err[keep])
    for _ in range(max_iter):
        if not np.isfinite(mean):
            break
        new_keep = keep & (np.abs(z0 - mean) <= n_sigma * err)
        if not new_keep.any() or np.array_equal(new_keep, keep):
            break
        keep = new_keep
        mean = weighted_z0_mean(z0[keep], err[keep])
    return mean


ANNEALING = (64.0, 16.0, 4.0, 2.0, 1.5, 1.0)


def adaptive_z_fit(
    z0: np.ndarray,
    z0_err: np.ndarray,
    seed: float,
    chi2_cut: float = 9.0,
    temperatures: tuple[float, ...] = ANNEALING,
    prior: np.ndarray | None = None,
    chi2_extra: np.ndarray | None = None,
    tol: float = 1e-7,
    max_iter: int = 50,
) -> float:
    """Adaptive (annealed) one-dimensional vertex fit of ``z`` [mm].

    AMVF-style track weights ``w = 1 / (1 + exp((chi2 - chi2_cut) / (2T)))``
    with ``chi2 = ((z0 - z) / z0_err)**2``, recomputed at each temperature of
    the deterministic-annealing schedule until ``z`` moves by less than
    ``tol``. ``z`` is the inverse-variance mean with these weights, multiplied
    by ``prior`` (e.g. GNN edge scores) when given. ``chi2_extra`` is a fixed
    per-track chi2 term added before weighting (e.g. the transverse d0 term
    of :func:`beamline_z_measurements`). Starts from ``seed``;
    returns ``seed`` when no track is usable or all weights vanish.
    """
    z0 = np.asarray(z0, dtype=np.float64)
    err = np.asarray(z0_err, dtype=np.float64)
    ok = np.isfinite(z0) & np.isfinite(err) & (err > 0.0)
    if prior is not None:
        pr = np.asarray(prior, dtype=np.float64)
        ok &= np.isfinite(pr) & (pr > 0.0)
        pr = pr[ok]
    else:
        pr = 1.0
    if chi2_extra is not None:
        extra = np.asarray(chi2_extra, dtype=np.float64)
        ok &= np.isfinite(extra)
    if not ok.any():
        return float(seed)
    extra = extra[ok] if chi2_extra is not None else 0.0
    z0, ivar = z0[ok], 1.0 / err[ok] ** 2
    z = float(seed)
    for t in temperatures:
        for _ in range(max_iter):
            chi2 = (z0 - z) ** 2 * ivar + extra
            arg = np.clip((chi2 - chi2_cut) / (2.0 * t), -700.0, 700.0)
            w = pr * ivar / (1.0 + np.exp(arg))
            sw = np.sum(w)
            if not sw > 0.0:
                return float(z)
            z_new = float(np.sum(w * z0) / sw)
            done = abs(z_new - z) < tol
            z = z_new
            if done:
                break
    return z


def beamline_z_measurements(
    d0: np.ndarray,
    z0: np.ndarray,
    var_d0: np.ndarray,
    var_z0: np.ndarray,
    cov_d0z0: np.ndarray,
    beam_sigma_xy: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-track z measurement for a vertex constrained to the beamline.

    The vertex sits on the beam axis (transverse spread ``beam_sigma_xy``),
    so a track's perigee residual is ``(d0, z0 - z)`` with covariance
    ``[[var_d0 + s_b^2, cov], [cov, var_z0]]`` (d0 relative to the beam
    position). Profiling out the d0 component gives an equivalent 1D
    measurement ``z0 - cov / V_dd * d0`` with variance ``var_z0 - cov^2 / V_dd``
    and a fixed chi2 term ``d0^2 / V_dd``, so the 2D chi2 equals
    ``extra + (z_eff - z)^2 / var_eff``. Returns ``(z_eff, err_eff, extra)``.
    """
    v_dd = np.asarray(var_d0, dtype=np.float64) + beam_sigma_xy**2
    cov = np.asarray(cov_d0z0, dtype=np.float64)
    d0 = np.asarray(d0, dtype=np.float64)
    z_eff = np.asarray(z0, dtype=np.float64) - cov / v_dd * d0
    var_eff = np.asarray(var_z0, dtype=np.float64) - cov**2 / v_dd
    return z_eff, np.sqrt(np.clip(var_eff, 1e-12, None)), d0**2 / v_dd


def adaptive_beamline_fit(
    d0: np.ndarray,
    z0: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    var_d0: np.ndarray,
    var_z0: np.ndarray,
    cov_d0z0: np.ndarray,
    seed_z: float,
    beam_xy: tuple[float, float] = (0.0, 0.0),
    beam_sigma_xy: float = 0.012,
    chi2_cut: float = 9.0,
    temperatures: tuple[float, ...] = ANNEALING,
    prior: np.ndarray | None = None,
    tol: float = 1e-7,
    max_iter: int = 50,
) -> tuple[float, float, float]:
    """Adaptive 3D vertex fit (x, y, z) [mm] with a beam-spot constraint.

    Straight-line linearisation of the perigee parameters (defined at the
    point of closest approach to the beam axis): a vertex at (x, y, z)
    predicts ``d0 = -x sin(phi) + y cos(phi)`` and
    ``z0 = z - (x cos(phi) + y sin(phi)) cot(theta)``. Both signs were measured
    on the r16443 ntuple by regressing d0 and z0 - z_true of the true HS tracks
    on the true HS (x, y) (slopes -0.86 and +0.80 against the opposite-sign
    and this model); note that ``RecoTrack_x0/y0`` satisfy
    ``d0 = x0 sin(phi) - y0 cos(phi)``, the opposite sign, so they are not the
    point of closest approach in the usual convention. Each track enters with
    its 2x2 (d0, z0) covariance; the transverse position has a Gaussian prior
    at ``beam_xy`` of width ``beam_sigma_xy``. Track weights follow
    :func:`adaptive_z_fit` (annealed, chi2 from the 2D residual), optionally
    multiplied by ``prior``. Returns ``(x, y, z)``; the seed when no track is
    usable.
    """
    d0, z0, phi, theta = (np.asarray(a, dtype=np.float64) for a in (d0, z0, phi, theta))
    vdd, vzz, vdz = (
        np.asarray(a, dtype=np.float64) for a in (var_d0, var_z0, cov_d0z0)
    )
    det = vdd * vzz - vdz**2
    ok = np.isfinite(d0 + z0 + phi + theta + det) & (det > 0.0) & (vdd > 0.0)
    pr = np.ones_like(d0) if prior is None else np.asarray(prior, dtype=np.float64)
    ok &= np.isfinite(pr) & (pr > 0.0)
    x = np.array([beam_xy[0], beam_xy[1], float(seed_z)])
    if not ok.any():
        return float(x[0]), float(x[1]), float(x[2])
    d0, z0, phi, theta, pr = d0[ok], z0[ok], phi[ok], theta[ok], pr[ok]
    # Inverse covariances, (n, 2, 2).
    vinv = np.empty((len(d0), 2, 2))
    vinv[:, 0, 0] = vzz[ok] / det[ok]
    vinv[:, 1, 1] = vdd[ok] / det[ok]
    vinv[:, 0, 1] = vinv[:, 1, 0] = -vdz[ok] / det[ok]
    cot = 1.0 / np.tan(theta)
    s, c = np.sin(phi), np.cos(phi)
    a = np.zeros((len(d0), 2, 3))
    a[:, 0, 0], a[:, 0, 1] = -s, c  # measured: d0 = -x sin(phi) + y cos(phi)
    a[:, 1, 0], a[:, 1, 1], a[:, 1, 2] = -c * cot, -s * cot, 1.0
    m = np.stack([d0, z0], axis=1)
    at_vinv = np.einsum("nij,nik->njk", a, vinv)  # (n, 3, 2)
    info_trk = np.einsum("njk,nkl->njl", at_vinv, a)  # (n, 3, 3)
    rhs_trk = np.einsum("njk,nk->nj", at_vinv, m)  # (n, 3)
    p_info = np.diag([beam_sigma_xy**-2, beam_sigma_xy**-2, 0.0])
    p_rhs = p_info @ np.array([beam_xy[0], beam_xy[1], 0.0])
    for t in temperatures:
        for _ in range(max_iter):
            r = m - np.einsum("nij,j->ni", a, x)
            chi2 = np.einsum("ni,nij,nj->n", r, vinv, r)
            arg = np.clip((chi2 - chi2_cut) / (2.0 * t), -700.0, 700.0)
            w = pr / (1.0 + np.exp(arg))
            info = p_info + np.einsum("n,njl->jl", w, info_trk)
            if not np.sum(w) > 0.0 or abs(np.linalg.det(info)) < 1e-300:
                return float(x[0]), float(x[1]), float(x[2])
            x_new = np.linalg.solve(info, p_rhs + np.einsum("n,nj->j", w, rhs_trk))
            done = abs(x_new[2] - x[2]) < tol
            x = x_new
            if done:
                break
    return float(x[0]), float(x[1]), float(x[2])
