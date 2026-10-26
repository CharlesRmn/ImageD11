"""Hough trajectory seeds, angular balance and public decoupling integration."""

from __future__ import print_function

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from ImageD11.sinograms import sinogram as sino


def observations(count, sigma=.08):
    rng = np.random.RandomState(17)
    centres = np.array([[-20., -10.], [15., -15.], [0., 25.], [30., 20.]])[:count]
    angles = [rng.uniform(0, 360, n) for n in (300, 180, 100, 60)[:count]]
    omega = np.concatenate(angles)
    dty = np.concatenate([sino._refine_predict(a, [c], 2.5).ravel()
                          for a, c in zip(angles, centres)])
    dty += rng.normal(0, sigma, len(dty))
    omega[::3] += 720
    order = rng.permutation(len(omega))
    centres = centres[np.lexsort((centres[:, 1], centres[:, 0]))]
    return omega[order], dty[order], centres


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_hough_recovers_unequal_branches_and_automatic_count(count):
    omega, dty, expected = observations(count)
    fitted, y0, labels, diagnostic = sino._decouple_auto_components(
        omega, dty, .5, .2, fixed_y0=2.5, hough=True)
    assert diagnostic["selected"] == count
    assert diagnostic["ransac_confidence"] is None
    assert y0 == 2.5
    np.testing.assert_allclose(fitted, expected, atol=.06)
    assert set(labels[labels >= 0]) == set(range(count))


def test_hough_is_independent_of_arrival_order_seed_and_trial_budget(monkeypatch):
    omega, dty, _ = observations(3)

    def no_ransac(*args, **kwargs):
        raise AssertionError("Hough initialization called RANSAC")

    monkeypatch.setattr(sino, "_decouple_ransac_one", no_ransac)
    monkeypatch.setattr(sino, "_decouple_ransac_one_manual", no_ransac)
    first = sino._decouple_fit_components(
        omega, dty, .5, .2, 2.5, trials=0, seed=0, n_components=3, hough=True)
    second = sino._decouple_fit_components(
        omega[::-1], dty[::-1], .5, .2, 2.5, trials=1, seed=91,
        n_components=3, hough=True)
    np.testing.assert_allclose(first[0], second[0], atol=1e-7)
    np.testing.assert_array_equal(first[2], second[2][::-1])


def test_sector_balance_rejects_dense_local_distractor():
    omega = np.linspace(0, 359, 240)
    fragment = np.linspace(65, 70, 2000)
    dty = np.r_[sino._refine_predict(omega, [[-20., -10.]], 2.5).ravel(),
                sino._refine_predict(fragment, [[25., 20.]], 2.5).ravel()]
    all_omega = np.r_[omega, fragment]
    centre, _ = sino._decouple_hough_one(all_omega, dty, .5, 2.5, 12)
    np.testing.assert_allclose(centre, [-20., -10.], atol=.05)
    _, _, _, diagnostic = sino._decouple_auto_components(
        all_omega, dty, .5, .2, 2.5, hough=True)
    assert diagnostic["selected"] == 1


@pytest.mark.parametrize("sigma", [.3, .8])
def test_hough_does_not_turn_broad_branches_into_extra_grains(sigma):
    rng = np.random.RandomState(19)
    angles = rng.uniform(0, 360, 600)
    centres = np.array([[-20., -10.], [15., -15.]])
    omega = np.tile(angles, 2)
    dty = np.concatenate([sino._refine_predict(angles, [c], 2.5).ravel() for c in centres])
    dty += rng.normal(0, sigma, len(dty))
    _, _, _, diagnostic = sino._decouple_auto_components(
        omega, dty, .5, .2, 2.5, hough=True)
    assert diagnostic["selected"] == 2


def test_half_turn_crossing_zero_and_cosine_singularities():
    omega = np.tile(337.5 + 15 * np.arange(12), 4)
    omega = np.r_[omega, [90., 270., 450., -90.]]
    dty = sino._refine_predict(omega, [[-20., -10.]], 2.5).ravel()
    centre, _ = sino._decouple_hough_one(omega, dty, .5, 2.5, 12)
    np.testing.assert_allclose(centre, [-20., -10.], atol=1e-7)


@pytest.mark.parametrize("axis", [None, np.nan, np.inf])
def test_hough_requires_calibrated_axis_before_reading_peaks(axis):
    with pytest.raises(ValueError, match="calibrated y0"):
        sino.split_grain_sinogram(None, None, None, y0=axis, initialization="hough")


def test_hough_rejects_parallel_measurement_lines():
    with pytest.raises(RuntimeError, match="distinct projection directions"):
        sino._decouple_hough_one(np.zeros(20), np.arange(20.), .5, 0.)


@pytest.mark.parametrize("width", [0., -1., np.nan, np.inf])
def test_hough_rejects_invalid_voting_width(width):
    with pytest.raises(ValueError, match="positive and finite"):
        sino._decouple_hough_one(np.arange(20.), np.zeros(20), width, 0.)


def test_hough_retains_branches_with_background_outliers():
    omega, dty, expected = observations(2)
    rng = np.random.RandomState(41)
    omega = np.r_[omega, rng.uniform(0, 360, 80)]
    dty = np.r_[dty, rng.uniform(-45, 45, 80)]
    fitted, _, labels = sino._decouple_fit_components(
        omega, dty, .5, .2, 2.5, hough=True)
    np.testing.assert_allclose(fitted, expected, atol=.06)
    assert np.all(np.bincount(labels[labels >= 0]) >= 170)


@pytest.mark.parametrize("automatic", [False, True])
def test_public_hough_split_runs_actual_feedback(automatic, monkeypatch):
    angles = np.arange(0., 360., 6.)
    grid = np.arange(-.6, .60001, .01)
    centres = np.array([[-.16, -.08], [.18, .12]])
    positions = sino._refine_predict(angles, centres, .03)
    signals = np.exp(-.5 * ((grid[:, None, None] - positions[None]) / .025) ** 2)
    intensity = signals.sum(axis=2)
    yi, ai = np.nonzero(intensity > .001)
    cf = sino.ImageD11.columnfile.newcolumnfile([])
    cf.nrows = len(yi)
    columns = dict(omega=angles[ai], dty=grid[yi], gx=ai + 1.,
                   gy=np.ones(len(yi)), gz=np.ones(len(yi)),
                   eta=np.full(len(yi), 30.), sum_intensity=intensity[yi, ai],
                   spot3d_id=np.arange(len(yi)), source_2d_row=np.arange(len(yi)))
    for name, values in columns.items():
        cf.addcolumn(values, name)
    ds = sino.ImageD11.sinograms.dataset.DataSet()
    ds.ystep, ds.ybincens, ds.y0, ds._pk2d = .01, grid, .03, columns
    grain = sino.ImageD11.grain.grain(np.eye(3))
    monkeypatch.setattr(sino, "_refine_colfile_from_rows", lambda *args: cf.copy())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = sino.split_grain_sinogram(
            ds, grain, cf, initialization="hough", n_components=None if automatic else 2,
            width=.035, margin=.01, width_2d=.08, roi_radius=.12,
            iterations=2, verbose=False)
    assert result.stats["initialization"] == "hough"
    assert len(result.grainsinos) == 2 and result.history
    np.testing.assert_allclose(result.centres, centres, atol=.02)
    totals = result.peak_weights.sum(axis=1)
    np.testing.assert_allclose(totals[totals > 0], 1.)
    for gs in result.grainsinos:
        assert gs.spatial_split_mask.any()
        np.testing.assert_array_equal(gs.grain.ubi, grain.ubi)


def test_batch_hough_uses_dataset_axis_and_preserves_single_parent(monkeypatch):
    omega, dty, _ = observations(1)
    cf = SimpleNamespace(titles=[], nrows=len(omega), omega=omega, dty=dty)
    grain = SimpleNamespace(ubi=np.eye(3))
    ds = SimpleNamespace(ystep=.25, y0=2.5)
    monkeypatch.setattr(sino, "_refine_peak_owners", lambda *args: np.zeros(len(omega)))
    monkeypatch.setattr(sino, "_decouple_eligible_4d",
                        lambda *args: (np.arange(len(omega)), None))
    report = []
    updated, results = sino.decouple_grain_sinograms(
        ds, [grain], cf, grain_ids=None, initialization="hough", report=report, verbose=False)
    assert updated[0] is grain and not results
    assert report[0]["status"] == "unchanged"
    assert report[0]["n_components"] == 1
