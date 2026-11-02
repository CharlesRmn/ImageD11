"""Sample-space grain supports from Chan-Vese level-set segmentation."""

from __future__ import print_function

import warnings
import threading

import numpy as np
import pytest
from scipy.ndimage import binary_dilation, gaussian_filter, label

from ImageD11.sinograms import sinogram as sino


@pytest.mark.parametrize("level", [.12, .8, .95])
def test_contour_recovers_boundary_beyond_initial_threshold(level):
    rows, cols = np.indices((101, 101))
    truth = (rows - 50) ** 2 + (cols - 50) ** 2 <= 18 ** 2
    image = gaussian_filter(truth.astype(float), 1)
    before = image.copy()
    mask = sino._refine_connected_grain_mask(
        image, [0., 0.], 1., level=level, dilation=0)
    smooth = gaussian_filter(image, 1)
    seed = smooth >= level * smooth.max()
    seed_error = np.count_nonzero(seed != truth)
    assert np.count_nonzero(mask != truth) < .1 * seed_error
    assert np.sum(mask & truth) / float(np.sum(mask | truth)) > .98
    np.testing.assert_array_equal(image, before)


def test_contour_keeps_centre_component_and_dilates_without_circular_clipping():
    rows, cols = np.indices((101, 101))
    image = (((rows - 50) ** 2 + (cols - 50) ** 2 < 8 ** 2).astype(float)
             + 2. * ((rows - 50) ** 2 + (cols - 74) ** 2 < 7 ** 2))
    mask = sino._refine_connected_grain_mask(
        image, [0., 0.], 1., dilation=0)
    dilated = sino._refine_connected_grain_mask(
        image, [0., 0.], 1., dilation=30)
    assert mask[50, 50] and not mask[50, 74]
    assert label(mask)[1] == 1
    np.testing.assert_array_equal(
        dilated, binary_dilation(mask, iterations=30))


def test_contour_split_retains_both_seed_lobes_and_excludes_other_regions(monkeypatch):
    rows, cols = np.indices((101, 101))
    lower = (rows - 30) ** 2 + (cols - 50) ** 2 <= 12 ** 2
    upper = (rows - 70) ** 2 + (cols - 50) ** 2 <= 12 ** 2
    neck = (rows >= 30) & (rows <= 70) & (np.abs(cols - 50) <= 2)
    distant = (rows - 50) ** 2 + (cols - 85) ** 2 <= 8 ** 2
    image = (lower | upper | neck | distant).astype(float)

    def split_contour(density, num_iter, init_level_set, **kwargs):
        # Model curvature smoothing breaking the neck while a separate
        # bright region is also present in the evolved contour.
        assert init_level_set[31, 51] and init_level_set[71, 51]
        assert not init_level_set[51, 86]
        return np.pad(lower | upper | distant, 1).astype(np.int8)

    monkeypatch.setattr('skimage.segmentation.morphological_chan_vese', split_contour)
    mask = sino._refine_connected_grain_mask(
        image, [-20., 0.], 1., level=.2, dilation=0)
    assert mask[30, 50] and mask[70, 50]
    assert not mask[50, 85]
    assert label(mask)[1] == 2


def test_previous_support_keeps_lobes_when_threshold_seed_disconnects():
    rows, cols = np.indices((101, 101))
    lower = (rows - 30) ** 2 + (cols - 50) ** 2 <= 12 ** 2
    upper = (rows - 70) ** 2 + (cols - 50) ** 2 <= 12 ** 2
    distant = (rows - 50) ** 2 + (cols - 85) ** 2 <= 8 ** 2
    image = (lower | upper | distant).astype(float)
    mask = sino._refine_connected_grain_mask(
        image, [-20., 0.], 1., level=.2, dilation=0,
        reference_mask=lower | upper)
    assert mask[30, 50] and mask[70, 50]
    assert not mask[50, 85]
    assert label(mask)[1] == 2


def test_contour_recovers_elongated_grain():
    rows, cols = np.indices((101, 101))
    truth = ((rows - 50.) / 12.) ** 2 + ((cols - 50.) / 30.) ** 2 <= 1
    image = gaussian_filter(truth.astype(float), 1)
    mask = sino._refine_connected_grain_mask(
        image, [0., 0.], 1., dilation=0)
    assert mask[50, 77]
    assert np.sum(mask & truth) / float(np.sum(mask | truth)) > .95
    expected = sino._refine_connected_grain_mask(
        image, [0., 0.], 1., dilation=0,
        smoothed_image=gaussian_filter(image, 1))
    np.testing.assert_array_equal(mask, expected)


@pytest.mark.parametrize("centre", [[0., 0.], [-50., 50.]])
def test_empty_core_is_rejected(centre):
    with pytest.raises(RuntimeError, match="No reconstructed core"):
        sino._refine_connected_grain_mask(np.zeros((101, 101)), centre, 1.)


def test_boundary_grain_has_a_background_border_and_retains_coordinates():
    rows, cols = np.indices((101, 101))
    image = ((rows - 3) ** 2 + (cols - 3) ** 2 <= 12 ** 2).astype(float)
    centre = [-47., 47.]
    mask = sino._refine_connected_grain_mask(image, centre, 1., dilation=0)
    assert mask.shape == image.shape and mask.dtype == np.bool_
    assert mask[3, 3] and not mask[50, 50]


@pytest.mark.parametrize("workers", [None, 1, 2, 4])
@pytest.mark.parametrize("iterations", [0, 3])
def test_feedback_conserves_budget_and_matches_serial_execution(monkeypatch, workers, iterations):
    # Gaussian spatial phantoms have Gaussian line projections. Distinct
    # integer g-vectors here identify the synthetic reflection at each angle.
    angles = np.arange(0., 360., 10.)
    grid = np.arange(-.6, .60001, .01)
    centres = np.array([[-.16, -.08], [.18, .12]])
    positions = sino._refine_predict(angles, centres, .03)
    signals = np.exp(-.5 * ((grid[:, None, None] - positions[None]) / .025) ** 2)
    intensity = signals.sum(axis=2)
    yi, ai = np.nonzero(intensity > .001)
    weights = signals[yi, ai]
    weights = weights / weights.sum(axis=1)[:, None]
    budget = np.full(len(yi), .8)
    cf = sino.ImageD11.columnfile.newcolumnfile([])
    cf.nrows = len(yi)
    columns = dict(omega=angles[ai], dty=grid[yi], gx=ai + 1.,
                   gy=np.ones(len(yi)), gz=np.ones(len(yi)),
                   eta=np.full(len(yi), 30.), sum_intensity=intensity[yi, ai],
                   spot3d_id=ai, source_2d_row=np.arange(len(yi)))
    for name, values in columns.items():
        cf.addcolumn(values, name)
    ds = sino.ImageD11.sinograms.dataset.DataSet()
    ds.ystep, ds.ybincens = .01, grid
    ds._pk2d = columns
    grains = [sino.ImageD11.grain.grain(np.eye(3), translation=np.r_[c, 0.])
              for c in centres]
    options = dict(cf_2d=cf, initial_weights=weights * budget[:, None],
                   centres=centres, y0=.03,
                   intensity_budget=budget, iterations=iterations, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, serial = sino.refine_grain_sinograms(ds, grains, workers=1, **options)
        monkeypatch.setattr(sino.ImageD11.cImageD11, "cores_available", lambda: 8)
        expected_workers = min(8 if workers is None else workers, len(grains))
        tables, threads = [], set()
        barrier = threading.Barrier(2) if expected_workers > 1 else None
        rebuild = sino._refine_rebuild_sinogram

        def checked_rebuild(gs, table, *args, **kwargs):
            tables.append(table)
            threads.add(threading.current_thread().ident)
            if barrier is not None:
                barrier.wait(timeout=30)
            return rebuild(gs, table, *args, **kwargs)

        monkeypatch.setattr(sino, "_refine_rebuild_sinogram", checked_rebuild)
        worker_options = {} if workers is None else dict(workers=workers)
        updated, result = sino.refine_grain_sinograms(ds, grains, **dict(options, **worker_options))
        assert all(table is tables[0] for table in tables)
        assert len(threads) == expected_workers
        np.testing.assert_array_equal(result.peak_weights, serial.peak_weights)
        for actual, expected in zip(result.grainsinos, serial.grainsinos):
            np.testing.assert_array_equal(actual.spatial_split_mask, expected.spatial_split_mask)
            np.testing.assert_array_equal(actual.recons["iradon"], expected.recons["iradon"])
    for name, values in columns.items():
        np.testing.assert_array_equal(cf[name], values)
    np.testing.assert_array_equal(options["initial_weights"], weights * budget[:, None])
    active = result.peak_weights.sum(axis=1) > 0
    assert active.any()
    assert len(result.history) == len(serial.history)
    assert bool(result.history) == bool(iterations)
    np.testing.assert_allclose(result.peak_weights[active].sum(axis=1), .8)
    assert np.all(result.peak_weights >= 0)
    for k, gs in enumerate(result.grainsinos):
        np.testing.assert_array_equal(updated[k].ubi, grains[k].ubi)


@pytest.mark.parametrize("workers", [0, -1, 1.5, True, np.bool_(True)])
def test_refinement_rejects_invalid_workers(workers):
    with pytest.raises(ValueError, match="workers must be a positive integer"):
        sino.refine_grain_sinograms(None, [], workers=workers)
