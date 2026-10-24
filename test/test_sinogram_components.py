"""Spatial component detection, explicit overrides and child ID preservation."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from ImageD11.sinograms import sinogram as sino


@pytest.mark.parametrize('count', [2, 3, 4])
@pytest.mark.parametrize('fixed_y0', [None, 2.5])
def test_fit_manual_components(count, fixed_y0):
    centres = np.array([[-20., -10.], [15., -15.], [0., 25.], [30., 20.]])[:count]
    angles = np.linspace(0, 359, 240)
    omega = np.tile(angles, count)
    dty = np.concatenate([
        sino._refine_predict(angles, [centre], 2.5).ravel() for centre in centres
    ])
    dty += np.random.RandomState(11).normal(0, .08, len(dty))
    fitted, y0, labels = sino._decouple_fit_components(
        omega, dty, .5, .2, fixed_y0=fixed_y0, trials=150,
        n_components=count,
    )
    order = np.lexsort((centres[:, 1], centres[:, 0]))
    np.testing.assert_allclose(fitted, centres[order], atol=.04)
    assert abs(y0 - 2.5) < .04
    assert set(labels[labels >= 0]) == set(range(count))
    seeds, _, _, widths = sino._decouple_select_full_branches(
        omega, dty, fitted, y0, .1, .5, .2, manual_width=.5,
    )
    assert widths.shape == (36, count, 2)
    assert all(np.sum(seeds == k) >= 200 for k in range(count))


@pytest.mark.parametrize('selection, expected', [
    ({0: 3, 2: 4}, [(0, 3, 4), (2, 5, 6, 7)]),
    ([0, 2], [(0, 3), (2, 4)]),
    ([0, 2], [(0,), (2, 3, 4)]),
])
def test_component_child_ids(selection, expected):
    grains = [SimpleNamespace() for _ in range(3)]
    table = SimpleNamespace(titles=[])

    def split(*args, **kwargs):
        count = kwargs['n_components']
        if isinstance(selection, list):
            assert count is None
            parent = next(i for i, g in enumerate(grains) if g is args[1])
            count = len(next(ids for ids in expected if ids[0] == parent))
        return SimpleNamespace(
            grainsinos=[SimpleNamespace(grain=SimpleNamespace()) for _ in range(count)],
            source_2d_rows=np.empty(0, dtype=int), peak_weights=np.empty((0, count)),
        )

    with patch.object(sino, '_refine_peak_owners', return_value=np.array([0, 2])), \
            patch.object(sino, 'split_grain_sinogram', side_effect=split):
        updated, results = sino.decouple_grain_sinograms(
            None, grains, table, grain_ids=selection,
        )
    assert [result.child_ids for result in results] == expected
    assert updated[1] is grains[1]
    for result in results:
        for gs, gid in zip(result.grainsinos, result.child_ids):
            assert updated[gid] is gs.grain
            assert gs.grain.gid == gid


@pytest.mark.parametrize('count', [1, 0, -1, 2.5, True])
def test_invalid_component_counts(count):
    with pytest.raises(ValueError, match='Component counts'):
        sino.decouple_grain_sinograms(
            None, [None], SimpleNamespace(titles=[]), grain_ids={0: count},
        )


def test_saved_assignment_prepares_hkl_for_sinogram():
    cf = sino.ImageD11.columnfile.newcolumnfile(
        ['gx', 'gy', 'gz', 'sum_intensity', 'source_2d_row'])
    cf.set_bigarray(np.array([
        [1., 2., 3.4], [0., 1., 0.], [0., 0., 1.],
        [10., 20., 30.], [4., 5., 6.],
    ]))
    gs = sino.GrainSinogram.__new__(sino.GrainSinogram)
    gs.ds = SimpleNamespace()
    gs.grain = SimpleNamespace(ubi=np.eye(3))
    with patch.object(sino, '_saved_split_assignment',
                      return_value=(np.array([4, 5, 6]), np.array([.5, .25, 1.]))), \
            patch.object(sino, '_refine_colfile_from_rows', return_value=cf):
        gs.prepare_peaks_from_4d(
            SimpleNamespace(parameters=None), None, None, 0, None,
            phase_name='alpha')
    assert all(name in gs.cf_for_sino.titles for name in 'hkl')
    np.testing.assert_array_equal(gs.cf_for_sino.h, [1, 2])
    np.testing.assert_array_equal(gs.cf_for_sino.k, [0, 1])
    np.testing.assert_array_equal(gs.cf_for_sino.l, [0, 0])
    np.testing.assert_allclose(gs.cf_for_sino.sum_intensity, [5., 5.])
    np.testing.assert_array_equal(gs.source_2d_rows, [4, 5])
    np.testing.assert_allclose(gs.peak_fractions, [.5, .25])


@pytest.mark.parametrize('count', [1, 2, 3, 4])
@pytest.mark.parametrize('fixed_y0', [None, 2.5])
def test_automatic_components(count, fixed_y0):
    rng = np.random.RandomState(17)
    centres = np.array([[-20., -10.], [15., -15.], [0., 25.], [30., 20.]])[:count]
    angles = np.sort(rng.uniform(0, 360, 180))
    omega = np.tile(angles, count)
    dty = np.concatenate([
        sino._refine_predict(angles, [centre], 2.5).ravel() for centre in centres
    ]) + rng.normal(0, .08, len(omega))
    # Irregular samples, shuffled arrival order, repeated turns and background peaks.
    background = 30
    omega = np.r_[omega, rng.uniform(0, 360, background)]
    dty = np.r_[dty, rng.uniform(-45, 45, background)]
    omega[::3] += 720
    order = rng.permutation(len(omega))
    fitted, y0, labels, diagnostics = sino._decouple_auto_components(
        omega[order], dty[order], .5, .2, fixed_y0=fixed_y0,
        trials=80, max_components=6)
    expected_order = np.lexsort((centres[:, 1], centres[:, 0]))
    assert diagnostics['selected'] == count
    assert not diagnostics['reached_limit']
    np.testing.assert_allclose(fitted, centres[expected_order], atol=.05)
    assert abs(y0 - 2.5) < .05
    assert set(labels[labels >= 0]) == set(range(count))


def test_automatic_count_rejects_local_distractor():
    rng = np.random.RandomState(5)
    omega = np.r_[rng.uniform(0, 360, 240), rng.uniform(65, 70, 90)]
    dty = np.r_[sino._refine_predict(omega[:240], [[-20., -10.]], 2.5).ravel(),
                sino._refine_predict(omega[240:], [[25., 20.]], 2.5).ravel()]
    dty += rng.normal(0, .08, len(dty))
    _, _, _, diagnostics = sino._decouple_auto_components(
        omega, dty, .5, .2, trials=80, max_components=4)
    assert diagnostics['selected'] == 1


def test_automatic_count_requires_angular_coverage():
    omega = np.linspace(2, 20, 100)
    dty = sino._refine_predict(omega, [[1., 2.]], 0).ravel()
    with pytest.raises(ValueError, match='four occupied'):
        sino._decouple_auto_components(omega, dty, .5, .2)


def test_automatic_single_branch_envelope():
    rng = np.random.RandomState(8)
    omega = rng.uniform(0, 360, 2400)
    dty = sino._refine_predict(omega, [[-20., -10.]], 2.5).ravel()
    dty += rng.normal(0, .15, len(omega))
    labels, _, _, widths = sino._decouple_select_full_branches(
        omega, dty, np.array([[-20., -10.]]), 2.5, .1, .5, .2)
    assert widths.shape == (36, 1, 2)
    assert np.isfinite(widths).all()
    assert np.sum(labels == 0) > 2000


@pytest.mark.parametrize('count', [1, 3])
@pytest.mark.parametrize('reuse_fit', [False, True])
def test_detected_count_reaches_refinement(count, reuse_fit):
    omega = np.linspace(0, 359, 120)
    ds = SimpleNamespace(ystep=1., ybincens=np.arange(-100., 101.),
                         pk2d={'spot3d_id': np.arange(120)})
    grain = SimpleNamespace(ubi=np.eye(3), name='parent')
    cf4 = SimpleNamespace(nrows=120, omega=omega, dty=np.zeros(120),
                          spot3d_id=np.arange(120), parameters=None)
    cf2 = SimpleNamespace(nrows=120, omega=omega, dty=np.zeros(120),
                          eta=np.ones(120), sum_intensity=np.ones(120))
    cf2.filter = lambda mask: None  # All fixture peaks pass validity checks.
    centres = np.column_stack((np.arange(count) * 10., np.zeros(count)))
    labels = np.arange(120) % count
    diagnostic = dict(selected=count, counts=[count], scores=[.1],
                      ambiguous=False, reached_limit=False)
    refined = SimpleNamespace(
        grainsinos=[SimpleNamespace() for _ in range(count)], centres=centres,
        source_2d_rows=np.arange(120), peak_weights=np.zeros((120, count)),
        history=[], converged=True, stats={})
    with patch.object(sino, '_decouple_auto_components',
                      return_value=(centres, 0., labels, diagnostic)) as detect, \
            patch.object(sino, '_refine_errors',
                         side_effect=lambda grains, cf: np.zeros((cf.nrows, len(grains)))), \
            patch.object(sino, '_refine_colfile_from_rows', return_value=cf2), \
            patch.object(sino, '_decouple_select_full_branches',
                         return_value=(labels, None, None, np.ones((36, count, 2)))), \
            patch.object(sino, 'refine_grain_sinograms', return_value=([], refined)) as refine:
        result = sino.split_grain_sinogram(
            ds, grain, cf4, verbose=False,
            _initial_fit=(centres, 0., labels, diagnostic) if reuse_fit else None)
    assert detect.call_count == (0 if reuse_fit else 1)
    assert len(refine.call_args[0][1]) == count
    assert refine.call_args[1]['compatibility'].shape == (120, count)
    assert refine.call_args[1]['initial_weights'].shape == (120, count)
    np.testing.assert_allclose(refine.call_args[1]['initial_weights'].sum(axis=1), 1.)
    assert result.stats['component_selection'] == diagnostic
    assert len(result.grainsinos) == count


def test_automatic_count_without_supported_track_is_informative():
    omega = np.tile(np.arange(0, 360, 30), 3)
    with patch.object(sino, '_decouple_fit_components', return_value=(
            np.zeros((1, 2)), 0., np.full(len(omega), -1))):
        with pytest.raises(RuntimeError, match='could not validate one'):
            sino._decouple_auto_components(omega, np.zeros(len(omega)), .5, .2)


def test_automatic_count_reports_search_limit():
    omega = np.tile(np.linspace(0, 359, 150), 2)
    dty = np.concatenate([
        sino._refine_predict(omega[:150], [centre], 2.5).ravel()
        for centre in [[-20., -10.], [15., -15.]]])
    _, _, _, diagnostics = sino._decouple_auto_components(
        omega, dty, .5, .2, trials=60, max_components=2)
    assert diagnostics['selected'] == 2
    assert diagnostics['reached_limit']


@pytest.mark.parametrize('limit', [0, -1, 2.5, True])
def test_automatic_count_validates_limit(limit):
    with pytest.raises(ValueError, match='max_components'):
        sino._decouple_auto_components(np.arange(100), np.arange(100),
                                       .5, .2, max_components=limit)


def test_batch_screens_all_and_only_splits_supported_parents():
    grains = [SimpleNamespace(gid=i) for i in range(7)]
    table = SimpleNamespace(titles=[], nrows=140, omega=np.arange(140.),
                            dty=np.zeros(140))
    owners = np.repeat(np.arange(7), 20)
    report = []

    def detect(omega, *args):
        gid = int(omega[0] // 20)
        if gid == 4:
            raise ValueError('insufficient angles')
        count = 1 if gid == 0 else 2
        return (np.zeros((count, 2)), 0., np.zeros(len(omega), dtype=int),
                dict(selected=count, ambiguous=gid == 2, reached_limit=gid == 3))

    def split(ds, grain, cf, **kwargs):
        if grain.gid == 5:
            raise RuntimeError('not enough resolved 2D peaks')
        assert grain.gid in (1, 6)
        assert kwargs['n_components'] is None
        assert len(kwargs['_initial_fit'][0]) == 2
        np.testing.assert_array_equal(kwargs['peak_mask'], owners == grain.gid)
        return SimpleNamespace(
            grainsinos=[SimpleNamespace(grain=SimpleNamespace()) for _ in range(2)],
            source_2d_rows=np.empty(0, dtype=int), peak_weights=np.empty((0, 2)))

    with patch.object(sino, '_refine_peak_owners', return_value=owners), \
            patch.object(sino, '_refine_errors', return_value=np.zeros((140, 1))), \
            patch.object(sino, '_decouple_auto_components', side_effect=detect) as detector, \
            patch.object(sino, 'split_grain_sinogram', side_effect=split) as splitter:
        updated, results = sino.decouple_grain_sinograms(
            SimpleNamespace(ystep=1.), grains, table, grain_ids=None,
            report=report, verbose=False)
    assert detector.call_count == 7
    assert splitter.call_count == 3
    assert [r['status'] for r in report] == [
        'unchanged', 'split', 'review', 'review', 'failed', 'failed', 'split']
    assert 'insufficient angles' in report[4]['error']
    assert 'resolved 2D peaks' in report[5]['error']
    assert [r.child_ids for r in results] == [(1, 7), (6, 8)]
    for gid in (0, 2, 3, 4, 5):
        assert updated[gid] is grains[gid]
    assert len(updated) == 9
    assert [g.gid for g in grains] == list(range(7))


def test_batch_excludes_known_twins_from_spatial_screening():
    grains = [SimpleNamespace(gid=i) for i in range(3)]
    cf = SimpleNamespace(titles=[], nrows=60, omega=np.arange(60.), dty=np.zeros(60))
    report = []
    twin_result = SimpleNamespace(
        grainsinos=[SimpleNamespace(grain=SimpleNamespace()) for _ in range(2)],
        source_2d_rows=np.empty(0, dtype=int), peak_weights=np.empty((0, 2)))
    with patch.object(sino, '_refine_peak_owners', return_value=np.repeat(np.arange(3), 20)), \
            patch.object(sino, '_refine_errors', return_value=np.zeros((60, 1))), \
            patch.object(sino, '_decouple_auto_components', return_value=(
                np.zeros((1, 2)), 0., np.zeros(20, dtype=int),
                dict(ambiguous=False, reached_limit=False))) as detector, \
            patch.object(sino, 'split_grain_sinogram', return_value=twin_result) as splitter:
        updated, results = sino.decouple_grain_sinograms(
            SimpleNamespace(ystep=1.), grains, cf, grain_ids=None,
            twin_pairs=[(1, 2)], report=report, verbose=False)
    assert detector.call_count == 1
    assert splitter.call_count == 1
    assert splitter.call_args[1]['twin_grain'] is grains[2]
    assert splitter.call_args[1]['n_components'] == 2
    assert '_initial_fit' not in splitter.call_args[1]
    assert results[0].child_ids == (1, 2)
    assert updated[0] is grains[0]
    assert len(updated) == 3
    assert [row['status'] for row in report] == ['unchanged', 'split']


def test_explicit_selection_remains_fail_fast():
    with patch.object(sino, '_refine_peak_owners', return_value=np.array([0])), \
            patch.object(sino, 'split_grain_sinogram', side_effect=RuntimeError('bad peaks')):
        with pytest.raises(RuntimeError, match='Decoupling parent IDs'):
            sino.decouple_grain_sinograms(None, [None], SimpleNamespace(titles=[]), grain_ids=[0])


def test_batch_no_splits_preserves_strong_table_return():
    grain = SimpleNamespace()
    cf = SimpleNamespace(titles=[], nrows=20, omega=np.arange(20.), dty=np.zeros(20))
    strong = SimpleNamespace(titles=[])
    with patch.object(sino, '_refine_peak_owners', return_value=np.zeros(20, dtype=int)), \
            patch.object(sino, '_refine_errors', return_value=np.zeros((20, 1))), \
            patch.object(sino, '_decouple_auto_components', return_value=(
                np.zeros((1, 2)), 0., np.zeros(20, dtype=int),
                dict(ambiguous=False, reached_limit=False))), \
            patch.object(sino, 'split_grain_sinogram') as splitter, \
            patch.object(sino, 'decoupled_strong_columnfile', return_value=strong) as remerge:
        updated, results, plot_cf = sino.decouple_grain_sinograms(
            SimpleNamespace(ystep=1.), [grain], cf, grain_ids=None,
            cf_strong=strong, verbose=False)
    assert updated[0] is grain
    assert results == []
    assert plot_cf is strong
    splitter.assert_not_called()
    assert remerge.call_args[0][2] == []


def test_batch_with_real_trajectory_detection():
    rng = np.random.RandomState(29)
    angles = np.sort(rng.uniform(0, 360, 180))
    omega = np.r_[angles, angles, angles, angles[:5]]
    dty = np.r_[
        sino._refine_predict(angles, [[-20., -10.]], 2.5).ravel(),
        sino._refine_predict(angles, [[15., -15.]], 2.5).ravel(),
        sino._refine_predict(angles, [[0., 25.]], 2.5).ravel(),
        np.zeros(5)]
    dty += rng.normal(0, .08, len(dty))
    owners = np.r_[np.zeros(180, dtype=int), np.ones(360, dtype=int), np.full(5, 2)]
    cf = SimpleNamespace(titles=[], nrows=len(dty), omega=omega, dty=dty)
    grains = [SimpleNamespace(gid=i) for i in range(3)]
    result = SimpleNamespace(
        grainsinos=[SimpleNamespace(grain=SimpleNamespace()) for _ in range(2)],
        source_2d_rows=np.empty(0, dtype=int), peak_weights=np.empty((0, 2)))
    report = []
    with patch.object(sino, '_refine_peak_owners', return_value=owners), \
            patch.object(sino, '_refine_errors', return_value=np.zeros((len(dty), 1))), \
            patch.object(sino, 'split_grain_sinogram', return_value=result) as splitter:
        updated, results = sino.decouple_grain_sinograms(
            SimpleNamespace(ystep=.1), grains, cf, grain_ids=None,
            width=.5, margin=.2, trials=80, max_components=4,
            report=report, verbose=False)
    assert [row['status'] for row in report] == ['unchanged', 'split', 'failed']
    assert [row['n_components'] for row in report] == [1, 2, None]
    assert splitter.call_count == 1
    assert splitter.call_args[0][1] is grains[1]
    assert splitter.call_args[1]['_initial_fit'][3]['selected'] == 2
    assert updated[0] is grains[0]
    assert updated[2] is grains[2]
    assert results[0].child_ids == (1, 3)
