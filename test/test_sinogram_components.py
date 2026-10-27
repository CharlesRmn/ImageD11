"""Spatial component detection, explicit overrides and child ID preservation."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
    ({0: None, 2: None}, [(0,), (2, 3, 4)]),
])
def test_component_child_ids(selection, expected):
    grains = [SimpleNamespace() for _ in range(3)]
    table = SimpleNamespace(titles=[])

    def split(*args, **kwargs):
        assert kwargs['workers'] == 2
        count = kwargs['n_components']
        if isinstance(selection, list):
            assert count == 2
        if count is None:
            parent = next(i for i, g in enumerate(grains) if g is args[1])
            count = len(next(ids for ids in expected if ids[0] == parent))
        return SimpleNamespace(
            grainsinos=[SimpleNamespace(grain=SimpleNamespace()) for _ in range(count)],
            source_2d_rows=np.empty(0, dtype=int), peak_weights=np.empty((0, count)),
        )

    with patch.object(sino, '_refine_peak_owners', return_value=np.array([0, 2])), \
            patch.object(sino, 'split_grain_sinogram', side_effect=split):
        updated, results = sino.decouple_grain_sinograms(
            None, grains, table, grain_ids=selection, workers=2,
         initialization='trajectory')
    assert [result.child_ids for result in results] == expected
    assert updated[1] is grains[1]
    for result in results:
        for gs, gid in zip(result.grainsinos, result.child_ids):
            assert updated[gid] is gs.grain
            assert gs.grain.gid == gid


@pytest.mark.parametrize('count', [1, 0, -1, 2.5, True, 5])
def test_invalid_component_counts(count):
    with pytest.raises(ValueError, match='Component counts'):
        sino.decouple_grain_sinograms(
            None, [None], SimpleNamespace(titles=[]), grain_ids={0: count},
         initialization='trajectory')


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
    with pytest.raises(ValueError, match='180 degrees'):
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
    manual_centres = centres + np.array([2., 3.])
    diagnostic = dict(selected=count, counts=[count], scores=[.1],
                      ambiguous=False, reached_limit=False)
    refined = SimpleNamespace(
        grainsinos=[SimpleNamespace() for _ in range(count)], centres=centres,
        source_2d_rows=np.arange(120), peak_weights=np.zeros((120, count)),
        history=[], converged=True, stats={})
    with patch.object(sino, '_decouple_auto_components',
                      return_value=(centres, 0., labels, diagnostic)) as detect, \
            patch.object(sino, '_decouple_fit_components',
                         return_value=(manual_centres, 0., labels)) as manual_fit, \
            patch.object(sino, '_refine_errors',
                         side_effect=lambda grains, cf: np.zeros((cf.nrows, len(grains)))), \
            patch.object(sino, '_refine_colfile_from_rows', return_value=cf2), \
            patch.object(sino, '_decouple_select_full_branches',
                         return_value=(labels, None, None, np.ones((36, count, 2)))), \
            patch.object(sino, 'refine_grain_sinograms', return_value=([], refined)) as refine:
        result = sino.split_grain_sinogram(
            ds, grain, cf4, verbose=False, n_components=None,
            _initial_fit=(centres, 0., labels, diagnostic) if reuse_fit else None,
            ransac_confidence=.975, initialization='trajectory')
    assert manual_fit.call_count == int(count > 1)
    if count > 1:
        assert manual_fit.call_args.kwargs['confidence'] is None
        np.testing.assert_array_equal(refine.call_args.kwargs['centres'], manual_centres)
    assert detect.call_count == (0 if reuse_fit else 1)
    if not reuse_fit:
        assert detect.call_args.args[-1] == .975
    assert len(refine.call_args[0][1]) == count
    assert refine.call_args[1]['compatibility'].shape == (120, count)
    assert refine.call_args[1]['initial_weights'].shape == (120, count)
    np.testing.assert_allclose(refine.call_args[1]['initial_weights'].sum(axis=1), 1.)
    assert result.stats['component_selection'] == diagnostic
    assert len(result.grainsinos) == count


def test_automatic_count_without_supported_track_is_informative():
    omega = np.tile(np.arange(0, 360, 15), 3)
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
        assert args[-1] == .975
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
            patch.object(sino, '_refine_errors', side_effect=lambda grains, cf, rows=None:
                         np.zeros((cf.nrows if rows is None else len(rows), len(grains)))), \
            patch.object(sino, '_decouple_auto_components', side_effect=detect) as detector, \
            patch.object(sino, 'split_grain_sinogram', side_effect=split) as splitter:
        updated, results = sino.decouple_grain_sinograms(
            SimpleNamespace(ystep=1.), grains, table, grain_ids=None,
            report=report, verbose=False, ransac_confidence=.975, initialization='trajectory')
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
            patch.object(sino, '_refine_errors', side_effect=lambda grains, cf, rows=None:
                         np.zeros((cf.nrows if rows is None else len(rows), len(grains)))), \
            patch.object(sino, '_decouple_auto_components', return_value=(
                np.zeros((1, 2)), 0., np.zeros(20, dtype=int),
                dict(ambiguous=False, reached_limit=False))) as detector, \
            patch.object(sino, 'split_grain_sinogram', return_value=twin_result) as splitter:
        updated, results = sino.decouple_grain_sinograms(
            SimpleNamespace(ystep=1.), grains, cf, grain_ids=None,
            twin_pairs=[(1, 2)], report=report, verbose=False, initialization='trajectory')
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
            sino.decouple_grain_sinograms(None, [None], SimpleNamespace(titles=[]), grain_ids=[0], initialization='trajectory')


def test_batch_no_splits_preserves_strong_table_return():
    grain = SimpleNamespace()
    cf = SimpleNamespace(titles=[], nrows=20, omega=np.arange(20.), dty=np.zeros(20))
    strong = SimpleNamespace(titles=[])
    with patch.object(sino, '_refine_peak_owners', return_value=np.zeros(20, dtype=int)), \
            patch.object(sino, '_refine_errors', side_effect=lambda grains, cf, rows=None:
                         np.zeros((cf.nrows if rows is None else len(rows), len(grains)))), \
            patch.object(sino, '_decouple_auto_components', return_value=(
                np.zeros((1, 2)), 0., np.zeros(20, dtype=int),
                dict(ambiguous=False, reached_limit=False))), \
            patch.object(sino, 'split_grain_sinogram') as splitter, \
            patch.object(sino, 'decoupled_strong_columnfile', return_value=strong) as remerge:
        updated, results, plot_cf = sino.decouple_grain_sinograms(
            SimpleNamespace(ystep=1.), [grain], cf, grain_ids=None,
            cf_strong=strong, verbose=False, initialization='trajectory')
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
            patch.object(sino, '_refine_errors', side_effect=lambda grains, cf, rows=None:
                         np.zeros((cf.nrows if rows is None else len(rows), len(grains)))), \
            patch.object(sino, 'split_grain_sinogram', return_value=result) as splitter:
        updated, results = sino.decouple_grain_sinograms(
            SimpleNamespace(ystep=.1), grains, cf, grain_ids=None,
            width=.5, margin=.2, trials=80, max_components=4,
            report=report, verbose=False, initialization='trajectory')
    assert [row['status'] for row in report] == ['unchanged', 'split', 'failed']
    assert [row['n_components'] for row in report] == [1, 2, None]
    assert splitter.call_count == 1
    assert splitter.call_args[0][1] is grains[1]
    assert splitter.call_args[1]['_initial_fit'][3]['selected'] == 2
    assert updated[0] is grains[0]
    assert updated[2] is grains[2]
    assert results[0].child_ids == (1, 3)


@pytest.mark.parametrize('fixed_y0', [None, 2.5])
@pytest.mark.parametrize('outliers', [False, True])
def test_batched_ransac_matches_reference_and_random_stream(fixed_y0, outliers):
    rng = np.random.RandomState(91)
    omega = rng.uniform(-360, 720, 240)
    dty = sino._refine_predict(omega, [[-20., -10.]], 2.5).ravel()
    dty += rng.normal(0, .08, len(dty))
    if outliers:
        dty[::4] += rng.uniform(-30, 30, len(dty[::4]))
    width = .5
    trials = 150
    reference_rng = np.random.RandomState(42)
    actual_rng = np.random.RandomState(42)
    X = sino._refine_design(omega)
    A = np.column_stack((X, np.ones(len(X)))) if fixed_y0 is None else X
    target = dty if fixed_y0 is None else dty - fixed_y0
    best, best_score = None, -np.inf
    for _ in range(trials):
        ix = reference_rng.choice(len(target), A.shape[1], replace=False)
        if np.linalg.cond(A[ix]) > 50:
            continue
        beta = np.linalg.lstsq(A[ix], target[ix], rcond=None)[0]
        inside = np.abs(np.dot(A, beta) - target) < width
        bins = np.floor(np.mod(omega[inside], 360) / 15).astype(int)
        score = inside.sum() * min(len(np.unique(bins)), 12) / 12.0
        if score > best_score:
            best, best_score = beta, score
    inside = np.abs(np.dot(A, best) - target) < width
    fit = sino.least_squares(
        lambda beta: np.dot(A[inside], beta) - target[inside],
        best, loss='soft_l1', f_scale=width / 2)
    actual_centre, actual_y0 = sino._decouple_ransac_one(
        omega, dty, width, actual_rng, trials, fixed_y0)
    np.testing.assert_allclose(actual_centre, fit.x[:2], rtol=1e-8, atol=1e-8)
    assert abs(actual_y0 - (fit.x[2] if fixed_y0 is None else fixed_y0)) < 1e-8
    expected_state, actual_state = reference_rng.get_state(), actual_rng.get_state()
    np.testing.assert_array_equal(expected_state[1], actual_state[1])
    assert expected_state[2:] == actual_state[2:]


def test_batched_ransac_rejects_ill_conditioned_samples():
    with pytest.raises(RuntimeError, match='RANSAC failed'):
        sino._decouple_ransac_one(
            np.zeros(40), np.arange(40.), .5, np.random.RandomState(42), 60)


def test_short_track_is_rejected_before_nonlinear_refinement():
    omega = np.linspace(10, 150, 400)
    dty = sino._refine_predict(omega, [[15., -15.]], 2.5).ravel()
    with patch.object(sino, 'least_squares') as refine:
        with pytest.raises(RuntimeError, match='insufficient angular coverage'):
            sino._decouple_ransac_one(
                omega, dty, .5, np.random.RandomState(42), 600,
                fixed_y0=2.5, min_angle_bins=12)
    refine.assert_not_called()


def test_four_tracks_are_a_valid_physical_limit():
    angles = np.linspace(0, 359, 180)
    omega = np.tile(angles, 4)
    centres = np.array([[-20., -10.], [15., -15.], [0., 25.], [30., 20.]])
    dty = np.concatenate([sino._refine_predict(angles, [c], 2.5).ravel() for c in centres])
    with patch.object(sino, '_decouple_fit_components', wraps=sino._decouple_fit_components) as fit:
        fitted, _, _, diagnostics = sino._decouple_auto_components(
            omega, dty, .5, .2, trials=80, max_components=8)
    assert len(fitted) == 4
    assert diagnostics['selected'] == 4
    assert not diagnostics['reached_limit']
    assert max(call.args[7] for call in fit.call_args_list) == 4


def test_dense_partial_track_does_not_count_as_a_grain():
    rng = np.random.RandomState(5)
    angles = rng.uniform(0, 360, 300)
    fragment = rng.uniform(40, 170, 450)
    omega = np.r_[angles, fragment]
    dty = np.r_[sino._refine_predict(angles, [[-20., -10.]], 2.5).ravel(),
                sino._refine_predict(fragment, [[25., 20.]], 2.5).ravel()]
    dty += rng.normal(0, .08, len(dty))
    _, _, _, diagnostics = sino._decouple_auto_components(
        omega, dty, .5, .2, trials=80)
    assert diagnostics['selected'] == 1


def test_half_turn_support_across_zero_is_sufficient():
    omega = np.tile(337.5 + 15 * np.arange(12), 4)
    dty = sino._refine_predict(omega, [[-20., -10.]], 2.5).ravel()
    _, _, _, diagnostics = sino._decouple_auto_components(
        omega, dty, .5, .2, trials=60)
    assert diagnostics['selected'] == 1
    assert diagnostics['minimum_angle_coverage'] == 180


@pytest.mark.parametrize('count', [1, 2])
def test_eligible_hkl_scoring_only_visits_owned_rows(count):
    rng = np.random.RandomState(13)
    n = 240
    gv = rng.randint(-3, 4, (n, 3)).astype(float)
    gv += rng.normal(0, .002, gv.shape)
    rows = np.sort(rng.choice(n, 40, replace=False))
    mask = np.zeros(n, dtype=bool)
    mask[rows] = True
    omega = rng.uniform(0, 360, n)
    omega[rows[0]] = np.nan
    gv[rows[1], 0] += .3
    cf = SimpleNamespace(nrows=n, gx=gv[:, 0], gy=gv[:, 1], gz=gv[:, 2],
                         omega=omega, dty=np.zeros(n))
    grains = [SimpleNamespace(ubi=np.eye(3) * (k + 1)) for k in range(count)]
    reference = sino._refine_errors(grains, cf)
    expected = np.flatnonzero(mask & np.isfinite(omega)
                             & (reference < .05).any(axis=1))
    with patch.object(sino, '_refine_errors', wraps=sino._refine_errors) as score:
        selected, compatible = sino._decouple_eligible_4d(grains, cf, .05, mask)
    np.testing.assert_array_equal(selected, expected)
    np.testing.assert_array_equal(score.call_args.kwargs['rows'], rows)
    np.testing.assert_array_equal(compatible[rows], reference[rows] < .05)
    assert compatible.shape == (n, count)
    assert not compatible[~mask].any()


def test_adaptive_ransac_uses_fewer_samples_on_clean_tracks():
    omega = np.linspace(0, 359, 360)
    dty = sino._refine_predict(omega, [[-20., -10.]], 2.5).ravel()
    results = []
    draws = []
    for confidence in (None, .999):
        rng = np.random.RandomState(42)
        sampler = SimpleNamespace(choice=Mock(wraps=rng.choice))
        result = sino._decouple_ransac_one(
            omega, dty, .5, sampler, 600, min_angle_bins=12,
            confidence=confidence)
        results.append(result)
        draws.append(sampler.choice.call_count)
    assert draws == [600, 32]
    np.testing.assert_allclose(results[0][0], results[1][0], atol=1e-8)
    assert abs(results[0][1] - results[1][1]) < 1e-8


@pytest.mark.parametrize('confidence', [0., 1., -.1, 1.1])
def test_invalid_ransac_confidence_is_rejected(confidence):
    with pytest.raises(ValueError, match='confidence'):
        sino._decouple_auto_components(np.arange(100), np.arange(100),
                                       .5, .2, confidence=confidence)
    with pytest.raises(ValueError, match='confidence'):
        sino.decouple_grain_sinograms(
            None, [], SimpleNamespace(titles=[]), grain_ids=None,
            ransac_confidence=confidence, initialization='trajectory')


def test_adaptive_confidence_controls_sample_budget():
    omega = np.linspace(0, 359, 600)
    dty = sino._refine_predict(omega, [[-20., -10.]], 2.5).ravel()
    # Thirty percent inliers, with remaining observations distributed far away.
    rng = np.random.RandomState(91)
    dty[::3] += rng.uniform(80, 160, len(dty[::3]))
    dty[1::3] -= rng.uniform(80, 160, len(dty[1::3]))
    draws = []
    for confidence in (.9, .99999):
        rng = np.random.RandomState(42)
        sampler = SimpleNamespace(choice=Mock(wraps=rng.choice))
        centre, _ = sino._decouple_ransac_one(
            omega, dty, .5, sampler, 600, min_angle_bins=12,
            confidence=confidence)
        np.testing.assert_allclose(centre, [-20., -10.], atol=.05)
        draws.append(sampler.choice.call_count)
    assert 32 <= draws[0] < draws[1] <= 600


def test_confidence_is_forwarded_through_public_split():
    # Invalid confidence must fail before touching dataset/peak geometry.
    with pytest.raises(ValueError, match='confidence'):
        sino.split_grain_sinogram(None, None, None, ransac_confidence=1., initialization='trajectory')


@pytest.mark.parametrize('seed', [17, 71])
def test_adaptive_detection_retains_unequal_strength_tracks(seed):
    rng = np.random.RandomState(seed)
    centres = np.array([[-20., -10.], [15., -15.], [0., 25.], [30., 20.]])
    omegas = [np.sort(rng.uniform(0, 360, n)) for n in (300, 180, 100, 60)]
    omega = np.concatenate(omegas)
    dty = np.concatenate([
        sino._refine_predict(angles, [centre], 2.5).ravel()
        for angles, centre in zip(omegas, centres)])
    dty += rng.normal(0, .08, len(dty))
    expected = centres[np.lexsort((centres[:, 1], centres[:, 0]))]
    for confidence in (None, .999):
        fitted, _, _, diagnostics = sino._decouple_auto_components(
            omega, dty, .5, .2, trials=150, confidence=confidence)
        assert diagnostics['selected'] == 4
        np.testing.assert_allclose(fitted, expected, atol=.05)


@pytest.mark.parametrize('sigma', [.3, .5, .8])
def test_broad_two_tracks_do_not_split_into_four(sigma):
    rng = np.random.RandomState(17)
    angles = np.sort(rng.uniform(0, 360, 600))
    omega = np.tile(angles, 2)
    centres = np.array([[-20., -10.], [15., -15.]])
    dty = np.concatenate([sino._refine_predict(angles, [c], 2.5).ravel() for c in centres])
    dty += rng.normal(0, sigma, len(dty))
    _, _, _, diagnostics = sino._decouple_auto_components(
        omega, dty, .5, .2, trials=100)
    assert diagnostics['selected'] == 2
    # The old distance-only score still demonstrates the over-splitting defect.
    assert diagnostics['counts'][np.argmin(diagnostics['scores'])] == 4
    assert diagnostics['counts'][np.argmin(diagnostics['selection_scores'])] == 2


def test_normalized_density_does_not_reward_duplicate_curves():
    omega = np.linspace(0, 359, 240)
    centre = np.array([[-20., -10.]])
    dty = sino._refine_predict(omega, centre, 2.5).ravel()
    one, _ = sino._decouple_validation_losses(omega, dty, omega, dty, centre, 2.5, .5)
    two, _ = sino._decouple_validation_losses(
        omega, dty, omega, dty, np.repeat(centre, 2, axis=0), 2.5, .5)
    np.testing.assert_allclose(one, two)


@pytest.mark.parametrize('count', [1, 3, 4])
def test_likelihood_keeps_broad_single_and_multiple_tracks(count):
    rng = np.random.RandomState(17)
    angles = np.sort(rng.uniform(0, 360, 600))
    omega = np.tile(angles, count)
    centres = np.array([[-20., -10.], [15., -15.], [0., 25.], [30., 20.]])[:count]
    dty = np.concatenate([sino._refine_predict(angles, [c], 2.5).ravel() for c in centres])
    dty += rng.normal(0, .8, len(dty))
    _, _, _, diagnostics = sino._decouple_auto_components(
        omega, dty, .5, .2, trials=100)
    assert diagnostics['selected'] == count


# Captured from the original scalar, fixed-trial PCG64 initializer, with
# RandomState(19) observations, sigma 0.12, width 0.5, margin 0.2 and 100 trials.
@pytest.mark.parametrize(
    'count,fixed_y0,expected_centres,expected_y0,label_hash,seed_hash',
[(2,
  None,
  [[-19.9947268117583, -9.998812186989824], [15.002416849171869, -15.006744402690735]],
  2.4966603203156037,
  'f59523dfba43349dd54354ec8cb5a3a75580bd7acd8602a8e1ebeb199fddd1c4',
  'feced5775dc4823c9b0f0e859bf202c5783bfefa5c84667412c1a0f0092796e0'),
 (2,
  2.5,
  [[-19.994450462819408, -9.998671012781609], [15.002685384017488, -15.006663619871222]],
  2.5,
  'f59523dfba43349dd54354ec8cb5a3a75580bd7acd8602a8e1ebeb199fddd1c4',
  'c6b49ac554b76e9f6a5344e0ba86d07017065d2c0e306add8474e529c2bab762'),
 (3,
  None,
  [[-19.99469709163691, -9.998483117379001],
   [-0.006490971568655929, 25.0149491612981],
   [15.001907806019348, -15.006970461547274]],
  2.5003097120949427,
  '024b855d5853004591e0e414d5ad405d8577e16abbb556e7a05acdcb950422ff',
  '49d7aabacfe11c5c8a2a55da4ccafdfc5399daa249054e5523781e928e4252ee'),
 (3,
  2.5,
  [[-19.99472405363844, -9.99849530408344],
   [-0.00651424766550727, 25.014934063428473],
   [15.001886067380088, -15.006976793440115]],
  2.5,
  '024b855d5853004591e0e414d5ad405d8577e16abbb556e7a05acdcb950422ff',
  '87260b1b0180f69327c057e34aa3b744f144f2a8b1434af90c3473bf8b8bd697'),
 (4,
  None,
  [[-19.99463396738545, -9.998456594628884],
   [-0.00645056289970875, 25.01485458020164],
   [15.002526354092852, -15.00724556628362],
   [29.987813198718847, 19.996047249962576]],
  2.501024496753858,
  'be21f2853556d75877b8471efc4e8335921f470c488ef55b5f168f8aba285082',
  '327ae1bf4843433cbb16a02d1d92042240fd2f793d0492017b228dcd324398c4'),
 (4,
  2.5,
  [[-19.994725428635157, -9.998492903470556],
   [-0.006528597658990772, 25.014795392918206],
   [15.002459857185771, -15.007269023576713],
   [29.987755526055093, 19.99599454190479]],
  2.5,
  'be21f2853556d75877b8471efc4e8335921f470c488ef55b5f168f8aba285082',
  'c8fcf024ce87fc384d3f519a61aa240c439d8daf0bb6b499303dbdf005497a0c')])
def test_legacy_trajectory_matches_scalar_reference(
        count, fixed_y0, expected_centres, expected_y0, label_hash, seed_hash):
    import hashlib
    if not hasattr(np.random, 'Generator'):
        pytest.skip('Original PCG64 stream requires NumPy 1.17 or later')
    rng = np.random.RandomState(19)
    angles = np.sort(rng.uniform(0, 360, 600))
    centres = np.array([[-20., -10.], [15., -15.], [0., 25.], [30., 20.]])[:count]
    omega = np.tile(angles, count)
    dty = np.concatenate([
        sino._refine_predict(angles, [centre], 2.5).ravel() for centre in centres])
    dty += rng.normal(0, .12, len(dty))
    fitted, y0, labels = sino._decouple_fit_components(
        omega, dty, .5, .2, fixed_y0, 100, 42, count)
    np.testing.assert_allclose(fitted, expected_centres, rtol=0, atol=1e-8)
    assert abs(y0 - expected_y0) < 1e-8
    assert hashlib.sha256(repr(labels.tolist()).encode('ascii')).hexdigest() == label_hash
    seeds, _, _, _ = sino._decouple_select_full_branches(
        omega, dty, fitted, y0, .1, .5, .2)
    assert hashlib.sha256(repr(seeds.tolist()).encode('ascii')).hexdigest() == seed_hash


@pytest.mark.parametrize('count', [None, 3])
def test_fixed_trajectory_count_bypasses_count_detector(count):
    ds = SimpleNamespace(ystep=1., ybincens=np.arange(-100., 101.))
    cf = SimpleNamespace(omega=np.arange(40.), dty=np.zeros(40))
    kwargs = {} if count is None else dict(n_components=count)
    with patch.object(sino, '_decouple_eligible_4d',
                      return_value=(np.arange(40), np.ones((40, 1), dtype=bool))), \
            patch.object(sino, '_decouple_auto_components',
                         side_effect=AssertionError('Detector reached from manual split')), \
            patch.object(sino, '_decouple_fit_components',
                         side_effect=RuntimeError('manual initializer reached')) as fit:
        with pytest.raises(RuntimeError, match='manual initializer reached'):
            sino.split_grain_sinogram(ds, SimpleNamespace(), cf,
                                     ransac_confidence=.999, **kwargs, initialization='trajectory')
    assert fit.call_args.args[7] == (2 if count is None else count)
    assert fit.call_args.kwargs['confidence'] is None


@pytest.mark.parametrize('count', [2, 3, 4])
def test_spatial_threshold_counts_disconnected_blobs(count):
    ri, rj = np.indices((101, 101))
    pixels = [(25, 25), (75, 25), (25, 75), (75, 75)][:count]
    image = sum(np.exp(-((ri - x) ** 2 + (rj - y) ** 2) / 18.) for x, y in pixels)
    centres, masks, radii, _ = sino._decouple_threshold_components(image, count, .01, .12, 2)
    assert len(masks) == count
    assert all(mask.any() for mask in masks)
    assert np.all(radii > 0)
    # Physical coordinate conversion: sample x increases with row; y decreases with column.
    expected = np.array([[(x - 50) * .01, -(y - 50) * .01] for x, y in pixels])
    expected = expected[np.lexsort((expected[:, 1], expected[:, 0]))]
    np.testing.assert_allclose(centres, expected, atol=.001)


@pytest.mark.parametrize('keep_largest', [False, True])
def test_spatial_threshold_uses_fewer_blobs(keep_largest):
    ri, rj = np.indices((101, 101))
    image = np.exp(-((ri - 50) ** 2 + (rj - 50) ** 2) / 18.)
    diagnostics = {}
    centres, masks, radii, _ = sino._decouple_threshold_components(
        image, 2, .01, .12, 2, keep_largest=keep_largest, diagnostics=diagnostics)
    assert centres.shape == (1, 2)
    assert len(masks) == len(radii) == 1
    assert diagnostics['expected'] == 2
    assert diagnostics['found'] == 1
    assert diagnostics['excluded_region_ids'] == []


@pytest.mark.parametrize('count', [1, 2])
@pytest.mark.parametrize('workers', [None, 2])
def test_default_spatial_split_bypasses_all_sinusoid_initializers(count, workers):
    ds = SimpleNamespace(ystep=1., ybincens=np.arange(-100., 101.), y0=0.,
                         pk2d={'spot3d_id': np.arange(120)})
    grain = SimpleNamespace(ubi=np.eye(3), name='parent')
    cf4 = SimpleNamespace(nrows=120, omega=np.arange(120.), dty=np.zeros(120),
                          spot3d_id=np.arange(120), parameters=None)
    cf2 = SimpleNamespace(nrows=120, omega=np.arange(120.), dty=np.zeros(120),
                          eta=np.ones(120), sum_intensity=np.ones(120), filter=lambda mask: None)
    centres = np.array([[-20., -10.], [15., 15.]])[:count]
    weights = np.eye(count)[np.arange(120) % count]
    refined = SimpleNamespace(grainsinos=[SimpleNamespace() for _ in range(count)],
                              centres=centres, source_2d_rows=np.arange(120), peak_weights=weights,
                              history=[], converged=True, stats={})
    with patch.object(sino, '_refine_errors',
                      side_effect=lambda grains, cf: np.zeros((cf.nrows, len(grains)))), \
            patch.object(sino, '_refine_colfile_from_rows', return_value=cf2), \
            patch.object(sino, '_decouple_spatial_initialization',
                         return_value=(centres, weights, np.full(count, 10.))) as spatial, \
            patch.object(sino, '_decouple_auto_components', side_effect=AssertionError('sin fit')), \
            patch.object(sino, '_decouple_fit_components', side_effect=AssertionError('sin fit')), \
            patch.object(sino, '_decouple_ransac_one', side_effect=AssertionError('RANSAC')), \
            patch.object(sino, '_decouple_select_full_branches', side_effect=AssertionError('trajectory')), \
            patch.object(sino, 'refine_grain_sinograms', return_value=([], refined)) as feedback:
        result = sino.split_grain_sinogram(ds, grain, cf4, verbose=False, workers=workers)
    assert feedback.call_args.kwargs['workers'] == workers
    assert len(feedback.call_args.args[1]) == count
    assert feedback.call_args.kwargs['compatibility'].shape == (120, count)
    assert spatial.call_count == 1
    assert spatial.call_args.kwargs['keep_largest'] is True
    assert spatial.call_args.args[4] == ds.y0
    np.testing.assert_array_equal(feedback.call_args.kwargs['initial_weights'], weights)
    np.testing.assert_array_equal(feedback.call_args.kwargs['centres'], centres)
    assert feedback.call_args.kwargs['y0'] == ds.y0
    assert result.stats['initialization'] == 'spatial_threshold'
    assert np.all(result.labels4d == -1)


@pytest.mark.parametrize('requested_count', [2, 3])
def test_spatial_initializer_uses_real_back_and_forward_projection(requested_count):
    angles = np.arange(0., 360., 6.)
    grid = np.arange(-.65, .65001, .01)
    centres = np.array([[-.2, -.1], [.2, .15]])
    a = np.deg2rad(angles)
    positions = -np.sin(a)[:, None] * centres[:, 0] - np.cos(a)[:, None] * centres[:, 1]
    signals = np.exp(-.5 * ((grid[:, None, None] - positions[None, :, :]) / .025) ** 2) * 100
    sino_image = signals.sum(axis=2)
    keep = sino_image > .01
    y_index, angle_index = np.nonzero(keep)
    cf = SimpleNamespace(nrows=len(y_index), omega=angles[angle_index], dty=grid[y_index],
                         gx=angle_index + 1., gy=np.ones(len(y_index)), gz=np.ones(len(y_index)),
                         eta=np.full(len(y_index), 30.), sum_intensity=sino_image[keep])
    ds = SimpleNamespace(ystep=.01, ybincens=grid)
    parent = SimpleNamespace(recons={})

    def reconstruct(gs, cf2, weights, y0, shift, pad, filter_name):
        # Use the same per-projection normalization as GrainSinogram.build_sinogram.
        image = sino_image / sino_image.max(axis=0)[None, :]
        gs.recons['iradon'] = sino.ImageD11.sinograms.roi_iradon.run_iradon(
            image, angles, pad=pad, shift=shift, workers=1, filter_name=filter_name)

    with patch.object(sino, 'GrainSinogram', return_value=parent), \
            patch.object(sino, '_refine_rebuild_sinogram', side_effect=reconstruct), \
            patch.object(sino, '_decouple_ransac_one', side_effect=AssertionError('RANSAC')):
        fitted, weights, radii = sino._decouple_spatial_initialization(
            ds, SimpleNamespace(ubi=np.eye(3)), cf, requested_count, 0., .12, 2, .02, .75, 'hann', 5)
    np.testing.assert_allclose(fitted, centres, atol=.02)
    active = weights.sum(axis=1) > 0
    np.testing.assert_allclose(weights[active].sum(axis=1), 1.)
    assert np.all(weights >= 0) and np.all(weights <= 1)
    assert np.all((weights > 0).sum(axis=0) >= 20)
    for k in range(2):
        separated = np.abs(positions[angle_index, 0] - positions[angle_index, 1]) > .25
        core = separated & (np.abs(cf.dty - positions[angle_index, k]) < .025)
        assert core.any()
        np.testing.assert_allclose(weights[core, k], 1.)


def test_largest_spatial_regions_exclude_and_report_26_speckles():
    ri, rj = np.indices((161, 161))
    image = (np.exp(-((ri - 40) ** 2 + (rj - 40) ** 2) / 72.)
             + .8 * np.exp(-((ri - 120) ** 2 + (rj - 120) ** 2) / 50.))
    speckles = [(i, j) for i in range(10, 151, 20) for j in range(10, 151, 20)
                if (i - 40) ** 2 + (j - 40) ** 2 > 35 ** 2
                and (i - 120) ** 2 + (j - 120) ** 2 > 35 ** 2][:26]
    assert len(speckles) == 26
    for i, j in speckles:
        image[i, j] = 2.
    before = image.copy()
    diagnostics = {}
    centres, masks, radii, _ = sino._decouple_threshold_components(
        image, 2, .01, .12, 2, keep_largest=True, diagnostics=diagnostics)
    assert diagnostics['found'] == 28
    assert len(diagnostics['selected_region_ids']) == 2
    assert len(diagnostics['excluded_region_ids']) == 26
    assert len(diagnostics['regions']) == 28
    kept = [r for r in diagnostics['regions'] if r['selected']]
    excluded = [r for r in diagnostics['regions'] if not r['selected']]
    assert min(r['pixels'] for r in kept) > max(r['pixels'] for r in excluded)
    assert diagnostics['excluded_pixels'] == sum(r['pixels'] for r in excluded)
    np.testing.assert_allclose(diagnostics['excluded_density'], sum(r['density_sum'] for r in excluded))
    np.testing.assert_allclose(centres, [[-.4, .4], [.4, -.4]], atol=.002)
    np.testing.assert_array_equal(before, image)
    assert len(masks) == 2 and len(radii) == 2
    with pytest.raises(RuntimeError, match='found 28 disconnected blobs; expected 2'):
        sino._decouple_threshold_components(image, 2, .01, .12, 2, keep_largest=False)


def test_spatial_threshold_rejects_zero_blobs():
    with pytest.raises(RuntimeError, match='found 0 disconnected blobs'):
        sino._decouple_threshold_components(np.ones((10, 10)), 2, .01, 2., 2)
