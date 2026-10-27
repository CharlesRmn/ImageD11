"""Interactive Hough candidate selection and explicit grain recovery."""
from __future__ import print_function

from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib import pyplot as plt

from ImageD11.nbGui import nb_utils as utils
from ImageD11.grain import grain
from ImageD11.columnfile import newcolumnfile


def peak_table():
    rng = np.random.RandomState(18)
    angles = rng.uniform(-720, 1080, 300)
    centre = np.array([.2, -.1])
    y0 = .035
    dty = y0 - centre[0] * np.sin(np.deg2rad(angles)) - centre[1] * np.cos(np.deg2rad(angles))
    cf = newcolumnfile([])
    cf.nrows = 2 * len(angles)
    for name, values in dict(
        omega=np.tile(angles, 2), dty=np.r_[dty, dty + .25],
        grain_id=np.r_[np.full(len(angles), -1), np.zeros(len(angles))],
        sum_intensity=np.ones(cf.nrows)).items():
        cf.addcolumn(values, name)
    return cf, centre, y0


def test_picker_maps_irregular_multiturn_angles_and_preserves_inputs(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    import IPython.display
    monkeypatch.setattr(IPython.display, 'display', lambda *args: None)
    cf, centre, y0 = peak_table()
    before = cf.dty.copy()
    inside = np.zeros((101, 101), dtype=bool)
    inside[30:50, 60:80] = True
    picker = utils.hough_grain_picker(cf, y0, .5, bins=101, spatial_mask=inside)
    try:
        assert len(picker['rows']) == 300
        assert len(picker['centres']) > 0
        np.testing.assert_allclose(picker['centres'][0], centre, atol=.011)
        tables = utils.candidate_peak_tables(picker, .02, [0])
        assert tables[0].nrows == 300
        assert np.all(tables[0].grain_id == -1)
        tables[0].dty[:] = 0
        np.testing.assert_array_equal(cf.dty, before)
        picker['controls']['threshold'].value = .3
        assert len(picker['centres']) > 0
    finally:
        plt.close(picker['figure'])
        picker['output'].close()
        for widget in picker['controls'].values():
            widget.close()


def test_indexing_snapshots_centres_and_forwards_existing_options(monkeypatch):
    cf, centre, y0 = peak_table()
    picker = dict(cf=cf, rows=np.arange(300), omega=cf.omega[:300],
                  dty=cf.dty[:300], y0=y0, centres=centre[None].copy())
    options = dict(unitcell=object(), forgen=[0, 1], foridx=[0, 1, 2], hkl_tols=[.02])
    proposal = grain(np.eye(3))
    calls = []

    def index(table, **kwargs):
        assert table.nrows == 300
        assert kwargs == options
        calls.append(table)
        return [proposal], SimpleNamespace()

    monkeypatch.setattr(utils, 'do_index', index)
    results = utils.index_hough_candidates(picker, .02, options, candidate_ids=[0])
    picker['centres'][:] = 99
    assert len(calls) == 1
    np.testing.assert_array_equal(results[0]['centre'], centre)
    assert results[0]['grains'][0] is proposal


def test_explicit_acceptance_is_repeatable_and_reassigns_new_peaks():
    existing = grain(np.eye(3))
    angle = .37
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    proposal = grain(rotation.T)
    results = {2: dict(centre=np.array([.2, -.1]), grains=[proposal])}
    updated = utils.add_hough_grains([existing], results, [(2, 0)])
    repeated = utils.add_hough_grains([existing], results, [(2, 0)])
    assert len(updated) == len(repeated) == 2
    assert updated[0] is existing and updated[1] is not proposal
    assert updated[1].gid == 1
    assert proposal.translation is None and existing.translation is None
    np.testing.assert_array_equal(updated[1].translation, [.2, -.1, 0])
    np.testing.assert_array_equal(updated[1].ubi, proposal.ubi)
    cf = newcolumnfile([])
    cf.nrows = 2
    vectors = np.array([[1., 0., 0.], rotation[:, 0]])
    for k, name in enumerate(('gx', 'gy', 'gz')):
        cf.addcolumn(vectors[:, k], name)
    utils.assign_peaks_to_grains(updated, cf, .01)
    np.testing.assert_array_equal(cf.grain_id, [0, 1])
    with pytest.raises(ValueError, match='unique'):
        utils.add_hough_grains([existing], results, [(2, 0), (2, 0)])


def test_notebook_recovery_acceptance_and_disabled_path(monkeypatch):
    import json
    from pathlib import Path

    notebook = json.loads((Path(__file__).resolve().parents[1] /
        'ImageD11/nbGui/S3DXRD/tomo_1_index.ipynb').read_text())
    cells = [''.join(c.get('source', [])) for c in notebook['cells']
             if c['cell_type'] == 'code']
    picker_cell = next(c for c in cells if c.startswith('hough_picker = None'))
    index_cell = next(c for c in cells if c.startswith('# Run after adjusting the GUI'))
    accept_cell = next(c for c in cells if c.startswith('# Automatic acceptance with optional override;'))
    cf, centre, y0 = peak_table()
    original = grain(np.eye(3))
    proposed = grain(np.eye(3) * 2)
    monkeypatch.setattr(utils, 'hough_grain_picker', lambda *a, **k:
        dict(centres=centre[None], spacing=.01, controls=dict(smoothing=SimpleNamespace(value=1))))
    monkeypatch.setattr(utils, 'index_hough_candidates', lambda *a, **k:
        {0: dict(centre=centre, grains=[proposed], peaks=cf, indexer=object())})
    monkeypatch.setattr(utils, 'assign_peaks_to_grains', lambda *a, **k: None)
    monkeypatch.setattr(utils, 'plot_index_results', lambda *a, **k: None)
    monkeypatch.setattr(utils, 'plot_grain_sinograms', lambda *a, **k: None)
    env = dict(np=np, utils=utils, recover_hough_grains=True,
               hough_use_sample_mask=False, hough_radius=.5, hough_bins=101,
               ds=SimpleNamespace(y0=y0, ystep=.01), cf_strong_before_hough_recovery=cf,
               grains_before_hough_recovery=[original], hough_candidate_ids=[0],
               accepted_hough_grains=[(0, 0)], hough_trajectory_width=.02,
               ucell=object(), cf_strong_dstol=.005, rings_for_gen=[0, 1],
               rings_for_scoring=[0, 1, 2], hkl_tols_seq=[.01], fracs=[.9],
               cosine_tol=.01, max_grains=10, peak_assign_tol=.05)
    for cell in (picker_cell, index_cell, accept_cell, accept_cell):
        exec(compile(cell, '<notebook>', 'exec'), env)
    assert len(env['grains_before_decoupling']) == 2
    np.testing.assert_array_equal(env['grains'][1].translation, np.r_[centre, 0])
    assert env['grains'][0] is original
    monkeypatch.setattr(utils, 'select_hough_grains', lambda *a: ([(0, 0)], []))
    env.update(hough_candidate_ids=None, hough_trajectory_width=None,
               accepted_hough_grains=None)
    for cell in (picker_cell, index_cell, accept_cell):
        exec(compile(cell, '<notebook>', 'exec'), env)
    assert env['selected_hough_candidates'] == [0]
    assert env['resolved_hough_width'] == .03
    assert env['selected_hough_grains'] == [(0, 0)]
    assert len(env['grains_before_decoupling']) == 2
    env['recover_hough_grains'] = False
    for cell in (picker_cell, index_cell, accept_cell):
        exec(compile(cell, '<notebook>', 'exec'), env)
    assert env['hough_picker'] is None
    assert env['hough_results'] == {}
    assert env['grains_before_decoupling'] == [original]


def test_automatic_acceptance_requires_new_support_and_rejects_duplicates():
    existing = grain(np.eye(3))
    angle = .37
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    proposal = grain(rotation.T)
    cf = newcolumnfile([])
    cf.nrows = 4
    vectors = np.array([[1., 0., 0.], rotation[:, 0],
                        rotation[:, 1], rotation[:, 0] * 2])
    for k, name in enumerate(('gx', 'gy', 'gz')):
        cf.addcolumn(vectors[:, k], name)
    result = dict(peaks=cf, grains=[existing, proposal],
                  indexer=SimpleNamespace(minpks=3))
    choices, report = utils.select_hough_grains([existing], {0: result, 1: result}, .01)
    assert choices == [(0, 1)]
    assert [record['new_peaks'] for record in report] == [0, 3, 0, 0]
    result['indexer'].minpks = 4
    choices, report = utils.select_hough_grains([existing], {0: result}, .01)
    assert choices == []
