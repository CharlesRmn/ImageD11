"""Saved fractional peak assignments used by direct refinement."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ImageD11.sinograms import sinogram as sino


def test_saved_assignment_prepares_hkl_for_sinogram():
    cf = sino.ImageD11.columnfile.newcolumnfile([])
    cf.nrows = 3
    columns = dict(gx=[1., 2., 3.4], gy=[0., 1., 0.], gz=[0., 0., 1.],
                   sum_intensity=[10., 20., 30.], source_2d_row=[4., 5., 6.])
    for name, values in columns.items():
        cf.addcolumn(np.asarray(values), name)
    gs = sino.GrainSinogram.__new__(sino.GrainSinogram)
    gs.ds = SimpleNamespace()
    gs.grain = SimpleNamespace(ubi=np.eye(3))
    with patch.object(sino, '_saved_peak_assignment',
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


def direct_refinement_inputs(tmp_path):
    angles = np.arange(0., 360., 10.)
    bins = np.arange(-.6, .60001, .01)
    centres = np.array([[-.16, -.08], [.18, .12]])
    positions = sino._refine_predict(angles, centres, .03)
    signals = np.exp(-.5 * ((bins[:, None, None] - positions[None]) / .025) ** 2)
    intensity = signals.sum(axis=2)
    yi, ai = np.nonzero(intensity > .001)
    cf = sino.ImageD11.columnfile.newcolumnfile([])
    cf.nrows = len(yi)
    columns = dict(omega=angles[ai], dty=bins[yi], gx=ai + 1.,
                   gy=np.ones(len(yi)), gz=np.ones(len(yi)),
                   eta=np.full(len(yi), 30.), sum_intensity=intensity[yi, ai],
                   spot3d_id=np.arange(len(yi)),
                   grain_id=signals[yi, ai].argmax(axis=1))
    for name, values in columns.items():
        cf.addcolumn(values, name)
    ds = sino.ImageD11.sinograms.dataset.DataSet()
    ds.ystep, ds.ybincens = .01, bins
    ds._pk2d = columns
    ds.grainsfile = str(tmp_path / 'grains.h5')
    grains = [sino.ImageD11.grain.grain(np.eye(3), translation=np.r_[c, 0.])
              for c in centres]
    return ds, grains, cf


def test_direct_refinement_export_and_saved_assignment_round_trip(tmp_path, monkeypatch):
    import h5py
    import warnings

    ds, grains, cf = direct_refinement_inputs(tmp_path)
    original_intensity = cf.sum_intensity.copy()

    def original_rows(dataset, rows, parameters):
        subset = cf.copyrows(rows)
        subset.addcolumn(np.asarray(rows), 'source_2d_row')
        return subset

    monkeypatch.setattr(sino, '_refine_colfile_from_rows', original_rows)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        updated, result = sino.refine_grain_sinograms(
            ds, grains, cf, y0=.03, use_saved_assignments=False,
            workers=1, iterations=2, verbose=False)
    assert len(updated) == len(grains) == 2
    assert result.parent_ids == result.child_ids == (0, 1)
    total = result.peak_weights.sum(axis=1)
    assert (total > 0).any()
    np.testing.assert_allclose(total[total > 0], 1.)
    np.testing.assert_array_equal(cf.sum_intensity, original_intensity)
    assert sino.save_refined_grains(ds, updated, result, phase_name='Fe') is None
    restored = sino.read_h5(ds.grainsfile, ds, group_name='Fe')
    for gid, gs in enumerate(restored):
        assert gs.load_saved_peak_assignment(gid, phase_name='Fe')
        np.testing.assert_array_equal(gs.grain.ubi, grains[gid].ubi)
        np.testing.assert_array_equal(gs.source_2d_rows, result.grainsinos[gid].source_2d_rows)
        np.testing.assert_array_equal(gs.peak_fractions, result.grainsinos[gid].peak_fractions)
        np.testing.assert_array_equal(gs.spatial_split_mask, result.grainsinos[gid].spatial_split_mask)
    with h5py.File(ds.grainsfile, 'r') as saved:
        assert saved['Fe/0'].attrs['refinement_parent_ids'].tolist() == [0, 1]


def test_export_without_refinement_preserves_other_groups_and_makes_backup(tmp_path):
    import h5py

    ds, grains, _ = direct_refinement_inputs(tmp_path)
    with h5py.File(ds.grainsfile, 'w') as saved:
        saved.create_dataset('masks/Fe', data=np.ones((3, 3), dtype=bool))
        saved.create_group('other_phase')
    backup = sino.save_refined_grains(ds, grains, None, phase_name='Fe')
    assert backup.is_file()
    with h5py.File(ds.grainsfile, 'r') as saved:
        assert set(saved['Fe']) == {'0', '1'}
        assert 'other_phase' in saved and 'masks/Fe' in saved
        for gid, grain in enumerate(grains):
            np.testing.assert_array_equal(saved['Fe/' + str(gid) + '/ubi'][:], grain.ubi)
