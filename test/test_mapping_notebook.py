"""Mapping reconstruction setup must work without hidden kernel state."""
from __future__ import print_function

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from ImageD11.sinograms.geometry import sino_shift_and_pad
from ImageD11.sinograms.sinogram import GrainSinogram
from ImageD11.sinograms.dataset import DataSet
from ImageD11.grain import grain


def mapping_setup_cell():
    path = Path(__file__).resolve().parents[1] / 'ImageD11/nbGui/S3DXRD/tomo_2_map.ipynb'
    notebook = json.loads(path.read_text())
    cells = [''.join(c.get('source', [])) for c in notebook['cells']
             if c['cell_type'] == 'code']
    return next(c for c in cells if c.startswith('# Reconstruction geometry and saved sample mask'))


@pytest.mark.parametrize('y0', [0., .03])
def test_mapping_setup_and_first_reconstruction_from_fresh_namespace(tmp_path, y0):
    ds = DataSet()
    ds.y0, ds.ystep, ds.ybincens = y0, .01, np.linspace(-.3, .3, 61)
    ds.grainsfile = str(tmp_path / 'grains.h5')
    shift, pad = sino_shift_and_pad(ds.y0, len(ds.ybincens), min(ds.ybincens), ds.ystep)
    size = len(ds.ybincens) + pad
    mask = np.zeros((size, size), dtype=bool)
    mask[5:-5, 5:-5] = True
    with h5py.File(ds.grainsfile, 'w') as saved:
        saved.create_dataset('masks/Fe', data=mask)
    gs = GrainSinogram(grain(np.eye(3)), ds)
    gs.ssino = np.ones((61, 3), dtype=np.float32)
    gs.sinoangles = np.array([0., 60., 120.])
    env = dict(ds=ds, gs=gs, phase_str='Fe', h5py=h5py, Path=Path, y0_manual=None,
               sino_shift_and_pad=sino_shift_and_pad,
               cImageD11=SimpleNamespace(cores_available=lambda: 1))
    exec(compile(mapping_setup_cell(), '<mapping setup>', 'exec'), env)
    exec('gs.update_recon_parameters(pad=pad, shift=shift, mask=whole_sample_mask, y0=ds.y0)\n'
         'gs.recon(workers=1)', env)
    assert env['pad'] == pad and env['shift'] == shift and env['nthreads'] == 1
    np.testing.assert_array_equal(env['whole_sample_mask'], mask)
    assert gs.recons['iradon'].shape == mask.shape
    assert np.all(gs.recons['iradon'][~mask] == 0)


@pytest.mark.parametrize('file_exists', [False, True])
def test_mapping_setup_falls_back_when_no_parameters_or_mask_are_saved(tmp_path, file_exists):
    path = tmp_path / 'grains.h5'
    if file_exists:
        with h5py.File(str(path), 'w') as saved:
            saved.create_group('Fe')
    ds = SimpleNamespace(y0=.03, ystep=.01, ybincens=np.linspace(-.3, .3, 61),
                         grainsfile=str(path))
    env = dict(ds=ds, phase_str='Fe', h5py=h5py, Path=Path, y0_manual=None,
               sino_shift_and_pad=sino_shift_and_pad,
               cImageD11=SimpleNamespace(cores_available=lambda: 1))
    exec(compile(mapping_setup_cell(), '<mapping setup>', 'exec'), env)
    assert (env['shift'], env['pad']) == sino_shift_and_pad(.03, 61, -.3, .01)
    assert env['whole_sample_mask'] is None


@pytest.mark.parametrize('manual_y0', [None, .024])
def test_mapping_setup_loads_saved_geometry_and_respects_manual_axis(tmp_path, manual_y0):
    path = tmp_path / 'grains.h5'
    saved_y0 = .025
    shift, pad = sino_shift_and_pad(saved_y0, 61, -.3, .01)
    mask = np.ones((61 + pad, 61 + pad), dtype=bool)
    with h5py.File(str(path), 'w') as saved:
        attrs = saved.create_group('Fe/0/recon_parameters').attrs
        attrs['recon_y0'], attrs['recon_shift'], attrs['recon_pad'] = saved_y0, shift, pad
        saved.create_dataset('masks/Fe', data=mask)
    ds = SimpleNamespace(y0=.03, ystep=.01, ybincens=np.linspace(-.3, .3, 61),
                         grainsfile=str(path))
    calls = []

    def calculate(*args):
        calls.append(args)
        return sino_shift_and_pad(*args)

    env = dict(ds=ds, phase_str='Fe', h5py=h5py, Path=Path, y0_manual=manual_y0,
               sino_shift_and_pad=calculate,
               cImageD11=SimpleNamespace(cores_available=lambda: 1))
    exec(compile(mapping_setup_cell(), '<mapping setup>', 'exec'), env)
    expected_y0 = saved_y0 if manual_y0 is None else manual_y0
    assert ds.y0 == expected_y0
    assert (env['shift'], env['pad']) == sino_shift_and_pad(expected_y0, 61, -.3, .01)
    np.testing.assert_array_equal(env['whole_sample_mask'], mask)
    assert len(calls) == int(manual_y0 is not None)
