"""Lightweight checks for the Anri notebook workflow, without experimental data."""
from __future__ import print_function

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import h5py
import IPython.display
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import nbformat
import numpy as np
import pytest
from IPython.core.inputtransformer2 import TransformerManager

from ImageD11.sinograms.tensor_map import TensorMap
from ImageD11.sinograms.geometry import sino_shift_and_pad
from ImageD11.sinograms import dataset


def code_cells(name):
    path = (Path(__file__).resolve().parents[1] / 'ImageD11' / 'nbGui' /
            'anri' / name)
    notebook = json.loads(path.read_text())
    return [''.join(cell['source']) for cell in notebook['cells']
            if cell['cell_type'] == 'code']


@pytest.mark.parametrize('name', [
    'tomo_2_map.ipynb', 'tomo_3_refinement.ipynb', '4_visualise.ipynb'])
def test_notebook_code_compiles(name):
    transformer = TransformerManager()
    for source in code_cells(name):
        compile(transformer.transform_cell(source), name, 'exec')


@pytest.mark.parametrize('name', [
    'tomo_2_map.ipynb', 'tomo_3_refinement.ipynb', '4_visualise.ipynb'])
@pytest.mark.parametrize('namespace_loaded', [False, True])
def test_checkout_selection_precedes_import(monkeypatch, name, namespace_loaded):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(str(root.parent))
    source = next(cell for cell in code_cells(name)
                  if 'local_candidates' in cell or 'for candidate in [' in cell)
    start = source.index('if IMAGED11_PATH is not None:')
    setup = source[start:source.index('import ImageD11', start)]
    modules = {'ImageD11': SimpleNamespace(__file__=None)} if namespace_loaded else {}
    system = SimpleNamespace(modules=modules, path=[])
    namespace = dict(IMAGED11_PATH=None, ANRI_PATH=None,
                     CHECKOUT_PATH=None, sys=system)
    if namespace_loaded:
        with pytest.raises(RuntimeError, match='Restart'):
            exec(compile(setup, '<checkout setup>', 'exec'), namespace)
    else:
        exec(compile(setup, '<checkout setup>', 'exec'), namespace)
        assert system.path[0] == str(root)
        assert namespace['PYTHONPATH'] == str(root)


@pytest.fixture
def validation_cache(tmp_path):
    runtime = ('diff --git a/ImageD11/sinograms/pbp_refine.py b/ImageD11/sinograms/pbp_refine.py\n'
               '--- a/ImageD11/sinograms/pbp_refine.py\n'
               '+++ b/ImageD11/sinograms/pbp_refine.py\n'
               '@@ -1 +1 @@\n-old_runtime\n+new_runtime\n')
    notebook = ('diff --git a/ImageD11/nbGui/anri/4_visualise.ipynb b/ImageD11/nbGui/anri/4_visualise.ipynb\n'
                '--- a/ImageD11/nbGui/anri/4_visualise.ipynb\n'
                '+++ b/ImageD11/nbGui/anri/4_visualise.ipynb\n'
                '@@ -1 +1 @@\n-old_plot\n+new_plot\n')
    old_patch = runtime + notebook
    current_patch = old_patch.replace('+new_plot', '+changed_plot')
    commit = 'a' * 40

    def git(args, **kwargs):
        return SimpleNamespace(stdout=commit if 'rev-parse' in args else current_patch)

    namespace = dict(hashlib=hashlib, h5py=h5py, np=np, Path=Path,
                     numba=SimpleNamespace(__version__='test-numba'),
                     subprocess=SimpleNamespace(run=git), PYTHONPATH=str(tmp_path),
                     VALIDATION=tmp_path)
    cells = code_cells('4_visualise.ipynb')
    setup = next(cell for cell in cells if 'PATCH=subprocess.run' in cell)
    exec(compile(setup[setup.index('PATCH='):], '<source identity>', 'exec'), namespace)
    helpers = next(cell for cell in cells if 'def run_variant(' in cell)
    exec(compile(helpers, '<validation helpers>', 'exec'), namespace)
    collect_inputs = namespace['validation_inputs']
    namespace['validation_inputs'] = lambda manager, fold: ('inputs', dict(fold=fold, files=[]))
    namespace['read_map'] = lambda path: 'cached map'
    output = tmp_path / 'cached.h5'
    with h5py.File(str(output), 'w') as h:
        h.attrs['executed_source_sha256'] = hashlib.sha256(old_patch.encode()).hexdigest()
        h.attrs['validation_inputs_sha256'] = 'inputs'
    diagnostics = tmp_path / 'physical_validation_diagnostics.h5'
    with h5py.File(str(diagnostics), 'w') as h:
        provenance = h.create_group('provenance')
        provenance.attrs['source_commit'] = commit
        provenance.attrs['source_repository'] = 'original checkout'
        provenance.create_dataset('executed_source_diff', data=old_patch,
                                  dtype=h5py.string_dtype('utf-8'))
    return SimpleNamespace(namespace=namespace, output=output, diagnostics=diagnostics,
                           manager=SimpleNamespace(), old_patch=old_patch, runtime=runtime,
                           collect_inputs=collect_inputs)


@pytest.mark.parametrize('fold', [None, 0, 1])
def test_plot_only_change_reuses_legacy_cache_with_exact_source_evidence(validation_cache, fold):
    cache = validation_cache
    original_cache = cache.output.read_bytes()
    original_diagnostics = cache.diagnostics.read_bytes()
    ns = cache.namespace
    assert ns['PATCH'] != cache.old_patch
    assert ns['run_variant'](cache.manager, cache.output, fold=fold) == 'cached map'
    assert cache.output.read_bytes() == original_cache
    assert cache.diagnostics.read_bytes() == original_diagnostics


@pytest.mark.parametrize('failure,message', [
    ('runtime', 'Runtime source changed'),
    ('inputs', 'Inputs changed'),
    ('hash', 'Legacy cache source hash'),
    ('file', 'Missing diagnostics provenance'),
    ('diff', 'Missing executed source diff'),
    ('commit', 'Source commit changed or missing'),
    ('missing_commit', 'Source commit changed or missing'),
])
def test_legacy_cache_rejects_unproven_source_or_changed_inputs(validation_cache, failure, message):
    cache = validation_cache
    ns = cache.namespace
    if failure == 'file':
        cache.diagnostics.unlink()
    elif failure in ('runtime', 'diff', 'commit', 'missing_commit'):
        with h5py.File(str(cache.diagnostics), 'r+') as h:
            if failure in ('runtime', 'diff'):
                del h['provenance/executed_source_diff']
                if failure == 'runtime':
                    changed = cache.old_patch.replace('+new_runtime', '+different_runtime')
                    h.create_dataset('provenance/executed_source_diff', data=changed,
                                     dtype=h5py.string_dtype('utf-8'))
                    with h5py.File(str(cache.output), 'r+') as output:
                        output.attrs['executed_source_sha256'] = hashlib.sha256(changed.encode()).hexdigest()
            elif failure == 'commit':
                h['provenance'].attrs['source_commit'] = 'b' * 40
            else:
                del h['provenance'].attrs['source_commit']
    else:
        with h5py.File(str(cache.output), 'r+') as h:
            h.attrs['validation_inputs_sha256' if failure == 'inputs' else 'executed_source_sha256'] = 'mismatch'
    with pytest.raises(AssertionError, match=message):
        ns['run_variant'](cache.manager, cache.output)
    if failure in ('runtime', 'diff', 'commit', 'missing_commit'):
        source = next(cell for cell in code_cells('4_visualise.ipynb')
                      if cell.startswith('DIAG='))
        with pytest.raises(AssertionError, match=message):
            exec(compile(source, '<diagnostics rejection>', 'exec'), ns)


@pytest.mark.parametrize('failure', [None, 'runtime', 'commit', 'inputs'])
def test_modern_cache_checks_runtime_identity_and_inputs(validation_cache, failure):
    cache = validation_cache
    ns = cache.namespace
    with h5py.File(str(cache.output), 'r+') as h:
        h.attrs['runtime_source_sha256'] = ns['SOURCE_SHA']
        if failure == 'inputs':
            h.attrs['validation_inputs_sha256'] = 'changed'
    # A modern identity directly proves its runtime source, without legacy diagnostics.
    cache.diagnostics.unlink()
    if failure in ('runtime', 'commit'):
        patch = ns['RUNTIME_PATCH'] + '+changed runtime\n' if failure == 'runtime' else ns['RUNTIME_PATCH']
        commit = 'b' * 40 if failure == 'commit' else ns['SOURCE_COMMIT']
        ns['SOURCE_SHA'] = hashlib.sha256((commit + '\n' + patch).encode()).hexdigest()
    if failure:
        with pytest.raises(AssertionError, match='Inputs changed' if failure == 'inputs' else 'Runtime source changed'):
            ns['run_variant'](cache.manager, cache.output)
    else:
        assert ns['run_variant'](cache.manager, cache.output) == 'cached map'


def test_new_cache_keeps_full_patch_and_adds_runtime_identity(validation_cache):
    cache = validation_cache
    ns = cache.namespace
    output = cache.output.with_name('new_cache.h5')
    result = object()

    def refine(output_filename, **kwargs):
        with h5py.File(output_filename, 'w'):
            pass
        return result

    manager = SimpleNamespace(run_refine=refine, icolf=SimpleNamespace(nrows=2),
                              dset=SimpleNamespace(dsfile='dataset.h5'))
    ns.update(time=SimpleNamespace(monotonic=lambda: 1.0), log=lambda message: None)
    assert ns['run_variant'](manager, output) is result
    with h5py.File(str(output), 'r') as h:
        assert h.attrs['executed_source_sha256'] == ns['PATCH_SHA']
        assert h.attrs['runtime_source_sha256'] == ns['SOURCE_SHA']
        assert h.attrs['validation_inputs_sha256'] == 'inputs'
        assert json.loads(h.attrs['validation_inputs_manifest_json']) == dict(fold=None, files=[])
    assert ns['run_variant'](manager, output) == 'cached map'


@pytest.fixture
def validation_input_files(tmp_path):
    geometry, phase, par = [tmp_path / name for name in ('geometry.par', 'Fe.par', 'pars.json')]
    geometry.write_text('distance 200\n')
    phase.write_text('cell__a 2.86\n')
    par.write_text(json.dumps(dict(geometry=dict(file=geometry.name),
                                   phases=dict(Fe=dict(file=phase.name)))))
    ds = dataset.DataSet(analysispath=str(tmp_path))
    ds.parfile = str(par)
    ds.shape = (2, 3)
    ds.omega = np.array([[0., 1., 2.], [0., 1., 2.]])
    ds.dty = np.array([[0., 0., 0.], [1., 1., 1.]])
    ds.scans = ['1.1', '2.1']
    ds.guessbins()
    ds.save()
    paths = [tmp_path / name for name in ('manager.h5', 'peaks.h5', 'map.h5')]
    for path in paths:
        path.write_bytes(path.name.encode())
    manager = SimpleNamespace(own_filename=str(paths[0]), icolf_filename=str(paths[1]),
                              pbpmap_filename=str(paths[2]), dset=ds)
    return manager, paths + [Path(ds.dsfile), par, geometry, phase]


def test_validation_manifest_preserves_legacy_digest_and_tracks_every_input(
        validation_cache, validation_input_files):
    cache = validation_cache
    manager, paths = validation_input_files
    for fold in (None, 0, 1):
        key, manifest = cache.collect_inputs(manager, fold)
        expected = hashlib.sha256((repr(fold) + np.__version__ + 'test-numba').encode())
        for path in paths:
            expected.update(str(path.resolve()).encode())
            expected.update(path.read_bytes())
        assert key == expected.hexdigest()
        assert manifest['format'] == 'file-bytes-v1'
        assert manifest['fold'] == fold
        assert manifest['numpy'] == np.__version__
        assert manifest['numba'] == 'test-numba'
        assert [entry['path'] for entry in manifest['files']] == [str(p.resolve()) for p in paths]
        for entry, path in zip(manifest['files'], paths):
            assert entry['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
            assert entry['size_bytes'] == path.stat().st_size
            assert entry['mtime_ns'] == path.stat().st_mtime_ns
    key, _ = cache.collect_inputs(manager, None)
    paths[-1].touch()
    assert cache.collect_inputs(manager, None)[0] == key  # stats are diagnostics, not identity
    paths[-1].write_text('cell__a 2.87\n')
    assert cache.collect_inputs(manager, None)[0] != key


def test_identical_dataset_save_can_invalidate_byte_guard_without_value_changes(
        validation_cache, validation_input_files):
    cache = validation_cache
    manager, paths = validation_input_files
    path = Path(manager.dset.dsfile)
    with h5py.File(str(path), 'r') as h:
        attrs = dict(h.attrs)
        arrays = {name: value[()] for name, value in h.items()}
    key, manifest = cache.collect_inputs(manager, None)
    manager.dset.save()  # reproduces the former final visualisation cell, on synthetic data only
    with h5py.File(str(path), 'r') as h:
        assert set(h.attrs) == set(attrs)
        assert set(h) == set(arrays)
        for name, value in attrs.items():
            np.testing.assert_array_equal(h.attrs[name], value)
        for name, value in arrays.items():
            np.testing.assert_array_equal(h[name][()], value)
    changed_key, changed_manifest = cache.collect_inputs(manager, None)
    assert changed_key != key
    changed = [old['path'] for old, new in zip(manifest['files'], changed_manifest['files'])
               if old['sha256'] != new['sha256']]
    assert changed == [str(path.resolve())]
    with h5py.File(str(cache.output), 'r+') as h:
        h.attrs['validation_inputs_sha256'] = key
    original = cache.output.read_bytes()
    cache.namespace['validation_inputs'] = cache.collect_inputs
    with pytest.raises(AssertionError, match='legacy cache.*cannot identify'):
        cache.namespace['run_variant'](manager, cache.output)
    assert cache.output.read_bytes() == original


def test_new_manifest_mismatch_reports_recorded_and_current_inputs(
        validation_cache, validation_input_files):
    cache = validation_cache
    manager, paths = validation_input_files
    key, manifest = cache.collect_inputs(manager, None)
    with h5py.File(str(cache.output), 'r+') as h:
        h.attrs['validation_inputs_sha256'] = key
        h.attrs['validation_inputs_manifest_json'] = json.dumps(manifest)
    paths[1].write_bytes(b'changed measured peaks')
    new_key, new_manifest = cache.collect_inputs(manager, None)
    original = cache.output.read_bytes()
    with pytest.raises(AssertionError) as error:
        cache.namespace['check_cached_variant'](cache.output, new_key, new_manifest)
    message = str(error.value)
    for text in (str(cache.output), key, new_key, str(paths[1].resolve()),
                 manifest['files'][1]['sha256'], new_manifest['files'][1]['sha256'],
                 'Recorded manifest', 'Current manifest', 'RESULTS cell', 'do not edit cache hashes'):
        assert text in message
    assert cache.output.read_bytes() == original


@pytest.mark.parametrize('change', [None, 'inputs', 'runtime', 'split', 'diagnostics'])
def test_validation_directory_preserves_legacy_run_and_routes_unproven_inputs(
        validation_cache, tmp_path, change):
    cache = validation_cache
    ns = cache.namespace
    ns['RUN'] = tmp_path
    root = tmp_path / 'validation'
    root.mkdir()
    full = root / 'conservative_seamfixed_refinedmap.h5'
    cache.output.rename(full)
    cache.diagnostics.rename(root / cache.diagnostics.name)
    split = root / 'conservative_split1_refinedmap.h5'
    split.write_bytes(full.read_bytes())
    if change in ('inputs', 'runtime', 'split'):
        with h5py.File(str(split if change == 'split' else full), 'r+') as h:
            h.attrs['validation_inputs_sha256' if change != 'runtime' else 'runtime_source_sha256'] = 'changed'
    elif change == 'diagnostics':
        with h5py.File(str(root / cache.diagnostics.name), 'r+') as h:
            h['provenance'].attrs['source_commit'] = 'changed'
    originals = {path: path.read_bytes() for path in root.iterdir()}
    directory = ns['validation_directory'](cache.manager)
    if change is None:
        assert directory == root
    else:
        assert directory == root / hashlib.sha256((ns['SOURCE_SHA'] + '\ninputs').encode()).hexdigest()
        assert directory.is_dir()
        assert list(directory.iterdir()) == []  # selecting a path does not run refinement
    assert ns['validation_directory'](cache.manager) == directory
    for path, original in originals.items():
        assert path.read_bytes() == original


def test_validation_directory_is_proactively_input_and_source_specific(validation_cache, tmp_path):
    ns = validation_cache.namespace
    ns['RUN'] = tmp_path
    first = ns['validation_directory'](validation_cache.manager)
    ns['validation_inputs'] = lambda manager, fold: ('different inputs', {})
    changed_inputs = ns['validation_directory'](validation_cache.manager)
    ns['SOURCE_SHA'] = 'different source'
    changed_source = ns['validation_directory'](validation_cache.manager)
    assert len({first, changed_inputs, changed_source}) == 3
    assert ns['validation_directory'](validation_cache.manager) == changed_source
    results_cell = next(cell for cell in code_cells('4_visualise.ipynb')
                        if cell.startswith('RESULTS={}'))
    assert results_cell.index('VALIDATION=validation_directory(r)') < results_cell.index('raw=run_variant(')


def test_visualisation_final_input_check_does_not_save_dataset(tmp_path):
    sources = code_cells('4_visualise.ipynb')
    assert not any('ds.save(' in source for source in sources)
    source = next(cell for cell in sources if cell.startswith('# Check that experimental inputs'))
    path = tmp_path / 'dataset.h5'
    path.write_bytes(b'unchanged experimental dataset')
    original = path.read_bytes()
    grains = tmp_path / 'grains.h5'
    manual = np.ones((2, 2), dtype=bool)
    with h5py.File(str(grains), 'w') as h:
        h.create_dataset('masks/Fe', data=manual)
    manifest = dict(inputs={str(path): hashlib.sha256(original).hexdigest()},
                    manual_mask_sha256=hashlib.sha256(manual.tobytes()).hexdigest())
    (tmp_path / 'input_manifest.json').write_text(json.dumps(manifest))

    def no_save():
        pytest.fail('Visualisation must not rewrite the DataSet')

    ns = dict(RUN=tmp_path, Path=Path, hashlib=hashlib, h5py=h5py, np=np,
              SOURCE_GRAINS=grains, phase_str='Fe', primary=manual, ds=SimpleNamespace(save=no_save))
    exec(compile(source, '<final input check>', 'exec'), ns)
    assert path.read_bytes() == original


def test_runtime_source_filter_and_commit_identity(validation_cache):
    cache = validation_cache
    ns = cache.namespace
    filtered = ns['runtime_source_patch']
    assert filtered(cache.old_patch) == filtered(ns['PATCH']) == cache.runtime
    assert filtered(cache.runtime) == cache.runtime
    assert filtered('') == ''
    for header in (
            'diff --git a/a.ipynb b/a.ipynb',
            'diff --git a/a plot.ipynb b/a plot.ipynb',
            'diff --git "a/a plot.ipynb" "b/a plot.ipynb"'):
        assert filtered(header + '\n+saved notebook output\n' + cache.runtime) == cache.runtime
    for header in ('diff --git a/a.py b/a.ipynb', 'diff --git a/a.ipynb b/a.py'):
        assert filtered(header + '\n+renamed runtime source\n')
    assert ns['SOURCE_SHA'] == hashlib.sha256(
        (ns['SOURCE_COMMIT'] + '\n' + cache.runtime).encode()).hexdigest()
    assert ns['SOURCE_SHA'] != hashlib.sha256(
        ('b' * 40 + '\n' + cache.runtime).encode()).hexdigest()


def test_diagnostics_reuse_preserves_original_full_provenance(validation_cache):
    cache = validation_cache
    ns = cache.namespace
    ns['RESULTS'] = {}
    cells = code_cells('4_visualise.ipynb')
    source = next(cell for cell in cells if cell.startswith('DIAG='))
    exec(compile(source, '<diagnostics reuse>', 'exec'), ns)
    with h5py.File(str(cache.diagnostics), 'r') as h:
        assert h['provenance/executed_source_diff'][()].decode() == cache.old_patch
        assert h['provenance'].attrs['source_repository'] == 'original checkout'
        assert h['provenance'].attrs['source_commit'] == ns['SOURCE_COMMIT']


@pytest.mark.parametrize('supported', [True, False])
def test_unit_cell_summary_pretty_print(capsys, supported):
    source = next(cell for cell in code_cells('4_visualise.ipynb')
                  if 'supported_cells = tmap.unitcell' in cell)
    cells = np.array([[2, 3, 4, 90, 91, 92], [4, 5, 6, 92, 93, 94],
                      [999, 999, 999, 999, 999, 999]], dtype=float)
    original = cells.copy()
    tmap = SimpleNamespace(
        plot=lambda field: None, euler=None, unitcell=cells,
        phase_ids=np.array([0, 0, -1] if supported else [-1, -1, -1]))
    namespace = dict(np=np, RESULTS={'test': dict(validated=tmap,
                                                 uc={'reference': 'test'})})
    exec(compile(source, '<unit-cell summary>', 'exec'), namespace)
    output = capsys.readouterr().out
    assert "test reference cell:\n{'reference': 'test'}" in output
    if supported:
        assert 'Mean split-qualified refined cell:' in output
        for key, value in zip(('a (Angstrom)', 'b (Angstrom)', 'c (Angstrom)',
                               'alpha (deg)', 'beta (deg)', 'gamma (deg)'),
                              (3, 4, 5, 91, 92, 93)):
            assert "'%s': %.1f" % (key, value) in output
        assert '999' not in output
    else:
        assert 'No split-qualified voxels for unit-cell statistics' in output
    np.testing.assert_array_equal(cells, original)


@pytest.mark.parametrize('supported', [True, False])
def test_strain_plots_preserve_support_and_tensor_shear(supported):
    source = next(cell for cell in code_cells('4_visualise.ipynb')
                  if 'strain_components = ' in cell)
    strain = np.zeros((3, 4, 3, 3))
    strain[..., 0, 0] = 1e-4
    strain[..., 1, 1] = 2e-4
    strain[..., 2, 2] = 3e-4
    strain[..., 0, 1] = strain[..., 1, 0] = 4e-5
    support = np.full((3, 4), supported, dtype=bool)
    support[0, 0] = False
    tmap = TensorMap(maps={
        'eps_sample': TensorMap.recon_order_to_map_order(strain),
        'phase_ids': TensorMap.recon_order_to_map_order(
            np.where(support, 0, -1))})
    saved = []

    def snapshot(fig, name):
        saved.append(name)
        plt.close(fig)

    namespace = dict(np=np, plt=plt, TensorMap=TensorMap, snap=snapshot,
                     RESULTS={'test': dict(validated=tmap, qualified=support)})
    exec(compile(source, '<strain plots>', 'exec'), namespace)
    if supported:
        assert len(saved) == 2
        assert np.isnan(namespace['eps'][0, 0]).all()
        np.testing.assert_allclose(namespace['hydrostatic'][support], 2e-4)
        expected = np.sqrt((2.0 / 3.0) * (2e-8 + 2 * (4e-5) ** 2))
        np.testing.assert_allclose(namespace['equivalent'][support], expected)
    else:
        assert not saved


@pytest.mark.parametrize('manual_limit', [None, 500.0])
def test_strain_display_limits_ignore_isolated_outliers(manual_limit):
    source = next(cell for cell in code_cells('4_visualise.ipynb')
                  if 'strain_components = ' in cell)
    source = source.replace('strain_limit_microstrain = None',
                            'strain_limit_microstrain = %r' % manual_limit)
    strain = np.zeros((20, 20, 3, 3))
    strain[..., 0, 0] = 1e-4
    strain[..., 1, 1] = 2e-4
    strain[..., 2, 2] = 3e-4
    strain[0, 0, 0, 0] = 1.0
    original = strain.copy()
    support = np.ones((20, 20), dtype=bool)
    tmap = TensorMap(maps={
        'eps_sample': TensorMap.recon_order_to_map_order(strain),
        'phase_ids': TensorMap.recon_order_to_map_order(np.zeros(support.shape, int))})
    namespace = dict(np=np, plt=plt, TensorMap=TensorMap,
                     RESULTS={'test': dict(validated=tmap, qualified=support)},
                     snap=lambda fig, name: plt.close(fig))
    exec(compile(source, '<strain display limits>', 'exec'), namespace)
    if manual_limit is None:
        np.testing.assert_allclose(namespace['limit'], 300)
        np.testing.assert_allclose(namespace['hydro_limit'], 200)
        assert 0 < namespace['equivalent_limit'] < 300
    else:
        for key in ('limit', 'hydro_limit', 'equivalent_limit'):
            assert namespace[key] == manual_limit
    np.testing.assert_array_equal(namespace['eps'], original)
    np.testing.assert_array_equal(
        TensorMap.map_order_to_recon_order(tmap.eps_sample, 0), original)


def split_metric_helper(name='split_strain_metrics'):
    source = next(cell for cell in code_cells('4_visualise.ipynb')
                  if cell.startswith('def split_strain_metrics('))
    namespace = {}
    exec(compile(source, '<split strain metrics>', 'exec'), namespace)
    return namespace[name]


@pytest.fixture
def split_report(monkeypatch):
    displayed = []
    monkeypatch.setattr(IPython.display, 'display', displayed.append)
    return SimpleNamespace(
        displayed=displayed,
        show=split_metric_helper('display_split_strain_summary'))


def test_split_summary_markdown_formatting_and_integer_counts(split_report):
    eps0 = np.zeros((4, 3, 3))
    eps1 = np.zeros_like(eps0)
    eps0[:, 0, 0] = np.array([-2, 0, 2, 4]) * 1e-6
    rows = split_metric_helper()(eps0, eps1, np.ones(4, bool), [(0, 0)])
    original = dict(rows['eps_xx'])
    split_report.show('test', 'common_before_angle', rows, 4, 10)
    assert len(split_report.displayed) == 1
    report = split_report.displayed[0]
    assert isinstance(report, IPython.display.Markdown)
    text = report.data
    for expected in (
            '### test: Common Support Before Angular Qualification',
            'Voxels: 4 of 10.', 'Delta = split0 - split1',
            'tensor shear, not engineering shear',
            '| Component | Support (voxels) | Finite (pairs) | Bias (microstrain) | sigma_delta (microstrain) | MAD scatter (microstrain) | RMS (microstrain) |',
            '| eps_xx | 4 | 4 | 1 | 2.58 | 2.97 | 2.45 |',
            '| Component | p50 (microstrain) | p95 (microstrain) | p99 (microstrain) |',
            '| eps_xx | 2 | 3.7 | 3.94 |',
            'Conditional precision (assumptions above; not certified uncertainty or resolution)',
            '| Component | Per-half sigma (microstrain) | Full-data sigma (microstrain) |',
            '| eps_xx | 1.83 | 1.29 |'):
        assert expected in text
    assert rows['eps_xx'] == original

    # Counts must not be rounded to three significant digits along with metrics.
    rows['eps_xx'].update(n_support=12345, n_finite=12344)
    split_report.show('test', 'qualified', rows, 12345, 20000)
    text = split_report.displayed[1].data
    assert '### test: Qualified Support (split angle <= 0.5 deg)' in text
    assert 'Voxels: 12345 of 20000.' in text
    assert '| eps_xx | 12345 | 12344 | 1 | 2.58 |' in text


@pytest.mark.parametrize('component', [
    (0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)])
def test_split_strain_metric_formulas_and_bias(component):
    metrics = split_metric_helper()
    eps0 = np.zeros((4, 3, 3))
    eps1 = np.full((4, 3, 3), 20e-6)
    i, j = component
    eps0[:, i, j] = eps1[:, i, j] + np.array([-2, 0, 2, 4]) * 1e-6
    key = 'eps_%s%s' % ('xyz'[i], 'xyz'[j])
    support = np.ones(4, dtype=bool)
    original0, original1 = eps0.copy(), eps1.copy()
    row = metrics(eps0, eps1, support, [component])[key]
    assert row['n_support'] == row['n_finite'] == 4
    expected_sigma = np.sqrt(20.0 / 3.0)
    expected = dict(bias=1, sigma_delta=expected_sigma,
                    mad_sigma_delta=1.482602218505602 * 2,
                    rms_delta=np.sqrt(6), abs_p50=2,
                    abs_p95=3.7, abs_p99=3.94,
                    conditional_sigma_half=expected_sigma / np.sqrt(2),
                    conditional_sigma_full=expected_sigma / 2)
    assert set(row) == set(expected) | {'n_support', 'n_finite'}
    for name, value in expected.items():
        np.testing.assert_allclose(row[name], value, rtol=1e-12)
    shifted = metrics(eps0 + 10e-6, eps1, support, [component])[key]
    np.testing.assert_allclose(shifted['bias'], 11)
    np.testing.assert_allclose(shifted['rms_delta'], np.sqrt(126))
    for name in ('sigma_delta', 'mad_sigma_delta',
                 'conditional_sigma_half', 'conditional_sigma_full'):
        np.testing.assert_allclose(shifted[name], row[name])
    np.testing.assert_array_equal(eps0, original0)
    np.testing.assert_array_equal(eps1, original1)


def test_split_strain_metrics_masked_nonfinite_pairs():
    eps0 = np.ones((6, 3, 3)) * 2e-6
    eps1 = np.zeros_like(eps0)
    eps0[:, 0, 0] = np.array([2, 999, np.nan, np.inf, 8, -np.inf]) * 1e-6
    eps1[4, 0, 0] = np.nan
    support = np.array([True, False, True, True, True, True])
    rows = split_metric_helper()(eps0, eps1, support, [(0, 0), (0, 1)])
    assert rows['eps_xx']['n_support'] == rows['eps_xy']['n_support'] == 5
    assert rows['eps_xx']['n_finite'] == 1
    assert rows['eps_xy']['n_finite'] == 5
    np.testing.assert_allclose(rows['eps_xx']['bias'], 2)
    np.testing.assert_allclose(rows['eps_xy']['bias'], 2)
    assert np.isnan(rows['eps_xx']['sigma_delta'])
    eps0[:] = np.nan
    empty = split_metric_helper()(eps0, eps1, support, [(0, 0)])['eps_xx']
    assert empty['n_support'] == 5
    assert empty['n_finite'] == 0
    assert all(np.isnan(value) for key, value in empty.items()
               if key not in ('n_support', 'n_finite'))


@pytest.mark.parametrize('size,selected', [(0, False), (3, False), (1, True)])
def test_split_strain_metrics_empty_and_single(size, selected):
    eps0 = np.ones((size, 3, 3)) * -3e-6
    eps1 = np.zeros_like(eps0)
    support = np.full(size, selected, dtype=bool)
    with np.errstate(all='raise'):
        row = split_metric_helper()(eps0, eps1, support, [(1, 2)])['eps_yz']
    n = int(support.sum())
    assert row['n_support'] == row['n_finite'] == n
    for key in ('sigma_delta', 'conditional_sigma_half', 'conditional_sigma_full'):
        assert np.isnan(row[key])
    if n:
        np.testing.assert_allclose(row['bias'], -3)
        for key in ('rms_delta', 'abs_p50', 'abs_p95', 'abs_p99'):
            np.testing.assert_allclose(row[key], 3)
        assert row['mad_sigma_delta'] == 0
    else:
        assert all(np.isnan(value) for key, value in row.items()
                   if key not in ('n_support', 'n_finite'))


@pytest.mark.parametrize('qualified_count', [0, 1, 2])
@pytest.mark.parametrize('common_finite', [False, True])
def test_split_metric_report_support_selection_and_plot(
        capsys, split_report, qualified_count, common_finite):
    sources = code_cells('4_visualise.ipynb')
    source = next(cell for cell in sources
                  if cell.startswith('STRAIN_SPLIT_METRICS = {}'))
    components = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
    shape = (2, 3)
    eps1 = np.full(shape + (3, 3), 20e-6)
    eps0 = eps1 + np.array([[1, 2, 100], [4, 5, 6]])[..., None, None] * 1e-6
    full_eps = eps1.copy()
    full_eps[1, 2, 0, 0] = np.nan
    valid = np.ones(shape, dtype=bool)
    strain_ok = np.full(shape, common_finite, dtype=bool)
    strain_ok[1, 1] = False
    split_valid = valid.copy()
    split_valid[1, 0] = False
    qualified = np.zeros(shape, dtype=bool)
    qualified.flat[:qualified_count] = common_finite
    result = dict(eps=full_eps, valid=valid, strain_ok=strain_ok,
                  qualified=qualified,
                  splits=[dict(eps=eps0, valid=split_valid, strain_ok=valid),
                          dict(eps=eps1, valid=valid, strain_ok=valid)])
    snapshots = []

    def snapshot(fig, name):
        snapshots.append(name)
        assert len(fig.axes) == 6
        plt.close(fig)

    namespace = dict(np=np, plt=plt, RESULTS={'test': result},
                     strain_components=components, snap=snapshot,
                     display_split_strain_summary=split_report.show,
                     split_strain_metrics=split_metric_helper())
    original_qualified, original_valid = qualified.copy(), split_valid.copy()
    original_eps0, original_eps1 = eps0.copy(), eps1.copy()
    exec(compile(source, '<split metric report>', 'exec'), namespace)
    rows = namespace['STRAIN_SPLIT_METRICS']['test']
    assert set(rows) == {'common_before_angle', 'qualified'}
    for i, j in components:
        key = 'eps_%s%s' % ('xyz'[i], 'xyz'[j])
        before = rows['common_before_angle'][key]
        after = rows['qualified'][key]
        assert before['n_support'] == before['n_finite'] == (3 if common_finite else 0)
        assert after['n_support'] == after['n_finite'] == (qualified_count if common_finite else 0)
        if common_finite:
            np.testing.assert_allclose(before['bias'], 103.0 / 3)
            if qualified_count:
                np.testing.assert_allclose(after['bias'], (1 + qualified_count) / 2.0)
    np.testing.assert_array_equal(result['qualified'], original_qualified)
    np.testing.assert_array_equal(result['splits'][0]['valid'], original_valid)
    np.testing.assert_array_equal(result['splits'][0]['eps'], original_eps0)
    np.testing.assert_array_equal(result['splits'][1]['eps'], original_eps1)
    assert snapshots == ['test_split_strain_component_differences']
    assert len(split_report.displayed) == 2
    before_text, after_text = [report.data for report in split_report.displayed]
    assert 'Common Support Before Angular Qualification' in before_text
    assert 'Qualified Support (split angle <= 0.5 deg)' in after_text
    for text, count in ((before_text, 3 if common_finite else 0),
                        (after_text, qualified_count if common_finite else 0)):
        assert 'Voxels: %d of 6.' % count in text
        for i, j in components:
            component = 'eps_%s%s' % ('xyz'[i], 'xyz'[j])
            assert '| %s | %d | %d |' % (component, count, count) in text
        if not count:
            assert '| eps_xx | 0 | 0 | nan | nan | nan | nan |' in text
    if common_finite and qualified_count == 1:
        assert '| eps_xx | 1 | 1 | 1 | nan | 0 | 1 |' in after_text
    output = capsys.readouterr().out
    for text in ('microstrain', 'ddof=1', '1.482602218505602',
                 'independent equal-variance balanced', 'not certified',
                 'Shared origins/calibration', 'selection bias'):
        assert text in output


@pytest.mark.parametrize('manual_limit,percentile', [
    (None, 99.0), (None, 95.0), (175.0, 99.0)])
def test_split_histogram_display_limits_and_untrimmed_metrics(
        monkeypatch, capsys, split_report, manual_limit, percentile):
    source = next(cell for cell in code_cells('4_visualise.ipynb')
                  if cell.startswith('STRAIN_SPLIT_METRICS = {}'))
    source = source.replace('split_strain_limit_microstrain = None',
                            'split_strain_limit_microstrain = %r' % manual_limit)
    source = source.replace('split_strain_display_percentile = 99.0',
                            'split_strain_display_percentile = %r' % percentile)
    components = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
    eps1 = np.full((400, 3, 3), 20e-6)
    eps0 = eps1.copy()
    for k, (i, j) in enumerate(components):
        delta = np.linspace(-100, 100, 400) * (k + 1)
        delta[0], delta[-1] = -1e8 * (k + 1), 1e8 * (k + 1)
        eps0[:, i, j] += delta * 1e-6
    valid = np.ones(400, dtype=bool)
    qualified = np.arange(400) % 2 == 0
    result = dict(eps=eps1, valid=valid, strain_ok=valid, qualified=qualified,
                  splits=[dict(eps=eps0, valid=valid, strain_ok=valid),
                          dict(eps=eps1, valid=valid, strain_ok=valid)])
    histograms, figures = [], []
    original_hist = matplotlib.axes.Axes.hist

    def histogram(ax, values, *args, **kwargs):
        counts, bins, patches = original_hist(ax, values, *args, **kwargs)
        histograms.append((values.copy(), counts, bins))
        return counts, bins, patches

    def snapshot(fig, name):
        figures.append(fig)
        plt.close(fig)

    monkeypatch.setattr(matplotlib.axes.Axes, 'hist', histogram)
    namespace = dict(np=np, plt=plt, RESULTS={'test': result}, snap=snapshot,
                     strain_components=components,
                     display_split_strain_summary=split_report.show,
                     split_strain_metrics=split_metric_helper())
    original0, original1 = eps0.copy(), eps1.copy()
    original_qualified = qualified.copy()
    exec(compile(source, '<split histogram display>', 'exec'), namespace)
    output = capsys.readouterr().out
    assert 'tails outside the plotted range are omitted' in output
    assert 'all metric values unchanged' in output
    assert len(histograms) == 12
    rows = namespace['STRAIN_SPLIT_METRICS']['test']
    for k, (i, j) in enumerate(components):
        component = 'eps_%s%s' % ('xyz'[i], 'xyz'[j])
        delta = (eps0[:, i, j] - eps1[:, i, j]) * 1e6
        limit = (np.percentile(np.abs(delta), percentile)
                 if manual_limit is None else manual_limit)
        np.testing.assert_allclose(figures[0].axes[k].get_xlim(), [-limit, limit])
        for offset, (support_name, support) in enumerate([
                ('common_before_angle', valid), ('qualified', qualified)]):
            values, counts, bins = histograms[2 * k + offset]
            np.testing.assert_allclose(bins, np.linspace(-limit, limit, 61))
            np.testing.assert_array_equal(values, delta[support])
            outside = int(np.count_nonzero(np.abs(values) > limit))
            assert counts.sum() == values.size - outside
            assert ('test %s %s outside display range: %d of %d' %
                    (component, support_name, outside, values.size)) in output
            expected = split_metric_helper()(eps0, eps1, support, components)[component]
            for key, value in expected.items():
                np.testing.assert_allclose(rows[support_name][component][key], value)
            assert rows[support_name][component]['sigma_delta'] > 1e6
        np.testing.assert_array_equal(histograms[2 * k][2], histograms[2 * k + 1][2])
    np.testing.assert_array_equal(eps0, original0)
    np.testing.assert_array_equal(eps1, original1)
    np.testing.assert_array_equal(result['qualified'], original_qualified)


@pytest.mark.parametrize('kind,manual_limit', [
    ('zero', None), ('empty', None), ('nonfinite', None), ('overflow', None),
    ('zero', 0.0), ('zero', -1.0), ('zero', float('inf')), ('zero', float('nan'))])
def test_split_histogram_positive_fallback(split_report, kind, manual_limit):
    source = next(cell for cell in code_cells('4_visualise.ipynb')
                  if cell.startswith('STRAIN_SPLIT_METRICS = {}'))
    source = source.replace('split_strain_limit_microstrain = None',
                            'split_strain_limit_microstrain = manual_limit')
    eps1 = np.full((0 if kind == 'empty' else 2, 3, 3), 20e-6)
    eps0 = eps1.copy()
    if kind == 'nonfinite':
        eps0[:] = np.nan
    elif kind == 'overflow':
        eps0[:] = np.finfo(float).max
        eps1[:] = -np.finfo(float).max
    valid = np.ones(eps0.shape[0], dtype=bool)
    result = dict(eps=eps1, valid=valid, strain_ok=valid, qualified=valid,
                  splits=[dict(eps=eps0, valid=valid, strain_ok=valid),
                          dict(eps=eps1, valid=valid, strain_ok=valid)])
    figures = []

    def snapshot(fig, name):
        figures.append(fig)
        plt.close(fig)

    namespace = dict(np=np, plt=plt, RESULTS={'test': result}, snap=snapshot,
                     strain_components=((0, 0),), manual_limit=manual_limit,
                     display_split_strain_summary=split_report.show,
                     split_strain_metrics=split_metric_helper())
    with np.errstate(all='raise'):
        exec(compile(source, '<split histogram fallback>', 'exec'), namespace)
    np.testing.assert_allclose(figures[0].axes[0].get_xlim(), [-1, 1])
    row = namespace['STRAIN_SPLIT_METRICS']['test']['common_before_angle']['eps_xx']
    assert row['n_finite'] == (2 if kind == 'zero' else 0)


@pytest.mark.parametrize('omega_bin_width', [.1, None])
def test_spatial_sampling_report_and_notebook_schema(capsys, omega_bin_width):
    path = (Path(__file__).resolve().parents[1] / 'ImageD11' / 'nbGui' /
            'anri' / '4_visualise.ipynb')
    notebook = nbformat.read(str(path), as_version=4)
    nbformat.validate(notebook)
    ids = [cell['id'] for cell in notebook.cells]
    assert ids.index('anri-strain-diagnostics') < ids.index('anri-split-strain-metric-helper')
    source = next(cell for cell in code_cells('4_visualise.ipynb')
                  if "sampling = result['spatial_sampling']" in cell)
    namespace = dict(RESULTS={'test': dict(step=.002, spatial_sampling=dict(
        dty_step=.002, omega_bin_width_deg=omega_bin_width, assumed_beam_size=.009))})
    exec(compile(source, '<spatial sampling report>', 'exec'), namespace)
    output = capsys.readouterr().out
    for text in ('native dty motor units', 'derived', '0.002', '0.009',
                 "beam_size/3 (native dty motor units)': '0.003'", 'ordinal', 'API default ystep',
                 'Actual spatial resolution not measured', 'no PSF/edge benchmark'):
        assert text in output
    expected_omega = "'0.1'" if omega_bin_width is not None else 'None'
    assert "omega bin width (deg; may be irregular/interlaced)': " + expected_omega in output
    assert namespace['RESULTS']['test']['spatial_sampling']['assumed_beam_size'] == .009


@pytest.mark.parametrize('manual_y0', [None, .024])
def test_mapping_saved_geometry_and_manual_override(tmp_path, manual_y0):
    source = next(cell for cell in code_cells('tomo_2_map.ipynb')
                  if cell.startswith('# Reconstruction geometry and saved sample mask'))
    path = tmp_path / 'grains.h5'
    saved_y0 = .025
    shift, pad = sino_shift_and_pad(saved_y0, 61, -.3, .01)
    mask = np.ones((61 + pad, 61 + pad), dtype=bool)
    with h5py.File(str(path), 'w') as output:
        attrs = output.create_group('Fe/0/recon_parameters').attrs
        attrs['recon_y0'] = saved_y0
        attrs['recon_shift'] = shift
        attrs['recon_pad'] = pad
        output.create_dataset('masks/Fe', data=mask)
    dataset = SimpleNamespace(y0=.03, ystep=.01,
                              ybincens=np.linspace(-.3, .3, 61),
                              grainsfile=str(path))
    namespace = dict(ds=dataset, phase_str='Fe', h5py=h5py,
                     y0_manual=manual_y0,
                     y0_guess=.03 if manual_y0 is None else manual_y0,
                     sino_shift_and_pad=sino_shift_and_pad)
    exec(compile(source, '<mapping geometry>', 'exec'), namespace)
    expected = saved_y0 if manual_y0 is None else manual_y0
    assert dataset.y0 == expected
    assert (namespace['shift'], namespace['pad']) == sino_shift_and_pad(
        expected, 61, -.3, .01)
    np.testing.assert_array_equal(namespace['whole_sample_mask'], mask)
