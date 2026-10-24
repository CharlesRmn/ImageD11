# Grain peak refinement by back and forward projection

`refine_grain_sinograms` refines peak selection for one or more indexed grains
while preserving their count, IDs and UBIs. Each iteration reconstructs the
current weighted peaks, extracts a connected spatial support around each
grain centre, projects that support at the measured angles, and updates the
intensity fractions of compatible peaks. Peaks outside every compatible
support are rejected. Shared measured intensity is divided among contributing
grains; it is never counted more than once.

```python
from ImageD11.sinograms.sinogram import (
    refine_grain_sinograms, save_refined_grains,
)
from ImageD11.nbGui import nb_utils as utils

grains_refined, refinement, cf_strong_refined = refine_grain_sinograms(
    ds, grains, cf_4d,
    cf_strong=cf_strong_before_decoupling,
    hkl_tol=peak_assign_tol,
    iterations=10,
    mask_level=0.12,
    dilation=2,
    phase_name=phase_str,
)
utils.plot_grainsino_grid(refinement.grainsinos)

# After inspecting the masks, fractions and convergence:
save_refined_grains(ds, grains_refined, refinement, phase_name=phase_str)
ds.save()
```

Omit `cf_strong` to receive two return values. The optional third value remains
a merged 4D plotting table restricted to the original strong subset. Keep the
original tables for reruns: plotting contributions can repeat a `spot3d_id`.

`grain_ids=[0, 2]` restricts refinement to those original list positions. Grains
outside that list retain their objects and assigned peak pools. Their saved
fractional contributions also reserve intensity when a measured peak is shared.
`refinement.intensity_budget` records the fraction available to the selected
grains. A partial export preserves matching saved assignments of untouched
grains in the same phase.

Initialization uses existing 4D labels or the best HKL match, plus saved
fractions when present. Strong-table labels override labels for that subset.
At least twenty contributing 2D peaks are required per selected grain.
Existing translations initialize spatial centres; missing translations are
fitted to a single trajectory. With `y0=None`, individual offset estimates are
combined into one common median rotation axis. Missing positions are then
refitted with that fixed axis; supplied/current positions are retained.
`centres`, `y0` and `roi_radius` can be supplied explicitly. A scalar `y0`
sets the common calibrated axis; explicit per-grain arrays remain supported.
`roi_radius` accepts scalar or per-grain values.

Spatial decoupling and twins now use the same master function. Their specialized
initializers detect spatial trajectories or use twin-exclusive reflections, then pass
`cf_2d`, `initial_weights` and compatibility masks to `refine_grain_sinograms`.
They consider compatible unassigned peaks alongside the selected parents'
assigned peaks, allowing a spatial branch missed during indexing to be recovered.
Peaks assigned to other parents are excluded, and overlapping independent splits
are checked for double allocation of measured intensity.
To jointly refine all grains immediately after decoupling, supply the in-memory
results, without an intermediate export:

```python
grains_refined, refinement = refine_grain_sinograms(
    ds, grains, cf_4d,
    initial_results=decoupling_results,
    hkl_tol=peak_assign_tol,
    phase_name=phase_str,
)
```

Both notebooks expose `refine_all_grain_peaks` and `peak_refinement_options` in
their parameter cells. The indexing notebook also exposes `grain_ids_to_decouple`,
`twin_pairs_to_decouple` and the specialized `decoupling_options`; those fitting
controls override the common feedback settings. A calibrated `y0` in
`decoupling_options` also applies to joint refinement in the indexing notebook.
Parameter trials use preserved grain and peak snapshots.
Indexing trials initialize from original peaks and current in-memory decoupling
results (`use_saved_assignments=False` by default in that notebook).

The mapping notebook restores fractional assignments, masks and fitted offsets
through `load_saved_peak_assignment` and `prepare_peaks_from_4d`, even when its
rebuilt strong table assigns no peaks to a spatial child. It combines the offsets
into one common median `y0`, shift and padding for all grains. With refinement disabled,
saved fractions retain the original peak subset. Enable refinement to reconsider
the newly selected mapping peaks, including higher-angle projections. The mapping
loop preserves the original `cf_strong` and returns `cf_strong_refined` for plotting.
Its existing HDF5 export cells write the final fractions and reconstructions;
datasets are replaced when the peak or projection count changes. Feedback currently
requires `is_half_scan=False`.

Inspect `refinement.converged` and `refinement.history`, which report weight
changes, retained intensity, rejected peaks, shared peaks and calibrated
residuals. `weight_change` averages the absolute changes of individual grain
fractions over active peaks, preserving the original two-grain decoupling
tolerance. Convergence also requires `maximum_change < 5 * weight_tolerance`.
These are model-dependent assignments: an incorrect centre or a mask
contaminated by reconstruction streaks can bias the feedback. Adjust
`mask_level`, `dilation`, `roi_radius` and `support_level` against the observed
grain shapes. Full scans on a uniform increasing dty grid are currently required.

Feedback backprojections evaluate only each grain's fixed spatial ROI and a
four-pixel halo on each axis. The halo preserves the Gaussian-smoothed density
used for masks and forward projection. Image dimensions and coordinates stay
the same throughout. For shifted scans, regions reaching the detector boundary
retain full backprojection to preserve native interpolation behavior.
The last iteration reconstructs the full image, including
when convergence occurs early; `iterations=0` also returns a full reconstruction.
This reduces repeated backprojection work without changing peak fractions,
spatial masks or the returned full reconstructions. The final iteration can take
longer because it reconstructs the full image.

Warm single-thread synthetic benchmarks with 401 translation bins, 180 initial
projections per grain and 403-by-403 reconstructions measured:

| Grains | Iterations | Full feedback | ROI feedback | Speedup |
| --- | --- | --- | --- | --- |
| 3 | 3 | 2.83 s | 1.06 s | 2.66x |
| 10 | 3 | 14.60 s | 5.77 s | 2.53x |
| 3 | 10 | 8.27 s | 1.74 s | 4.75x |

An off-centre scan with `y0=0.23` and 447-by-447 reconstructions took 9.97 s
with full feedback and 1.93 s with ROI feedback for three grains and ten
iterations (5.16x).

These comparisons excluded compilation and produced identical peak fractions,
sinograms, masks, full reconstructions and diagnostics except elapsed times.
The gain depends on grain ROI size, the number of iterations and how many
boundary regions require full backprojection.

With `verbose=True` (the default), a notebook/terminal progress bar tracks peak
preparation, initial reconstruction, each grain's projection/reconstruction,
final masks and the optional merged plotting table. The total adjusts when
feedback converges early. Per-iteration messages report retained, rejected and
shared peaks; additions, removals and reassignments; retained intensity; weight
changes; and elapsed seconds. `verbose=False` disables the bar and messages;
numerical warnings retain their existing behavior.

`refinement.stats` contains the final summary, including per-grain counts and
assigned intensities, projection counts and spatial-mask pixel counts. Shared
measured peaks count once in `retained` and once
per contributing grain in `counts`. `retained_intensity` retains its existing
definition relative to all candidate intensity; `retained_available_intensity`
uses the intensity available after reserving unselected grains' fractions.
`added`, `removed` and `reassigned` compare final assignments with the initial
seeds. Reassignment requires a change of the dominant grain and a change larger
than `weight_tolerance` in its previous fraction. History entries compare each
iteration with its preceding assignments and include `changed` (any fraction
changes above the tolerance) and `elapsed_seconds`. Iteration timing includes
reconstruction; total timing also includes preparation and finalization.

```python
print(refinement.stats)
print(refinement.history[-1] if refinement.history else "Initialization only")
```

### Automatic spatial component counts

Use `grain_ids_to_decouple = [0, 2]` in the indexing notebook. Each selected
parent's number of spatial trajectories is estimated automatically, up to
`decoupling_options["max_components"]` (default 8). Only selected parents are
examined. A single detected trajectory retains one child and appends no grains.
Child 0 keeps the parent ID; additional children are appended.

Candidate counts are fitted with a common rotation-axis offset, using the
existing centreline width and margin. Three validation folds hold out entire
30-degree angular sectors, keeping repeated turns together. Each track needs
at least eight training peaks spanning three 15-degree sectors. Scores average
capped squared prediction errors across occupied angular sectors and add a
parameter-count penalty. The smallest count within one standard error of the
best score is selected. The search stops at the first unsupported count.
At least four occupied 30-degree sectors are required.

`result.stats["component_selection"]` reports candidate counts, scores, the
selected count, ambiguity within that score tolerance, and whether the selected
count reached the search limit. These are model diagnostics, not calibrated
probabilities. Increase `max_components` if the selected count reaches it.
Inspect reconstructed spatial masks before export. Nearby tracks, broad grains,
weak branches, limited angles or a poorly chosen centreline width can make the
count uncertain; estimating trajectories does not establish physical grain identity.

`decouple_grain_sinograms(grain_ids=[0, 2])` uses the same automatic selection.
A direct `split_grain_sinogram` call defaults to `n_components=None` for automatic
counting. Explicit mappings such as `{0: 3, 2: 4}` and direct integer
`n_components` values remain available as overrides (at least two).
Known twin pairs retain exactly two distinct orientations. Spatial components
of one parent inherit its UBI and share the existing conserved-intensity
refinement loop. Detection operates on eligible merged 4D peaks before loading
that parent's 2D contributions; candidate residual arrays cost approximately
8 bytes per eligible 4D peak per candidate component, not per global 2D peak.

### Batch screening of all grains

The indexing notebook defaults to `grain_ids_to_decouple = None`. This screens
every original parent, excluding members of explicit known twin pairs. Only a
supported, unambiguous count above one and below `max_components` triggers 2D
preparation and reconstruction. The validated detection fit is reused.
Single-track parents retain their original grain objects and peak assignments.
Children are appended only for successful decouplings, preserving parent IDs.

```python
report = []
grains, results, cf_strong_decoupled = decouple_grain_sinograms(
    ds, grains_before_decoupling, cf_4d,
    grain_ids=None,
    twin_pairs=twin_pairs_to_decouple,
    cf_strong=cf_strong_before_decoupling,
    report=report,
    **decoupling_parameters,
)
```

`report` receives dictionaries with `parent_ids`, `child_ids`, `n_components`,
`status`, `component_selection` and `error`. Status is `split`, `unchanged`,
`review` or `failed`. Ambiguous counts and counts at the search limit need
`review`; failed screening/reconstruction records the error. Both retain the
original parents, and processing continues. Global input and intensity-
conservation errors still stop processing. Known twin pairs retain their two
existing IDs and UBIs and are processed once.

The existing two/three return values are unchanged; `results` contains only
completed splits or twin refinements. The notebook collects `decoupling_report`
and prints its summary and unresolved entries. `plot_decoupling=False` avoids
creating a figure for each split during batch execution; enable it for inspection.
Reruns start from preserved original grains and peaks. Mapping restores the
exported split assignments without screening them again. In the mapping
notebook, `batch_mode=True` selects a noninteractive Matplotlib backend and
uses an existing whole-sample mask or thresholding instead of mask drawing.
Set `batch_mode=False` to use interactive plotting and mask drawing. Use
`grain_ids_to_decouple=[]` to skip spatial screening, an explicit list to process
selected parents, or a mapping for manual component counts. Explicit selections
keep their previous fail-fast behavior and can reconstruct a single component.
Batch failures are recorded in memory/logs; inspect the report before treating
the exported grain count as fully resolved.

### Refinement statistics and graphics

Both notebooks call `utils.report_sinogram_refinement(result)` after joint
refinement. The indexing notebook also reports each completed spatial/twin split.
The text includes measured candidate, retained, rejected, exclusive and shared
peak counts; additions, removals and reassignments relative to initialization;
retained intensity fractions, convergence, calibrated residual and elapsed time.
The per-grain table compares initial/final contributing peak counts and includes
assigned intensity, projection count and mask size. Shared measured peaks count
once globally and once per contributing grain in the per-grain counts.

The overview plots initial/final counts, assigned intensity, weight changes,
retained available intensity, iteration-to-iteration assignment changes and the
calibrated relative residual. Detailed previews show each selected grain's final
sinogram, reconstruction with support outline, and fractional peak assignments.
Iteration charts explicitly indicate when there are no measurements, including
initialization-only runs. There are no stored initial sinograms or reconstructions,
so spatial previews show final results rather than an invented before/after.

`plot_refinement_report=False` keeps the textual report and suppresses figures.
`refinement_report_limit=4` caps grain previews per result and the number of split
results plotted during batch runs; numerical summaries still cover every result.
The notebooks explicitly display and close report figures, including with the
mapping notebook's noninteractive batch backend. Assignment graphics sample at
most 10000 candidate rows per grain, while all statistics use the full data.

```python
figures = utils.report_sinogram_refinement(refinement, max_grains=4)
# Optional standalone graphics; filenames/destination are chosen by the caller.
for index, figure in enumerate(figures):
    figure.savefig("refinement_report_{}.png".format(index), dpi=160)
```
