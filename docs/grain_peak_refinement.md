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

### Spatial threshold decoupling (default manual workflow)

Set `grain_ids_to_decouple = {parent_id: expected_count}` with the actual parent
IDs. A list requests two components per selected parent. The indexing notebook
starts with an empty list, and `decoupling_options["initialization"] = "spatial"`.
Direct `split_grain_sinogram` calls use this same spatial mode by default.

The confirmed workflow is reconstruction-first:

1. Reconstruct all compatible parent peaks into one image.
2. Smooth positive density with the same one-pixel Gaussian used by feedback.
3. Threshold at `mask_level` times the image maximum and label disconnected blobs.
4. Fill holes and apply the existing `dilation` to each blob; determine its centre
   and enclosing radius from image coordinates, not from a sinusoid fit.
5. Forward-project each mask onto measured peak positions to initialize conserved
   fractional assignments, then run the existing back/forward projection loop.

Supply a calibrated `y0`, or store it as `ds.y0`. Spatial mode does not estimate
the rotation axis by fitting peak trajectories and does not assume zero when it
is missing. `mask_level`, `dilation`, `support_level` and `roi_radius` control the
spatial supports. Sinusoid width, angular support, trial count and random seeds
are not used by this initialization path. Known twin pairs retain their separate
exclusive-reflection initialization.

With `spatial_keep_largest=True` (the selected default), extra thresholded regions
are ranked by pixel area and the expected number of largest regions are retained
as spatial seeds. Every excluded region is reported. Set the option to `False`
for strict count matching. Fewer regions than expected always raises; touching
regions are not forcibly subdivided. Initial `labels4d`
are -1 because assignments are made at the underlying 2D peak level. Results mark
`stats["initialization"] = "spatial_threshold"`. Exported fractional provenance
and the mapping notebook's saved-assignment reader are unchanged.

The exact older source was not recoverable from the commit, stash or dangling
Git objects. This reconstructs the user-confirmed thresholded-blob workflow;
it is not a claim of byte-for-byte recovery. Tests exercise actual iradon and
forward-projection kernels on synthetic blobs and check centre recovery, peak
selection and intensity conservation. Real-scan reconstruction quality remains
unverified without the scan and its former settings.

The previous RANSAC-based mode remains explicitly available with
`initialization="trajectory"`. Its fixed-count legacy initializer and scalar/PCG64
regression fixtures are retained for comparison, but it is not the default spatial
workflow. Automatic sinusoid-count screening also requires this explicit mode.

### Automatic spatial component counts

Set `decoupling_options["initialization"] = "trajectory"` and use
`grain_ids_to_decouple = {0: None, 2: None}` to opt selected parents into
automatic sinusoid-count estimation, up to
`decoupling_options["max_components"]` (default 4; larger requested limits are capped at four). Only selected parents are
examined. A single detected trajectory retains one child and appends no grains.
Child 0 keeps the parent ID; additional children are appended.

Candidate counts are fitted with a common rotation-axis offset, using the
existing centreline width and margin. Three validation folds hold out entire
30-degree angular sectors, keeping repeated turns together. Each track must occupy at least twelve 15-degree sectors (180 degrees of
angular support) in the full data. Training folds retain two thirds of the
angular sectors and require eight occupied sectors (120 degrees) per track. Selection scores average normalized Gaussian-mixture negative log likelihoods
across occupied angular sectors and add a parameter-count penalty. Branch widths
and mixture weights are estimated only from training data. The smallest count
within one standard error of the best selection score is selected. The search stops at the first unsupported count.
At least twelve occupied 15-degree sectors are required. Partial residual
tracks are rejected during RANSAC scoring and before nonlinear refinement.

`result.stats["component_selection"]` reports candidate counts, the selected
count, ambiguity and whether a computational search limit was reached.
`selection_scores` and `selection_standard_error` are the likelihood-based values
used for selection; lower scores are better. `selection_method` identifies the
metric. The original `scores` and `standard_error` fields retain the legacy
capped-distance diagnostics for comparison. These are model diagnostics, not calibrated
probabilities. A selected count of four is valid because four is the known physical maximum.
A search limit below four can still leave the result unresolved.
Inspect reconstructed spatial masks before export. Nearby tracks, broad grains,
weak branches, limited angles or a poorly chosen centreline width can make the
count uncertain; estimating trajectories does not establish physical grain identity.

`decouple_grain_sinograms(grain_ids={0: None, 2: None})` uses automatic selection.
A direct `split_grain_sinogram(..., initialization="trajectory", n_components=None)`
opts into automatic counting; the default is a spatial two-blob split. Explicit mappings such as `{0: 3, 2: 4}` and direct integer
`n_components` values remain available as overrides (two to four).
Known twin pairs retain exactly two distinct orientations. Spatial components
of one parent inherit its UBI and share the existing conserved-intensity
refinement loop. Detection operates on eligible merged 4D peaks before loading
that parent's 2D contributions; candidate residual arrays cost approximately
8 bytes per eligible 4D peak per candidate component, not per global 2D peak.

### Batch screening of all grains

Opt into sinusoid-based batch estimation with `grain_ids_to_decouple = None`
and `decoupling_options["initialization"] = "trajectory"`. This screens
every original parent, excluding members of explicit known twin pairs. Only a
supported, unambiguous count above one within the four-track maximum triggers 2D
preparation and reconstruction. The detected count is passed to the original manual spatial initializer.
Single-track parents retain their original grain objects and peak assignments.
Children are appended only for successful decouplings, preserving parent IDs.

```python
report = []
trajectory_parameters = dict(decoupling_parameters, initialization="trajectory")
grains, results, cf_strong_decoupled = decouple_grain_sinograms(
    ds, grains_before_decoupling, cf_4d,
    grain_ids=None,
    twin_pairs=twin_pairs_to_decouple,
    cf_strong=cf_strong_before_decoupling,
    report=report,
    **trajectory_parameters,
)
```

`report` receives dictionaries with `parent_ids`, `child_ids`, `n_components`,
`status`, `component_selection` and `error`. Status is `split`, `unchanged`,
`review` or `failed`. Ambiguous counts and counts at a search limit below four need
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

### Screening performance

The timings below describe the batching optimization before the subsequent
four-track/180-degree screening constraints. They are historical comparisons,
not timings measured with those new constraints.

RANSAC screening batches the tiny, well-conditioned sample-system solves and
reuses angular groups when scoring trials. It keeps the sample order, RNG stream,
trial count, angular validation folds, support checks and selection penalties.
Scoring stops only when the existing score's mathematical upper bound has been
reached; all random samples have already been drawn. Direct solves replace least
squares for these square systems, so roundoff can differ slightly. The final
robust trajectory refinement is unchanged. No notebook settings need changing.

Complete automatic trajectory-count selection, with 600 RANSAC trials and a
search limit of six components, measured the following median times over three
repeated calls in one Python process. These are synthetic irregular-angle scans
with 1200 observations per track, Gaussian dty noise of sigma 0.08, fitting width
0.5 and margin 0.2. Python 3.12, NumPy 1.26.4 and SciPy 1.11.4 were used with
one BLAS thread; global HKL selection, 2D peak loading and reconstruction are
excluded.

| True tracks | Observations | Previous screening | Batched screening | Speedup |
| --- | --- | --- | --- | --- |
| 1 | 1200 | 1.929 s | 0.589 s | 3.27x |
| 2 | 2400 | 5.571 s | 2.703 s | 2.06x |
| 4 | 4800 | 20.295 s | 14.915 s | 1.36x |

These cases retained identical selected counts, peak labels, ambiguity and limit
flags. Centres matched within rtol/atol 1e-8, and candidate scores within
rtol 1e-7 / atol 1e-9. Real-scan timing has not been measured. The benefit depends
on track count, peak count and how much time the final nonlinear fits consume.

Each temporary float scoring matrix is bounded to approximately 1 MiB, or one
observation vector when that is larger. Angular sector/order arrays add roughly
16 bytes per eligible merged 4D observation during one RANSAC fit. No arrays are
added per global 2D peak. The code does not construct an observations-by-all-trials
matrix.

Screening now tests at most four tracks, using the user's minimum of 180 degrees
of support per track, measured by occupied 15-degree sectors modulo 360. This
counts angular support rather than the linear max-minus-min angle, so repeated
turns and the zero-degree boundary are handled consistently. It does not impose
an intensity cutoff: a weak track may still count if its angular support qualifies.
The existing trials, validation folds and width/margin tolerances are retained;
short residual fragments are rejected before nonlinear fits. Explicit component
counts are restricted to two through four. Notebook defaults now use four, and
older settings above four are capped during automatic screening.

### Candidate-only HKL scoring and adaptive RANSAC

Ownership masking now precedes HKL scoring. Each parent scores only its owned
and unassigned candidate rows; known twin pairs score their union. Original
4D row indices/order and the compatibility matrix layout are preserved. A full
mask uses the original direct path to avoid unnecessary column copies. Only
candidate g-vector columns are gathered; no new arrays are added per global 2D
peak. The existing full-length boolean compatibility matrix remains available
for indexed access by the downstream twin initializer.

For optional adaptive count searches, set
`decoupling_options["ransac_confidence"] = 0.999`. The notebook defaults to `None`.
RANSAC draws batches of at most 32 samples and uses the best supported model's
inlier count to update its trial limit. For a minimal sample of size s, the
all-inlier sampling probability is the product of (I-j)/(N-j), j=0..s-1, because
sampling is without replacement. The nominal required trial count is
ceil(log(1-confidence)/log(1-probability)). `trials=600` remains the hard cap;
a low inlier fraction can prevent reaching the target. Well-conditioned sample
checks, 180-degree track support, robust refinement and angular validation remain
in place.

This is a nominal RANSAC sampling target, not a calibrated probability that the
selected grain count is correct; conditioning and model mismatch affect it.
Adaptive stopping consumes fewer random draws, so later hypotheses and fitted
values may differ from fixed-trial mode. `ransac_confidence=None` disables
adaptive stopping and retains the fixed-trial sampling stream. Increasing the
confidence generally requires more samples, up to the existing hard cap.

Repeated synthetic benchmarks (median of three calls, one BLAS thread,
Python 3.12 / NumPy 1.26.4 / SciPy 1.11.4) measured:

| Workload | Previous | Optimized | Speedup |
| --- | --- | --- | --- |
| HKL eligibility, 500000 4D rows / 20 parents, masks included | 0.275 s | 0.047 s | 5.84x |
| Automatic counting, one track / 600 observations | 0.380 s | 0.081 s | 4.69x |
| Automatic counting, two tracks / 1200 observations | 1.342 s | 0.311 s | 4.32x |
| Automatic counting, four tracks / 2400 observations | 5.504 s | 2.703 s | 2.04x |

Counting used irregular full-turn angles, sigma 0.08 dty noise, width 0.5,
margin 0.2, a four-track maximum and a 600-trial cap; optimized counting used
confidence 0.999. Both paths selected the expected counts, with fitted centres
within 0.04 dty units of the synthetic centres. Eligibility returned identical
selected rows. Counting timings exclude eligibility, 2D preparation and
reconstruction; the separate speedup factors should not be multiplied.
Real-scan timing and real-scan detection differences remain unmeasured.

### Preventing broad branches from being counted twice

The former nearest-trajectory distance score could keep improving when extra
sinusoids divided the width of a broad branch. This reproduced four selected
components in synthetic signals containing only two Gaussian-broadened tracks,
with sigma 0.3, 0.5 and 0.8 against a fitting width of 0.5.

Selection now scores a normalized mixture of track densities on held-out angular
sectors. Each track has a training-estimated RMS width and frequency weight;
width is bounded below by the existing robust-fit scale, `width / 2`. Mixture
weights sum to one, so adding curves no longer provides free density. The
penalty counts two position coordinates, one width and the mixture weights,
plus the shared offset when fitted. The per-observation penalty is
`0.5 * parameter_count * log(n) / n`, with `parameter_count = 4*k - 1` plus
one for a fitted offset. The existing one-standard-error rule prefers a smaller
count with comparable predictive quality.

The reproduced two-track signals now select two. Broad one-, three- and
four-track regression signals retain their respective counts, as do unequal-
strength and background-contaminated cases. These checks are synthetic; the
Gaussian profile approximation and count selection still need validation on
real scan data. The old distance scores are retained separately in diagnostics
and can continue to favour four; inspect `selection_scores` for the actual choice.

Screening may still test up to four candidates. Scores are not necessarily
monotonic: a genuine three-track mixture can score worse at two tracks before
improving sharply at three. Consequently a first-failed-improvement stopping
rule is not applied. The maximum remains a search ceiling, not a requested count.
Earlier timing tables predate this selection-metric change.

### Extra threshold regions and exclusion reporting

For manual counts, `spatial_keep_largest=True` keeps the requested number of
largest connected regions by pixel area. This policy was explicitly selected for
the case where parent 5 produced 28 regions with two grains expected. Ties use
region ID; children retain the existing physical-centre ordering. Region areas
are measured before hole filling and dilation. This does not classify every
excluded region as noise: the diagnostic report exposes the selection for review.
Original peak tables are unchanged; only the initial spatial supports are selected.

`result.stats["spatial_components"]` contains the found/expected counts, retained
and excluded region IDs, every region's pixel area and integrated smoothed
reconstruction density, and the total excluded area/density. Reconstructed density
is an image diagnostic, not measured diffraction peak intensity. The initial image
and region labels are available in `result.spatial_component_diagnostics`; they
remain in memory and are not added to the HDF5 export schema.

`report_sinogram_refinement` now prints the region table and displays the initial
smoothed reconstruction, retained-region outlines and a ranked region-area chart.
It respects the existing `plot_refinement_report` and preview limits. These images
cost one reconstruction-sized float image and one integer label image per split,
not arrays indexed by every global 2D peak. The text report still covers all regions
when plotting is disabled.

The iteration-limit warning is independent of component-count selection. Results
retain `converged=False` and their iteration diagnostics when the limit is reached;
this change does not suppress that warning or alter convergence thresholds.
