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
    cf_strong=cf_strong_before_peak_refinement,
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
`centres` and `y0` can be supplied explicitly. A scalar `y0`
sets the common calibrated axis; explicit per-grain arrays remain supported.

Both notebooks expose `refine_all_grain_peaks` and `peak_refinement_options` in
their parameter cells. Indexing trials start directly from the indexed grains
and original peaks (`use_saved_assignments=False` by default in that notebook).
Supply calibrated `y0` in `peak_refinement_options` when available.

The mapping notebook restores fractional assignments, masks and fitted offsets
through `load_saved_peak_assignment` and `prepare_peaks_from_4d`, even when its
rebuilt strong table assigns no peaks to a refined grain. It combines the offsets
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
fractions over active peaks. Convergence also requires `maximum_change < 5 * weight_tolerance`.
These are model-dependent assignments: an incorrect centre or a mask
contaminated by reconstruction streaks can bias the feedback. Adjust
`mask_level`, `dilation` and `support_level` against the observed
grain shapes. Full scans on a uniform increasing dty grid are currently required.

Feedback backprojections and segmentation now cover the full reconstruction.
The former fixed circular ROI could cut through a grain, rejecting its outer
peaks during feedback and introducing an artificial reconstruction boundary.
There is no radius cutoff on the connected support or its dilation. This also
removes the former ROI backprojection acceleration; runtime has not been measured
for this change. Previously saved cropped assignments require refinement from
original peak assignments (`use_saved_assignments=False`, without cropped
`initial_results`) to avoid reusing the affected seeds.

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

### Refinement statistics and graphics

Both notebooks call `utils.report_sinogram_refinement(result)` after joint
refinement.
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

### Sample-space level-set masks

Refinement and its final masks now use morphological Chan–Vese segmentation
over the full reconstruction image. `mask_level` selects an initial connected
core nearest the grain centre; it is no longer the final mask cutoff. After
clipping negative density and Gaussian-smoothing with sigma one pixel, evolve
this seed for 40 steps with one curvature-smoothing operation per step and
equal foreground/background weights. Density is normalized by its image maximum.
The image has a one-pixel zero background border. Distant reconstruction
streaks can enter the region estimates; inspect the selected connected support.

Keep every evolved component overlapping the selected initial connected core,
fill holes and apply `dilation`, without a circular cutoff. Curvature smoothing
can break a narrow grain neck; retaining only the component nearest the centre
would discard the other lobe and reject its peaks on the next feedback pass.
Separate evolved regions that do not overlap the selected seed are excluded.
On subsequent feedback passes, select all threshold components overlapping the
previous spatial support, so a newly disconnected neck cannot reset the seed
to just one lobe. The final mask uses the same continuity rule. Pixels still
need current reconstructed density to enter the threshold seed; previous masks
do not force density or peak assignments to remain present. A missing overlap
raises an error rather than silently switching to another spatial component.
Empty initial or evolved cores raise an error.
The same mask helper serves every refinement iteration and the final masks. Peak compatibility, projection support thresholds, intensity calibration,
conservation and stopping rules are unchanged. Chan–Vese assumes distinguishable
mean foreground/background densities; inspect the masks on real scans.

The implementation uses scikit-image's
[morphological Chan–Vese](https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.morphological_chan_vese).
