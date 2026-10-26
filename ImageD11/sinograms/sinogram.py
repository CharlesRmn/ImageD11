from functools import partial
import copy
import os
from pathlib import Path
import shutil
import tempfile
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter

import h5py
import numba
import numpy as np
from scipy.optimize import least_squares
from tqdm.auto import tqdm
from scipy.ndimage import (
    binary_dilation, binary_fill_holes, gaussian_filter,
    gaussian_filter1d, label as ndi_label,
)

import ImageD11.grain
import ImageD11.cImageD11
import ImageD11.columnfile
import ImageD11.sinograms.dataset
import ImageD11.sinograms.properties
import ImageD11.sinograms.roi_iradon
import ImageD11.sinograms.geometry
from ImageD11.sinograms.geometry import recon_to_sample, sino_shift_and_pad
from ImageD11.refinegrains import lf, polarization

# OPTIONAL structure factor correction for when we build sinograms.
class StructureFactorCorrection:
    """hkl lookup + structure factors used to correct peak intensities when
    building a sinogram. Build once per phase and reuse across grains.

    sf_dense : 3D float array; sf_dense[h - h0, k - k0, l - l0] is the
               structure-factor scale for that reflection (intensities are
               divided by this). Entries where valid is False are ignored.
    valid    : 3D bool array, same shape, True where a reflection is tabulated.
    hkl_min  : length-3 int (h0, k0, l0) mapped to index (0,0,0).
    """
    def __init__(self, sf_dense, valid, hkl_min):
        self.sf_dense = np.asarray(sf_dense, dtype=float)
        self.valid = np.asarray(valid, dtype=bool)
        self.hkl_min = np.asarray(hkl_min, dtype=int)

    def valid_mask(self, hkl):
        shifted = hkl - self.hkl_min[:, np.newaxis]
        shape = np.array(self.valid.shape)
        in_bounds = np.all((shifted >= 0) & (shifted < shape[:, np.newaxis]), axis=0)
        mask = np.zeros(hkl.shape[1], dtype=bool)
        mask[in_bounds] = self.valid[
            shifted[0, in_bounds], shifted[1, in_bounds], shifted[2, in_bounds]
        ]
        return mask

    def sf_values(self, hkl):
        """Structure-factor scale for each peak (shape (N,)); intensities are
        divided by this. hkl (shape (3, N)) must already be filtered via valid_mask."""
        shifted = hkl - self.hkl_min[:, np.newaxis]
        return self.sf_dense[shifted[0], shifted[1], shifted[2]]


class GrainSinogram:
    """Class to build, hold and reconstruct sinograms of grains for Scanning 3DXRD data"""

    def __init__(self, grain_obj, dataset):
        if not isinstance(grain_obj, ImageD11.grain.grain):
            raise TypeError("grain_obj must be an ImageD11.grain.grain instance")
        if not isinstance(dataset, ImageD11.sinograms.dataset.DataSet):
            raise TypeError(
                "dataset must be an ImageD11.sinograms.dataset.DataSet instance"
            )

        self.grain = grain_obj
        self.ds = dataset

        # Peak information to build the sinogram
        self.cf_for_sino = None

        # Sinogram information
        self.sinoangles = None
        self.sino = None
        self.ssino = None  # sorted by angle

        # Reconstruction information
        # self.recons is a dict that stores reconstructions by method
        # i.e. self.recons["iradon"] is the iradon reconstruction
        self.recons = {}
        self.recon_mask = None
        self.recon_pad = None
        self.recon_shift = None
        self.recon_niter = None
        self.recon_y0 = None
        # Optional fractional assignment from peak refinement or spatial/twin splitting.
        self.source_2d_rows = None
        self.peak_fractions = None
        self.spatial_split_mask = None
        self.peak_assignment_kind = None

    def load_saved_peak_assignment(self, grain_label, phase_name=None):
        """Restore saved fractional provenance, spatial mask and y0.

        Returns False when no matching saved assignment exists. No peak geometry
        is computed here.
        """
        assignment = _saved_split_assignment(self, grain_label, phase_name)
        if assignment is None:
            return False
        self.source_2d_rows, self.peak_fractions = assignment
        self.grain.npks_4d = len(np.unique(self.ds.pk2d['spot3d_id'][self.source_2d_rows]))
        return True

    def load_saved_split(self, grain_label, phase_name=None):
        """Compatibility name for restoring split or refined peak assignments."""
        return self.load_saved_peak_assignment(grain_label, phase_name)


    def prepare_peaks_from_2d(self, cf_2d, grain_label, hkltol=0.25, gv=None):
        """cf_2d is an already spatially-corrected 2D columnfile with geometry.
        Keeps only the columns the sinogram build needs, to save memory."""
        if gv is None:
            gv = np.ascontiguousarray(np.array((cf_2d.gx, cf_2d.gy, cf_2d.gz), dtype=float).T)
        labels = np.zeros(cf_2d.nrows, "i")
        drlv2 = np.ones(cf_2d.nrows, "d")
        ImageD11.cImageD11.score_and_assign(
            self.grain.ubi, gv, hkltol, drlv2, labels, grain_label)
        cols = ("dty", "omega", "gx", "gy", "gz", "eta", "sum_intensity", "tth")
        self.cf_for_sino = cf_2d.copyrows_cols(labels == grain_label, cols)
        self.compute_hkl(store=True)   # adds h, k, l; score_and_assign enforced hkltol


    def prepare_peaks_from_weights(self, cf_2d, weights):
        """Prepare fractional 2D contributions, copying only participating rows.

        Intensities always come from the original columnfile. Fractions are
        retained when source_2d_row is available for subsequent HDF5 export.
        """
        weights = np.asarray(weights)
        if weights.shape != (cf_2d.nrows,) or not np.isfinite(weights).all():
            raise ValueError("Expected one finite intensity fraction per 2D peak")
        if np.any(weights < 0) or np.any(weights > 1):
            raise ValueError("Intensity fractions must be in [0, 1]")
        keep = weights > 0
        if keep.sum() < 20:
            raise ValueError("Too few contributing peaks to reconstruct this child")
        cf = cf_2d.copyrows(keep)
        cf.addcolumn(cf.sum_intensity * weights[keep], "sum_intensity")
        self.cf_for_sino = cf
        self.grain.npks = cf.nrows
        _, hkl = self.compute_hkl(store=True)
        # Encode bounded Miller indices so uniqueness uses a 1D integer sort.
        hkl = hkl - hkl.min(axis=1, keepdims=True)
        hkl_ids = np.ravel_multi_index(hkl, hkl.max(axis=1) + 1)
        self.grain.nuniq = len(np.unique(hkl_ids))
        self.source_2d_rows = None
        self.peak_fractions = None
        if "source_2d_row" in cf.titles:
            self.source_2d_rows = cf.source_2d_row.astype(np.int64)
            self.peak_fractions = weights[keep].copy()

    def prepare_peaks_from_4d(self, cf_4d, gord, inds, grain_label, spatial_corrector, hkltol=0.25,
                              phase_name=None, use_saved_split=True):
        """Subset THIS grain's 2D peaks out of ds.pk2d, spatially correct and add
        geometry on that subset only, then compute/store hkl and apply hkl tolerance.
        Never materialises a global cf_2d."""

        # Saved fractional assignments take precedence over orientation-only labels.
        if use_saved_split:
            assignment = _saved_split_assignment(self, grain_label, phase_name)
            if assignment is not None:
                rows, fractions = assignment
                flt = _refine_colfile_from_rows(self.ds, rows, cf_4d.parameters)
                flt.addcolumn(fractions, "split_fraction")
                hkl = np.column_stack((flt.gx, flt.gy, flt.gz)) @ self.grain.ubi.T
                flt.filter(np.linalg.norm(hkl - np.rint(hkl), axis=1) < hkltol)
                flt.addcolumn(flt.sum_intensity * flt.split_fraction, "sum_intensity")
                self.source_2d_rows = flt.source_2d_row.astype(np.int64)
                self.peak_fractions = flt.split_fraction.copy()
                self.cf_for_sino = flt
                self.compute_hkl(store=True)
                return
            
        if "grain_id" not in cf_4d.titles:
            raise ValueError("cf_4d does not contain grain assignments!")
        idx = gord[inds[grain_label + 1] : inds[grain_label + 2]]
        p2d = {p: self.ds.pk2d[p][idx] for p in self.ds.pk2d}
        flt = self.ds.get_colfile_from_peaks_dict(p2d, spatial_corrector)
        flt.parameters = cf_4d.parameters
        flt.updateGeometry()
        self.cf_for_sino = flt
        hkl_real, hkl_int = self.compute_hkl(store=True)
        self.filter_hkltol(hkltol, hkl_real, hkl_int)

    def compute_hkl(self, store=True):
        """Integer hkl for self.cf_for_sino from grain.ubi. Returns (hkl_real, hkl_int),
        both (3, N). If store, writes integer 'h','k','l' columns."""
        cf = self.cf_for_sino
        hkl_real = np.dot(self.grain.ubi, (cf.gx, cf.gy, cf.gz))
        hkl_int = np.round(hkl_real).astype(int)
        if store:
            cf.addcolumn(hkl_int[0], "h")
            cf.addcolumn(hkl_int[1], "k")
            cf.addcolumn(hkl_int[2], "l")
        return hkl_real, hkl_int

    def filter_hkltol(self, hkltol=0.25, hkl_real=None, hkl_int=None):
        """Drop peaks whose g-vector is > hkltol from an integer hkl."""
        cf = self.cf_for_sino
        if hkl_real is None:
            hkl_real, hkl_int = self.compute_hkl(store=False)
        dh2 = ((hkl_real - hkl_int) ** 2).sum(axis=0)
        cf.filter(dh2 < hkltol * hkltol)

    def filter_valid_hkls(self, sf_correction):
        """Drop peaks whose integer hkl is not tabulated in sf_correction.
        Needs h,k,l columns (prepare_peaks_* / compute_hkl provides them)."""
        cf = self.cf_for_sino
        cf.filter(sf_correction.valid_mask(np.array((cf.h, cf.k, cf.l))))

    def compute_corrected_intensity(self, out_column="corr_sum_intensity",
                                    in_column="sum_intensity",
                                    sf_correction=None, apply_lf_pol=True):
        """Write a corrected-intensity column and return its name.
        correction = lf/pol (needs tth,eta) and/or 1/structure-factor (needs every hkl
        tabulated — run filter_valid_hkls first)."""
        cf = self.cf_for_sino
        corr = np.ones(cf.nrows, "d")
        if apply_lf_pol:
            corr *= lf(cf.tth, cf.eta)
            corr /= polarization(cf.tth, cf.eta)
        if sf_correction is not None:
            corr /= sf_correction.sf_values(np.array((cf.h, cf.k, cf.l)))
        cf.addcolumn(cf[in_column] * corr, out_column)
        return out_column
    
    def build_sinogram(self, columns=("omega",), intensity_column="sum_intensity",
                       normalise=True):
        """Build the sinogram from self.cf_for_sino. Assumes preparation is done:
        'h','k','l' columns exist, any hkl/validity filtering already applied, and the
        signal lives in intensity_column. normalise divides each projection row by its max
        (original default); pass False to keep a corrected intensity scale."""
        cf = self.cf_for_sino
        for a in columns:
            assert a in cf.titles
        assert "omega" in columns
        assert intensity_column in cf.titles
        for c in "hkl":
            assert c in cf.titles, "call compute_hkl() / prepare_peaks_* first"
    
        hkl = np.array((cf.h, cf.k, cf.l))
        NY = len(self.ds.ybincens)
        iy = np.round((cf.dty - self.ds.ybincens[0])
                      / (self.ds.ybincens[1] - self.ds.ybincens[0])).astype(int)
        etasigns = np.sign(cf.eta)
    
        hklmin = hkl.min(axis=1)
        dh = hkl - hklmin[:, np.newaxis]
        de = (etasigns.astype(int) + 1) // 2
        pkmsk = np.zeros(list(dh.max(axis=1) + 1) + [2], int)
        pkmsk[dh[0], dh[1], dh[2], de] = 1
        pkrow = np.cumsum(pkmsk.ravel()).reshape(pkmsk.shape) - 1
        pkhkle = np.arange(np.prod(pkmsk.shape), dtype=int)[pkmsk.flat == 1]
        pkindices = np.array(np.unravel_index(pkhkle, pkmsk.shape))
        pkindices[:3] += hklmin[:, np.newaxis]
        npks = pkmsk.sum()
        destRow = pkrow[dh[0], dh[1], dh[2], de]
        sino = np.zeros((npks, NY), "f")
        hits = np.zeros((npks, NY), "f")
        adr = (destRow * NY + iy).astype("q")
    
        sig = cf[intensity_column].astype("f")
        ImageD11.cImageD11.put_incr64(sino, adr, sig)
        ImageD11.cImageD11.put_incr64(hits, adr, np.ones(len(de), dtype="f"))
    
        angs = {}
        for name in columns:
            if name == intensity_column:
                continue
            angs[name] = np.zeros((npks, NY), "f")
            ImageD11.cImageD11.put_incr64(angs[name], adr, (cf[name] * sig).astype("f"))
        sinoangles = angs["omega"].sum(axis=1) / sino.sum(axis=1)
    
        if normalise:
            self.proj_scale = sino.max(axis=1)
            self.sino = sino / self.proj_scale[:, np.newaxis]
        else:
            self.proj_scale = np.ones(npks, "f")
            self.sino = sino
    
        order = np.lexsort((np.arange(npks), sinoangles))
        self.sinoangles = sinoangles[order]
        self.ssino = self.sino[order].T
        self.proj_scale = self.proj_scale[order]
        if len(columns) > 1:
            self.angle_wt_sinos = {
                name: angs[name][order].T for name in columns if name != intensity_column}
            if intensity_column in columns:
                self.angle_wt_sinos[intensity_column] = sino[order].T
        self.hits = hits
        self.hkle = pkindices[:, order]


    def update_lab_position_from_peaks(self, cf_4d, grain_label, y0_guess, phase_name=None):
        """Updates translation of self.grain using peaks in assigned 4D colfile.
        Also updates self.recon_y0 with centre"""
        # Spatial split centres and y0 are already saved. Orientation-only
        # assignment in tomo_2_map may give the appended child zero 4D peaks.
        if self.load_saved_split(grain_label, phase_name):
            return
        mask_4d = cf_4d.grain_id == grain_label
        omega = cf_4d.omega[mask_4d]
        dty = cf_4d.dty[mask_4d]
        sx, sy, y0 = ImageD11.sinograms.geometry.sx_sy_y0_from_dty_omega(dty, omega, y0_guess)
        self.grain.translation = np.array([sx, sy, 0])
        self.recon_y0 = y0

    def update_lab_position_from_recon(self, method="iradon"):
        """Updates translation of self.grain by finding centre-of-mass of reconstruction
        Only really valid for very small grains.
        Does not update translation if no fitting result found."""
        fitting_result = ImageD11.sinograms.geometry.fit_sample_position_from_recon(
            self.recons[method], self.ds.ystep
        )
        if fitting_result is None:
            # indicate that the grain has a dodgy reconstruction
            # useful for filtering out bad grains
            self.bad_recon = True
        else:
            x, y = fitting_result
            self.grain.translation = np.array([x, y, 0])

    def correct_halfmask(self):
        """Applies halfmask correction to sinogram"""
        self.ssino = ImageD11.sinograms.roi_iradon.apply_halfmask_to_sino(
            self.ssino, real_mask=getattr(self.ds, "ybin_real_mask", None))

    def correct_ring_current(self, is_half_scan=False, min_ring_current_frac=0.5):
        """Corrects each row of the sinogram to the ring current of the corresponding scan"""

        # ignore ring current values below min_ring_current
        mean_ring_current = np.mean(self.ds.ring_currents_per_scan_scaled)
        min_ring_current = min_ring_current_frac * np.max(
            self.ds.ring_currents_per_scan_scaled
        )
        ring_current_to_use = np.where(
            self.ds.ring_currents_per_scan_scaled < min_ring_current,
            mean_ring_current,
            self.ds.ring_currents_per_scan_scaled,
        )

        if is_half_scan:
            correction = ring_current_to_use
            addition_length = len(self.ds.ybincens) - len(ring_current_to_use)
            correction_halfmask_addition = np.zeros(addition_length)
            correction_halfmask_addition.fill(correction.max())
            correction = np.concatenate((correction, correction_halfmask_addition))
        else:
            correction = ring_current_to_use
        self.ssino = self.ssino / correction[:, None]

    def update_recon_parameters(
        self, pad=None, shift=None, mask=None, niter=None, y0=None
    ):
        """Update some or all of the reconstruction parameters in one go"""

        if pad is not None:
            self.recon_pad = pad
        if shift is not None:
            self.recon_shift = shift
        if mask is not None:
            self.recon_mask = mask
        if niter is not None:
            self.recon_niter = niter
        if y0 is not None:
            self.recon_y0 = y0

    def recon(self, method="iradon", workers=1, projections=None, **extra_args):
        """Performs reconstruction given reconstruction method

        method = "iradon", "mlem", "astra"
        workers = threads to use - passed to the method

        projections = which projections to use for reconstruction using a subset of data (twins)
                iradon( self.sino[ :, projections ], theta = self.sinoangles[ projections ], ... )
                effectively a sinogram mask - perhaps this should move to iradon code.

        extra_args = passed to the projection function

           iradon -> ImageD11.sinograms.roi_iradon.run_iradon
           { pad=20, shift=0, workers=1, mask=None,
                      apply_halfmask=False,
                      mask_central_zingers=False,
                      central_mask_radius=25,
                      filter_name='hamming'}
           mlem : ImageD11.sinograms.roi_iradon.run_mlem
              { mask=None, pad=20, shift=0, workers=1, niter=20, apply_halfmask=False,
             mask_central_zingers=False, central_mask_radius=25 }
           astra : run_astra in this file
               { shift=0, pad=0, mask=None, niter=100, astra_method='SIRT_CUDA', workers=None }
        """
        if method not in ["iradon", "mlem", "astra"]:
            raise ValueError("Unsupported method!")
        if self.ssino is None or self.sinoangles is None:
            raise ValueError(
                "Sorted sino or sinoangles are missing/have not been computed, unable to reconstruct."
            )
        if projections is None:
            sino = self.ssino
            angles = self.sinoangles
        else:
            sino = self.ssino[:, projections]
            angles = self.sinoangles[projections]
            print("Using subset", sino.shape, "from", self.ssino.shape)
        if method == "iradon":
            recon_function = ImageD11.sinograms.roi_iradon.run_iradon
        elif method == "mlem":
            # MLEM has niter as an extra argument
            recon_function = ImageD11.sinograms.roi_iradon.run_mlem
            # Overwrite the default argument if self.recon_niter is set
            if self.recon_niter is not None and "niter" not in extra_args:
                extra_args["niter"] = self.recon_niter
            # TODO: We should think about default reconstruction arguments in more detail
            # At the moment, we could pass None for pad or shift etc to recon_function if they are not manually set, which seems dangerous
            # Do we check for None at the start of this function?
            # Or do we initialise them to sensible default values inside __init__?
            
        elif method == "astra":
            recon_function = run_astra

        # These two are properties of the sinogram and of the finished
        # reconstruction, not of the backend, so handle them here rather than
        # inside each run_* function. Previously only iradon and mlem accepted
        # them, and passing them with method="astra" raised TypeError.
        apply_halfmask = extra_args.pop("apply_halfmask", False)
        mask_central_zingers = extra_args.pop("mask_central_zingers", False)
        central_mask_radius = extra_args.pop("central_mask_radius", 25)

        if apply_halfmask:
            axis_row = None
            if self.recon_shift is not None:
                axis_row = sino.shape[0] // 2 - self.recon_shift
            sino = ImageD11.sinograms.roi_iradon.apply_halfmask_to_sino(
                sino, axis_row=axis_row,
                real_mask=getattr(self.ds, "ybin_real_mask", None))
        
        recon = recon_function(
            sino=sino,
            angles=angles,
            pad=self.recon_pad,
            shift=self.recon_shift,
            workers=workers,
            mask=self.recon_mask,
            **extra_args
        )

        if mask_central_zingers:
            recon = ImageD11.sinograms.roi_iradon.correct_recon_central_zingers(
                recon, radius=central_mask_radius)

        self.recons[method] = recon
        return recon

    def get_shape_mask(self, method="iradon", cutoff=None):
        """Gets a boolean mask representing the grain shape from the grain reconstruction.
        Performs a binary threshold of the reconstruction.
        Uses manual threshold value if cutoff value provided.
        Performs global Otsu threshold if no value provided."""
        image = self.recons[method]
        if cutoff is None:
            threshold = threshold_otsu(image)
        else:
            threshold = cutoff
        binary_image = image > threshold
        return binary_image

    def mask_central_zingers(self, method="iradon", radius=25):
        self.recons[method] = (
            ImageD11.sinograms.roi_iradon.correct_recon_central_zingers(
                self.recons[method], radius=radius
            )
        )

    def to_h5py_group(self, parent_group, group_name):
        """Creates a H5Py group for this GrainSinogram.
        parent_group is the parent H5py Group
        group_name is the name of the H5py Group for this name
        Very useful for saving lists of GrainSinograms to an H5 file"""

        # create a group for this specific GrainSinogram
        grain_group = parent_group.require_group(group_name)

        # save peak information
        # peak_info_group = grain_group.require_group("peak_info")
        # for peak_info_attr in ["etasigns_2d_strong", "hkl_2d_strong"]:
        #     peak_info_var = getattr(self, peak_info_attr)
        #     if peak_info_var is not None:
        #         save_array(peak_info_group, peak_info_attr, peak_info_var)

        if self.source_2d_rows is not None:
            assignment = grain_group.require_group("split_peak_assignment")
            save_array(assignment, "source_2d_row", self.source_2d_rows)
            save_array(assignment, "intensity_fraction", self.peak_fractions)
            grain_group.attrs["peak_assignment_kind"] = self.peak_assignment_kind or "refined"
        if self.spatial_split_mask is not None:
            save_array(grain_group, "spatial_split_mask", self.spatial_split_mask)

        # save sinograms
        sinogram_group = grain_group.require_group("sinograms")

        for sino_attr in ["sino", "ssino", "sinoangles"]:
            sino_var = getattr(self, sino_attr)
            if sino_var is not None:
                save_array(sinogram_group, sino_attr, sino_var)

        # save reconstruction parameters as attributes

        recon_par_group = grain_group.require_group("recon_parameters")

        for recon_par_attr in ["recon_pad", "recon_shift", "recon_niter", "recon_y0"]:
            recon_par_var = getattr(self, recon_par_attr)
            if recon_par_var is not None:
                recon_par_group.attrs[recon_par_attr] = recon_par_var

        if self.recon_mask is not None:
            save_array(recon_par_group, "recon_mask", self.recon_mask)

        # save reconstructions

        recon_group = grain_group.require_group("recons")

        for recon_attr in self.recons.keys():
            save_array(recon_group, recon_attr, self.recons[recon_attr])

        return grain_group

    @classmethod
    def from_h5py_group(cls, group, ds, grain):
        """Creates a GrainSinogram object from an h5py group, dataset and grain object"""
        grainsino_obj = GrainSinogram(grain_obj=grain, dataset=ds)

        if "sinograms" in group.keys():
            for sino_attr in group["sinograms"].keys():
                sino_var = group["sinograms"].get(sino_attr)[:]
                setattr(grainsino_obj, sino_attr, sino_var)

        if "recon_parameters" in group.keys():
            for recon_par_attr in [
                "recon_pad",
                "recon_shift",
                "recon_niter",
                "recon_y0",
            ]:
                if group["recon_parameters"].attrs.get(recon_par_attr) is not None:
                    recon_par_var = group["recon_parameters"].attrs.get(recon_par_attr)[
                        ()
                    ]
                    setattr(grainsino_obj, recon_par_attr, recon_par_var)

            if "recon_mask" in group["recon_parameters"]:
                grainsino_obj.recon_mask = group["recon_parameters"]["recon_mask"][:]

        if "recons" in group.keys():
            for recon_attr in group["recons"].keys():
                recon_var = group["recons"].get(recon_attr)[:]
                grainsino_obj.recons[recon_attr] = recon_var

        if "split_peak_assignment" in group:
            assignment = group["split_peak_assignment"]
            grainsino_obj.source_2d_rows = assignment["source_2d_row"][:]
            grainsino_obj.peak_fractions = assignment["intensity_fraction"][:]
            grainsino_obj.peak_assignment_kind = group.attrs.get("peak_assignment_kind", "split")
        if "spatial_split_mask" in group:
            grainsino_obj.spatial_split_mask = group["spatial_split_mask"][:]
        return grainsino_obj


from scipy.ndimage import shift as ndi_shift

def _astra_geoms(n, pad, angles):
    """Volume + projection geometry matching roi_iradon's conventions."""
    import astra
    N = n + pad                                    # reconstruction size, as in run_iradon
    (pb, pa), _ = ImageD11.sinograms.roi_iradon._sinogram_pad(n, N)   # same padding iradon uses internally
    L = n + pb + pa
    if L % 2 == 0:                                 # odd detector -> centre on integer index
        pa += 1
        L += 1
    # roi_iradon puts the origin on integer index N//2; ASTRA centres the volume on
    # the origin, i.e. (N-1)/2. Offset the window by half a pixel when N is even.
    off = 0.5 if N % 2 == 0 else 0.0
    vol_geom = astra.create_vol_geom(N, N,
                                     -N / 2.0 - off, N / 2.0 - off,
                                     -N / 2.0 + off, N / 2.0 + off)
    proj_geom = astra.create_proj_geom("parallel", 1.0, L, angles)
    return vol_geom, proj_geom, (pb, pa)

def shift_sinogram(sino, shift_amount, order=1):
    """
    Shift a sinogram along dty by a possibly non-integer
    amount, equivalent to astra.functions.geom_postalignment(proj_geom, shift).
    """
    if shift_amount == 0:
        return sino
    return ndi_shift(
        sino,
        shift=(shift_amount, 0),   # shift along dty axis only, not angles
        order=order,               # linear spline interpolation for sub-pixel accuracy
        mode="constant",
        cval=0.0,
    )

def run_astra(
    sino,
    angles,
    shift=0,
    pad=0,
    mask=None,
    niter=100,
    astra_method="SIRT_CUDA",
    workers=None,
    min_constraint=0,
    max_constraint=None,
):
    import astra
    angles = np.radians(angles)
    allowed_methods = [
        "BP", "FBP", "SIRT",
        "BP_CUDA", "FBP_CUDA", "SIRT_CUDA", "SART_CUDA", "CGLS_CUDA", "EM_CUDA",
    ]
    if astra_method not in allowed_methods:
        raise ValueError("Unsupported method!")

    manual_mask = None
    if astra_method in ["EM_CUDA", "FBP", "FBP_CUDA", "BP_CUDA"] and mask is not None:
        manual_mask = mask.copy()
        mask = None

    vol_geom, proj_geom, (pb, pa) = _astra_geoms(sino.shape[0], pad, angles)
    sino = np.pad(sino, ((pb, pa), (0, 0)), mode="constant")

    if shift != 0:
        if astra_method == "FBP":       # FBP rejects vec geometries
            sino = shift_sinogram(sino, shift, order=1)
        else:
            proj_geom = astra.functions.geom_postalignment(proj_geom, shift)
    
    proj_id = astra.create_projector("linear", proj_geom, vol_geom)
    proj_data_id = astra.data2d.create("-sino", proj_geom, data=sino.T)
    if astra_method == "EM_CUDA":
        # initialise with ones
        rec_id = astra.data2d.create("-vol", vol_geom, 1.0)
    else:
        rec_id = astra.data2d.create("-vol", vol_geom, 0.0)

    cfg = astra.creators.astra_dict(astra_method)

    if "CUDA" not in astra_method:
        cfg["ProjectorId"] = proj_id
    cfg["ProjectionDataId"] = proj_data_id
    cfg["ReconstructionDataId"] = rec_id
    cfg["option"] = {}
    if astra_method not in ["EM_CUDA", "FBP"]:
        # These used to be hard-coded to [0, 1], which silently saturated the
        # reconstruction whenever the sinogram was not normalised to a peak of 1
        # (build_sinogram(normalise=False), correct_ring_current). Non-negativity
        # is kept by default; the upper clamp is now opt-in.
        if min_constraint is not None:
            cfg["option"]["MinConstraint"] = min_constraint
        if max_constraint is not None:
            cfg["option"]["MaxConstraint"] = max_constraint

    if mask is not None:
        mask_id = astra.data2d.create("-vol", vol_geom, mask)
        cfg["option"]["ReconstructionMaskId"] = mask_id

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, iterations=niter)
    recon = astra.data2d.get(rec_id)

    # Clean up.
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.projector.delete(proj_id)
    astra.data2d.delete(proj_data_id)

    if mask is not None:
        astra.data2d.delete(mask_id)

    if manual_mask is not None:
        # manually mask
        recon = np.where(manual_mask, recon, 0.0)

    return recon


def write_h5(filename, list_of_sinos, overwrite_grains=False, group_name="grains"):
    """Write list of GrainSinogram objects to H5Py file
    If overwrite_grains is True, will replace grains in group_name"""
    with h5py.File(filename, "a") as hout:
        grains_group = hout.require_group(group_name)

        for gsinc, gs in enumerate(list_of_sinos):
            group_name = str(gsinc)
            gs.to_h5py_group(parent_group=grains_group, group_name=group_name)
            if overwrite_grains:
                gs.grain.to_h5py_group(parent_group=grains_group, group_name=group_name)


def read_h5(filename, ds, group_name="grains"):
    """Read list of GrainSinogram objects from H5Py file
    Will also create self.grain objects from the H5Py file
    Because GrainSinogram objects can't exist without corresponding grain objects"""
    with h5py.File(filename, "r") as hin:
        grains_group = hin[group_name]
        gs_objects = []

        # take all the keys in the grains group, sort them by integer value, iterate
        for gid_string in sorted(grains_group.keys(), key=lambda x: int(x)):
            grain_group = grains_group[gid_string]

            # import the grain object
            g = ImageD11.grain.grain.from_h5py_group(grain_group)

            # create the GrainSinogram object
            gs = GrainSinogram.from_h5py_group(grain_group, ds, g)

            gs_objects.append(gs)

    return gs_objects


def write_slice_recon(filename, slice_arrays):
    rgb_x_array, rgb_y_array, rgb_z_array, grain_labels_array, raw_intensity_array = (
        slice_arrays
    )
    with h5py.File(filename, "a") as hout:
        slice_group = hout.require_group("slice_recon")
        save_array(slice_group, "intensity", raw_intensity_array)
        save_array(slice_group, "labels", grain_labels_array)

        save_array(slice_group, "ipf_x_col_map", rgb_x_array).attrs["CLASS"] = "IMAGE"
        save_array(slice_group, "ipf_y_col_map", rgb_y_array).attrs["CLASS"] = "IMAGE"
        save_array(slice_group, "ipf_z_col_map", rgb_z_array).attrs["CLASS"] = "IMAGE"


def write_pbp_strain(filename, pbp_strain_slice_arrays):
    (
        ubi_map,
        eps_map,
        misorientation_map,
        ipf_x_col_map,
        ipf_y_col_map,
        ipf_z_col_map,
    ) = pbp_strain_slice_arrays
    with h5py.File(filename, "a") as hout:
        slice_group = hout.require_group("pbp_strain_recon")
        save_array(slice_group, "ubi", ubi_map)
        save_array(slice_group, "eps", eps_map)
        save_array(slice_group, "misorientation", misorientation_map)

        save_array(slice_group, "ipf_x_col_map", ipf_x_col_map).attrs["CLASS"] = "IMAGE"
        save_array(slice_group, "ipf_y_col_map", ipf_y_col_map).attrs["CLASS"] = "IMAGE"
        save_array(slice_group, "ipf_z_col_map", ipf_z_col_map).attrs["CLASS"] = "IMAGE"


def read_slice_recon(filename):
    with h5py.File(filename, "r") as hin:
        slice_group = hin["slice_recon"]
        intensity = slice_group["intensity"][:]
        labels = slice_group["labels"][:]
        ipf_x = slice_group["ipf_x_col_map"][:]
        ipf_y = slice_group["ipf_y_col_map"][:]
        ipf_z = slice_group["ipf_z_col_map"][:]

    return ipf_x, ipf_y, ipf_z, labels, intensity


def build_slice_arrays(
    grainsinos, cutoff_level=0.0, method="iradon", grain_labels=None
):
    """Build grain maps from individual grain reonstructions
    Optionally provide a different list of grain labels to label the grains"""

    if grain_labels is not None:
        assert len(grainsinos) == len(grain_labels)

    first_recon = grainsinos[0].recons[method]
    grain_labels_array = np.zeros_like(first_recon) - 1

    redx = np.zeros_like(first_recon)
    grnx = np.zeros_like(first_recon)
    blux = np.zeros_like(first_recon)

    redy = np.zeros_like(first_recon)
    grny = np.zeros_like(first_recon)
    bluy = np.zeros_like(first_recon)

    redz = np.zeros_like(first_recon)
    grnz = np.zeros_like(first_recon)
    bluz = np.zeros_like(first_recon)

    raw_intensity_array = np.zeros_like(first_recon)

    raw_intensity_array.fill(cutoff_level)

    def norm(r):
        m = r > r.max() * 0.2
        return (r / r[m].mean()).clip(0, 1)

    for i, gs in enumerate(grainsinos):

        if grain_labels is not None:
            label = grain_labels[i]
        else:
            label = i

        g_raw_intensity = norm(gs.recons[method])

        g_raw_intensity_mask = g_raw_intensity > raw_intensity_array

        g_raw_intensity_map = g_raw_intensity[g_raw_intensity_mask]

        raw_intensity_array[g_raw_intensity_mask] = g_raw_intensity_map

        redx[g_raw_intensity_mask] = g_raw_intensity_map * gs.grain.rgb_x[0]
        grnx[g_raw_intensity_mask] = g_raw_intensity_map * gs.grain.rgb_x[1]
        blux[g_raw_intensity_mask] = g_raw_intensity_map * gs.grain.rgb_x[2]

        redy[g_raw_intensity_mask] = g_raw_intensity_map * gs.grain.rgb_y[0]
        grny[g_raw_intensity_mask] = g_raw_intensity_map * gs.grain.rgb_y[1]
        bluy[g_raw_intensity_mask] = g_raw_intensity_map * gs.grain.rgb_y[2]

        redz[g_raw_intensity_mask] = g_raw_intensity_map * gs.grain.rgb_z[0]
        grnz[g_raw_intensity_mask] = g_raw_intensity_map * gs.grain.rgb_z[1]
        bluz[g_raw_intensity_mask] = g_raw_intensity_map * gs.grain.rgb_z[2]

        grain_labels_array[g_raw_intensity_mask] = label

    raw_intensity_array[raw_intensity_array == cutoff_level] = 0

    # redx has shape (i, j)
    # (redx, grnx, blux) has shape (3, i, j)
    # transpose changes this to (i, j, 3) needed for mpl imshow
    # crucially: (i, j) unaffected

    rgb_x_array = np.transpose((redx, grnx, blux), axes=(1, 2, 0))
    rgb_y_array = np.transpose((redy, grny, bluy), axes=(1, 2, 0))
    rgb_z_array = np.transpose((redz, grnz, bluz), axes=(1, 2, 0))

    return (
        rgb_x_array,
        rgb_y_array,
        rgb_z_array,
        grain_labels_array,
        raw_intensity_array,
    )


from skimage.filters import threshold_otsu
from skimage.filters import threshold_otsu, threshold_local, threshold_yen, threshold_li
from skimage.morphology import disk, white_tophat, remove_small_holes
from skimage.segmentation import watershed
from skimage.measure import label


def norm(r):
    m = r > r.max() * 0.2
    return (r / r[m].mean()).clip(0, 1)


def filter_normalise_recon(recon):
    """Function to filter and normalise reconstructions"""
    # determine threshold of grain vs background
    thresh = threshold_yen(recon)

    # determine binary mask of grain
    binary = recon > thresh

    # this still leaves some hot pixels behind
    footprint = disk(8)
    res = white_tophat(binary, footprint)
    binary_clean = binary ^ res

    # binary_clean might contain holes
    binary_clean_noholes = remove_small_holes(binary_clean)

    # fig, ax = plt.subplots()
    # ax.imshow(binary_clean_noholes, origin="lower")
    # plt.show()

    # determine masked recon image
    recon_masked = np.where(binary_clean_noholes, recon, 0)

    # to normalise, we first need to segment into contiguous blobs
    # normalise each of them separetely
    # label image regions
    label_image = label(binary_clean_noholes)

    recon_masked_normed = np.zeros_like(recon)

    for region_id in np.unique(label_image.ravel())[1:]:
        recon_this_region_only = np.where(label_image == region_id, recon_masked, 0)
        recon_this_region_only_normed = norm(recon_this_region_only)
        recon_masked_normed = np.where(
            label_image == region_id, recon_this_region_only_normed, recon_masked_normed
        )

    return recon_masked_normed


def build_slice_arrays_clean(grainsinos, method="iradon", grain_labels=None):
    """Build grain maps from individual grain reonstructions
    Cleans and normalises reconstructions before building slice arrays
    Saves cleaned and normalised reconstruction to gs.recons[method_norm]
    Optionally provide a different list of grain labels to label the grains"""

    if grain_labels is not None:
        assert len(grainsinos) == len(grain_labels)

    first_recon = grainsinos[0].recons[method]
    grain_labels_array = np.zeros_like(first_recon) - 1

    redx = np.zeros_like(first_recon)
    grnx = np.zeros_like(first_recon)
    blux = np.zeros_like(first_recon)

    redy = np.zeros_like(first_recon)
    grny = np.zeros_like(first_recon)
    bluy = np.zeros_like(first_recon)

    redz = np.zeros_like(first_recon)
    grnz = np.zeros_like(first_recon)
    bluz = np.zeros_like(first_recon)

    raw_intensity_array = np.zeros_like(first_recon)

    for i, gs in enumerate(grainsinos):

        if grain_labels is not None:
            label = grain_labels[i]
        else:
            label = i

        # the grain should win where its intensity is greater than what's there before
        g_raw_intensity_map = filter_normalise_recon(gs.recons[method])
        gs.recons[method + "_norm"] = g_raw_intensity_map

        g_raw_intensity_mask = g_raw_intensity_map > raw_intensity_array

        raw_intensity_array = np.where(
            g_raw_intensity_mask, g_raw_intensity_map, raw_intensity_array
        )

        redx = np.where(
            g_raw_intensity_mask, g_raw_intensity_map * gs.grain.rgb_x[0], redx
        )
        redy = np.where(
            g_raw_intensity_mask, g_raw_intensity_map * gs.grain.rgb_y[0], redy
        )
        redz = np.where(
            g_raw_intensity_mask, g_raw_intensity_map * gs.grain.rgb_z[0], redz
        )

        grnx = np.where(
            g_raw_intensity_mask, g_raw_intensity_map * gs.grain.rgb_x[1], grnx
        )
        grny = np.where(
            g_raw_intensity_mask, g_raw_intensity_map * gs.grain.rgb_y[1], grny
        )
        grnz = np.where(
            g_raw_intensity_mask, g_raw_intensity_map * gs.grain.rgb_z[1], grnz
        )

        blux = np.where(
            g_raw_intensity_mask, g_raw_intensity_map * gs.grain.rgb_x[2], blux
        )
        bluy = np.where(
            g_raw_intensity_mask, g_raw_intensity_map * gs.grain.rgb_y[2], bluy
        )
        bluz = np.where(
            g_raw_intensity_mask, g_raw_intensity_map * gs.grain.rgb_z[2], bluz
        )

        grain_labels_array[g_raw_intensity_mask] = label

    # redx has shape (i, j)
    # (redx, grnx, blux) has shape (3, i, j)
    # transpose changes this to (i, j, 3) needed for mpl imshow
    # crucially: (i, j) unaffected

    rgb_x_array = np.transpose((redx, grnx, blux), axes=(1, 2, 0))
    rgb_y_array = np.transpose((redy, grny, bluy), axes=(1, 2, 0))
    rgb_z_array = np.transpose((redz, grnz, bluz), axes=(1, 2, 0))

    return (
        rgb_x_array,
        rgb_y_array,
        rgb_z_array,
        grain_labels_array,
        raw_intensity_array,
    )


# TODO: Use Silx Nexus IO instead?
# This is a temporary sensible middle ground!
def save_array(grp, name, ary):
    # TODO: Move this helper function somewhere else
    cmp = {"compression": "gzip", "compression_opts": 2, "shuffle": True}

    # Refinement can change peak counts, projection counts or reconstruction size.
    if name in grp:
        previous = grp[name]
        if previous.shape != ary.shape or not np.can_cast(ary.dtype, previous.dtype):
            attributes = dict(previous.attrs)
            del grp[name]
            hds = grp.create_dataset(name, data=ary, **cmp)
            hds.attrs.update(attributes)
            return hds
    hds = grp.require_dataset(name, shape=ary.shape, dtype=ary.dtype, **cmp)
    hds[:] = ary
    return hds

# TODO go to some sort of Numba zone or C code?
@numba.njit(parallel=True)
def pmax(ary):
    """Find the min/max of an array in parallel"""
    mx = ary.flat[0]
    mn = ary.flat[0]
    for i in numba.prange(1, ary.size):
        mx = max(ary.flat[i], mx)
        mn = min(ary.flat[i], mn)
    return mn, mx


@numba.njit(parallel=True)
def palloc(shape, dtype):
    """Allocate and fill an array with zeros in parallel"""
    ary = np.empty(shape, dtype=dtype)
    for i in numba.prange(ary.size):
        ary.flat[i] = 0
    return ary


# counting sort by grain_id
@numba.njit
def counting_sort(ary, maxval=None, minval=None):
    """Radix sort for integer array. Single threaded. O(n)
    Numpy should be doing this...
    """
    if maxval is None:
        assert minval is None
        minval, maxval = pmax(ary)  # find with a first pass
    maxval = int(maxval)
    minval = int(minval)
    histogram = palloc((maxval - minval + 1,), np.int64)
    indices = palloc((maxval - minval + 2,), np.int64)
    result = palloc(ary.shape, np.int64)
    for gid in ary:
        histogram[gid - minval] += 1
    indices[0] = 0
    for i in range(len(histogram)):
        indices[i + 1] = indices[i] + histogram[i]
    i = 0
    for gid in ary:
        j = gid - minval
        result[indices[j]] = i
        indices[j] += 1
        i += 1
    return result, histogram


@numba.njit(parallel=True)
def find_grain_id(spot3d_id, grain_id, spot2d_label, grain_label, order, nthreads=20):
    """
    Assignment grain labels into the peaks 2d array
    spot3d_id = the 3d spot labels that are merged and indexed
    grain_id = the grains assigned to the 3D merged peaks
    spot2d_label = the 3d label for each 2d peak
    grain_label => output, which grain is this peak
    order = the order to traverse spot2d_label sorted
    """
    assert spot3d_id.shape == grain_id.shape
    assert spot2d_label.shape == grain_label.shape
    assert spot2d_label.shape == order.shape
    T = nthreads
    # print("Using", T, "threads")
    for tid in numba.prange(T):
        pcf = 0  # thread local I hope?
        for i in order[tid::T]:
            grain_label[i] = -1
            pkid = spot2d_label[i]
            while pcf < spot3d_id.size and spot3d_id[pcf] < pkid:
                pcf += 1
            if pcf < spot3d_id.size and spot3d_id[pcf] == pkid:
                grain_label[i] = grain_id[pcf]


def get_2d_peaks_from_4d_peaks(p2d, cf, return_labels=False):
    """inds tells you, for each grain, where its 2D spots live in the sorted order.
    return_labels also returns grain_2d_id: one int per 2D peak giving its grain,
    a compact selector into p2d you can store once and reuse."""
    assert (np.argsort(cf.spot3d_id) == np.arange(cf.nrows)).all()
    numba_order, numba_histo = counting_sort(p2d["spot3d_id"])
    grain_2d_id = palloc(p2d["spot3d_id"].shape, np.dtype(int))
    cleanid = cf.grain_id.copy()
    find_grain_id(cf.spot3d_id, cleanid, p2d["spot3d_id"], grain_2d_id, numba_order)
    # prepare_peaks_from_4d indexes grain g at g+1: reserve the -1 slot
    # even when every measured peak has a nonnegative grain assignment.
    largest = max(0, int(grain_2d_id.max())) if grain_2d_id.size else 0
    gord, counts = counting_sort(grain_2d_id, maxval=largest, minval=-1)
    inds = np.concatenate(((0,), np.cumsum(counts)))
    if return_labels:
        return gord, inds, grain_2d_id
    return gord, inds


# Grain peak refinement, spatial decoupling and twin-aware peak sharing.
def _refine_group_rows(values):
    """Sort once and reuse contiguous angle/reflection groups during feedback."""
    order = np.argsort(values, kind="stable")
    boundaries = np.r_[0, 1 + np.flatnonzero(np.diff(values[order]))  , len(order)]
    return order, boundaries


@numba.njit
def _refine_project_kernel(
    sx,
    sy,
    density,
    omega,
    dty,
    order,
    boundaries,
    y0,
    ystep,
    threshold,
    gaussian,
    padding,
):
    """Splat sparse pixels and interpolate measurements, grouped by angle.

    A serial kernel avoids launching a large thread pool for small grain ROIs.
    Both output arrays are indexed by the original peak rows.
    """
    profile = np.zeros(len(omega), dtype=np.float64)
    support = np.zeros(len(omega), dtype=np.bool_)
    for group in range(len(boundaries) - 1):
        begin, end = boundaries[group], boundaries[group + 1]
        if begin == end:
            continue
        angle = omega[order[begin]] * np.pi / 180.0
        sinomega, cosomega = np.sin(angle), np.cos(angle)
        positions = (-sx * sinomega - sy * cosomega) / ystep
        lower = np.floor(positions).astype(np.int64)
        offset = lower.min() - padding
        size = lower.max() - offset + padding + 2
        hist = np.zeros(size, dtype=np.float64)
        for pixel in range(len(density)):
            index = lower[pixel] - offset
            fraction = positions[pixel] - lower[pixel]
            hist[index] += density[pixel] * (1.0 - fraction)
            hist[index + 1] += density[pixel] * fraction
        if len(gaussian) > 1:
            filtered = np.zeros(size, dtype=np.float64)
            radius = len(gaussian) // 2
            for index in range(size):
                for tap in range(len(gaussian)):
                    # Match scipy.ndimage's half-sample symmetric 'reflect' mode.
                    source = (index + tap - radius) % (2 * size)
                    if source >= size:
                        source = 2 * size - source - 1
                    filtered[index] += hist[source] * gaussian[tap]
            hist = filtered
        cutoff = threshold * hist.max()
        # Interpolate on the physical grid, including exact boundary samples.
        # Converting dty back to steps can round a boundary just outside the ROI.
        grid = y0 + (offset + np.arange(size)) * ystep
        rows = order[begin:end]
        values = np.interp(dty[rows], grid, hist)
        for index in range(len(rows)):
            row = rows[index]
            if dty[row] < grid[0] or dty[row] > grid[-1]:
                continue
            profile[row] = values[index]
            support[row] = values[index] >= cutoff
    return profile, support


def _refine_mask_geometry(shape, centre, ystep, radius):
    ri = np.arange(shape[0])[:, None]
    rj = np.arange(shape[1])[None, :]
    sx, sy = recon_to_sample(ri, rj, shape, ystep)
    distances = (sx - centre[0]) ** 2 + (sy - centre[1]) ** 2
    return distances <= radius**2, distances


def _refine_reconstruction_mask(roi, shift, ny):
    """Restrict feedback backprojection without changing boundary interpolation."""
    # gaussian_filter(..., 1) has a four-pixel radius on each axis.
    mask = binary_dilation(roi, structure=np.ones((9, 9), dtype=bool))
    if abs(shift) >= 1:
        # Native iradon zero-pads projection shifts. Larger shifts can make
        # its interpolation grid nonmonotonic at the detector boundaries,
        # where np.interp's result depends on the sequence of queried pixels.
        # Restrict acceleration to pixels projecting strictly inside those
        # boundaries, or retain the original full-image query sequence.
        ri, rj = np.nonzero(mask)
        centre = roi.shape[0] // 2
        distance = (ri - centre) ** 2 + (rj - centre) ** 2
        if distance.size and distance.max() >= ((ny - 1) // 2) ** 2:
            return None
    return mask


def _refine_design(omega):
    a = np.deg2rad(omega)
    return np.column_stack((-np.sin(a), -np.cos(a)))


def _refine_predict(omega, centres, y0):
    return _refine_design(omega) @ np.asarray(centres).T + y0


def _decouple_classify(omega, dty, centres, y0, width, margin):
    distances = np.abs(np.asarray(dty)[:, None] - _refine_predict(omega, centres, y0))
    nearest = distances.argmin(axis=1)
    ordered = np.sort(distances, axis=1)
    separation = ordered[:, 1] - ordered[:, 0] if len(centres) > 1 else np.full(len(dty), np.inf)
    confident = (ordered[:, 0] <= width) & (separation >= margin)
    return (np.where(confident, nearest, -1), distances)


def _decouple_ransac_one_manual(omega, dty, width, rng, trials, fixed_y0=None):
    """Legacy scalar, fixed-trial initializer for the opt-in trajectory mode."""
    X = _refine_design(omega)
    A = np.column_stack((X, np.ones(len(X)))) if fixed_y0 is None else X
    target = dty if fixed_y0 is None else dty - fixed_y0
    npars = A.shape[1]
    if len(target) < max(8, npars):
        raise ValueError("Too few residual peaks for the second trajectory")
    best = None
    best_score = -np.inf
    for _ in range(trials):
        ix = rng.choice(len(target), npars, replace=False)
        if np.linalg.cond(A[ix]) > 50:
            continue
        beta = np.linalg.lstsq(A[ix], target[ix], rcond=None)[0]
        residual = np.abs(np.dot(A, beta) - target)
        inside = residual < width
        bins = np.floor(np.mod(omega[inside], 360) / 15).astype(int)
        score = inside.sum() * min(len(np.unique(bins)), 12) / 12
        if score > best_score:
            best, best_score = (beta, score)
    if best is None:
        raise RuntimeError("RANSAC failed; check angular coverage")
    inside = np.abs(np.dot(A, best) - target) < width
    fit = least_squares(
        lambda b: np.dot(A[inside], b) - target[inside],
        best,
        loss="soft_l1",
        f_scale=width / 2,
    )
    beta = fit.x
    y0 = beta[2] if fixed_y0 is None else fixed_y0
    return (beta[:2], y0)


def _decouple_ransac_one(omega, dty, width, rng, trials, fixed_y0=None, min_angle_bins=0, confidence=None):
    """Fit a trajectory with optional nominal-confidence adaptive sampling."""
    if confidence is not None and not 0 < confidence < 1:
        raise ValueError("RANSAC confidence must be between zero and one, or None")
    X = _refine_design(omega)
    A = np.column_stack((X, np.ones(len(X)))) if fixed_y0 is None else X
    target = dty if fixed_y0 is None else dty - fixed_y0
    npars = A.shape[1]
    if len(target) < max(8, npars):
        raise ValueError("Too few residual peaks for the second trajectory")
    sectors = np.floor(np.mod(omega, 360) / 15).astype(int)
    order = np.argsort(sectors)
    boundaries = np.r_[0, np.flatnonzero(np.diff(sectors[order])) + 1]
    if len(boundaries) < min_angle_bins:
        raise RuntimeError("Residual signal has insufficient angular coverage")
    # Bound each scoring float matrix to 1 MiB (or one observation vector).
    block = max(1, min(64, 131072 // len(target)))
    best = None
    best_score = -np.inf
    best_hits = 0
    upper_score = len(target) * min(len(boundaries), 12) / 12.0
    drawn, limit = 0, trials
    while drawn < limit:
        # Fixed mode consumes the historical full RNG stream. Adaptive mode
        # draws small batches so successful tracks avoid further sampling.
        batch = trials if confidence is None else min(32, limit - drawn)
        samples = np.asarray([rng.choice(len(target), npars, replace=False)
                              for _ in range(batch)], dtype=int).reshape(-1, npars)
        drawn += batch
        matrices = A[samples]
        good = np.linalg.cond(matrices) <= 50
        betas = np.linalg.solve(matrices[good], target[samples[good]][:, :, None])[:, :, 0]
        for start in range(0, len(betas), block):
            trial_betas = betas[start:start + block]
            inside = np.abs(np.dot(A, trial_betas.T) - target[:, None]) < width
            coverage = np.count_nonzero(
                np.logical_or.reduceat(inside[order], boundaries, axis=0), axis=0)
            hits = inside.sum(axis=0)
            scores = hits * np.minimum(coverage, 12) / 12.0
            scores[coverage < min_angle_bins] = -np.inf
            winner = int(np.argmax(scores))
            if scores[winner] > best_score:
                best = trial_betas[winner]
                best_score = scores[winner]
                best_hits = int(hits[winner])
            if best_score == upper_score:
                break
        if confidence is None:
            break
        if best is not None and best_hits >= npars:
            # Exact probability of drawing an all-inlier minimal sample
            # without replacement; confidence is the nominal RANSAC target.
            offsets = np.arange(npars, dtype=float)
            probability = np.prod((best_hits - offsets) / (len(target) - offsets))
            if probability >= 1:
                limit = drawn
            elif probability > 0:
                needed = int(np.ceil(np.log1p(-confidence) / np.log1p(-probability)))
                limit = min(trials, max(drawn, needed))
    if best is None:
        raise RuntimeError("RANSAC failed; check angular coverage")
    inside = np.abs(A @ best - target) < width
    fit = least_squares(
        lambda b: A[inside] @ b - target[inside],
        best,
        loss="soft_l1",
        f_scale=width / 2,
    )
    beta = fit.x
    y0 = beta[2] if fixed_y0 is None else fixed_y0
    return (beta[:2], y0)


def _decouple_fit_components(omega, dty, width, margin, fixed_y0=None, trials=600, seed=42, n_components=2, min_angle_bins=0, confidence=None):
    omega, dty = (np.asarray(omega), np.asarray(dty))
    if width <= 0 or margin < 0 or len(omega) != len(dty):
        raise ValueError("Invalid trajectory fitting inputs")
    if not np.isfinite(omega).all() or not np.isfinite(dty).all():
        raise ValueError("Trajectory fitting requires finite observations")
    X = _refine_design(omega)
    manual = min_angle_bins == 0 and confidence is None
    if manual and hasattr(np.random, "Generator"):
        # The pre-automation manual initializer used the PCG64 stream.
        rng = np.random.Generator(np.random.PCG64(seed))
    else:
        rng = np.random.RandomState(seed)

    def fit_one(angles, values, offset):
        if manual:
            return _decouple_ransac_one_manual(angles, values, width, rng, trials, offset)
        return _decouple_ransac_one(
            angles, values, width, rng, trials, offset, min_angle_bins, confidence)

    candidates = []
    angular_bins = np.floor(np.mod(omega, 360) / 15).astype(int)

    def supported(labels):
        return all(np.sum(labels == k) >= 8
                   and len(np.unique(angular_bins[labels == k])) >= min_angle_bins
                   for k in range(n_components))

    for _ in range(8):
        c1, offset = fit_one(omega, dty, fixed_y0)
        centres = [c1]
        for component in range(1, n_components):
            distances = np.abs(dty[:, None] - _refine_predict(omega, centres, offset))
            remaining = distances.min(axis=1) > width
            try:
                centre, _ = fit_one(omega[remaining], dty[remaining], offset)
            except (ValueError, RuntimeError):
                break
            centres.append(centre)
        if len(centres) != n_components:
            continue
        centres = np.asarray(centres)
        previous = None
        for iteration in range(30):
            labels, distances = _decouple_classify(
                omega, dty, centres, offset, width, margin
            )
            keep = labels >= 0
            if not supported(labels):
                break
            if previous is not None and np.array_equal(labels, previous):
                break
            previous = labels.copy()
            initial = (
                centres.ravel()
                if fixed_y0 is not None
                else np.r_[centres.ravel(), offset]
            )

            def residual(p):
                cc = p[:2 * n_components].reshape(n_components, 2)
                yy = fixed_y0 if fixed_y0 is not None else p[2 * n_components]
                return np.sum(X[keep] * cc[labels[keep]], axis=1) + yy - dty[keep]

            result = least_squares(residual, initial, loss="soft_l1", f_scale=width / 2)
            centres = result.x[:2 * n_components].reshape(n_components, 2)
            offset = fixed_y0 if fixed_y0 is not None else result.x[2 * n_components]
        labels, distances = _decouple_classify(
            omega, dty, centres, offset, width, margin
        )
        if not supported(labels):
            continue
        score = np.mean(np.minimum(distances.min(axis=1) ** 2, width**2))
        candidates.append((score, centres.copy(), float(offset), labels))
    if not candidates:
        raise RuntimeError(
            "Requested component count is not supported by the peaks. Inspect branches and adjust WIDTH/Y0."
        )
    _, centres, offset, labels = min(candidates, key=lambda item: item[0])
    order = np.lexsort((centres[:, 1], centres[:, 0]))
    centres = centres[order]
    labels, _ = _decouple_classify(omega, dty, centres, offset, width, margin)
    return (centres, offset, labels)


def _decouple_validation_losses(train_omega, train_dty, omega, dty,
                                centres, y0, width):
    """Score held-out peaks using normalized, width-aware track densities.

    Training nearest-track assignments estimate component weights and RMS
    widths. The scale floor is the existing robust fit scale, width / 2.
    Return likelihood loss and the legacy capped-distance loss separately.
    """
    train_residual = train_dty[:, None] - _refine_predict(train_omega, centres, y0)
    nearest = np.abs(train_residual).argmin(axis=1)
    residual = dty[:, None] - _refine_predict(omega, centres, y0)
    log_density = np.full(residual.shape, -np.inf)
    for k in range(len(centres)):
        values = train_residual[nearest == k, k]
        if not len(values):
            continue
        scale = max(width / 2.0, np.sqrt(np.mean(values ** 2)))
        weight = float(len(values)) / len(train_dty)
        log_density[:, k] = (np.log(weight) - np.log(scale * np.sqrt(2 * np.pi))
                             - 0.5 * (residual[:, k] / scale) ** 2)
    likelihood = -np.logaddexp.reduce(log_density, axis=1)
    distance = np.minimum(np.abs(residual).min(axis=1) / width, 1.0) ** 2
    return likelihood, distance


def _decouple_auto_components(omega, dty, width, margin, fixed_y0=None,
                             trials=600, seed=42, max_components=4, confidence=0.999):
    """Select trajectories by blocked angular prediction and a one-SE rule.

    Selection uses normalized Gaussian mixture likelihoods with branch widths
    learned on training angles. Legacy capped-distance scores remain in diagnostics.
    Each fold holds out 30-degree sectors, including all repeated turns at those
    angles. Each full-data track needs 180 degrees of support (12 occupied
    15-degree sectors); each two-thirds training fold needs eight sectors.
    Returned diagnostics describe candidate counts, not a probability of grains.
    """
    if confidence is not None and not 0 < confidence < 1:
        raise ValueError("RANSAC confidence must be between zero and one, or None")
    if (isinstance(max_components, (bool, np.bool_))
            or not isinstance(max_components, (int, np.integer)) or max_components < 1):
        raise ValueError("max_components must be a positive integer")
    max_components = min(max_components, 4)
    omega, dty = np.asarray(omega), np.asarray(dty)
    if (omega.ndim != 1 or dty.shape != omega.shape or len(omega) < 24
            or width <= 0 or margin < 0
            or not np.isfinite(omega).all() or not np.isfinite(dty).all()):
        raise ValueError("Automatic counting needs at least 24 finite omega/dty pairs")
    sectors = np.floor(np.mod(omega, 360) / 30).astype(int)
    occupied = np.unique(sectors)
    if len(np.unique(np.floor(np.mod(omega, 360) / 15).astype(int))) < 12:
        raise ValueError("Automatic counting needs at least 180 degrees of support (12 occupied 15-degree sectors)")
    # Assign occupied angular blocks, rather than observations, to validation folds.
    fold_of_sector = np.zeros(12, dtype=int)
    fold_of_sector[occupied] = np.arange(len(occupied)) % 3
    folds = fold_of_sector[sectors]
    fine_sectors = np.floor(np.mod(omega, 360) / 15).astype(int)
    fits, losses, legacy_losses, counts, penalties, legacy_penalties = [], [], [], [], [], []

    def supported(labels, angles, count, minimum_bins):
        return all(np.sum(labels == k) >= 8
                   and len(np.unique(angles[labels == k])) >= minimum_bins
                   for k in range(count))

    for count in range(1, min(max_components, len(omega) // 8) + 1):
        try:
            fit = _decouple_fit_components(
                omega, dty, width, margin, fixed_y0, trials, seed, count, min_angle_bins=12, confidence=confidence)
            if not supported(fit[2], fine_sectors, count, 12):
                raise RuntimeError("Insufficient angular support for fitted tracks")
            heldout = np.empty(len(omega))
            legacy_heldout = np.empty(len(omega))
            for fold in range(3):
                train = folds != fold
                centres, offset, labels = _decouple_fit_components(
                    omega[train], dty[train], width, margin, fixed_y0,
                    trials, seed + fold + 1, count, min_angle_bins=8, confidence=confidence)
                if not supported(labels, fine_sectors[train], count, 8):
                    raise RuntimeError("Insufficient angular support in a training fold")
                heldout[~train], legacy_heldout[~train] = _decouple_validation_losses(
                    omega[train], dty[train], omega[~train], dty[~train],
                    centres, offset, width)
        except (ValueError, RuntimeError):
            if count == 1:
                raise RuntimeError("Automatic counting could not validate one trajectory; check WIDTH/Y0 and angular coverage")
            break
        fits.append(fit)
        counts.append(count)
        losses.append(np.array([heldout[sectors == b].mean() for b in occupied]))
        legacy_losses.append(np.array([legacy_heldout[sectors == b].mean() for b in occupied]))
        # Two centre coordinates and one width per track, count - 1 weights,
        # and the shared rotation-axis offset when it is not fixed.
        parameters = 4 * count - 1 + int(fixed_y0 is None)
        penalties.append(0.5 * parameters * np.log(len(omega)) / len(omega))
        legacy_parameters = 2 * count + int(fixed_y0 is None)
        legacy_penalties.append(legacy_parameters * np.log(len(omega)) / len(omega))
    losses = np.asarray(losses)
    scores = losses.mean(axis=1) + penalties
    best = int(np.argmin(scores))
    standard_error = losses[best].std(ddof=1) / np.sqrt(len(occupied))
    chosen = int(np.flatnonzero(scores <= scores[best] + standard_error)[0])
    legacy_losses = np.asarray(legacy_losses)
    legacy_scores = legacy_losses.mean(axis=1) + legacy_penalties
    legacy_best = int(np.argmin(legacy_scores))
    diagnostics = dict(
        counts=counts, scores=legacy_scores.tolist(), selected=counts[chosen],
        standard_error=float(legacy_losses[legacy_best].std(ddof=1) / np.sqrt(len(occupied))),
        selection_scores=scores.tolist(), selection_standard_error=float(standard_error),
        selection_method="normalized Gaussian mixture likelihood", minimum_angle_coverage=180,
        ransac_confidence=confidence,
        ambiguous=int(np.sum(scores <= scores[best] + standard_error)) > 1,
        reached_limit=counts[chosen] == max_components and max_components < 4,
    )
    centres, offset, labels = fits[chosen]
    return centres, offset, labels, diagnostics


def _decouple_fit_two(omega, dty, width, margin, fixed_y0=None, trials=600, seed=42):
    return _decouple_fit_components(omega, dty, width, margin, fixed_y0, trials, seed)


def _decouple_select_full_branches(
    omega,
    dty,
    centres,
    y0,
    step,
    fit_width,
    margin,
    manual_width=None,
    angle_bin=10,
    density_fraction=0.08,
):
    """Learn connected density envelopes in resolved angles; reject overlap."""
    from scipy.ndimage import gaussian_filter1d

    omega, dty = (np.asarray(omega), np.asarray(dty))
    if step <= 0 or not 0 < angle_bin <= 180 or (not 0 < density_fraction < 1):
        raise ValueError("Check step, ANGLE_BIN_2D and DENSITY_FRACTION_2D")
    trajectories = _refine_predict(omega, centres, y0)
    residuals = dty[:, None] - trajectories
    n_components = len(centres)
    nearest = np.abs(residuals).argmin(axis=1)
    separation = np.full(len(omega), np.inf)
    nearest_trajectory = trajectories[np.arange(len(omega)), nearest]
    for k in range(n_components):
        distance = np.abs(trajectories[:, k] - nearest_trajectory)
        distance[nearest == k] = np.inf
        np.minimum(separation, distance, out=separation)
    resolved = separation > max(6 * step, 4 * fit_width)
    nbins = int(np.ceil(360 / angle_bin))
    bin_centres = (np.arange(nbins) + 0.5) * 360 / nbins
    bin_ids = np.floor(np.mod(omega, 360) * nbins / 360).astype(int)
    widths = np.full((nbins, n_components, 2), np.nan)
    if manual_width is None:
        search = (max(6 * step, 0.45 * np.max(separation)) if n_components > 1
                  else max(6 * step, 2 * fit_width, np.percentile(np.abs(residuals), 95)))
        edges = np.arange(-search, search + step, step)
        mids = (edges[:-1] + edges[1:]) / 2
        for b in range(nbins):
            for k in range(n_components):
                m = (bin_ids == b) & resolved & (nearest == k)
                if m.sum() < 15:
                    continue
                values = residuals[m, k]
                hist = gaussian_filter1d(
                    np.histogram(values, edges)[0].astype(float), 1
                )
                near = np.flatnonzero(np.abs(mids) <= max(fit_width, 2 * step))
                peak = near[np.argmax(hist[near])]
                if hist[peak] <= 0:
                    continue
                threshold = max(1.0, density_fraction * hist[peak])
                lo = hi = peak
                while lo > 0 and hist[lo - 1] >= threshold:
                    lo -= 1
                while hi < len(hist) - 1 and hist[hi + 1] >= threshold:
                    hi += 1
                widths[b, k] = [max(2 * step, -edges[lo]), max(2 * step, edges[hi + 1])]
        for k in range(n_components):
            valid = np.isfinite(widths[:, k, 0])
            if valid.sum() < 3:
                raise RuntimeError(
                    "Too few resolved angle bins to learn envelopes for child {}: "
                    "{} usable bins (need at least 3), {} resolved 2D peaks. "
                    "Maximum sampled branch separation is {:.3g} dty steps; "
                    "automatic estimation requires more than {:.3g} steps. "
                    "Supply width_2d (branch half-width in dty units), or increase "
                    "angle_bin if resolved peaks are spread too sparsely. "
                    "Increasing angle_bin cannot resolve overlapping branches.".format(
                        k,
                        int(valid.sum()),
                        int(np.sum(resolved & (nearest == k))),
                        separation.max() / step,
                        max(6 * step, 4 * fit_width) / step,
                    )
                )
            for side in (0, 1):
                widths[:, k, side] = np.interp(
                    bin_centres, bin_centres[valid], widths[valid, k, side], period=360
                )
    else:
        if manual_width <= 0:
            raise ValueError("WIDTH_2D must be positive")
        widths[:] = manual_width
    per_peak = np.empty((len(omega), n_components, 2))
    for k in range(n_components):
        for side in (0, 1):
            per_peak[:, k, side] = np.interp(
                np.mod(omega, 360), bin_centres, widths[:, k, side], period=360
            )
    supports = (residuals >= -per_peak[:, :, 0]) & (residuals <= per_peak[:, :, 1])
    labels = np.full(len(omega), -1, dtype=int)
    exclusive = supports.sum(axis=1) == 1
    ordered = np.sort(np.abs(residuals), axis=1)
    safe = exclusive if n_components == 1 else exclusive & (ordered[:, 1] - ordered[:, 0] >= margin)
    labels[safe] = supports[safe].argmax(axis=1)
    return (labels, np.abs(residuals), bin_centres, widths)


def _refine_rebuild_sinogram(
    gs, cf2, weights, y0, shift, pad, filter_name="hann", recon_mask=None
):
    gs.prepare_peaks_from_weights(cf2, weights)
    gs.build_sinogram()
    gs.update_recon_parameters(y0=y0, shift=shift, pad=pad)
    # Assign directly: update_recon_parameters(mask=None) preserves the old mask.
    gs.recon_mask = recon_mask
    gs.recon(method="iradon", workers=1, filter_name=filter_name)
    gs.recon_mask = np.ones(gs.recons["iradon"].shape, dtype=bool)
    return gs


def _decouple_rebuild_split_sinogram(*args, **kwargs):
    """Compatibility wrapper for reconstruction from fractional peak weights."""
    return _refine_rebuild_sinogram(*args, **kwargs)


def _refine_connected_grain_mask(
    image,
    centre,
    ystep,
    radius,
    level=0.15,
    dilation=2,
    geometry=None,
    smoothed_image=None,
):
    if geometry is None:
        geometry = _refine_mask_geometry(image.shape, centre, ystep, radius)
    roi, distances = geometry
    if smoothed_image is None:
        smoothed_image = gaussian_filter(np.maximum(image, 0), 1)
    if not roi.any() or smoothed_image[roi].max() <= 0:
        raise RuntimeError(
            "No reconstructed core inside ROI; inspect centre and radius"
        )
    components, count = ndi_label(
        roi & (smoothed_image >= level * smoothed_image[roi].max())
    )
    if count == 0:
        raise RuntimeError("No connected core; lower mask_level")
    nearest_label = components.flat[
        np.argmin(np.where(components > 0, distances, np.inf))
    ]
    mask = binary_fill_holes(components == nearest_label)
    if dilation:
        mask = binary_dilation(mask, iterations=dilation)
    return mask & roi


def _refine_project_grain_profile(
    mask,
    image,
    omega,
    dty,
    y0,
    ystep,
    threshold=0.02,
    beam_sigma_steps=0.75,
    groups=None,
    smoothed_image=None,
):
    """Forward-project a grain ROI with a conserved, angle-independent mass.

    ``groups`` and ``smoothed_image`` can be reused from the feedback loop.
    The Gaussian kernel has the same radius and boundary convention as
    scipy.ndimage.gaussian_filter1d with its default truncate=4.
    """
    if ystep <= 0 or not 0 < threshold < 1 or beam_sigma_steps < 0:
        raise ValueError("Invalid projection controls")
    ri, rj = np.nonzero(mask)
    if len(ri) == 0:
        raise ValueError("Cannot project an empty mask")
    sx, sy = recon_to_sample(ri, rj, mask.shape, ystep)
    if smoothed_image is None:
        smoothed_image = gaussian_filter(np.maximum(image, 0), 1)
    density = smoothed_image[ri, rj]
    mass = density.sum()
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("No positive density inside the mask")
    if groups is None:
        groups = _refine_group_rows(np.asarray(omega))
    if beam_sigma_steps > 0:
        radius = int(4 * beam_sigma_steps + 0.5)
        taps = np.arange(-radius, radius + 1)
        gaussian = np.exp(-0.5 * (taps / beam_sigma_steps) ** 2)
        gaussian /= gaussian.sum()
    else:
        gaussian = np.ones(1)
    return _refine_project_kernel(
        sx,
        sy,
        density / mass,
        omega,
        dty,
        groups[0],
        groups[1],
        y0,
        ystep,
        threshold,
        gaussian,
        max(3, int(np.ceil(4 * beam_sigma_steps))),
    )


def _refine_shared_peak_weights(
    profiles, supports, intensity, reflection_ids, minimum=5, groups=None
):
    """Calibrate on exclusive reflections without rescanning the entire table.

    Returns fractional weights, predicted total intensity and calibration flags.
    """
    exclusive = supports.sum(axis=1) == 1
    scores = np.zeros_like(profiles)
    calibrated = np.zeros_like(supports)
    if groups is None:
        groups = [
            _refine_group_rows(
                reflection_ids[:, k] if reflection_ids.ndim == 2 else reflection_ids
            )
            for k in range(profiles.shape[1])
        ]
    for k in range(profiles.shape[1]):
        cutoff = 0.1 * profiles[:, k].max()
        trusted = (
            exclusive & supports[:, k] & (profiles[:, k] > cutoff) & (intensity > 0)
        )
        ratios = np.divide(
            intensity,
            profiles[:, k],
            out=np.zeros_like(intensity, dtype=float),
            where=trusted,
        )
        fallback = np.median(ratios[trusted]) if trusted.any() else 1.0
        order, boundaries = groups[k]
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            rows = order[begin:end]
            usable = rows[trusted[rows]]
            if len(usable) >= minimum:
                amplitude = np.median(ratios[usable])
                calibrated[rows, k] = True
            else:
                amplitude = fallback
            scores[rows, k] = profiles[rows, k] * amplitude * supports[rows, k]
    # No contributing grain has independent evidence for this shared reflection.
    unknown = (supports.sum(axis=1) > 1) & ~(calibrated & supports).any(axis=1)
    scores[unknown] = profiles[unknown] * supports[unknown]
    total = scores.sum(axis=1)
    weights = np.divide(
        scores, total[:, None], out=np.zeros_like(scores), where=total[:, None] > 0
    )
    return weights, total, calibrated


def _decouple_shared_peak_weights(*args, **kwargs):
    """Compatibility wrapper for the common multi-grain intensity partition."""
    return _refine_shared_peak_weights(*args, **kwargs)


@dataclass
class GrainRefinementResult:
    """Jointly refined peaks and reconstructions without creating new grains.

    Columns of peak_weights correspond to child_ids, which retain the original
    grain-list indices. source_2d_rows indexes ds.pk2d. Weights sum to one for
    retained peaks when all grains are refined, and zero for rejected peaks.
    intensity_budget excludes fixed contributions belonging to unselected
    grains. y0 contains one value per grain.
    Shared contributions are model estimates; UBIs are not refined here.
    """

    grainsinos: list
    centres: np.ndarray
    y0: np.ndarray
    source_2d_rows: np.ndarray
    peak_weights: np.ndarray
    intensity_budget: np.ndarray
    history: list
    converged: bool
    parent_ids: tuple
    child_ids: tuple
    stats: dict = field(default_factory=dict)


def _refine_peak_statistics(weights, intensity, budget):
    """Count measured peaks separately from per-grain contributions."""
    totals = weights.sum(axis=1)
    contributions = (weights > 0).sum(axis=1)
    assigned_intensity = float(np.sum(intensity * totals))
    available_intensity = float(np.sum(intensity * budget))
    original_intensity = float(intensity.sum())
    return dict(
        candidates=len(weights),
        retained=int((contributions > 0).sum()),
        rejected=int((contributions == 0).sum()),
        exclusive=int((contributions == 1).sum()),
        shared=int((contributions > 1).sum()),
        counts=(weights > 0).sum(axis=0).astype(int).tolist(),
        grain_intensities=np.asarray(intensity @ weights).tolist(),
        assigned_intensity=assigned_intensity,
        available_intensity=available_intensity,
        retained_intensity=(
            assigned_intensity / original_intensity if original_intensity else 0.0
        ),
        retained_available_intensity=(
            assigned_intensity / available_intensity if available_intensity else 0.0
        ),
    )


def _refine_assignment_changes(
    weights, old_active, old_owners, old_fractions, tolerance
):
    """Report gains, losses and significant changes of the dominant grain."""
    active = weights.sum(axis=1) > 0
    owners = weights.argmax(axis=1)
    previous_contribution = weights[np.arange(len(weights)), old_owners]
    return dict(
        added=int((~old_active & active).sum()),
        removed=int((old_active & ~active).sum()),
        reassigned=int(
            (
                old_active
                & active
                & (old_owners != owners)
                & (np.abs(previous_contribution - old_fractions) > tolerance)
            ).sum()
        ),
    )


def _refine_initial_results(grains, results):
    """Validate in-memory assignments against the current grain list."""
    provided = {}
    for result in results:
        if len(result.child_ids) != len(result.grainsinos):
            raise ValueError("Initial results must have one grain ID per sinogram")
        for gid, gs in zip(result.child_ids, result.grainsinos):
            if (
                not isinstance(gid, (int, np.integer))
                or isinstance(gid, bool)
                or gid in provided
                or gid < 0
                or gid >= len(grains)
            ):
                raise ValueError(
                    "Initial results must refer to unique existing grain IDs"
                )
            if (
                gs.source_2d_rows is None
                or gs.peak_fractions is None
                or not np.allclose(gs.grain.ubi, grains[gid].ubi)
            ):
                raise ValueError(
                    "Initial result does not match its current grain or peak assignments"
                )
            if grains[gid].translation is not None and (
                gs.grain.translation is None
                or not np.allclose(gs.grain.translation, grains[gid].translation)
            ):
                raise ValueError(
                    "Initial result centre does not match its current grain"
                )
            provided[gid] = gs
    return provided


def _refine_seed_assignments(
    ds,
    grains,
    ids,
    cf_4d,
    cf_strong,
    hkl_tol,
    phase_name,
    use_saved_assignments,
    provided,
):
    """Seed selected grains and reserve saved intensity of unselected grains."""
    if len(np.unique(cf_4d.spot3d_id)) != cf_4d.nrows:
        raise ValueError("cf_4d must contain unique spot3d_id values")
    owners = _refine_peak_owners(grains, cf_4d, cf_strong, hkl_tol)
    saved = [None] * len(ids)
    for k, gid in enumerate(ids):
        gs = provided.get(gid)
        if gs is None:
            gs = GrainSinogram(copy.deepcopy(grains[gid]), ds)
            found = use_saved_assignments and gs.load_saved_peak_assignment(
                gid, phase_name
            )
        else:
            found = True
        if found:
            saved[k] = gs
    fixed = []
    for gid in set(range(len(grains))) - set(ids):
        gs = provided.get(gid)
        if gs is None:
            gs = GrainSinogram(copy.deepcopy(grains[gid]), ds)
            found = gs.load_saved_peak_assignment(gid, phase_name)
        else:
            found = True
        if found:
            fixed.append(gs)
    pool = cf_4d.spot3d_id[np.isin(owners, ids) | (owners < 0)]
    rows = np.flatnonzero(np.isin(ds.pk2d["spot3d_id"], pool))
    extra = [gs.source_2d_rows for gs in saved if gs is not None]
    if extra:
        rows = np.unique(np.concatenate([rows] + extra))
    cf2 = _refine_colfile_from_rows(ds, rows, cf_4d.parameters)
    budget = np.ones(cf2.nrows)
    for gs in fixed:
        present = np.isin(gs.source_2d_rows, rows)
        np.add.at(
            budget,
            np.searchsorted(rows, gs.source_2d_rows[present]),
            -gs.peak_fractions[present],
        )
    if np.any(budget < -1e-8):
        raise ValueError("Saved unselected grains overallocate measured peak intensity")
    budget = np.clip(budget, 0, 1)
    spot_order = np.argsort(cf_4d.spot3d_id)
    spot_ids = cf_4d.spot3d_id[spot_order]
    mapped = np.isin(cf2.spot3d_id, spot_ids)
    row_owners = np.full(cf2.nrows, -1, dtype=np.int64)
    row_owners[mapped] = owners[
        spot_order[np.searchsorted(spot_ids, cf2.spot3d_id[mapped])]
    ]
    weights = np.column_stack([row_owners == gid for gid in ids]).astype(float)
    if extra:
        claimed = np.isin(rows, np.concatenate(extra))
        weights[claimed] = 0
        for k, gs in enumerate(saved):
            if gs is not None:
                weights[np.searchsorted(rows, gs.source_2d_rows), k] = gs.peak_fractions
    return cf2, weights, budget, saved, owners


def _refine_seed_geometry(originals, ids, cf_4d, owners, saved, centres, y0, step):
    """Initialize grain positions using one automatically estimated scan axis."""
    n = len(ids)
    supplied_centres = centres is not None
    centres = (
        np.asarray(centres, dtype=float).copy()
        if supplied_centres
        else np.empty((n, 2), dtype=float)
    )
    if centres.shape != (n, 2):
        raise ValueError("centres must have shape (n_grains, 2)")
    y0s = (
        np.broadcast_to(np.asarray(y0, dtype=float), (n,)).copy()
        if y0 is not None
        else np.empty(n, dtype=float)
    )
    fitted_centres = []
    for k, original in enumerate(originals):
        owned = owners == ids[k]
        finite = owned & np.isfinite(cf_4d.omega) & np.isfinite(cf_4d.dty)
        omega, dty = cf_4d.omega[finite], cf_4d.dty[finite]
        gs = saved[k]
        translation = original.translation
        if not supplied_centres:
            if translation is not None and np.isfinite(translation[:2]).all():
                centres[k] = translation[:2]
            else:
                X = _refine_design(omega)
                A = np.column_stack((X, np.ones(len(X)))) if y0 is None else X
                if len(A) < A.shape[1] or np.linalg.matrix_rank(A) < A.shape[1]:
                    raise ValueError(
                        "Grain {} needs angular coverage or supplied centres/y0".format(
                            ids[k]
                        )
                    )
                target = dty if y0 is None else dty - y0s[k]
                initial = np.linalg.lstsq(A, target, rcond=None)[0]
                fit = least_squares(
                    lambda p: A @ p - target,
                    initial,
                    loss="soft_l1",
                    f_scale=2 * step,
                )
                centres[k] = fit.x[:2]
                if y0 is None:
                    y0s[k] = fit.x[2]
                    fitted_centres.append(k)
                continue
        if y0 is None:
            if gs is not None and gs.recon_y0 is not None:
                y0s[k] = gs.recon_y0
            elif len(dty):
                y0s[k] = np.median(dty - _refine_design(omega) @ centres[k])
            else:
                raise ValueError(
                    "Grain {} needs assigned 4D peaks or supplied y0".format(ids[k])
                )
    if y0 is None:
        # Individual centroid fits are noisy estimates of the same scan axis.
        # Do not turn those errors into grain-dependent reconstruction shifts.
        y0s.fill(float(np.median(y0s)))
        for k in fitted_centres:
            finite = (
                (owners == ids[k]) & np.isfinite(cf_4d.omega) & np.isfinite(cf_4d.dty)
            )
            X = _refine_design(cf_4d.omega[finite])
            target = cf_4d.dty[finite] - y0s[k]
            fit = least_squares(
                lambda p: X @ p - target,
                centres[k],
                loss="soft_l1",
                f_scale=2 * step,
            )
            centres[k] = fit.x
    return centres, y0s


def refine_grain_sinograms(
    ds,
    grains,
    cf_4d=None,
    grain_ids=None,
    cf_strong=None,
    hkl_tol=0.05,
    y0=None,
    centres=None,
    roi_radius=None,
    iterations=10,
    mask_level=0.12,
    dilation=2,
    support_level=0.02,
    beam_sigma_steps=0.75,
    damping=0.5,
    weight_tolerance=0.005,
    filter_name="hann",
    minimum_exclusive=5,
    verbose=True,
    phase_name=None,
    use_saved_assignments=True,
    initial_results=(),
    cf_2d=None,
    initial_weights=None,
    compatibility=None,
    intensity_budget=None,
):
    """Refine peak selection by back/forward projection for any number of grains.

    Returns ``(updated_grains, result)``; supplying cf_strong adds an updated
    merged 4D plotting table as a third value. Grain count, original list positions and UBIs are
    preserved. Grain labels use these list positions, as in the mapping notebooks.
    Inputs and files are unchanged. Export result with save_refined_grains() to
    reuse the fractions in the mapping notebook.

    verbose=True displays notebook/terminal progress through preparation,
    reconstruction, projection and final masks, with per-iteration statistics.
    result.stats contains final counts, assigned intensity and elapsed time;
    result.history records weight changes and assignment transitions per
    iteration. verbose=False disables the bar and printed statistics.

    Intermediate backprojections use each fixed grain ROI plus the halo needed
    for Gaussian smoothing. The final reconstruction covers the full image.

    Normal mode uses original cf_4d peaks, their grain_id labels (or best HKL
    matches), and existing saved fractional assignments to seed each grain.
    All HKL-compatible candidates in the selected parents' peak pool can be
    recovered or reassigned using spatial support. Unassigned peaks are also
    candidates; peaks owned by unselected grains are excluded. At least twenty
    contributing 2D peaks per grain are required; grain_ids can restrict the
    operation to grains with sufficient data.
    initial_results can supply in-memory decoupling/refinement results as seeds,
    allowing joint refinement immediately after splitting, before any export.

    Without supplied centres, saved/current translations are used when present;
    otherwise a robust single trajectory is fitted to assigned 4D peaks.
    With y0=None, individual estimates are reduced to one common median scan
    axis and missing positions are refitted using that fixed axis. Supplied
    positions are retained. Explicit y0 may be scalar or one value per selected
    grain; a scalar sets the common calibrated scan axis. roi_radius is likewise
    scalar or per grain; its default comes from seed trajectory widths.

    Seeded mode accepts an original cf_2d with source_2d_row provenance and an
    (n_peaks, n_grains) initial_weights matrix. centres and y0 must be supplied.
    compatibility can further restrict allowed contributions. Spatial splits
    and twins use this mode after their specialized initialization, then share
    this exact refinement loop. intensity_budget optionally limits each peak's
    available fraction in seeded mode. Saved contributions of unselected grains
    are always reserved in normal mode. This routine requires full, uniform dty
    scans.
    """
    from ImageD11.sinograms.geometry import sino_shift_and_pad

    started = perf_counter()
    ids = tuple(range(len(grains))) if grain_ids is None else tuple(grain_ids)
    if (
        not ids
        or len(set(ids)) != len(ids)
        or any(
            not isinstance(gid, (int, np.integer))
            or isinstance(gid, bool)
            or gid < 0
            or gid >= len(grains)
            for gid in ids
        )
    ):
        raise ValueError("grain_ids must contain valid, unique original grain indices")
    step = float(ds.ystep)
    bins = np.asarray(ds.ybincens)
    if (
        not np.isfinite(step)
        or step <= 0
        or len(bins) < 2
        or not np.isfinite(bins).all()
        or not np.allclose(np.diff(bins), step)
    ):
        raise ValueError("Peak refinement requires a uniform increasing dty grid")
    if (
        not np.isfinite(hkl_tol)
        or hkl_tol <= 0
        or not isinstance(iterations, (int, np.integer))
        or iterations < 0
        or not 0 < damping <= 1
        or not 0 < mask_level < 1
        or not 0 < support_level < 1
        or not np.isfinite(beam_sigma_steps)
        or beam_sigma_steps < 0
        or not np.isfinite(weight_tolerance)
        or weight_tolerance <= 0
        or minimum_exclusive < 1
        or not isinstance(dilation, (int, np.integer))
        or dilation < 0
    ):
        raise ValueError("Invalid peak refinement controls")
    provenance = ("source_4d_row", "source_2d_row", "split_fraction")
    for name, table in (("cf_4d", cf_4d), ("cf_strong", cf_strong)):
        if table is not None and any(title in table.titles for title in provenance):
            raise ValueError(
                "{} must be an original 4D peak table, not plotting contributions".format(
                    name
                )
            )
    originals = [grains[gid] for gid in ids]
    n = len(ids)
    total_steps = 1 + 2 * n * (iterations + 1) + int(cf_strong is not None)
    with tqdm(
        total=total_steps,
        desc="Preparing refinement",
        unit="step",
        disable=not verbose,
        dynamic_ncols=True,
    ) as progress:
        seed_results = (
            [initial_results]
            if hasattr(initial_results, "grainsinos")
            else list(initial_results or ())
        )
        provided = _refine_initial_results(grains, seed_results)
        if cf_2d is None:
            if cf_4d is None:
                raise ValueError("Supply cf_4d, or cf_2d with initial_weights")
            if (
                initial_weights is not None
                or compatibility is not None
                or intensity_budget is not None
            ):
                raise ValueError(
                    "initial_weights, compatibility and intensity_budget require cf_2d"
                )
            cf2, weights, budget, saved, owners = _refine_seed_assignments(
                ds,
                grains,
                ids,
                cf_4d,
                cf_strong,
                hkl_tol,
                phase_name,
                use_saved_assignments,
                provided,
            )
            centres, y0s = _refine_seed_geometry(
                originals, ids, cf_4d, owners, saved, centres, y0, step
            )
        else:
            if seed_results:
                raise ValueError(
                    "Use initial_results in normal mode, or initial_weights with cf_2d"
                )
            if initial_weights is None or centres is None or y0 is None:
                raise ValueError(
                    "Seeded refinement requires initial_weights, centres and y0"
                )
            cf2 = cf_2d.copy()
            if "source_2d_row" not in cf2.titles:
                raise ValueError("cf_2d must contain source_2d_row provenance")
            weights = np.asarray(initial_weights, dtype=float).copy()
            budget = (
                np.ones(cf2.nrows)
                if intensity_budget is None
                else np.asarray(intensity_budget, dtype=float).copy()
            )
            centres = np.asarray(centres, dtype=float).copy()
            y0s = np.broadcast_to(np.asarray(y0, dtype=float), (n,)).copy()
        if (
            centres.shape != (n, 2)
            or not np.isfinite(centres).all()
            or not np.isfinite(y0s).all()
        ):
            raise ValueError("Expected finite centres and y0 for every selected grain")
        if (
            weights.shape != (cf2.nrows, n)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or np.any(weights.sum(axis=1) > 1 + 1e-8)
        ):
            raise ValueError(
                "Initial weights must be finite, nonnegative and conserve each peak's intensity"
            )
        if (
            budget.shape != (cf2.nrows,)
            or not np.isfinite(budget).all()
            or np.any(budget < 0)
            or np.any(budget > 1)
        ):
            raise ValueError(
                "intensity_budget must contain one finite fraction in [0, 1] per peak"
            )
        rows = cf2.source_2d_row.astype(np.int64)
        if (
            not np.array_equal(rows, cf2.source_2d_row)
            or len(np.unique(rows)) != len(rows)
            or np.any(rows < 0)
            or np.any(rows >= len(ds.pk2d["omega"]))
        ):
            raise ValueError("source_2d_row must uniquely index the current dataset")
        if not np.allclose(
            cf2.sum_intensity, ds.pk2d["sum_intensity"][rows], equal_nan=True
        ):
            raise ValueError(
                "cf_2d must contain original intensities before fractional weighting"
            )
        errors = _refine_errors(originals, cf2)
        compatible = errors < hkl_tol
        if compatibility is not None:
            allowed = np.asarray(compatibility, dtype=bool)
            if allowed.shape != compatible.shape:
                raise ValueError("compatibility must have shape (n_peaks, n_grains)")
            compatible &= allowed
        yi = np.rint((cf2.dty - bins[0]) / step)
        valid = (
            np.isfinite(cf2.omega)
            & np.isfinite(cf2.dty)
            & np.isfinite(cf2.eta)
            & np.isfinite(cf2.sum_intensity)
            & (cf2.sum_intensity > 0)
            & (yi >= 0)
            & (yi < len(bins))
            & compatible.any(axis=1)
            & (budget > 0)
        )
        input_candidates = cf2.nrows
        cf2.filter(valid)
        weights, compatible, budget = weights[valid], compatible[valid], budget[valid]
        weights[~compatible] = 0
        total = weights.sum(axis=1)
        np.divide(weights, total[:, None], out=weights, where=total[:, None] > 0)
        weights *= budget[:, None]
        radii = (
            np.empty(n, dtype=float)
            if roi_radius is None
            else np.broadcast_to(np.asarray(roi_radius, dtype=float), (n,)).copy()
        )
        for k in range(n):
            seeded = weights[:, k] > 0
            if seeded.sum() < 20:
                raise ValueError(
                    "Grain {} has {} contributing seed peaks; at least 20 are required".format(
                        ids[k], int(seeded.sum())
                    )
                )
            if roi_radius is None:
                prediction = _refine_design(cf2.omega[seeded]) @ centres[k] + y0s[k]
                radii[k] = max(
                    6 * step,
                    1.5 * np.percentile(np.abs(cf2.dty[seeded] - prediction), 90),
                )
        if not np.isfinite(radii).all() or np.any(radii <= 0):
            raise ValueError("ROI radii must be positive and finite")
        geometry = [sino_shift_and_pad(yy, len(bins), bins.min(), step) for yy in y0s]
        pad = max(gpad for _, gpad in geometry)
        shape = (len(bins) + pad, len(bins) + pad)
        mask_geometry = [
            _refine_mask_geometry(shape, centres[k], step, radii[k]) for k in range(n)
        ]
        reconstruction_masks = [
            (
                _refine_reconstruction_mask(roi, geometry[k][0], len(bins))
                if iterations
                else None
            )
            for k, (roi, _) in enumerate(mask_geometry)
        ]
        progress.update(1)
        seed_stats = _refine_peak_statistics(weights, cf2.sum_intensity, budget)
        seed_active = weights.sum(axis=1) > 0
        seed_owners = weights.argmax(axis=1)
        seed_fractions = weights[np.arange(len(weights)), seed_owners]
        if verbose:
            progress.write(
                "Peak refinement: {} grains, {} candidate 2D peaks, {} seeded, "
                "{} filtered; seed counts={}".format(
                    n,
                    len(weights),
                    seed_stats["retained"],
                    input_candidates - len(weights),
                    seed_stats["counts"],
                )
            )
            if np.all(y0s == y0s[0]):
                progress.write(
                    "Rotation-axis dty: {:.6g} (shared by {} grains)".format(y0s[0], n)
                )
        progress.set_description("Initial reconstructions")
        sinos, reflection_groups, reflection_ids = [], [], []
        gv2d = np.column_stack((cf2.gx, cf2.gy, cf2.gz))
        reflection_cache = {}
        for k, original in enumerate(originals):
            child = copy.deepcopy(original)
            child.gid = ids[k]
            z = original.translation[2] if original.translation is not None else 0
            child.translation = np.r_[centres[k], z]
            gs = GrainSinogram(child, ds)
            gs.peak_assignment_kind = "refined"
            _refine_rebuild_sinogram(
                gs,
                cf2,
                weights[:, k],
                y0s[k],
                geometry[k][0],
                pad,
                filter_name,
                recon_mask=reconstruction_masks[k],
            )
            sinos.append(gs)
            key = original.ubi.tobytes()
            if key not in reflection_cache:
                hkl = np.rint(gv2d @ original.ubi.T).astype(int)
                _, labels = np.unique(
                    np.column_stack((hkl, np.sign(cf2.eta))),
                    axis=0,
                    return_inverse=True,
                )
                reflection_cache[key] = (labels, _refine_group_rows(labels))
            labels, groups = reflection_cache[key]
            reflection_ids.append(labels)
            reflection_groups.append(groups)
            progress.update(1)
        reflection_ids = np.column_stack(reflection_ids)
        angle_groups = _refine_group_rows(cf2.omega)
        history, converged = [], False
        for iteration in range(iterations):
            iteration_started = perf_counter()
            progress.set_description("Iteration {}: projection".format(iteration + 1))
            smoothed = [
                gaussian_filter(np.maximum(gs.recons["iradon"], 0), 1) for gs in sinos
            ]
            projections, mask_pixels = [], []
            for k, gs in enumerate(sinos):
                mask = _refine_connected_grain_mask(
                    gs.recons["iradon"],
                    centres[k],
                    step,
                    radii[k],
                    mask_level,
                    dilation,
                    mask_geometry[k],
                    smoothed[k],
                )
                mask_pixels.append(int(mask.sum()))
                projections.append(
                    _refine_project_grain_profile(
                        mask,
                        gs.recons["iradon"],
                        cf2.omega,
                        cf2.dty,
                        y0s[k],
                        step,
                        support_level,
                        beam_sigma_steps,
                        angle_groups,
                        smoothed[k],
                    )
                )
                progress.update(1)
            progress.set_description(
                "Iteration {}: peak assignment".format(iteration + 1)
            )
            profiles = np.column_stack([p[0] for p in projections])
            supports = np.column_stack([p[1] for p in projections]) & compatible
            target, predicted, calibrated = _refine_shared_peak_weights(
                profiles,
                supports,
                cf2.sum_intensity * budget,
                reflection_ids,
                minimum_exclusive,
                reflection_groups,
            )
            target *= budget[:, None]
            overlap = supports.sum(axis=1) > 1
            previous = weights.copy()
            previous[~supports] = 0
            previous_total = previous.sum(axis=1)
            np.divide(
                previous,
                previous_total[:, None],
                out=previous,
                where=previous_total[:, None] > 0,
            )
            previous *= budget[:, None]
            damp = overlap & (previous_total > 0)
            new_weights = target.copy()
            new_weights[damp] = (1 - damping) * previous[damp] + damping * target[damp]
            active = supports.any(axis=1) | (weights.sum(axis=1) > 0)
            if not active.any():
                raise ValueError(
                    "Feedback rejected all peaks; inspect ROI and mask_level"
                )
            change = np.abs(new_weights - weights)
            # Keep the original decoupling tolerance in units of a single
            # fractional assignment, rather than summing changes over grains.
            delta, maximum = float(change[active].mean()), float(change.max())
            old_owners = weights.argmax(axis=1)
            iteration_stats = _refine_peak_statistics(
                new_weights, cf2.sum_intensity, budget
            )
            iteration_stats.update(
                _refine_assignment_changes(
                    new_weights,
                    weights.sum(axis=1) > 0,
                    old_owners,
                    weights[np.arange(len(weights)), old_owners],
                    weight_tolerance,
                )
            )
            iteration_stats["changed"] = int(
                (change.max(axis=1) > weight_tolerance).sum()
            )
            weights = new_weights
            counts = iteration_stats["counts"]
            if min(counts) < 20:
                raise ValueError(
                    "Grain {} retained fewer than 20 peaks; inspect its ROI and mask_level".format(
                        ids[int(np.argmin(counts))]
                    )
                )
            trusted = supports.any(axis=1) & np.all(~supports | calibrated, axis=1)
            available = cf2.sum_intensity * budget
            residual = (
                float(
                    np.median(
                        np.abs(predicted[trusted] - available[trusted])
                        / available[trusted]
                    )
                )
                if trusted.any()
                else None
            )
            progress.set_description(
                "Iteration {}: reconstruction".format(iteration + 1)
            )
            stable = delta < weight_tolerance and maximum < 5 * weight_tolerance
            final_iteration = stable or iteration + 1 == iterations
            for k, gs in enumerate(sinos):
                _refine_rebuild_sinogram(
                    gs,
                    cf2,
                    weights[:, k],
                    y0s[k],
                    geometry[k][0],
                    pad,
                    filter_name,
                    recon_mask=None if final_iteration else reconstruction_masks[k],
                )
                progress.update(1)
            if not np.allclose(
                weights[supports.any(axis=1)].sum(axis=1), budget[supports.any(axis=1)]
            ):
                raise RuntimeError("Intensity partition lost normalization")
            iteration_stats.update(
                iteration=iteration + 1,
                weight_change=delta,
                maximum_change=maximum,
                calibrated_relative_residual=residual,
                mask_pixels=mask_pixels,
                projections=[len(gs.sinoangles) for gs in sinos],
                elapsed_seconds=perf_counter() - iteration_started,
            )
            history.append(iteration_stats)
            progress.set_postfix(
                kept=iteration_stats["retained"],
                shared=iteration_stats["shared"],
                change="{:.3g}".format(delta),
            )
            if verbose:
                progress.write(
                    "Iteration {}: kept={}/{}, rejected={}, shared={}, added={}, "
                    "removed={}, reassigned={}, intensity={:.1%}, "
                    "weight change={:.4g} (max={:.4g}), {:.2f}s".format(
                        iteration + 1,
                        iteration_stats["retained"],
                        len(weights),
                        iteration_stats["rejected"],
                        iteration_stats["shared"],
                        iteration_stats["added"],
                        iteration_stats["removed"],
                        iteration_stats["reassigned"],
                        iteration_stats["retained_available_intensity"],
                        delta,
                        maximum,
                        iteration_stats["elapsed_seconds"],
                    )
                )
            if stable:
                converged = True
                progress.total = progress.n + n + int(cf_strong is not None)
                progress.refresh()
                break
        if iterations and not converged:
            warnings.warn(
                "Peak refinement reached its iteration limit; inspect diagnostics",
                RuntimeWarning,
            )
        progress.set_description("Final grain masks")
        updated = list(grains)
        for k, gs in enumerate(sinos):
            contributing = weights[:, k] > 0
            gs.source_2d_rows = cf2.source_2d_row[contributing].astype(np.int64)
            gs.peak_fractions = weights[contributing, k].copy()
            gs.grain.npks_4d = len(np.unique(cf2.spot3d_id[contributing]))
            gs.spatial_split_mask = _refine_connected_grain_mask(
                gs.recons["iradon"],
                centres[k],
                step,
                radii[k],
                mask_level,
                dilation,
                mask_geometry[k],
            )
            updated[ids[k]] = gs.grain
            progress.update(1)
        result = GrainRefinementResult(
            sinos,
            centres,
            y0s,
            cf2.source_2d_row.astype(np.int64),
            weights,
            budget,
            history,
            converged,
            ids,
            ids,
        )
        plotting_table = None
        if cf_strong is not None:
            progress.set_description("Merging plotting peaks")
            plotting_table = refined_strong_columnfile(ds, cf_strong, [result])
            progress.update(1)
        result.stats.update(_refine_peak_statistics(weights, cf2.sum_intensity, budget))
        result.stats.update(
            _refine_assignment_changes(
                weights,
                seed_active,
                seed_owners,
                seed_fractions,
                weight_tolerance,
            )
        )
        result.stats.update(
            grains=n,
            input_candidates=input_candidates,
            filtered=input_candidates - len(weights),
            seeded=seed_stats["retained"],
            initial_counts=seed_stats["counts"],
            iterations=len(history),
            converged=converged,
            elapsed_seconds=perf_counter() - started,
            mask_pixels=[int(gs.spatial_split_mask.sum()) for gs in sinos],
            projections=[len(gs.sinoangles) for gs in sinos],
        )
        for name in ("weight_change", "maximum_change", "calibrated_relative_residual"):
            result.stats[name] = history[-1][name] if history else None
        progress.set_description("Peak refinement complete")
        if verbose:
            status = (
                ("converged" if converged else "iteration limit reached")
                if iterations
                else "initialization only"
            )
            progress.write(
                "Peak refinement {} after {} iterations in {:.2f}s: kept={}/{}, "
                "rejected={}, shared={}, intensity={:.1%}; per-grain counts={}".format(
                    status,
                    len(history),
                    result.stats["elapsed_seconds"],
                    result.stats["retained"],
                    len(weights),
                    result.stats["rejected"],
                    result.stats["shared"],
                    result.stats["retained_available_intensity"],
                    result.stats["counts"],
                )
            )
        if cf_strong is not None:
            return updated, result, plotting_table
        return updated, result


@dataclass
class GrainDecouplingResult:
    """Initial grains, reconstructed shapes and conserved 2D intensity fractions.

    ``source_2d_rows`` indexes ``ds.pk2d``. ``peak_weights[:, k]`` is the
    fraction assigned to child k; zero-sum rows are outside all supports.
    UBIs are initial copies, not separately refined orientations. Shared
    contributions are model estimates, not independently measured intensities.
    """

    grainsinos: list
    centres: np.ndarray
    y0: float
    source_2d_rows: np.ndarray
    peak_weights: np.ndarray
    selected_4d_rows: np.ndarray
    labels4d: np.ndarray
    history: list
    converged: bool
    parent_ids: tuple = ()
    child_ids: tuple = ()
    stats: dict = field(default_factory=dict)


def _refine_colfile_from_rows(ds, rows, parameters):
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= len(ds.pk2d["omega"])):
        raise ValueError("Saved 2D peak rows are outside the current dataset")
    peaks = {name: values[rows] for name, values in ds.pk2d.items()}
    cf = ds.get_colfile_from_peaks_dict(peaks_dict=peaks)
    cf.parameters = copy.deepcopy(parameters)
    cf.updateGeometry()
    cf.addcolumn(rows, "source_2d_row")
    return cf


def _validate_peak_assignment(ds, rows, fractions):
    """Validate fractional provenance before using saved rows for indexing."""
    rows = np.asarray(rows)
    fractions = np.asarray(fractions, dtype=float)
    if (
        rows.ndim != 1
        or fractions.shape != rows.shape
        or rows.dtype.kind not in "iuf"
        or not np.isfinite(rows).all()
        or np.any(rows < 0)
        or np.any(rows >= len(ds.pk2d["omega"]))
        or len(np.unique(rows)) != len(rows)
    ):
        raise ValueError("Saved peak rows must uniquely index the current dataset")
    integer_rows = rows.astype(np.int64)
    if not np.array_equal(integer_rows, rows):
        raise ValueError("Saved peak rows must be integer indices")
    if (
        not np.isfinite(fractions).all()
        or np.any(fractions <= 0)
        or np.any(fractions > 1)
    ):
        raise ValueError("Invalid saved peak fractions")
    return integer_rows, fractions


def _saved_split_assignment(gs, grain_label, phase_name):
    if gs.source_2d_rows is not None:
        return _validate_peak_assignment(gs.ds, gs.source_2d_rows, gs.peak_fractions)
    filename = getattr(gs.ds, "grainsfile", None)
    if filename is None or not os.path.exists(filename):
        return None
    candidates = []
    with h5py.File(filename, "r") as source:
        phases = [phase_name] if phase_name is not None else list(source.keys())
        for phase in phases:
            if phase not in source or not isinstance(source[phase], h5py.Group):
                continue
            group = source[phase].get(str(grain_label))
            if (
                not isinstance(group, h5py.Group)
                or "split_peak_assignment" not in group
                or "ubi" not in group
            ):
                continue
            # Reject stale assignments when indexing changed the grain identity.
            if not np.allclose(group["ubi"][:], gs.grain.ubi, rtol=1e-9, atol=1e-10):
                continue
            if gs.grain.translation is not None:
                if "translation" not in group or not np.allclose(
                    group["translation"][:], gs.grain.translation, rtol=1e-9, atol=1e-10
                ):
                    continue
            assignment = group["split_peak_assignment"]
            parameters = group.get("recon_parameters")
            saved_y0 = (
                parameters.attrs.get("recon_y0") if parameters is not None else None
            )
            saved_mask = (
                group["spatial_split_mask"][:]
                if "spatial_split_mask" in group
                else None
            )
            candidates.append(
                (
                    assignment["source_2d_row"][:],
                    assignment["intensity_fraction"][:],
                    saved_y0,
                    saved_mask,
                    group.attrs.get("peak_assignment_kind", "split"),
                )
            )
    if len(candidates) > 1:
        raise ValueError(
            "Saved peak assignments match several phases; supply phase_name"
        )
    if not candidates:
        return None
    rows, fractions, saved_y0, saved_mask, assignment_kind = candidates[0]
    rows, fractions = _validate_peak_assignment(gs.ds, rows, fractions)
    gs.peak_assignment_kind = assignment_kind
    if saved_y0 is not None:
        gs.recon_y0 = float(saved_y0)
    if saved_mask is not None:
        gs.spatial_split_mask = saved_mask
    return rows, fractions


def _refine_errors(grains, cf, rows=None):
    """HKL residuals for all rows, or only the supplied row indices."""
    columns = (cf.gx, cf.gy, cf.gz)
    if rows is not None:
        columns = tuple(column[rows] for column in columns)
    gv = np.column_stack(columns)
    errors = []
    previous = None
    for grain in grains:
        if grain is not previous:
            hkl = gv @ grain.ubi.T
            hkl -= np.rint(hkl)
            error = np.sqrt(np.einsum("ij,ij->i", hkl, hkl))
        errors.append(error)
        previous = grain
    return np.column_stack(errors)


def _decouple_fit_twins(omega, dty, compatible, width, y0):
    exclusive = compatible.sum(axis=1) == 1
    labels = compatible.argmax(axis=1)
    X = _refine_design(omega)
    columns = 4 if y0 is not None else 5
    A = np.zeros((int(exclusive.sum()), columns))
    choices = labels[exclusive]
    for k in (0, 1):
        if np.sum(exclusive & (labels == k)) < 8:
            raise ValueError(
                "Each twin needs at least eight orientation-exclusive 4D peaks"
            )
        A[choices == k, 2 * k : 2 * k + 2] = X[exclusive][choices == k]
    if y0 is None:
        A[:, 4] = 1
    if np.linalg.matrix_rank(A) < columns:
        raise ValueError(
            "Insufficient twin-exclusive angular coverage; supply calibrated y0"
        )
    target = dty[exclusive] - (y0 if y0 is not None else 0)
    initial = np.linalg.lstsq(A, target, rcond=None)[0]
    fit = least_squares(
        lambda p: A @ p - target, initial, loss="soft_l1", f_scale=width
    )
    return fit.x[:4].reshape(2, 2), float(y0 if y0 is not None else fit.x[4])


def _decouple_eligible_4d(grains, cf_4d, hkl_tol, peak_mask=None):
    """Keep finite, compatible merged peaks in their original row order."""
    if peak_mask is not None:
        peak_mask = np.asarray(peak_mask, dtype=bool)
        if peak_mask.shape != (cf_4d.nrows,):
            raise ValueError("peak_mask must contain one value per 4D peak")
    if peak_mask is None or np.all(peak_mask):
        errors = _refine_errors(grains, cf_4d)
        compatible = errors < hkl_tol
        finite_errors = np.isfinite(errors).all(axis=1)
    else:
        rows = np.flatnonzero(peak_mask)
        if len(rows) < 20:
            raise ValueError("Too few compatible 4D peaks to decouple")
        errors = _refine_errors(grains, cf_4d, rows=rows)
        compatible = np.zeros((cf_4d.nrows, len(grains)), dtype=bool)
        compatible[rows] = (errors < hkl_tol) & np.isfinite(errors).all(axis=1)[:, None]
        finite_errors = peak_mask
    eligible = (np.isfinite(cf_4d.omega) & np.isfinite(cf_4d.dty)
                & finite_errors & compatible.any(axis=1))
    selected = np.flatnonzero(eligible)
    if len(selected) < 20:
        raise ValueError("Too few compatible 4D peaks to decouple")
    return selected, compatible


def _decouple_threshold_components(image, count, step, level, dilation,
                                   keep_largest=False, diagnostics=None):
    """Segment thresholded cores, optionally retaining the largest count by area."""
    smooth = gaussian_filter(np.maximum(image, 0), 1)
    if not np.isfinite(smooth).all() or smooth.max() <= 0:
        raise RuntimeError("Parent reconstruction has no finite positive density")
    components, found = ndi_label(smooth >= level * smooth.max())
    if found < count or (found != count and not keep_largest):
        raise RuntimeError(
            "Spatial threshold found {} disconnected blobs; expected {}. "
            "Inspect the parent reconstruction and mask_level; no blobs were "
            "discarded, merged or split.".format(found, count))
    areas = np.bincount(components.ravel(), minlength=found + 1)[1:]
    density = np.bincount(components.ravel(), weights=smooth.ravel(), minlength=found + 1)[1:]
    region_ids = np.arange(1, found + 1)
    ranking = np.lexsort((region_ids, -areas))
    selected = region_ids[ranking[:count]]
    excluded = region_ids[ranking[count:]]
    if diagnostics is not None:
        diagnostics.update(
            expected=count, found=found, mask_level=level,
            selection="largest pixel area" if keep_largest else "exact count",
            selected_region_ids=selected.tolist(), excluded_region_ids=excluded.tolist(),
            excluded_pixels=int(areas[excluded - 1].sum()),
            excluded_density=float(density[excluded - 1].sum()),
            thresholded_density=float(density.sum()),
            regions=[dict(region_id=int(region_ids[k]), pixels=int(areas[k]),
                          density_sum=float(density[k]), selected=bool(region_ids[k] in selected))
                     for k in ranking],
            image=smooth, labels=components)
    centres, masks, radii = [], [], []
    for label_id in selected:
        core = components == label_id
        ri, rj = np.nonzero(core)
        sx, sy = recon_to_sample(ri, rj, image.shape, step)
        mass = smooth[core]
        centre = np.array([np.average(sx, weights=mass), np.average(sy, weights=mass)])
        mask = binary_fill_holes(core)
        if dilation:
            mask = binary_dilation(mask, iterations=dilation)
        ri, rj = np.nonzero(mask)
        sx, sy = recon_to_sample(ri, rj, image.shape, step)
        # Include the full thresholded/dilated region and one pixel of margin.
        radius = np.sqrt(((sx - centre[0]) ** 2 + (sy - centre[1]) ** 2).max()) + step
        centres.append(centre)
        masks.append(mask)
        radii.append(radius)
    centres = np.asarray(centres)
    order = np.lexsort((centres[:, 1], centres[:, 0]))
    if diagnostics is not None:
        diagnostics["selected_region_ids"] = selected[order].tolist()
    return (centres[order], [masks[k] for k in order],
            np.asarray(radii)[order], smooth)


def _decouple_spatial_initialization(ds, grain, cf2, count, y0, mask_level,
                                     dilation, support_level, beam_sigma_steps,
                                     filter_name, minimum_exclusive,
                                     keep_largest=True, diagnostics=None):
    """Reconstruct once, segment disconnected blobs and project their supports."""
    parent = GrainSinogram(copy.deepcopy(grain), ds)
    shift, pad = sino_shift_and_pad(y0, len(ds.ybincens), min(ds.ybincens), ds.ystep)
    _refine_rebuild_sinogram(parent, cf2, np.ones(cf2.nrows), y0, shift, pad, filter_name)
    centres, masks, radii, smooth = _decouple_threshold_components(
        parent.recons["iradon"], count, float(ds.ystep), mask_level, dilation,
        keep_largest=keep_largest, diagnostics=diagnostics)
    profiles = np.empty((cf2.nrows, count))
    supports = np.empty((cf2.nrows, count), dtype=bool)
    angle_groups = _refine_group_rows(cf2.omega)
    for k, mask in enumerate(masks):
        profiles[:, k], supports[:, k] = _refine_project_grain_profile(
            mask, parent.recons["iradon"], cf2.omega, cf2.dty, y0, ds.ystep,
            support_level, beam_sigma_steps, angle_groups, smooth)
    gv = np.column_stack((cf2.gx, cf2.gy, cf2.gz))
    hkl = np.rint(np.dot(gv, grain.ubi.T)).astype(int)
    _, reflection_ids = np.unique(
        np.column_stack((hkl, np.sign(cf2.eta))), axis=0, return_inverse=True)
    groups = _refine_group_rows(reflection_ids)
    weights, _, _ = _refine_shared_peak_weights(
        profiles, supports, cf2.sum_intensity, reflection_ids,
        minimum_exclusive, [groups] * count)
    if np.any((weights > 0).sum(axis=0) < 20):
        raise RuntimeError("A thresholded blob has fewer than 20 projected seed peaks")
    return centres, weights, radii


def split_grain_sinogram(
    ds,
    grain,
    cf_4d,
    twin_grain=None,
    hkl_tol=0.05,
    y0=None,
    width=None,
    width_2d=None,
    margin=None,
    angle_bin=10,
    density_fraction=0.08,
    trials=600,
    seed=42,
    iterations=10,
    mask_level=0.12,
    dilation=2,
    roi_radius=None,
    support_level=0.02,
    beam_sigma_steps=0.75,
    damping=0.5,
    weight_tolerance=0.005,
    filter_name="hann",
    minimum_exclusive=5,
    verbose=True,
    peak_mask=None,
    n_components=2,
    max_components=4,
    _initial_fit=None,
    ransac_confidence=0.999,
    initialization="spatial",
    spatial_keep_largest=True,
):
    """Decouple one mixed orientation, or two known twin-related orientations.

    Single orientation: reconstruct the parent's peaks and threshold disconnected
    blobs to initialize the requested count (default two). Project these masks to
    select seed peaks, then refine by back/forward projection. This default path
    fits no sinusoids and requires calibrated y0 or ds.y0. When extra blobs are
    present, spatial_keep_largest=True retains the requested count by pixel area
    and reports exclusions. False requires an exact count; too few always raises.

    initialization="trajectory" explicitly enables the separate sinusoid-based
    initializer, with n_components=None opting into automatic count estimation.

    With ``twin_grain``, preserve the two distinct UBIs. Reflections compatible
    with only one UBI seed its shape; only peaks compatible with both can be
    shared. This also permits co-centred twins provided exclusive reflections
    resolve their shapes. The twin law/second orientation is NOT inferred.
    Both initializers delegate reconstruction and projection feedback to
    refine_grain_sinograms().

    Uses full, uniformly increasing dty scans. No files or input grains are
    changed. Returned shapes/UBIs require inspection and separate refinement.
    ``peak_mask`` restricts rows before HKL scoring, retaining original indices.
    ``ransac_confidence`` applies only to automatic count estimation. Spatial
    initialization with a fixed count always uses the original full-trial solver.
    """
    if initialization not in ("spatial", "trajectory"):
        raise ValueError("initialization must be 'spatial' or 'trajectory'")
    spatial = twin_grain is None and initialization == "spatial"
    if spatial and n_components is None:
        raise ValueError("Spatial initialization requires an explicit expected component count")
    if ransac_confidence is not None and not 0 < ransac_confidence < 1:
        raise ValueError("RANSAC confidence must be between zero and one, or None")
    if n_components is not None and (isinstance(n_components, (bool, np.bool_))
            or not isinstance(n_components, (int, np.integer)) or not 2 <= n_components <= 4):
        raise ValueError("n_components must be an integer from two to four")
    if twin_grain is not None and n_components not in (None, 2):
        raise ValueError("Twin pairs require exactly two components")
    step = float(ds.ystep)
    bins = np.asarray(ds.ybincens)
    if step <= 0 or len(bins) < 2 or not np.allclose(np.diff(bins), step):
        raise ValueError("Decoupling requires a uniform increasing dty grid")
    width = float(2 * step if width is None else width)
    margin = float(step if margin is None else margin)
    if (
        hkl_tol <= 0
        or width <= 0
        or margin < 0
        or iterations < 0
        or not 0 < damping <= 1
        or not 0 < mask_level < 1
        or weight_tolerance <= 0
        or minimum_exclusive < 1
        or dilation < 0
    ):
        raise ValueError("Invalid decoupling controls")
    originals = [grain] if twin_grain is None else [grain, twin_grain]
    selected, compatible4d = _decouple_eligible_4d(
        originals, cf_4d, hkl_tol, peak_mask)
    omega, dty = cf_4d.omega[selected], cf_4d.dty[selected]
    selection = None
    if spatial:
        fitted_y0 = y0 if y0 is not None else getattr(ds, "y0", None)
        if fitted_y0 is None or not np.isfinite(fitted_y0):
            raise ValueError("Spatial initialization needs calibrated y0 or ds.y0; no sinusoid is fitted")
        fitted_y0 = float(fitted_y0)
        labels4d = np.full(len(selected), -1, dtype=int)
        originals = [grain] * n_components
    elif twin_grain is None:
        if n_components is None:
            initial = (_decouple_auto_components(
                omega, dty, width, margin, y0, trials, seed, max_components, ransac_confidence)
                if _initial_fit is None else _initial_fit)
            centres, fitted_y0, labels4d, selection = initial
            n_components = len(centres)
            if verbose:
                print("Detected {} spatial component(s); scores: {}{}".format(
                    n_components, dict(zip(selection["counts"], selection.get("selection_scores", selection["scores"]))),
                    " (ambiguous)" if selection["ambiguous"] else ""))
            if selection["reached_limit"]:
                warnings.warn("Automatic count reached max_components; inspect tracks or increase the limit", RuntimeWarning)
        # Automatic detection chooses the count only. Spatial initialization
        # always uses the original fixed-count solver for multi-grain splits.
        if n_components > 1:
            centres, fitted_y0, labels4d = _decouple_fit_components(
                omega, dty, width, margin, y0, trials, seed, n_components,
                confidence=None)
        originals = [grain] * n_components
    else:
        n_components = 2
        centres, fitted_y0 = _decouple_fit_twins(
            omega, dty, compatible4d[selected], width, y0
        )
        labels4d = np.full(len(selected), -2, dtype=int)
        exclusive = compatible4d[selected].sum(axis=1) == 1
        labels4d[exclusive] = compatible4d[selected][exclusive].argmax(axis=1)
    # np.isin handles partial/unsorted 4D tables without the mapping kernel's bounds assumptions.
    rows = np.flatnonzero(np.isin(ds.pk2d["spot3d_id"], cf_4d.spot3d_id[selected]))
    cf2 = _refine_colfile_from_rows(ds, rows, cf_4d.parameters)
    errors2d = _refine_errors(originals, cf2)
    compatible2d = errors2d < hkl_tol
    yindices = np.rint((cf2.dty - bins[0]) / step)
    valid2d = (
        np.isfinite(errors2d).all(axis=1)
        & np.isfinite(cf2.omega)
        & np.isfinite(cf2.dty)
        & np.isfinite(cf2.eta)
        & np.isfinite(cf2.sum_intensity)
        & (cf2.sum_intensity > 0)
        & (yindices >= 0)
        & (yindices < len(bins))
        & compatible2d.any(axis=1)
    )
    cf2.filter(valid2d)
    compatible2d = compatible2d[valid2d]
    if cf2.nrows < 20 * n_components:
        raise ValueError("Too few valid compatible 2D peaks to decouple")
    spatial_diagnostics = {}
    if spatial:
        centres, weights, spatial_radii = _decouple_spatial_initialization(
            ds, grain, cf2, n_components, fitted_y0, mask_level, dilation,
            support_level, beam_sigma_steps, filter_name, minimum_exclusive,
            keep_largest=spatial_keep_largest, diagnostics=spatial_diagnostics)
        if verbose and spatial_diagnostics:
            print("Spatial initialization kept {} of {} regions; excluded {} regions ({} pixels).".format(
                n_components, spatial_diagnostics["found"],
                len(spatial_diagnostics["excluded_region_ids"]), spatial_diagnostics["excluded_pixels"]))
    elif twin_grain is None:
        labels, _, _, envelopes = _decouple_select_full_branches(
            cf2.omega,
            cf2.dty,
            centres,
            fitted_y0,
            step,
            width,
            margin,
            width_2d,
            angle_bin,
            density_fraction,
        )
        size = np.percentile(envelopes, 90, axis=(0, 2))
    else:
        labels = np.full(cf2.nrows, -1, dtype=int)
        exclusive = compatible2d.sum(axis=1) == 1
        labels[exclusive] = compatible2d[exclusive].argmax(axis=1)
        residual = np.abs(
            cf2.dty[:, None] - _refine_predict(cf2.omega, centres, fitted_y0)
        )
        if any(np.sum(labels == k) < 20 for k in range(n_components)):
            raise ValueError(
                "Each twin needs enough exclusive 2D peaks to seed its reconstruction"
            )
        size = np.array([np.percentile(residual[labels == k, k], 90) for k in range(n_components)])
    if not spatial:
        weights = np.column_stack([labels == k for k in range(n_components)]).astype(float)
    radii = (
        (spatial_radii if spatial else np.maximum(6 * step, 1.5 * size))
        if roi_radius is None
        else np.broadcast_to(np.asarray(roi_radius, float), (n_components,)).copy()
    )
    if not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError("ROI radii must be positive and finite")
    seeds = []
    for k, original in enumerate(originals):
        child = copy.deepcopy(original)
        child.name = "{}_split_{}".format(getattr(original, "name", "grain"), k)
        if hasattr(child, "intensity_info"):
            del child.intensity_info
        seeds.append(child)
    _, refined = refine_grain_sinograms(
        ds,
        seeds,
        cf_2d=cf2,
        initial_weights=weights,
        compatibility=compatible2d,
        centres=centres,
        y0=fitted_y0,
        roi_radius=radii,
        hkl_tol=hkl_tol,
        iterations=iterations,
        mask_level=mask_level,
        dilation=dilation,
        support_level=support_level,
        beam_sigma_steps=beam_sigma_steps,
        damping=damping,
        weight_tolerance=weight_tolerance,
        filter_name=filter_name,
        minimum_exclusive=minimum_exclusive,
        verbose=verbose,
        use_saved_assignments=False,
    )
    for gs in refined.grainsinos:
        gs.peak_assignment_kind = "split"
    refined.stats["initialization"] = "spatial_threshold" if spatial else "trajectory"
    if selection is not None:
        refined.stats["component_selection"] = selection
    result = GrainDecouplingResult(
        refined.grainsinos,
        refined.centres,
        fitted_y0,
        refined.source_2d_rows,
        refined.peak_weights,
        selected,
        labels4d,
        refined.history,
        refined.converged,
        stats=refined.stats,
    )
    if spatial:
        result.spatial_component_diagnostics = spatial_diagnostics
        result.stats["spatial_components"] = {
            key: value for key, value in spatial_diagnostics.items() if key not in ("image", "labels")}
    return result


def _refine_peak_owners(grains, cf_4d, cf_strong, hkl_tol):
    """Assign each merged peak to one parent before independent splits.

    Existing 4D assignments are used when available. Otherwise score against
    all original grains. Strong-subset assignments override those labels.
    """
    if "grain_id" in cf_4d.titles:
        owners = cf_4d.grain_id.astype(np.int64).copy()
    else:
        owners = np.full(cf_4d.nrows, -1, dtype=np.int64)
        best = np.full(cf_4d.nrows, hkl_tol)
        for gid, grain in enumerate(grains):
            errors = _refine_errors([grain], cf_4d)[:, 0]
            better = errors < best
            owners[better] = gid
            best[better] = errors[better]
    if cf_strong is not None:
        if "grain_id" not in cf_strong.titles:
            raise ValueError("cf_strong must contain grain_id assignments")
        spot_ids, order = np.unique(cf_strong.spot3d_id, return_index=True)
        if len(spot_ids) != cf_strong.nrows:
            raise ValueError("cf_strong must contain unique spot3d_id values")
        matched = np.isin(cf_4d.spot3d_id, spot_ids)
        indices = order[np.searchsorted(spot_ids, cf_4d.spot3d_id[matched])]
        owners[matched] = cf_strong.grain_id[indices].astype(np.int64)
    return owners


# Compatibility names for the earlier decoupling-specific helpers.
_decouple_group_rows = _refine_group_rows
_decouple_project_kernel = _refine_project_kernel
_decouple_mask_geometry = _refine_mask_geometry
_decouple_design = _refine_design
_decouple_predict = _refine_predict
_decouple_connected_grain_mask = _refine_connected_grain_mask
_decouple_project_grain_profile = _refine_project_grain_profile
_decouple_colfile_from_rows = _refine_colfile_from_rows
_decouple_errors = _refine_errors
_decouple_peak_owners = _refine_peak_owners


def decouple_grain_sinograms(
    ds, grains, cf_4d, grain_ids=(), twin_pairs=(), cf_strong=None, report=None, **kwargs
):
    """Return updated grains and results for selected IDs/pairs or batch screening.

    The default spatial mode requires explicit expected counts. With
    initialization="trajectory", grain_ids=None screens all parents except known twin pairs. Only
    unambiguous multi-track parents below the search limit are reconstructed.
    Single-track, uncertain and failed parents retain their original objects.
    An optional report list receives one status dictionary per processed entry.
    Per-parent ValueError/RuntimeError failures are reported and skipped in batch
    mode; explicit selections retain fail-fast behavior. Global intensity
    conflicts still raise. Return shapes are unchanged.

    grain_ids accepts a list (two manual components per parent) or a mapping such as
    {0: 3, 2: 4} giving the child count per parent. IDs are original list
    positions. Child 0 retains its parent ID; other children are appended. Twin pairs retain both indices
    and both original UBIs. Inputs are unchanged; entries must be disjoint.
    Each entry considers peaks owned by its parent IDs and compatible
    unassigned peaks, so incomplete indexing cannot remove a spatial branch.
    Peaks assigned to other parents are excluded. Ownership comes from
    cf_4d.grain_id, or the best HKL match among all original grains if absent;
    cf_strong.grain_id takes precedence for its subset. Twin pairs jointly
    partition the union of both parents' peaks.

    If ``cf_strong`` is supplied, return a third value: an updated merged
    4D plotting columnfile restricted to the original strong subset.
    Untouched rows are preserved. Split rows are remerged from weighted 2D
    contributions, once per contributing child. Repeated spot3d_id values
    identify shared parent peaks; use this table for plotting, not indexing.
    Keep the original cf_strong for reruns; the returned table must not be
    fed back as an input alongside the original parents.
    """
    for name, cf in (("cf_4d", cf_4d), ("cf_strong", cf_strong)):
        if cf is not None and any(
            title in cf.titles
            for title in ("source_4d_row", "source_2d_row", "split_fraction")
        ):
            raise ValueError(
                "{} contains decoupled plotting contributions. Rerun with the "
                "original peak table and grains_before_decoupling; keep the "
                "original strong table as cf_strong_before_decoupling and store "
                "the third return value as cf_strong_decoupled.".format(name)
            )
    confidence = kwargs.get("ransac_confidence", 0.999)
    if confidence is not None and not 0 < confidence < 1:
        raise ValueError("RANSAC confidence must be between zero and one, or None")
    batch = grain_ids is None
    if batch and kwargs.get("initialization", "spatial") != "trajectory":
        raise ValueError("Spatial decoupling requires manual expected counts; automatic sinusoid screening requires initialization='trajectory'")
    if batch and report is None:
        report = []
    pairs = [tuple(pair) for pair in twin_pairs]
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("Each twin pair must contain two existing grain IDs")
    paired = {gid for pair in pairs for gid in pair}
    ids = [gid for gid in range(len(grains)) if gid not in paired] if batch else list(grain_ids)
    component_counts = dict(grain_ids) if hasattr(grain_ids, "items") else dict.fromkeys(ids, None if batch else 2)
    if any(count is not None and (isinstance(count, (bool, np.bool_))
           or not isinstance(count, (int, np.integer)) or not 2 <= count <= 4)
           for count in component_counts.values()):
        raise ValueError("Component counts must be integers from two to four")
    entries = ids + [gid for pair in pairs for gid in pair]
    if any(
        not isinstance(gid, (int, np.integer))
        or isinstance(gid, bool)
        or gid < 0
        or gid >= len(grains)
        for gid in entries
    ) or len(set(entries)) != len(entries):
        raise ValueError(
            "Grain IDs must be valid, unique, and disjoint from twin pairs"
        )
    updated = list(grains)
    results = []
    owners = (
        _refine_peak_owners(grains, cf_4d, cf_strong, kwargs.get("hkl_tol", 0.05))
        if entries
        else None
    )
    work = [(gid,) for gid in ids] + pairs
    if batch:
        work = tqdm(work, desc="Screening grain sinograms", disable=not kwargs.get("verbose", True))
    for entry in work:
        record = dict(parent_ids=entry, child_ids=entry, n_components=None,
                      status=None, component_selection=None, error=None)
        peak_mask = np.isin(owners, entry) | (owners < 0)
        split_options = dict(kwargs)
        try:
            if batch and len(entry) == 1:
                selected, _ = _decouple_eligible_4d(
                    [grains[entry[0]]], cf_4d, kwargs.get("hkl_tol", 0.05), peak_mask)
                width = kwargs.get("width")
                margin = kwargs.get("margin")
                initial = _decouple_auto_components(
                    cf_4d.omega[selected], cf_4d.dty[selected],
                    2 * float(ds.ystep) if width is None else float(width),
                    float(ds.ystep) if margin is None else float(margin),
                    kwargs.get("y0"), kwargs.get("trials", 600),
                    kwargs.get("seed", 42), kwargs.get("max_components", 4),
                    kwargs.get("ransac_confidence", 0.999))
                selection = initial[3]
                record.update(n_components=len(initial[0]), component_selection=selection)
                if selection["ambiguous"] or selection["reached_limit"]:
                    record["status"] = "review"
                elif len(initial[0]) == 1:
                    record["status"] = "unchanged"
                if record["status"] is not None:
                    if report is not None:
                        report.append(record)
                    continue
                split_options["_initial_fit"] = initial
            result = split_grain_sinogram(
                ds,
                grains[entry[0]],
                cf_4d,
                twin_grain=grains[entry[1]] if len(entry) == 2 else None,
                peak_mask=peak_mask,
                n_components=component_counts[entry[0]] if len(entry) == 1 else 2,
                **split_options
            )
        except (ValueError, RuntimeError) as exc:
            if not batch:
                raise type(exc)("Decoupling parent IDs {}: {}".format(entry, exc)) from exc
            record.update(status="failed", error=str(exc))
            if report is not None:
                report.append(record)
            if kwargs.get("verbose", True):
                tqdm.write("Decoupling parent IDs {} failed: {}".format(entry, exc))
            continue
        targets = ((entry[0],) + tuple(range(len(updated), len(updated) + len(result.grainsinos) - 1))
                   if len(entry) == 1 else entry)
        result.parent_ids, result.child_ids = entry, targets
        for k, target in enumerate(targets):
            g = result.grainsinos[k].grain
            g.gid = target
            if target == len(updated):
                updated.append(g)
            else:
                updated[target] = g
        results.append(result)
        record.update(status="split", n_components=len(result.grainsinos), child_ids=targets)
        if report is not None:
            report.append(record)
    if results:
        rows = np.concatenate([result.source_2d_rows for result in results])
        fractions = np.concatenate(
            [result.peak_weights.sum(axis=1) for result in results]
        )
        _, inverse = np.unique(rows, return_inverse=True)
        overallocated = np.bincount(inverse, weights=fractions) > 1 + 1e-8
        if np.any(overallocated):
            conflicting_rows = np.unique(rows)[overallocated]
            conflicts = []
            for i, first in enumerate(results):
                first_rows = first.source_2d_rows[first.peak_weights.sum(axis=1) > 0]
                for second in results[i + 1 :]:
                    second_rows = second.source_2d_rows[
                        second.peak_weights.sum(axis=1) > 0
                    ]
                    shared = np.intersect1d(
                        np.intersect1d(first_rows, second_rows), conflicting_rows
                    )
                    if shared.size:
                        conflicts.append(
                            "{} and {} ({} rows)".format(
                                first.parent_ids, second.parent_ids, shared.size
                            )
                        )
            raise ValueError(
                "Selected decouplings share 2D intensity: {}. "
                "Use twin_pairs for known, distinct twin orientations and remove "
                "those IDs from grain_ids; otherwise inspect duplicate orientations "
                "and hkl_tol. Input grains are unchanged.".format("; ".join(conflicts))
            )
    if batch and kwargs.get("verbose", True):
        print("Decoupling batch: {}".format({
            status: sum(row["status"] == status for row in report)
            for status in ("split", "unchanged", "review", "failed")}))
    if cf_strong is not None:
        return updated, results, decoupled_strong_columnfile(ds, cf_strong, results)
    return updated, results


def decoupled_strong_columnfile(ds, cf_strong, results):
    """Compatibility wrapper for merged fractional plotting contributions."""
    return refined_strong_columnfile(ds, cf_strong, results)


def refined_strong_columnfile(ds, cf_strong, results):
    """Remerge refined 2D contributions into the original strong 4D subset.

    Rows assigned to a result's parents and recovered unassigned rows are
    replaced. Centroids use
    intensity times split fraction, following pk2dmerge's arithmetic mean
    convention. ``source_4d_row`` points to cf_strong, not ds.pk2d.
    ``split_fraction`` is the retained fraction of the original intensity.
    """
    if "grain_id" not in cf_strong.titles:
        raise ValueError("cf_strong must contain grain_id assignments")
    spot_ids, order = np.unique(cf_strong.spot3d_id, return_index=True)
    if len(spot_ids) != cf_strong.nrows:
        raise ValueError("cf_strong must contain unique spot3d_id values")
    parents = [gid for result in results for gid in result.parent_ids]
    source_spots = ds.pk2d["spot3d_id"]
    claimed_spots = [
        source_spots[result.source_2d_rows[result.peak_weights.sum(axis=1) > 0]]
        for result in results
    ]
    claimed_spots = np.concatenate(claimed_spots) if claimed_spots else np.empty(0)
    rebuilt = np.isin(cf_strong.grain_id, parents) | (
        (cf_strong.grain_id < 0) & np.isin(cf_strong.spot3d_id, claimed_spots)
    )
    unchanged = np.flatnonzero(~rebuilt)
    source_rows = list(unchanged)
    assignments = []
    for result in results:
        claimed = source_spots[
            result.source_2d_rows[result.peak_weights.sum(axis=1) > 0]
        ]
        parent_rows = np.flatnonzero(
            np.isin(cf_strong.grain_id, result.parent_ids)
            | ((cf_strong.grain_id < 0) & np.isin(cf_strong.spot3d_id, claimed))
        )
        eligible = np.isin(
            source_spots[result.source_2d_rows], cf_strong.spot3d_id[parent_rows]
        )
        for k, gid in enumerate(result.child_ids):
            used = eligible & (result.peak_weights[:, k] > 0)
            rows = result.source_2d_rows[used]
            fractions = result.peak_weights[used, k]
            strong_rows = order[np.searchsorted(spot_ids, source_spots[rows])]
            grouped, boundaries = _refine_group_rows(strong_rows)
            for start, stop in zip(boundaries[:-1], boundaries[1:]):
                if start == stop:
                    continue
                indices = grouped[start:stop]
                source_rows.append(int(strong_rows[indices[0]]))
                assignments.append((gid, rows[indices], fractions[indices]))
    source_rows = np.asarray(source_rows, dtype=np.int64)
    cf = cf_strong.copyrows(source_rows)
    cf.addcolumn(source_rows, "source_4d_row")
    cf.addcolumn(np.ones(len(source_rows)), "split_fraction")
    centroid_columns = [
        name
        for name in ("omega", "dty", "s_raw", "f_raw", "sc", "fc")
        if name in cf.titles and name in ds.pk2d
    ]
    for target, (gid, rows, fractions) in enumerate(assignments, start=len(unchanged)):
        intensity = ds.pk2d["sum_intensity"][rows] * fractions
        total = intensity.sum()
        cf.grain_id[target] = gid
        cf.sum_intensity[target] = total
        original = cf_strong.sum_intensity[source_rows[target]]
        cf.split_fraction[target] = total / original if original > 0 else 0
        for name in centroid_columns:
            cf[name][target] = np.average(ds.pk2d[name][rows], weights=intensity)
        if "npk2d" in cf.titles:
            cf.npk2d[target] = len(rows)
        if "Number_of_pixels" in cf.titles and "Number_of_pixels" in ds.pk2d:
            cf.Number_of_pixels[target] = ds.pk2d["Number_of_pixels"][rows].sum()
    # Recompute detector correction/geometry for the changed merged centroids.
    if assignments and "s_raw" in centroid_columns and "f_raw" in centroid_columns:
        peaks = {name: cf[name].copy() for name in cf.titles}
        corrected = ds.get_colfile_from_peaks_dict(peaks_dict=peaks)
        corrected.parameters = copy.deepcopy(cf_strong.parameters)
        corrected.updateGeometry()
        # Preserve every original value for untouched rows.
        for name in cf_strong.titles:
            corrected[name][: len(unchanged)] = cf_strong[name][unchanged]
        cf = corrected
    return cf


def decoupled_plot_columnfile(ds, cf_strong, results):
    """Build a 2D plot table for all grains without modifying input peaks.

    ``source_2d_row`` indexes ds.pk2d; ``split_fraction`` records each
    contribution. ``sum_intensity`` already includes that fraction.
    """
    if "grain_id" not in cf_strong.titles:
        raise ValueError("cf_strong must contain grain_id assignments")
    spot_ids, order = np.unique(cf_strong.spot3d_id, return_index=True)
    if len(spot_ids) != cf_strong.nrows:
        raise ValueError("cf_strong must contain unique spot3d_id values")
    source_spots = ds.pk2d["spot3d_id"]
    rows = np.flatnonzero(np.isin(source_spots, spot_ids))
    grain_ids = cf_strong.grain_id[order[np.searchsorted(spot_ids, source_spots[rows])]]
    parents = [gid for result in results for gid in result.parent_ids]
    claimed = [
        result.source_2d_rows[result.peak_weights.sum(axis=1) > 0] for result in results
    ]
    claimed = np.concatenate(claimed) if claimed else np.empty(0, dtype=np.int64)
    keep = ~np.isin(grain_ids, parents) & ~np.isin(rows, claimed)
    row_parts = [rows[keep]]
    id_parts = [grain_ids[keep]]
    fraction_parts = [np.ones(keep.sum())]
    for result in results:
        for k, gid in enumerate(result.child_ids):
            fractions = result.peak_weights[:, k]
            used = fractions > 0
            row_parts.append(result.source_2d_rows[used])
            id_parts.append(np.full(used.sum(), gid, dtype=np.int64))
            fraction_parts.append(fractions[used])
    rows = np.concatenate(row_parts).astype(np.int64)
    fractions = np.concatenate(fraction_parts)
    cf = _refine_colfile_from_rows(ds, rows, cf_strong.parameters)
    cf.addcolumn(np.concatenate(id_parts).astype(np.int64), "grain_id")
    cf.addcolumn(fractions, "split_fraction")
    cf.addcolumn(cf.sum_intensity * fractions, "sum_intensity")
    return cf


def plot_decoupled_sinograms(results):
    """Compatibility wrapper for nbGui.nb_utils.plot_decoupled_sinograms."""
    from ImageD11.nbGui.nb_utils import plot_decoupled_sinograms

    return plot_decoupled_sinograms(results)


def save_refined_grains(ds, grains, result, phase_name=None):
    """Atomically export a refinement result and fractions for later mapping."""
    return _save_grain_peak_assignments(ds, grains, [result], phase_name)


def save_decoupled_grains(ds, grains, results, phase_name=None):
    """Atomically export spatial/twin splits and fractions for later mapping."""
    return _save_grain_peak_assignments(ds, grains, results, phase_name)


def _save_grain_peak_assignments(ds, grains, results, phase_name=None):
    """Export the phase and fractional sinograms, backing up existing data.

    Replaces this phase's indexed grain group, like a fresh indexing export;
    other phases/groups are preserved. Existing maps of this phase become stale.
    Fractional provenance allows prepare_peaks_from_4d() to reuse the split.
    """
    for result in results:
        if len(result.child_ids) != len(result.grainsinos):
            raise ValueError("Expected one output grain ID per sinogram")
    phase = "grains" if phase_name is None else phase_name
    path = Path(ds.grainsfile)
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = path.with_name(path.name + ".before_decoupling_" + stamp + ".bak")
        shutil.copy2(path, backup)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".h5", dir=path.parent
    )
    os.close(fd)
    try:
        if path.exists():
            shutil.copy2(path, temporary)
        with h5py.File(temporary, "a") as fout:
            previous = fout[phase] if phase in fout else None
            if phase in fout:
                del fout[phase]
            group = fout.create_group(phase)
            refined_ids = {gid for result in results for gid in result.child_ids}
            for gid, g in enumerate(grains):
                old = previous.get(str(gid)) if previous is not None else None
                identity_matches = (
                    isinstance(old, h5py.Group)
                    and "ubi" in old
                    and np.allclose(old["ubi"][:], g.ubi, rtol=1e-9, atol=1e-10)
                )
                if identity_matches and g.translation is not None:
                    identity_matches = "translation" in old and np.allclose(
                        old["translation"][:], g.translation, rtol=1e-9, atol=1e-10
                    )
                if gid not in refined_ids and identity_matches:
                    fout.copy(old, group, name=str(gid))
                else:
                    g.to_h5py_group(group, str(gid))
            written = set()
            for result in results:
                for k, gid in enumerate(result.child_ids):
                    if gid in written or not 0 <= gid < len(grains):
                        raise ValueError("Invalid or duplicate output grain IDs")
                    gs = result.grainsinos[k]
                    if not np.allclose(grains[gid].ubi, gs.grain.ubi):
                        raise ValueError(
                            "Peak refinement result does not match output grain list"
                        )
                    translation = grains[gid].translation
                    result_translation = gs.grain.translation
                    if (translation is None) != (result_translation is None) or (
                        translation is not None
                        and not np.allclose(
                            translation, result_translation, rtol=1e-9, atol=1e-10
                        )
                    ):
                        raise ValueError(
                            "Peak refinement result does not match output grain position"
                        )
                    gs.to_h5py_group(group, str(gid))
                    kind = gs.peak_assignment_kind or "split"
                    prefix = "split" if kind == "split" else "refinement"
                    group[str(gid)].attrs[prefix + "_parent_ids"] = result.parent_ids
                    group[str(gid)].attrs[prefix + "_converged"] = result.converged
                    written.add(gid)
            for gid, g in enumerate(grains):
                if not np.allclose(group[str(gid)]["ubi"][:], g.ubi):
                    raise RuntimeError("Staged grain verification failed")
            assignments = [
                g["split_peak_assignment"]
                for g in group.values()
                if "split_peak_assignment" in g
            ]
            if assignments:
                checked = [
                    _validate_peak_assignment(
                        ds, a["source_2d_row"][:], a["intensity_fraction"][:]
                    )
                    for a in assignments
                ]
                rows = np.concatenate([rows for rows, _ in checked])
                fractions = np.concatenate([fractions for _, fractions in checked])
                _, inverse = np.unique(rows, return_inverse=True)
                if np.any(np.bincount(inverse, weights=fractions) > 1 + 1e-8):
                    raise ValueError(
                        "Staged grain assignments double-count measured 2D intensity"
                    )
            tmap_name = "TensorMap_" + phase
            if tmap_name in fout:
                fout[tmap_name].attrs["requires_reconstruction_after_decoupling"] = True
                fout[tmap_name].attrs[
                    "requires_reconstruction_after_peak_refinement"
                ] = True
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return backup
