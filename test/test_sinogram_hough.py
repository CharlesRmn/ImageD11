"""Shared Hough voting used by missing-grain recovery."""
from __future__ import print_function

import numpy as np
import pytest

from ImageD11.sinograms import sinogram as sino


@pytest.mark.parametrize("angle_start", [-45., 315.])
def test_hough_votes_recover_centre_across_zero_and_axis_aligned_angles(angle_start):
    omega = np.arange(angle_start, angle_start + 180., 1.)
    centre = np.array([.2, -.1])
    design = sino._refine_design(omega)
    target = np.dot(design, centre)
    votes = sino._hough_votes(
        design, target, np.ones(len(omega)), np.array([-.5, -.5]), .01, 101)
    row, col = np.unravel_index(np.argmax(votes), votes.shape)
    np.testing.assert_allclose(np.array([row, col]) * .01 - .5, centre, atol=.011)
    assert np.isfinite(votes).all() and np.all(votes >= 0)
