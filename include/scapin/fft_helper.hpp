/**
 * @file fft_utils.h
 *
 * @brief Helper functions to work with discrete Fourier transforms.
 *
 * This module is heavily inspired by the `numpy.fft.helper` module, see
 *
 * <https://docs.scipy.org/doc/numpy/reference/routines.fft.html#helper-routines>
 *
 */
#pragma once

#include "core.hpp"

/**
 * @brief Return the Discrete Fourier Transform sample frequencies.
 *
 * The returned float array `freq` contains the frequency bin centers in cycles
 * per unit of the sample spacing (with zero at the start). For instance, if the
 * sample spacing is in seconds, then the frequency unit is cycles/second.
 *
 * Given a window length `n` and a sample spacing `d`
 *
 *     f = [0, 1, ..., n/2-1, -n/2, ..., -1] / (d*n) if n is even
 *     f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n) if n is odd
 *
 * If `freq == NULL`, then a new `double[n]` array is allocated and
 * returned. Otherwise, `freq` must be a preallocated `double[n]` array, which
 * is modified in place and returned.
 */
DllExport double *fft_helper_fftfreq(size_t n, double d, double *freq);

