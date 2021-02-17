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

#include <cmath>
#include <numbers>

#include "core.hpp"

namespace fft_helper {

/**
 * Return the Discrete Fourier Transform sample frequencies.
 *
 * The returned float array `freq` contains the frequency bin centers in cycles
 * per unit of the sample spacing (with zero at the start). For instance, if the
 * sample spacing is in seconds, then the frequency unit is cycles/second.
 *
 * Given a window length `n` and a sample spacing `d`
 *
 *     f = [0, 1, ..., n/2-1, -n/2, ..., -1] / (d * n) if n is even
 *     f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d * n) if n is odd
 *
 * If `freq == nullptr`, then a `new double[n]` array is allocated and
 * returned. Otherwise, `freq` must be a preallocated `double[n]` array, which
 * is modified in place and returned.
 */
DllExport double *fftfreq(int n, double d, double *freq) {
  if (freq == nullptr) {
    freq = new double[n];
  }
  int const m = n / 2;
  int const rem = n % 2;
  double const f = 1. / (d * n);
  for (int i = 0; i < m + rem; i++) {
    freq[i] = f * i;
  }
  for (int i = m + rem; i < n; i++) {
    freq[i] = -f * (n - i);
  }
  return freq;
}

/**
 * Return the Discrete Fourier Transform sample wavenumbers.
 *
 * The returned float array `k` contains the wavenumbers bin centers in radians
 * per unit of the L (with zero at the start). For instance, if the
 * sample L is in seconds, then the frequency unit is radians/second.
 *
 * Given a window length `n` and the sample `L`
 *
 *     k = [0, 1, ..., n/2-1, -n/2, ..., -1] * Δk if n is even
 *     k = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] * Δk if n is odd
 *
 * where `Δk = 2π / (n * L)`.
 *
 * If `k == nullptr`, then a `new double[n]` array is allocated and
 * returned. Otherwise, `k` must be a preallocated `double[n]` array, which
 * is modified in place and returned.
 */
DllExport double *fftwavnum(int n, double L, double *k) {
  if (k == nullptr) {
    k = new double[n];
  }
  int const m = n / 2;
  int const rem = n % 2;
  double const delta_k = 2 * std::numbers::pi / (n * L);
  for (int i = 0; i < m + rem; i++) {
    k[i] = delta_k * i;
  }
  for (int i = m + rem; i < n; i++) {
    k[i] = -delta_k * (n - i);
  }
  return k;
}
}  // namespace fft_helper