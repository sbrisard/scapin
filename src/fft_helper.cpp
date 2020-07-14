#include "scapin/fft_helper.hpp"

namespace fft_helper {

double *fftfreq(size_t n, double d, double *freq) {
  if (freq == nullptr) {
    freq = new double[n];
  }
  size_t const m = n / 2;
  size_t const rem = n % 2;
  double const f = 1. / (d * n);
  for (size_t i = 0; i < m + rem; i++) {
    freq[i] = f * i;
  }
  for (size_t i = m + rem; i < n; i++) {
    freq[i] = -f * (n - i);
  }
  return freq;
}

double *fftwavnum(size_t n, double L, double *k) {
  if (k == nullptr) {
    k = new double[n];
  }
  size_t const m = n / 2;
  size_t const rem = n % 2;
  double const delta_k = 2 * M_PI / (n * L);
  for (size_t i = 0; i < m + rem; i++) {
    k[i] = delta_k * i;
  }
  for (size_t i = m + rem; i < n; i++) {
    k[i] = -delta_k * (n - i);
  }
  return k;
}

}  // namespace fft_helper
