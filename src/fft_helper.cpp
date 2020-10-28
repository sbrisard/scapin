#include "scapin/fft_helper.hpp"

namespace fft_helper {

double *fftfreq(int n, double d, double *freq) {
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

double *fftwavnum(int n, double L, double *k) {
  if (k == nullptr) {
    k = new double[n];
  }
  int const m = n / 2;
  int const rem = n % 2;
  double const delta_k = 2 * M_PI / (n * L);
  for (int i = 0; i < m + rem; i++) {
    k[i] = delta_k * i;
  }
  for (int i = m + rem; i < n; i++) {
    k[i] = -delta_k * (n - i);
  }
  return k;
}

}  // namespace fft_helper
