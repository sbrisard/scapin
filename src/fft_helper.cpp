#include <scapin/fft_helper.hpp>

double *fft_helper_fftfreq(std::size_t n, double d, double *freq) {
  if (freq == nullptr) {
    freq = new double[n];
  }
  std::size_t const m = n / 2;
  std::size_t const rem = n % 2;
  double const f = 1. / (d * n);
  for (std::size_t i = 0; i < m + rem; i++) {
    freq[i] = f * i;
  }
  for (std::size_t i = m + rem; i < n; i++) {
    freq[i] = -f * (n - i);
  }
  return freq;
}
