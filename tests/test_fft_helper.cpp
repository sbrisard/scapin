#include <scapin/fft_helper.hpp>
#include "catch2/catch.hpp"

void test_fftfreq(std::size_t n, double d, bool inplace, int cycles[]) {
  double *out = inplace ? new double[n] : nullptr;
  double *act = fft_helper_fftfreq(n, d, out);
  if (inplace) {
    REQUIRE(act == out);
  }
  for (size_t i = 0; i < n; i++) {
    double exp_i = cycles[i] / (d * n);
    REQUIRE(exp_i == act[i]);
  }
  delete [] out;
}

void test_fftfreq(std::size_t n, double d, int cycles[]) {
  SECTION("in-place output") { test_fftfreq(n, d, true, cycles); }
  SECTION("out-of-place output") { test_fftfreq(n, d, false, cycles); }
}

TEST_CASE("fftfreq") {
  double d = 1.0;

  SECTION("even input") {
    std::size_t n = 8;
    int cycles[] = {0, 1, 2, 3, -4, -3, -2, -1};
    test_fftfreq(n, d, cycles);
  }

  SECTION("odd input") {
    std::size_t n = 9;
    int cycles[] = {0, 1, 2, 3, 4, -4, -3, -2, -1};
    test_fftfreq(n, d, cycles);
  }
}
