#define _USE_MATH_DEFINES

#include <array>
#include <cmath>
#include <complex>
#include <iostream>

#include <fftw3.h>
#include <numeric>

#include "scapin/ms94.hpp"
#include "scapin/scapin.hpp"

using complex128 = std::complex<double>;

class FFTWComplexBuffer {
 public:
  const size_t size;
  fftw_complex *const c_data;
  complex128 *const cpp_data;

  FFTWComplexBuffer(size_t n)
      : size(n),
        c_data((fftw_complex *)fftw_malloc(n * sizeof(fftw_complex))),
        cpp_data(reinterpret_cast<complex128 *const>(c_data)) {}

  ~FFTWComplexBuffer() { fftw_free(c_data); }
  FFTWComplexBuffer(const FFTWComplexBuffer &) = delete;
  FFTWComplexBuffer &operator=(const FFTWComplexBuffer &) = delete;
  FFTWComplexBuffer(const FFTWComplexBuffer &&) = delete;
  FFTWComplexBuffer &operator=(const FFTWComplexBuffer &&) = delete;
};

/**
 * Compute in Fourier space the nearest-neighbour interpolation kernel.
 *
 * Let `xc[0], ..., xc[nc-1]` be data points defined on the coarse grid (size:
 * `nc`). The nearest-neighbour interpolation `xf` is defined on the fine grid
 * (size: `nf = r * nc`) as follows: `xf[i] = xc[i / r]`.
 *
 * In Fourier space, this formula reads: `yf[i] = kernel[i] * yc[i % nc]`,
 * where `yc` and `yf` are the DFT of `xc` and `xf`, respectively.
 *
 * If `kernel == nullptr`, it is allocated. Otherwise, it must be a preallocated
 * `std::complex<double>[nf]` array.
 *
 * @param nc the size of the coarse grid
 * @param r the refinement factor
 * @param kernel the coefficients of the kernel
 * @returns kernel
 */
complex128 *fft_nninterp(size_t nc, size_t r, complex128 *kernel) {
  const size_t nf = r * nc;
  if (kernel == nullptr) {
    kernel = new complex128[nf];
  }
  kernel[0] = r;
  for (size_t i = 1; i < nf; i++) {
    const double xc = i * M_PI / (double)nc;
    const double xf = i * M_PI / (double)nf;
    const complex128 z{cos((r - 1) * xf), -sin((r - 1) * xf)};
    kernel[i] = sin(xc) / sin(xf) * z;
  }
  return kernel;
}

template <typename T>
class Range {
 public:
  const T start;
  const T end;
  Range(T start, T end) : start(start), end(end) {}
  Range(T end) : start(0), end(end) {}

  std::string repr() const {
    std::ostringstream stream;
    stream << "Range<" << typeid(T).name() << ">(start=" << start << ","
           << "end=" << end << ")";
    return stream.str();
  }
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const Range<T> &range) {
  return os << range.repr();
}

template <typename T, size_t DIM>
class CartesianIndices {
 private:
  static std::array<T, DIM> init_multi_index(std::array<Range<T>, DIM> ranges) {
    std::array<T, DIM> multi_index;
    for (size_t k = 0; k < DIM; k++) {
      multi_index[k] = ranges[k].start;
    }
    return multi_index;
  }

 public:
  typedef std::array<T, DIM> MultiIndex;
  const std::array<Range<T>, DIM> ranges;
  CartesianIndices(std::array<Range<T>, DIM> ranges) : ranges(ranges){};

  std::string repr() const {
    std::ostringstream stream;
    stream << "CartesianIndices<" << typeid(T).name() << "," << DIM << ">(";
    for (auto range : ranges) {
      stream << range << ",";
    }
    stream << ")";
    return stream.str();
  }

  class CartesianIndex {
   private:
    std::array<T, DIM> multi_index;
    T linear_index;

   public:
    CartesianIndex() : multi_index(init_multi_index(ranges)){};

    CartesianIndex(const std::array<T, DIM> multi_index)
        : multi_index(multi_index){};

    std::string repr() const {
      std::ostringstream stream;
      stream << "CartesianIndex<" << typeid(T).name() << "," << DIM;
      for (auto i_k : multi_index) {
        stream << i_k << ",";
      }
      return stream.str();
    }

    CartesianIndex &operator++() {
      size_t k = DIM - 1;
      for (size_t kk = 0; kk < DIM; ++kk, --k) {
        multi_index[k] += 1;
        if (multi_index[k] < ranges[k].end) {
          break;
        }
        multi_index[k] = 0;
      }
      return *this;
    }
  };

  void iterate() {
    auto index = CartesianIndex();
    size_t num_iters{1};
    for (size_t k = 0; k < DIM; k++) {
      num_iters *= ranges[k].end - ranges[k].start;
    }
    for (size_t iter = 0; iter < num_iters; iter++) {
      ++index;
      std::cout << index << std::endl;
    }
  }
};

template <typename T, size_t DIM>
std::ostream &operator<<(std::ostream &os,
                         const CartesianIndices<T, DIM> &indices) {
  return os << indices.repr();
}

template <typename T, size_t DIM>
std::ostream &operator<<(
    std::ostream &os, const CartesianIndices<T, DIM>::CartesianIndex &index) {
  return os << index.repr();
}

int main() {
  std::array<Range<size_t>, 2> ranges{Range<size_t>(4), Range<size_t>(3)};
  for (auto range : ranges) {
    std::cout << range << std::endl;
  }
  CartesianIndices<size_t, 2> indices{ranges};
  std::cout << indices << std::endl;
  indices.iterate();

  Hooke<complex128, 2> gamma{1.0, 0.3};
  using MultiIndex = std::array<size_t, gamma.dim>;

  std::array<double, gamma.dim> patch_ratio;
  patch_ratio.fill(0.125);

  /*
   * tau_in: value of the prescribed polarization inside the patch
   * tau_out: value of the prescribed polarization outside the patch
   */
  std::array<double, gamma.isize> tau_in, tau_out;
  tau_in.fill(0.);
  tau_in[gamma.isize - 1] = 1.;
  tau_out.fill(0.);

  std::array<double, gamma.dim> L{1., 1.};
  MultiIndex grid_size{16, 16};

  size_t num_cells = std::accumulate(grid_size.cbegin(), grid_size.cend(),
                                     std::size_t{1}, std::multiplies<size_t>());
  FFTWComplexBuffer tau{num_cells * gamma.isize};

  // TODO: This is not dimension independent
  for (size_t i0 = 0; i0 < grid_size[0]; i0++) {
    for (size_t i1 = 0; i1 < grid_size[1]; i1++) {
    }
  }
  //  std::array<double, dim> L = {1.0, 1.0};
  //  MultiIndex Nc = {16, 16};
  //
  //  size_t num_refinements = 6;
  //  size_t r_max = 1 << num_refinements;
  //  MultiIndex Nf = {r_max * Nc[0], r_max * Nc[1]};
  //
  //  FFTWComplexBuffer tau_c{Nc[0] * Nc[1] * gamma.isize};
  //  tau_c.cpp_data[0] = 1.;
  //  for (size_t i = 0; i < tau_c.size; i++) tau_c.cpp_data[i] = 0.;
  //
  //  for (size_t r = 1; r <= r_max; r *= 2) {
  //    MultiIndex N = {r * Nc[0], r * Nc[1]};
  //    MoulinecSuquet94<decltype(gamma)> gamma_h{gamma, N, L};
  //    std::cout << gamma_h << std::endl;
  //
  //    int n[] = {static_cast<int>(N[0]), static_cast<int>(N[1])};
  //    FFTWComplexBuffer tau{N[0] * N[1] * gamma.isize};
  //    FFTWComplexBuffer eps{N[0] * N[1] * gamma.osize};
  //    auto p = fftw_plan_many_dft(dim, n, gamma.isize, tau.c_data, nullptr,
  //                                gamma.isize, 1, tau.c_data, nullptr,
  //                                gamma.isize, 1, FFTW_FORWARD,
  //                                FFTW_ESTIMATE);
  //
  //    for (MultiIndex i{0}, ic{0}; i[0] < N[0]; i[0]++) {
  //      ic[0] = i[0] * Nc[0] / N[0];
  //      for (i[1] = 0; i[1] < N[1]; i[1]++) {
  //        ic[1] = i[1] * Nc[1] / N[1];
  //        for (size_t j = 0; j < gamma.isize; j++) {
  //          auto adr = (i[0] * N[1] + i[1]) * gamma.isize + j;
  //          auto adr_c = (ic[0] * Nc[1] + ic[1]) * gamma.isize + j;
  //          tau.cpp_data[adr] = tau_c.cpp_data[adr_c];
  //        }
  //      }
  //    }
  //
  //    fftw_execute(p);
  //    fftw_destroy_plan(p);
  //
  //    auto tau_i = tau.cpp_data;
  //    auto eps_i = eps.cpp_data;
  //    for (MultiIndex i{0}; i[0] < N[0]; i[0]++) {
  //      for (i[1] = 0; i[1] < N[1]; i[1]++) {
  //        gamma_h.apply(i.data(), tau_i, eps_i);
  //        tau_i += gamma.isize;
  //        eps_i += gamma.osize;
  //      }
  //    }
  //
  //    auto p2 = fftw_plan_many_dft(dim, n, gamma.osize, eps.c_data, nullptr,
  //                                 gamma.osize, 1, eps.c_data, nullptr,
  //                                 gamma.osize, 1, FFTW_BACKWARD,
  //                                 FFTW_ESTIMATE);
  //    fftw_execute(p2);
  //    fftw_destroy_plan(p2);
  //  }
}
