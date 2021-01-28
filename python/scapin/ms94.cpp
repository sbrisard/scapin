#include <iostream>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "scapin/hooke.hpp"
#include "scapin/ms94.hpp"

namespace py = pybind11;

using float64 = double;
using complex128 = std::complex<float64>;

template <typename T>
using PyArray = py::array_t<T, py::array::c_style>;

template <typename T, int SIZE>
auto make_tuple_from_std_array(std::array<T, SIZE> a) {
  static_assert((SIZE >= 0) || (SIZE < 5), "unexpected size");
  if constexpr (SIZE == 0)
    return py::make_tuple();
  else if constexpr (SIZE == 1)
    return py::make_tuple(a[0]);
  else if constexpr (SIZE == 2)
    return py::make_tuple(a[0], a[1]);
  else if constexpr (SIZE == 3)
    return py::make_tuple(a[0], a[1], a[2]);
  else if constexpr (SIZE == 4)
    return py::make_tuple(a[0], a[1], a[2], a[3]);
}

template <typename GREENC>
auto create_ms94(GREENC& greenc, py::array_t<int> N, py::array_t<double> L) {
  scapin::MoulinecSuquet94<GREENC> greend{greenc, N.data(), L.data()};
  return greend;
}

template <typename GREENC>
auto create_binding(py::module m, const char* name) {
  using MS94 = scapin::MoulinecSuquet94<GREENC>;
  using Scalar = typename GREENC::Scalar;
  return py::class_<MS94>(m, name)
      .def(py::init(&create_ms94<GREENC>))
      .def_readonly("gamma", &MS94::gamma)
      .def_property_readonly(
          "N",
          [](const MS94& self) { return make_tuple_from_std_array(self.N); })
      .def_property_readonly(
          "L",
          [](const MS94& self) { return make_tuple_from_std_array(self.L); })
      .def("__repr__", &MS94::repr)
      .def("apply", [](MS94& self, PyArray<Scalar> tau, PyArray<Scalar> out) {
        self.apply(tau.data(), out.mutable_data());
      });
}

PYBIND11_MODULE(ms94, m) {
  m.doc() = "";

  create_binding<scapin::Hooke<float64, 2>>(m,
                                            "MoulinecSuquet94HookeFloat64_2D");
  create_binding<scapin::Hooke<float64, 3>>(m,
                                            "MoulinecSuquet94HookeFloat64_3D");
  create_binding<scapin::Hooke<complex128, 2>>(
      m, "MoulinecSuquet94HookeComplex128_2D");
  create_binding<scapin::Hooke<complex128, 3>>(
      m, "MoulinecSuquet94HookeComplex128_3D");
}
