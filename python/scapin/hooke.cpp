#define _USE_MATH_DEFINES

#include <cmath>
#include <complex>
#include <iostream>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "scapin/hooke.hpp"

namespace py = pybind11;

using float64 = double;
using complex128 = std::complex<float64>;

template <typename T, size_t DIM>
auto create_binding(py::module m, const char* name) {
  using Hooke = scapin::Hooke<T, DIM>;
  return py::class_<Hooke>(m, name)
      .def(py::init<double, double>())
      .def_readonly_static("dim", &Hooke::dim)
      .def_readonly_static("isize", &Hooke::isize)
      .def_readonly_static("osize", &Hooke::osize)
      .def_readonly("mu", &Hooke::mu)
      .def_readonly("nu", &Hooke::nu)
      .def_property_readonly_static(
          "dtype", [](py::object) { return py::dtype::of<T>(); })
      .def("__repr__", &Hooke::repr)
      .def("apply", [](Hooke& self, py::array_t<double, py::array::c_style> k,
                       py::array_t<T, py::array::c_style> tau,
                       py::array_t<T, py::array::c_style> out) {
        self.apply(k.data(), tau.data(), out.mutable_data());
      });  // TODO: check that stride==1, ndim==1, shape>=isize, etc.
}

PYBIND11_MODULE(hooke, m) {
  m.doc() = "Python bindings to the scapin/hooke.hpp header-only library";

  m.attr("__version__") = py::cast(__SCAPIN_VERSION__);
  m.attr("__author__") = py::cast(__SCAPIN_AUTHOR__);

  create_binding<float64, 2>(m, "HookeFloat64_2D");
  create_binding<float64, 3>(m, "HookeFloat64_3D");
  create_binding<complex128, 2>(m, "HookeComplex128_2D");
  create_binding<complex128, 3>(m, "HookeComplex128_3D");
}
