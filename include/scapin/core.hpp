#pragma once

#if _WIN32
#define DllExport __declspec(dllexport)
#else
#define DllExport
#endif

namespace scapin {
template <int DIM>
concept spatial_dimension = (DIM == 2) || (DIM == 3);

template <class T>
struct remove_complex {
  typedef T type;
};

template <class T>
struct remove_complex<std::complex<T>> {
  typedef T type;
};

template <class T>
using remove_complex_t = typename remove_complex<T>::type;

template <typename Iterator>
std::string repr(Iterator first, Iterator last) {
  std::ostringstream stream;
  stream << "{";
  for (auto i = first; i != last; i++) stream << *i << ",";
  stream << "}";
  return stream.str();
}

}  // namespace scapin