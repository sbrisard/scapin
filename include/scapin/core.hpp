#pragma once

#if _WIN32
#define DllExport __declspec(dllexport)
#else
#define DllExport
#endif

namespace scapin {
template <int DIM>
concept spatial_dimension = (DIM == 2) || (DIM == 3);

template <typename Iterator>
std::string repr(Iterator first, Iterator last) {
  std::ostringstream stream;
  stream << "{";
  for (auto i = first; i != last; i++) stream << *i << ",";
  stream << "}";
  return stream.str();
}

}  // namespace scapin