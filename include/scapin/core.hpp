#pragma once

#if _WIN32
#define DllExport __declspec(dllexport)
#else
#define DllExport
#endif

namespace scapin {
template <typename T>
int dimensionality();

template <typename Iterator>
std::string repr(Iterator first, Iterator last) {
  std::ostringstream stream;
  stream << "{";
  for (auto i = first; i != last; i++) stream << *i << ",";
  stream << "}";
  return stream.str();
}

}  // namespace scapin