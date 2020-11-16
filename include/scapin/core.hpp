#pragma once

#if _WIN32
#define DllExport __declspec(dllexport)
#else
#define DllExport
#endif

namespace scapin {
template <typename T>
int dimensionality();
}