#pragma once

#if _WIN32
#define DllExport __declspec(dllexport)
#else
#define DllExport
#endif

template <typename T>
size_t dimensionality();
