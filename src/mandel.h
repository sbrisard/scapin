#pragma once

#if _WIN32
#define DllExport __declspec(dllexport)
#else
#define DllExport
#endif

#include <math.h>

DllExport void mandel3d_tens2_dot_vec(double const *, double const *, double *);
