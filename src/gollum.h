#ifndef __GOLLUM_H__
#define __GOLLUM_H__

#include <glib.h>
#include <math.h>

#if _WIN32
#define DllExport __declspec(dllexport)
#else
#define DllExport
#endif

DllExport void green_apply(double *, double *, double, double, double *);

#endif
