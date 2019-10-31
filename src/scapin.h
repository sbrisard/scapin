#ifndef __SCAPIN_H__
#define __SCAPIN_H__

#include <glib.h>
#include <math.h>
#include <stdlib.h>

#if _WIN32
#define DllExport __declspec(dllexport)
#else
#define DllExport
#endif

typedef struct ScapinGreenOperatorType_ ScapinGreenOperatorType;
typedef struct ScapinGreenOperator_ ScapinGreenOperator;

typedef void scapin_grop_dispose_t();
typedef void scapin_grop_apply_t(ScapinGreenOperator *, double *, double *);

struct ScapinGreenOperatorType_ {
  char *name;
  size_t ndims;
  size_t isize;
  size_t osize;
  scapin_grop_apply_t *apply;
  scapin_grop_dispose_t *dispose;
};

struct ScapinGreenOperator_ {
  ScapinGreenOperatorType *type;
  void *data;
};

DllExport ScapinGreenOperator *scapin_grop_new();
DllExport void scapin_grop_free(ScapinGreenOperator *);

DllExport ScapinGreenOperator *scapin_grop_hooke_new(size_t, double, double);
DllExport double scapin_grop_hooke_mu(ScapinGreenOperator *op);
DllExport double scapin_grop_hooke_nu(ScapinGreenOperator *op);

#endif
