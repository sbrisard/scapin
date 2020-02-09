#ifndef __SCAPIN_H__
#define __SCAPIN_H__

#if _WIN32
#define DllExport __declspec(dllexport)
#else
#define DllExport
#endif

typedef struct ScapinGreenOperatorType_ ScapinGreenOperatorType;
typedef struct ScapinGreenOperator_ ScapinGreenOperator;

typedef void scapin_grop_dispose_t();
typedef void scapin_grop_apply_t(ScapinGreenOperator const *, double const *,
                                 double const *, double *);

struct ScapinGreenOperatorType_ {
  char const *name;
  size_t const dim;
  size_t const isize;
  size_t const osize;
  scapin_grop_apply_t  * const apply;
  scapin_grop_dispose_t * const dispose;
};

struct ScapinGreenOperator_ {
  ScapinGreenOperatorType const *type;
  void *data;
};

DllExport ScapinGreenOperator *scapin_grop_new();
DllExport void scapin_grop_free(ScapinGreenOperator *);

DllExport ScapinGreenOperator *scapin_grop_hooke_new(size_t, double, double);
DllExport double scapin_grop_hooke_mu(ScapinGreenOperator *op);
DllExport double scapin_grop_hooke_nu(ScapinGreenOperator *op);

#endif
