#ifndef __SCAPIN_H__
#define __SCAPIN_H__

#if _WIN32
#define DllExport __declspec(dllexport)
#else
#define DllExport
#endif

template <size_t DIM>
class Hooke {
 public:
  const double mu;
  const double nu;
  const size_t isize;
  const size_t osize;
  Hooke(const double mu, const double nu)
      : mu(mu), nu(nu), isize((DIM * (DIM + 1)) / 2), osize(isize) {}

  void apply(const double* k, const double* tau, double* out);
};

template <size_t DIM>
void Hooke<DIM>::apply(const double *k, const double *tau, double *out) {
  if constexpr (DIM == 2) {
    double const k2 = k[0] * k[0] + k[1] * k[1];
    double tau_k[] = {tau[0] * k[0] + M_SQRT1_2 * tau[2] * k[1],
                      tau[1] * k[1] + M_SQRT1_2 * tau[2] * k[0]};
    double const n_tau_n = (k[0] * tau_k[0] + k[1] * tau_k[1]) / k2;
    double const const1 = n_tau_n / (1. - this->nu);
    double const const2 = 1. / (2. * this->mu * k2);
    out[0] = const2 * (k[0] * (2. * tau_k[0] - const1 * k[0]));
    out[1] = const2 * (k[1] * (2. * tau_k[1] - const1 * k[1]));
    double const const3 = M_SQRT2 * const2;
    out[2] =
        const3 * (k[0] * tau_k[1] + k[1] * tau_k[0] - const1 * k[0] * k[1]);
  }
  if constexpr (DIM == 3) {
    double const k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
    double tau_k[] = {
        tau[0] * k[0] + M_SQRT1_2 * (tau[5] * k[1] + tau[4] * k[2]),
        tau[1] * k[1] + M_SQRT1_2 * (tau[5] * k[0] + tau[3] * k[2]),
        tau[2] * k[2] + M_SQRT1_2 * (tau[4] * k[0] + tau[3] * k[1])};
    double const n_tau_n =
        (k[0] * tau_k[0] + k[1] * tau_k[1] + k[2] * tau_k[2]) / k2;
    double const const1 = n_tau_n / (1. - this->nu);
    double const const2 = 1. / (2. * this->mu * k2);
    out[0] = const2 * (k[0] * (2. * tau_k[0] - const1 * k[0]));
    out[1] = const2 * (k[1] * (2. * tau_k[1] - const1 * k[1]));
    out[2] = const2 * (k[2] * (2. * tau_k[2] - const1 * k[2]));
    double const const3 = M_SQRT2 * const2;
    out[3] =
        const3 * (k[1] * tau_k[2] + k[2] * tau_k[1] - const1 * k[1] * k[2]);
    out[4] =
        const3 * (k[2] * tau_k[0] + k[0] * tau_k[2] - const1 * k[2] * k[0]);
    out[5] =
        const3 * (k[0] * tau_k[1] + k[1] * tau_k[0] - const1 * k[0] * k[1]);
  }
}

#endif
