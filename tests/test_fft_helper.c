#include <glib.h>
#include <limits.h>
#include <scapin/fft_helper.h>
#include <stdbool.h>

typedef struct test_fftfreq_data_ {
  size_t n;
  double d;
  bool inplace;
  int *cycles;
} test_fftfreq_data;

test_fftfreq_data *test_fftfreq_data_new(size_t n, double d, bool inplace,
                                         int *cycles) {
  test_fftfreq_data *data = malloc(sizeof(test_fftfreq_data));
  data->n = n;
  data->d = d;
  data->inplace = inplace;
  data->cycles = malloc(n * sizeof(int));
  for (size_t i = 0; i < n; i++) {
    data->cycles[i] = cycles[i];
  }
  return data;
}

void test_fftfreq_data_free(test_fftfreq_data *data) {
  free(data->cycles);
  free(data);
}

void test_fftfreq(gconstpointer data) {
  test_fftfreq_data const *data_ = data;

  double *freq;
  if (data_->inplace) {
    freq = malloc(data_->n * sizeof(double));
  } else {
    freq = NULL;
  }

  double *act = fft_helper_fftfreq(data_->n, data_->d, freq);
  if (data_->inplace) {
    g_assert_cmpuint(act, ==, freq);
  }

  for (size_t i = 0; i < data_->n; i++) {
    double const exp_i = data_->cycles[i] / (data_->d * data_->n);
    g_assert_cmpfloat(act[i], ==, exp_i);
  }

  if (!data_->inplace) {
    free(freq);
  }
  free(act);
}

void setup_test_fftfreq() {
  int even[] = {0, 1, 2, 3, -4, -3, -2, -1};
  int odd[] = {0, 1, 2, 3, 4, -4, -3, -2, -1};
  g_test_add_data_func_full("/fft_helper/fftfreq/even/in-place",
                            test_fftfreq_data_new(8, 1., true, even),
                            test_fftfreq, test_fftfreq_data_free);
  g_test_add_data_func_full("/fft_helper/fftfreq/even/out-of-place",
                            test_fftfreq_data_new(8, 1., false, even),
                            test_fftfreq, test_fftfreq_data_free);
  g_test_add_data_func_full("/fft_helper/fftfreq/odd/in-place",
                            test_fftfreq_data_new(9, 1., true, odd),
                            test_fftfreq, test_fftfreq_data_free);
  g_test_add_data_func_full("/fft_helper/fftfreq/odd/out-of-place",
                            test_fftfreq_data_new(9, 1., false, odd),
                            test_fftfreq, test_fftfreq_data_free);
}

int main(int argc, char **argv) {
  g_test_init(&argc, &argv, NULL);
  setup_test_fftfreq();
  return g_test_run();
}
