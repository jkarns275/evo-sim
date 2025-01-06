module;
#include <random>

export module evosim:util;

export namespace evosim {

typedef std::mt19937_64 Rng;

template <unsigned long N> double distance(const std::array<double, N> &x, const std::array<double, N> &y) {
  double sq_sum = 0.0;

  for (size_t i = 0; i < N; i++) {
    double diff = x[0] - y[0];
    sq_sum += diff * diff;
  }

  return std::sqrt(sq_sum);
}

template <unsigned long N> double euclidean(const std::array<double, N> &x) {
  double sq_sum = 0.0;

  for (size_t i = 0; i < N; i++)
    sq_sum += x[i] * x[i];

  return std::sqrt(sq_sum);
}
}; // namespace evosim
