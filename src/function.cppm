module;
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <format>
#include <utility>

export module evosim:function;
import :util;

export namespace evosim {

template <unsigned N> struct FitnessFunction {
  virtual ~FitnessFunction() {}

  virtual double operator()(const std::array<double, N> &x) const = 0;
  virtual std::array<double, N> global_optimum() const = 0;
  virtual std::pair<double, double> domain() const = 0;
  virtual bool converged(const std::array<double, N> &x) = 0;
  virtual std::string to_string() const = 0;
};

template <unsigned N> struct ScottDeJongBasins : public FitnessFunction<N> {
  double A, B, sigma;

  ScottDeJongBasins(double A, double B, double sigma = 2.5) : A(A), B(B), sigma(sigma) {}

  double operator()(const std::array<double, N> &x) const override {
    double a_sq_sum = 0.0;
    double b_sq_sum = 0.0;
    for (size_t i = 0; i < N; i++) {
      double a_diff = x[i] - 2 * sigma;
      a_sq_sum += a_diff * a_diff;

      double b_diff = x[i] + 2 * sigma;
      b_sq_sum += b_diff * b_diff;
    }

    return std::max(std::abs(A), std::abs(B)) - A * std::exp(-1.0 / (2.0 * sigma * sigma) * a_sq_sum) -
           B * std::exp(-1.0 / (2.0 * sigma * sigma) * b_sq_sum);
  }

  std::array<double, N> global_optimum() const override {
    if (A >= B)
      return {sigma * 2.0};
    else
      return {sigma * -2.0};
  }

  std::pair<double, double> domain() const override { return {-10.0, 10.0}; }

  bool converged(const std::array<double, N> &x) {
    std::array<double, N> go = this->global_optimum();
    std::array<double, N> lo = {-go[0]};
    return distance(go, x) < distance(lo, x);
  }

  std::string to_string() const override { return std::format("Scott-DeJong( {:.2f}, {:.2f} )", A, B); }
};

template <unsigned N> struct Schwefel : public FitnessFunction<N> {
  double operator()(std::array<double, N> &x) const override {
    double total = 0.0;

    for (auto xi : x)
      total += 50.0 * xi * std::sin(std::sqrt(std::abs(50.0 * xi)));

    return 418.9829 * (double)N - total;
  }

  std::array<double, N> global_optimum() const override { return {-420.9687 / 50}; }

  std::pair<double, double> domain() const override { return {-10.0, 10.0}; }
};

}; // namespace evosim
