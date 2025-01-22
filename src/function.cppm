module;
#include <boost/math/distributions/normal.hpp>
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

  virtual bool converged(const std::array<double, N> &x) {
    std::array<double, N> go = this->global_optimum();
    std::array<double, N> lo = {-go[0]};
    return distance(go, x) < distance(lo, x);
  }

  virtual std::string to_string() const = 0;
};

template <unsigned N> struct Flat : public FitnessFunction<N> {

  Flat() {}

  double operator()(const std::array<double, N> &x) const override { return 1.0; }

  std::array<double, N> global_optimum() const override { return {0.0}; }

  std::pair<double, double> domain() const override { return {-10.0, 10.0}; }

  bool converged(const std::array<double, N> &x) override {
    std::array<double, N> go = this->global_optimum();
    std::array<double, N> lo = {-go[0]};
    return distance(go, x) < distance(lo, x);
  }

  std::string to_string() const override { return "Const(1)"; }
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

  std::array<double, N> global_optimum() const override { return array_of<N>(sigma * 2.0 * (A >= B ? 1 : -1)); }

  std::pair<double, double> domain() const override { return {-10.0, 10.0}; }

  bool converged(const std::array<double, N> &x) override {
    std::array<double, N> go = this->global_optimum();
    std::array<double, N> lo = array_of<N>(-go[0]);
    return distance(go, x) < distance(lo, x);
  }

  std::string to_string() const override { return std::format("Scott-DeJong( {:.2f}; {:.2f} )", A, B); }
};

template <unsigned N> struct Schwefel : public FitnessFunction<N> {
  double operator()(const std::array<double, N> &x) const override {
    double total = 0.0;

    for (auto xi : x)
      total += 50.0 * xi * std::sin(std::sqrt(std::abs(50.0 * xi)));

    return 418.9829 * (double)N - total;
  }

  std::array<double, N> global_optimum() const override { return array_of<N>(-420.9687 / 50); }

  std::pair<double, double> domain() const override { return {-10.0, 10.0}; }

  std::string to_string() const override { return "Schwefel()"; }
};

template <unsigned N> struct Rosenbrock : public FitnessFunction<N> {
  double operator()(const std::array<double, N> &x) const override {
    double sum = 0.0;
    for (int i = 0; i < N - 1; i++) {
      double left = (x[i + 1] - x[i] * x[i]);
      left *= left * 100.0;

      double right = 1.0 - x[i];
      right *= right;

      sum += left + right;
    }

    return sum;
  }

  std::array<double, N> global_optimum() const override { return array_of<N>(1.0); }

  std::pair<double, double> domain() const override { return {-2.0, 2.0}; }

  std::string to_string() const override { return std::format("Rosenbrock()"); }
};

template <unsigned N> struct SphericalGaussian : public FitnessFunction<N> {
  std::array<boost::math::normal_distribution<double>, N> dist;

  SphericalGaussian(std::array<double, N> mu, double sigma = 1.0) {
    for (size_t i = 0; i < N; i++)
      dist[i] = boost::math::normal_distribution<double>(mu[i], sigma);
  }

  double operator()(const std::array<double, N> &x) const override {
    double sum = 1.0;
    for (int i = 0; i < N; i++)
      sum += boost::math::pdf(dist[i], x[i]);

    return sum;
  }

  std::array<double, N> global_optimum() const override {
    std::array<double, N> optimum;

    for (size_t i = 0; i < N; i++)
      optimum[i] = dist[i].mean();

    return optimum;
  }

  std::pair<double, double> domain() const override { return {-INFINITY, INFINITY}; }

  std::string to_string() const override {
    return std::format("Spherical-Gaussian( mu = ...; {:.2f} )", dist[0].scale());
  }
};

template <unsigned N> struct InvSphericalGaussian : public FitnessFunction<N> {
  boost::math::normal_distribution<double> dist;
  const double max_value;

  static double calc_max_value(double mu, double sigma) {
    boost::math::normal_distribution<double> dist(mu, sigma);
    double prod = 1.0;
    for (int i = 0; i < N; i++)
      prod *= boost::math::pdf(dist, mu);

    return prod;
  }

  InvSphericalGaussian(double mu = 0.0, double sigma = 1.0) : dist(mu, sigma), max_value(calc_max_value(mu, sigma)) {}

  double operator()(const std::array<double, N> &x) const override {
    double product = 1.0;
    for (int i = 0; i < N; i++)
      product += boost::math::pdf(dist, x[i]);

    return max_value - product;
  }

  std::array<double, N> global_optimum() const override { return {dist.mean()}; }

  std::pair<double, double> domain() const override { return {-INFINITY, INFINITY}; }

  std::string to_string() const override {
    return std::format("Inv-Spherical-Gaussian( {:.2f}; {:.2f} )", dist.mean(), dist.scale());
  }
};
}; // namespace evosim
