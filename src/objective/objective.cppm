module;
#include <boost/math/distributions/normal.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <numbers>
#include <string>
#include <string_view>
#include <utility>

export module evosim:objective;
import :core;
import :concepts;

export namespace evosim {

template <class Derived, unsigned N>
struct ObjectiveBase {
  static constexpr std::string_view name = "ObjectiveBase";

  double operator()(const std::array<double, N>& x) const {
    return static_cast<const Derived*>(this)->operator_impl(x);
  }

  std::array<double, N> global_optimum() const {
    return static_cast<const Derived*>(this)->global_optimum_impl();
  }

  std::pair<double, double> domain() const {
    return static_cast<const Derived*>(this)->domain_impl();
  }

  bool converged(const std::array<double, N>& x) const {
    return static_cast<const Derived*>(this)->converged_impl(x);
  }

  std::string to_string() const {
    return std::string{Derived::name};
  }

  // default implementations to be overridden
  double operator_impl(const std::array<double, N>&) const {
    return 0.0;
  }

  std::array<double, N> global_optimum_impl() const {
    return {};
  }

  std::pair<double, double> domain_impl() const {
    return {-10.0, 10.0};
  }

  bool converged_impl(const std::array<double, N>& x) const {
    auto go = global_optimum();
    std::array<double, N> lo{};
    for (size_t i = 0; i < N; i++)
      lo[i] = -go[i];
    return distance_squared(go, x) < distance_squared(lo, x);
  }
};

template <unsigned N>
struct FlatObj : public ObjectiveBase<FlatObj<N>, N> {
  static constexpr std::string_view name = "Const(1)";

  double operator_impl(const std::array<double, N>&) const {
    return 1.0;
  }

  std::array<double, N> global_optimum_impl() const {
    return {};
  }

  std::pair<double, double> domain_impl() const {
    return {-10.0, 10.0};
  }

  bool converged_impl(const std::array<double, N>& x) const {
    std::array<double, N> go{};
    std::array<double, N> lo{};
    return distance_squared(go, x) < distance_squared(lo, x);
  }
};

template <unsigned N>
struct ScottDeJongBasinsObj : public ObjectiveBase<ScottDeJongBasinsObj<N>, N> {
  static constexpr std::string_view name = "Scott-DeJong";
  double A, B, sigma;

  ScottDeJongBasinsObj(double A_, double B_, double sigma_ = 2.5)
      : A(A_), B(B_), sigma(sigma_) {}

  double operator_impl(const std::array<double, N>& x) const {
    double a_sq = 0.0, b_sq = 0.0;
    for (size_t i = 0; i < N; i++) {
      double ad = x[i] - 2 * sigma;
      a_sq += ad * ad;
      double bd = x[i] + 2 * sigma;
      b_sq += bd * bd;
    }
    return std::max(std::abs(A), std::abs(B)) -
        A * std::exp(-1.0 / (2 * sigma * sigma) * a_sq) -
        B * std::exp(-1.0 / (2 * sigma * sigma) * b_sq);
  }

  std::array<double, N> global_optimum_impl() const {
    return array_of<N>(sigma * 2.0 * (A >= B ? 1 : -1));
  }

  std::pair<double, double> domain_impl() const {
    return {-10.0, 10.0};
  }

  bool converged_impl(const std::array<double, N>& x) const {
    auto go = array_of<N>(2 * sigma);
    auto lo = array_of<N>(-2 * sigma);
    return distance_squared(go, x) < distance_squared(lo, x);
  }

  std::string to_string() const {
    return std::format("Scott-DeJong( {:.2f}; {:.2f} )", A, B);
  }
};

template <unsigned N>
struct SchwefelObj : public ObjectiveBase<SchwefelObj<N>, N> {
  static constexpr std::string_view name = "Schwefel";

  double operator_impl(const std::array<double, N>& x) const {
    double total = 0.0;
    for (auto xi : x)
      total += 50.0 * xi * std::sin(std::sqrt(std::abs(50.0 * xi))) * (1.0 / N);
    return (418.9829 - total) / N;
  }

  std::array<double, N> global_optimum_impl() const {
    return array_of<N>(420.9687 / 50);
  }

  std::pair<double, double> domain_impl() const {
    return {-10.0, 10.0};
  }

  bool converged_impl(const std::array<double, N>& x) const {
    return distance_squared(global_optimum_impl(), x) < 40.0;
  }
};

template <unsigned N>
struct RosenbrockObj : public ObjectiveBase<RosenbrockObj<N>, N> {
  static constexpr std::string_view name = "Rosenbrock";

  double operator_impl(const std::array<double, N>& x) const {
    double sum = 0.0;
    std::array<double, N> sx;
    for (size_t i = 0; i < N; i++)
      sx[i] = x[i] / 10.0 * 2.048;
    for (int i = 0; i < N - 1; i++) {
      double l = sx[i + 1] - sx[i] * sx[i];
      l *= l * 100.0;
      double r = 1.0 - sx[i];
      r *= r;
      sum += l + r;
    }
    return sum;
  }

  std::array<double, N> global_optimum_impl() const {
    return array_of<N>(1.0);
  }

  std::pair<double, double> domain_impl() const {
    return {-10.0, 10.0};
  }

  bool converged_impl(const std::array<double, N>& x) const {
    return distance(x, global_optimum_impl()) < 0.5;
  }
};

template <unsigned N>
struct AckleyObj : public ObjectiveBase<AckleyObj<N>, N> {
  static constexpr std::string_view name = "Ackley";
  std::array<double, N> center;
  bool inverse;

  AckleyObj(std::array<double, N> c, bool inv = false)
      : center(c), inverse(inv) {}

  double operator_impl(const std::array<double, N>& x) const {
    const double a = 20, b = 0.2, c = 2 * std::numbers::pi;
    double lsum = 0, rsum = 0;
    for (size_t i = 0; i < N; i++) {
      double s = ((x[i] - center[i]) / 10) * 32.768;
      lsum += s * s;
      rsum += std::cos(c * s);
    }
    lsum /= (double)N;
    rsum /= (double)N;
    lsum = -b * std::sqrt(lsum);
    double v = -a * std::exp(lsum) - std::exp(rsum) + a + std::exp(1);
    return inverse ? std::max(1.0, 24.0 - v) : std::max(0.01, v);
  }

  std::array<double, N> global_optimum_impl() const {
    return center;
  }

  std::pair<double, double> domain_impl() const {
    return {-10.0, 10.0};
  }

  bool converged_impl(const std::array<double, N>& x) const {
    return distance(x, center) < 0.05;
  }
};

template <unsigned N>
struct SphericalObj : public ObjectiveBase<SphericalObj<N>, N> {
  static constexpr std::string_view name = "Spherical";
  std::array<double, N> center;
  bool inverse;

  SphericalObj(std::array<double, N> c, bool inv = false)
      : center(c), inverse(inv) {}

  double operator_impl(const std::array<double, N>& x) const {
    double s = 0;
    for (size_t i = 0; i < N; i++) {
      double d = x[i] - center[i];
      s += d * d;
    }
    return inverse ? 1.0 / s : s;
  }

  std::array<double, N> global_optimum_impl() const {
    return center;
  }

  std::pair<double, double> domain_impl() const {
    return {-10.0, 10.0};
  }

  bool converged_impl(const std::array<double, N>& x) const {
    return distance(x, center) < 0.05;
  }
};

template <unsigned N>
struct SphericalGaussianObj : public ObjectiveBase<SphericalGaussianObj<N>, N> {
  static constexpr std::string_view name = "Spherical-Gaussian";
  std::array<boost::math::normal_distribution<double>, N> dist;

  SphericalGaussianObj(std::array<double, N> mu, double sigma = 1.0) {
    for (size_t i = 0; i < N; i++)
      dist[i] = boost::math::normal_distribution<double>(mu[i], sigma);
  }

  double operator_impl(const std::array<double, N>& x) const {
    double s = 1.0;
    for (int i = 0; i < N; i++)
      s += boost::math::pdf(dist[i], x[i]);
    return s;
  }

  std::array<double, N> global_optimum_impl() const {
    std::array<double, N> o;
    for (size_t i = 0; i < N; i++)
      o[i] = dist[i].mean();
    return o;
  }

  std::pair<double, double> domain_impl() const {
    return {-INFINITY, INFINITY};
  }
};

template <unsigned N>
struct InvSphericalGaussianObj
    : public ObjectiveBase<InvSphericalGaussianObj<N>, N> {
  static constexpr std::string_view name = "Inv-Spherical-Gaussian";
  boost::math::normal_distribution<double> dist;
  double max_value;

  static double calc_max(double mu, double sigma) {
    boost::math::normal_distribution<double> d(mu, sigma);
    double p = 1.0;
    for (int i = 0; i < N; i++)
      p *= boost::math::pdf(d, mu);
    return p;
  }

  InvSphericalGaussianObj(double mu = 0.0, double sigma = 1.0)
      : dist(mu, sigma), max_value(calc_max(mu, sigma)) {}

  double operator_impl(const std::array<double, N>& x) const {
    double p = 1.0;
    for (int i = 0; i < N; i++)
      p += boost::math::pdf(dist, x[i]);
    return max_value - p;
  }

  std::array<double, N> global_optimum_impl() const {
    return {dist.mean()};
  }

  std::pair<double, double> domain_impl() const {
    return {-INFINITY, INFINITY};
  }
};

// Compatibility alias removed - old code uses evosim:function partition
// FitnessFunction virtual base. New code should use Objective concept directly.

} // namespace evosim
