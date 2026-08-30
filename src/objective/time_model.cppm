module;
#include <array>
#include <string>
#include <string_view>
#include <utility>

export module evosim:time_model;
import :core;
import :objective;
import :fitness;

export namespace evosim {

// TimeModel conceptually same signature as Objective but distinct type for
// clarity. We alias template to ObjectiveBase for now to share implementation,
// but with separate name.
template <class Derived, unsigned N>
using TimeModelBase = ObjectiveBase<Derived, N>;

// Specific time model aliases for clarity in code – they are just objectives
// reused as time functions.
template <unsigned N>
using FlatTime = FlatObj<N>;
template <unsigned N>
using ScottDeJongBasinsTime = ScottDeJongBasinsObj<N>;

// Backpropagation time wrapper: models local search evaluation time as 2 * base
// time model. Takes BaseTime as template template parameter to wrap any
// existing time model.
template <template <unsigned> class BaseTime, unsigned N>
struct BackpropagationTime
    : public ObjectiveBase<BackpropagationTime<BaseTime, N>, N> {
  static constexpr std::string_view name = "BackpropagationTime";
  BaseTime<N> base;

  BackpropagationTime() = default;

  explicit BackpropagationTime(const BaseTime<N>& b) : base(b) {}

  explicit BackpropagationTime(BaseTime<N>&& b) : base(std::move(b)) {}

  double operator_impl(const std::array<double, N>& x) const {
    return 2.0 * base(x);
  }

  std::array<double, N> global_optimum_impl() const {
    return base.global_optimum();
  }

  std::pair<double, double> domain_impl() const {
    return base.domain();
  }

  bool converged_impl(const std::array<double, N>& x) const {
    return base.converged(x);
  }

  std::array<double, N> gradient_impl(const std::array<double, N>& x) const {
    auto g = base.gradient(x);
    for (auto& v : g)
      v *= 2.0;
    return g;
  }

  std::string to_string() const {
    return std::string{"2*"} + base.to_string();
  }
};

// Backpropagation time wrapper for old FitnessFunction types (Flat, etc.)
// Models local search evaluation time as 2 * base time.
template <template <unsigned> class Base, unsigned M>
struct BackpropagationTimeStatic : public Base<M> {
  static constexpr std::string_view name = "BackpropagationTimeStatic";
  BackpropagationTimeStatic() = default;

  double operator()(const std::array<double, M>& x) const override {
    return 2.0 * Base<M>::operator()(x);
  }

  std::string to_string() const override {
    return std::string{"2*"} + Base<M>::to_string();
  }
};

} // namespace evosim
