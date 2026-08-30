module;
#include <array>
#include <cmath>
#include <variant>

export module evosim:core;
import :rng_util;

export namespace evosim {

// Re-export RNG utilities from :rng_util for backward compatibility
// so existing code that imports :core still sees Rng, make_rng, point_in_sphere
using evosim::Rng;
using evosim::NORMAL_DISTRIBUTION;
using evosim::make_rng;
using evosim::point_in_sphere;

template <unsigned long N>
inline double distance(
    const std::array<double, N>& x,
    const std::array<double, N>& y) {
  double s = 0;

  for (size_t i = 0; i < N; i++) {
    double d = x[i] - y[i];
    s += d * d;
  }

  return std::sqrt(s);
}

template <unsigned long N>
inline double distance_squared(
    const std::array<double, N>& x,
    const std::array<double, N>& y) {
  double s = 0;

  for (size_t i = 0; i < N; i++) {
    double d = x[i] - y[i];
    s += d * d;
  }

  return s;
}

template <unsigned long N>
inline constexpr std::array<double, N> array_of(double v) {
  std::array<double, N> d;
  d.fill(v);
  return d;
}

template <unsigned long N>
inline double euclidean(const std::array<double, N>& x) {
  double s = 0;

  for (auto v : x)
    s += v * v;

  return std::sqrt(s);
}

template <typename VariantType, typename T, std::size_t index = 0>
constexpr std::size_t variant_index() {
  static_assert(std::variant_size_v<VariantType> > index, "Type not found");
  if constexpr (index == std::variant_size_v<VariantType>) {
    return index;
  } else if constexpr (
      std::is_same_v<std::variant_alternative_t<index, VariantType>, T>) {
    return index;
  } else {
    return variant_index<VariantType, T, index + 1>();
  }
}

} // namespace evosim
