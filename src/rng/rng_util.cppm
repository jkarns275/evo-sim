module;
#include <array>
#include <cmath>
#include <random>

export module evosim:rng_util;
import :mwc192;
import :fast_uniform;
import :ziggurat;

export namespace evosim {

// Use 32-bit variant by default as user requested - faster since we only call for 32 bits at a time in many places (parent selection, Bernoulli trials).
// Each 64-bit MWC192 next() provides two 32-bit values, doubling throughput for 32-bit consumers.
// For double conversion needing 53 bits, next_uniform_double() combines two 32-bit values or calls next64() directly.
using Rng = MWC192<uint32_t>;
inline const std::normal_distribution<double> NORMAL_DISTRIBUTION{0.0, 1.0};

template <typename RngType = Rng>
inline RngType make_rng() {
  std::random_device rd;
  RngType rng{rd()};
  if constexpr (requires { rng.next(); }) {
    for (int i = 0; i < 100; i++) (void)rng.next();
  } else {
    for (int i = 0; i < 100; i++) (void)rng();
  }
  return rng;
}

template <typename RngType = Rng>
inline RngType make_rng(uint64_t seed) {
  RngType rng{seed};
  if constexpr (requires { rng.next(); }) {
    for (int i = 0; i < 100; i++) (void)rng.next();
  } else {
    for (int i = 0; i < 100; i++) (void)rng();
  }
  return rng;
}

// Explicit specializations for std::mt19937 and std::mt19937_64 to avoid next() call
template <>
inline std::mt19937 make_rng<std::mt19937>() {
  std::random_device rd;
  std::mt19937 rng{rd()};
  for (int i = 0; i < 100; i++) (void)rng();
  return rng;
}

template <>
inline std::mt19937 make_rng<std::mt19937>(uint64_t seed) {
  std::mt19937 rng{static_cast<uint32_t>(seed)};
  for (int i = 0; i < 100; i++) (void)rng();
  return rng;
}

template <>
inline std::mt19937_64 make_rng<std::mt19937_64>() {
  std::random_device rd;
  std::mt19937_64 rng{rd()};
  for (int i = 0; i < 100; i++) (void)rng();
  return rng;
}

template <>
inline std::mt19937_64 make_rng<std::mt19937_64>(uint64_t seed) {
  std::mt19937_64 rng{seed};
  for (int i = 0; i < 100; i++) (void)rng();
  return rng;
}

template <unsigned long N, typename RNG = Rng>
inline std::array<double, N> point_in_sphere(RNG& rng, double radius) {
  std::normal_distribution<double> normal(0.0, 1.0);
  std::array<double, N> dir;
  double ss = 0;

  for (size_t i = 0; i < N; i++) {
    double v = normal(rng);
    ss += v * v;
    dir[i] = v;
  }

  double mag = std::sqrt(ss);
  double r = std::generate_canonical<double, 52>(rng) * radius;

  for (size_t i = 0; i < N; i++)
    dir[i] = r * dir[i] / mag;

  return dir;
}

} // namespace evosim
