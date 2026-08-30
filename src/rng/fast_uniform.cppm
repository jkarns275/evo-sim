module;
#include <cstdint>
#include <limits>
#include <random>

export module evosim:fast_uniform;

export namespace evosim {

// Fast uniform integer generation in range [0, n) without std::uniform_int_distribution overhead.
// Generator-agnostic struct templated on RNG type, using fast multiplication-based range reduction.
// Much faster than std::uniform_int_distribution which does rejection sampling and generic case handling.
// Bias minimal for small n relative to 2^32/2^64, acceptable for EA simulation parent selection, tournament, etc.
// We only need next_uniform_uint32 as user requested, since we only call for 32 bits at a time in hot paths.
// Each call returns uint32_t in [0, n), using fast method: (rng() * n) >> 32 for 32-bit RNG, >> 64 for 64-bit.

struct FastUniformInt {
  // Generate uniform uint32_t in [0, n) using fast multiplication method, generator-agnostic.
  // Works with any RNG engine having operator()() returning uint32_t or uint64_t.
  // For 32-bit RNG result_type, uses (uint64_t)rng() * n >> 32.
  // For 64-bit RNG result_type, uses (__uint128_t)rng() * n >> 64.
  // Much faster than std::uniform_int_distribution which constructs distribution object and does rejection sampling.
  template <typename Rng>
  [[gnu::hot, gnu::always_inline]]
  inline uint32_t operator()(Rng& rng, uint32_t n) const {
    using result_type = typename Rng::result_type;
    if constexpr (sizeof(result_type) == 4) {
      // 32-bit RNG: 32x32->64 multiply, shift by 32
      return static_cast<uint32_t>((uint64_t)rng() * n >> 32);
    } else {
      // 64-bit RNG: 64x64->128 multiply, shift by 64
      return static_cast<uint32_t>((__uint128_t)rng() * (__uint128_t)n >> 64);
    }
  }

  // Convenience method named as user requested: next_uniform_uint32
  template <typename Rng>
  inline uint32_t next_uniform_uint32(Rng& rng, uint32_t n) const {
    return operator()(rng, n);
  }
};

// New interface aligned with std::uniform_int_distribution for easy swapping via typedef.
// Usage: uniform_int_dist_t(n)(generator) returns value in [0, n).
// This allows configuring the uniform int distribution type via a single typedef
// in traits or a config header, enabling benchmarks of FastUniformInt vs std::uniform_int_distribution.
struct FastUniformIntDist {
  uint32_t n;
  explicit FastUniformIntDist(uint32_t n) : n(n) {}
  template <typename RNG>
  [[gnu::hot, gnu::always_inline]]
  inline uint32_t operator()(RNG& rng) const {
    return FastUniformInt{}(rng, n);
  }
};

template <typename T = uint32_t>
struct StdUniformIntDist {
  T n;
  explicit StdUniformIntDist(T n) : n(n) {}
  template <typename RNG>
  [[gnu::hot, gnu::always_inline]]
  inline T operator()(RNG& rng) const {
    std::uniform_int_distribution<T> dist(0, n - 1);
    return dist(rng);
  }
};

// Default uniform int distribution type used throughout the codebase.
// Change this typedef to StdUniformIntDist<uint32_t> to use std::uniform_int_distribution
// instead of FastUniformInt, for benchmarking or comparison.
// All call sites use: uniform_int_dist_t(n)(generator) to get value in [0, n).
using uniform_int_dist_t = FastUniformIntDist;

// Helper to get uniform_int_dist_t from Traits if defined, otherwise use global.
// Usage: typename get_uniform_dist<Traits>::type
template <typename T, typename = void>
struct get_uniform_dist {
    using type = uniform_int_dist_t;
};
template <typename T>
struct get_uniform_dist<T, std::void_t<typename T::uniform_int_dist_t>> {
    using type = typename T::uniform_int_dist_t;
};

} // namespace evosim
