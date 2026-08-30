module;
#include <cstdint>
#include <limits>
#include <random>
#include <utility>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

export module evosim:mwc192;

export namespace evosim {

// MWC192 - Marsaglia multiply-with-carry generator with period ~2^191
// Borrowed from https://prng.di.unimi.it/MWC192.c by Sebastiano Vigna, public domain.
// State: x, y, c (three uint64_t). Must be initialized with 0 < c < MWC_A2 - 1.
// Templated on result_type to support uint32_t and uint64_t variants as requested.
// The 32-bit variant is faster since we only call for 32 bits at a time in many places (parent selection, Bernoulli trials),
// and each 64-bit next() provides two 32-bit values, doubling throughput for 32-bit consumers.
// For NEON, with 32-bit result type we can fit 4 uint32_t per 128-bit NEON register vs 2 uint64_t, maximizing vector width even more.

template <typename ResultType = uint64_t>
struct MWC192 {
  using result_type = ResultType;

  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }

  static constexpr uint64_t MWC_A2 = 0xffa04e67b3c95d86ULL;
  static constexpr uint64_t MWC_A3 = 0x2c4b6e14f2cc8f9bULL;
  static constexpr size_t NUM_STREAMS = 4;
  // Increased buffer size from 16/8 to 256/128 to reduce refill frequency.
  // 256 * 4 bytes = 1KB for 32-bit, fits comfortably in L1 cache.
  // 128 * 8 bytes = 1KB for 64-bit.
  static constexpr size_t BUFFER_SIZE = (sizeof(ResultType) == 4) ? 256 : 128;
  uint64_t sx[NUM_STREAMS];
  uint64_t sy[NUM_STREAMS];
  uint64_t sc[NUM_STREAMS];
  result_type buffer[BUFFER_SIZE];
  size_t buffer_index = BUFFER_SIZE; // start empty, first call to operator() will refill

  MWC192() { seed(); }
  explicit MWC192(uint64_t seed) { this->seed(seed); }
  // Legacy constructor not used with multi-stream version; initialize arrays directly
  MWC192(uint64_t x_, uint64_t y_, uint64_t c_ = 1) {
    seed(x_ ^ y_ ^ c_);
  }

  void seed(uint64_t s = 0x123456789abcdefULL) {
    for (size_t i = 0; i < NUM_STREAMS; i++) {
      uint64_t z2 = s + 0x9e3779b97f4a7c15ULL + i * 0x9e3779b97f4a7c15ULL;
      z2 = (z2 ^ (z2 >> 30)) * 0xbf58476d1ce4e5b9ULL;
      z2 = (z2 ^ (z2 >> 27)) * 0x94d049bb133111ebULL;
      sx[i] = z2 ^ (z2 >> 31);
      z2 = sx[i] + 0x9e3779b97f4a7c15ULL;
      z2 = (z2 ^ (z2 >> 30)) * 0xbf58476d1ce4e5b9ULL;
      z2 = (z2 ^ (z2 >> 27)) * 0x94d049bb133111ebULL;
      sy[i] = z2 ^ (z2 >> 31);
      sc[i] = 1;
    }
    for (int i = 0; i < 10; i++) (void)next();
    for (int i = 0; i < 10; i++) {
      for (size_t j = 0; j < NUM_STREAMS; j++) {
        const __uint128_t t = (__uint128_t)MWC_A2 * (__uint128_t)sx[j] + (__uint128_t)sc[j];
        sx[j] = sy[j]; sy[j] = (uint64_t)t; sc[j] = (uint64_t)(t >> 64);
      }
    }
    buffer_index = BUFFER_SIZE;
  }

  void seed(uint64_t x_, uint64_t y_, uint64_t c_ = 1) {
    seed(x_ ^ y_ ^ c_);
  }

  inline uint64_t next() {
    // Advance stream 0 and return its result (other streams advanced in refill)
    const uint64_t result = sy[0];
    const __uint128_t t = (__uint128_t)MWC_A2 * (__uint128_t)sx[0] + (__uint128_t)sc[0];
    sx[0] = sy[0]; sy[0] = (uint64_t)t; sc[0] = (uint64_t)(t >> 64);
    return result;
  }

  // Fast next64 that bypasses 32-bit buffering, for when we need full 64 bits (e.g., for double conversion needing 53 bits)

  [[gnu::hot, gnu::always_inline]]
  inline void refill() {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Number of 64-bit values actually generated per refill: for 64-bit result type, it's BUFFER_SIZE (8);
    // for 32-bit result type, each 64-bit value gives 2 uint32, so we generate BUFFER_SIZE/2 64-bit values (8) to fill 16-entry buffer.
    // This treats buffer as array of 32-bit values directly and indexes into it, with BUFFER_SIZE = 2x actual 64-bit values, as user requested.
    constexpr size_t NUM_64BIT_VALUES = (sizeof(ResultType) == 4) ? (BUFFER_SIZE / 2) : BUFFER_SIZE;
    constexpr size_t ROUNDS = NUM_64BIT_VALUES / NUM_STREAMS; // e.g., 8/4 = 2 rounds, each round steps all 4 streams once
    size_t buf_pos = 0;
    for (size_t round = 0; round < ROUNDS; round++) {
      for (size_t s = 0; s < NUM_STREAMS; s += 2) {
        uint64_t r0 = sy[s];
        uint64_t r1 = sy[s+1];
        if constexpr (sizeof(ResultType) == 8) {
          buffer[buf_pos++] = static_cast<result_type>(r0);
          buffer[buf_pos++] = static_cast<result_type>(r1);
        } else {
          // 32-bit variant: extract low and high 32 bits from each 64-bit result, giving 4 uint32 per 2 streams
          buffer[buf_pos++] = static_cast<result_type>(r0 & 0xffffffffULL);
          buffer[buf_pos++] = static_cast<result_type>(r0 >> 32);
          buffer[buf_pos++] = static_cast<result_type>(r1 & 0xffffffffULL);
          buffer[buf_pos++] = static_cast<result_type>(r1 >> 32);
        }
        __uint128_t t0 = (__uint128_t)MWC_A2 * (__uint128_t)sx[s] + (__uint128_t)sc[s];
        __uint128_t t1 = (__uint128_t)MWC_A2 * (__uint128_t)sx[s+1] + (__uint128_t)sc[s+1];
        sx[s] = sy[s]; sx[s+1] = sy[s+1];
        sy[s] = (uint64_t)t0; sy[s+1] = (uint64_t)t1;
        sc[s] = (uint64_t)(t0 >> 64); sc[s+1] = (uint64_t)(t1 >> 64);
      }
    }
    buffer_index = 0;
#else
    for (size_t i = 0; i < BUFFER_SIZE; ) {
      size_t stream = (i / (sizeof(ResultType) == 4 ? 2 : 1)) % NUM_STREAMS;
      uint64_t result = sy[stream];
      const __uint128_t t = (__uint128_t)MWC_A2 * (__uint128_t)sx[stream] + (__uint128_t)sc[stream];
      sx[stream] = sy[stream]; sy[stream] = (uint64_t)t; sc[stream] = (uint64_t)(t >> 64);
      if constexpr (sizeof(ResultType) == 8) {
        buffer[i++] = static_cast<result_type>(result);
      } else {
        buffer[i++] = static_cast<result_type>(result & 0xffffffffULL);
        if (i < BUFFER_SIZE) buffer[i++] = static_cast<result_type>(result >> 32);
      }
    }
    buffer_index = 0;
#endif
  }

  [[gnu::hot, gnu::always_inline]]
  inline result_type operator()() {
    if (buffer_index >= BUFFER_SIZE) {
      refill();
    }
    return buffer[buffer_index++];
  }
};

} // namespace evosim
