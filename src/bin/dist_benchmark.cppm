module;
#include <spdlog/spdlog.h>
#include <chrono>
#include <random>
#include <vector>

export module evosim.main;
import evosim;

using namespace evosim;
const unsigned N = 10;

// Simple benchmark of distributions in isolation, not entire EA algorithm,
// to get clear signal on distribution performance as requested.

template <typename Rng, typename Dist>
double benchmark_distribution(
    const std::string& label,
    Rng& rng,
    Dist& dist,
    size_t iterations = 100'000'000) {
  volatile double sink = 0.0; // prevent optimization from removing the loop
  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < iterations; i++) {
    sink += dist(rng);
  }

  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();
  double ns_per_op = ms * 1e6 / iterations;
  spdlog::info("{}: {:.2f} ms for {} ops ({:.2f} ns/op)", label, ms, iterations, ns_per_op);
  return ms;
}

template <typename Rng>
double benchmark_fast_uniform_double(
    const std::string& label,
    Rng& rng,
    size_t iterations = 100'000'000) {
  volatile double sink = 0.0;
  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < iterations; i++) {
    // Fast path: (rng() >> 11) * 2^-53, avoids std::uniform_real_distribution overhead
    if constexpr (std::is_same_v<typename Rng::result_type, uint32_t>) {
      uint64_t r = ((uint64_t)rng() << 32) | rng();
      sink += (r >> 11) * 0x1.0p-53;
    } else {
      sink += (rng() >> 11) * 0x1.0p-53;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();
  double ns_per_op = ms * 1e6 / iterations;
  spdlog::info("{}: {:.2f} ms for {} ops ({:.2f} ns/op)", label, ms, iterations, ns_per_op);
  return ms;
}

template <typename Rng>
double benchmark_fast_uniform_uint64(
    const std::string& label,
    Rng& rng,
    uint64_t range,
    size_t iterations = 100'000'000) {
  volatile uint64_t sink = 0;
  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < iterations; i++) {
    // Fast range reduction: (rng() * range) >> 64 for 64-bit, >> 32 for 32-bit
    if constexpr (std::is_same_v<typename Rng::result_type, uint32_t>) {
      sink += (uint64_t)rng() * range >> 32;
    } else {
      sink += (__uint128_t)rng() * (__uint128_t)range >> 64;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();
  double ns_per_op = ms * 1e6 / iterations;
  spdlog::info("{}: {:.2f} ms for {} ops ({:.2f} ns/op)", label, ms, iterations, ns_per_op);
  return ms;
}

int main() {
  initialize_logger();
  spdlog::set_level(spdlog::level::info);

  spdlog::info("=== Distribution Benchmark (isolated, not entire EA) ===");
  spdlog::info("Measuring time per operation for various RNG engines and distributions");
  spdlog::info("Iterations per test: 100 million (100M) for stable measurement");
  spdlog::info("");

  const size_t ITERS = 100'000'000;

  // Test 1: Old RNG engine (mt19937_64) with std::normal_distribution
  {
    spdlog::info("--- Test 1: std::mt19937_64 engine ---");
    std::mt19937_64 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    benchmark_distribution("mt19937_64 + std::normal_distribution<float>", rng, dist, ITERS);
    spdlog::info("");
  }

  // Test 2: New RNG engine 64-bit with std::normal_distribution
  {
    spdlog::info("--- Test 2: MWC192<uint64_t> engine ---");
    MWC192<uint64_t> rng(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    benchmark_distribution("MWC192<uint64_t> + std::normal_distribution<float>", rng, dist, ITERS);
    spdlog::info("");
  }

  // Test 3: New RNG engine 32-bit with std::normal_distribution
  {
    spdlog::info("--- Test 3: MWC192<uint32_t> engine ---");
    MWC192<uint32_t> rng(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    benchmark_distribution("MWC192<uint32_t> + std::normal_distribution<float>", rng, dist, ITERS);
    spdlog::info("");
  }

  // Test 4: New RNG 32-bit with ziggurat (fast normal)
  {
    spdlog::info("--- Test 4: MWC192<uint32_t> + Ziggurat ---");
    MWC192<uint32_t> rng(42);
    ziggurat_normal_distribution<float> dist(0.0f, 0.5f);
    benchmark_distribution("MWC192<uint32_t> + ziggurat_normal_distribution<float>", rng, dist, ITERS);
    spdlog::info("");
  }

  // Test 5: Fast uniform double path vs std::uniform_real_distribution
  {
    spdlog::info("--- Test 5: Uniform double generation ---");
    {
      MWC192<uint32_t> rng(42);
      std::uniform_real_distribution<double> dist(0.0, 1.0);
      benchmark_distribution("MWC192<uint32_t> + std::uniform_real_distribution<double>", rng, dist, ITERS);
    }
    {
      MWC192<uint32_t> rng(42);
      benchmark_fast_uniform_double("MWC192<uint32_t> + fast (rng>>11)*2^-53", rng, ITERS);
    }
    spdlog::info("");
  }

  // Test 6: Fast uniform int range reduction vs std::uniform_int_distribution
  {
    spdlog::info("--- Test 6: Uniform int in range [0, 20) ---");
    {
      MWC192<uint32_t> rng(42);
      std::uniform_int_distribution<size_t> dist(0, 19);
      benchmark_distribution("MWC192<uint32_t> + std::uniform_int_distribution<size_t>(0,19)", rng, dist, ITERS);
    }
    {
      MWC192<uint32_t> rng(42);
      benchmark_fast_uniform_uint64("MWC192<uint32_t> + fast (rng*20>>32)", rng, 20, ITERS);
    }
    spdlog::info("");
  }

  // Test 7: Raw RNG throughput (just rng(), no distribution)
  {
    spdlog::info("--- Test 7: Raw RNG throughput (no distribution) ---");
    {
      std::mt19937_64 rng(42);
      volatile uint64_t sink = 0;
      auto start = std::chrono::high_resolution_clock::now();
      for (size_t i = 0; i < ITERS; i++) {
        sink += rng();
      }
      auto end = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration<double, std::milli>(end - start).count();
      spdlog::info("mt19937_64 raw: {:.2f} ms for {} ops ({:.2f} ns/op)", ms, ITERS, ms * 1e6 / ITERS);
    }
    {
      MWC192<uint64_t> rng(42);
      volatile uint64_t sink = 0;
      auto start = std::chrono::high_resolution_clock::now();
      for (size_t i = 0; i < ITERS; i++) {
        sink += rng();
      }
      auto end = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration<double, std::milli>(end - start).count();
      spdlog::info("MWC192<uint64_t> raw: {:.2f} ms for {} ops ({:.2f} ns/op)", ms, ITERS, ms * 1e6 / ITERS);
    }
    {
      MWC192<uint32_t> rng(42);
      volatile uint32_t sink = 0;
      auto start = std::chrono::high_resolution_clock::now();
      for (size_t i = 0; i < ITERS; i++) {
        sink += rng();
      }
      auto end = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration<double, std::milli>(end - start).count();
      spdlog::info("MWC192<uint32_t> raw: {:.2f} ms for {} ops ({:.2f} ns/op)", ms, ITERS, ms * 1e6 / ITERS);
    }
    spdlog::info("");
  }

  spdlog::info("=== Summary ===");
  spdlog::info("This benchmark isolates distribution performance from EA algorithm overhead.");
  spdlog::info("Compare per-operation nanoseconds to identify hotspots:");
  spdlog::info("  - Raw RNG: measures engine speed alone (MWC192 vs mt19937)");
  spdlog::info("  - std::normal_distribution vs ziggurat: measures distribution algorithm overhead (Box-Muller with transcendentals vs Ziggurat with tables)");
  spdlog::info("  - std::uniform_real_distribution vs fast path: measures distribution overhead for uniform doubles");
  spdlog::info("  - std::uniform_int_distribution vs fast range reduction: measures distribution overhead for bounded ints");
  spdlog::info("");
  spdlog::info("In the full EA simulation, these distributions are called many times per step:");
  spdlog::info("  - Parent selection: 2 uniform_int per breed");
  spdlog::info("  - Tournament in try_insert: 1 uniform_int per step");
  spdlog::info("  - Crossover points: 2 uniform_int per TwoPoint crossover");
  spdlog::info("  - Mutation Bernoulli: N=10 generate_canonical per breed (fast path uses 1 rng call for 32 bits)");
  spdlog::info("  - Mutation perturbation: ~1 normal per breed on average (p=1/N per dimension)");
  spdlog::info("  - With HillClimbLS: Steps*N normals per local search (e.g., 100 normals per LS call)");
  spdlog::info("");
  spdlog::info("Ziggurat and fast paths reduce per-call overhead significantly, especially for normal distribution");
  spdlog::info("where Box-Muller does log, sqrt, sin, cos per call vs Ziggurat's mostly table lookups and comparisons.");

  return 0;
}
