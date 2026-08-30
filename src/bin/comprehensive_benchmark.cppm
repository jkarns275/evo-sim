module;
#include <spdlog/spdlog.h>
#include <chrono>
#include <random>
#include <vector>
#include <string>
#include <iomanip>

export module evosim.main;
import evosim;

using namespace evosim;
const unsigned N = 10;

// Zero time function for benchmarking (no local search time)
template <unsigned M>
struct ZeroTime : public Flat<M> {
  double operator()(const std::array<double, M>&) const {
    return 0.0;
  }
  std::string to_string() const { return "Zero"; }
};

// Fully old baseline: std::mt19937_64 + per-dim GaussianMutationPolicy + std::uniform + old queue
// This represents the original code before any optimizations:
// - Old RNG: std::mt19937_64 instead of MWC192
// - Std sampling: std::uniform_int_distribution instead of FastUniformInt
// - Inefficient gaussian: std::normal_distribution (Box-Muller) per-dim loop
// - Old queue: std::priority_queue based PriorityQueue instead of PriorityQueueLowCopy
template <unsigned M>
struct FullyOldTraits {
  static constexpr std::string_view name = "Fully Old: mt19937_64 + per-dim std + std::uniform + old queue";
  static constexpr std::string_view static_name = "Fully Old: mt19937_64 + per-dim std + std::uniform + old queue";
  using genome_t = Genome<M>;
  using mutation_t = GaussianMutationPolicy<Genome<M>, M>;
  using crossover_t = TwoPointCrossoverPolicy<Genome<M>, M, StdUniformIntDist<uint32_t>>;
  using local_search_t = NoOpLocalSearch<Genome<M>, M>;
  using ls_time_model_t = FlatObj<M>;
  using init_policy_t = SimulatedInit<M>;
  using rng_t = std::mt19937_64;
  using queue_t = PriorityQueue<genome_t>;  // old std::priority_queue based
  using uniform_int_dist_t = StdUniformIntDist<uint32_t>;
};

// Baseline traits: std::mt19937_64 + per-dim GaussianMutationPolicy
// Uses std::uniform_int_distribution for all uniform int draws (parent selection, etc.)
// Uses new PriorityQueueLowCopy (so only RNG, normal, and uniform are old)
template <unsigned M>
struct BaselineTraits {
  static constexpr std::string_view name = "Baseline: mt19937_64 + per-dim std normal";
  static constexpr std::string_view static_name = "Baseline: mt19937_64 + per-dim std normal";
  using genome_t = Genome<M>;
  using mutation_t = GaussianMutationPolicy<Genome<M>, M>;
  using crossover_t = TwoPointCrossoverPolicy<Genome<M>, M, StdUniformIntDist<uint32_t>>;
  using local_search_t = NoOpLocalSearch<Genome<M>, M>;
  using ls_time_model_t = FlatObj<M>;
  using init_policy_t = SimulatedInit<M>;
  using rng_t = std::mt19937_64;
  using queue_t = PriorityQueueLowCopy<genome_t>;
  using uniform_int_dist_t = StdUniformIntDist<uint32_t>;
};

// Ziggurat only: mt19937_64 + per-dim FastGaussian (ziggurat)
// Uses std::uniform_int_distribution for all uniform int draws
template <unsigned M>
struct ZigguratTraits {
  static constexpr std::string_view name = "Ziggurat: mt19937_64 + per-dim ziggurat";
  static constexpr std::string_view static_name = "Ziggurat: mt19937_64 + per-dim ziggurat";
  using genome_t = Genome<M>;
  using mutation_t = FastGaussianMutationPolicy<Genome<M>, M>;
  using crossover_t = TwoPointCrossoverPolicy<Genome<M>, M, StdUniformIntDist<uint32_t>>;
  using local_search_t = NoOpLocalSearch<Genome<M>, M>;
  using ls_time_model_t = FlatObj<M>;
  using init_policy_t = SimulatedInit<M>;
  using rng_t = std::mt19937_64;
  using queue_t = PriorityQueueLowCopy<genome_t>;
  using uniform_int_dist_t = StdUniformIntDist<uint32_t>;
};

// Custom RNG only: MWC192 + per-dim GaussianMutationPolicy
template <unsigned M>
struct CustomRngTraits {
  static constexpr std::string_view name = "Custom RNG: MWC192 + per-dim std normal";
  static constexpr std::string_view static_name = "Custom RNG: MWC192 + per-dim std normal";
  using genome_t = Genome<M>;
  using mutation_t = GaussianMutationPolicy<Genome<M>, M>;
  using crossover_t = TwoPointCrossoverPolicy<Genome<M>, M>;
  using local_search_t = NoOpLocalSearch<Genome<M>, M>;
  using ls_time_model_t = FlatObj<M>;
  using init_policy_t = SimulatedInit<M>;
  using rng_t = MWC192<uint32_t>;
  using queue_t = PriorityQueueLowCopy<genome_t>;
};


// Custom RNG + Ziggurat: MWC192 + per-dim FastGaussian
template <unsigned M>
struct CustomRngZigguratTraits {
  static constexpr std::string_view name = "Custom RNG + Ziggurat: MWC192 + per-dim ziggurat";
  static constexpr std::string_view static_name = "Custom RNG + Ziggurat: MWC192 + per-dim ziggurat";
  using genome_t = Genome<M>;
  using mutation_t = FastGaussianMutationPolicy<Genome<M>, M>;
  using crossover_t = TwoPointCrossoverPolicy<Genome<M>, M>;
  using local_search_t = NoOpLocalSearch<Genome<M>, M>;
  using ls_time_model_t = FlatObj<M>;
  using init_policy_t = SimulatedInit<M>;
  using rng_t = MWC192<uint32_t>;
  using queue_t = PriorityQueueLowCopy<genome_t>;
};

// Geometric skipping with std normal: MWC192 + geometric + std normal
template <unsigned M>
struct GeometricStdTraits {
  static constexpr std::string_view name = "Geometric: MWC192 + geometric std normal";
  static constexpr std::string_view static_name = "Geometric: MWC192 + geometric std normal";
  using genome_t = Genome<M>;
  using mutation_t = GeometricGaussianMutationPolicy<Genome<M>, M, false>;
  using crossover_t = TwoPointCrossoverPolicy<Genome<M>, M>;
  using local_search_t = NoOpLocalSearch<Genome<M>, M>;
  using ls_time_model_t = FlatObj<M>;
  using init_policy_t = SimulatedInit<M>;
  using rng_t = MWC192<uint32_t>;
  using queue_t = PriorityQueueLowCopy<genome_t>;
};

// Geometric skipping with ziggurat: MWC192 + geometric + ziggurat
template <unsigned M>
struct GeometricZigguratTraits {
  static constexpr std::string_view name = "Geometric Ziggurat: MWC192 + geometric ziggurat";
  static constexpr std::string_view static_name = "Geometric Ziggurat: MWC192 + geometric ziggurat";
  using genome_t = Genome<M>;
  using mutation_t = GeometricGaussianMutationPolicy<Genome<M>, M, true>;
  using crossover_t = TwoPointCrossoverPolicy<Genome<M>, M>;
  using local_search_t = NoOpLocalSearch<Genome<M>, M>;
  using ls_time_model_t = FlatObj<M>;
  using init_policy_t = SimulatedInit<M>;
  using rng_t = MWC192<uint32_t>;
  using queue_t = PriorityQueueLowCopy<genome_t>;
};

// Geometric single mutation: MWC192 + single uniform index + ziggurat
template <unsigned M>
struct GeometricSingleTraits {
  static constexpr std::string_view name = "Geometric Single: MWC192 + single ziggurat";
  static constexpr std::string_view static_name = "Geometric Single: MWC192 + single ziggurat";
  using genome_t = Genome<M>;
  using mutation_t = GeometricSingleMutationPolicy<Genome<M>, M, true>;
  using crossover_t = TwoPointCrossoverPolicy<Genome<M>, M>;
  using local_search_t = NoOpLocalSearch<Genome<M>, M>;
  using ls_time_model_t = FlatObj<M>;
  using init_policy_t = SimulatedInit<M>;
  using rng_t = MWC192<uint32_t>;
  using queue_t = PriorityQueueLowCopy<genome_t>;
};

template <unsigned M>
struct StdUniformTraits {
  static constexpr std::string_view name = "Std Uniform: MWC192 + single ziggurat + std::uniform_int";
  static constexpr std::string_view static_name = "Std Uniform: MWC192 + single ziggurat + std::uniform_int";
  using genome_t = Genome<M>;
  using mutation_t = GeometricSingleMutationPolicy<Genome<M>, M, true, StdUniformIntDist<uint32_t>>;
  using crossover_t = TwoPointCrossoverPolicy<Genome<M>, M, StdUniformIntDist<uint32_t>>;
  using local_search_t = NoOpLocalSearch<Genome<M>, M>;
  using ls_time_model_t = FlatObj<M>;
  using init_policy_t = SimulatedInit<M>;
  using rng_t = MWC192<uint32_t>;
  using queue_t = PriorityQueueLowCopy<genome_t>;
  using uniform_int_dist_t = StdUniformIntDist<uint32_t>;
};

// Old queue vs new queue: MWC192 + geometric ziggurat + old PriorityQueue
template <unsigned M>
struct OldQueueTraits {
  static constexpr std::string_view name = "Old Queue: MWC192 + geometric ziggurat + std::priority_queue";
  static constexpr std::string_view static_name = "Old Queue: MWC192 + geometric ziggurat + std::priority_queue";
  using genome_t = Genome<M>;
  using mutation_t = GeometricGaussianMutationPolicy<Genome<M>, M, true>;
  using crossover_t = TwoPointCrossoverPolicy<Genome<M>, M>;
  using local_search_t = NoOpLocalSearch<Genome<M>, M>;
  using ls_time_model_t = FlatObj<M>;
  using init_policy_t = SimulatedInit<M>;
  using rng_t = MWC192<uint32_t>;
  using queue_t = PriorityQueue<genome_t>;  // old std::priority_queue based
};

template <typename Traits>
double run_benchmark(const std::string& label, int runs = 1000) {
  const int POP_SIZE = 10;
  const int NP = 10;
  const int NGENOMES = 1000;

  ScottDeJongBasins<N> fitness(1.0, 2.0);
  ScottDeJongBasins<N> time_fn(1.0, 2.0);
  ZeroTime<N> ls_time;

  auto start = std::chrono::high_resolution_clock::now();

  for (int run = 0; run < runs; run++) {
    typename Traits::rng_t rng{static_cast<uint64_t>(run)};
    std::vector<typename Traits::genome_t> init_pop;
    init_pop.reserve(POP_SIZE);
    for (int i = 0; i < POP_SIZE; i++) {
      typename Traits::genome_t g(rng, time_fn, std::normal_distribution<double>{0.0, 1.0});
      g.clear_fitness();
      g.set_fitness(fitness);
      init_pop.push_back(std::move(g));
    }
    Simulation<Traits, N> s(
        init_pop, NP, NGENOMES, false, fitness, time_fn, ls_time, run);
    s.run();
  }

  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();
  spdlog::info("{:<55} {:>8.2f} ms  ({:>6.2f} ms/run)", label, ms, ms / runs);
  return ms;
}

int main() {
  initialize_logger();
  spdlog::set_level(spdlog::level::info);

  spdlog::info("=== Comprehensive EA Benchmark ===");
  spdlog::info("Measuring performance impact of incremental optimizations:");
  spdlog::info("  1. Fully Old: mt19937_64 + per-dim std normal + std::uniform + old queue (original code)");
  spdlog::info("  2. Baseline: mt19937_64 + per-dim std normal + std::uniform + new queue (only queue optimized)");
  spdlog::info("  3. Custom RNG: MWC192 instead of mt19937_64, per-dim std normal, new queue");
  spdlog::info("  4. Ziggurat: mt19937_64 + ziggurat instead of std::normal, new queue");
  spdlog::info("  5. Custom RNG + Ziggurat: MWC192 + ziggurat, per-dim loop, new queue");
  spdlog::info("  6. Geometric skipping: MWC192 + geometric + std normal (fast bit-cast uniform), new queue");
  spdlog::info("  7. Geometric Ziggurat: MWC192 + geometric + ziggurat (fast bit-cast), new queue");
  spdlog::info("  8. Geometric Single: MWC192 + single uniform index + ziggurat (exactly one mutation), new queue");
  spdlog::info("  9. Std Uniform: MWC192 + single ziggurat + std::uniform_int (vs FastUniformInt), new queue");
  spdlog::info(" 10. Old Queue: MWC192 + geometric ziggurat + old std::priority_queue (queue only)");
  spdlog::info("");
  spdlog::info("Configuration: N=10, POP_SIZE=10, NP=10, NGENOMES=1000, Fitness=ScottDeJongBasins");
  spdlog::info("");

  const int RUNS = 50;

  std::vector<std::pair<std::string, double>> results;

  results.emplace_back("1. Fully Old (mt19937_64 + per-dim std + std::uniform + old Q)",
      run_benchmark<FullyOldTraits<N>>("1. Fully Old (mt19937_64 + per-dim std + old Q)", RUNS));
  results.emplace_back("2. Baseline (mt19937_64 + per-dim std + std::uniform + new Q)",
      run_benchmark<BaselineTraits<N>>("2. Baseline (mt19937_64 + per-dim std + new Q)", RUNS));
  results.emplace_back("3. Custom RNG (MWC192 + per-dim std)",
      run_benchmark<CustomRngTraits<N>>("3. Custom RNG (MWC192 + per-dim std)", RUNS));
  results.emplace_back("4. Ziggurat (mt19937_64 + per-dim ziggurat)",
      run_benchmark<ZigguratTraits<N>>("4. Ziggurat (mt19937_64 + per-dim ziggurat)", RUNS));
  results.emplace_back("5. Custom RNG + Ziggurat (MWC192 + per-dim ziggurat)",
      run_benchmark<CustomRngZigguratTraits<N>>("5. Custom RNG + Ziggurat", RUNS));
  results.emplace_back("6. Geometric Std (MWC192 + geometric std)",
      run_benchmark<GeometricStdTraits<N>>("6. Geometric Std (MWC192 + geometric std)", RUNS));
  results.emplace_back("7. Geometric Ziggurat (MWC192 + geometric ziggurat)",
      run_benchmark<GeometricZigguratTraits<N>>("7. Geometric Ziggurat (MWC192 + geometric)", RUNS));
  results.emplace_back("8. Geometric Single (MWC192 + single ziggurat)",
      run_benchmark<GeometricSingleTraits<N>>("8. Geometric Single (MWC192 + single)", RUNS));
  results.emplace_back("9. Std Uniform (MWC192 + single ziggurat + std::uniform)",
      run_benchmark<StdUniformTraits<N>>("9. Std Uniform (MWC192 + single + std::uniform)", RUNS));
  results.emplace_back("10. Old Queue (MWC192 + geometric ziggurat + old queue)",
      run_benchmark<OldQueueTraits<N>>("10. Old Queue (MWC192 + geometric + old Q)", RUNS));

  spdlog::info("");
  spdlog::info("=== Summary (speedup vs baseline) ===");
  double baseline = results[0].second;
  spdlog::info("{:<55} {:>10} {:>10} {:>10}", "Configuration", "Time (ms)", "Speedup", "Reduction");
  spdlog::info("{:<55} {:>10} {:>10} {:>10}", std::string(55, '-'), std::string(10, '-'), std::string(10, '-'), std::string(10, '-'));
  for (auto& [label, ms] : results) {
    double speedup = baseline / ms;
    double reduction = (baseline - ms) / baseline * 100.0;
    spdlog::info("{:<55} {:>10.2f} {:>9.2f}x {:>9.1f}%", label, ms, speedup, reduction);
  }

  spdlog::info("");
  spdlog::info("Key insights:");
  spdlog::info("  - Custom RNG (MWC192) vs mt19937_64: measures RNG speed impact");
  spdlog::info("  - Ziggurat vs std::normal: measures normal distribution speed");
  spdlog::info("  - Geometric skipping vs per-dim loop: reduces RNG calls from N to ~1 per child");
  spdlog::info("  - Geometric Single vs Geometric: exactly one mutation vs Bernoulli(p) distribution");
  spdlog::info("  - FastUniformInt vs std::uniform_int: measures fast range reduction impact (config 7 vs 8)");
  spdlog::info("    FastUniformInt uses (rng * n) >> 32 instead of std::uniform_int_distribution");
  spdlog::info("    which does rejection sampling and generic case handling. Saves ~8 ns per draw.");
  spdlog::info("    In full EA: 6 draws per child (2 parents, 2 crossover, 1 mut idx, 1 pop) * 1M children = 6M draws.");
  spdlog::info("  - Old Queue vs PriorityQueueLowCopy: measures queue optimization impact");
  spdlog::info("    PriorityQueueLowCopy stores 16-byte HeapEntry vs 104-byte Genome, reducing cache pressure.");
  spdlog::info("");
  spdlog::info("For isolated FastUniformInt microbenchmark, also run: ./src/uniform_benchmark");

  return 0;
}
