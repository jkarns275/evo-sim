module;
#include <spdlog/spdlog.h>
#include <chrono>
#include <random>

export module evosim.main;
import evosim;

using namespace evosim;
const unsigned N = 10;

template <unsigned M>
struct ZeroTime : public Flat<M> {
  double operator()(const std::array<double, M>&) const {
    return 0.0;
  }
  std::string to_string() const { return "Zero"; }
};

template <typename Traits>
double run_benchmark(const std::string& label, int runs = 2000) {
  const int POP_SIZE = 10;
  const int NP = 10;
  const int NGENOMES = 1000;

  Flat<N> fitness;
  Flat<N> time_fn;
  ZeroTime<N> ls_time;

  auto start = std::chrono::high_resolution_clock::now();

  for (int run = 0; run < runs; run++) {
    // Create initial population manually to avoid std::function type issues with generic lambda
    Rng rng{static_cast<uint64_t>(run)};
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
  spdlog::info("{}: {} runs in {:.2f} ms ({:.2f} ms/run)", label, runs, ms, ms / runs);
  return ms;
}

int main() {
  initialize_logger();
  spdlog::set_level(spdlog::level::info);

  spdlog::info("=== Gaussian Mutation Benchmark: std::normal_distribution vs Ziggurat ===");
  spdlog::info("Comparing GaussianMutationPolicy (std::normal_distribution with Box-Muller) vs");
  spdlog::info("         FastGaussianMutationPolicy (cxx::ziggurat_normal_distribution with Ziggurat algorithm)");
  spdlog::info("Both using MWC192<uint32_t> RNG and PriorityQueueLowCopy, NoOp local search baseline");
  spdlog::info("");

  const int RUNS = 2000;

  double std_ms = run_benchmark<SDBTraits<N>>("std::normal_distribution (Box-Muller)", RUNS);
  double zig_ms = run_benchmark<SDBTraitsFastGauss<N>>("Ziggurat (cxx::ziggurat_normal_distribution)", RUNS);
  double geo_std_ms = run_benchmark<SDBTraitsGeometric<N>>("Geometric std::normal_distribution", RUNS);
  double geo_zig_ms = run_benchmark<SDBTraitsGeometricZiggurat<N>>("Geometric Ziggurat", RUNS);
  double geo_single_ms = run_benchmark<SDBTraitsGeometricSingle<N>>("Geometric Single (1 mutation)", RUNS);

  spdlog::info("");
  spdlog::info("=== Results ===");
  spdlog::info("std::normal_distribution: {:.2f} ms", std_ms);
  spdlog::info("Ziggurat: {:.2f} ms ({:.2f}x speedup, {:.1f}% time reduction)", zig_ms, std_ms / zig_ms, (std_ms - zig_ms) / std_ms * 100.0);
  spdlog::info("Geometric std: {:.2f} ms ({:.2f}x vs std, {:.1f}% reduction)", geo_std_ms, std_ms / geo_std_ms, (std_ms - geo_std_ms) / std_ms * 100.0);
  spdlog::info("Geometric Ziggurat: {:.2f} ms ({:.2f}x vs std, {:.1f}% reduction)", geo_zig_ms, std_ms / geo_zig_ms, (std_ms - geo_zig_ms) / std_ms * 100.0);
  spdlog::info("Geometric Single: {:.2f} ms ({:.2f}x vs std, {:.1f}% reduction)", geo_single_ms, std_ms / geo_single_ms, (std_ms - geo_single_ms) / std_ms * 100.0);

  if (zig_ms < std_ms) {
    spdlog::info("Ziggurat is faster as expected - Ziggurat algorithm uses precomputed tables and rejection sampling, typically 2-3x faster than Box-Muller which does log, sqrt, sin, cos per call.");
  } else {
    spdlog::info("Ziggurat is not faster - Box-Muller may be well optimized by compiler, or ziggurat overhead outweighs benefits for this workload with low mutation rate (p=1/N per dimension).");
  }

  return 0;
}
