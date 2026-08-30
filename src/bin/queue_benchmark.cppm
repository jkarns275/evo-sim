module;
#include <spdlog/sinks/basic_file_sink.h>
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

// Traits using old PriorityQueue
template <unsigned M>
struct SDBTraitsOldQ {
  static constexpr std::string_view name = "OldQ";
  static constexpr std::string_view static_name = "OldQ";
  using genome_t = Genome<M>;
  using mutation_t = GaussianMutationPolicy<Genome<M>, M>;
  using crossover_t = TwoPointCrossoverPolicy<Genome<M>, M>;
  using local_search_t = NoOpLocalSearch<Genome<M>, M>;
  using ls_time_model_t = Flat<M>;
  using init_policy_t = SimulatedInit<M>;
  using queue_t = PriorityQueue<Genome<M>>;
  using rng_t = MWC192<uint32_t>;
};

// Traits using new PriorityQueueLowCopy
template <unsigned M>
struct SDBTraitsNewQ {
  static constexpr std::string_view name = "NewQ";
  static constexpr std::string_view static_name = "NewQ";
  using genome_t = Genome<M>;
  using mutation_t = GaussianMutationPolicy<Genome<M>, M>;
  using crossover_t = TwoPointCrossoverPolicy<Genome<M>, M>;
  using local_search_t = NoOpLocalSearch<Genome<M>, M>;
  using ls_time_model_t = Flat<M>;
  using init_policy_t = SimulatedInit<M>;
  using queue_t = PriorityQueueLowCopy<Genome<M>>;
  using rng_t = MWC192<uint32_t>;
};

template <typename Traits>
double run_benchmark(const std::string& label, int runs = 1000) {
  const int POP_SIZE = 10;
  const int NP = 10;
  const int NGENOMES = 1000;

  Flat<N> fitness;
  Flat<N> time_fn;
  ZeroTime<N> ls_time;

  auto factory = [](Rng& rng, auto& tf) {
    return Genome<N>(rng, tf, std::normal_distribution<double>{0.0, 1.0});
  };

  auto start = std::chrono::high_resolution_clock::now();

  for (int run = 0; run < runs; run++) {
    Simulation<Traits, N> s(
        POP_SIZE, NP, NGENOMES, false, fitness, time_fn, factory, ls_time, run);
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

  spdlog::info("=== Priority Queue Head-to-Head Benchmark ===");
  spdlog::info("Genome<10> size: 104 bytes, HeapEntry size: 16 bytes");
  spdlog::info("Each push/pop does ~4 heap trickle moves of 104 bytes (old) vs 16 bytes (new)");

  const int RUNS = 1000;

  double old_ms = run_benchmark<SDBTraitsOldQ<N>>("Old PriorityQueue (std::priority_queue<Genome>)", RUNS);
  double new_ms = run_benchmark<SDBTraitsNewQ<N>>("New PriorityQueueLowCopy (slot map + heap of time,key)", RUNS);

  double speedup = old_ms / new_ms;
  double reduction = (old_ms - new_ms) / old_ms * 100.0;

  spdlog::info("=== Results ===");
  spdlog::info("Old: {:.2f} ms, New: {:.2f} ms", old_ms, new_ms);
  spdlog::info("Speedup: {:.2f}x ({:.1f}% time reduction)", speedup, reduction);

  if (speedup > 1.0) {
    spdlog::info("New queue is faster as hypothesized - custom priority queue with slot map reduces heap trickle cost significantly.");
  } else {
    spdlog::info("New queue is not faster - heap trickle cost may be less significant than hypothesized, or slot map overhead outweighs benefits for this workload.");
  }

  return 0;
}
