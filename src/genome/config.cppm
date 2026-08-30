module;
#include <string>

export module evosim:genome_config;
import :genome_base;
import :genome_counter;
import :operators;
import :objective;
import :mwc192;
import :queue;

import :init;

export namespace evosim {

template <unsigned N>
struct SDBGenomeNoCOConfig {
  const std::string name = "Gaussian Mutation; No-Op Crossover";
  static constexpr std::string_view static_name =
      "Gaussian Mutation; No-Op Crossover";

  using genome_t = ::evosim::Genome<N>;
  using mutation_t = GaussianMutationPolicy<::evosim::Genome<N>, N>;
  using crossover_t = NopCrossoverPolicy<::evosim::Genome<N>, N>;
  using local_search_t = NoOpLocalSearch<::evosim::Genome<N>, N>;
  using ls_time_model_t = FlatObj<N>;
  using init_policy_t = SimulatedInit<N>;
  using rng_t = MWC192<uint32_t>;
  using queue_t = PriorityQueueLowCopy<genome_t>;
  using old_queue_t = PriorityQueue<genome_t>;
};

template <unsigned N>
struct SDBGenomeConfig {
  const std::string name = "Gaussian Mutation; Two-Point Crossover";
  static constexpr std::string_view static_name =
      "Gaussian Mutation; Two-Point Crossover";

  using genome_t = ::evosim::Genome<N>;
  using mutation_t = GaussianMutationPolicy<::evosim::Genome<N>, N>;
  using crossover_t = TwoPointCrossoverPolicy<::evosim::Genome<N>, N>;
  using local_search_t = NoOpLocalSearch<::evosim::Genome<N>, N>;
  using ls_time_model_t = FlatObj<N>;
  using init_policy_t = SimulatedInit<N>;
  using rng_t = MWC192<uint32_t>;
  using queue_t = PriorityQueueLowCopy<genome_t>;
  using old_queue_t = PriorityQueue<genome_t>;
};

template <unsigned N>
struct CounterGenomeConfig {
  const std::string name =
      "Genome with Counters; Gaussian Mutation; Two-Point Crossover";
  static constexpr std::string_view static_name =
      "Genome with Counters; Gaussian Mutation; Two-Point Crossover";

  using genome_t = ::evosim::GenomeWithCounter<N>;
  using mutation_t = GaussianMutationPolicy<::evosim::GenomeWithCounter<N>, N>;
  using crossover_t =
      TwoPointCrossoverPolicy<::evosim::GenomeWithCounter<N>, N>;
  using local_search_t = NoOpLocalSearch<::evosim::GenomeWithCounter<N>, N>;
  using ls_time_model_t = FlatObj<N>;
  using init_policy_t = SimulatedInit<N>;
  using rng_t = MWC192<uint32_t>;
  using queue_t = PriorityQueueLowCopy<genome_t>;
  using old_queue_t = PriorityQueue<genome_t>;
};

} // namespace evosim
