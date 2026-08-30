module;
#include <string_view>

export module evosim:traits;
import :genome_base;
import :genome_counter;
import :operators;
import :init;
import :mwc192;
import :queue;
import :time_model;

export namespace evosim {

template <unsigned N>
struct GenomeTraitsBase {
  static constexpr std::string_view name = "base";
  static constexpr std::string_view static_name = "base";
  using genome_t = Genome<N>;
  using mutation_t = GeometricGaussianMutationPolicy<Genome<N>, N, true>;
  using crossover_t = NopCrossoverPolicy<Genome<N>, N>;
  using local_search_t = NoOpLocalSearch<Genome<N>, N>;
  using ls_time_model_t = FlatObj<N>;
  using init_policy_t = SimulatedInit<N>;
  using rng_t = MWC192<uint32_t>;
  using queue_t = PriorityQueueLowCopy<genome_t>;
  using old_queue_t = PriorityQueue<genome_t>; // old std::priority_queue based implementation kept as option // old std::priority_queue based implementation kept as option per user request
};

template <unsigned N>
struct SDBGenomeNoCOTraits {
  static constexpr std::string_view name = "Geometric Mutation; No-Op Crossover";
  static constexpr std::string_view static_name =
      "Geometric Mutation; No-Op Crossover";
  using genome_t = Genome<N>;
  using mutation_t = GeometricGaussianMutationPolicy<Genome<N>, N, false>;
  using crossover_t = NopCrossoverPolicy<Genome<N>, N>;
  using local_search_t = NoOpLocalSearch<Genome<N>, N>;
  using ls_time_model_t = FlatObj<N>;
  using init_policy_t = SimulatedInit<N>;
  using rng_t = MWC192<uint32_t>;
  using queue_t = PriorityQueueLowCopy<genome_t>;
  using old_queue_t = PriorityQueue<genome_t>; // old std::priority_queue based implementation kept as option // old std::priority_queue based implementation kept as option per user request
};

template <unsigned N>
struct SDBTraits {
  static constexpr std::string_view name =
      "Geometric Mutation; Two-Point Crossover";
  static constexpr std::string_view static_name =
      "Geometric Mutation; Two-Point Crossover";
  using genome_t = Genome<N>;
  using mutation_t = GeometricGaussianMutationPolicy<Genome<N>, N, false>;
  using crossover_t = TwoPointCrossoverPolicy<Genome<N>, N>;
  using local_search_t = NoOpLocalSearch<Genome<N>, N>;
  using ls_time_model_t = FlatObj<N>;
  using init_policy_t = SimulatedInit<N>;
  using rng_t = MWC192<uint32_t>;
  using queue_t = PriorityQueueLowCopy<genome_t>;
  using old_queue_t = PriorityQueue<genome_t>; // old std::priority_queue based implementation kept as option // old std::priority_queue based implementation kept as option per user request
};

template <unsigned N>
struct CounterTraits {
  static constexpr std::string_view name =
      "Genome with Counters; Geometric Mutation; Two-Point Crossover";
  static constexpr std::string_view static_name =
      "Genome with Counters; Geometric Mutation; Two-Point Crossover";
  using genome_t = GenomeWithCounter<N>;
  using mutation_t = GeometricGaussianMutationPolicy<GenomeWithCounter<N>, N, false>;
  using crossover_t = TwoPointCrossoverPolicy<GenomeWithCounter<N>, N>;
  using local_search_t = NoOpLocalSearch<GenomeWithCounter<N>, N>;
  using ls_time_model_t = FlatObj<N>;
  using init_policy_t = SimulatedInit<N>;
  using rng_t = MWC192<uint32_t>;
  using queue_t = PriorityQueueLowCopy<genome_t>;
  using old_queue_t = PriorityQueue<genome_t>; // old std::priority_queue based implementation kept as option // old std::priority_queue based implementation kept as option per user request
};

// Fast Gaussian mutation traits using ziggurat instead of std::normal_distribution, for benchmarking as requested.
// Much faster normal RNG via ziggurat algorithm (precomputed tables, rejection sampling) vs Box-Muller in std::normal_distribution.
// Created as new traits for benchmarking to compare against std::normal_distribution baseline.

template <unsigned N>
struct SDBTraitsFastGauss {
  static constexpr std::string_view name =
      "Geometric Ziggurat; Two-Point Crossover";
  static constexpr std::string_view static_name =
      "Geometric Ziggurat; Two-Point Crossover";
  using genome_t = Genome<N>;
  using mutation_t = GeometricGaussianMutationPolicy<Genome<N>, N, true>;
  using crossover_t = TwoPointCrossoverPolicy<Genome<N>, N>;
  using local_search_t = NoOpLocalSearch<Genome<N>, N>;
  using ls_time_model_t = FlatObj<N>;
  using init_policy_t = SimulatedInit<N>;
  using rng_t = MWC192<uint32_t>;
  using queue_t = PriorityQueueLowCopy<genome_t>;
  using old_queue_t = PriorityQueue<genome_t>;
};

template <unsigned N>
struct SDBGenomeNoCOTraitsFastGauss {
  static constexpr std::string_view name =
      "Geometric Ziggurat; No-Op Crossover";
  static constexpr std::string_view static_name =
      "Geometric Ziggurat; No-Op Crossover";
  using genome_t = Genome<N>;
  using mutation_t = GeometricGaussianMutationPolicy<Genome<N>, N, true>;
  using crossover_t = NopCrossoverPolicy<Genome<N>, N>;
  using local_search_t = NoOpLocalSearch<Genome<N>, N>;
  using ls_time_model_t = FlatObj<N>;
  using init_policy_t = SimulatedInit<N>;
  using rng_t = MWC192<uint32_t>;
  using queue_t = PriorityQueueLowCopy<genome_t>;
  using old_queue_t = PriorityQueue<genome_t>;
};

template <unsigned N>
struct CounterTraitsFastGauss {
  static constexpr std::string_view name =
      "Genome with Counters; Geometric Ziggurat; Two-Point Crossover";
  static constexpr std::string_view static_name =
      "Genome with Counters; Geometric Ziggurat; Two-Point Crossover";
  using genome_t = GenomeWithCounter<N>;
  using mutation_t = GeometricGaussianMutationPolicy<GenomeWithCounter<N>, N, true>;
  using crossover_t = TwoPointCrossoverPolicy<GenomeWithCounter<N>, N>;
  using local_search_t = NoOpLocalSearch<GenomeWithCounter<N>, N>;
  using ls_time_model_t = FlatObj<N>;
  using init_policy_t = SimulatedInit<N>;
  using rng_t = MWC192<uint32_t>;
  using queue_t = PriorityQueueLowCopy<genome_t>;
  using old_queue_t = PriorityQueue<genome_t>;
};

// Geometric-skipping Gaussian mutation traits for benchmarking.
// Uses geometric distribution to skip non-mutated dimensions, reducing RNG calls
// from N per child to ~1. Uses std::generate_canonical as requested.
template <unsigned N>
struct SDBTraitsGeometric {
  static constexpr std::string_view name =
      "Geometric Gaussian Mutation; Two-Point Crossover";
  static constexpr std::string_view static_name =
      "Geometric Gaussian Mutation; Two-Point Crossover";
  using genome_t = Genome<N>;
  using mutation_t = GeometricGaussianMutationPolicy<Genome<N>, N, false>;
  using crossover_t = TwoPointCrossoverPolicy<Genome<N>, N>;
  using local_search_t = NoOpLocalSearch<Genome<N>, N>;
  using ls_time_model_t = FlatObj<N>;
  using init_policy_t = SimulatedInit<N>;
  using rng_t = MWC192<uint32_t>;
  using queue_t = PriorityQueueLowCopy<genome_t>;
  using old_queue_t = PriorityQueue<genome_t>;
};

template <unsigned N>
struct SDBTraitsGeometricZiggurat {
  static constexpr std::string_view name =
      "Geometric Ziggurat Mutation; Two-Point Crossover";
  static constexpr std::string_view static_name =
      "Geometric Ziggurat Mutation; Two-Point Crossover";
  using genome_t = Genome<N>;
  using mutation_t = GeometricGaussianMutationPolicy<Genome<N>, N, true>;
  using crossover_t = TwoPointCrossoverPolicy<Genome<N>, N>;
  using local_search_t = NoOpLocalSearch<Genome<N>, N>;
  using ls_time_model_t = FlatObj<N>;
  using init_policy_t = SimulatedInit<N>;
  using rng_t = MWC192<uint32_t>;
  using queue_t = PriorityQueueLowCopy<genome_t>;
  using old_queue_t = PriorityQueue<genome_t>;
};

template <unsigned N>
struct SDBTraitsGeometricSingle {
  static constexpr std::string_view name =
      "Geometric Single Mutation; Two-Point Crossover";
  static constexpr std::string_view static_name =
      "Geometric Single Mutation; Two-Point Crossover";
  using genome_t = Genome<N>;
  using mutation_t = GeometricGaussianMutationPolicy<Genome<N>, N, true>;
  using crossover_t = TwoPointCrossoverPolicy<Genome<N>, N>;
  using local_search_t = NoOpLocalSearch<Genome<N>, N>;
  using ls_time_model_t = FlatObj<N>;
  using init_policy_t = SimulatedInit<N>;
  using rng_t = MWC192<uint32_t>;
  using queue_t = PriorityQueueLowCopy<genome_t>;
  using old_queue_t = PriorityQueue<genome_t>;
};

template <unsigned N>
struct SDBTraitsBP {
  static constexpr std::string_view name =
      "Gaussian Mutation; Two-Point Crossover; Backpropagation LS";
  static constexpr std::string_view static_name =
      "Gaussian Mutation; Two-Point Crossover; Backpropagation LS";
  using genome_t = Genome<N>;
  using mutation_t = GeometricGaussianMutationPolicy<Genome<N>, N, true>;
  using crossover_t = TwoPointCrossoverPolicy<Genome<N>, N>;
  using local_search_t = BackpropagationLS<Genome<N>, FlatObj<N>, N>;
  using ls_time_model_t = BackpropagationTimeStatic<Flat, N>;
  using init_policy_t = SimulatedInit<N>;
  using rng_t = MWC192<uint32_t>;
  using queue_t = PriorityQueueLowCopy<genome_t>;
  using old_queue_t = PriorityQueue<genome_t>; // old std::priority_queue based implementation kept as option // old std::priority_queue based implementation kept as option per user request
};

template <unsigned N>
struct SDBGenomeNoCOTraitsBP {
  static constexpr std::string_view name =
      "Gaussian Mutation; No-Op Crossover; Backpropagation LS";
  static constexpr std::string_view static_name =
      "Gaussian Mutation; No-Op Crossover; Backpropagation LS";
  using genome_t = Genome<N>;
  using mutation_t = GeometricGaussianMutationPolicy<Genome<N>, N, true>;
  using crossover_t = NopCrossoverPolicy<Genome<N>, N>;
  using local_search_t = BackpropagationLS<Genome<N>, FlatObj<N>, N>;
  using ls_time_model_t = BackpropagationTimeStatic<Flat, N>;
  using init_policy_t = SimulatedInit<N>;
  using rng_t = MWC192<uint32_t>;
  using queue_t = PriorityQueueLowCopy<genome_t>;
  using old_queue_t = PriorityQueue<genome_t>; // old std::priority_queue based implementation kept as option // old std::priority_queue based implementation kept as option per user request
};

template <unsigned N>
struct CounterTraitsBP {
  static constexpr std::string_view name =
      "Genome with Counters; Gaussian Mutation; Two-Point Crossover; "
      "Backpropagation LS";
  static constexpr std::string_view static_name =
      "Genome with Counters; Gaussian Mutation; Two-Point Crossover; Backpropagation LS";
  using genome_t = GenomeWithCounter<N>;
  using mutation_t = GeometricGaussianMutationPolicy<GenomeWithCounter<N>, N, true>;
  using crossover_t = TwoPointCrossoverPolicy<GenomeWithCounter<N>, N>;
  using local_search_t = BackpropagationLS<GenomeWithCounter<N>, FlatObj<N>, N>;
  using ls_time_model_t = BackpropagationTimeStatic<Flat, N>;
  using init_policy_t = SimulatedInit<N>;
  using rng_t = MWC192<uint32_t>;
  using queue_t = PriorityQueueLowCopy<genome_t>;
  using old_queue_t = PriorityQueue<genome_t>; // old std::priority_queue based implementation kept as option // old std::priority_queue based implementation kept as option per user request
};

} // namespace evosim
