module;
#include <algorithm>
#include <cassert>
#include <format>
#include <functional>
#include <variant>
#include <vector>

export module evosim:init;
import :core;
import :objective;
import :genome_base;
import :population;
import :queue;

export namespace evosim {
namespace init_policy {
struct SynchronousGenerations {
  unsigned int n;
};

struct ConstTimeAsynchronousEvaluations {
  unsigned int n;

  ConstTimeAsynchronousEvaluations(unsigned int n) : n(n) {
    assert(n > 0);
  }
};

struct Simulated {};

using InitType = std::variant<
    SynchronousGenerations,
    ConstTimeAsynchronousEvaluations,
    Simulated>;
} // namespace init_policy

template <typename VariantType, typename T, std::size_t index = 0>
constexpr std::size_t variant_index() {
  if constexpr (index == std::variant_size_v<VariantType>)
    return index;
  else if constexpr (
      std::is_same_v<std::variant_alternative_t<index, VariantType>, T>)
    return index;
  else
    return variant_index<VariantType, T, index + 1>();
}

// Init policies operating on a Sim-like object that provides the same public
// interface as old Simulation / new Simulation2:
//  - population object with size(), data() returning vector<Genome>&,
//  sort_by_fitness_old(), try_insert(Genome&, Rng&), set_rng(Rng*)
//  - pending_evaluation queue with push(), pop(), top(), size(), empty(), clear
//  via pop loop
//  - generator Rng member
//  - np uint32_t
//  - fitness_value_fn and time_value_fn pointers to FitnessFunction<N>
//  - step_count int64_t
//  - breed() method returning Genome
//  - try_insert(Genome&) method that clears fitness and inserts via tournament
//  - step() method

template <unsigned N>
struct SynchronousGenerationsInit {
  unsigned int n = 0;
  SynchronousGenerationsInit() = default;

  explicit SynchronousGenerationsInit(unsigned int n_) : n(n_) {}

  template <class Sim, class Factory>
  void apply(Sim& sim, uint32_t pop_size, Factory& factory) const {
    using Genome = typename Sim::Genome;
    // Build initial population via factory, set fitness, then sort
    std::vector<Genome> init_vec;
    init_vec.reserve(pop_size);
    for (int i = 0; i < (int)pop_size; i++) {
      Genome g = factory(sim.generator, sim.time_value_fn);
      g.clear_fitness();
      g.set_fitness(sim.fitness_value_fn);
      init_vec.push_back(std::move(g));
    }
    sim.population = Population<Genome>(std::move(init_vec));
    sim.population.set_rng(&sim.generator);
    sim.population.sort_by_fitness_old();
    for (int i = 0; i < (int)n; i++) {
      std::vector<Genome> new_pop;
      new_pop.reserve(sim.population.size());
      for (size_t j = 0; j < sim.population.size(); j++)
        new_pop.push_back(sim.breed());
      for (size_t j = 0; j < sim.population.size(); j++)
        sim.try_insert(new_pop[j]);
    }
    sim.step_count += sim.population.size() * n;
  }
};

template <unsigned N>
struct SimulatedInit {
  template <class Sim, class Factory>
  void apply(Sim& sim, uint32_t pop_size, Factory& factory) const {
    using Genome = typename Sim::Genome;
    // Build initial population via async simulation of evaluations, matching
    // old Simulation constructor logic exactly
    std::vector<Genome> init_vec;
    init_vec.reserve(pop_size);
    while (init_vec.size() < (size_t)pop_size) {
      while (sim.pending_evaluation.size() < sim.np) {
        sim.pending_evaluation.push(factory(sim.generator, sim.time_value_fn));
      }
      Genome genome = std::move(sim.pending_evaluation.top());
      sim.pending_evaluation.pop();
      genome.set_fitness(sim.fitness_value_fn);
      auto it = std::upper_bound(
          init_vec.begin(),
          init_vec.end(),
          genome,
          [](const Genome& l, const Genome& r) {
            return *l.fitness < *r.fitness;
          });
      init_vec.emplace(it, std::move(genome));
    }
    sim.population = Population<typename Sim::Genome>(std::move(init_vec));
    sim.population.set_rng(&sim.generator);
  }
};

template <unsigned N>
struct ConstTimeAsyncInit {
  unsigned int n;

  ConstTimeAsyncInit(unsigned int n) : n(n) {}

  template <class Sim, class Factory>
  void apply(Sim& sim, uint32_t pop_size, Factory& factory) const {
    // First do simulated init to fill population
    SimulatedInit<N> base;
    base.apply(sim, pop_size, factory);
    // Then step nevals times using flat time model temporarily as old code does
    FlatObj<N> flat_time_fn{};
    auto* saved_time_fn = sim.time_value_fn;
    sim.time_value_fn = &flat_time_fn;
    for (int i = 0; i < (int)n; i++)
      sim.step();
    // Rebuild pending queue with new times using flat time model then restore
    // original time fn after
    std::vector<typename Sim::Genome> genomes;
    genomes.reserve(sim.pending_evaluation.size());
    while (!sim.pending_evaluation.empty()) {
      genomes.push_back(
          std::move(
              const_cast<typename Sim::Genome&>(sim.pending_evaluation.top())));
      sim.pending_evaluation.pop();
    }
    for (typename Sim::Genome& g : genomes) {
      g.set_finish_time(sim.time_value_fn.operator()(g.x));
      sim.pending_evaluation.push(std::move(g));
    }
    sim.time_value_fn = saved_time_fn;
  }
};

template <unsigned N>
struct InitDispatcher {
  template <class Sim, class Factory>
  static void apply(
      Sim& sim,
      uint32_t pop_size,
      Factory& factory,
      const typename init_policy::InitType& init_type) {
    std::visit(
        [&](auto&& cfg) {
          using T = std::decay_t<decltype(cfg)>;
          if constexpr (
              std::is_same_v<T, init_policy::SynchronousGenerations>) {
            SynchronousGenerationsInit<N> policy{cfg.n};
            policy.apply(sim, pop_size, factory);
          } else if constexpr (
              std::
                  is_same_v<T, init_policy::ConstTimeAsynchronousEvaluations>) {
            ConstTimeAsyncInit<N> policy{cfg.n};
            policy.apply(sim, pop_size, factory);
          } else if constexpr (std::is_same_v<T, init_policy::Simulated>) {
            SimulatedInit<N> policy;
            policy.apply(sim, pop_size, factory);
          }
        },
        init_type);
  }
};

inline std::string init_type_to_string(const init_policy::InitType& init_type) {
  return std::visit(
      [](auto&& cfg) -> std::string {
        using T = std::decay_t<decltype(cfg)>;
        if constexpr (std::is_same_v<T, init_policy::SynchronousGenerations>) {
          return std::format("SynchronousGenerations({})", cfg.n);
        } else if constexpr (std::is_same_v<
                                 T,
                                 init_policy::ConstTimeAsynchronousEvaluations>) {
          return std::format("ConstTimeAsynchronousEvaluations({})", cfg.n);
        } else if constexpr (std::is_same_v<T, init_policy::Simulated>) {
          return "Simulated";
        } else {
          return "Unknown";
        }
      },
      init_type);
}

} // namespace evosim
