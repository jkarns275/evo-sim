module;
#include <algorithm>
#include <cassert>
#include <concepts>
#include <functional>
#include <memory>
#include <queue>
#include <random>
#include <type_traits>
#include <variant>
#include <vector>

export module evosim:simulation;
import :core;
import :fast_uniform;
import :genome_base;
import :genome_counter;
import :population;
import :queue;

import :init;
import :traits;
import :concepts;
import :operators;
import :genome_config;
import :fitness;

export namespace evosim {

template <
    typename Traits,
    unsigned N,
    typename FitnessObj = FitnessFunction<N>,
    typename TimeObj = FitnessFunction<N>,
    typename LsTimeObj = FitnessFunction<N>>
  requires GenomeTraitsType<Traits> && Objective<FitnessObj, N> && Objective<TimeObj, N> && Objective<LsTimeObj, N>
struct Simulation {
  typedef typename Traits::genome_t Genome;
  typedef typename Traits::mutation_t Mutation;
  typedef typename Traits::crossover_t Crossover;
  typedef typename Traits::local_search_t LocalSearch;

  using Queue = typename Traits::queue_t;
  Queue pending_evaluation;
  Population<Genome, typename Traits::rng_t, typename get_uniform_dist<Traits>::type> population;
  typename Traits::rng_t generator;
  FitnessObj& fitness_value_fn;
  TimeObj& time_value_fn;
  LsTimeObj& ls_time_fn;
  uint32_t np;
  int64_t step_count = 0;
  int64_t max_steps;
  bool use_sweet = false;

  bool sort_by_fitness(const Genome& left, const Genome& right) const {
    return *left.fitness < *right.fitness;
  }

  struct FitnessComparator {
    Simulation& sim;

    FitnessComparator(Simulation& sim) : sim(sim) {}

    bool operator()(const Genome& l, const Genome& r) const {
      return sim.sort_by_fitness(l, r);
    }
  };

  const Genome& select() {
    return population.select_uniform(generator);
  }

  bool done() {
    return step_count >= max_steps;
  }

  void try_insert(Genome& element) {
    element.clear_fitness();
    element.set_fitness(fitness_value_fn);

    population.try_insert(element, generator);
  }

  Genome breed() {
    size_t n_genomes =
        population.size() + (use_sweet ? pending_evaluation.size() : 0);

    // Use fast range reduction instead of std::uniform_int_distribution for speed.
    // std::uniform_int_distribution is slow due to generic case handling and rejection sampling overhead.
    // Fast path: (rng() * n) >> 32 for 32-bit, >> 64 for 64-bit, minimal bias acceptable for EA.
    // Now using uniform_int_dist_t which defaults to FastUniformIntDist but can be
    // configured via typedef to StdUniformIntDist for benchmarking, or via
    // Traits::uniform_int_dist_t for per-trait configuration.
    using UniformDist = typename get_uniform_dist<Traits>::type;
    size_t parent0_index = UniformDist(static_cast<uint32_t>(n_genomes))(generator);
    size_t parent1_index;
    do {
      parent1_index = UniformDist(static_cast<uint32_t>(n_genomes))(generator);
    } while (parent1_index == parent0_index);

    const Genome& parent0 = parent0_index < population.size()
        ? population[parent0_index]
        : pending_evaluation[parent0_index - population.size()];

    const Genome& parent1 = parent1_index < population.size()
        ? population[parent1_index]
        : pending_evaluation[parent1_index - population.size()];

    Genome child = Crossover::apply(parent0, parent1, generator);
    child = Mutation::apply(child, generator);

    return child;
  }

  void reproduce(double current_time) {
    Genome child = this->breed();

    // Local search hook - atomic step as per updated plan
    LocalSearch::apply(child, generator);

    double ls_t = ls_time_fn(child.x);
    double ev_t = time_value_fn(child.x);

    child.set_finish_time(current_time + ls_t + ev_t);

    auto [min, max] = fitness_value_fn.domain();
    child.bound(min, max);

    pending_evaluation.push(std::move(child));
  }

  void step() {
    assert(pending_evaluation.size() == np);

    Genome element = std::move(pending_evaluation.top());
    pending_evaluation.pop();

    try_insert(element);

    reproduce(element.get_finish_time());

    step_count += 1;
  }

 public:
  Simulation(
      std::vector<Genome>& pop_init,
      uint32_t np,
      int64_t max_steps,
      bool use_sweet,
      FitnessObj& fitness_value_fn,
      TimeObj& time_value_fn,
      FitnessObj& ls_time_fn,
      uint64_t seed = 0)
      : population(std::move(pop_init)),
        generator(seed ? make_rng<typename Traits::rng_t>(seed) : make_rng<typename Traits::rng_t>()),
        fitness_value_fn(fitness_value_fn),
        time_value_fn(time_value_fn),
        ls_time_fn(ls_time_fn),
        np(np),
        max_steps(max_steps),
        use_sweet(use_sweet) {
    population.set_rng(&generator);
    assert(population.size() > 0);
    assert(max_steps > 0);
    initialize_simulation();
  }

  Simulation(
      uint32_t pop_size,
      uint32_t np,
      int64_t max_steps,
      bool use_sweet,
      FitnessObj& fitness_value_fn,
      TimeObj& time_value_fn,
      std::function<Genome(typename Traits::rng_t&, FitnessFunction<N>&)> factory,
      FitnessObj& ls_time_fn,
      uint64_t seed = 0)
      : generator(seed ? make_rng<typename Traits::rng_t>(seed) : make_rng<typename Traits::rng_t>()),
        fitness_value_fn(fitness_value_fn),
        time_value_fn(time_value_fn),
        ls_time_fn(ls_time_fn),
        np(np),
        max_steps(max_steps),
        use_sweet(use_sweet) {
    population.set_rng(&generator);
    // Use init policy from traits as default, no runtime dispatch glue needed
    typename Traits::init_policy_t init_policy{};
    init_policy.apply(*this, pop_size, factory);
    this->time_value_fn = time_value_fn;
    initialize_simulation();
  }

  void initialize_simulation() {
    population.sort_by_fitness_old();
    while (pending_evaluation.size() < np) {
      Genome child = Mutation::apply(select(), generator);
      child.clear_fitness();
      child.set_fitness(fitness_value_fn);
      child.set_finish_time(time_value_fn(child.x));
      pending_evaluation.push(std::move(child));
    }
  }

  void run() {
    step_count = 0;
    while (!done())
      step();
  }

  bool converged_to_global_best() const {
    return fitness_value_fn.converged(population.data()[0].x);
  }

  Genome& get_best_genome() {
    return population.data()[0];
  }

  const std::vector<Genome>& get_population_data() const {
    return population.data();
  }

  size_t pending_size() const {
    return pending_evaluation.size();
  }

  double pending_top_time() const {
    return pending_evaluation.empty()
        ? 0.0
        : pending_evaluation.top().get_finish_time();
  }
};

} // namespace evosim
