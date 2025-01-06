module;
#include <assert.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <concepts>
#include <functional>
#include <queue>
#include <random>
#include <type_traits>
#include <vector>

export module evosim:simulator;
import :util;
import :function;
import :genome;

export namespace evosim {

enum InitType { UNIFORM, SIMULATED };

/// Evolutionary algorithm simulation for some quasi-genome Ty.
///
/// The simulation will maintain a population of genomes, reproduction will occur using tournamant selection.
template <typename GC, unsigned N>
  requires std::derived_from<GC, GenomeConfig<N>>
struct Simulation {

  static_assert(std::derived_from<typename GC::Genome, Genome<N>>);
  /// Re-definition of `Genome` to correspond to the genome type as specified by the GenomeConfig class (i.e. GC).
  typedef GC::Genome Genome;

  static_assert(std::derived_from<typename GC::Mutation, Mutation<Genome, N>>);
  /// Re-definition of `Mutation` to correspond to the mutation type as specified by the GenomeConfig class (i.e. GC).
  typedef GC::Mutation Mutation;
  static_assert(!std::is_abstract_v<Mutation>);

  static_assert(std::derived_from<typename GC::Crossover, Crossover<Genome, N>>);
  /// Re-definition of `Crossover` to correspond to the mutation type as specified by the GenomeConfig class (i.e. GC).
  typedef GC::Crossover Crossover;
  static_assert(!std::is_abstract_v<Crossover>);

  /// Genomes pending evaluation. Must be ordered such that the smallest value is considered the largest as this is a
  /// max heap and we want to pull the most recently evaluated genome.
  std::priority_queue<Genome, std::vector<Genome>, std::greater<Genome>> pending_evaluation;

  /// Population of solutions sorted from highest fitness (best) to lowest fitness (worst).
  std::vector<Genome> population;

  /// RNG Engine
  Rng generator;

  /// Fitness landscape function.
  FitnessFunction<N> &fitness_value_fn;

  /// Function used to compute the amount of simulated time a genome will take to evaluate.
  FitnessFunction<N> &time_value_fn;

  /// The number of simulator worker processors -- this is the max size of the `pending_evaluation` queue.
  uint32_t np;

  /// How many steps have occurred thus far in the simulation.
  int64_t step_count = 0;

  /// Number of steps which constitute a single run of the simulation.
  int64_t max_steps;

  bool sort_by_fitness(const Genome &left, const Genome &right) const {
    return fitness_value_fn(left.x) < fitness_value_fn(right.x);
  }

  struct FitnessComparator {
    Simulation &sim;

    FitnessComparator(Simulation &sim) : sim(sim) {}

    bool operator()(const Genome &left, const Genome &right) const { return sim.sort_by_fitness(left, right); }
  };

  /// Uniformly selects a random member of the population and returns a reference to it.
  const Genome &select() {
    assert(population.size() > 0);
    size_t index = std::uniform_int_distribution<size_t>(0, population.size() - 1)(generator);
    return population[index];
  }

  bool done() { return step_count >= max_steps; }

  /// A single virtual step of the simulated asynchronous algorithm. This consists of "receiving" a genome from a worker
  /// by removing a genome from the pending evaluation queue, then inserting that genome in the population using
  /// tournamant selection. Subsequently, a new random genome is selected and mutated -- this child genome is
  /// subsequently added into the evaluation queue.
  void step() {
    assert(pending_evaluation.size() == np);

    // Get most recently evaluated genome, and insert it if it passes tournamant selection.
    Genome element = std::move(pending_evaluation.top());
    pending_evaluation.pop();
    // spdlog::info("Child finish time = {}", element.time_finished);
    // spdlog::info("Child time value  = {}", time_value_fn(element.x));
    // spdlog::info("Child             = {}", element.to_string());
    // spdlog::info("Global best       = {}", population[0].to_string());
    // spdlog::info("Current time = {}", element.get_finish_time());

    // Tournamant selection at index.
    size_t index = std::uniform_int_distribution<size_t>(0, population.size() - 1)(generator);
    const Genome &insertion_element = population[index];

    // If by the fitness ordering the insertion genome is less than the new genome, remove it. Then, in sorted order
    // insert the new genome. i.e. tournamant selection.
    if (!this->sort_by_fitness(insertion_element, element)) {
      population.erase(population.begin() + index);
      auto it = std::upper_bound(population.begin(), population.end(), element, FitnessComparator(*this));
      population.emplace(it, std::move(element));
    }

    // Randomly select a parent genome uniformly.
    size_t parent0_index = std::uniform_int_distribution<size_t>(0, population.size() - 1)(generator);
    const Genome &parent0 = population[parent0_index];

    size_t parent1_index;
    do {
      parent1_index = std::uniform_int_distribution<size_t>(0, population.size() - 1)(generator);
    } while (parent1_index == parent0_index);
    const Genome &parent1 = population[parent1_index];

    // Breed a child
    Genome child = Crossover()(parent0, parent1, generator);

    // Mutate that child
    child = Mutation()(child, generator);

    // Set that child's finish time to be equal to the parents finish time added to its own evaluation time.
    // spdlog::info("Time value = {}", time_value_fn(child));
    child.set_finish_time(element.get_finish_time() + time_value_fn(child.x));

    auto [min, max] = fitness_value_fn.domain();
    child.bound(min, max);

    // Add to execution queue.
    pending_evaluation.emplace(std::move(child));

    step_count += 1;
  }

public:
  /// Creates a new simulation with the supplied initial population. The size of this initial population will be
  /// considered the target size of the population during the simulation.
  ///
  /// The simulation will create `np` mutants and add them to the pending evaluation queue.
  Simulation(std::vector<Genome> &population, uint32_t np, int64_t max_steps, FitnessFunction<N> &fitness_value_fn,
             FitnessFunction<N> &time_value_fn)
      : population(std::move(population)), np(np), max_steps(max_steps), fitness_value_fn(fitness_value_fn),
        time_value_fn(time_value_fn) {
    assert(population.size() > 0);
    assert(max_steps > 0);

    initialize_rng();
    initialize_simulation();
  }

  Simulation(uint32_t pop_size, uint32_t np, int64_t max_steps, FitnessFunction<N> &fitness_value_fn,
             FitnessFunction<N> &time_value_fn, std::function<Genome(Rng &, FitnessFunction<N> &)> factory,
             InitType init_type = UNIFORM)
      : np(np), max_steps(max_steps), fitness_value_fn(fitness_value_fn), time_value_fn(time_value_fn) {
    initialize_rng();

    switch (init_type) {
    case UNIFORM:
      for (int i = 0; i < pop_size; i++)
        population.push_back(factory(generator, time_value_fn));
      break;
    case SIMULATED:
      while (population.size() < pop_size) {
        while (pending_evaluation.size() < np) {
          pending_evaluation.push(factory(generator, time_value_fn));
        }

        Genome genome = pending_evaluation.top();
        pending_evaluation.pop();

        auto it = std::upper_bound(population.begin(), population.end(), genome, FitnessComparator(*this));
        population.emplace(it, std::move(genome));
      }
      break;
    }
    initialize_simulation();
  }

  void initialize_rng() {
    std::random_device rd;
    generator.seed(rd());

    for (int i = 0; i < 1000; i++)
      std::generate_canonical<double, 52>(generator);
  }

  void initialize_simulation() {
    std::sort(population.begin(), population.end(), FitnessComparator(*this));

    while (pending_evaluation.size() < np)
      pending_evaluation.push(Mutation()(select(), generator));
  }

  /// Runs the simulation for `max_steps`. This resets `step_count`, meaning it can be called multiple times
  void run() {
    step_count = 0;
    while (!done()) {
      step();
    }
  }

  bool converged_to_global_best() const { return fitness_value_fn.converged(population.at(0).x); }

  Genome &get_best_genome() { return population[0]; }
};

}; // namespace evosim
