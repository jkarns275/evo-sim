module;
#include <assert.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <concepts>
#include <functional>
#include <queue>
#include <random>
#include <type_traits>
#include <variant>
#include <vector>

export module evosim:simulator;
import :util;
import :function;
import :genome;

export namespace evosim {
namespace init_type {

/// This init type initializes the population using the factory function supplied to the simulation, then runs N
/// generations of synchronous evolution (genomes inserted using tournamant selection).
struct SynchronousGenerations {
  /// The number of synchronous generations to simulate
  unsigned int n;
};

/// This init type is like init_type::Simulated, but uses a constant time function.
struct ConstTimeAsynchronousEvaluations {
  /// The number of genomes to evaluate asynchronously using a constant time function,
  unsigned int n;

  ConstTimeAsynchronousEvaluations(unsigned int n) : n(n) { assert(n > 0); }
};

/// Asynchronously evaluates genomes until the population is full.
struct Simulated {};

typedef std::variant<init_type::SynchronousGenerations, init_type::ConstTimeAsynchronousEvaluations,
                     init_type::Simulated>
    InitType;
}; // namespace init_type

using init_type::InitType;

std::string init_type_to_string(InitType init_type) {
  switch (init_type.index()) {
  case variant_index<InitType, init_type::SynchronousGenerations>():
    return std::format("SynchronousGenerations({})", std::get<init_type::SynchronousGenerations>(init_type).n);
  case variant_index<InitType, init_type::ConstTimeAsynchronousEvaluations>():
    return std::format("ConstTimeAsynchronousEvaluations({})",
                       std::get<init_type::ConstTimeAsynchronousEvaluations>(init_type).n);
  case variant_index<InitType, init_type::Simulated>():
    return "Simulated";
  }
}

template <class _Tp, class _Container, class _Compare>
class priority_queue : public std::priority_queue<_Tp, _Container, _Compare> {
public:
  _Container::const_iterator cbegin() const noexcept { return this->c.cbegin(); }
  _Container::const_iterator cend() const noexcept { return this->c.cend(); }
};

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
  priority_queue<Genome, std::vector<Genome>, std::greater<Genome>> pending_evaluation;

  /// Population of solutions sorted from highest fitness (best) to lowest fitness (worst).
  std::vector<Genome> population;

  /// RNG Engine
  Rng generator;

  /// Fitness landscape function.
  FitnessFunction<N> *fitness_value_fn;

  /// Function used to compute the amount of simulated time a genome will take to evaluate.
  FitnessFunction<N> *time_value_fn;

  /// The number of simulator worker processors -- this is the max size of the `pending_evaluation` queue.
  uint32_t np;

  /// How many steps have occurred thus far in the simulation.
  int64_t step_count = 0;

  /// Number of steps which constitute a single run of the simulation.
  int64_t max_steps;

  /// Whether to employ Selection WhilE EvaluaTing (i.e. SWEET)
  bool use_sweet = false;

  bool sort_by_fitness(const Genome &left, const Genome &right) const { return *left.fitness < *right.fitness; }

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

  void try_insert(Genome &element) {
    element.clear_fitness();
    element.set_fitness(*fitness_value_fn);

    size_t index = std::uniform_int_distribution<size_t>(0, population.size() - 1)(generator);
    Genome &insertion_element = population[index];

    // If by the fitness ordering the insertion genome is less than the new genome, remove it. Then, in sorted order
    // insert the new genome. i.e. tournamant selection.
    if (!this->sort_by_fitness(insertion_element, element)) {
      population.erase(population.begin() + index);
      auto it = std::upper_bound(population.begin(), population.end(), element, FitnessComparator(*this));
      population.emplace(it, element);
    }
  }

  Genome breed() {
    size_t n_genomes = population.size() + (use_sweet ? pending_evaluation.size() : 0);

    // Randomly select a parent genome uniformly.
    size_t parent0_index = std::uniform_int_distribution<size_t>(0, n_genomes - 1)(generator);
    const Genome &parent0 = parent0_index < population.size() ? population[parent0_index]
                                                              : population[parent0_index - pending_evaluation.size()];

    size_t parent1_index;
    do {
      parent1_index = std::uniform_int_distribution<size_t>(0, n_genomes - 1)(generator);
    } while (parent1_index == parent0_index);
    const Genome &parent1 = parent0_index < population.size() ? population[parent1_index]
                                                              : population[parent1_index - pending_evaluation.size()];
    // Breed a child
    Genome child = Crossover()(parent0, parent1, generator);

    // Mutate that child
    child = Mutation()(child, generator);

    return child;
  }

  void reproduce(double current_time) {
    Genome child = breed();

    // Set that child's finish time to be equal to the parents finish time added to its own evaluation time.
    // spdlog::info("Time value = {}", time_value_fn(child));
    child.set_finish_time(current_time + (*time_value_fn)(child.x));

    auto [min, max] = fitness_value_fn->domain();
    child.bound(min, max);

    // Add to execution queue.
    pending_evaluation.emplace(std::move(child));
  }

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
    try_insert(element);

    reproduce(element.get_finish_time());

    step_count += 1;
  }

public:
  /// Creates a new simulation with the supplied initial population. The size of this initial population will be
  /// considered the target size of the population during the simulation.
  ///
  /// The simulation will create `np` mutants and add them to the pending evaluation queue.
  Simulation(std::vector<Genome> &population, uint32_t np, int64_t max_steps, bool use_sweet,
             FitnessFunction<N> *fitness_value_fn, FitnessFunction<N> *time_value_fn)
      : population(std::move(population)), generator(make_rng()), np(np), max_steps(max_steps),
        fitness_value_fn(fitness_value_fn), time_value_fn(time_value_fn), use_sweet(use_sweet) {
    assert(population.size() > 0);
    assert(max_steps > 0);

    initialize_simulation();
  }

  Simulation(uint32_t pop_size, uint32_t np, int64_t max_steps, bool use_sweet, FitnessFunction<N> *fitness_value_fn,
             FitnessFunction<N> *time_value_fn, std::function<Genome(Rng &, FitnessFunction<N> &)> factory,
             InitType init_type = init_type::SynchronousGenerations{0})
      : generator(make_rng()), np(np), max_steps(max_steps), fitness_value_fn(fitness_value_fn),
        time_value_fn(time_value_fn), use_sweet(use_sweet) {

    switch (init_type.index()) {
    case variant_index<InitType, init_type::SynchronousGenerations>(): {
      int ngens = std::get<init_type::SynchronousGenerations>(init_type).n;

      for (int i = 0; i < pop_size; i++) {
        population.push_back(factory(generator, *time_value_fn));
        population[i].clear_fitness();
        population[i].set_fitness(*fitness_value_fn);
      }

      for (int i = 0; i < ngens; i++) {
        std::vector<Genome> new_pop;
        for (int j = 0; j < population.size(); j++)
          new_pop.push_back(breed());

        for (int j = 0; j < population.size(); j++)
          try_insert(new_pop[j]);
      }

      step_count += population.size() * ngens;

      break;
    }

    case variant_index<InitType, init_type::Simulated>():
    case variant_index<InitType, init_type::ConstTimeAsynchronousEvaluations>(): {
      int nevals = 0;
      Flat<N> flat_time_fn{};

      if (auto i = std::get_if<init_type::ConstTimeAsynchronousEvaluations>(&init_type)) {
        nevals = i->n;
        this->time_value_fn = &flat_time_fn;
      }

      while (population.size() < pop_size) {
        while (pending_evaluation.size() < np) {
          pending_evaluation.push(factory(generator, *time_value_fn));
        }

        Genome genome = pending_evaluation.top();
        pending_evaluation.pop();
        genome.set_fitness(*fitness_value_fn);

        auto it = std::upper_bound(population.begin(), population.end(), genome, FitnessComparator(*this));
        population.emplace(it, std::move(genome));
      }

      if (init_type.index() != variant_index<InitType, init_type::ConstTimeAsynchronousEvaluations>())
        break;

      for (int i = 0; i < nevals; i++)
        step();

      // Re-create pending_evaluation w/ the user supplied time_value_fn. The order that exists as of now is incorrect.
      std::vector<Genome> genomes(pending_evaluation.cbegin(), pending_evaluation.cend());
      while (pending_evaluation.size())
        pending_evaluation.pop();

      for (Genome &genome : genomes) {
        genome.set_finish_time(time_value_fn->operator()(genome.x));
        pending_evaluation.push(genome);
      }
    }
    }

    // This value may have been overwritten in the switch above, set it to the supplied value.
    this->time_value_fn = time_value_fn;
    initialize_simulation();
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

  bool converged_to_global_best() const { return fitness_value_fn->converged(population.at(0).x); }

  Genome &get_best_genome() { return population[0]; }
};

}; // namespace evosim
