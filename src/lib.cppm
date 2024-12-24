module;
#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <cmath>
#include <concepts>
#include <functional>
#include <numbers>
#include <queue>
#include <random>
#include <stdint.h>
#include <type_traits>
#include <vector>

export module evosim;

export namespace evosim {

typedef std::mt19937_64 Rng;

class TimedEval {
  double time_finished;

public:
  TimedEval() : time_finished(0.0) {}

  void set_finish_time(double t) { time_finished = t; }
  double get_finish_time() const { return time_finished; }

  /// Obtains the time this object will take to evaluate. By default, 1 "second".
  virtual double get_eval_time() { return 1.0; }

  /// Define partial ordering for these types over timed_finish
  std::partial_ordering operator<=>(const TimedEval &other) const {
    return this->time_finished <=> other.time_finished;
  }
};

/// Template class for an ordering of TimedEval objects. This exists as a template to avoid dynamic dispatch when such
/// objects are compared.
///
/// This is meant for ordering the time in a max-heap, but we want to obtain the smallest values out of the max heap,
/// so we invert the comparison.
template <typename T>
  requires std::derived_from<T, TimedEval>
struct TimedEvalOrd {
  bool operator()(const T &left, const T &right) const { return !(left <= right); }
};

template <typename T> class Mutator {
  virtual T operator()(const T &parent, Rng &generator) const = 0;
};

template <double mean, double std> constexpr double gaussian_pdf(double x) {
  const double denom = std::sqrt(2.0 * std::numbers::pi * std * std);
  const double exponent = ((x - mean) * (x - mean)) / (2.0 * std * std);
  return (1.0 / denom) * std::pow(std::numbers::e, -exponent);
}

template <unsigned int D> struct SphericalGaussianMixtureFitness {
  double radius;
  std::array<double, D> global_min_position;
  std::array<double, D> local_min_position;

  SphericalGaussianMixtureFitness(double radius, std::array<double, D> global_min_position,
                                  std::array<double, D> local_min_position)
      : radius(radius), global_min_position(global_min_position), local_min_position(local_min_position) {}
};

template <unsigned int N> struct ScottDeJongBasins : public TimedEval {
  double A, B;
  std::array<double, N> x;
  double sigma;

  ScottDeJongBasins(double A, double B, Rng &rng, double sigma = 2.5) : A(A), B(B), sigma(sigma) {
    for (size_t i = 0; i < N; i++)
      x[i] = std::uniform_real_distribution<double>(-1.0, 1.0)(rng);
  }

  ScottDeJongBasins(double A, double B, std::array<double, N> x, double sigma = 2.5) : A(A), B(B), x(x), sigma(sigma) {}

  std::string to_string() const {
    std::string sb;

    for (int i = 0; i < 10; i++) {
      sb += std::format("{}, ", x[0]);
    }

    sb.pop_back();
    sb.pop_back();

    return std::format("{{ {} }}", sb);
  }

  double distance(const ScottDeJongBasins<N> &other) const {
    assert(this->A == other.A);
    assert(this->B == other.B);
    double square_sum = 0.0;

    for (size_t i = 0; i < N; i++) {
      double diff = x[i] - other.x[i];
      square_sum += diff * diff;
    }

    return std::sqrt(square_sum);
  }

  double operator()() const {
    double a_sq_sum = 0.0;
    double b_sq_sum = 0.0;
    for (size_t i = 0; i < N; i++) {
      double a_diff = x[i] - 2 * sigma;
      a_sq_sum += a_diff * a_diff;

      double b_diff = x[i] + 2 * sigma;
      b_sq_sum += b_diff * b_diff;
    }

    return std::max(A, B) - A * std::exp(-1.0 / (2.0 * sigma * sigma) * a_sq_sum) -
           B * std::exp(-1.0 / (2.0 * sigma * sigma) * b_sq_sum);
  }

  /// Ordering of ScottDeJongBasins<N, A, B> based upon fitness, where lower fitness is better.
  struct Ord {
    bool operator()(const ScottDeJongBasins<N> &left, const ScottDeJongBasins<N> &right) { return left() < right(); }
  };

  /// Mutation for ScottDeJongBasins<N, A, B>. Randomly mutates one element of the genome
  template <double Scale> struct Mutation : Mutator<ScottDeJongBasins<N>> {
    ScottDeJongBasins<N> operator()(const ScottDeJongBasins<N> &parent, Rng &engine) const override {
      unsigned int i = std::uniform_int_distribution<unsigned int>(0, N - 1)(engine);
      double offset = std::uniform_real_distribution<double>(-Scale, Scale)(engine);

      ScottDeJongBasins<N> child(parent);
      child.x[i] += offset;

      return child;
    }
  };

  static ScottDeJongBasins<N> positive_basin(double A, double B, double sigma = 2.5) {
    return ScottDeJongBasins(A, B, {sigma * 2});
  }
  static ScottDeJongBasins<N> negative_basin(double A, double B, double sigma = 2.5) {
    return ScottDeJongBasins(A, B, {sigma * -2});
  }
};

/// Evolutionary algorithm simulation for some quasi-genome Ty.
///
/// The simulation will maintain a population of genomes, reproduction will occur using tournamant selection with
/// ordering FOrd.
template <typename T, typename M, typename FOrd, typename TOrd = TimedEvalOrd<T>>
  requires std::derived_from<T, TimedEval> && std::derived_from<M, Mutator<T>> && (!std::is_abstract_v<M>)
class Simulation {

  /// Genomes pending evaluation. Must be ordered such that the smallest value is considered the largest as this is a
  /// max heap and we want to pull the most recently evaluated genome.
  std::priority_queue<T, std::vector<T>, TOrd> pending_evaluation;

  /// Population of solutions sorted from highest fitness (best) to lowest fitness (worst).
  std::vector<T> population;

  /// RNG Engine
  Rng rng;

  /// The number of simulator worker processors -- this is the max size of the `pending_evaluation` queue.
  uint32_t np;

  /// How many steps have occurred thus far in the simulation.
  int64_t step_count = 0;

  /// Number of steps which constitute a single run of the simulation.
  int64_t max_steps;

  /// Uniformly selects a random member of the population and returns a reference to it.
  const T &select() {
    assert(population.size() > 0);
    size_t index = std::uniform_int_distribution<size_t>(0, population.size() - 1)(rng);
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
    T element = std::move(pending_evaluation.top());
    pending_evaluation.pop();

    // Tournamant selection at index.
    size_t index = std::uniform_int_distribution<size_t>(0, population.size() - 1)(rng);
    const T &insertion_element = population[index];

    // If by the fitness ordering the insertion genome is less than the new genome, remove it. Then, in sorted order
    // insert the new genome. i.e. tournamant selection.
    if (!FOrd()(insertion_element, element)) {
      population.erase(population.begin() + index);
      auto it = std::upper_bound(population.begin(), population.end(), element, FOrd());
      population.emplace(it, std::move(element));
    }

    // Randomly select a parent genome uniformly.
    size_t parent_index = std::uniform_int_distribution<size_t>(0, population.size() - 1)(rng);
    const T &parent = population[parent_index];

    // Create a child with mutation
    T child = M()(parent, rng);

    // Set that child's finish time to be equal to the parents finish time added to its own evaluation time.
    child.set_finish_time(element.get_finish_time() + child.get_eval_time());

    // Add to execution queue.
    pending_evaluation.emplace(std::move(child));

    step_count += 1;
  }

public:
  /// Creates a new simulation with the supplied initial population. The size of this initial population will be
  /// considered the target size of the population during the simulation.
  ///
  /// The simulation will create `np` mutants and add them to the pending evaluation queue.
  Simulation(std::vector<T> &population, uint32_t np, int64_t max_steps)
      : population(std::move(population)), np(np), max_steps(max_steps) {
    assert(population.size() > 0);
    assert(max_steps > 0);

    initialize_rng();
    initialize_simulation();
  }

  Simulation(uint32_t pop_size, uint32_t np, int64_t max_steps, std::function<T(Rng &)> factory)
      : np(np), max_steps(max_steps) {
    initialize_rng();

    for (int i = 0; i < pop_size; i++)
      population.push_back(factory(rng));

    initialize_simulation();
  }

  void initialize_rng() {
    std::random_device rd;
    rng.seed(rd());

    for (int i = 0; i < 1000; i++)
      std::generate_canonical<double, 52>(rng);
  }

  void initialize_simulation() {
    std::sort(population.begin(), population.end(), FOrd());

    for (uint32_t i = 0; i < np; i++)
      pending_evaluation.push(M()(select(), rng));
  }

  /// Runs the simulation for `max_steps`. This resets `step_count`, meaning it can be called multiple times
  void run() {
    step_count = 0;
    while (!done()) {
      step();
    }
  }

  T &get_best_genome() { return population[0]; }
};

} // namespace evosim
