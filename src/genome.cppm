module;
#include <array>
#include <compare>
#include <concepts>
#include <format>
#include <random>
#include <string>

export module evosim:genome;
import :util;
import :function;

export namespace evosim {

template <unsigned N> struct Genome {
  double time_finished;
  std::array<double, N> x;

  Genome(Rng &rng, const FitnessFunction<N> &time_value_fn) {
    for (int i = 0; i < N; i++)
      x[i] = std::uniform_real_distribution<double>(-0.01, 0.01)(rng);

    time_finished = time_value_fn(x);
  }

  virtual ~Genome() {}

  void set_finish_time(double t) { time_finished = t; }
  double get_finish_time() const { return time_finished; }

  void bound(double min, double max) {
    for (size_t i = 0; i < N; i++)
      x[i] = std::min(max, std::max(min, x[i]));
  }

  /// Define partial ordering for these types over timed_finish
  std::partial_ordering operator<=>(const Genome &other) const { return this->time_finished <=> other.time_finished; }

  std::string to_string() const {
    std::string s("{ ");

    for (double d : x) {
      s += std::format("{}, ", d);
    }

    s.pop_back();
    s.pop_back();
    s += " }";

    return s;
  }
};

/// Template class for an ordering of TimedEval objects. This exists as a template to avoid dynamic dispatch when such
/// objects are compared.
///
/// This is meant for ordering the time in a max-heap, but we want to obtain the smallest values out of the max heap,
/// so we invert the comparison.
template <typename T, unsigned N>
  requires std::derived_from<T, Genome<N>>
struct TimedEvalOrd {
  bool operator()(const T &left, const T &right) const { return !(left <= right); }
};

/// Template abstract class for a Mutation over Genome<N>
template <typename G, unsigned N>
  requires std::derived_from<G, Genome<N>>
struct Mutation {
  virtual G operator()(const G &parent, Rng &generator) const = 0;
};

/// Template abstract class for a Crossover operation over Genome<N>.
template <typename G, unsigned N>
  requires std::derived_from<G, Genome<N>> && (N > 3)
class Crossover {
  virtual G operator()(const G &p0, const G &p1, Rng &generator) const = 0;
};

/// Gaussian mutation operation over Genome<N>.
///
/// There is a 1 / N probability for each weight to be modified by a number drawn from N(0.0, 0.5).
template <typename G, unsigned N>
  requires std::derived_from<G, Genome<N>>
struct GaussianMutation : Mutation<G, N> {
  G operator()(const G &parent, Rng &generator) const override {
    const double p = 1.0 / (double)N;
    std::normal_distribution<double> dist(0.0, 0.5);

    G child(parent);

    // Modify each weight w/ a gaussian generated value at probability 1 / N = p
    for (int i = 0; i < N; i++)
      if (std::generate_canonical<float, 32>(generator) < p) {
        child.x[i] += dist(generator);
        child.x[i] = std::max(std::min(child.x[i], 10.0), -10.0);
      }

    return child;
  }
};

template <typename G, unsigned N>
  requires std::derived_from<G, Genome<N>>
struct TwoPointCrossover : Crossover<G, N> {
  G operator()(const G &p0, const G &p1, Rng &generator) const override {

    G child(p0);

    auto start_dist = std::uniform_int_distribution<size_t>(0, N - 3);
    size_t start = start_dist(generator);
    auto end_dist = std::uniform_int_distribution<size_t>(start, N - 1);
    size_t end = end_dist(generator);

    std::copy(p1.x.begin() + start, p1.x.begin() + end, child.x.begin() + start);

    return child;
  }
};

template <typename G, unsigned N>
  requires std::derived_from<G, Genome<N>>
struct NopCrossover : Crossover<G, N> {
  G operator()(const G &p0, const G &p1, Rng &generator) const override {
    G child(p0);
    return child;
  }
};

///
template <unsigned N> struct GenomeConfig {
  const std::string name;

  typedef Genome<N> Genome;
  typedef Mutation<Genome, N> Mutation;
  typedef Crossover<Genome, N> Crossover;
};

template <unsigned N> struct SDBGenomeConfig : public GenomeConfig<N> {
  const std::string name = "Gaussian Mutation; No-Op Crossover";

  typedef Genome<N> Genome;
  typedef GaussianMutation<Genome, N> Mutation;
  typedef NopCrossover<Genome, N> Crossover;
};

}; // namespace evosim
