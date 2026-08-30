module;
#include <array>
#include <compare>
#include <concepts>
#include <format>
#include <limits>
#include <optional>
#include <random>
#include <string>

export module evosim:genome_base;
import :core;
import :fitness;

export namespace evosim {

template <typename T>
concept ProbabilityDistribution = requires(T dist, std::minstd_rand0& rng) {
  typename T::result_type;
  { dist(rng) } -> std::convertible_to<typename T::result_type>;
};

template <unsigned N>
struct Genome {
  std::array<double, N> x;
  double time_finished;
  std::optional<double> fitness = std::nullopt;

  template <typename RNG, ProbabilityDistribution Dist>
  Genome(RNG& rng, const FitnessFunction<N>& time_value_fn, Dist d) {
    for (int i = 0; i < N; i++)
      x[i] = d(rng);

    time_finished = time_value_fn(x);
  }

 protected:
  Genome(
      const std::array<double, N>& x,
      const double time_finished,
      const std::optional<double> fitness)
      : x(x), time_finished(time_finished), fitness(fitness) {}

 public:
  Genome() = default;
  Genome(const Genome<N>& other) = default;
  Genome(Genome<N>&& other) noexcept = default;

  Genome<N>& operator=(const Genome<N>& other) = default;
  Genome<N>& operator=(Genome<N>&& other) noexcept = default;

  virtual ~Genome() = default;

  void clear_fitness() {
    fitness = std::nullopt;
  }

  double set_fitness(FitnessFunction<N>& fitness_function) {
    if (!fitness.has_value())
      fitness = fitness_function(x);

    return *fitness;
  }

  void set_finish_time(double t) {
    time_finished = t;
  }

  double get_finish_time() const {
    return time_finished;
  }

  void bound(double min, double max) {
    for (size_t i = 0; i < N; i++)
      x[i] = std::min(max, std::max(min, x[i]));
  }

  /// Define partial ordering for these types over timed_finish
  std::partial_ordering operator<=>(const Genome& other) const {
    return this->time_finished <=> other.time_finished;
  }

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

} // namespace evosim
