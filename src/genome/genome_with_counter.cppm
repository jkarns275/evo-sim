module;
#include <array>
#include <string>

export module evosim:genome_counter;
import :genome_base;
import :core;
import :fitness;

export namespace evosim {

template <unsigned N>
struct GenomeWithCounter : public Genome<N> {
  unsigned int counter = 0;

  template <typename Dist>
    requires ProbabilityDistribution<Dist>
  GenomeWithCounter(Rng& rng, const FitnessFunction<N>& time_value_fn, Dist d)
      : Genome<N>(rng, time_value_fn, d) {}

  GenomeWithCounter() = default;
  GenomeWithCounter(const GenomeWithCounter<N>& other) = default;
  GenomeWithCounter(GenomeWithCounter<N>&& other) noexcept = default;

  GenomeWithCounter<N>& operator=(const GenomeWithCounter<N>& other) = default;
  GenomeWithCounter<N>& operator=(GenomeWithCounter<N>&& other) noexcept =
      default;

  void inc() {
    counter += 1;
  }

  struct SortByCounter {
    bool operator()(
        const GenomeWithCounter<N>* left,
        const GenomeWithCounter<N>* right) const {
      return left->counter < right->counter;
    }
  };
};

} // namespace evosim
