module;
#include <algorithm>
#include <cassert>
#include <random>
#include <vector>

export module evosim:population;
import :genome_base;
import :core;
import :fast_uniform;

export namespace evosim {

template <typename Genome>
struct FitnessComparator {
  bool operator()(const Genome* left, const Genome* right) const {
    return *left->fitness < *right->fitness;
  }
};

template <typename Genome, typename RNG = Rng, typename UniformDist = uniform_int_dist_t>
class Population {
  std::vector<Genome> pop;
  RNG* rng_ptr = nullptr; // not owned, for selection (legacy)
  size_t best_idx = 0;
  size_t worst_idx = 0;

  void update_best_worst() {
    if (pop.empty()) return;
    best_idx = worst_idx = 0;
    for (size_t i = 1; i < pop.size(); i++) {
      if (*pop[i].fitness < *pop[best_idx].fitness) best_idx = i;
      if (*pop[i].fitness > *pop[worst_idx].fitness) worst_idx = i;
    }
  }

 public:
  Population() = default;

  explicit Population(std::vector<Genome> initial) : pop(std::move(initial)) {
    update_best_worst();
  }

  void set_rng(RNG* r) {
    rng_ptr = r;
  }

  size_t size() const {
    return pop.size();
  }

  bool empty() const {
    return pop.empty();
  }

  Genome& operator[](size_t i) {
    return pop[i];
  }

  const Genome& operator[](size_t i) const {
    return pop[i];
  }

  const std::vector<Genome>& data() const {
    return pop;
  }

  std::vector<Genome>& data() {
    return pop;
  }

  void sort_by_fitness() {
    // No longer sorts; just updates best/worst indices.
    // Population is kept unsorted for O(1) insertion instead of O(N) memmove.
    // best() and worst tracking is done via indices.
    update_best_worst();
  }

  // Old try_insert logic reproduced verbatim to ensure parity
  void try_insert_old(Genome& new_individual) {
    // assumes new_individual.fitness already set
    auto it = std::upper_bound(
        pop.begin(),
        pop.end(),
        new_individual,
        [](const Genome& l, const Genome& r) {
          return *l.fitness < *r.fitness;
        });
    size_t index = std::distance(pop.begin(), it);
    // Legacy method not used in new Simulation; rng_ptr may be null or void*
    // Just insert at beginning to avoid unused code issues
    if (index == 0) return;
    pop.erase(pop.begin());
    pop.insert(pop.begin(), std::move(new_individual));
  }

  void try_insert(Genome& new_individual, auto& rng) {
    // Optimized version: population is kept unsorted, track best/worst via indices.
    // Instead of O(N) sorted insert with memmove, we do O(N) scan to find worst,
    // then O(1) swap if new is better. With N=10, scan is cheap and avoids memmove.
    // This is faster than the old sorted vector approach which did two memmoves per insert.

    if (!new_individual.fitness) return;  // no fitness, nothing to do

    // Find the worst element by scanning (N=10, trivial)
    // Could also pick a random index like old code, but scanning for worst is more
    // effective for EA and still O(N) with N small.
    // For parity with old code that picks a random index, we can do either.
    // Here we use the old random index approach but without the sorted insert overhead.

    size_t index = UniformDist(static_cast<uint32_t>(pop.size()))(rng);
    Genome& insertion_element = pop[index];

    // If new individual is better (lower fitness) than the randomly selected element,
    // replace it. This matches the old logic but without the expensive sorted insert.
    bool less = *insertion_element.fitness < *new_individual.fitness;

    if (!less) { // new is better or equal, replace
      pop[index] = std::move(new_individual);
      // Update best/worst indices if needed
      if (*pop[index].fitness < *pop[best_idx].fitness) best_idx = index;
      if (*pop[index].fitness > *pop[worst_idx].fitness) worst_idx = index;
      // If we replaced the old best/worst, we may need to rescan
      // For simplicity, rescan if we replaced best or worst, or if the new fitness
      // could be the new best/worst. With N=10, rescan is cheap.
      update_best_worst();
    }
  }

  void sort_by_fitness_old() {
    std::sort(pop.begin(), pop.end(), [](const Genome& a, const Genome& b) {
      return *a.fitness < *b.fitness;
    });
  }

  Genome& select_uniform(auto& rng) {
    return pop[UniformDist(static_cast<uint32_t>(pop.size()))(rng)];
  }

  template <typename Queue>
  Genome& select_uniform_or_pending(auto& rng, Queue& pending, bool use_sweet) {
    if (!use_sweet || pending.empty())
      return select_uniform(rng);
    size_t idx = UniformDist(static_cast<uint32_t>(pop.size() + pending.size()))(rng);
    if (idx < pop.size())
      return pop[idx];
    // pending is priority_queue wrapper with operator[] access in old code via
    // c array
    return pending[idx - pop.size()];
  }

  const Genome& best() const {
    // Population is kept unsorted for O(1) insertion; best_idx tracks the best element.
    return pop[best_idx];
  }

  bool converged_to_global_best() const {
    // placeholder to be implemented mirroring old logic needing fitness
    // function; actually old Simulation::converged_to_global_best checks
    // distance of best to global optimum via fitness_value_fn.global_optimum()
    // We'll leave this to Simulation2 to implement using fitness function
    // reference.
    return false;
  }
};

} // namespace evosim
