module;
#include <format>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

export module evosim.main;
import evosim;

using namespace evosim;

const unsigned N = 8;

int main() {
  initialize_logger();
  spdlog::set_level(spdlog::level::info);

  const int POP_SIZE = 10;
  const int NP = 10;
  const int NGENOMES = 100;
  const uint64_t SEED = 42;

  spdlog::info(
      "Parity test - running old Simulation and new Simulation2 with fixed seed {}",
      SEED);

  Flat<N> fitness;
  Flat<N> time_fn;

  struct ZeroTime : Flat<N> {
    double operator()(const std::array<double, N>&) const {
      return 0.0;
    }

    std::string to_string() const {
      return "Zero";
    }
  };

  ZeroTime ls_time;

  auto factory = [](Rng& rng, auto& tf) {
    return Genome<N>(rng, tf, std::normal_distribution<double>{0.0, 1.0});
  };

  {
    Simulation<SDBTraits<N>, N> old_sim(
        POP_SIZE,
        NP,
        NGENOMES,
        false,
        fitness,
        time_fn,
        factory,
        ls_time,
        SEED);
    Simulation<SDBTraits<N>, N> new_sim(
        POP_SIZE,
        NP,
        NGENOMES,
        false,
        fitness,
        time_fn,
        factory,
        ls_time,
        SEED);

    // Compare initial population sizes and sorted fitness
    auto cmp = [](const auto& a, const auto& b) {
      return *a.fitness < *b.fitness;
    };
    auto old_pop = old_sim.get_population_data();
    auto new_pop = new_sim.get_population_data();
    std::sort(old_pop.begin(), old_pop.end(), cmp);
    std::vector<Genome<N>> new_pop_copy = new_pop;
    std::sort(new_pop_copy.begin(), new_pop_copy.end(), cmp);
    bool init_match = old_pop.size() == new_pop_copy.size();
    for (size_t i = 0; i < old_pop.size() && init_match; i++) {
      if (std::abs(*old_pop[i].fitness - *new_pop_copy[i].fitness) > 1e-9)
        init_match = false;
      for (size_t j = 0; j < N; j++)
        if (std::abs(old_pop[i].x[j] - new_pop_copy[i].x[j]) > 1e-9)
          init_match = false;
    }
    if (!init_match) {
      spdlog::error("Initial population parity FAILED");
      return 1;
    }
    spdlog::info("Initial population parity PASSED");

    // Step-by-step compare
    int steps = NGENOMES; // compare first 100 steps
    bool all_match = true;
    for (int s = 0; s < steps; s++) {
      old_sim.step();
      new_sim.step();

      auto old_p = old_sim.get_population_data();
      auto new_p = new_sim.get_population_data();
      std::sort(old_p.begin(), old_p.end(), cmp);
      std::sort(new_p.begin(), new_p.end(), cmp);
      if (old_p.size() != new_p.size()) {
        all_match = false;
        spdlog::error(
            "Step {} size mismatch {} vs {}", s, old_p.size(), new_p.size());
        break;
      }
      for (size_t i = 0; i < old_p.size(); i++) {
        if (std::abs(*old_p[i].fitness - *new_p[i].fitness) > 1e-9) {
          all_match = false;
          spdlog::error(
              "Fitness mismatch at step {} idx {} old {} new {}",
              s,
              i,
              *old_p[i].fitness,
              *new_p[i].fitness);
          break;
        }
        for (size_t j = 0; j < N; j++)
          if (std::abs(old_p[i].x[j] - new_p[i].x[j]) > 1e-6) {
            all_match = false;
            spdlog::error(
                "X mismatch at step {} idx {} dim {} old {} new {}",
                s,
                i,
                j,
                old_p[i].x[j],
                new_p[i].x[j]);
            break;
          }
        if (!all_match)
          break;
      }
      if (!all_match) {
        spdlog::error("Population mismatch at step {}", s);
        // print old population fitnesses
        std::string old_str = "old fitness: ";
        for (auto& g : old_p)
          old_str += std::format("{:.4f} ", *g.fitness);
        spdlog::error("{}", old_str);
        std::string new_str = "new fitness: ";
        for (auto& g : new_p)
          new_str += std::format("{:.4f} ", *g.fitness);
        spdlog::error("{}", new_str);
        break;
      }

      // compare pending queue top time
      double old_top = old_sim.pending_evaluation.top().get_finish_time();
      double new_top = new_sim.pending_top_time();
      if (std::abs(old_top - new_top) > 1e-9) {
        all_match = false;
        spdlog::error(
            "Pending top time mismatch at step {} {} vs {}",
            s,
            old_top,
            new_top);
        break;
      }
    }
    if (all_match) {
      spdlog::info("Parity test PASSED for {} steps", steps);
      return 0;
    } else {
      spdlog::error("Parity test FAILED");
      return 2;
    }
  }
}
