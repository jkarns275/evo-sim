module;

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>

#include <random>

export module evosim.main;

import evosim;

using namespace evosim;

const unsigned N = 10;

// Zero time model for NoOp local search baseline - returns 0 time cost
template <unsigned M>
struct ZeroTime : public Flat<M> {
  double operator()(const std::array<double, M>&) const {
    return 0.0;
  }

  std::string to_string() const {
    return "Zero";
  }

  std::array<double, M> gradient(const std::array<double, M>&) const {
    return array_of<M>(0.0);
  }
};

template <unsigned M>
struct ZeroTime2 : public Flat<M> {
  double operator()(const std::array<double, M>&) const {
    return 0.0;
  }

  std::string to_string() const {
    return "Zero";
  }

  std::array<double, M> gradient(
      const std::array<double, M>& x) const {
    return array_of<M>(0.0);
  }
};

int main(int argc, char** argv) {
  initialize_logger();

  // Memetic experiments: compare NoOp local search baseline vs Backpropagation
  // local search across standard fitness landscapes, modeling LS time as
  // 2*t(x).

  const int POP_SIZE = 10;
  const int NUMBER_SIMULATED_PROCESSORS = 10;
  const int NUMBER_GENOMES = 1000;

  spdlog::info("-------------------------------------------");
  spdlog::info("Memetic Experiments: NoOp vs Backpropagation Local Search");
  spdlog::info("-------------------------------------------");

  // Experiment 1: Flat fitness with Flat time, compare NoOp vs BP LS
  // Expect similar behavior since Flat has zero gradient, but time cost differs
  // 1*t vs 3*t with BP (2*t LS + 1*t eval)
  {
    spdlog::info("Experiment 1: Flat fitness, Flat time, NoOp vs BP");

    auto csv_logger = spdlog::basic_logger_st(
        "memetic_flat", "results/memetic_flat.csv", true);
    csv_logger->set_pattern("%v");
    csv_logger->info("policy,run,converged,best_fitness,total_time");

    auto factory = [](Rng& rng, auto& time_fn) {
      return Genome<N>(
          rng, time_fn, std::normal_distribution<double>{0.0, 1.0});
    };

    Flat<N> fitness;
    Flat<N> time_fn;
    ZeroTime<N> zero_time;

    // NoOp baseline using SDBTraits (NoOpLocalSearch) with zero LS time
    {
// Use run_experiment with old Simulation path for baseline to compare
      // against existing literature baseline, then use Simulation2 for memetic
      // path below via manual loop for now until run_experiment migrated fully
      // to support ls_time parameter generically. For now manual loop for both
      // to ensure same code path structure using Simulation2 with different
      // traits.

      for (int run = 0; run < 1000; run++) {
        Rng rng{static_cast<uint64_t>(run)};
        std::vector<Genome<N>> init_pop;
        init_pop.reserve(POP_SIZE);
        for (int i = 0; i < POP_SIZE; i++) {
          Genome<N> g = factory(rng, time_fn);
          g.clear_fitness();
          g.set_fitness(fitness);
          init_pop.push_back(std::move(g));
        }
        Simulation<SDBTraits<N>, N> s(
            init_pop,
            NUMBER_SIMULATED_PROCESSORS,
            NUMBER_GENOMES,
            false,
            fitness,
            time_fn,
            zero_time,
            run); // seed for deterministic-ish per run variation; using run as
                  // seed for reproducibility
        s.run();
        bool conv = s.converged_to_global_best();
        double best_fit = s.get_best_genome().fitness.value_or(-1);
        // Approximate total time as step_count * avg time? For simplicity use
        // step_count as proxy; real total time would need tracking last finish
        // time, which Simulation currently does not expose directly easily.
        // We'll use step_count for now as placeholder and improve later.
        csv_logger->info(
            "NoOp,{},{},{},{}", run, conv ? 1 : 0, best_fit, s.step_count);
        if (run % 100 == 0)
          spdlog::info("NoOp run {} done", run);
      }
    }

    // Backpropagation LS using SDBTraitsBP with BackpropagationTime wrapper for
    // ls_time = 2*t
    {
      // Define BackpropagationTime wrapper around Flat time model for LS time
      // =2*t
      BackpropagationTimeStatic<Flat, N> bp_time;

      for (int run = 0; run < 1000; run++) {
        Rng rng{static_cast<uint64_t>(run + 10000)};
        std::vector<Genome<N>> init_pop;
        init_pop.reserve(POP_SIZE);
        for (int i = 0; i < POP_SIZE; i++) {
          Genome<N> g = factory(rng, time_fn);
          g.clear_fitness();
          g.set_fitness(fitness);
          init_pop.push_back(std::move(g));
        }
        Simulation<SDBTraitsBP<N>, N, Flat<N>, Flat<N>, BackpropagationTimeStatic<Flat, N>> s(
            init_pop,
            NUMBER_SIMULATED_PROCESSORS,
            NUMBER_GENOMES,
            false,
            fitness,
            time_fn,
            bp_time,
            run + 10000); // different seed offset to avoid overlap but still
                          // deterministic
        s.run();
        bool conv = s.converged_to_global_best();
        double best_fit = s.get_best_genome().fitness.value_or(-1);
        csv_logger->info(
            "Backprop,{},{},{},{}", run, conv ? 1 : 0, best_fit, s.step_count);
        if (run % 100 == 0)
          spdlog::info("Backprop run {} done", run);
      }
    }
  }

  // Experiment 2: Scott-DeJong basins with asymmetric A B values, compare NoOp
  // vs BP across time bias conditions similar to reproduction experiments but
  // adding memetic dimension.
  {
    spdlog::info("Experiment 2: Scott-DeJong basins memetic comparison");
    const int POP_SIZE2 = 10;
    const int NP2 = 10;
    const int NGEN2 = 1000;

    auto csv_logger =
        spdlog::basic_logger_st("memetic_sdb", "results/memetic_sdb.csv", true);
    csv_logger->set_pattern("%v");
    csv_logger->info("policy,A,B,time_model,run,converged,best_x0,best_x1");

    std::vector<std::pair<double, double>> ab_values = {
        {10, 10}, {10, 15}, {15, 10}};

    for (auto [A, B] : ab_values) {
      // NoOp baseline
      {
        auto factory = [=](Rng& rng, auto& time_fn) {
          return Genome<2>(
              rng, time_fn, std::normal_distribution<double>{0.0, 1.0});
        };
        ScottDeJongBasins<2> fitness(A, B);
        Flat<2> time_model;
        ZeroTime2<2> zero_ls;

        for (int run = 0; run < 1000; run++) {
          Rng rng{static_cast<uint64_t>(run)};
          std::vector<Genome<2>> init_pop;
          init_pop.reserve(POP_SIZE2);
          for (int i = 0; i < POP_SIZE2; i++) {
            Genome<2> g = factory(rng, time_model);
            g.clear_fitness();
            g.set_fitness(fitness);
            init_pop.push_back(std::move(g));
          }
          Simulation<SDBTraits<2>, 2, ScottDeJongBasins<2>, Flat<2>> s(
              init_pop,
              NP2,
              NGEN2,
              false,
              fitness,
              time_model,
              zero_ls,
              run);
          s.run();
          auto best = s.get_best_genome();
          bool conv = fitness.converged(best.x);
          csv_logger->info(
              "NoOp,{},{},Flat,{},{},{:.4f},{:.4f}",
              A,
              B,
              run,
              conv ? 1 : 0,
              best.x[0],
              best.x[1]);
        }
      }
      // Backpropagation LS with time = 2*t and with fitness-biased time model
      // as well to explore interaction with evaluation-time bias mitigation via
      // memetic local search
      {
        auto factory = [=](Rng& rng, auto& time_fn) {
          return Genome<2>(
              rng, time_fn, std::normal_distribution<double>{0.0, 1.0});
        };
        ScottDeJongBasins<2> fitness(A, B);
        // Use fitness as time model to create evaluation-time bias condition,
        // then BP LS time =2*t will amplify effect, allowing us to test whether
        // memetic local search mitigates ETB.
        ScottDeJongBasins<2> time_model(A, B);
        // For now use Flat base for simplicity to demonstrate structure; TODO:
        // extend BackpropagationTimeStatic to accept runtime base instance
        // parameters via factory or via traits.
        Flat<2> flat_for_bp;
        BackpropagationTimeStatic<Flat, 2> bp_time;

        for (int run = 0; run < 1000; run++) {
          Rng rng{static_cast<uint64_t>(run + 20000)};
          std::vector<Genome<2>> init_pop;
          init_pop.reserve(POP_SIZE2);
          for (int i = 0; i < POP_SIZE2; i++) {
            Genome<2> g = factory(rng, time_model);
            g.clear_fitness();
            g.set_fitness(fitness);
            init_pop.push_back(std::move(g));
          }
          Simulation<SDBTraitsBP<2>, 2, ScottDeJongBasins<2>, ScottDeJongBasins<2>, BackpropagationTimeStatic<Flat, 2>> s(
              init_pop,
              NP2,
              NGEN2,
              false,
              fitness,
              time_model,
              bp_time,
              run + 20000);
          s.run();
          auto best = s.get_best_genome();
          bool conv = fitness.converged(best.x);
          csv_logger->info(
              "Backprop,{},{},FitnessBias,{},{},{:.4f},{:.4f}",
              A,
              B,
              run,
              conv ? 1 : 0,
              best.x[0],
              best.x[1]);
        }
      }
    }
  }

  spdlog::info(
      "Memetic experiments complete - results in results/memetic_*.csv");
  return 0;
}
