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
const int POP_SIZE = 10;
const int NUMBER_SIMULATED_PROCESSORS = 10;
const int NUMBER_GENOMES = 1000;

const int NUMBER_THREADS = 10;
const int NUMBER_RUNS = 1'000'000;

template <class GC>
void modulation_experiment(double A, double B, std::string logger_name) {
  auto csv_logger = spdlog::basic_logger_st(
      logger_name, "results/" + logger_name + ".csv", true);
  csv_logger->set_pattern("%v");
  csv_logger->info("Variance,Converged Percentage,95% CI");
  auto fitness = std::make_unique<ScottDeJongBasins<N>>(A, B);
  std::vector<std::unique_ptr<FitnessFunction<N>>> time_functions;

  // time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new
  // Flat<N>()));
  // time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new
  // ScottDeJongBasins<N>(A, B)));
  time_functions.emplace_back(
      std::unique_ptr<FitnessFunction<N>>(new ScottDeJongBasins<N>(A, -B)));

  SimulationConfig sc{
      POP_SIZE, NUMBER_SIMULATED_PROCESSORS, NUMBER_GENOMES, init_policy::Simulated{}, false};

  // Genomes are initialized at 0^N at the start of the search.

  for (int i = 0; i < time_functions.size(); i++) {
    for (int u = 0; u < 15; u++) {
      auto factory = [=](Rng& rng, auto& time_value) {
        return Genome<N>(
            rng,
            time_value,
            std::normal_distribution<double>{0.0, std::sqrt((double)u)});
      };
      FitnessFunction<N>* time = time_functions[i].get();
      ExpConfig<N> fc{NUMBER_THREADS, NUMBER_RUNS, *fitness, *time, factory};
      spdlog::info(
          "Initialization Distribution: N({}, {:.4f})",
          0.0,
          std::sqrt((double)u));
      auto [p, ci] = run_experiment<GC, N>(fc, sc, *csv_logger);
      csv_logger->info("{:.4f},{:.4f},{:.4f}", (double)u, p, ci);
    }
  }
}

/// This binary runs a series of experiments that represent a superset of the
/// experiments run in "Evaluation-Time Bias in Asynchronous Evolutionary
/// Algorithms" by Eric O. Scott and Kenneth A. De Jong.
///
/// There are some additional figures / visualizations in the paper that are not
/// produced by this script.
int main(int argc, char** argv) {
  initialize_logger();

  // Experiment 1: Converging to Fast Optimum on Two-Basin Objective
  //
  // These experiments will only be run on the Scott-DeJong basin function as it
  // depends on having two identical local minima.
  //
  // Groups:
  // - Constant-Eval Time
  // - Eval Time = Fitness
  // - A fast, B slow.

  spdlog::info("-------------------------------------------");
  spdlog::info("Modulation Experiments: A = B with Crossover");
  spdlog::info("-------------------------------------------");
  modulation_experiment<SDBGenomeConfig<N>>(10, 10, "modulate_a_eq_b_co");

  spdlog::info("-------------------------------------------");
  spdlog::info("Modulation Experiments: A = B with Crossover");
  spdlog::info("-------------------------------------------");
  modulation_experiment<SDBGenomeNoCOConfig<N>>(
      10, 10, "modulate_a_eq_b_no_co");

  spdlog::info("-------------------------------------------");
  spdlog::info("Modulation Experiments: 1.5A = B with Crossover");
  spdlog::info("-------------------------------------------");
  modulation_experiment<SDBGenomeConfig<N>>(10, 15, "modulate_a_lt_b_co");

  spdlog::info("-------------------------------------------");
  spdlog::info("Modulation Experiments: 1.5A = B with Crossover");
  spdlog::info("-------------------------------------------");
  modulation_experiment<SDBGenomeNoCOConfig<N>>(
      10, 15, "modulate_a_lt_b_no_co");
  return 0;
}
