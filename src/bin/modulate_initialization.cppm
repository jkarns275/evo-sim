module;

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>

#include <random>

export module evosim.main;

import evosim;

using namespace evosim;

const unsigned N = 8;

/// This binary runs a series of experiments that represent a superset of the experiments run in "Evaluation-Time Bias
/// in Asynchronous Evolutionary Algorithms" by Eric O. Scott and Kenneth A. De Jong.
///
/// There are some additional figures / visualizations in the paper that are not produced by this script.
int main(int argc, char **argv) {

  initialize_logger();

  auto csv_logger = spdlog::basic_logger_st("experiment_output_csv", "results.csv", true);
  csv_logger->set_pattern("%v");
  csv_logger->info("Experiment Name, Converged Percentage, 95% CI, NSim, NGenomes");

  // Experiment 1: Converging to Fast Optimum on Two-Basin Objective
  //
  // These experiments will only be run on the Scott-DeJong basin function as it depends on having two identical local
  // minima.
  //
  // Groups:
  // - Constant-Eval Time
  // - Eval Time = Fitness
  // - A fast, B slow.
  const int POP_SIZE = 10;
  const int NUMBER_SIMULATED_PROCESSORS = 10;
  const int NUMBER_GENOMES = 1000;

  const int NUMBER_THREADS = 10;
  const int NUMBER_RUNS = 1'000'00;

  spdlog::info("-------------------------------------------");
  spdlog::info("Modulation Experiments: A = B");
  spdlog::info("-------------------------------------------");

  {
    const double A = 10.0;
    const double B = A;

    const InitType INIT_TYPE = init_type::Simulated{};

    auto fitness = std::make_unique<ScottDeJongBasins<N>>(A, B);
    std::vector<std::unique_ptr<FitnessFunction<N>>> time_functions;

    // time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new Flat<N>()));
    // time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new ScottDeJongBasins<N>(A, B)));
    time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new ScottDeJongBasins<N>(A, -B)));

    SimulationConfig sc{POP_SIZE, NUMBER_SIMULATED_PROCESSORS, NUMBER_GENOMES, INIT_TYPE, false};

    // Genomes are initialized at 0^N at the start of the search.

    for (int i = 0; i < time_functions.size(); i++) {
      for (int u = 0; u < 11; u++) {
        auto factory = [=](Rng &rng, FitnessFunction<N> &time_value) {
          return Genome(rng, time_value, std::uniform_real_distribution<double>{(double)-u, (double)u});
        };
        FitnessFunction<N> *time = time_functions[i].get();
        ExpConfig<N> fc{NUMBER_THREADS, NUMBER_RUNS, *fitness, *time, factory};
        spdlog::info("Range: [-{}, {}]", u * 1.0, u * 1.0);
        run_experiment<SDBGenomeConfig<N>, N>(fc, sc, *csv_logger);
      }
    }
  }

  // spdlog::info("-------------------------------------------");
  // spdlog::info("Modulation Experiments: B = 1.5A");
  // spdlog::info("-------------------------------------------");

  // {
  //   const double A = 10.0;
  //   const double B = 1.5 * A;

  //   const InitType INIT_TYPE = init_type::Simulated{};

  //   auto fitness = std::make_unique<ScottDeJongBasins<N>>(A, B);
  //   std::vector<std::unique_ptr<FitnessFunction<N>>> time_functions;

  //   time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new Flat<N>()));
  //   time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new ScottDeJongBasins<N>(A, B)));
  //   time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new ScottDeJongBasins<N>(A, -B)));

  //   SimulationConfig sc{POP_SIZE, NUMBER_SIMULATED_PROCESSORS, NUMBER_GENOMES, INIT_TYPE, false};

  //   // Genomes are initialized at 0^N at the start of the search.

  //   for (int i = 0; i < time_functions.size(); i++) {
  //     for (int u = 0; u < 11; u++) {
  //       auto factory = [=](Rng &rng, FitnessFunction<N> &time_value) { return Genome(rng, time_value, 1.0 * u); };
  //       FitnessFunction<N> *time = time_functions[i].get();
  //       ExpConfig<N> fc{NUMBER_THREADS, NUMBER_RUNS, *fitness, *time, factory};

  //       run_experiment<SDBGenomeConfig<N>, N>(fc, sc, *csv_logger);
  //     }
  //   }
  // }
  return 0;
}
