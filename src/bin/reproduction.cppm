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

  const int N_REPEATS = 100000;

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
  const int NUMBER_RUNS = 100'000;

  const InitType INIT_TYPE = init_type::Simulated{};

  spdlog::info("-------------------------------------------");
  spdlog::info("Reproducibility Experiments: A = B");
  spdlog::info("-------------------------------------------");

  {
    const double A = 10.0;
    const double B = A;

    auto fitness = std::make_unique<ScottDeJongBasins<N>>(A, B);
    std::vector<std::unique_ptr<FitnessFunction<N>>> time_functions;

    time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new Flat<N>()));
    time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new ScottDeJongBasins<N>(A, B)));
    time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new ScottDeJongBasins<N>(A, -B)));

    SimulationConfig sc{POP_SIZE, NUMBER_SIMULATED_PROCESSORS, NUMBER_GENOMES, INIT_TYPE, false};

    // Genomes are initialized at 0^N at the start of the search.
    auto factory = [=](Rng &rng, FitnessFunction<N> &time_value) {
      return Genome(rng, time_value, std::normal_distribution<double>{0.0, 1.0});
    };

    for (int i = 0; i < time_functions.size(); i++) {
      FitnessFunction<N> *time = time_functions[i].get();
      ExpConfig<N> fc{NUMBER_THREADS, NUMBER_RUNS, *fitness, *time, factory};

      run_experiment<SDBGenomeConfig<N>, N>(fc, sc, *csv_logger);
    }
  }

  spdlog::info("-------------------------------------------");
  spdlog::info("Reproducibility Experiments: B = 1.5A");
  spdlog::info("-------------------------------------------");
  {
    const double A = 10.0;
    const double B = 1.5 * A;

    auto fitness = std::make_unique<ScottDeJongBasins<N>>(A, B);
    std::vector<std::unique_ptr<FitnessFunction<N>>> time_functions;

    time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new Flat<N>()));
    time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new ScottDeJongBasins<N>(A, B)));
    time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new ScottDeJongBasins<N>(-A, B)));
    time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new ScottDeJongBasins<N>(A, -B)));

    SimulationConfig sc{POP_SIZE, NUMBER_SIMULATED_PROCESSORS, NUMBER_GENOMES, INIT_TYPE, false};

    // Genomes are initialized at 0^N at the start of the search.
    auto factory = [=](Rng &rng, FitnessFunction<N> &time_value) {
      return Genome(rng, time_value, std::normal_distribution<double>{0, 1});
    };

    for (int i = 0; i < time_functions.size(); i++) {
      FitnessFunction<N> *time = time_functions[i].get();
      ExpConfig<N> fc{NUMBER_THREADS, NUMBER_RUNS, *fitness, *time, factory};

      run_experiment<SDBGenomeConfig<N>, N>(fc, sc, *csv_logger);
    }
  }

  spdlog::info("-------------------------------------------");
  spdlog::info("Extended Experiments: A = B; C = xD for x in 1...10");
  spdlog::info("-------------------------------------------");
  {
    const double A = 10.0;
    const double B = A;

    auto fitness = std::make_unique<ScottDeJongBasins<N>>(A, B);
    std::vector<std::unique_ptr<FitnessFunction<N>>> time_functions;

    const double D_MIN = 1.0;
    const double D_MAX = 10.0;
    const int GRID_SIZE = 100;
    const double STEP_SIZE = (D_MAX - D_MIN) / STEP_SIZE;

    for (int i = 0; i < GRID_SIZE; i++) {
      double d = D_MIN + i * STEP_SIZE;
      time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new ScottDeJongBasins<N>(d * A, B)));
    }

    SimulationConfig sc{POP_SIZE, NUMBER_SIMULATED_PROCESSORS, NUMBER_GENOMES, INIT_TYPE, false};

    // Genomes are initialized at 0^N at the start of the search.
    auto factory = [=](Rng &rng, FitnessFunction<N> &time_value) {
      return Genome(rng, time_value, std::normal_distribution<double>{0, 1});
    };

    for (int i = 0; i < time_functions.size(); i++) {
      FitnessFunction<N> *time = time_functions[i].get();
      ExpConfig<N> fc{NUMBER_THREADS, NUMBER_RUNS, *fitness, *time, factory};

      run_experiment<SDBGenomeConfig<N>, N>(fc, sc, *csv_logger);
    }
  }

  spdlog::info("-------------------------------------------");
  spdlog::info("Extended Experiments: A = B; C = -xD for x in 1...10");
  spdlog::info("-------------------------------------------");
  {
    const double A = 10.0;
    const double B = A;

    auto fitness = std::make_unique<ScottDeJongBasins<N>>(A, B);
    std::vector<std::unique_ptr<FitnessFunction<N>>> time_functions;

    const double D_MIN = 1.0;
    const double D_MAX = 10.0;
    const int GRID_SIZE = 100;
    const double STEP_SIZE = (D_MAX - D_MIN) / STEP_SIZE;

    for (int i = 0; i < GRID_SIZE; i++) {
      double d = D_MIN + i * STEP_SIZE;
      time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new ScottDeJongBasins<N>(-d * A, B)));
    }

    SimulationConfig sc{POP_SIZE, NUMBER_SIMULATED_PROCESSORS, NUMBER_GENOMES, INIT_TYPE, false};

    // Genomes are initialized at 0^N at the start of the search.
    auto factory = [=](Rng &rng, FitnessFunction<N> &time_value) {
      return Genome(rng, time_value, std::normal_distribution<double>{0, 1});
    };

    for (int i = 0; i < time_functions.size(); i++) {
      FitnessFunction<N> *time = time_functions[i].get();
      ExpConfig<N> fc{NUMBER_THREADS, NUMBER_RUNS, *fitness, *time, factory};

      run_experiment<SDBGenomeConfig<N>, N>(fc, sc, *csv_logger);
    }
  }
  return 0;
}
