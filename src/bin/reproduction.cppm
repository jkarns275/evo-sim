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

/// This binary runs a series of experiments that represent a superset of the experiments run in "Evaluation-Time Bias
/// in Asynchronous Evolutionary Algorithms" by Eric O. Scott and Kenneth A. De Jong.
///
/// There are some additional figures / visualizations in the paper that are not produced by this script.
int main(int argc, char **argv) {

  initialize_logger();

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
  const int NUMBER_RUNS = 1'000'00;

  const InitType INIT_TYPE = init_type::Simulated{};
  spdlog::info("-------------------------------------------");
  spdlog::info("Reproducibility Experiments: Flat Fitness Landscape");
  spdlog::info("-------------------------------------------");
  {
    const int N = 2;
    double A = 10.0;
    double B = 15.0;

    auto csv_logger = spdlog::basic_logger_st("repro_flat_fitness", "repro_flat_fitness.csv", true);
    csv_logger->set_pattern("%v");
    csv_logger->info("x0,x1,Duration");

    auto factory = [=](Rng &rng, FitnessFunction<N> &time_value) {
      return Genome(rng, time_value, std::normal_distribution<double>{0.0, 1.0});
    };
    auto fitness = std::unique_ptr<FitnessFunction<N>>(new Flat<N>());
    auto time = std::unique_ptr<FitnessFunction<N>>(new ScottDeJongBasins<N>(A, -B));

    for (int i = 0; i < 5000; i++) {
      Simulation<SDBGenomeNoCOConfig<2>, 2> s(POP_SIZE, 10, 100, false, fitness.get(), time.get(), factory, INIT_TYPE);
      s.run();

      const Genome<N> &g = s.select();
      csv_logger->info("{:.4f},{:.4f},{:.4f}", g.x[0], g.x[1], (*time)(g.x));
    }
  }

  spdlog::info("-------------------------------------------");
  spdlog::info("Reproducibility Experiments: Wall-clock time Evaluation Sequence ");
  spdlog::info("-------------------------------------------");
  {
    const int N = 2;
    double A = 100.0;
    double B = 100.0;

    auto csv_logger = spdlog::basic_logger_st("repro_wall_clock_time_seq", "repro_wall_clock_seq.csv", true);
    csv_logger->set_pattern("%v");
    csv_logger->info("x0,x1,Start,Duration");

    auto factory = [=](Rng &rng, FitnessFunction<N> &time_value) {
      return Genome(rng, time_value, std::normal_distribution<double>{0.0, 1.0});
    };
    auto fitness = std::unique_ptr<FitnessFunction<N>>(new ScottDeJongBasins<N>(A, B));
    auto time = std::unique_ptr<FitnessFunction<N>>(new ScottDeJongBasins<N>(-A, B));

    SimulationWithLogging<SDBGenomeNoCOConfig<2>, 2> s(POP_SIZE, 32, 1000, false, fitness.get(), time.get(), factory,
                                                       csv_logger, INIT_TYPE);
    s.run();
  }

  spdlog::info("-------------------------------------------");
  spdlog::info("Reproducibility Experiments: A = B");
  spdlog::info("-------------------------------------------");
  {
    auto csv_logger = spdlog::basic_logger_st("repro_results_a_eq_b", "repro_a_eq_b.csv", true);
    csv_logger->set_pattern("%v");
    csv_logger->info("Experiment Name,Converged Percentage,95% CI");

    const double A = 10.0;
    const double B = A;

    auto fitness = std::make_unique<ScottDeJongBasins<N>>(A, B);
    std::vector<std::unique_ptr<FitnessFunction<N>>> time_functions;

    time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new ScottDeJongBasins<N>(-A, B)));
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

      auto [prop, ci] = run_experiment<SDBGenomeConfig<N>, N>(fc, sc, *csv_logger);
      csv_logger->info("{},{:.4f},{:.4f}", time->to_string(), prop, ci);
    }
  }

  spdlog::info("-------------------------------------------");
  spdlog::info("Reproducibility Experiments: B = 1.5A");
  spdlog::info("-------------------------------------------");
  {
    auto csv_logger = spdlog::basic_logger_st("repro_results_b_gt_a", "repro_b_gt_a.csv", true);
    csv_logger->set_pattern("%v");
    csv_logger->info("Experiment Name,Converged Percentage,95% CI");

    const double A = 10.0;
    const double B = 1.5 * A;

    auto fitness = std::make_unique<ScottDeJongBasins<N>>(A, B);
    std::vector<std::unique_ptr<FitnessFunction<N>>> time_functions;

    time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new ScottDeJongBasins<N>(-A, B)));
    time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new Flat<N>()));
    time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new ScottDeJongBasins<N>(A, B)));
    time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new ScottDeJongBasins<N>(A, -B)));

    SimulationConfig sc{POP_SIZE, NUMBER_SIMULATED_PROCESSORS, NUMBER_GENOMES, INIT_TYPE, false};

    // Genomes are initialized at 0^N at the start of the search.
    auto factory = [=](Rng &rng, FitnessFunction<N> &time_value) {
      return Genome(rng, time_value, std::normal_distribution<double>{0, 1});
    };

    for (int i = 0; i < time_functions.size(); i++) {
      FitnessFunction<N> *time = time_functions[i].get();
      ExpConfig<N> fc{NUMBER_THREADS, NUMBER_RUNS, *fitness, *time, factory};

      auto [prop, ci] = run_experiment<SDBGenomeConfig<N>, N>(fc, sc, *csv_logger);
      csv_logger->info("{},{:.4f},{:.4f}", time->to_string(), prop, ci);
    }
  }

  spdlog::info("-------------------------------------------");
  spdlog::info("Extended Experiments: A = B, no crossover");
  spdlog::info("-------------------------------------------");
  {
    auto csv_logger = spdlog::basic_logger_st("repro_results_a_eq_b_no_co", "repro_a_eq_b_no_co.csv", true);
    csv_logger->set_pattern("%v");
    csv_logger->info("Experiment Name,Converged Percentage,95% CI");

    const double A = 10.0;
    const double B = A;

    auto fitness = std::make_unique<ScottDeJongBasins<N>>(A, B);
    std::vector<std::unique_ptr<FitnessFunction<N>>> time_functions;

    time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new ScottDeJongBasins<N>(-A, B)));
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

      auto [prop, ci] = run_experiment<SDBGenomeNoCOConfig<N>, N>(fc, sc, *csv_logger);
      csv_logger->info("{},{:.4f},{:.4f}", time->to_string(), prop, ci);
    }
  }

  spdlog::info("-------------------------------------------");
  spdlog::info("Extended Experiments: B = 1.5A, no crossover");
  spdlog::info("-------------------------------------------");
  {
    auto csv_logger = spdlog::basic_logger_st("repro_results_b_gt_a_no_co", "repro_b_gt_a_no_co.csv", true);
    csv_logger->set_pattern("%v");
    csv_logger->info("Experiment Name,Converged Percentage,95% CI");

    const double A = 10.0;
    const double B = 1.5 * A;

    auto fitness = std::make_unique<ScottDeJongBasins<N>>(A, B);
    std::vector<std::unique_ptr<FitnessFunction<N>>> time_functions;

    time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new ScottDeJongBasins<N>(-A, B)));
    time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new Flat<N>()));
    time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new ScottDeJongBasins<N>(A, B)));
    time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new ScottDeJongBasins<N>(A, -B)));

    SimulationConfig sc{POP_SIZE, NUMBER_SIMULATED_PROCESSORS, NUMBER_GENOMES, INIT_TYPE, false};

    // Genomes are initialized at 0^N at the start of the search.
    auto factory = [=](Rng &rng, FitnessFunction<N> &time_value) {
      return Genome(rng, time_value, std::normal_distribution<double>{0, 1});
    };

    for (int i = 0; i < time_functions.size(); i++) {
      FitnessFunction<N> *time = time_functions[i].get();
      ExpConfig<N> fc{NUMBER_THREADS, NUMBER_RUNS, *fitness, *time, factory};

      auto [prop, ci] = run_experiment<SDBGenomeNoCOConfig<N>, N>(fc, sc, *csv_logger);
      csv_logger->info("{},{:.4f},{:.4f}", time->to_string(), prop, ci);
    }
  }

  /*
  spdlog::info("-------------------------------------------");
  spdlog::info("Extended Experiments: A = B; C = xD for x in 1...10");
  spdlog::info("-------------------------------------------");
  {
    auto csv_logger = spdlog::basic_logger_st("extended_c_xd", "extended_c_xd.csv", true);
    csv_logger->set_pattern("%v");
    csv_logger->info("Experiment Name,Converged Percentage,95% CI");

    const double A = 10.0;
    const double B = A;

    auto fitness = std::make_unique<ScottDeJongBasins<N>>(A, B);
    std::vector<std::unique_ptr<FitnessFunction<N>>> time_functions;

    const double D_MIN = 1.0;
    const double D_MAX = 10.0;
    const int GRID_SIZE = 10;
    const double STEP_SIZE = (D_MAX - D_MIN) / GRID_SIZE;

    for (int i = 0; i <= GRID_SIZE; i++) {
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

      auto [prop, ci] = run_experiment<SDBGenomeConfig<N>, N>(fc, sc, *csv_logger);
      csv_logger->info("{},{:.4f},{:.4f}", time->to_string(), prop, ci);
    }
  }

  spdlog::info("-------------------------------------------");
  spdlog::info("Extended Experiments: A = B; C = -xD for x in 1...10");
  spdlog::info("-------------------------------------------");
  {
    auto csv_logger = spdlog::basic_logger_st("extended_c_neg_xd", "extended_c_neg_xd.csv", true);
    csv_logger->set_pattern("%v");
    csv_logger->info("Experiment Name,Converged Percentage,95% CI");

    const double A = 10.0;
    const double B = A;

    auto fitness = std::make_unique<ScottDeJongBasins<N>>(A, B);
    std::vector<std::unique_ptr<FitnessFunction<N>>> time_functions;

    const double D_MIN = 1.0;
    const double D_MAX = 10.0;
    const int GRID_SIZE = 10;
    const double STEP_SIZE = (D_MAX - D_MIN) / GRID_SIZE;

    for (int i = 0; i <= GRID_SIZE; i++) {
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

      auto [prop, ci] = run_experiment<SDBGenomeConfig<N>, N>(fc, sc, *csv_logger);
      csv_logger->info("{},{:.4f},{:.4f}", time->to_string(), prop, ci);
    }
  }
  */
  return 0;
}
