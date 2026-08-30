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

/*
 * Groups: CNT SWT GEN
 *          0   0   0
 *          0   0   1
 *          0   1   0
 *          0   1   1
 *          1   0   0
 *          1   0   1
 *          1   1   0
 *          1   1   1
 */

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
  const int POP_SIZE = 10;
  const int NUMBER_SIMULATED_PROCESSORS = 10;
  const int NUMBER_GENOMES = 1000;

  const int NUMBER_THREADS = 10;
  const int NUMBER_RUNS = 1'000'000;
  std::vector<std::pair<
      std::unique_ptr<FitnessFunction<N>>,
      std::vector<std::unique_ptr<FitnessFunction<N>>>>>
      landscapes;

  {
    std::vector<std::pair<int, int>> values = {
        {10, 10}, {10, 15}, {10, 20}, {10, 25}, {10, 30}};
    for (auto [A, B] : values) {
      std::vector<std::unique_ptr<FitnessFunction<N>>> time_functions;
      auto fitness = std::make_unique<ScottDeJongBasins<N>>(A, B);

      time_functions.emplace_back(
          std::unique_ptr<FitnessFunction<N>>(new ScottDeJongBasins<N>(-A, B)));
      time_functions.emplace_back(
          std::unique_ptr<FitnessFunction<N>>(new Flat<N>()));
      time_functions.emplace_back(
          std::unique_ptr<FitnessFunction<N>>(new ScottDeJongBasins<N>(A, B)));
      time_functions.emplace_back(
          std::unique_ptr<FitnessFunction<N>>(new ScottDeJongBasins<N>(A, -B)));

      landscapes.emplace_back(std::move(fitness), std::move(time_functions));
    }

    // We propose the following set of optimization test functions for guaging
    // the efficacy of a evaluation time bias mitigation technique:
    // - Schwefel
    // - Rosenbrock
    //
    // For each of the fitness functions, we use the following time-functions
    // centered about the global minimum of the fitness functions:
    // - Inverse Spherical
    // - Inverse Ackley
    // - Flat
    // - Ackley
    // - Spherical
    // std::vector<std::unique_ptr<FitnessFunction<N>>> fitness_functions;
    // fitness_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new
    // Schwefel<N>()));
    // fitness_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new
    // Rosenbrock<N>()));

    // for (size_t i = 0; i < fitness_functions.size(); i++) {
    //   FitnessFunction<N> &f = *fitness_functions[i];
    //   std::vector<std::unique_ptr<FitnessFunction<N>>> time_functions;
    //   time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new
    //   Ackley<N>f.global_optimum(), true)));
    //   time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new
    //   Flat<N>()));
    //   time_functions.emplace_back(std::unique_ptr<FitnessFunction<N>>(new
    //   Ackley<N>f.global_optimum(), false)));

    //   landscapes.emplace_back(std::move(fitness_functions[i]),
    //   std::move(time_functions));
    // }
  }
  spdlog::info("-------------------------------------------");
  spdlog::info("Reproducibility Experiments: A = B Control");
  spdlog::info("-------------------------------------------");
  {
    auto csv_logger = spdlog::basic_logger_st(
        "mitigation_control", "results/mitigation_control.csv", true);
    csv_logger->set_pattern("%v");
    csv_logger->info("Experiment Name,Converged Percentage,95% CI");

    SimulationConfig sc{
        POP_SIZE,
        NUMBER_SIMULATED_PROCESSORS,
        NUMBER_GENOMES,
        init_policy::Simulated{},
        false};

    // Genomes are initialized at 0^N at the start of the search.
    auto factory = [=](Rng& rng, auto& time_value) {
      return Genome<N>(
          rng, time_value, std::normal_distribution<double>{0.0, 1.0});
    };

    for (auto& [fitness, time_functions] : landscapes) {
      for (int i = 0; i < time_functions.size(); i++) {
        FitnessFunction<N>* time = time_functions[i].get();
        ExpConfig<N, Genome<N>> fc{
            NUMBER_THREADS, NUMBER_RUNS, *fitness, *time, factory};

        auto [prop, ci] = run_experiment<
            SDBGenomeConfig<N>,
            N,
            Simulation<SDBGenomeConfig<N>, N>,
            Genome<N>>(fc, sc, *csv_logger);
        csv_logger->info("{},{:.4f},{:.4f}", time->to_string(), prop, ci);
      }
    }
  }

  spdlog::info("-------------------------------------------");
  spdlog::info("Reproducibility Experiments: A = B GEN");
  spdlog::info("-------------------------------------------");
  {
    auto csv_logger = spdlog::basic_logger_st(
        "mitigation_gen", "results/mitigation_gen.csv", true);
    csv_logger->set_pattern("%v");
    csv_logger->info("Experiment Name,Converged Percentage,95% CI");
    SimulationConfig sc{
        POP_SIZE,
        NUMBER_SIMULATED_PROCESSORS,
        NUMBER_GENOMES,
        init_policy::Simulated{},
        false};

    // Genomes are initialized at 0^N at the start of the search.
    auto factory = [=](Rng& rng, auto& time_value) {
      return Genome<N>(
          rng, time_value, std::normal_distribution<double>{0.0, 1.0});
    };

    for (auto& [fitness, time_functions] : landscapes)
      for (int i = 0; i < time_functions.size(); i++) {
        FitnessFunction<N>* time = time_functions[i].get();
        ExpConfig<N, Genome<N>> fc{
            NUMBER_THREADS, NUMBER_RUNS, *fitness, *time, factory};

        auto [prop, ci] = run_experiment<
            SDBGenomeConfig<N>,
            N,
            Simulation<SDBGenomeConfig<N>, N>,
            Genome<N>>(fc, sc, *csv_logger);
        csv_logger->info("{},{:.4f},{:.4f}", time->to_string(), prop, ci);
      }
  }

  spdlog::info("-------------------------------------------");
  spdlog::info("Reproducibility Experiments: A = B SWT");
  spdlog::info("-------------------------------------------");
  {
    auto csv_logger = spdlog::basic_logger_st(
        "mitigation_swt", "results/mitigation_swt.csv", true);
    csv_logger->set_pattern("%v");
    csv_logger->info("Experiment Name,Converged Percentage,95% CI");
    SimulationConfig sc{
        POP_SIZE, NUMBER_SIMULATED_PROCESSORS, NUMBER_GENOMES, init_policy::Simulated{}, true};

    // Genomes are initialized at 0^N at the start of the search.
    auto factory = [=](Rng& rng, auto& time_value) {
      return Genome<N>(
          rng, time_value, std::normal_distribution<double>{0.0, 1.0});
    };

    for (auto& [fitness, time_functions] : landscapes)
      for (int i = 0; i < time_functions.size(); i++) {
        FitnessFunction<N>* time = time_functions[i].get();
        ExpConfig<N, Genome<N>> fc{
            NUMBER_THREADS, NUMBER_RUNS, *fitness, *time, factory};

        auto [prop, ci] = run_experiment<
            SDBGenomeConfig<N>,
            N,
            Simulation<SDBGenomeConfig<N>, N>,
            Genome<N>>(fc, sc, *csv_logger);
        csv_logger->info("{},{:.4f},{:.4f}", time->to_string(), prop, ci);
      }
  }
  spdlog::info("-------------------------------------------");
  spdlog::info("Reproducibility Experiments: A = B w/ CNT");
  spdlog::info("-------------------------------------------");
  {
    auto csv_logger = spdlog::basic_logger_st(
        "mitigation_cnt", "results/mitigation_cnt.csv", true);
    csv_logger->set_pattern("%v");
    csv_logger->info("Experiment Name,Converged Percentage,95% CI");

    SimulationConfig sc{
        POP_SIZE,
        NUMBER_SIMULATED_PROCESSORS,
        NUMBER_GENOMES,
        init_policy::Simulated{},
        false};

    // Genomes are initialized at 0^N at the start of the search.
    auto factory = [=](Rng& rng, auto& time_value) {
      return GenomeWithCounter<N>(
          rng, time_value, std::normal_distribution<double>{0.0, 1.0});
    };

    for (auto& [fitness, time_functions] : landscapes)
      for (int i = 0; i < time_functions.size(); i++) {
        FitnessFunction<N>* time = time_functions[i].get();
        ExpConfig<N, GenomeWithCounter<N>> fc{
            NUMBER_THREADS, NUMBER_RUNS, *fitness, *time, factory};

        auto [prop, ci] = run_experiment<
            CounterGenomeConfig<N>,
            N,
            Simulation<CounterGenomeConfig<N>, N>,
            GenomeWithCounter<N>>(fc, sc, *csv_logger);
        csv_logger->info("{},{:.4f},{:.4f}", time->to_string(), prop, ci);
      }
  }

  return 0;
}
