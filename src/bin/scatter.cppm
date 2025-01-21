module;

#include <boost/math/distributions/normal.hpp>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>

#include <memory>
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

  const int NUMBER_THREADS = 1;
  const int NUMBER_RUNS = 1'000'000;

  std::normal_distribution<double> dist(0.0, 3);

  Rng rng = make_rng();

  spdlog::info("-------------------------------------------");
  spdlog::info("Scatter-plot Experiments: Spherical Function @ Global Minimum");
  spdlog::info("-------------------------------------------");

  {
    const double A = 10.0;
    const double B = A;

    const InitType INIT_TYPE = init_type::Simulated{};

    std::vector<std::pair<std::string, std::unique_ptr<FitnessFunction<N>>>> fitness_functions;

    fitness_functions.push_back(
        {"Scott-DeJong Basins", std::unique_ptr<FitnessFunction<N>>(new ScottDeJongBasins<N>(B, A))});

    for (auto &[name, fitness] : fitness_functions) {
      std::string csv_name(std::format("sph_at_min_{}", name));
      auto csv_logger = spdlog::basic_logger_st("experiment_output_csv", "results.csv", true);
      csv_logger->set_pattern("%v");
      csv_logger->info("Converged Percentage, 95% CI, Distance from Minimum");

      SimulationConfig sc{POP_SIZE, NUMBER_SIMULATED_PROCESSORS, NUMBER_GENOMES, INIT_TYPE, false};

      // Genomes are initialized at 0^N at the start of the search.
      auto factory = [=](Rng &rng, FitnessFunction<N> &time_value) {
        return Genome(rng, time_value, std::normal_distribution<double>{0.0, 1.0});
      };

      const int N_DATAPOINTS = 100;
      const int N_OUTER_THREADS = 10;

      std::vector<std::thread> threads;
      std::mutex mutex; // for CSV logger

      auto thread_fn = [&]() {
        std::vector<std::string> rows;
        for (int i = 0; i < N_DATAPOINTS / N_OUTER_THREADS; i++) {
          std::array<double, N> center = fitness->global_optimum();
          std::array<double, N> offset = point_in_sphere<N>(rng, 5.0);

          std::string s;
          for (int j = 0; j < N; j++) {
            center[j] += offset[j];
            s += std::format("{:.4f}, ", center[j]);
          }

          spdlog::info("Center: {{ {} }}", s);

          SphericalGaussian<N> time(center, 1.4);
          auto fitness_center = fitness->global_optimum();

          ExpConfig<N> fc{NUMBER_THREADS, NUMBER_RUNS, *fitness, time, factory};

          auto [prop, ci] = run_experiment<SDBGenomeConfig<N>, N>(fc, sc, *csv_logger);
          // rows.push_back(std::format("{:.4f}, {:.4f}, {:.4f}", prop, ci, ));
          rows.push_back(std::format("{:.4f}, {:.4f}, {:.4f}", prop, ci, distance(fitness_center, center)));
          // rows.push_back(std::format("{:.4f}, {:.4f}, {:.4f}", prop, ci, euclidean<N>(offset)));
        }

        std::lock_guard<std::mutex> guard(mutex);
        for (std::string &row : rows)
          csv_logger->info(row);
      };

      for (int i = 0; i < N_OUTER_THREADS; i++)
        threads.emplace_back(thread_fn);

      for (auto &thread : threads)
        thread.join();
    }
  }

  return 0;
}
