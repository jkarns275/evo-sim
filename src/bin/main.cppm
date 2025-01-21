module;

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>

export module evosim.main;

import evosim;

using namespace evosim;

const unsigned N = 8;
int main(int argc, char **argv) {

  initialize_logger();

  auto csv_logger = std::make_shared<spdlog::sinks::basic_file_sink_st>("results.csv", "csv_output");

  double Afactor = 1.0;
  // for (double Afactor : {1.0, 1.5, 2.0}) {
  for (int pop_size : {10, 15, 20}) {
    spdlog::info("####################################");
    double A = 10;
    double B = Afactor * A;

    std::vector<InitType> init_types;
    for (int i = 0; i < 2; i++) {
      init_types.push_back(init_type::Uniform{i});
    }
    init_types.push_back(init_type::Simulated{});

    for (auto init_type : init_types) {
      for (int i = 0; i <= 0; i++) {
        float range = i * 1;
        auto factory = [=](Rng &rng, FitnessFunction<N> &time_value) { return Genome(rng, time_value, range); };
        auto fitness = std::make_unique<ScottDeJongBasins<N>>(A, B);
        auto time = std::make_unique<SphericalGaussian<N>>(-2.5);
        SimulationConfig sc{pop_size, 10, 1000, init_type};
        ExpConfig<N> fc{pop_size, 10, *fitness, *time, factory};

        spdlog::info("Range = {}; Sim type = {}", range, init_type_to_string(init_type));
        run_experiment<SDBGenomeConfig<N>, N>(fc, sc, *csv_logger);
        spdlog::info("----------------------");
      }
    }
  }
  return 0;
}
