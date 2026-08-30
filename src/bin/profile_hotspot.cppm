module;
#include <spdlog/spdlog.h>
#include <chrono>
#include <random>

export module evosim.main;
import evosim;

using namespace evosim;
const unsigned N = 10;

int main() {
  initialize_logger();
  spdlog::set_level(spdlog::level::warn);

  const int POP_SIZE = 10;
  const int NP = 10;
  const int NGENOMES = 1000;
  const int RUNS = 100000; // 20k runs ~3 seconds, enough for sampling

  spdlog::warn("Starting hotspot profile: {} runs", RUNS);

  Flat<N> fitness;
  Flat<N> time_fn;

  struct ZeroTime : Flat<N> {
    double operator()(const std::array<double, N>&) const {
      return 0.0;
    }
    std::string to_string() const { return "Zero"; }
  } ls_time;

  auto factory = [](Rng& rng, auto& tf) {
    return Genome<N>(rng, tf, std::normal_distribution<double>{0.0, 1.0});
  };

  auto start = std::chrono::high_resolution_clock::now();

  for (int run = 0; run < RUNS; run++) {
    Simulation<SDBTraits<N>, N> s(
        POP_SIZE, NP, NGENOMES, false, fitness, time_fn, factory, ls_time, run);
    s.run();
    if (run % 5000 == 0 && run > 0) {
      spdlog::warn("Completed {} runs", run);
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();
  spdlog::warn("Done {} runs in {:.2f} ms", RUNS, ms);

  return 0;
}
