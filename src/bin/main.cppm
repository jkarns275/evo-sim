module;

#include <spdlog/spdlog.h>

#include <memory>
#include <random>
#include <vector>

export module evosim.main;

import evosim;

using namespace evosim;

template <unsigned N> struct FitnessConfig {
  FitnessFunction<N> &fitness_eval_fn;
  FitnessFunction<N> &time_eval_fn;
  std::function<Genome<N>(Rng &, FitnessFunction<N> &)> factory;

  FitnessConfig(FitnessFunction<N> &fitness_eval_fn, FitnessFunction<N> &time_eval_fn,
                std::function<Genome<N>(Rng &, FitnessFunction<N> &)> factory)
      : fitness_eval_fn(fitness_eval_fn), time_eval_fn(time_eval_fn), factory(factory) {}
};

template <typename GC, unsigned N> void run_experiment(FitnessConfig<N> fc, int nthreads, int nruns, int ngenomes) {
  std::vector<size_t> negative_count(nthreads, 0);
  std::vector<size_t> positive_count(nthreads, 0);
  std::vector<std::thread> threads;

  for (int t = 0; t < nthreads; t++) {
    auto f = [&](int t) {
      for (int i = 0; i < nruns / nthreads; i++) {
        Simulation<GC, N> s(10, 10, 1000, fc.fitness_eval_fn, fc.time_eval_fn, fc.factory);
        s.run();

        auto &best = s.get_best_genome();
        bool converged = s.converged_to_global_best();

        if (converged)
          positive_count[t] += 1;
        else
          negative_count[t] += 1;
      }
    };

    threads.emplace_back(f, t);
  }

  for (int i = 0; i < nthreads; i++)
    threads[i].join();

  double pc = 0.0, nc = 0.0;
  for (int i = 0; i < nthreads; i++) {
    pc += (double)positive_count[i];
    nc += (double)negative_count[i];
  }

  double converged_prop = pc / (pc + nc);
  double ci = wilson_confidence(pc, nc, 0.95);

  spdlog::info("{:} / {:}     {:} R {:} I", fc.fitness_eval_fn.to_string(), GC().name, pc + nc, ngenomes);
  spdlog::info("Converged % +/- 95% CI: {} +- {}", converged_prop * 100, ci);
}

const unsigned N = 8;
int main(int argc, char **argv) {
  double A = 10.0;
  double B = A;
  auto factory = [=](Rng &rng, FitnessFunction<N> &time_value) { return Genome(rng, time_value); };
  auto fitness = std::make_unique<ScottDeJongBasins<N>>(A, B);
  auto time = std::make_unique<ScottDeJongBasins<N>>(A, B);
  FitnessConfig<N> fc{*fitness, *time, factory};

  run_experiment<SDBGenomeConfig<N>, N>(fc, 10, 100000, 1000);
  return 0;
}
