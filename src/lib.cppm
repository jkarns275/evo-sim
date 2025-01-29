module;
#include <fmt/format.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <memory>
#include <random>
#include <vector>

export module evosim;
export import :function;
export import :simulator;
export import :genome;
export import :util;
export import :analysis;

export namespace evosim {

struct SimulationConfig {
  int pop_size, np, ngenomes;
  InitType init_type;
  bool use_sweet;

  SimulationConfig(int pop_size, int np, int ngenomes, InitType init_type, bool use_sweet)
      : pop_size(pop_size), np(np), ngenomes(ngenomes), init_type(init_type), use_sweet(use_sweet) {}
};

template <unsigned N, class G = Genome<N>> struct ExpConfig {
  int nthreads, nruns;
  FitnessFunction<N> &fitness_eval_fn;
  FitnessFunction<N> &time_eval_fn;
  std::function<G(Rng &, FitnessFunction<N> &)> factory;

  ExpConfig(int nthreads, int nruns, FitnessFunction<N> &fitness_eval_fn, FitnessFunction<N> &time_eval_fn,
            std::function<G(Rng &, FitnessFunction<N> &)> factory)
      : nthreads(nthreads), nruns(nruns), fitness_eval_fn(fitness_eval_fn), time_eval_fn(time_eval_fn),
        factory(factory) {}
};

template <typename GC, unsigned N, typename SC = Simulation<GC, N>, typename G = Genome<N>>
std::pair<double, double> run_experiment(ExpConfig<N, G> fc, SimulationConfig sc, spdlog::logger &logger) {
  std::vector<size_t> negative_count(fc.nthreads, 0);
  std::vector<size_t> positive_count(fc.nthreads, 0);
  std::vector<double> sum(fc.nthreads, 0.0);

  std::vector<std::thread> threads;
  auto f = [&](int t) {
    for (int i = 0; i < fc.nruns / fc.nthreads; i++) {
      SC s(sc.pop_size, sc.np, sc.ngenomes, sc.use_sweet, &fc.fitness_eval_fn, &fc.time_eval_fn, fc.factory,
           sc.init_type);
      s.run();

      // std::array<double, N> best = fc.fitness_eval_fn.global_optimum();
      sum[t] += *s.population[0].fitness / fc.nruns; // distance(s.population[0].x, best) / fc.nruns;
      bool converged = s.converged_to_global_best();

      if (converged)
        positive_count[t] += 1;
      else
        negative_count[t] += 1;
    }
  };

  for (int t = 0; t < fc.nthreads; t++) {
    threads.emplace_back(f, t);
  }

  for (int i = 0; i < fc.nthreads; i++)
    threads[i].join();

  double pc = 0.0, nc = 0.0;
  for (int i = 0; i < fc.nthreads; i++) {
    pc += (double)positive_count[i];
    nc += (double)negative_count[i];
  }

  double avg_fitness = 0.0;
  for (int i = 0; i < fc.nthreads; i++)
    avg_fitness += sum[i];

  double converged_prop = 100 * (pc / (pc + nc));
  double ci = converged_prop - 100.0 * wilson_confidence(pc, nc, 0.95);

  spdlog::info("Fitn: {:}", fc.fitness_eval_fn.to_string());
  spdlog::info("Time: {:}", fc.time_eval_fn.to_string());
  spdlog::info("Genome: {:}; {} Runs", GC().name, pc + nc);
  spdlog::info("Converged % +/- 95% CI: {} +- {}\n", converged_prop, ci);
  spdlog::info("Avg. Fitness: {}", avg_fitness);
  // logger.info("{}, {:.4f}, {:.4f}, {}, {}", "something", converged_prop, ci, pc + nc, sc.ngenomes);
  return {converged_prop, ci};
}

void initialize_logger() {
  std::vector<spdlog::sink_ptr> sinks;
  sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());

  sinks.push_back(std::make_shared<spdlog::sinks::basic_file_sink_st>("basic_file_log", "output.txt"));
  auto combined_logger = std::make_shared<spdlog::logger>("primary", begin(sinks), end(sinks));

  spdlog::register_logger(combined_logger);
  spdlog::set_default_logger(combined_logger);
}

bool mkdir(std::string path) {
  std::error_code err;
  return std::filesystem::create_directories(path, err) || std::filesystem::exists(path);
}
}; // namespace evosim
