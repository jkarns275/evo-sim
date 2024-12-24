module;

#include <format>
#include <random>
#include <spdlog/spdlog.h>
#include <vector>

export module evosim:main;

import evosim;

using namespace evosim;

typedef ScottDeJongBasins<2> sdb_basis;
typedef sdb_basis::Mutation<0.1> sdb_mut;

int main(int argc, char **argv) {

  double negative_count = 0;
  double positive_count = 0;
  double A = 2.0;

  for (double B : {2.0, 2.1, 2.2}) {
    spdlog::info("A={}, B={}", A, B);
    for (int i = 0; i < 100000; i++) {
      std::vector<sdb_basis> initial_population;
      std::function<sdb_basis(std::mt19937_64 &)> factory = [=](std::mt19937_64 &rng) { return sdb_basis(A, B, rng); };
      Simulation<sdb_basis, sdb_mut, sdb_basis::Ord, sdb_basis::Ord> s(10, 10, 1000, factory);
      s.run();

      sdb_basis &best = s.get_best_genome();

      // spdlog::info("{}", best.to_string());

      double pd = sdb_basis::positive_basin(A, B).distance(best);
      double nd = sdb_basis::negative_basin(A, B).distance(best);

      if (pd < nd) {
        // spdlog::info("Converged to positive basin.");
        positive_count += 1;
      } else {
        // spdlog::info("Converged to negative basin.");
        negative_count += 1;
      }
    }

    double basin_a_percentage = positive_count / (positive_count + negative_count);
    double basin_b_percentage = negative_count / (positive_count + negative_count);

    spdlog::info("A prop: {}", basin_a_percentage);
    spdlog::info("B prop: {}", basin_b_percentage);
  }

  return 0;
}
