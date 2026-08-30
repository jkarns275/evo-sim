module;
#include <boost/math/distributions/normal.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <numbers>
#include <random>
#include <string_view>

export module evosim:operators;
import :core;
import :genome_base;
import :ziggurat;
import :objective;


import :fast_uniform;


export namespace evosim {

// Static policy versions replacing old virtual Mutation/Crossover hierarchy.
// Old virtual interfaces and concrete implementations removed as deprecated code after migration to Simulation2 static polymorphism path and deletion of old simulator.cppm.

template <typename G, unsigned N>
struct GaussianMutationPolicy {
  static constexpr std::string_view name = "GaussianMutation";

  static G apply(const G& parent, auto& generator) {
    constexpr double p = 1.0 / static_cast<double>(N);
    std::normal_distribution<float> dist(0.0, 0.5);
    G child(parent);
    for (size_t i = 0; i < N; i++)
      if (std::generate_canonical<float, 32>(generator) < p)
        child.x[i] += dist(generator);
    return child;
  }
};

// Fast Gaussian mutation using ziggurat algorithm instead of std::normal_distribution
// Borrowed from https://github.com/snsinfu/cxx-ziggurat for ultra-fast normal RNG.
// Much faster than std::normal_distribution which uses Box-Muller with log, sqrt, sin, cos per call.
// Ziggurat uses precomputed tables and rejection sampling, typically 2-3x faster than Box-Muller.
// Created as new policy for benchmarking as requested, to compare against std::normal_distribution baseline.
template <typename G, unsigned N>
struct FastGaussianMutationPolicy {
  static constexpr std::string_view name = "FastGaussianMutation";

  static G apply(const G& parent, auto& generator) {
    constexpr uint32_t thresh = (1.0 / static_cast<double>(N)) * static_cast<float>(std::numeric_limits<uint32_t>::max());
    ziggurat_normal_distribution<float> dist(0.0f, 0.5f);
    G child(parent);
    for (size_t i = 0; i < N; i++) {
      if (static_cast<uint32_t>(generator()) < thresh) {
        child.x[i] += dist(generator);
      }
    }
    return child;
  }
};

// Geometric-skipping Gaussian mutation for benchmarking.
// Instead of N Bernoulli trials with p=1/N, sample the distance to the next
// mutated dimension using geometric distribution via -log(U)/-log1p(-p).
// Uses fast bit-cast uniform (generator() * 0x1.0p-32) instead of
// std::generate_canonical for speed, and __builtin_log for fast log.
// On average ~1 mutation per child for N=10, reducing RNG calls from N to ~1.
// Template parameter UseZiggurat selects between ziggurat and std::normal_distribution
// for the actual Gaussian delta.
template <typename G, unsigned N, bool UseZiggurat = true>
struct GeometricGaussianMutationPolicy {
  static constexpr std::string_view name = UseZiggurat
      ? "GeometricZigguratMutation"
      : "GeometricGaussianMutation";

  static G apply(const G& parent, auto& generator) {
    G child(parent);

    constexpr double p = 1.0 / double(N);
    const double log1p_neg_p = std::log1p(-p);

    static thread_local std::normal_distribution<float> std_dist(0.0f, 0.5f);
    static constexpr ziggurat_normal_distribution<float> zig_dist(0.0f, 0.5f);

    int idx = -1;
    while (true) {
      // Fast uniform in (0,1] using 32-bit RNG: 2^-32 to 1.0
      // Using bit-cast instead of std::generate_canonical for speed.
      // 32 bits is plenty for geometric skipping with p=1/N.
      double u = static_cast<double>(generator()) * 0x1.0p-32;
      if (u == 0.0) u = 0x1.0p-32;
      // __builtin_log is faster than std::log with -ffast-math
      int skip = static_cast<int>(__builtin_log(u) / log1p_neg_p);
      idx += 1 + skip;
      if (idx >= static_cast<int>(N)) break;

      float delta;
      if constexpr (UseZiggurat) {
        delta = zig_dist(generator);
      } else {
        delta = std_dist(generator);
      }
      child.x[idx] += delta;
      // Continue loop to allow multiple mutations per child (preserves
      // original Bernoulli(p) distribution). Uncomment break for exactly one:
      // break;
    }
    return child;
  }
};

// Geometric-skipping with exactly one mutation per child, selected uniformly.
// Even faster than geometric skipping when you want exactly one mutation.
// Uses UniformDist for index selection (defaults to uniform_int_dist_t which is
// FastUniformIntDist, but can be configured to StdUniformIntDist via traits).
template <typename G, unsigned N, bool UseZiggurat = true, typename UniformDist = uniform_int_dist_t>
struct GeometricSingleMutationPolicy {
  static constexpr std::string_view name = UseZiggurat
      ? "GeometricSingleZiggurat"
      : "GeometricSingleGaussian";

  static G apply(const G& parent, auto& generator) {
    G child(parent);
    static thread_local std::normal_distribution<float> std_dist(0.0f, 0.5f);
    static constexpr ziggurat_normal_distribution<float> zig_dist(0.0f, 0.5f);

    size_t idx = UniformDist(static_cast<uint32_t>(N))(generator);
    float delta;
    if constexpr (UseZiggurat) {
      delta = zig_dist(generator);
    } else {
      delta = std_dist(generator);
    }
    child.x[idx] += delta;
    return child;
  }
};

template <typename G, unsigned N, typename UniformDist = uniform_int_dist_t>
struct TwoPointCrossoverPolicy {
  static constexpr std::string_view name = "TwoPointCrossover";

  static G apply(const G& p0, const G& p1, auto& generator) {
    G child(p0);
    size_t start = UniformDist(static_cast<uint32_t>(N - 2))(generator);
    size_t end = start + UniformDist(static_cast<uint32_t>(N - start))(generator);
    std::copy(
        p1.x.begin() + start, p1.x.begin() + end, child.x.begin() + start);
    return child;
  }
};

template <typename G, unsigned N>
struct LineSearchCrossoverPolicy {
  static constexpr std::string_view name = "LineSearchCrossover";

  static G apply(const G& p0, const G& p1, auto& generator) {
    G child(p0);
    for (size_t i = 0; i < N; i++) {
      double grad = p0.x[i] - p1.x[i];
      child.x[i] = p0.x[i] +
          (generator.next_uniform_double() * 2.0 - 0.5) * grad;
    }
    return child;
  }
};

template <typename G, unsigned N>
struct NopCrossoverPolicy {
  static constexpr std::string_view name = "NopCrossover";

  static G apply(const G& p0, const G&, auto&) {
    return G(p0);
  }
};

template <typename G, unsigned N>
struct NoOpLocalSearch {
  static constexpr std::string_view name = "NoOpLS";

  template <typename Obj>
  static void apply(G&, auto&, const Obj&) {}

  // backward compatibility overload without objective
  static void apply(G& g, auto& rng) {
    apply(g, rng, nullptr);
  }
};

template <typename G, unsigned N, int Steps = 10, double StepSize = 0.1>
struct HillClimbLS {
  static constexpr std::string_view name = "HillClimbLS";

  template <typename Obj>
  static void apply(G& g, auto& rng, const Obj&) {
    std::normal_distribution<double> nd(0.0, StepSize);
    for (int s = 0; s < Steps; s++) {
      for (size_t i = 0; i < N; i++) {
        g.x[i] += nd(rng);
      }
    }
  }

  static void apply(G& g, auto& rng) {
    apply(g, rng, nullptr);
  }
};

// Backpropagation gradient per fitness function - primary template fails to
// compile to enforce explicit specialization per fitness function as requested
// for organizational clarity.
template <typename FitnessFn, unsigned N>
struct BackpropagationGradient {
  static_assert(
      sizeof(FitnessFn) == 0,
      "BackpropagationGradient must be specialized per fitness function type to provide analytic gradient implementation. Define explicit specialization template<> struct BackpropagationGradient<SpecificFitnessFn<N>, N> with static gradient method returning std::array<double,N>.");
};

// Explicit specializations for new CRTP objective types (FlatObj etc.)
template <unsigned N>
struct BackpropagationGradient<FlatObj<N>, N> {
  static std::array<double, N> gradient(const FlatObj<N>&, const std::array<double, N>&) {
    return array_of<N>(0.0);
  }
};
template <unsigned N>
struct BackpropagationGradient<ScottDeJongBasinsObj<N>, N> {
  static std::array<double, N> gradient(const ScottDeJongBasinsObj<N>& obj, const std::array<double, N>& x) {
    return obj.gradient(x);
  }
};
template <unsigned N>
struct BackpropagationGradient<SchwefelObj<N>, N> {
  static std::array<double, N> gradient(const SchwefelObj<N>& obj, const std::array<double, N>& x) {
    return obj.gradient(x);
  }
};
template <unsigned N>
struct BackpropagationGradient<RosenbrockObj<N>, N> {
  static std::array<double, N> gradient(const RosenbrockObj<N>& obj, const std::array<double, N>& x) {
    return obj.gradient(x);
  }
};
template <unsigned N>
struct BackpropagationGradient<AckleyObj<N>, N> {
  static std::array<double, N> gradient(const AckleyObj<N>& obj, const std::array<double, N>& x) {
    return obj.gradient(x);
  }
};
template <unsigned N>
struct BackpropagationGradient<SphericalObj<N>, N> {
  static std::array<double, N> gradient(const SphericalObj<N>& obj, const std::array<double, N>& x) {
    return obj.gradient(x);
  }
};
template <unsigned N>
struct BackpropagationGradient<SphericalGaussianObj<N>, N> {
  static std::array<double, N> gradient(const SphericalGaussianObj<N>& obj, const std::array<double, N>& x) {
    return obj.gradient(x);
  }
};
template <unsigned N>
struct BackpropagationGradient<InvSphericalGaussianObj<N>, N> {
  static std::array<double, N> gradient(const InvSphericalGaussianObj<N>& obj, const std::array<double, N>& x) {
    return obj.gradient(x);
  }
};

// Backpropagation local search policy using analytic gradient per fitness
// function. Template parameters: G genome type, FitnessFn fitness function type
// to specialize gradient computation, N dimensions, Steps number of gradient
// descent steps, StepSize learning rate, D number of dimensions to update per
// step (default N = all dimensions) as additional lever per user request.
template <
    typename G,
    typename FitnessFn,
    unsigned N,
    int Steps = 10,
    double StepSize = 0.01,
    unsigned D = N>
struct BackpropagationLS {
  static constexpr std::string_view name = "BackpropagationLS";
  static_assert(D <= N, "D must be <= N for partial dimension update");

  static void apply(G& g, auto& rng, const FitnessFn& obj) {
    // Single generic gradient descent loop relying on explicit
    // per-fitness-function specialization of BackpropagationGradient for
    // analytic gradient computation, as requested to avoid duplication across
    // specializations. No dynamic_cast fallback needed — original simulation
    // path does not use local search at all, and new Simulation2 path uses
    // compile-time traits to ensure type matches exactly at compile time per
    // user request to remove virtual gradient case.

    std::array<size_t, N> indices{};
    for (size_t i = 0; i < N; i++)
      indices[i] = i;

    // Use explicit specialization of BackpropagationGradient per fitness
    // function type for compile-time enforcement and organizational clarity as
    // requested. Default primary template will fail to compile if not
    // specialized.
    for (int s = 0; s < Steps; s++) {
      std::array<double, N> grad =
          BackpropagationGradient<FitnessFn, N>::gradient(obj, g.x);

      if constexpr (D == N) {
        for (size_t i = 0; i < N; i++) {
          g.x[i] -= StepSize * grad[i];
        }
      } else {
        // Partial dimension update lever: randomly select D distinct dimensions
        // each step without replacement using Fisher-Yates partial shuffle
        for (size_t i = 0; i < D; i++) {
          size_t j = i + FastUniformInt{}(rng, static_cast<uint32_t>(N - i));
          std::swap(indices[i], indices[j]);
        }
        for (size_t i = 0; i < D; i++) {
          size_t dim = indices[i];
          g.x[dim] -= StepSize * grad[dim];
        }
      }
    }
  }

  // Backward compatibility overload without objective — does nothing to
  // preserve old behavior during transition when objective not provided
  static void apply(G&, Rng&) {}
};

} // namespace evosim
