module;
#include <array>
#include <concepts>
#include <optional>
#include <string_view>
#include <type_traits>

export module evosim:concepts;
import :core;

export namespace evosim {

template <class T, unsigned N>
concept Objective = requires(const T& obj, const std::array<double, N>& x) {
  { obj(x) } -> std::convertible_to<double>;
  { obj.global_optimum() } -> std::same_as<std::array<double, N>>;
  { obj.domain() } -> std::same_as<std::pair<double, double>>;
  { obj.converged(x) } -> std::convertible_to<bool>;
  { T::name } -> std::convertible_to<std::string_view>;
};

template <class T, unsigned N>
concept TimeModel = requires(const T& tm, const std::array<double, N>& x) {
  { tm(x) } -> std::convertible_to<double>;
  { tm.global_optimum() } -> std::same_as<std::array<double, N>>;
  { tm.domain() } -> std::same_as<std::pair<double, double>>;
  { T::name } -> std::convertible_to<std::string_view>;
};

template <class T, unsigned N>
concept GenomeType = requires(T g) {
  typename T::value_type;
  requires std::same_as<typename T::value_type, std::array<double, N>>;
  { g.x } -> std::convertible_to<std::array<double, N>&>;
  { g.time_finished } -> std::convertible_to<double&>;
  { g.fitness } -> std::same_as<std::optional<double>&>;
};

template <class M, class G, unsigned N>
concept MutationPolicy = requires(const M& m, const G& parent, Rng& rng) {
  { M::apply(parent, rng) } -> std::same_as<G>;
  { M::name } -> std::convertible_to<std::string_view>;
};

template <class C, class G, unsigned N>
concept CrossoverPolicy =
    requires(const C& c, const G& p0, const G& p1, Rng& rng) {
      { C::apply(p0, p1, rng) } -> std::same_as<G>;
      { C::name } -> std::convertible_to<std::string_view>;
    };

template <class LS, class G, unsigned N, class Obj>
concept LocalSearchPolicy = requires(G& g, Rng& rng, const Obj& obj) {
  { LS::apply(g, rng, obj) } -> std::same_as<void>;
  { LS::name } -> std::convertible_to<std::string_view>;
};

template <class T>
concept GenomeTraitsType = requires {
  typename T::genome_t;
  typename T::mutation_t;
  typename T::crossover_t;
  typename T::local_search_t;
  typename T::ls_time_model_t;
  typename T::init_policy_t;
  typename T::rng_t;
  { T::static_name } -> std::convertible_to<std::string_view>;
};

} // namespace evosim
