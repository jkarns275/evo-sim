module;
#include <random>
#include <variant>

export module evosim:util;

export namespace evosim {

typedef std::mt19937_64 Rng;
inline const std::normal_distribution<double> NORMAL_DISTRIBUTION = std::normal_distribution<double>(0.0, 1.0);

/// Create a random number generator and warm it up
Rng make_rng() {
  std::random_device rd;
  Rng rng(rd());

  for (int i = 0; i < 1000; i++)
    std::uniform_int_distribution<unsigned long>(0, -1)(rng);

  return rng;
}

template <unsigned long N> double distance(const std::array<double, N> &x, const std::array<double, N> &y) {
  double sq_sum = 0.0;

  for (size_t i = 0; i < N; i++) {
    double diff = x[i] - y[i];
    sq_sum += diff * diff;
  }

  return std::sqrt(sq_sum);
}

template <unsigned long N> inline constexpr std::array<double, N> array_of(double value) {
  std::array<double, N> d;
  d.fill(value);
  return d;
}

template <unsigned long N> double euclidean(const std::array<double, N> &x) {
  double sq_sum = 0.0;

  for (size_t i = 0; i < N; i++)
    sq_sum += x[i] * x[i];

  return std::sqrt(sq_sum);
}

/*
u = np.random.normal(0,1,d)  # an array of d normally distributed random variables

norm=np.sum(u**2) **(0.5)

r = random()**(1.0/d)

x= r*u/norm
*/
template <unsigned long N> std::array<double, N> point_in_sphere(Rng &rng, double radius) {
  std::normal_distribution<double> normal(0.0, 1.0);

  std::array<double, N> direction;
  double sq_sum = 0;
  for (size_t i = 0; i < N; i++) {
    double value = normal(rng);
    sq_sum += value * value;
    direction[i] = value;
  }

  double mag = std::sqrt(sq_sum);
  double r = std::generate_canonical<double, 52>(rng) * radius;
  for (size_t i = 0; i < N; i++)
    direction[i] = r * direction[i] / mag;

  return direction;
}

template <typename VariantType, typename T, std::size_t index = 0> constexpr std::size_t variant_index() {
  static_assert(std::variant_size_v<VariantType> > index, "Type not found in variant");
  if constexpr (index == std::variant_size_v<VariantType>) {
    return index;
  } else if constexpr (std::is_same_v<std::variant_alternative_t<index, VariantType>, T>) {
    return index;
  } else {
    return variant_index<VariantType, T, index + 1>();
  }
}

}; // namespace evosim
