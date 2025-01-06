module;
#include <boost/math/distributions/normal.hpp>

export module evosim:analysis;

export namespace evosim {

double alpha_to_zscore(double alpha) {
  double p = 1 - (alpha / 2);
  return boost::math::quantile(boost::math::normal_distribution<>(), p);
}

/// Computes confidence interval for given samples from a binomial distribution.
double wilson_confidence(size_t ups, size_t downs, double confidence = 0.95) {
  double n = (double)ups + (double)downs;

  if (n == 0.0)
    return 0.0;

  double z = alpha_to_zscore(1 - confidence);

  double phat = (double)ups / n;
  return (phat + z * z / (2 * n) - z * std::sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n);
}

}; // namespace evosim

// def confidence(ups, downs):
//     n = ups + downs
//
//     if n == 0:
//         return 0
//
//     z = 1.0 #1.44 = 85%, 1.96 = 95%
//     phat = float(ups) / n
//     return ((phat + z*z/(2*n) - z * sqrt((phat*(1-phat)+z*z/(4*n))/n))/(1+z*z/n))
