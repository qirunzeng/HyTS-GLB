#pragma once
#include <cmath>
#include "lin_alg.h"

// Logistic link: mu(z) = 1/(1+exp(-z))
inline double sigmoid(double z) {
    if (z >= 0) {
        double ez = std::exp(-z);
        return 1.0 / (1.0 + ez);
    } else {
        double ez = std::exp(z);
        return ez / (1.0 + ez);
    }
}

inline double mu(double z) { return sigmoid(z); }
inline double mu_prime(double z) {
    double s = sigmoid(z);
    return s * (1.0 - s);
}

// Classic Bernoulli/logistic negative log-likelihood:
inline double nll_logistic(double z, int r01) {
    // stable log(1+exp(z))
    double log1pexp = (z > 0) ? (z + std::log1p(std::exp(-z))) : std::log1p(std::exp(z));
    return - (double)r01 * z + log1pexp;
}
