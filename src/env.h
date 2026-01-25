#pragma once
#include <cmath>
#include "lin_alg.h"
#include "instance.h"
#define pii std::pair<int, int>

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

// reward Bernoulli/logistic negative log-likelihood:
inline double nll_logistic(double z, int r) {
    // stable log(1 + exp(z))
    double log1pexp = (z > 0) ? (z + std::log1p(std::exp(-z))) : std::log1p(std::exp(z));
    return - (double)r * z + log1pexp;
}

// simulate one Bernoulli reward
inline int sample_reward(const Instance& inst, int arm, RNG& rng) {
    return rng.uniform01() < inst.means[arm];
}

// simulate one dueling outcome y ~ Bernoulli(sigmoid((x_j-x_k)^T theta*))
inline int sample_duel_outcome(const Instance& inst, int j, int k, RNG& rng) {
    return (rng.uniform01() < inst.gaps[j][k]) ? 1 : 0;
}

#include <vector>

struct MeanStd {
    double mean;
    double stdev;
};

static MeanStd mean_std(const std::vector<long long>& a) {
    MeanStd ms{0.0, 0.0};
    const size_t n = a.size();
    if (n == 0) return ms;

    long double sum = 0.0L;
    for (long long x : a) sum += (long double)x;
    const long double mean = sum / (long double)n;

    if (n <= 1) {
        ms.mean = (double)mean;
        ms.stdev = 0.0;
        return ms;
    }

    long double ss = 0.0L;
    for (long long x : a) {
        long double diff = (long double)x - mean;
        ss += diff * diff;
    }
    // sample stdev
    long double var = ss / (long double)(n - 1);
    ms.mean = (double)mean;
    ms.stdev = std::sqrt((double)var);
    return ms;
}


static MeanStd mean_std(const std::vector<double>& a) {
    MeanStd ms{0.0, 0.0};
    const size_t n = a.size();
    if (n == 0) return ms;

    long double sum = 0.0L;
    for (double x : a) sum += (long double)x;
    const long double mean = sum / (long double)n;

    if (n <= 1) {
        ms.mean = (double)mean;
        ms.stdev = 0.0;
        return ms;
    }

    long double ss = 0.0L;
    for (double x : a) {
        long double diff = (long double)x - mean;
        ss += diff * diff;
    }
    long double var = ss / (long double)(n - 1); // sample variance
    ms.mean = (double)mean;
    ms.stdev = std::sqrt((double)var);
    return ms;
}