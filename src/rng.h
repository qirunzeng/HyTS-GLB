#pragma once
#include <random>
#include <cmath>
#include "lin_alg.h"

struct RNG {
    std::mt19937_64 gen;
    explicit RNG(uint64_t seed) : gen(seed) {}

    double uniform01() {
        return std::uniform_real_distribution<double>(0.0,1.0)(gen);
    }

    int randint(int lo, int hi) { // inclusive
        return std::uniform_int_distribution<int>(lo, hi)(gen);
    }

    double normal01() {
        return std::normal_distribution<double>(0.0,1.0)(gen);
    }

    Vec random_unit_vec(int d) {
        Vec x(d);
        for (int i=0;i<d;++i) x[i]=normal01();
        double n = x.norm2();
        if (n<1e-12) x[0]=1.0, n=1.0;
        for (int i=0;i<d;++i) x[i]/=n;
        return x;
    }

    // random vector uniformly on ball ||x||<=1
    Vec random_ball_vec(int d) {
        Vec u = random_unit_vec(d);
        double r = std::pow(uniform01(), 1.0/d);
        for (int i=0;i<d;++i) u[i]*=r;
        return u;
    }
};
