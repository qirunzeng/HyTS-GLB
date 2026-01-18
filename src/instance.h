#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iomanip>
#include "lin_alg.h"
#include "rng.h"

struct Instance {
    int K=0;
    int d=0;
    double S=1.0;

    Vec theta_star;
    std::vector<Vec> x; // arms, each in R^d

    int true_best_arm() const {
        int best=0;
        double bestv = dot(x[0], theta_star);
        for (int i=1;i<K;++i) {
            double v = dot(x[i], theta_star);
            if (v > bestv) { 
                bestv = v; 
                best  = i; 
            }
        }
        return best;
    }

    Instance() = default;
    Instance(int K, int d, double S) : K(K), d(d), S(S), theta_star(d), x(K, Vec(d)) {}
};

inline void save_instance(const Instance& inst, const std::string& path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot open for writing: " + path);
    out << inst.K << " " << inst.d << " " << std::setprecision(17) << inst.S << "\n";
    for (int j = 0; j < inst.d; ++j) {
        out << std::setprecision(17) << inst.theta_star[j] << " \n"[j+1 == inst.d];
    }
    for (int i = 0; i < inst.K; ++i) for (int j = 0; j < inst.d; ++j) {
        out << std::setprecision(17) << inst.x[i][j] << " \n"[j+1 == inst.d];
    }
}

inline Instance load_instance(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open for reading: " + path);

    Instance inst;
    in >> inst.K >> inst.d >> inst.S;
    inst.theta_star = Vec(inst.d);
    for (int j=0;j<inst.d;++j) in >> inst.theta_star[j];

    inst.x.resize(inst.K);
    for (int i=0;i<inst.K;++i) {
        inst.x[i] = Vec(inst.d);
        for (int j=0;j<inst.d;++j) in >> inst.x[i][j];
    }
    return inst;
}

inline Instance generate_synthetic_instance(int K, int d, double S, RNG& rng) {
    Instance inst(K, d, S);
    double val = (S - 1.0) / std::sqrt((double)d);
    for (int j=0;j<d;++j) inst.theta_star[j]=val;
    for (int i=0;i<K;++i) {
        Vec xi = rng.random_ball_vec(d);
        inst.x[i]=xi;
    }
    return inst;
}
