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
    int K = 2;
    int d = 2;
    double S = 2.0;

    Vec theta_star;
    std::vector<Vec> x;    
    std::vector<std::vector<Vec>> g;

    int best = -1;

    int true_best_arm() const {
        return best;
    }

    std::vector<double> dots;
    std::vector<double> means;
    std::vector<std::vector<double>> gap_dots;
    std::vector<std::vector<double>> gaps;

    void reallocate() {
        dots.resize(K);
        means.resize(K);
        g.resize(K, std::vector<Vec> (K));
        gap_dots.resize(K, std::vector<double> (K));
        gaps.resize(K, std::vector<double> (K));
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
    for (int j = 0; j < inst.d; ++j) in >> inst.theta_star[j];

    inst.x.resize(inst.K);
    for (int i = 0; i < inst.K; ++i) {
        inst.x[i] = Vec(inst.d);
        for (int j = 0; j < inst.d; ++j) in >> inst.x[i][j];
    }
    
    return inst;
}

Vec unit_vec(int d, int i) {
    Vec ret(d, 0.);
    ret[i] = 1;
    return ret;
}

inline Instance K2d2() {
    Instance inst(2, 2, 2.0);
    inst.theta_star[0] = 1.0;
    inst.theta_star[1] = 1.0;
    inst.x[0][0] = 1.0;
    inst.x[0][1] = 0.0;
    inst.x[1][0] = 0;
    inst.x[1][1] = -1;
    return inst;
}

inline Instance generate_instance(int d, double S) {
    Instance inst(d+1, d, S);
    inst.theta_star[0] = S-1;
    for (int i = 0; i < d; ++i) {
        inst.x[i][i] = 1;
    }
    inst.x[d][0] = std::cos(0.1);
    inst.x[d][1] = std::sin(0.1);
    return inst;
}

inline Instance generate_synthetic_instance(int K, int d, double S, RNG& rng) {
    Instance inst(K, d, S);

    double val = (S - 1.) / std::sqrt((double)d);
    for (int j = 0; j < d; ++j) inst.theta_star[j] = val;

    double theta_norm2 = 0.0;
    for (int j = 0; j < d; ++j) {
        theta_norm2 += inst.theta_star[j] * inst.theta_star[j];
    }
    double theta_norm = std::sqrt(theta_norm2);

    if (theta_norm < 1e-12) {
        for (int i = 0; i < K; ++i) inst.x[i] = rng.random_ball_vec(d);
        return inst;
    }

    const double rho = 0.8;
    const double u_max = rho * theta_norm;

    std::vector<double> u(K);
    if (K == 1) {
        u[0] = 0.0;
    } else {
        u[0] = 0.9 * theta_norm;
        for (int i = 1; i < K; ++i) {
            double t = (double)(K - i - 1) / (double)(K - 1);
            u[i] = u_max * t; 
        }
    }

    for (int i = 0; i < K; ++i) {
        double alpha = u[i] / theta_norm2;
        Vec x_par(d);
        for (int j = 0; j < d; ++j) x_par[j] = alpha * inst.theta_star[j];

        double par_norm2 = 0.0;
        for (int j = 0; j < d; ++j) par_norm2 += x_par[j] * x_par[j];
        double ortho_radius = 0.0;
        if (par_norm2 < 1.0) ortho_radius = std::sqrt(1.0 - par_norm2);

        Vec v = rng.random_ball_vec(d);
        double vTtheta = 0.0;
        for (int j = 0; j < d; ++j) vTtheta += v[j] * inst.theta_star[j];
        double coeff = vTtheta / theta_norm2;
        for (int j = 0; j < d; ++j) v[j] -= coeff * inst.theta_star[j];

        double v_norm2 = 0.0;
        for (int j = 0; j < d; ++j) v_norm2 += v[j] * v[j];
        double v_norm = std::sqrt(v_norm2);

        Vec x = x_par;
        if (v_norm > 1e-12 && ortho_radius > 0.0) {
            double scale = ortho_radius;
            for (int j = 0; j < d; ++j) x[j] += (scale / v_norm) * v[j];
        }

        double x_norm2 = 0.0;
        for (int j = 0; j < d; ++j) x_norm2 += x[j] * x[j];
        if (x_norm2 > 1.0) {
            double x_norm = std::sqrt(x_norm2);
            for (int j = 0; j < d; ++j) x[j] /= x_norm;
        }

        inst.x[i] = x;
    }

    return inst;
}



