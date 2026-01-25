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
    std::vector<Vec> x;             // arms, each in R^d
    std::vector<std::vector<Vec>> g;// gaps

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

// inline Instance generate_synthetic_instance(int K, int d, double S, RNG& rng) {
//     Instance inst(K, d, S);

//     // Keep the same theta_star generation as before
//     double val = (S - 1.0) / std::sqrt((double)d);
//     for (int j = 0; j < d; ++j) inst.theta_star[j] = val;

//     const double theta_norm2 = dot(inst.theta_star, inst.theta_star);
//     if (K <= 0) return inst;

//     // Degenerate theta -> fall back to random arms
//     if (theta_norm2 <= 0.0) {
//         for (int i = 0; i < K; ++i) inst.x[i] = rng.random_ball_vec(d);
//         return inst;
//     }

//     const double theta_norm = std::sqrt(theta_norm2);

//     // Let desired rewards r_i = x_i^T theta be evenly spaced in [+r_max, -r_max].
//     // Choose r_max = ||theta|| so that feasibility with ||x||<=1 is guaranteed,
//     // because max possible dot(x,theta) over ||x||<=1 equals ||theta||.
//     const double r_max = theta_norm;

//     // Unit direction u = theta / ||theta||
//     Vec u = inst.theta_star;
//     for (int j = 0; j < d; ++j) u[j] /= theta_norm;

//     // d == 1: no orthogonal subspace; x is determined by the dot constraint.
//     if (d == 1) {
//         if (K == 1) {
//             inst.x[0][0] = 1.0; // dot = ||theta|| (since theta_norm = |theta|)
//             return inst;
//         }
//         for (int i = 0; i < K; ++i) {
//             const double t = (double)i / (double)(K - 1);
//             const double r = (1.0 - 2.0 * t) * r_max;
//             const double a = r / theta_norm;
//             inst.x[i][0] = a;
//         }
//         return inst;
//     }

//     // d >= 2
//     for (int i = 0; i < K; ++i) {
//         const double t = (K == 1) ? 0.0 : (double)i / (double)(K - 1);
//         const double r = (1.0 - 2.0 * t) * r_max;
//         double a = r / theta_norm;
//         if (a >  1.0) a =  1.0;
//         if (a < -1.0) a = -1.0;

//         // Sample a random direction v in the orthogonal complement of u
//         Vec v(d);
//         bool ok = false;
        
//         for (int tries = 0; tries < 200; ++tries) {
//             Vec z = rng.random_ball_vec(d);                   // random in unit ball
//             // Project z onto orthogonal complement: z_perp = z - u*(u^T z)
//             double proj = dot(z, u);
//             for (int j = 0; j < d; ++j) z[j] -= proj * u[j];

//             double zn2 = dot(z, z);
//             if (zn2 > 1e-12) {
//                 double inv = 1.0 / std::sqrt(zn2);
//                 for (int j = 0; j < d; ++j) v[j] = z[j] * inv; // normalize to unit
//                 ok = true;
//                 break;
//             }
//         }
//         if (!ok) {
//             int p = 0;
//             for (int j = 1; j < d; ++j) if (std::fabs(u[j]) < std::fabs(u[p])) p = j;
//             int q = (p == 0 ? 1 : 0);
//             for (int j = 0; j < d; ++j) v[j] = 0.0;
//             v[p] =  u[q];
//             v[q] = -u[p];
//             double vn2 = dot(v, v);
//             double inv = 1.0 / std::sqrt(vn2);
//             for (int j = 0; j < d; ++j) v[j] *= inv;
//         }

//         // Choose b so that ||x|| = 1 and v is orthogonal to u:
//         // x = a u + b v => ||x||^2 = a^2 + b^2, so set b = sqrt(1-a^2).
//         const double b = std::sqrt(std::max(0.0, 1.0 - a * a));

//         Vec xi(d);
//         for (int j = 0; j < d; ++j) {
//             xi[j] = a * u[j] + b * v[j];
//         }
//         inst.x[i] = xi;
//     }

//     return inst;
// }


Vec unit_vec(int d, int i) {
    Vec ret(d, 0.);
    ret[i] = 1;
    return ret;
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

    // theta_star: 全维相等（与你给的一致）
    double val = (S - 1.) / std::sqrt((double)d);
    for (int j = 0; j < d; ++j) inst.theta_star[j] = val;

    // 计算 ||theta||^2 与 ||theta||
    double theta_norm2 = 0.0;
    for (int j = 0; j < d; ++j) {
        theta_norm2 += inst.theta_star[j] * inst.theta_star[j];
    }
    double theta_norm = std::sqrt(theta_norm2);

    // 退化保护
    if (theta_norm < 1e-12) {
        for (int i = 0; i < K; ++i) inst.x[i] = rng.random_ball_vec(d);
        return inst;
    }

    // rho 越接近 1，覆盖越宽，但正交空间越小；一般 0.8~0.95
    const double rho = 0.8;
    const double u_max = rho * theta_norm;

    // 生成 u_i 在 [0, u_max] 上等距（更“均匀”）
    std::vector<double> u(K);
    if (K == 1) {
        u[0] = 0.0;
    } else {
        u[0] = 0.9 * theta_norm;
        for (int i = 1; i < K; ++i) {
            double t = (double)(K - i - 1) / (double)(K - 1);  // in [0,1]
            u[i] = u_max * t;  // in [0, u_max]
        }
    }

    // 构造每个 x_i，使得 x_i^T theta_star = u_i 且 ||x_i|| <= 1
    for (int i = 0; i < K; ++i) {
        // 平行分量 alpha * theta
        double alpha = u[i] / theta_norm2;
        Vec x_par(d);
        for (int j = 0; j < d; ++j) x_par[j] = alpha * inst.theta_star[j];

        // 剩余可用半径（用于正交分量）
        double par_norm2 = 0.0;
        for (int j = 0; j < d; ++j) par_norm2 += x_par[j] * x_par[j];
        double ortho_radius = 0.0;
        if (par_norm2 < 1.0) ortho_radius = std::sqrt(1.0 - par_norm2);

        // 取一个随机向量，并投影到 theta 的正交子空间
        Vec v = rng.random_ball_vec(d); // 用你现成的接口，避免依赖 normal
        double vTtheta = 0.0;
        for (int j = 0; j < d; ++j) vTtheta += v[j] * inst.theta_star[j];
        double coeff = vTtheta / theta_norm2;
        for (int j = 0; j < d; ++j) v[j] -= coeff * inst.theta_star[j];

        // 归一化 v，并缩放到 ortho_radius（填满剩余半径）
        double v_norm2 = 0.0;
        for (int j = 0; j < d; ++j) v_norm2 += v[j] * v[j];
        double v_norm = std::sqrt(v_norm2);

        Vec x = x_par;
        if (v_norm > 1e-12 && ortho_radius > 0.0) {
            double scale = ortho_radius;
            for (int j = 0; j < d; ++j) x[j] += (scale / v_norm) * v[j];
        }

        // 数值安全：若略超出单位球，归一化
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



