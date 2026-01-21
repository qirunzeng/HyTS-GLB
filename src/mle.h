#pragma once
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include "lin_alg.h"
#include "env.h"

// One observation can be reward or dueling.
// struct Obs {
//     bool is_duel = false;

//     int i   = -1;
//     int r01 =  0;

//     int j = -1, k = -1;
//     int y01 = 0;

//     Vec &feat; // x or g

//     Obs(int i, int r, Vec &x) : i(i), r01(r), feat(x) {}
//     Obs(int j, int k, int y, Vec &g) : j(j), k(k), y01(y), feat(g) {
//         is_duel = true;
//     }
// };

struct MLEConfig {
    int max_iter            = 50;
    double tol              = 1e-8;
    double step_backtrack   = 0.5;
    double min_step         = 1e-6;
};

inline Vec project_to_ball(const Vec& v, double S) {
    double n = v.norm2();
    if (n <= S) return v;
    if (n < 1e-12) return v;
    return (S / n) * v;
}

// Compute total negative log-likelihood, gradient, Hessian for logistic model over observations:
// For reward: z = x^T theta, r in {0,1}
// For duel:   z  = g^T theta, y in {0,1}
inline double total_nll_grad_hess(
    const std::vector<std::vector<int>>& r01s,
    const std::vector<std::vector<std::vector<int>>> y01s,
    const Vec& theta,
    Vec& grad,
    Mat& hess,
    double zeta_c,
    double zeta_d,
    const Instance& inst
) {
    int d = theta.dim();
    grad = Vec(d, 0.0);
    hess = Mat(d, 0.0);


    double loss = 0.0;
    for (int i = 0; i < inst.K; ++i) {
        double z = dot(inst.x[i], theta);
        double p = sigmoid(z);
        double w = mu_prime(z);
        for (int r = 0; r < 2; ++r) {
            loss += (1.0 / zeta_c) * nll_logistic(z, r) * r01s[i][r];
            double coef = (1.0 / zeta_c) * (p - (double)r);
            for (int j = 0; j < d;++j) {
                grad[j] += coef * inst.x[i][j] * r01s[i][r];
            }
        }
        hess = hess + (1.0 * (r01s[i][0] + r01s[i][1]) * (1.0 / zeta_c) * w) * outer(inst.x[i]);
    }

    for (int j = 0; j < inst.K; ++j) {
        for (int k = j+1; k < inst.K; ++k) {
            double z = dot(inst.g[j][k], theta);
            double p = sigmoid(z);
            double w = mu_prime(z);
            for (int r = 0; r < 2; ++r) {
                loss += (1.0 / zeta_d) * nll_logistic(z, r) * y01s[j][k][r];
                double coef = (1.0 / zeta_d) * (p - (double)r);
                for (int l = 0; l < d; ++l) {
                    grad[l] += coef * inst.g[j][k][l] * y01s[j][k][r];
                }
            }
            hess = hess + (1.0 * (y01s[j][k][0] + y01s[j][k][1]) * (1.0 / zeta_d) * w) * outer(inst.g[j][k]);
        }
    }
    return loss;
}

// Projected Newton with backtracking line search for constrained MLE over ||theta||<=S.
inline Vec constrained_mle_logistic(
    const std::vector<std::vector<int>>& r01s,
    const std::vector<std::vector<std::vector<int>>> y01s,
    int d,
    double S,
    double zeta_c,
    double zeta_d,
    const MLEConfig& cfg,
    Vec theta,
    const Instance& inst
) {
    double prev = 1e100;

    for (int it = 0; it < cfg.max_iter; ++it) {
        Vec g(d);
        Mat H(d);
        
        double f = total_nll_grad_hess(r01s, y01s, theta, g, H, zeta_c, zeta_d, inst);
        
        double gnorm = g.norm2();
        if (gnorm < cfg.tol) break;
        if (std::fabs(prev - f) < cfg.tol * (1.0 + std::fabs(f))) break;
        prev = f;

        // Newton direction: p = - H^{-1} g

        double reg = 1e-3; // can be tuned or made adaptive
        Mat Hreg = H;
        for (int j = 0; j < d; ++j) Hreg(j,j) += reg;

        Vec negg(d);
        for (int j = 0; j < d; ++j) negg[j] = -g[j];

        Vec p = solve_spd_cholesky(Hreg, negg);

        // backtracking line search with projection
        double step = 1.0;
        bool accepted=false;
        for (;;) {
            Vec cand = theta + step * p;
            cand = project_to_ball(cand, S);

            Vec cg(d);
            Mat cH(d);
            double cf = total_nll_grad_hess(r01s, y01s, cand, cg, cH, zeta_c, zeta_d, inst);

            // Armijo condition (simple)
            if (cf <= f - 1e-4 * step * gnorm * gnorm) {
                theta = cand;
                accepted=true;
                break;
            }
            step *= cfg.step_backtrack;
            if (step < cfg.min_step) break;
        }
        if (!accepted) {
            // fallback: small gradient step + projection
            double eta = 0.1 / (1.0 + it);
            Vec cand(d);
            for (int j = 0; j < d; ++j) cand[j] = theta[j] - eta * g[j];
            theta = project_to_ball(cand, S);
        }
    }
    return theta;
}
