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
    const int d = theta.dim();

    // 复位（如果 Vec/Mat 内部是 std::vector，尽量保证它们已分配好容量）
    grad = Vec(d, 0.0);
    hess = Mat(d, 0.0);

    const double inv_zc = 1.0 / zeta_c;
    const double inv_zd = 1.0 / zeta_d;

    double loss = 0.0;

    // ---- reward 部分 ----
    for (int i = 0; i < inst.K; ++i) {
        const int n0 = r01s[i][0];
        const int n1 = r01s[i][1];
        const int n  = n0 + n1;
        if (n == 0) continue;

        const double z = dot(inst.x[i], theta);
        const double p = sigmoid(z);
        const double w = mu_prime(z);

        // loss
        loss += inv_zc * ( (double)n1 * nll_logistic(z, 1) + (double)n0 * nll_logistic(z, 0) );

        // grad: (n*p - n1) * x
        const double gcoef = inv_zc * ( (double)n * p - (double)n1 );
        for (int j = 0; j < d; ++j) {
            grad[j] += gcoef * inst.x[i][j];
        }

        // hess: n * w * xx^T
        const double hcoef = inv_zc * (double)n * w;
        for (int a = 0; a < d; ++a) {
            const double xa = inst.x[i][a];
            for (int b = 0; b < d; ++b) {
                hess(a,b) += hcoef * xa * inst.x[i][b];
            }
        }
    }

    // ---- duel 部分 ----
    for (int j = 0; j < inst.K; ++j) {
        for (int k = j + 1; k < inst.K; ++k) {
            const int n0 = y01s[j][k][0];
            const int n1 = y01s[j][k][1];
            const int n  = n0 + n1;
            if (n == 0) continue;

            const double z = dot(inst.g[j][k], theta);
            const double p = sigmoid(z);
            const double w = mu_prime(z);

            loss += inv_zd * ( (double)n1 * nll_logistic(z, 1) + (double)n0 * nll_logistic(z, 0) );

            const double gcoef = inv_zd * ( (double)n * p - (double)n1 );
            for (int l = 0; l < d; ++l) {
                grad[l] += gcoef * inst.g[j][k][l];
            }

            const double hcoef = inv_zd * (double)n * w;
            for (int a = 0; a < d; ++a) {
                const double ga = inst.g[j][k][a];
                for (int b = 0; b < d; ++b) {
                    hess(a,b) += hcoef * ga * inst.g[j][k][b];
                }
            }
        }
    }

    return loss;
}

inline double total_nll_only(
    const std::vector<std::vector<int>>& r01s,
    const std::vector<std::vector<std::vector<int>>> y01s,
    const Vec& theta,
    double zeta_c,
    double zeta_d,
    const Instance& inst
) {
    const double inv_zc = 1.0 / zeta_c;
    const double inv_zd = 1.0 / zeta_d;

    double loss = 0.0;

    for (int i = 0; i < inst.K; ++i) {
        const int n0 = r01s[i][0];
        const int n1 = r01s[i][1];
        const int n  = n0 + n1;
        if (n == 0) continue;

        const double z = dot(inst.x[i], theta);
        loss += inv_zc * ( (double)n1 * nll_logistic(z, 1) + (double)n0 * nll_logistic(z, 0) );
    }

    for (int j = 0; j < inst.K; ++j) {
        for (int k = j + 1; k < inst.K; ++k) {
            const int n0 = y01s[j][k][0];
            const int n1 = y01s[j][k][1];
            const int n  = n0 + n1;
            if (n == 0) continue;

            const double z = dot(inst.g[j][k], theta);
            loss += inv_zd * ( (double)n1 * nll_logistic(z, 1) + (double)n0 * nll_logistic(z, 0) );
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

    // 复用缓冲区，避免每轮分配
    Vec g(d);
    Mat H(d);
    Vec negg(d);
    Vec p(d);
    Vec cand(d);

    for (int it = 0; it < cfg.max_iter; ++it) {
        const double f = total_nll_grad_hess(r01s, y01s, theta, g, H, zeta_c, zeta_d, inst);

        const double gnorm = g.norm2();
        if (gnorm < cfg.tol) break;
        if (std::fabs(prev - f) < cfg.tol * (1.0 + std::fabs(f))) break;
        prev = f;

        // Hreg = H + reg I（如果 Mat 复制很贵，可以改成 solve 内部加对角，但先保持简单）
        double reg = 1e-3;
        Mat Hreg = H;
        for (int j = 0; j < d; ++j) Hreg(j,j) += reg;

        for (int j = 0; j < d; ++j) negg[j] = -g[j];
        p = solve_spd_cholesky(Hreg, negg);

        // Armijo: g^T p
        double gTp = 0.0;
        for (int j = 0; j < d; ++j) gTp += g[j] * p[j];

        // 下降性保护
        if (gTp >= 0.0) {
            for (int j = 0; j < d; ++j) p[j] = -g[j];
            gTp = -gnorm * gnorm;
        }

        // backtracking line search (只算 NLL)
        double step = 1.0;
        bool accepted = false;
        const double c1 = 1e-4;

        for (;;) {
            // cand = project_to_ball(theta + step*p, S)
            for (int j = 0; j < d; ++j) cand[j] = theta[j] + step * p[j];
            cand = project_to_ball(cand, S);

            const double cf = total_nll_only(r01s, y01s, cand, zeta_c, zeta_d, inst);

            if (cf <= f + c1 * step * gTp) {
                theta = cand;
                accepted = true;
                break;
            }
            step *= cfg.step_backtrack;
            if (step < cfg.min_step) break;
        }

        if (!accepted) {
            // fallback: small gradient step + projection
            const double eta = 0.1 / (1.0 + it);
            for (int j = 0; j < d; ++j) cand[j] = theta[j] - eta * g[j];
            theta = project_to_ball(cand, S);
        }
    }

    return theta;
}
