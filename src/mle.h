#pragma once
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include "lin_alg.h"
#include "glm_logistic.h"

// One observation can be classic or dueling.
struct Obs {
    bool is_duel=false;

    // classic: arm i, reward r in {0,1}, feature x_i
    int i=-1;
    int r01=0;

    // duel: pair (j,k), y in {0,1}, feature g = x_j - x_k
    int j=-1, k=-1;
    int y01=0;

    Vec feat; // x or g
};

struct MLEConfig {
    int max_iter=50;
    double tol=1e-8;
    double step_backtrack=0.5;
    double min_step=1e-6;
};

inline Vec project_to_ball(const Vec& v, double S) {
    double n = v.norm2();
    if (n <= S) return v;
    if (n < 1e-12) return v;
    return (S / n) * v;
}

// Compute total negative log-likelihood, gradient, Hessian for logistic model over observations:
// For classic: z = x^T theta, r in {0,1}
// For duel:   z = g^T theta, y in {0,1}
inline double total_nll_grad_hess(
    const std::vector<Obs>& data,
    const Vec& theta,
    Vec& grad,
    Mat& hess,
    double zeta_c,
    double zeta_d
) {
    int d = theta.dim();
    grad = Vec(d, 0.0);
    hess = Mat(d, 0.0);

    double loss = 0.0;
    for (const auto& ob: data) {
        double z = dot(ob.feat, theta);
        double p = sigmoid(z);
        double w = mu_prime(z); // p(1-p)
        if (!ob.is_duel) {
            loss += (1.0 / zeta_c) * nll_logistic(z, ob.r01);
            // grad += (1/zeta_c) * (p - r) x
            double coef = (1.0 / zeta_c) * (p - (double)ob.r01);
            for (int j=0;j<d;++j) grad[j] += coef * ob.feat[j];
            // hess += (1/zeta_c) * w * x x^T
            Mat ox = outer(ob.feat);
            hess = hess + ( (1.0 / zeta_c) * w ) * ox;
        } else {
            loss += (1.0 / zeta_d) * nll_logistic(z, ob.y01);
            double coef = (1.0 / zeta_d) * (p - (double)ob.y01);
            for (int j=0;j<d;++j) grad[j] += coef * ob.feat[j];
            Mat og = outer(ob.feat);
            hess = hess + ( (1.0 / zeta_d) * w ) * og;
        }
    }
    return loss;
}

// Projected Newton with backtracking line search for constrained MLE over ||theta||<=S.
inline Vec constrained_mle_logistic(
    const std::vector<Obs>& data,
    int d,
    double S,
    double zeta_c,
    double zeta_d,
    const MLEConfig& cfg
) {
    Vec theta(d, 0.0); // start at 0
    double prev = 1e100;

    for (int it=0; it<cfg.max_iter; ++it) {
        Vec g(d);
        Mat H(d);
        double f = total_nll_grad_hess(data, theta, g, H, zeta_c, zeta_d);

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
            double cf = total_nll_grad_hess(data, cand, cg, cH, zeta_c, zeta_d);

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
            for (int j=0;j<d;++j) cand[j] = theta[j] - eta * g[j];
            theta = project_to_ball(cand, S);
        }
    }
    return theta;
}
