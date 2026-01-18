#pragma once
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>

// Reuse your project infrastructure (Instance, RNG, lin_alg, logistic link, Obs + constrained MLE,
// and sampling routine sample_classic_reward).
#include "hybrid_bai.h"   // includes: lin_alg.h, rng.h, instance.h, mle.h, glm_logistic.h

// ============================================================================
// GLGapE baseline (Kazerouni & Wein style) for Logistic GLM bandits
// - Classic feedback only (Bernoulli-logistic rewards)
// - Uses constrained MLE from mle.h (projected Newton + backtracking)
// - Avoids explicit matrix inverse: all quadratic forms use solve_spd_cholesky
// - Select-gap + Select-arm (L1 LP) + tracking
// ============================================================================

// -------- Simplex LP for min ||w||_1 s.t. X w = y --------
// Convert to: min sum u_i
// s.t.  X (w^+ - w^-) = y,  w^+, w^- >= 0
// objective: sum (w^+ + w^-)
// variables: z = [w^+ (K), w^- (K)] >= 0
// constraints: A z = y, where A = [X, -X] of size d x (2K)
struct SimplexEq {
    // minimize c^T x, subject to A x = b, x>=0
    // Big-M with artificials (sufficient for small d,K).
    int m=0, n=0;
    std::vector<std::vector<double>> A; // m x n
    std::vector<double> b;              // m
    std::vector<double> c;              // n
    std::vector<double> x;              // n solution

    bool solve(double M = 1e6, int maxit = 20000, double eps = 1e-10) {
        // Tableau with artificials: n + m variables
        int N = n + m;
        std::vector<std::vector<double>> T(m + 1, std::vector<double>(N + 1, 0.0));
        std::vector<int> basis(m, -1);

        for (int i = 0; i < m; ++i) {
            double bi = b[i];
            if (bi < 0) {
                bi = -bi;
                for (int j = 0; j < n; ++j) T[i][j] = -A[i][j];
            } else {
                for (int j = 0; j < n; ++j) T[i][j] = A[i][j];
            }
            T[i][n + i] = 1.0;  // artificial
            T[i][N] = bi;
            basis[i] = n + i;
        }

        // objective row: original + M * artificials
        for (int j = 0; j < n; ++j) T[m][j] = c[j];
        for (int j = n; j < N; ++j) T[m][j] = M;

        // canonicalize w.r.t. artificial basis
        for (int i = 0; i < m; ++i) {
            double coeff = T[m][basis[i]];
            if (std::fabs(coeff) > 0) {
                for (int j = 0; j <= N; ++j) T[m][j] -= coeff * T[i][j];
            }
        }

        auto pivot = [&](int r, int s) {
            double piv = T[r][s];
            if (std::fabs(piv) < eps) return;
            for (int j = 0; j <= N; ++j) T[r][j] /= piv;
            for (int i = 0; i <= m; ++i) if (i != r) {
                double f = T[i][s];
                if (std::fabs(f) > eps) {
                    for (int j = 0; j <= N; ++j) T[i][j] -= f * T[r][j];
                }
            }
            basis[r] = s;
        };

        for (int it = 0; it < maxit; ++it) {
            // entering: most negative reduced cost
            int s = -1;
            double best = -eps;
            for (int j = 0; j < N; ++j) {
                if (T[m][j] < best) { best = T[m][j]; s = j; }
            }
            if (s < 0) break; // optimal

            // leaving: min ratio
            int r = -1;
            double minratio = std::numeric_limits<double>::infinity();
            for (int i = 0; i < m; ++i) {
                double a = T[i][s];
                if (a > eps) {
                    double ratio = T[i][N] / a;
                    if (ratio < minratio) { minratio = ratio; r = i; }
                }
            }
            if (r < 0) return false; // unbounded

            pivot(r, s);
        }

        // extract solution for original vars
        x.assign(n, 0.0);
        for (int i = 0; i < m; ++i) {
            int var = basis[i];
            if (var >= 0 && var < n) x[var] = T[i][N];
        }

        // check artificials near zero
        for (int i = 0; i < m; ++i) {
            int var = basis[i];
            if (var >= n) {
                if (std::fabs(T[i][N]) > 1e-6) return false;
            }
        }
        return true;
    }
};

inline void solve_l1_min_w(
    const Instance& inst,
    const Vec& y,
    std::vector<double>& w_out
) {
    int K = inst.K;
    int d = inst.d;

    SimplexEq lp;
    lp.m = d;
    lp.n = 2 * K;
    lp.A.assign(d, std::vector<double>(2 * K, 0.0));
    lp.b.assign(d, 0.0);
    lp.c.assign(2 * K, 1.0);

    for (int i = 0; i < d; ++i) lp.b[i] = y[i];

    for (int a = 0; a < K; ++a) {
        for (int i = 0; i < d; ++i) {
            lp.A[i][a]     = inst.x[a][i];   // w^+
            lp.A[i][K + a] = -inst.x[a][i];  // w^-
        }
    }

    bool ok = lp.solve();
    w_out.assign(K, 0.0);
    if (!ok) return;

    for (int a = 0; a < K; ++a) {
        double wp = lp.x[a];
        double wm = lp.x[K + a];
        w_out[a] = wp - wm;
    }
}

// -------------------- Baseline config/result --------------------

struct GLGapEConfig {
    // stopping tolerance (paper's ε)
    double eps = 0.1;

    // confidence
    double delta = 0.05;

    // warm-up length E; if <0: E = min(K, 3d)
    int E = -1;

    double alpha = -1.0;

    // logistic has k_mu=1/4; c_mu depends on bounded |x^T theta|
    double c_mu = 1e-3;
    double k_mu = 0.25;

    // optional: downscale Ct in practice
    bool downscale_C = false;
    double C_scale = 1.0;

    // numerical ridge for M (design matrix)
    double ridge = 1e-6;

    // max total pulls to prevent infinite loops
    int max_steps = 200000;

    // constrained MLE config (reusing mle.h)
    MLEConfig mle_cfg;
};

struct GLGapEResult {
    int hat_arm = -1;
    int stop_t  = 0;     // number of classic pulls
    bool correct = false;
};

// -------------------- Internal helpers --------------------

inline void mat_add_inplace(Mat& A, const Mat& B) {
    if (A.n != B.n) throw std::runtime_error("mat_add_inplace dim mismatch");
    for (int i = 0; i < A.n * A.n; ++i) A.a[(size_t)i] += B.a[(size_t)i];
}

inline Mat compute_M_from_data(const Instance& inst, const std::vector<Obs>& data, double ridge) {
    int d = inst.d;
    Mat M(d, 0.0);
    for (const auto& ob : data) {
        if (ob.is_duel) continue; // baseline uses classic only
        mat_add_inplace(M, outer(inst.x[ob.i]));
    }
    for (int i = 0; i < d; ++i) M(i,i) += ridge;
    return M;
}

inline double quadform_Minv(const Mat& M, const Vec& y) {
    Vec x = solve_spd_cholesky(M, y);   // x = M^{-1} y
    return dot(y, x);
}

inline double Ct_value(int t, const GLGapEConfig& cfg, int d) {
    // Conservative, monotone Ct(t). (Matches typical GLM-BAI choices.)
    const double pi = 3.14159265358979323846;
    double alpha = (cfg.alpha < 0.0) ? 1.0 : cfg.alpha;

    int tt = std::max(2, t);
    double inside = (pi * pi * (double)d * (double)tt * (double)tt) / (6.0 * cfg.delta);
    double val = std::sqrt(std::max(0.0, 2.0 * (double)d * std::log((double)tt) * std::log(inside)));
    double C = alpha * val;
    if (cfg.downscale_C) C *= cfg.C_scale;
    return C;
}

// Return predicted best arm under theta_hat (by mean, which is monotone in x^T theta for logistic)
inline int argmax_mean(const Instance& inst, const Vec& theta_hat) {
    int best = 0;
    double bestz = dot(inst.x[0], theta_hat);
    for (int i = 1; i < inst.K; ++i) {
        double z = dot(inst.x[i], theta_hat);
        if (z > bestz) { bestz = z; best = i; }
    }
    return best;
}

// Compute beta(i,j) and the maximizer y = c x_i - c' x_j where (c,c') in {c_mu,k_mu}^2
inline void beta_and_y(
    const Instance& inst,
    const Vec& theta_hat,
    const Mat& M,
    const GLGapEConfig& cfg,
    int i, int j,
    double Ct,
    double& beta_out,
    Vec& y_out
) {
    (void)theta_hat; // not needed for corner-max version (depends only on bounds)

    double corners[2] = {cfg.c_mu, cfg.k_mu};
    double best = -1.0;
    Vec besty(inst.d, 0.0);

    for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) {
        double c1 = corners[a];
        double c2 = corners[b];

        // y = c1*x_i - c2*x_j, using only left scalar multiplication (lin_alg style)
        Vec y = (c1 * inst.x[i]) + ((-c2) * inst.x[j]);

        double n2 = quadform_Minv(M, y); // y^T M^{-1} y
        if (n2 > best) { best = n2; besty = y; }
    }

    y_out = besty;
    beta_out = Ct * std::sqrt(std::max(0.0, best));
}

// -------------------- Main baseline runner --------------------
// Uses only classic pulls; dueling is ignored for this baseline.
inline GLGapEResult run_glgape_baseline(
    const Instance& inst,
    const GLGapEConfig& cfg,
    RNG& rng
) {
    int K = inst.K;
    int d = inst.d;

    int E = cfg.E;
    if (E < 0) E = std::min(K, 3 * d);
    E = std::max(1, E);

    // data and counts
    std::vector<Obs> data;
    data.reserve((size_t)cfg.max_steps + 10);
    std::vector<int> T(K, 0);

    // warm-up: random arms
    for (int t = 0; t < E && (int)data.size() < cfg.max_steps; ++t) {
        int a = rng.randint(0, K - 1);
        int r01 = sample_classic_reward(inst, a, rng);

        Obs ob;
        ob.is_duel = false;
        ob.i = a;
        ob.r01 = r01;
        ob.feat = inst.x[a];
        data.push_back(ob);

        T[a] += 1;
    }

    std::vector<double> w;


    for (int t = (int)data.size() + 1; t <= cfg.max_steps; ++t) {
        if (t % 500 == 0) {
            std::cout << "Step: " << t << "\n";
        }

        Vec theta_hat = constrained_mle_logistic(
            data, d, inst.S,
            1.0, 1.0,
            cfg.mle_cfg
        );

        Mat M = compute_M_from_data(inst, data, cfg.ridge);

        int it = argmax_mean(inst, theta_hat);

        double Ct = Ct_value(t, cfg, d);

        int jt = -1;
        double Bt = -std::numeric_limits<double>::infinity();
        Vec y_t(d, 0.0);

        double zi = dot(inst.x[it], theta_hat);
        double mui = mu(zi);

        for (int j = 0; j < K; ++j) if (j != it) {
            double zj = dot(inst.x[j], theta_hat);
            double muj = mu(zj);
            double delta_hat = muj - mui;
            double beta_ij = 0.0;
            Vec ytmp(d, 0.0);
            beta_and_y(inst, theta_hat, M, cfg, it, j, Ct, beta_ij, ytmp);

            double Bcand = delta_hat + beta_ij;
            if (Bcand > Bt) { 
                Bt = Bcand; 
                jt = j; 
                y_t = ytmp; 
            }
        }
        

        if (jt < 0) {
            // fallback: should not happen
            GLGapEResult res;
            res.hat_arm = it;
            res.stop_t = (int)data.size();
            res.correct = (res.hat_arm == inst.true_best_arm());
            return res;
        }

        // 4) stopping rule
        if (Bt <= cfg.eps) {
            GLGapEResult res;
            res.hat_arm = it;
            res.stop_t = (int)data.size();
            res.correct = (res.hat_arm == inst.true_best_arm());
            return res;
        }

        // 5) select-arm: solve min ||w||_1 s.t. X w = y_t, convert to p, tracking
        solve_l1_min_w(inst, y_t, w);

        std::vector<double> p(K, 0.0);
        double s = 0.0;
        for (int a = 0; a < K; ++a) {
            s += std::fabs(w[a]);
        }

        int a_pick = -1;
        if (s <= 1e-12) {
            a_pick = rng.randint(0, K - 1);
        } else {
            for (int a = 0; a < K; ++a) p[a] = std::fabs(w[a]) / s;

            double best_ratio = std::numeric_limits<double>::infinity();
            for (int a = 0; a < K; ++a) if (p[a] > 1e-12) {
                double ratio = (double)T[a] / p[a];
                if (ratio < best_ratio) {
                    best_ratio = ratio;
                    a_pick = a;
                }
            }
            if (a_pick < 0) {
                a_pick = rng.randint(0, K - 1);
            }
        }

        int r01 = sample_classic_reward(inst, a_pick, rng);

        Obs ob;
        ob.is_duel = false;
        ob.i = a_pick;
        ob.r01 = r01;
        ob.feat = inst.x[a_pick];
        data.push_back(ob);

        T[a_pick] += 1;
    }

    GLGapEResult res;
    Vec theta_hat = constrained_mle_logistic(
        data, d, inst.S, 1.0, 1.0, cfg.mle_cfg
    );
    res.hat_arm = argmax_mean(inst, theta_hat);
    res.stop_t = (int)data.size();
    res.correct = (res.hat_arm == inst.true_best_arm());
    return res;
}
