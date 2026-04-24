#pragma  once
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>

#include "env.h"
#include "rng.h"
#include "instance.h"
#include "mle.h"

struct SimplexEq {
    int m=0, n=0;
    std::vector<std::vector<double>> A;
    std::vector<double> b;  
    std::vector<double> c; 
    std::vector<double> x; 

    bool solve(double M = 1e6, int maxit = 20000, double eps = 1e-6) {
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
            T[i][n + i] = 1.0;
            T[i][N] = bi;
            basis[i] = n + i;
        }

        for (int j = 0; j < n; ++j) T[m][j] = c[j];
        for (int j = n; j < N; ++j) T[m][j] = M;

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
            int s = -1;
            double best = -eps;
            for (int j = 0; j < N; ++j) {
                if (T[m][j] < best) { best = T[m][j]; s = j; }
            }
            if (s < 0) break;

            int r = -1;
            double minratio = std::numeric_limits<double>::infinity();
            for (int i = 0; i < m; ++i) {
                double a = T[i][s];
                if (a > eps) {
                    double ratio = T[i][N] / a;
                    if (ratio < minratio) { minratio = ratio; r = i; }
                }
            }
            if (r < 0) return false;

            pivot(r, s);
        }

        x.assign(n, 0.0);
        for (int i = 0; i < m; ++i) {
            int var = basis[i];
            if (var >= 0 && var < n) x[var] = T[i][N];
        }

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
            lp.A[i][a]     = inst.x[a][i];
            lp.A[i][K + a] = -inst.x[a][i];
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


struct GLGapEConfig {
    double eps = 0.1;

    double delta = 0.05;

    int E = -1;

    double kappa = -1.0;

    double alpha = -1.0;

    double c_mu = 1e-3;
    double k_mu = 0.25;

    bool downscale_C = false;
    double C_scale = 1.0;

    double ridge = 1e-6;

    int max_steps = 200000;

    MLEConfig mle_cfg;
};

struct GLGapEResult {
    int hat_arm = -1;
    int stop_t  = 0;  
    bool correct = false;
};


#pragma once
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include "lin_alg.h"

inline double lambda_max_spd_power(const Mat& M, int max_iter = 500, double tol = 1e-12) {
    const int n = M.n;
    if (n <= 0) throw std::runtime_error("lambda_max_spd_power: empty matrix");
    Vec v(n, 0.0);
    for (int i = 0; i < n; ++i) v[i] = 1.0 / std::sqrt((double)n);

    double lambda = 0.0;
    for (int it = 0; it < max_iter; ++it) {
        Vec w = mat_vec(M, v);
        double nw = w.norm2();
        if (nw <= 1e-18) throw std::runtime_error("lambda_max_spd_power: numerical breakdown");
        for (int i = 0; i < n; ++i) w[i] /= nw;

        Vec Mw = mat_vec(M, w);
        double lambda_new = dot(w, Mw);

        if (std::fabs(lambda_new - lambda) <= tol * std::max(1.0, std::fabs(lambda_new))) {
            lambda = lambda_new;
            break;
        }
        lambda = lambda_new;
        v = w;
    }
    return lambda;
}

inline double lambda_min_spd_inv_power(const Mat& M, int max_iter = 500, double tol = 1e-12) {
    const int n = M.n;
    if (n <= 0) throw std::runtime_error("lambda_min_spd_inv_power: empty matrix");

    Vec v(n, 0.0);
    for (int i = 0; i < n; ++i) v[i] = 1.0 / std::sqrt((double)n);

    double lambda_min = 0.0;
    for (int it = 0; it < max_iter; ++it) {
        Vec w = solve_spd_cholesky(M, v);

        double nw = w.norm2();
        if (nw <= 1e-18) {
            throw std::runtime_error("lambda_min_spd_inv_power: numerical breakdown (nw ~ 0)");
        }
        for (int i = 0; i < n; ++i) w[i] /= nw;

        Vec Mw = mat_vec(M, w);
        double lambda_new = dot(w, Mw);

        if (lambda_new <= 0.0) {
            throw std::runtime_error("lambda_min_spd_inv_power: M not SPD (lambda_new <= 0)");
        }

        if (std::fabs(lambda_new - lambda_min) <= tol * std::max(1.0, std::fabs(lambda_new))) {
            lambda_min = lambda_new;
            break;
        }
        lambda_min = lambda_new;
        v = w;
    }

    if (lambda_min <= 0.0) throw std::runtime_error("lambda_min_spd_inv_power: failed (lambda_min<=0)");
    return lambda_min;
}

inline double compute_kappa_L1_from_M(const Mat& M) {
    double lambda0 = lambda_min_spd_inv_power(M);
    if (lambda0 <= 0.0) throw std::runtime_error("compute_kappa: lambda0 must be positive");

    return std::sqrt(3.0 + 2.0 * std::log(1.0 + 2.0 / lambda0));
}

inline void mat_add_inplace(Mat& A, const Mat& B, int cnt = 1) {
    if (A.n != B.n) throw std::runtime_error("mat_add_inplace dim mismatch");
    for (int i = 0; i < A.n * A.n; ++i) {
        A.a[(size_t)i] += B.a[(size_t)i] * cnt;
    }
}

inline Mat compute_M_from_data(const Instance& inst, double ridge, const std::vector<int>& T) {
    int d = inst.d;
    Mat M(d, 0.0);

    for (int i = 0; i < inst.K; ++i) {
        mat_add_inplace(M, outer(inst.x[i]), T[i]);
    }

    for (int i = 0; i < d; ++i) M(i,i) += ridge;
    return M;
}

inline double quadform_Minv(const Mat& M, const Vec& y) {
    Vec x = solve_spd_cholesky(M, y); 
    return dot(y, x);
}

const double pi = 3.14159265358979323846;

inline double Ct_value(int t, const GLGapEConfig& cfg, int d) {
    int tt = std::max(2, t);
    double inside = (pi * pi * (double)d * (double)tt * (double)tt) / (6.0 * cfg.delta);
    double val = std::sqrt(std::max(0.0, 2.0 * (double)d * std::log((double)tt) * std::log(inside)));
    double C = cfg.alpha * val;
    if (cfg.downscale_C) C *= cfg.C_scale;
    return C;
}

inline int argmax_mean(const Instance& inst, const Vec& theta_hat) {
    int best = 0;
    double bestz = dot(inst.x[0], theta_hat);
    for (int i = 1; i < inst.K; ++i) {
        double z = dot(inst.x[i], theta_hat);
        if (z > bestz) { bestz = z; best = i; }
    }
    return best;
}

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
    (void)theta_hat;

    double corners[2] = {cfg.c_mu, cfg.k_mu};
    double best = -1.0;
    Vec besty(inst.d, 0.0);

    for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) {
        double c1 = corners[a];
        double c2 = corners[b];

        Vec y = (c1 * inst.x[i]) + ((-c2) * inst.x[j]);

        double n2 = quadform_Minv(M, y);  
        if (n2 > best) { best = n2; besty = y; }
    }

    y_out = besty;
    beta_out = Ct * std::sqrt(std::max(0.0, best));
}

inline GLGapEResult run_glgape_baseline(
    Instance& inst,
    GLGapEConfig& cfg,
    RNG& rng
) {
    int K = inst.K;
    int d = inst.d;

    int E = cfg.E;
    if (E < 0) E = std::min(K, 3 * d);
    E = std::max(1, E);

    std::vector<int> T(K, 0);


    std::vector<std::vector<int>> r01s(inst.K, std::vector<int> (2, 0));
    std::vector<std::vector<std::vector<int>>> y01s(inst.K, std::vector<std::vector<int>> (inst.K, std::vector<int>(2, 0)));

    for (int t = 0; t < E && t < cfg.max_steps; ++t) {
        int arm = t % K;

        r01s[arm][sample_reward(inst, arm, rng)]++;
        T[arm] += 1;
    }

    Mat M = compute_M_from_data(inst, cfg.ridge, T);
    cfg.kappa = compute_kappa_L1_from_M(M);
    cfg.alpha = 2 * cfg.kappa * 1 / cfg.c_mu;

    std::vector<double> w;

    Vec theta_hat(inst.d, 0.0);

    

    int t;
    for (t = E + 1; t <= cfg.max_steps; ++t) {
        theta_hat = constrained_mle_logistic(
            r01s, y01s, d, inst.S,
            1.0, 1.0,
            cfg.mle_cfg,
            theta_hat,
            inst
        );

        M = compute_M_from_data(inst, cfg.ridge, T);

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
            GLGapEResult res;
            res.hat_arm = it;
            res.stop_t = t;
            res.correct = (res.hat_arm == inst.true_best_arm());
            return res;
        }

        if (Bt <= cfg.eps) {
            GLGapEResult res;
            res.hat_arm = it;
            res.stop_t = t;
            res.correct = (res.hat_arm == inst.true_best_arm());
            return res;
        }

        solve_l1_min_w(inst, y_t, w);

        std::vector<double> p(K, 0.0);
        double s = 0.0;
        for (int a = 0; a < K; ++a) {
            s += std::fabs(w[a]);
        }

        int arm = -1;
        if (s <= 1e-12) {
            arm = rng.randint(0, K - 1);
        } else {
            for (int a = 0; a < K; ++a) p[a] = std::fabs(w[a]) / s;

            double best_ratio = std::numeric_limits<double>::infinity();
            for (int a = 0; a < K; ++a) if (p[a] > 1e-12) {
                double ratio = (double)T[a] / p[a];
                if (ratio < best_ratio) {
                    best_ratio = ratio;
                    arm = a;
                }
            }
            if (arm < 0) {
                arm = rng.randint(0, K - 1);
            }
        }

        r01s[arm][sample_reward(inst, arm, rng)]++;
        T[arm] += 1;
    }

    GLGapEResult res;
    res.hat_arm = argmax_mean(inst, theta_hat);
    res.stop_t = t;
    res.correct = (res.hat_arm == inst.true_best_arm());
    return res;
}
