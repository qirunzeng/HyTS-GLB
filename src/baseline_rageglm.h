#pragma once
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <iostream>

#include "env.h"
#include "rng.h"
#include "instance.h"
#include "mle.h"
#include "lin_alg.h"
struct RAGEGLMConfig {
    double delta = 0.05;

    bool include_dueling_pairs_as_actions = false;
    bool include_reward = true;

    double eps_round = 0.10;

    int max_steps = 500000;

    int burnin_n = -1;

    int fw_iters = 100;

    double ridge = 1e-6;

    MLEConfig mle_cfg;
};

struct RAGEGLMResult {
    int hat_arm = -1;
    int stop_t = 0;
    bool correct = false;
    int rounds = 0;
};


struct ObsActionRef {
    bool is_duel = false;
    int a = -1;  
    int j = -1, k = -1; 
    const Vec* v = nullptr;
};

inline std::vector<ObsActionRef> build_all_actions(const Instance& inst, bool include_reward, bool include_duels) {
    std::vector<ObsActionRef> acts;
    const int K = inst.K;
    acts.reserve((size_t)K + (include_duels ? (size_t)K * (size_t)(K - 1) / 2 : 0));

    if (include_reward) {
        for (int a = 0; a < K; ++a) {
            ObsActionRef ar;
            ar.is_duel = false;
            ar.a = a;
            ar.v = &inst.x[a];
            acts.push_back(ar);
        }
    }
    if (include_duels) {
        for (int j = 0; j < K; ++j) {
            for (int k = j + 1; k < K; ++k) {
                ObsActionRef ar;
                ar.is_duel = true;
                ar.j = j;
                ar.k = k;
                ar.v = &inst.g[j][k];
                acts.push_back(ar);
            }
        }
    }
    return acts;
}

inline double gamma_d(int d, int teff, double delta) {
    return (double)d + std::log(6.0 * (2.0 + (double)teff) / std::max(1e-300, delta));
}

inline double max_arm_norm(const Instance& inst) {
    double L = 0.0;
    for (int a = 0; a < inst.K; ++a) {
        L = std::max(L, std::sqrt(dot(inst.x[a], inst.x[a])));
    }
    return L;
}

inline int argmax_z_all(const Instance& inst, const Vec& theta_hat) {
    int z = 0;
    double best = -1e300;
    for (int a = 0; a < inst.K; ++a) {
        double v = dot(inst.x[a], theta_hat);
        if (v > best) { best = v; z = a; }
    }
    return z;
}

inline int argmax_z_active(const Instance& inst, const Vec& theta_hat, const std::vector<int>& active) {
    int z = active[0];
    double best = -1e300;
    for (int a : active) {
        double v = dot(inst.x[a], theta_hat);
        if (v > best) { best = v; z = a; }
    }
    return z;
}

void epsilon_round_counts(
    const std::vector<double>& lambda_in,
    int n,
    double eps_round,
    std::vector<int>& out_counts
) {
    const int K = (int)lambda_in.size();
    out_counts.assign(K, 0);
    if (n <= 0 || K <= 0) return;

    std::vector<double> lambda(K, 0.0);
    double s = 0.0;
    for (int a = 0; a < K; ++a) {
        lambda[a] = std::max(0.0, lambda_in[a]);
        s += lambda[a];
    }
    if (s <= 1e-15) {
        int q = n / K;
        int r = n % K;
        for (int a = 0; a < K; ++a) out_counts[a] = q + (a < r ? 1 : 0);
        return;
    }
    for (int a = 0; a < K; ++a) lambda[a] /= s;

    struct Frac { int a; double frac; };
    std::vector<Frac> fracs;
    fracs.reserve((size_t)K);

    int used = 0;
    for (int a = 0; a < K; ++a) {
        const double raw = (double)n * lambda[a];
        const int base = (int)std::floor(raw);
        out_counts[a] = base;
        used += base;
        fracs.push_back({a, raw - (double)base});
    }

    int rem = n - used;
    if (rem <= 0) return;

    std::sort(fracs.begin(), fracs.end(), [](const Frac& u, const Frac& v) {
        return u.frac > v.frac;
    });
    for (int i = 0; i < rem; ++i) {
        out_counts[fracs[i % K].a] += 1;
    }

    int sum = 0;
    for (int c : out_counts) sum += c;
    if (sum != n) {
        int diff = n - sum;
        if (diff > 0) out_counts[0] += diff;
        else if (diff < 0) out_counts[0] = std::max(0, out_counts[0] + diff);
    }

    (void)eps_round; 
}




Mat fisher_matrix_from_lambda(
    const Instance& inst,
    const std::vector<ObsActionRef>& acts,
    const std::vector<double>& lambda,
    const Vec& theta,
    double ridge
) {
    const int d = inst.d;
    Mat H(d, 0.0);

    const int M = (int)acts.size();
    if ((int)lambda.size() != M) {
        throw std::runtime_error("fisher_matrix_from_lambda: lambda size mismatch with acts");
    }

    for (int m = 0; m < M; ++m) {
        const double lm = lambda[m];
        if (lm <= 0.0) continue;
        const Vec& x = *acts[m].v;
        const double z = dot(x, theta);
        const double w = mu_prime(z);
        for (int i = 0; i < d; ++i) {
            const double xi = x[i];
            for (int j = 0; j < d; ++j) {
                H(i, j) += lm * w * xi * x[j];
            }
        }
    }
    for (int i = 0; i < d; ++i) H(i, i) += ridge;
    return H;
}

Mat fisher_matrix_from_lambda_curv(
    int d,
    const std::vector<ObsActionRef>& acts,
    const std::vector<double>& lambda,
    const std::vector<double>& curv,
    double ridge
) {
    Mat H(d, 0.0);

    const int M = (int)acts.size();
    if ((int)lambda.size() != M || (int)curv.size() != M) {
        throw std::runtime_error("fisher_matrix_from_lambda_curv: size mismatch");
    }

    for (int m = 0; m < M; ++m) {
        const double lm = lambda[m];
        if (lm <= 0.0) continue;
        const Vec& x = *acts[m].v;
        const double coef = lm * curv[m];
        for (int i = 0; i < d; ++i) {
            const double xi = x[i];
            for (int j = 0; j < d; ++j) {
                H(i, j) += coef * xi * x[j];
            }
        }
    }
    for (int i = 0; i < d; ++i) H(i, i) += ridge;
    return H;
}


std::vector<double> approx_fw_design(
    const Instance& inst,
    const std::vector<ObsActionRef>& acts,
    const std::vector<Vec>& D,
    const Vec& theta_prev,
    int iters,
    double ridge
) {
    const int M = (int)acts.size();
    if (M <= 0) return {};

    std::vector<double> lambda(M, 1.0 / (double)M);
    std::vector<double> curv(M, 0.0);
    for (int m = 0; m < M; ++m) {
        curv[m] = mu_prime(dot(*acts[m].v, theta_prev));
    }

    for (int t = 0; t < iters; ++t) {
        Mat H = fisher_matrix_from_lambda_curv(inst.d, acts, lambda, curv, ridge);

        int worst_i = 0;
        double worstv = quad_form_inv_spd(H, D[0]);
        for (int i = 1; i < (int)D.size(); ++i) {
            const double v = quad_form_inv_spd(H, D[i]);
            if (v > worstv) { worstv = v; worst_i = i; }
        }
        const Vec& y = D[worst_i];
        Vec v = solve_spd_cholesky(H, y);

        int best_m = 0;
        double best_score = -1.0;
        for (int m = 0; m < M; ++m) {
            const Vec& x = *acts[m].v;
            const double ip = dot(x, v);
            const double score = curv[m] * ip * ip;
            if (score > best_score) { best_score = score; best_m = m; }
        }

        const double eta = 2.0 / (double)(t + 2);
        for (int m = 0; m < M; ++m) lambda[m] *= (1.0 - eta);
        lambda[best_m] += eta;
    }

    double s = 0.0;
    for (double v : lambda) s += std::max(0.0, v);
    if (s <= 1e-15) {
        std::fill(lambda.begin(), lambda.end(), 1.0 / (double)M);
    } else {
        for (double& v : lambda) v = std::max(0.0, v) / s;
    }
    return lambda;
}

std::vector<double> approx_burnin_design(
    const Instance& inst,
    const std::vector<ObsActionRef>& acts,
    int iters,
    double ridge
) {
    std::vector<Vec> D;
    D.reserve(acts.size());
    for (const ObsActionRef& act : acts) D.push_back(*act.v);
    Vec theta0(inst.d, 0.0);
    return approx_fw_design(inst, acts, D, theta0, iters, ridge);
}

std::vector<double> approx_rage_design(
    const Instance& inst,
    const std::vector<ObsActionRef>& acts,
    const std::vector<int>& active,
    int zhat,
    const Vec& theta_prev,
    int iters,
    double ridge
) {
    std::vector<Vec> D;
    D.reserve((size_t)inst.K + active.size());
    for (int a = 0; a < inst.K; ++a) D.push_back(inst.x[a]);
    for (int idx : active) {
        if (idx == zhat) continue;
        D.push_back(inst.x[zhat] - inst.x[idx]);
    }
    return approx_fw_design(inst, acts, D, theta_prev, iters, ridge);
}
RAGEGLMResult run_rageglm_baseline(
    Instance& inst,
    RAGEGLMConfig& cfg,
    RNG& rng
) {
    const int K = inst.K;
    const int d = inst.d;

    std::vector<std::vector<int>> r01s(K, std::vector<int>(2, 0));
    std::vector<std::vector<std::vector<int>>> y01s(K,
        std::vector<std::vector<int>>(K, std::vector<int>(2, 0))
    );

    const std::vector<ObsActionRef> acts = build_all_actions(inst, cfg.include_reward, cfg.include_dueling_pairs_as_actions);
    const int M = (int)acts.size();

    const double L = max_arm_norm(inst);
    const double kappa0 = mu_prime(L * inst.S);
    const double kappa0_inv = 1.0 / std::max(1e-12, kappa0);

    const int teff = M;
    const int z_size = K;
    const int x_size = M;
    const int union_size = std::max(z_size, x_size);
    const double gd = gamma_d(d, teff, cfg.delta);
    const int r_eps = (int)std::ceil(((double)d * (double)(d + 1) + 2.0) / std::max(1e-12, cfg.eps_round));

    int n0 = 0;
    if (cfg.burnin_n > 0) {
        n0 = cfg.burnin_n;
    } else {
        double inside;
        if (!cfg.include_reward) {
            inside = 2.0 * (double)(K * (K-1) / 2) * (2.0 + (double)(K * (K-1) / 2)) / std::max(1e-6, cfg.delta);
        } else if (!cfg.include_dueling_pairs_as_actions) {
            inside = 2.0 * (double)K * (2.0 + (double)K) / std::max(1e-6, cfg.delta);
        } else {
            inside = 2.0 * (double)(K + K * (K-1) / 2) * (2.0 + (double)(K + K * (K-1) / 2)) / std::max(1e-6, cfg.delta);
        }
        const double nn = 3.0 * (1.0 + cfg.eps_round) * kappa0_inv * (double)d * gd * std::log(inside);
        n0 = (int)std::ceil(std::max(nn, (double)r_eps));
    }

    int t = 0;

    std::vector<double> lambda0 = approx_burnin_design(inst, acts, cfg.fw_iters, cfg.ridge);
    std::vector<int> count0;
    epsilon_round_counts(lambda0, n0, cfg.eps_round, count0);


    for (int m = 0; m < M; ++m) {
        for (int c = 0; c < count0[m]; ++c) {
            const ObsActionRef& act = acts[m];
            if (!act.is_duel) {
                const int r = sample_reward(inst, act.a, rng);
                r01s[act.a][r]++;
            } else {
                const int y = sample_duel_outcome(inst, act.j, act.k, rng);
                y01s[act.j][act.k][y]++;
            }
            t++;
            if (t >= cfg.max_steps) break;
        }
        if (t >= cfg.max_steps) break;
    }

    Vec theta_hat(d, 0.0);
    theta_hat = constrained_mle_logistic(
        r01s, y01s, d, inst.S,
        1.0, 1.0,
        cfg.mle_cfg,
        theta_hat,
        inst
    );

    std::vector<int> active;
    active.reserve((size_t)K);
    for (int a = 0; a < K; ++a) active.push_back(a);

    int k = 1;
    while ((int)active.size() > 1 && t < cfg.max_steps) {
        double denom = 2.0 * (double)k * (double)k * (double)union_size * (2.0 + (double)x_size);
        const double delta_k = cfg.delta / std::max(1.0, denom);

        const int z_hat = argmax_z_active(inst, theta_hat, active);

        std::vector<double> lambda = approx_rage_design(inst, acts, active, z_hat, theta_hat, cfg.fw_iters, cfg.ridge);

        Mat H = fisher_matrix_from_lambda(inst, acts, lambda, theta_hat, cfg.ridge);

        double fval = 0.0;

        double maxX = 0.0;
        for (int m = 0; m < M; ++m) {
            const Vec& x = *acts[m].v; 
            maxX = std::max(maxX, quad_form_inv_spd(H, x));
        }
        fval = std::max(fval, gd * maxX);


        const double c352 = 3.5 * 3.5;

        double maxD = 0.0;
        for (int ii = 0; ii < (int)active.size(); ++ii) {
            for (int jj = ii + 1; jj < (int)active.size(); ++jj) {
                Vec diff = inst.x[active[ii]] - inst.x[active[jj]];
                maxD = std::max(maxD, quad_form_inv_spd(H, diff));
            }
        }
        fval = std::max(fval, std::pow(2.0, 2.0 * (double)k) * c352 * maxD);

        int nk = (int)std::ceil(std::max(
            3.0 * (1.0 + cfg.eps_round) * fval * std::log(1.0 / std::max(1e-300, delta_k)),
            (double)r_eps
        ));
        if (t + nk > cfg.max_steps) nk = cfg.max_steps - t;
        if (nk <= 0) break;

        std::vector<int> countk;
        epsilon_round_counts(lambda, nk, cfg.eps_round, countk);

        std::vector<std::vector<int>> round_r01s(K, std::vector<int>(2, 0));
        std::vector<std::vector<std::vector<int>>> round_y01s(K,
            std::vector<std::vector<int>>(K, std::vector<int>(2, 0))
        );

        for (int m = 0; m < M; ++m) {
            for (int c = 0; c < countk[m]; ++c) {
                const ObsActionRef& act = acts[m];
                if (!act.is_duel) {
                    const int r = sample_reward(inst, act.a, rng);
                    round_r01s[act.a][r]++;
                } else {
                    const int y = sample_duel_outcome(inst, act.j, act.k, rng);
                    round_y01s[act.j][act.k][y]++;
                }
                t++;
                if (t >= cfg.max_steps) break;
            }
            if (t >= cfg.max_steps) break;
        }

        theta_hat = constrained_mle_logistic(
            round_r01s, round_y01s, d, inst.S,
            1.0, 1.0,
            cfg.mle_cfg,
            theta_hat,
            inst
        );

        const int z_new = argmax_z_active(inst, theta_hat, active);

        std::vector<int> next_active;
        next_active.reserve(active.size());
        for (int a : active) {
            if (a == z_new) { next_active.push_back(a); continue; }
            const double gap_est = dot(inst.g[z_new][a], theta_hat);
            if (gap_est < std::pow(2.0, -(double)k)) next_active.push_back(a);
        }
        active.swap(next_active);

        k++;
        if (k > 60) break;
    }

    RAGEGLMResult res;
    res.hat_arm = (active.empty() ? argmax_z_all(inst, theta_hat) : active[0]);
    res.stop_t = t;
    res.correct = (res.hat_arm == inst.true_best_arm());
    res.rounds = k - 1;
    return res;
}
