#pragma once
// ============================================================================
// RAGE-GLM-style baseline (Jun et al., 2021) for *logistic* GLM best-arm ID.
//
// This version fixes the 
//   (1) uses ACTIVE-set argmax z_hat in Alg. 1 line 11
//   (2) uses the paper constant (3.5)^2
//   (3) replaces multinomial rounding by epsilon-rounding
//
// Patch (2026-01-22): support observation action set
//   all = {i : 0<=i<K} U {(j,k): 0<=j<k<K} via cfg.include_dueling_pairs_as_actions,
// while the active set / elimination / stopping still concerns only the K reward arms.
// ============================================================================

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

// ---------------- public API ----------------

struct RAGEGLMConfig {
    double delta = 0.05;

    // If true, allow selecting dueling pairs (j<k) as additional observation actions.
    // NOTE: this enlarges the observation action set, but the elimination/stop logic still only concerns the K reward arms.
    bool include_dueling_pairs_as_actions = false;

    // Paper's rounding tolerance ε (used in sample size formula and rounding).
    double eps_round = 0.10;

    int max_steps = 500000;

    // If > 0, override burn-in sample size.
    int burnin_n = -1;

    // Frank-Wolfe / greedy iterations for design approximation.
    int fw_iters = 100;

    // Ridge to keep Fisher information SPD.
    double ridge = 1e-6;

    // MLE configuration (Newton iterations, line search, etc.)
    MLEConfig mle_cfg;
};

struct RAGEGLMResult {
    int hat_arm = -1;
    int stop_t = 0;
    bool correct = false;
    int rounds = 0;
};

// ---------------- internal helpers ----------------

struct ObsActionRef {
    bool is_duel = false; // false: reward arm; true: dueling pair (j<k)
    int a = -1;           // arm id if !is_duel
    int j = -1, k = -1;   // pair ids if is_duel
    const Vec* v = nullptr; // feature vector: &inst.x[a] or &inst.g[j][k]
};

inline std::vector<ObsActionRef> build_all_actions(const Instance& inst, bool include_duels) {
    std::vector<ObsActionRef> acts;
    const int K = inst.K;
    acts.reserve((size_t)K + (include_duels ? (size_t)K * (size_t)(K - 1) / 2 : 0));

    for (int a = 0; a < K; ++a) {
        ObsActionRef ar;
        ar.is_duel = false;
        ar.a = a;
        ar.v = &inst.x[a];
        acts.push_back(ar);
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
    // gamma(d) = d + log(6(2+teff)/delta)
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

// epsilon-rounding to integer counts summing to n, with tolerance eps_round
void epsilon_round_counts(
    const std::vector<double>& lambda_in,
    int n,
    double eps_round,
    std::vector<int>& out_counts
) {
    const int K = (int)lambda_in.size();
    out_counts.assign(K, 0);
    if (n <= 0 || K <= 0) return;

    // normalize
    std::vector<double> lambda(K, 0.0);
    double s = 0.0;
    for (int a = 0; a < K; ++a) {
        lambda[a] = std::max(0.0, lambda_in[a]);
        s += lambda[a];
    }
    if (s <= 1e-15) {
        // fallback uniform
        int q = n / K;
        int r = n % K;
        for (int a = 0; a < K; ++a) out_counts[a] = q + (a < r ? 1 : 0);
        return;
    }
    for (int a = 0; a < K; ++a) lambda[a] /= s;

    // base allocation
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

    // If eps_round is too small, we effectively do largest fractional parts.
    // Otherwise, we still do the same (simple, deterministic).
    std::sort(fracs.begin(), fracs.end(), [](const Frac& u, const Frac& v) {
        return u.frac > v.frac;
    });
    for (int i = 0; i < rem; ++i) {
        out_counts[fracs[i % K].a] += 1;
    }

    // Sanity: enforce sum
    int sum = 0;
    for (int c : out_counts) sum += c;
    if (sum != n) {
        // adjust
        int diff = n - sum;
        if (diff > 0) out_counts[0] += diff;
        else if (diff < 0) out_counts[0] = std::max(0, out_counts[0] + diff);
    }

    (void)eps_round; // reserved for potential advanced rounding
}

// Fisher-aware rounding:
// 目标：在给定 theta_ref (通常用上一轮 theta_hat) 下，使得
//   H_counts(theta_ref) ≈ n * H(lambda, theta_ref)
// 并用 n/(1+eps) 的缩放预留 slack，剩余样本用 leverage-greedy 补齐。
//
// 注意：这是“对 theta_ref 的有效 rounding”（工程常用），并非论文要求的 ∀theta 严格版本。

// static Mat fisher_matrix_from_counts_theta(
//     const Instance& inst,
//     const std::vector<ObsActionRef>& acts,
//     const std::vector<int>& counts,
//     const Vec& theta_ref,
//     double ridge
// ) {
//     const int d = inst.d;
//     Mat H(d, 0.0);
//     const int M = (int)acts.size();
//     for (int m = 0; m < M; ++m) {
//         const int cm = counts[m];
//         if (cm <= 0) continue;
//         const Vec& x = *acts[m].v;
//         const double z = dot(x, theta_ref);
//         const double w = mu_prime(z);
//         const double scale = (double)cm * w;
//         for (int i = 0; i < d; ++i) {
//             const double xi = x[i];
//             for (int j = 0; j < d; ++j) {
//                 H(i, j) += scale * xi * x[j];
//             }
//         }
//     }
//     for (int i = 0; i < d; ++i) H(i, i) += ridge;
//     return H;
// }

// // Compute leverage score of action m under current SPD matrix H:
// //   score = w * x^T H^{-1} x
// static double leverage_score(
//     const Instance& inst,
//     const ObsActionRef& act,
//     const Mat& H,
//     const Vec& theta_ref
// ) {
//     const Vec& x = *act.v;
//     const double z = dot(x, theta_ref);
//     const double w = mu_prime(z);
//     const double q = quad_form_inv_spd(H, x); // x^T H^{-1} x
//     return w * q;
// }

// void fisher_round_counts(
//     const Instance& inst,
//     const std::vector<ObsActionRef>& acts,
//     const std::vector<double>& lambda_in,
//     int n,
//     double eps_round,
//     const Vec& theta_ref,
//     double ridge,
//     std::vector<int>& out_counts
// ) {
//     const int M = (int)acts.size();
//     out_counts.assign(M, 0);
//     if (n <= 0 || M <= 0) return;

//     // --- normalize lambda over M actions ---
//     std::vector<double> lambda(M, 0.0);
//     double s = 0.0;
//     for (int m = 0; m < M; ++m) {
//         lambda[m] = std::max(0.0, lambda_in[m]);
//         s += lambda[m];
//     }
//     if (s <= 1e-15) {
//         // fallback uniform
//         int q = n / M;
//         int r = n % M;
//         for (int m = 0; m < M; ++m) out_counts[m] = q + (m < r ? 1 : 0);
//         return;
//     }
//     for (int m = 0; m < M; ++m) lambda[m] /= s;

//     // --- base allocation uses n/(1+eps) scaling (leave slack for greedy fill) ---
//     const double n_core = (double)n / (1.0 + std::max(0.0, eps_round));

//     int used = 0;
//     std::vector<int> support;
//     support.reserve((size_t)M);

//     for (int m = 0; m < M; ++m) {
//         if (lambda[m] <= 0.0) continue;
//         support.push_back(m);
//         const double raw = n_core * lambda[m];
//         const int base = (int)std::floor(raw);
//         if (base > 0) {
//             out_counts[m] = base;
//             used += base;
//         }
//     }

//     // Ensure we always use exactly n samples
//     int rem = n - used;
//     if (rem <= 0) {
//         // adjust down if numerical drift
//         int sum = 0;
//         for (int c : out_counts) sum += c;
//         if (sum > n) {
//             int diff = sum - n;
//             for (int m = 0; m < M && diff > 0; ++m) {
//                 int dec = std::min(out_counts[m], diff);
//                 out_counts[m] -= dec;
//                 diff -= dec;
//             }
//         }
//         return;
//     }

//     // --- leverage-greedy fill for remaining samples ---
//     // Strategy:
//     //   each step: recompute H from counts (cheap if rem small), pick action maximizing leverage, add 1.
//     // Acceleration:
//     //   recompute H only every "rebuild_period" steps; between rebuilds, keep adding to the current best action.
//     const int rebuild_period = 32; // tuneable: larger -> faster, smaller -> closer to greedy
//     int step = 0;

//     // initial H from base counts
//     Mat H = fisher_matrix_from_counts_theta(inst, acts, out_counts, theta_ref, ridge);

//     while (rem > 0) {
//         if (step % rebuild_period == 0) {
//             H = fisher_matrix_from_counts_theta(inst, acts, out_counts, theta_ref, ridge);
//         }

//         int best_m = 0;
//         double best = -1.0;

//         // Search over support only for speed (if you want full M, loop over m=0..M-1)
//         for (int idx = 0; idx < (int)support.size(); ++idx) {
//             const int m = support[idx];
//             const double sc = leverage_score(inst, acts[m], H, theta_ref);
//             if (sc > best) { best = sc; best_m = m; }
//         }

//         // add one sample to best_m
//         out_counts[best_m] += 1;
//         rem -= 1;
//         step += 1;
//     }

//     // final sanity
//     int sum = 0;
//     for (int c : out_counts) sum += c;
//     if (sum != n) {
//         int diff = n - sum;
//         if (diff > 0) out_counts[support.empty() ? 0 : support[0]] += diff;
//         else if (diff < 0) {
//             int m0 = support.empty() ? 0 : support[0];
//             out_counts[m0] = std::max(0, out_counts[m0] + diff);
//         }
//     }
// }


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
        // H += lm * w * x x^T
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

// ---------------- design approximations ----------------

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

    for (int t = 0; t < iters; ++t) {
        Mat H = fisher_matrix_from_lambda(inst, acts, lambda, theta_prev, ridge);

        // Find worst direction in D: argmax_{y in D} y^T H^{-1} y
        int worst_i = 0;
        double worstv = quad_form_inv_spd(H, D[0]);
        for (int i = 1; i < (int)D.size(); ++i) {
            const double v = quad_form_inv_spd(H, D[i]);
            if (v > worstv) { worstv = v; worst_i = i; }
        }
        const Vec& y = D[worst_i];
        Vec v = solve_spd_cholesky(H, y); // v = H^{-1} y

        // Linear oracle over observation actions: argmax_{m} mu'(x_m^T theta) * (x_m^T v)^2
        int best_m = 0;
        double best_score = -1.0;
        for (int m = 0; m < M; ++m) {
            const Vec& x = *acts[m].v;
            const double z = dot(x, theta_prev);
            const double w = mu_prime(z);
            const double ip = dot(x, v);
            const double score = w * ip * ip;
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

// Burn-in design: D = {x_a}_{a=0}^{K-1} (i.e., we only care about covering reward arms)
std::vector<double> approx_burnin_design(
    const Instance& inst,
    const std::vector<ObsActionRef>& acts,
    int iters,
    double ridge
) {
    std::vector<Vec> D;
    D.reserve((size_t)inst.K);
    for (int a = 0; a < inst.K; ++a) D.push_back(inst.x[a]);
    Vec theta0(inst.d, 0.0);
    return approx_fw_design(inst, acts, D, theta0, iters, ridge);
}

// RAGE-style design for a given active set: D = {x_a}_{a=0}^{K-1} U {x_{zhat} - x_i : i in active, i != zhat}
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

// ---------------- main algorithm ----------------

RAGEGLMResult run_rageglm_baseline(
    Instance& inst,
    RAGEGLMConfig& cfg,
    RNG& rng
) {
    const int K = inst.K;
    const int d = inst.d;

    // reward outcomes
    std::vector<std::vector<int>> r01s(K, std::vector<int>(2, 0));
    // dueling outcomes (ONLY upper triangle j<k is used by the MLE code in mle.h)
    std::vector<std::vector<std::vector<int>>> y01s(K,
        std::vector<std::vector<int>>(K, std::vector<int>(2, 0))
    );

    // Observation action set: all = arms U {j<k pairs}
    const std::vector<ObsActionRef> acts = build_all_actions(inst, cfg.include_dueling_pairs_as_actions);
    const int M = (int)acts.size();

    // Burn-in (Alg. 2)
    const double L = max_arm_norm(inst);
    const double kappa0 = mu_prime(L * inst.S);
    const double kappa0_inv = 1.0 / std::max(1e-12, kappa0);

    const int teff = K;
    const double gd = gamma_d(d, teff, cfg.delta);
    const int r_eps = (int)std::ceil((double)d * (double)d / std::max(1e-12, cfg.eps_round));

    int n0 = 0;
    if (cfg.burnin_n > 0) {
        n0 = cfg.burnin_n;
    } else {
        // n0 = 3(1+eps) kappa0^{-1} d gamma(d) log(2|X|(2+|X|)/delta)
        double inside;
        if (!cfg.include_dueling_pairs_as_actions) {
            inside = 2.0 * (double)K * (2.0 + (double)K) / std::max(1e-6, cfg.delta);
        } else {
            inside = 2.0 * (double)(K + K * (K-1) / 2) * (2.0 + (double)(K + K * (K-1) / 2)) / std::max(1e-6, cfg.delta);
        }
        const double nn = 3.0 * (1.0 + cfg.eps_round) * kappa0_inv * (double)d * gd * std::log(inside);
        n0 = (int)std::ceil(std::max(nn, (double)r_eps));
    }

    int t = 0;

    // Burn-in design over observation actions, but only covering reward arms (D = {x_a})
    std::vector<double> lambda0 = approx_burnin_design(inst, acts, cfg.fw_iters, cfg.ridge);
    std::vector<int> count0;
    epsilon_round_counts(lambda0, n0, cfg.eps_round, count0);
    // fisher_round_counts(
    //     inst, acts, lambda0, n0, cfg.eps_round,
    //     /*theta_ref=*/Vec(inst.d, 0.0),
    //     cfg.ridge,
    //     count0
    // );


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

    // Elimination rounds (Alg. 1)
    std::vector<int> active;
    active.reserve((size_t)K);
    for (int a = 0; a < K; ++a) active.push_back(a);

    int k = 1;
    int XS = K + (cfg.include_dueling_pairs_as_actions ? K * (K-1) / 2 : 0);
    while ((int)active.size() > 1 && t < cfg.max_steps) {
        double denom = 2.0 * (double)k * (double)k * (double)XS * (2.0 + (double)XS);
        const double delta_k = cfg.delta / std::max(1.0, denom);

        // IMPORTANT: z_hat over ACTIVE set (paper Alg.1 line 11)
        const int z_hat = argmax_z_active(inst, theta_hat, active);

        // Design over ALL observation actions, but the D-set is still driven by ACTIVE arms
        std::vector<double> lambda = approx_rage_design(inst, acts, active, z_hat, theta_hat, cfg.fw_iters, cfg.ridge);

        Mat H = fisher_matrix_from_lambda(inst, acts, lambda, theta_hat, cfg.ridge);

        // f(λ) computation
        double fval = 0.0;

        // First term: gamma(d) * max_{x in X} ||x||^2_{H^{-1}}
        double maxX = 0.0;
        for (int m = 0; m < M; ++m) {
            const Vec& x = *acts[m].v;          // 可能是 inst.x[a] 或 inst.g[j][k]
            maxX = std::max(maxX, quad_form_inv_spd(H, x));
        }
        fval = std::max(fval, gd * maxX);


        // Second term: (2 / 2^k) * (3.5)^2 * max_{z,z' in Z_k} ||z-z'||^2_{H^{-1}}
        // (paper constant 3.5 from the confidence bound)
        const double c352 = 3.5 * 3.5;

        double maxD = 0.0;
        for (int ii = 0; ii < (int)active.size(); ++ii) {
            for (int jj = ii + 1; jj < (int)active.size(); ++jj) {
                Vec diff = inst.x[active[ii]] - inst.x[active[jj]];
                maxD = std::max(maxD, quad_form_inv_spd(H, diff));
            }
        }
        fval = std::max(fval, (2.0 / std::pow(2.0, (double)k)) * c352 * maxD);

        // nk = ceil( max( 3(1+eps) f(λ) log(1/delta_k), r(ε) ) )
        int nk = (int)std::ceil(std::max(
            3.0 * (1.0 + cfg.eps_round) * fval * std::log(1.0 / std::max(1e-300, delta_k)),
            (double)r_eps
        ));
        if (t + nk > cfg.max_steps) nk = cfg.max_steps - t;
        if (nk <= 0) break;

        std::vector<int> countk;
        epsilon_round_counts(lambda, nk, cfg.eps_round, countk);

        // fisher_round_counts(
        //     inst, acts, lambda, nk, cfg.eps_round,
        //     /*theta_ref=*/theta_hat,  // 用上一轮/当前的 theta_hat 作为参考
        //     cfg.ridge,
        //     countk
        // );


        // Collect nk samples according to rounded design
        for (int m = 0; m < M; ++m) {
            for (int c = 0; c < countk[m]; ++c) {
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

        theta_hat = constrained_mle_logistic(
            r01s, y01s, d, inst.S,
            1.0, 1.0,
            cfg.mle_cfg,
            theta_hat,
            inst
        );

        // elimination: keep arms with estimated gap below 2^{-k}
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
