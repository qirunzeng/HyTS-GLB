#pragma once
#include <vector>
#include <cmath>
#include <limits>
#include <iostream>
#include <algorithm>
#include "lin_alg.h"
#include "rng.h"
#include "mle.h"
#include "env.h"
#include "cholesky.h"

inline double beta_t(double delta, int d, double S, double L_t, bool printInfo = false) {
    double ct = std::min(1.0, double(d) / (2.0 * S * L_t));
    double val = std::log(1.0/delta) - (double)d * std::log(ct) + 2.0 * S * L_t * ct;
    if (printInfo) {
        std::cout << "beta_t: delta=" << delta << ", d=" << d << ", S=" << S << ", L_t=" << L_t << ", ct=" << ct << ", val=" << val << std::endl;
    }
    return val;
}

inline double lipschitz_Lt_bernoulli(int t_c, int t_d, double S) {
    return (1.0 + S/2.0) * (t_c + 2*t_d);
}

struct HybridConfig {
    double delta = 0.05;
    double duel_bound = 2.;
    int max_steps = 200000;

    int fw_iters = 20;
    int update_period = 1;

    double Rs_c = 1.0;
    double Rs_d = 1.0;
    double zeta_c = 1.0;
    double zeta_d = 1.0;
    double sc, sd;

    double lambda = 1e-6;

    MLEConfig mle_cfg;

    bool reward_only = false;
    bool duel_only = false;
    double cc = 1., cd = 1.;

    void get_sc_sd(double S) {
        sc = 1.0 / (2.0 * (1. + S * Rs_c));
        sd = 1.0 / (2.0 * (1. + 2 * S * Rs_d));
    }
};

struct RunSummary {
    int stop_time = 0;
    double c_c = 0, c_d = 0;
    int pred_best = -1;
    int true_best = -1;
    bool correct = false;
};

struct PairAction {
    int j = 0;
    int k = 1;
};

inline std::vector<PairAction> build_pair_actions(int K);

struct HybridState {
    std::vector<PairAction> pairs;
    std::vector<int> Nc;
    std::vector<int> Nd;
    std::vector<double> W_arm;
    std::vector<double> W_pair;
    std::vector<double> w_arm;
    std::vector<double> w_pair;
    std::vector<double> curv_arm;
    std::vector<double> curv_pair;

    explicit HybridState(int K)
        : pairs(build_pair_actions(K)),
          Nc(K, 0),
          Nd(pairs.size(), 0),
          W_arm(K, 0.0),
          W_pair(pairs.size(), 0.0),
          w_arm(K, 0.0),
          w_pair(pairs.size(), 0.0),
          curv_arm(K, 0.0),
          curv_pair(pairs.size(), 0.0) {}
};

inline std::vector<PairAction> build_pair_actions(int K) {
    std::vector<PairAction> pairs;
    pairs.reserve((size_t)K * (size_t)(K - 1) / 2);
    for (int j = 0; j < K; ++j) {
        for (int k = j + 1; k < K; ++k) {
            pairs.push_back({j, k});
        }
    }
    return pairs;
}

inline Mat hessian_classic_only(const std::vector<int>& Nc, const Vec& theta_hat, double zeta_c, const Instance& inst) {
    Mat H(inst.d, 0.0);
    for (int i = 0; i < inst.K; ++i) {
        if (Nc[i] == 0) continue;
        double z = dot(inst.x[i], theta_hat);
        double coef = (double)Nc[i] * mu_prime(z) / zeta_c;
        for (int a = 0; a < inst.d; ++a) {
            const double xa = inst.x[i][a];
            for (int b = 0; b < inst.d; ++b) {
                H(a,b) += coef * xa * inst.x[i][b];
            }
        }
    }
    return H;
}

inline Mat hessian_duel_only(const std::vector<std::vector<int>>& Nd, const Vec& theta_hat, double zeta_d, const Instance& inst) {
    Mat H(inst.d, 0.0);
    for (int j = 0; j < inst.K; ++j) {
        for (int k = j+1; k < inst.K; ++k) {
            if (Nd[j][k] == 0) continue;
            double z = dot(inst.g[j][k], theta_hat);
            double coef = (double)Nd[j][k] * mu_prime(z) / zeta_d;
            for (int a = 0; a < inst.d; ++a) {
                const double ga = inst.g[j][k][a];
                for (int b = 0; b < inst.d; ++b) {
                    H(a,b) += coef * ga * inst.g[j][k][b];
                }
            }
        }
    }
    return H;
}

inline Mat hessian_duel_only_flat(
    const std::vector<int>& Nd,
    const std::vector<PairAction>& pairs,
    const Vec& theta_hat,
    double zeta_d,
    const Instance& inst
) {
    Mat H(inst.d, 0.0);
    for (int p = 0; p < (int)pairs.size(); ++p) {
        if (Nd[p] == 0) continue;
        const int j = pairs[p].j;
        const int k = pairs[p].k;
        const Vec& g = inst.g[j][k];
        double coef = (double)Nd[p] * mu_prime(dot(g, theta_hat)) / zeta_d;
        for (int a = 0; a < inst.d; ++a) {
            const double ga = g[a];
            for (int b = 0; b < inst.d; ++b) {
                H(a,b) += coef * ga * g[b];
            }
        }
    }
    return H;
}

inline Mat info_matrix_A(
    const Mat& Hc,
    const Mat& Hd,
    double S,
    double Rs_c,
    double Rs_d,
    double duel_bound
) {
    double ac = 1.0 / (2.0 * (1.0 + S * Rs_c));
    double ad = 1.0 / (2.0 * (1.0 + duel_bound * S * Rs_d));
    Mat A(Hc.n, 0.0);
    for (int i = 0; i < Hc.n * Hc.n; ++i) {
        A.a[i] = ac * Hc.a[i] + ad * Hd.a[i];
    }
    return A;
}

inline int predicted_best_arm(const Instance& inst, const Vec& theta_hat) {
    int best = 0;
    double bestv = dot(inst.x[0], theta_hat);
    for (int i = 1; i < inst.K; ++i) {
        double v = dot(inst.x[i], theta_hat);
        if (v > bestv) { 
            bestv=v; 
            best = i; 
        }
    }
    return best;
}

inline bool stop_condition(
    const Instance& inst,
    const Vec& theta_hat,
    const Mat& A,
    double beta,
    bool PrintInfo = false
) {
    int ihat = predicted_best_arm(inst, theta_hat);
    Chol L = chol_spd(A);

    bool ret = true;
    for (int i = 0; i < inst.K; ++i) if (i != ihat) {
        double gap = dot(inst.g[ihat][i], theta_hat);
        double q = quad_form_inv_chol(L, inst.g[ihat][i]);
        double rad = std::sqrt(std::max(0.0, beta * q));
        if (PrintInfo) {
            printf("%d-%d: gap = %lf, rad = %lf\n", ihat, i, gap, rad);
        }
        if (!(gap > rad)) {
            ret = false;
            break;
        }
    }
    return ret;
}

inline Mat compute_A_of_w_flat(
    const Instance& inst,
    const Vec& theta_hat,
    const HybridConfig& cfg,
    const std::vector<PairAction>& pairs,
    const std::vector<double>& w_arm,
    const std::vector<double>& w_pair,
    const std::vector<double>* curv_arm_cache = nullptr,
    const std::vector<double>* curv_pair_cache = nullptr
) {
    int d = inst.d;
    int K = inst.K;

    const double wc = 1.0 / (2.0 * (1.0 + inst.S * cfg.Rs_c));
    const double wd = 1.0 / (2.0 * (1.0 + cfg.duel_bound * inst.S * cfg.Rs_d));

    Mat A(d, cfg.lambda * (wc + wd));

    for (int i = 0; i < K; ++i) {
        const double wi = w_arm[i];
        if (wi <= 0.0) continue;
        const double curv = curv_arm_cache ? (*curv_arm_cache)[i] : mu_prime(dot(inst.x[i], theta_hat)) / cfg.zeta_c;
        const double coef = wc * wi * curv;
        for (int a = 0; a < d; ++a) {
            const double xa = inst.x[i][a];
            for (int b = 0; b < d; ++b) {
                A(a,b) += coef * xa * inst.x[i][b];
            }
        }
    }

    for (int p = 0; p < (int)pairs.size(); ++p) {
        const double wp = w_pair[p];
        if (wp <= 0.0) continue;
        const int j = pairs[p].j;
        const int k = pairs[p].k;
        const Vec& g = inst.g[j][k];
        const double curv = curv_pair_cache ? (*curv_pair_cache)[p] : mu_prime(dot(g, theta_hat)) / cfg.zeta_d;
        const double coef = wd * wp * curv;
        for (int a = 0; a < d; ++a) {
            const double ga = g[a];
            for (int b = 0; b < d; ++b) {
                A(a,b) += coef * ga * g[b];
            }
        }
    }
    return A;
}


static void compute_optimal_proportions_track_stop_flat(
    const Instance& inst,
    const Vec& theta_hat,
    const HybridConfig& cfg,
    const std::vector<PairAction>& pairs,
    std::vector<double>& w_arm,
    std::vector<double>& w_pair,
    std::vector<double>& curv_arm,
    std::vector<double>& curv_pair
) {
    const int K = inst.K;
    const int P = (int)pairs.size();

    std::fill(w_arm.begin(), w_arm.end(), 0.0);
    std::fill(w_pair.begin(), w_pair.end(), 0.0);

    if (cfg.reward_only) {
        const double mass_arm = 1.0 / (double)K;
        std::fill(w_arm.begin(), w_arm.end(), mass_arm);
    } else if (cfg.duel_only) {
        const double mass_pair = 1.0 / (double)P;
        std::fill(w_pair.begin(), w_pair.end(), mass_pair);
    } else {
        const double mass = 1.0 / (double)(K + P);
        std::fill(w_arm.begin(), w_arm.end(), mass);
        std::fill(w_pair.begin(), w_pair.end(), mass);
    }

    const int ihat = predicted_best_arm(inst, theta_hat);

    std::fill(curv_arm.begin(), curv_arm.end(), 0.0);
    if (!cfg.duel_only) {
        for (int i = 0; i < K; ++i) {
            curv_arm[i] = mu_prime(dot(inst.x[i], theta_hat)) / cfg.zeta_c;
        }
    }

    std::fill(curv_pair.begin(), curv_pair.end(), 0.0);
    if (!cfg.reward_only) {
        for (int p = 0; p < P; ++p) {
            const int j = pairs[p].j;
            const int k = pairs[p].k;
            curv_pair[p] = mu_prime(dot(inst.g[j][k], theta_hat)) / cfg.zeta_d;
        }
    }

    for (int m = 0; m < cfg.fw_iters; ++m) {
        Mat A = compute_A_of_w_flat(inst, theta_hat, cfg, pairs, w_arm, w_pair, &curv_arm, &curv_pair);
        Chol L = chol_spd(A);

        int idag = -1;
        double best_val = -1.0;
        for (int i = 0; i < K; ++i) if (i != ihat) {
            const double val = quad_form_inv_chol(L, inst.g[ihat][i]);
            if (val > best_val) {
                best_val = val;
                idag = i;
            }
        }
        if (idag < 0) return;

        Vec u = solve_chol(L, inst.g[ihat][idag]);

        int i_star = 0;
        double sc_best = -1.0;
        if (!cfg.duel_only) {
            for (int i = 0; i < K; ++i) {
                const double proj = dot(inst.x[i], u);
                const double sc = cfg.sc * curv_arm[i] * proj * proj;
                if (sc > sc_best) {
                    sc_best = sc;
                    i_star = i;
                }
            }
        }

        int p_star = 0;
        double sd_best = -1.0;
        if (!cfg.reward_only) {
            for (int p = 0; p < P; ++p) {
                const int j = pairs[p].j;
                const int k = pairs[p].k;
                const double proj = dot(inst.g[j][k], u);
                const double sd = cfg.sd * curv_pair[p] * proj * proj;
                if (sd > sd_best) {
                    sd_best = sd;
                    p_star = p;
                }
            }
        }

        const double gamma = 2.0 / (double)(m + 2);
        const double one_minus = 1.0 - gamma;

        for (double& w : w_arm) w *= one_minus;
        for (double& w : w_pair) w *= one_minus;

        if (sc_best >= sd_best) w_arm[i_star] += gamma;
        else                    w_pair[p_star] += gamma;
    }

    double sum = 0.0;
    for (double w : w_arm) sum += w;
    for (double w : w_pair) sum += w;
    if (sum > 0) {
        const double inv = 1.0 / sum;
        for (double& w : w_arm) w *= inv;
        for (double& w : w_pair) w *= inv;
    }
}

static void cost_optimal_proportions_track_stop_flat(
    const Instance& inst,
    const Vec& theta_hat,
    const HybridConfig& cfg,
    const std::vector<PairAction>& pairs,
    std::vector<double>& w_arm,
    std::vector<double>& w_pair,
    std::vector<double>& curv_arm,
    std::vector<double>& curv_pair
) {
    const int K = inst.K;
    const int P = (int)pairs.size();

    const double inv_cc = 1.0 / std::max(1e-12, cfg.cc);
    const double inv_cd = 1.0 / std::max(1e-12, cfg.cd);

    std::fill(w_arm.begin(), w_arm.end(), inv_cc);
    std::fill(w_pair.begin(), w_pair.end(), inv_cd);

    double sum0 = 0.0;
    for (double w : w_arm) sum0 += w;
    for (double w : w_pair) sum0 += w;
    if (sum0 > 0) {
        const double inv = 1.0 / sum0;
        for (double& w : w_arm) w *= inv;
        for (double& w : w_pair) w *= inv;
    }

    const int ihat = predicted_best_arm(inst, theta_hat);

    for (int i = 0; i < K; ++i) {
        curv_arm[i] = mu_prime(dot(inst.x[i], theta_hat)) / cfg.zeta_c;
    }

    for (int p = 0; p < P; ++p) {
        const int j = pairs[p].j;
        const int k = pairs[p].k;
        curv_pair[p] = mu_prime(dot(inst.g[j][k], theta_hat)) / cfg.zeta_d;
    }

    for (int m = 0; m < cfg.fw_iters; ++m) {
        Mat A = compute_A_of_w_flat(inst, theta_hat, cfg, pairs, w_arm, w_pair, &curv_arm, &curv_pair);
        Chol L = chol_spd(A);

        int idag = -1;
        double best_val = -1.0;
        for (int i = 0; i < K; ++i) if (i != ihat) {
            const double val = quad_form_inv_chol(L, inst.g[ihat][i]);
            if (val > best_val) {
                best_val = val;
                idag = i;
            }
        }
        if (idag < 0) return;

        Vec u = solve_chol(L, inst.g[ihat][idag]);

        int i_star = 0;
        double sc_best = -1.0;
        for (int i = 0; i < K; ++i) {
            const double proj = dot(inst.x[i], u);
            const double sc = cfg.sc * curv_arm[i] * proj * proj;
            if (sc > sc_best) {
                sc_best = sc;
                i_star = i;
            }
        }

        int p_star = 0;
        double sd_best = -1.0;
        for (int p = 0; p < P; ++p) {
            const int j = pairs[p].j;
            const int k = pairs[p].k;
            const double proj = dot(inst.g[j][k], u);
            const double sd = cfg.sd * curv_pair[p] * proj * proj;
            if (sd > sd_best) {
                sd_best = sd;
                p_star = p;
            }
        }

        const double gamma = 2.0 / (double)(m + 2);
        const double one_minus = 1.0 - gamma;

        for (double& w : w_arm) w *= one_minus;
        for (double& w : w_pair) w *= one_minus;

        const double eff_sc = sc_best / std::max(1e-12, cfg.cc);
        const double eff_sd = sd_best / std::max(1e-12, cfg.cd);

        if (eff_sc >= eff_sd) w_arm[i_star] += gamma;
        else                  w_pair[p_star] += gamma;
    }

    double sum = 0.0;
    for (double w : w_arm) sum += w;
    for (double w : w_pair) sum += w;
    if (sum > 0) {
        const double inv = 1.0 / sum;
        for (double& w : w_arm) w *= inv;
        for (double& w : w_pair) w *= inv;
    }
}


inline RunSummary run_cost(
    Instance& inst, 
    const HybridConfig& cfg, 
    RNG& rng
) {
    RunSummary out;
    out.true_best = inst.true_best_arm();

    HybridState state(inst.K);

    Vec theta_hat(inst.d, 0.0);
    bool have_design = false;

    
    std::vector<std::vector<int>> r01s(inst.K, std::vector<int> (2, 0));
    std::vector<std::vector<std::vector<int>>> y01s(inst.K, std::vector<std::vector<int>> (inst.K, std::vector<int>(2, 0)));


    int t = 0, t_c = 0;
    
    const int warmup_steps = inst.K * inst.d;
    const int update_period = std::max(1, cfg.update_period);

    for (; t < warmup_steps; t++, t_c++) {
        int a = t % inst.K;
        int r = sample_reward(inst, a, rng);
        r01s[a][r]++;
        state.Nc[a]++;
    }

    for (; t < cfg.max_steps; ++t) {

        const bool do_update = !have_design || ((t - warmup_steps) % update_period == 0);

        if (do_update) {
            theta_hat = constrained_mle_logistic(r01s, y01s, inst.d, inst.S, cfg.zeta_c, cfg.zeta_d, cfg.mle_cfg, theta_hat, inst);

            Mat Hc = hessian_classic_only(state.Nc, theta_hat, cfg.zeta_c, inst);
            Mat Hd = hessian_duel_only_flat(state.Nd, state.pairs, theta_hat, cfg.zeta_d, inst);
            Mat A  = info_matrix_A(Hc, Hd, inst.S, cfg.Rs_c, cfg.Rs_d, cfg.duel_bound) + Mat(inst.d, cfg.lambda);

            double Lt = lipschitz_Lt_bernoulli(t_c, t - t_c, inst.S);
            double beta = beta_t(cfg.delta, inst.d, inst.S, Lt);

            if (t > inst.d && stop_condition(inst, theta_hat, A, beta)) {
                break;
            }

            cost_optimal_proportions_track_stop_flat(
                inst, theta_hat, cfg, state.pairs,
                state.w_arm, state.w_pair,
                state.curv_arm, state.curv_pair
            );
            have_design = true;
        }
        for (int i = 0; i < inst.K; ++i) {
            state.W_arm[i] += state.w_arm[i];
        }
        for (int p = 0; p < (int)state.pairs.size(); ++p) {
            state.W_pair[p] += state.w_pair[p];
        }

        int best_i = 0;
        double best_val = (double)state.Nc[0] - state.W_arm[0];
        for (int i = 1; i < inst.K; ++i) {
            if (double v = (double)state.Nc[i] - state.W_arm[i]; v < best_val) { 
                best_val = v; 
                best_i = i;
            }
        }

        int best_p = 0;
        double best_duel_val = (double)state.Nd[0] - state.W_pair[0];
        for (int p = 1; p < (int)state.pairs.size(); ++p) {
            if (double v = (double)state.Nd[p] - state.W_pair[p]; v < best_duel_val) {
                best_duel_val = v;
                best_p = p;
            }
        }

        bool do_duel = (best_duel_val < best_val);

        if (!do_duel) {
            int r = sample_reward(inst, best_i, rng);
            r01s[best_i][r]++;
            state.Nc[best_i]++; t_c++;
        } else {
            const int best_j = state.pairs[best_p].j;
            const int best_k = state.pairs[best_p].k;
            int y = sample_duel_outcome(inst, best_j, best_k, rng);
            y01s[best_j][best_k][y]++;
            state.Nd[best_p]++;
        }
    }

    out.stop_time = t;
    out.c_c = t_c * cfg.cc;
    out.c_d = (t - t_c) * cfg.cd;
    out.pred_best = predicted_best_arm(inst, theta_hat);
    out.correct = (out.pred_best == out.true_best);
    return out;
}


inline RunSummary run_one(
    Instance& inst, 
    const HybridConfig& cfg, 
    RNG& rng
) {
    RunSummary out;
    out.true_best = inst.true_best_arm();

    HybridState state(inst.K);

    Vec theta_hat(inst.d, 0.0);
    bool have_design = false;

    
    std::vector<std::vector<int>> r01s(inst.K, std::vector<int> (2, 0));
    std::vector<std::vector<std::vector<int>>> y01s(inst.K, std::vector<std::vector<int>> (inst.K, std::vector<int>(2, 0)));

    int t = 0, t_c = 0;
    
    const int warmup_steps = inst.K * inst.d;
    const int update_period = std::max(1, cfg.update_period);

    for (; t < warmup_steps; t++, t_c++) {
        int a = t % inst.K;
        int r = sample_reward(inst, a, rng);
        r01s[a][r]++;
        state.Nc[a]++;
    }


    for (; t < cfg.max_steps; ++t) {

        const bool do_update = !have_design || ((t - warmup_steps) % update_period == 0);

        if (do_update) {
            theta_hat = constrained_mle_logistic(r01s, y01s, inst.d, inst.S, cfg.zeta_c, cfg.zeta_d, cfg.mle_cfg, theta_hat, inst);

            Mat Hc = hessian_classic_only(state.Nc, theta_hat, cfg.zeta_c, inst);
            Mat Hd = hessian_duel_only_flat(state.Nd, state.pairs, theta_hat, cfg.zeta_d, inst);
            Mat A  = info_matrix_A(Hc, Hd, inst.S, cfg.Rs_c, cfg.Rs_d, cfg.duel_bound) + Mat(inst.d, cfg.lambda);

            double Lt = lipschitz_Lt_bernoulli(t_c, t - t_c, inst.S);
            double beta = beta_t(cfg.delta, inst.d, inst.S, Lt);

            if (t > inst.d && stop_condition(inst, theta_hat, A, beta)) {
                break;
            }

            compute_optimal_proportions_track_stop_flat(
                inst, theta_hat, cfg, state.pairs,
                state.w_arm, state.w_pair,
                state.curv_arm, state.curv_pair
            );
            have_design = true;
        }
        for (int i = 0; i < inst.K; ++i) {
            state.W_arm[i] += state.w_arm[i];
        }
        for (int p = 0; p < (int)state.pairs.size(); ++p) {
            state.W_pair[p] += state.w_pair[p];
        }

        int best_i = 0;
        double best_val = (double)state.Nc[0] - state.W_arm[0];
        for (int i = 1; i < inst.K; ++i) {
            if (double v = (double)state.Nc[i] - state.W_arm[i]; v < best_val) { 
                best_val = v; 
                best_i = i;
            }
        }

        int best_p = 0;
        double best_duel_val = (double)state.Nd[0] - state.W_pair[0];
        for (int p = 1; p < (int)state.pairs.size(); ++p) {
            if (double v = (double)state.Nd[p] - state.W_pair[p]; v < best_duel_val) {
                best_duel_val = v;
                best_p = p;
            }
        }

        bool do_duel = (best_duel_val < best_val);

        if (!do_duel) {
            int r = sample_reward(inst, best_i, rng);
            r01s[best_i][r]++;
            state.Nc[best_i]++; t_c++;
        } else {
            const int best_j = state.pairs[best_p].j;
            const int best_k = state.pairs[best_p].k;
            int y = sample_duel_outcome(inst, best_j, best_k, rng);
            y01s[best_j][best_k][y]++;
            state.Nd[best_p]++;
        }
    }

    out.stop_time = t;
    out.c_c = t_c * cfg.cc;
    out.c_d = (t - t_c) * cfg.cd;
    out.pred_best = predicted_best_arm(inst, theta_hat);
    out.correct = (out.pred_best == out.true_best);
    return out;
}




inline RunSummary run_rand(Instance& inst, const HybridConfig& cfg, RNG& rng) {
    RunSummary out;
    out.true_best = inst.true_best_arm();

    std::vector<int> Nc(inst.K, 0);

    Vec theta_hat(inst.d, 0.0);

    std::vector<double> W_arm(inst.K, 0.0);

    int t = 0;
    
    std::vector<std::vector<int>> r01s(inst.K, std::vector<int> (2, 0));
    std::vector<std::vector<std::vector<int>>> y01s(inst.K, std::vector<std::vector<int>> (inst.K, std::vector<int>(2, 0)));

    for (int t_c = 0; t < cfg.max_steps; ++t) {

        theta_hat = constrained_mle_logistic(r01s, y01s, inst.d, inst.S, cfg.zeta_c, cfg.zeta_d, cfg.mle_cfg, theta_hat, inst);

        Mat Hc = hessian_classic_only(Nc, theta_hat, cfg.zeta_c, inst);
        Mat Hd(inst.d, 0.0);
        Mat A  = info_matrix_A(Hc, Hd, inst.S, cfg.Rs_c, cfg.Rs_d, cfg.duel_bound) + Mat(inst.d, cfg.lambda);

        double Lt = lipschitz_Lt_bernoulli(t_c, t - t_c, inst.S);
        double beta = beta_t(cfg.delta, inst.d, inst.S, Lt);

        if (t > inst.d && stop_condition(inst, theta_hat, A, beta)) {
            break;
        }

        int arm = int(rng.uniform01() * inst.K);
        if (arm == inst.K) {
            arm--;
        }

        r01s[arm][sample_reward(inst, arm, rng)]++;

        Nc[arm] += 1;
        t_c += 1;
    }

    out.stop_time = t;
    out.pred_best = predicted_best_arm(inst, theta_hat);
    out.correct = (out.pred_best == out.true_best);
    return out;
}





inline RunSummary run_rand_hybrid(Instance& inst, const HybridConfig& cfg, RNG& rng) {
    RunSummary out;
    out.true_best = inst.true_best_arm();

    std::vector<int> Nc(inst.K, 0);


    Vec theta_hat(inst.d, 0.0);

    std::vector<double> W_arm(inst.K, 0.0);

    std::vector<std::vector<int>> r01s(inst.K, std::vector<int> (2, 0));
    std::vector<std::vector<std::vector<int>>> y01s(inst.K, std::vector<std::vector<int>> (inst.K, std::vector<int>(2, 0)));

    int upp = inst.K + inst.K * (inst.K-1) / 2;

    std::vector<pii> indices(upp - inst.K);
    for (int j = 0, idx = 0; j < inst.K; ++j) {
        for (int k = j+1; k < inst.K; k++, idx++) {
            indices[idx] = {j, k};
        }
    }

    std::vector<std::vector<int>> Nd(inst.K, std::vector<int>(inst.K, 0));



    int t = 0, t_c = 0;
    


    for (t_c = 0; t < cfg.max_steps; ++t) {
     
        theta_hat = constrained_mle_logistic(r01s, y01s, inst.d, inst.S, cfg.zeta_c, cfg.zeta_d, cfg.mle_cfg, theta_hat, inst);

        Mat Hc = hessian_classic_only(Nc, theta_hat, cfg.zeta_c, inst);
        Mat Hd = hessian_duel_only(Nd, theta_hat, cfg.zeta_d, inst);
        Mat A  = info_matrix_A(Hc, Hd, inst.S, cfg.Rs_c, cfg.Rs_d, cfg.duel_bound) + Mat(inst.d, cfg.lambda);

        double Lt = lipschitz_Lt_bernoulli(t_c, t - t_c, inst.S);
        double beta = beta_t(cfg.delta, inst.d, inst.S, Lt);

        if (t > inst.d && stop_condition(inst, theta_hat, A, beta)) {
            break;
        }

        int arm = int(rng.uniform01() * upp);
        if (arm == upp) {
            arm--;
        }
        if (arm < inst.K) {
            r01s[arm][sample_reward(inst, arm, rng)]++;
            Nc[arm] += 1;
            t_c += 1;
        } else {
            arm -= inst.K;
            auto [j, k] = indices[arm];
            y01s[j][k][sample_duel_outcome(inst, j, k, rng)]++;
            Nd[j][k]++;
        }
    }

    out.stop_time = t;
    out.pred_best = predicted_best_arm(inst, theta_hat);
    out.correct = (out.pred_best == out.true_best);
    return out;
}
