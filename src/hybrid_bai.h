#pragma once
#include <vector>
#include <cmath>
#include <limits>
#include <iostream>
#include "lin_alg.h"
#include "rng.h"
#include "mle.h"
#include "env.h"
#include "cholesky.h"

// --- beta_t(delta) ---
inline double beta_t(double delta, int d, double S, double L_t) {
    double ct = std::min(1.0, double(d) / (2.0 * S * L_t));
    double val = std::log(1.0/delta) - (double)d * std::log(ct) + 2.0 * S * L_t * ct;
    return val;
}

// Bernoulli/logistic L_t upper bound (CR-GLB Table 1): (1 + S/2)(t-1)
inline double lipschitz_Lt_bernoulli(int t_c, int t_d, double S) {
    return (1.0 + S/2.0) * (t_c + 2*t_d);
}

struct HybridConfig {
    double delta = 0.05;
    double duel_bound = 2.;
    int max_steps = 200000;

    int fw_iters = 20;

    // self-concordance constants and zeta scalings
    double Rs_c = 1.0;
    double Rs_d = 1.0;
    double zeta_c = 1.0;
    double zeta_d = 1.0;
    double sc, sd;

    double lambda = 1e-6;    // ridge added to A_t for invertibility

    MLEConfig mle_cfg;

    bool reward_only = false;
    bool duel_only = false;
    double cc = 1., cd = 1.; // cost_c, cost_d

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

inline Mat hessian_classic_only(const std::vector<int> Nc, const Vec& theta_hat, double zeta_c, const Instance& inst) {
    Mat H(inst.d, 0.0);
    for (int i = 0; i < inst.K; ++i) {
        double z = dot(inst.x[i], theta_hat);
        double w = mu_prime(z);
        H = H + (1.0 * Nc[i] * (1.0/zeta_c) * w) * outer(inst.x[i]);
    }
    return H;
}

inline Mat hessian_duel_only(const std::vector<std::vector<int>> Nd, const Vec& theta_hat, double zeta_d, const Instance& inst) {
    Mat H(inst.d, 0.0);
    for (int j = 0; j < inst.K; ++j) {
        for (int k = j+1; k < inst.K; ++k) {
            double z = dot(inst.g[j][k], theta_hat);
            double w = mu_prime(z);
            H = H + (1.0 * Nd[j][k] * (1.0 / zeta_d) * w) * outer(inst.g[j][k]);
        }
    }
    return H;
}

// A_t := H_c /(2(1 + S Rs_c)) + H_d /(2(1 + 2 S Rs_d))
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
    return ac * Hc + ad * Hd;
}

inline int argmax_score(const std::vector<double>& v) {
    int id = 0;
    for (int i = 1; i < (int)v.size(); ++i) if (v[i] > v[id]) {
        id = i;
    }
    return id;
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

// stopping: for i_hat, for all i!=i_hat, require
// g^T theta_hat - sqrt(beta * g^T A^{-1} g) > 0
inline bool stop_condition(
    const Instance& inst,
    const Vec& theta_hat,
    const Mat& A,
    double beta,
    bool PrintInfo = false
) {
    int ihat = predicted_best_arm(inst, theta_hat);

    bool ret = true;
    for (int i = 0; i < inst.K; ++i) if (i != ihat) {
        double gap = dot(inst.g[ihat][i], theta_hat);
        double q = quad_form_inv_spd(A, inst.g[ihat][i]);
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

// choose which competitor is most uncertain (largest rad / max(mean_gap,eps))
inline int worst_competitor(
    const Instance& inst,
    const Vec& theta_hat,
    const Mat& A,
    double beta
) {
    int ihat = predicted_best_arm(inst, theta_hat);

    int worst = -1;
    double bestScore = -1.0;

    for (int i = 0; i < inst.K; ++i) if (i != ihat) {
        Vec g = inst.x[ihat] - inst.x[i];
        double mean_gap = std::max(1e-9, dot(g, theta_hat));
        double q = quad_form_inv_spd(A, g);
        double rad = std::sqrt(std::max(0.0, beta * q));
        double score = rad / mean_gap;
        if (score > bestScore) { bestScore = score; worst = i; }
    }
    if (worst < 0) worst = (ihat==0?1:0);
    return worst;
}


inline Mat compute_A_of_w(
    const Instance& inst,
    const Vec& theta_hat,
    const HybridConfig& cfg,
    const std::vector<double>& w_arm,
    const std::vector<std::vector<double>>& w_pair
) {
    int d = inst.d;
    int K = inst.K;

    Mat Ac(d, cfg.lambda), Ad(d, cfg.lambda);

    // classic
    for (int i = 0; i < K; ++i) {
        double wi = w_arm[i];
        if (wi <= 0.0) continue;
        double z = dot(inst.x[i], theta_hat);
        double curv = mu_prime(z) / cfg.zeta_c; // logistic: mu'(z)=sigmoid(z)(1-sigmoid(z))
        // Ac += wi * curv * x x^T
        for (int a = 0; a < d; ++a) for (int b = 0; b < d; ++b)
            Ac(a,b) += wi * curv * inst.x[i][a] * inst.x[i][b];
    }

    // dueling
    for (int j = 0; j < K; ++j) {
        for (int k = j + 1; k < K; ++k) {
            double wjk = w_pair[j][k];
            if (wjk <= 0.0) continue;
            Vec g = inst.x[j] - inst.x[k];
            double z = dot(g, theta_hat);
            double curv = mu_prime(z) / cfg.zeta_d;
            for (int a = 0; a < d; ++a) {
                for (int b = 0; b < d; ++b) {
                    Ad(a,b) += wjk * curv * g[a] * g[b];
                }
            }
        }
    }

    double wc = 1.0 / (2.0 * (1.0 + inst.S * cfg.Rs_c));
    double wd = 1.0 / (2.0 * (1.0 + 2.0 * inst.S * cfg.Rs_d));

    Mat A = wc * Ac + wd * Ad;
    return A;
}


static void compute_optimal_proportions_track_stop(
    const Instance& inst,
    const Vec& theta_hat,
    const HybridConfig& cfg,
    std::vector<double>& w_arm,
    std::vector<std::vector<double>>& w_pair
) {
    const int K = inst.K;

    for (int i = 0; i < K; ++i) w_arm[i] = 0.0;
    for (int j = 0; j < K; ++j) for (int k = 0; k < K; ++k) w_pair[j][k] = 0.0;

    const int P = K * (K - 1) / 2;

    if (cfg.reward_only) {
        const double mass_arm = 1.0 / (double)K;
        for (int i = 0; i < K; ++i) w_arm[i] = mass_arm;
    } else if (cfg.duel_only) {
        const double mass_pair = 1.0 / (double)P;
        for (int j = 0; j < K; ++j)
            for (int k = j + 1; k < K; ++k)
                w_pair[j][k] = mass_pair;
    } else {
        const double mass = 1.0 / (double)(K + P);
        for (int i = 0; i < K; ++i) {
            w_arm[i] = mass;
        }
        for (int j = 0; j < K; ++j) {
            for (int k = j + 1; k < K; ++k) {
                w_pair[j][k] = mass;
            }
        }
    }

    const int ihat = predicted_best_arm(inst, theta_hat);

    std::vector<double> curv_arm;
    std::vector<std::vector<double>> curv_pair;

    if (!cfg.duel_only) {
        curv_arm.resize(K);
        for (int i = 0; i < K; ++i) {
            const double z = dot(inst.x[i], theta_hat);
            curv_arm[i] = mu_prime(z) / cfg.zeta_c;
        }
    }
    if (!cfg.reward_only) {
        curv_pair.assign(K, std::vector<double>(K, 0.0));
        for (int j = 0; j < K; ++j) {
            for (int k = j + 1; k < K; ++k) {
                const double z = dot(inst.g[j][k], theta_hat);
                curv_pair[j][k] = mu_prime(z) / cfg.zeta_d;
            }
        }
    }

    for (int m = 0; m < cfg.fw_iters; ++m) {
        Mat A = compute_A_of_w(inst, theta_hat, cfg, w_arm, w_pair);

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
                const double sc = cfg.sc * curv_arm[i] * (proj * proj);
                if (sc > sc_best) { 
                    sc_best = sc; 
                    i_star = i; 
                }
            }
        }

        int j_star = 0, k_star = 1;
        double sd_best = -1.0;
        if (!cfg.reward_only) {
            for (int j = 0; j < K; ++j) {
                for (int k = j + 1; k < K; ++k) {
                    const double proj = dot(inst.g[j][k], u);
                    const double sd = cfg.sd * curv_pair[j][k] * (proj * proj);
                    if (sd > sd_best) { 
                        sd_best = sd; 
                        j_star = j; 
                        k_star = k; 
                    }
                }
            }
        }

        const double gamma = 2.0 / (double)(m + 2);
        const double one_minus = 1.0 - gamma;

        for (int i = 0; i < K; ++i) w_arm[i] *= one_minus;
        for (int j = 0; j < K; ++j) for (int k = j + 1; k < K; ++k) w_pair[j][k] *= one_minus;

        if (sc_best >= sd_best) {
            w_arm[i_star] += gamma;
        } else {
            w_pair[j_star][k_star] += gamma;
        }
    }

    // normalize
    double sum = 0.0;
    for (int i = 0; i < K; ++i) sum += w_arm[i];
    for (int j = 0; j < K; ++j) for (int k = j + 1; k < K; ++k) sum += w_pair[j][k];

    if (sum > 0) {
        const double inv = 1.0 / sum;
        for (int i = 0; i < K; ++i) w_arm[i] *= inv;
        for (int j = 0; j < K; ++j) for (int k = j + 1; k < K; ++k) w_pair[j][k] *= inv;
    }
}


static void cost_optimal_proportions_track_stop(
    const Instance& inst,
    const Vec& theta_hat,
    const HybridConfig& cfg,
    std::vector<double>& w_arm,
    std::vector<std::vector<double>>& w_pair
) {
    const int K = inst.K;

    for (int i = 0; i < K; ++i) w_arm[i] = 0.0;
    for (int j = 0; j < K; ++j) for (int k = 0; k < K; ++k) w_pair[j][k] = 0.0;

    // ---------- cost-aware initialization (both modalities enabled) ----------
    // w_arm[i] ∝ 1/cc, w_pair[j][k] ∝ 1/cd, then normalize to sum=1
    const double inv_cc = 1.0 / std::max(1e-12, cfg.cc);
    const double inv_cd = 1.0 / std::max(1e-12, cfg.cd);

    for (int i = 0; i < K; ++i) w_arm[i] = inv_cc;
    for (int j = 0; j < K; ++j) {
        for (int k = j + 1; k < K; ++k) {
            w_pair[j][k] = inv_cd;
        }
    }

    double sum0 = 0.0;
    for (int i = 0; i < K; ++i) {
        sum0 += w_arm[i];
    }
    for (int j = 0; j < K; ++j) {
        for (int k = j + 1; k < K; ++k) {
            sum0 += w_pair[j][k];
        }
    }

    if (sum0 > 0) {
        const double inv = 1.0 / sum0;
        for (int i = 0; i < K; ++i) w_arm[i] *= inv;
        for (int j = 0; j < K; ++j) for (int k = j + 1; k < K; ++k) w_pair[j][k] *= inv;
    }

    const int ihat = predicted_best_arm(inst, theta_hat);

    // ---------- precompute curvatures (both modalities) ----------
    std::vector<double> curv_arm(K, 0.0);
    for (int i = 0; i < K; ++i) {
        const double z = dot(inst.x[i], theta_hat);
        curv_arm[i] = mu_prime(z) / cfg.zeta_c;
    }

    std::vector<std::vector<double>> curv_pair(K, std::vector<double>(K, 0.0));
    for (int j = 0; j < K; ++j) {
        for (int k = j + 1; k < K; ++k) {
            const double z = dot(inst.g[j][k], theta_hat);
            curv_pair[j][k] = mu_prime(z) / cfg.zeta_d;
        }
    }

    // ---------- Frank–Wolfe ----------
    for (int m = 0; m < cfg.fw_iters; ++m) {
        Mat A = compute_A_of_w(inst, theta_hat, cfg, w_arm, w_pair);
        Chol L = chol_spd(A);

        int idag = -1;
        double best_val = -1.0;
        for (int i = 0; i < K; ++i) if (i != ihat) {
            const double val = quad_form_inv_chol(L, inst.g[ihat][i]);
            if (val > best_val) { best_val = val; idag = i; }
        }
        if (idag < 0) return;

        Vec u = solve_chol(L, inst.g[ihat][idag]);

        int i_star = 0;
        double sc_best = -1.0;
        for (int i = 0; i < K; ++i) {
            const double proj = dot(inst.x[i], u);
            const double sc = cfg.sc * curv_arm[i] * (proj * proj);
            if (sc > sc_best) { sc_best = sc; i_star = i; }
        }

        int j_star = 0, k_star = 1;
        double sd_best = -1.0;
        for (int j = 0; j < K; ++j) {
            for (int k = j + 1; k < K; ++k) {
                const double proj = dot(inst.g[j][k], u);
                const double sd = cfg.sd * curv_pair[j][k] * (proj * proj);
                if (sd > sd_best) { sd_best = sd; j_star = j; k_star = k; }
            }
        }

        const double gamma = 2.0 / (double)(m + 2);
        const double one_minus = 1.0 - gamma;

        for (int i = 0; i < K; ++i) w_arm[i] *= one_minus;
        for (int j = 0; j < K; ++j) for (int k = j + 1; k < K; ++k) w_pair[j][k] *= one_minus;

        // ---------- cost-aware choice: compare gain per unit cost ----------
        const double eff_sc = sc_best / std::max(1e-12, cfg.cc);
        const double eff_sd = sd_best / std::max(1e-12, cfg.cd);

        if (eff_sc >= eff_sd) w_arm[i_star] += gamma;
        else                  w_pair[j_star][k_star] += gamma;
    }

    // ---------- normalize ----------
    double sum = 0.0;
    for (int i = 0; i < K; ++i) sum += w_arm[i];
    for (int j = 0; j < K; ++j) for (int k = j + 1; k < K; ++k) sum += w_pair[j][k];

    if (sum > 0) {
        const double inv = 1.0 / sum;
        for (int i = 0; i < K; ++i) w_arm[i] *= inv;
        for (int j = 0; j < K; ++j) for (int k = j + 1; k < K; ++k) w_pair[j][k] *= inv;
    }
}


inline RunSummary run_cost(
    Instance& inst, 
    const HybridConfig& cfg, 
    RNG& rng
) {
    RunSummary out;
    out.true_best = inst.true_best_arm();

    std::vector<int> Nc(inst.K, 0);
    std::vector<std::vector<int>> Nd(inst.K, std::vector<int>(inst.K, 0));

    Vec theta_hat(inst.d, 0.0);

    std::vector<double> W_arm(inst.K, 0.0);
    std::vector<std::vector<double>> W_pair(inst.K, std::vector<double>(inst.K, 0.0));

    
    std::vector<std::vector<int>> r01s(inst.K, std::vector<int> (2, 0));
    std::vector<std::vector<std::vector<int>>> y01s(inst.K, std::vector<std::vector<int>> (inst.K, std::vector<int>(2, 0)));


    int t = 0, t_c = 0;
    
    for (; t < inst.K * inst.d; t++, t_c++) {
        int a = t % inst.K;
        int r = sample_reward(inst, a, rng);
        r01s[a][r]++;
        Nc[a]++;
    }

    for (; t < cfg.max_steps; ++t) {

        // if (t % 5000 == 0) {
        //     std::cout << "Step: " << t << "\n";
        // }
        theta_hat = constrained_mle_logistic(r01s, y01s, inst.d, inst.S, cfg.zeta_c, cfg.zeta_d, cfg.mle_cfg, theta_hat, inst);

        Mat Hc = hessian_classic_only(Nc, theta_hat, cfg.zeta_c, inst);
        Mat Hd = hessian_duel_only(Nd, theta_hat, cfg.zeta_d, inst);
        Mat A  = info_matrix_A(Hc, Hd, inst.S, cfg.Rs_c, cfg.Rs_d, cfg.duel_bound) + Mat(inst.d, cfg.lambda);

        double Lt = lipschitz_Lt_bernoulli(t_c, t - t_c, inst.S);
        double beta = beta_t(cfg.delta, inst.d, inst.S, Lt);

        if (t > inst.d && stop_condition(inst, theta_hat, A, beta)) {
            break;
        }


        std::vector<double> w_arm(inst.K, 0.0);
        std::vector<std::vector<double>> w_pair(inst.K, std::vector<double>(inst.K, 0.0));
        cost_optimal_proportions_track_stop(inst, theta_hat, cfg, w_arm, w_pair);
        for (int i = 0; i < inst.K; ++i) {
            W_arm[i] += w_arm[i];
        }
        for (int j = 0; j < inst.K; ++j) {
            for (int k = j + 1; k < inst.K; ++k) {
                W_pair[j][k] += w_pair[j][k];
            }
        }

        int best_i = 0;
        double best_val = (double)Nc[0] - W_arm[0];
        for (int i = 1; i < inst.K; ++i) {
            if (double v = (double)Nc[i] - W_arm[i]; v < best_val) { 
                best_val = v; 
                best_i = i;
            }
        }

        int best_j = 0, best_k = 1;
        double best_duel_val = (double)Nd[0][1] - W_pair[0][1];
        for (int j = 0; j < inst.K; ++j) for (int k = j + 1; k < inst.K; ++k) {
            if (double v = (double)Nd[j][k] - W_pair[j][k]; v < best_duel_val) { 
                best_duel_val = v; 
                best_j = j;
                best_k = k;
            }
        }

        bool do_duel = (best_duel_val < best_val);

        if (!do_duel) {
            int r = sample_reward(inst, best_i, rng);
            r01s[best_i][r]++;
            Nc[best_i]++; t_c++;
        } else {
            int y = sample_duel_outcome(inst, best_j, best_k, rng);
            y01s[best_j][best_k][y]++;
            Nd[best_j][best_k]++;
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

    std::vector<int> Nc(inst.K, 0);
    std::vector<std::vector<int>> Nd(inst.K, std::vector<int>(inst.K, 0));

    Vec theta_hat(inst.d, 0.0);

    std::vector<double> W_arm(inst.K, 0.0);
    std::vector<std::vector<double>> W_pair(inst.K, std::vector<double>(inst.K, 0.0));

    
    std::vector<std::vector<int>> r01s(inst.K, std::vector<int> (2, 0));
    std::vector<std::vector<std::vector<int>>> y01s(inst.K, std::vector<std::vector<int>> (inst.K, std::vector<int>(2, 0)));

    int t = 0, t_c = 0;
    
    for (; t < inst.K * inst.d; t++, t_c++) {
        int a = t % inst.K;
        int r = sample_reward(inst, a, rng);
        r01s[a][r]++;
        Nc[a]++;
    }


    for (; t < cfg.max_steps; ++t) {

        // if (t % 5000 == 0) {
        //     std::cout << "Step: " << t << "\n";
        // }
        theta_hat = constrained_mle_logistic(r01s, y01s, inst.d, inst.S, cfg.zeta_c, cfg.zeta_d, cfg.mle_cfg, theta_hat, inst);

        Mat Hc = hessian_classic_only(Nc, theta_hat, cfg.zeta_c, inst);
        Mat Hd = hessian_duel_only(Nd, theta_hat, cfg.zeta_d, inst);
        Mat A  = info_matrix_A(Hc, Hd, inst.S, cfg.Rs_c, cfg.Rs_d, cfg.duel_bound) + Mat(inst.d, cfg.lambda);

        double Lt = lipschitz_Lt_bernoulli(t_c, t - t_c, inst.S);
        double beta = beta_t(cfg.delta, inst.d, inst.S, Lt);

        if (t > inst.d && stop_condition(inst, theta_hat, A, beta)) {
            break;
        }


        std::vector<double> w_arm(inst.K, 0.0);
        std::vector<std::vector<double>> w_pair(inst.K, std::vector<double>(inst.K, 0.0));
        compute_optimal_proportions_track_stop(inst, theta_hat, cfg, w_arm, w_pair);
        for (int i = 0; i < inst.K; ++i) {
            W_arm[i] += w_arm[i];
        }
        for (int j = 0; j < inst.K; ++j) {
            for (int k = j + 1; k < inst.K; ++k) {
                W_pair[j][k] += w_pair[j][k];
            }
        }

        int best_i = 0;
        double best_val = (double)Nc[0] - W_arm[0];
        for (int i = 1; i < inst.K; ++i) {
            if (double v = (double)Nc[i] - W_arm[i]; v < best_val) { 
                best_val = v; 
                best_i = i;
            }
        }

        int best_j = 0, best_k = 1;
        double best_duel_val = (double)Nd[0][1] - W_pair[0][1];
        for (int j = 0; j < inst.K; ++j) for (int k = j + 1; k < inst.K; ++k) {
            if (double v = (double)Nd[j][k] - W_pair[j][k]; v < best_duel_val) { 
                best_duel_val = v; 
                best_j = j;
                best_k = k;
            }
        }

        bool do_duel = (best_duel_val < best_val);

        if (!do_duel) {
            int r = sample_reward(inst, best_i, rng);
            r01s[best_i][r]++;
            Nc[best_i]++; t_c++;
        } else {
            int y = sample_duel_outcome(inst, best_j, best_k, rng);
            y01s[best_j][best_k][y]++;
            Nd[best_j][best_k]++;
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

        // if (t % 5000 == 0) {
        //     std::cout << "Step: " << t << "\n";
        // }

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
        // if (t % 5000 == 0) {
        //     std::cout << "Step: " << t << "\n";
        // }

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
