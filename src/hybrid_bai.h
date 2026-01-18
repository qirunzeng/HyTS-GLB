#pragma once
#include <vector>
#include <cmath>
#include <limits>
#include <iostream>
#include "lin_alg.h"
#include "rng.h"
#include "instance.h"
#include "mle.h"
#include "glm_logistic.h"

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
    int max_steps = 20000;

    int fw_iters = 20;

    // self-concordance constants and zeta scalings
    double Rs_c = 1.0;
    double Rs_d = 1.0;
    double zeta_c = 1.0;
    double zeta_d = 1.0;

    double lambda = 1e-6;    // ridge added to A_t for invertibility

    MLEConfig mle_cfg;

    bool reward_only = false;
    bool duel_only = false;
};

struct RunSummary {
    int stop_time = 0;
    int pred_best = -1;
    int true_best = -1;
    bool correct = false;
};

inline Mat hessian_classic_only(const std::vector<Obs>& data, const Vec& theta_hat, int d, double zeta_c) {
    Mat H(d, 0.0);
    for (const auto& ob: data) if (!ob.is_duel) {
        double z = dot(ob.feat, theta_hat);
        double w = mu_prime(z);
        H = H + ((1.0/zeta_c) * w) * outer(ob.feat);
    }
    return H;
}

inline Mat hessian_duel_only(const std::vector<Obs>& data, const Vec& theta_hat, int d, double zeta_d) {
    Mat H(d, 0.0);
    for (const auto& ob: data) if (ob.is_duel) {
        double z = dot(ob.feat, theta_hat);
        double w = mu_prime(z);
        H = H + ((1.0/zeta_d) * w) * outer(ob.feat);
    }
    return H;
}

// A_t := H_c /(2(1 + S Rs_c)) + H_d /(2(1 + 2 S Rs_d))
inline Mat info_matrix_A(
    const Mat& Hc,
    const Mat& Hd,
    double S,
    double Rs_c,
    double Rs_d
) {
    double ac = 1.0 / (2.0 * (1.0 + S * Rs_c));
    double ad = 1.0 / (2.0 * (1.0 + 2.0 * S * Rs_d));
    return ac * Hc + ad * Hd;
}

inline int argmax_score(const std::vector<double>& v) {
    int id=0;
    for (int i = 1; i < (int)v.size(); ++i) if (v[i] > v[id]) {
        id = i;
    }
    return id;
}

inline int predicted_best_arm(const Instance& inst, const Vec& theta_hat) {
    int best=0;
    double bestv = dot(inst.x[0], theta_hat);
    for (int i=1;i<inst.K;++i) {
        double v = dot(inst.x[i], theta_hat);
        if (v > bestv) { 
            bestv=v; 
            best=i; 
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
    double beta, bool printInfo=false
) {
    int ihat = predicted_best_arm(inst, theta_hat);

    bool ret = true;
    if (printInfo) std::cout << "Predicted best arm: " << ihat << "\n";
    for (int i = 0; i < inst.K; ++i) if (i != ihat) {
        Vec g = inst.x[ihat] - inst.x[i];
        double mean_gap = dot(g, theta_hat);
        double q = quad_form_inv_spd(A, g);
        double rad = std::sqrt(std::max(0.0, beta * q));
        if (printInfo) std::cout << "  vs arm " << i << ": mean gap = " << mean_gap << ", rad = " << rad << "\n";
        // if (!(mean_gap > rad)) return false;
        if (!(mean_gap - rad > 1e-12)) ret = false;
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

    for (int i=0;i<inst.K;++i) if (i!=ihat) {
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


// simulate one classic Bernoulli reward
inline int sample_classic_reward(const Instance& inst, int arm, RNG& rng) {
    double z = dot(inst.x[arm], inst.theta_star);
    double p = sigmoid(z);
    return (rng.uniform01() < p) ? 1 : 0;
}

// simulate one dueling outcome y ~ Bernoulli(sigmoid((x_j-x_k)^T theta*))
inline int sample_duel_outcome(const Instance& inst, int j, int k, RNG& rng) {
    Vec g = inst.x[j] - inst.x[k];
    double z = dot(g, inst.theta_star);
    double p = sigmoid(z);
    return (rng.uniform01() < p) ? 1 : 0;
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
            for (int a = 0; a < d; ++a) for (int b = 0; b < d; ++b)
                Ad(a,b) += wjk * curv * g[a] * g[b];
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
    int K = inst.K;

    // 1) initialize
    for (int i = 0; i < K; ++i) w_arm[i] = 0.0;
    for (int j = 0; j < K; ++j) for (int k = 0; k < K; ++k) w_pair[j][k] = 0.0;

    int P = K * (K - 1) / 2;

    if (cfg.reward_only) {
        double mass_arm = 1.0 / (double)K;
        for (int i = 0; i < K; ++i) {
            w_arm[i] = mass_arm;
        }
    } else if (cfg.duel_only) {
        double mass_pair = 1.0 / (double)P;
        for (int j = 0; j < K; ++j) {
            for (int k = j + 1; k < K; ++k) {
                w_pair[j][k] = mass_pair;
            }
        }
    } else {
        double mass = 1.0 / (double)(K + P);
        for (int i = 0; i < K; ++i) {
            w_arm[i] = mass;
        }
        for (int j = 0; j < K; ++j) {
            for (int k = j + 1; k < K; ++k) {
                w_pair[j][k] = mass;
            }
        }
    }
    

    int ihat = predicted_best_arm(inst, theta_hat);

    for (int m = 0; m < cfg.fw_iters; ++m) {
        Mat A = compute_A_of_w(inst, theta_hat, cfg, w_arm, w_pair);

        int idag = -1;
        double best_val = -1.0;

        for (int i = 0; i < K; ++i) if (i != ihat) {
            Vec g = inst.x[ihat] - inst.x[i];
            double val = quad_form_inv_spd(A, g);
            if (val > best_val) { 
                best_val = val; 
                idag = i; 
            }
        }
        if (idag < 0) {
            return;
        }

        Vec g_dag = inst.x[ihat] - inst.x[idag];

        Vec u = solve_spd_cholesky(A, g_dag);

        int i_star = 0;
        double sc_best = -1.0;
        if (!cfg.duel_only) {
            for (int i = 0; i < K; ++i) {
                double z = dot(inst.x[i], theta_hat); // 内积
                double curv = mu_prime(z) / cfg.zeta_c;
                double proj = dot(inst.x[i], u);
                double sc = curv * (proj * proj);
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
                    Vec g = inst.x[j] - inst.x[k];
                    double z = dot(g, theta_hat);
                    double curv = mu_prime(z) / cfg.zeta_d;
                    double proj = dot(g, u);
                    double sd = curv * (proj * proj);
                    if (sd > sd_best) { 
                        sd_best = sd; 
                        j_star = j; 
                        k_star = k; 
                    }
                }
            }
        }

        double gamma = 2.0 / (double)(m + 2);

        if (sc_best >= sd_best) {
            for (int i = 0; i < K; ++i) {
                w_arm[i] *= (1.0 - gamma);
            }
            for (int j = 0; j < K; ++j) for (int k = j + 1; k < K; ++k) {
                w_pair[j][k] *= (1.0 - gamma);
            }
            w_arm[i_star] += gamma;
        } else {
            for (int i = 0; i < K; ++i) {
                w_arm[i] *= (1.0 - gamma);
            }
            for (int j = 0; j < K; ++j) for (int k = j + 1; k < K; ++k) {
                w_pair[j][k] *= (1.0 - gamma);
            }
            w_pair[j_star][k_star] += gamma;
        }
    }

    double sum = 0.0;
    for (int i = 0; i < K; ++i) sum += w_arm[i];
    for (int j = 0; j < K; ++j) for (int k = j + 1; k < K; ++k) sum += w_pair[j][k];
    if (sum > 0) {
        for (int i = 0; i < K; ++i) w_arm[i] /= sum;
        for (int j = 0; j < K; ++j) for (int k = j + 1; k < K; ++k) w_pair[j][k] /= sum;
    }
}

inline RunSummary run_one(const Instance& inst, const HybridConfig& cfg, RNG& rng) {
    RunSummary out;
    out.true_best = inst.true_best_arm();

    std::vector<int> Nc(inst.K, 0);
    std::vector<std::vector<int>> Nd(inst.K, std::vector<int>(inst.K, 0));

    std::vector<Obs> data;
    data.reserve(cfg.max_steps);

    Vec theta_hat(inst.d, 0.0);

    std::vector<double> W_arm(inst.K, 0.0);
    std::vector<std::vector<double>> W_pair(inst.K, std::vector<double>(inst.K, 0.0));

    int t = 0;
    
    for (int t_c = 0; t <= cfg.max_steps; ++t) {

        if (t % 500 == 0) {
            std::cout << "Step: " << t << "\n";
        }

        theta_hat = constrained_mle_logistic(data, inst.d, inst.S, cfg.zeta_c, cfg.zeta_d, cfg.mle_cfg);

        Mat Hc = hessian_classic_only(data, theta_hat, inst.d, cfg.zeta_c);
        Mat Hd = hessian_duel_only(data, theta_hat, inst.d, cfg.zeta_d);
        Mat A  = info_matrix_A(Hc, Hd, inst.S, cfg.Rs_c, cfg.Rs_d) + Mat(inst.d, cfg.lambda);

        double Lt = lipschitz_Lt_bernoulli(t_c, t-1 - t_c, inst.S);
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

        int t_now = (int)data.size();

        int best_i = 0;
        double best_val = (double)Nc[0] - W_arm[0];
        for (int i = 1; i < inst.K; ++i) {
            double v = (double)Nc[i] - W_arm[i];
            if (v < best_val) { best_val = v; best_i = i; }
        }

        int best_j = 0, best_k = 1;
        double best_duel_val = (double)Nd[0][1] - W_pair[0][1];
        for (int j = 0; j < inst.K; ++j) {
            for (int k = j + 1; k < inst.K; ++k) {
                double v = (double)Nd[j][k] - W_pair[j][k];
                if (v < best_duel_val) { 
                    best_duel_val = v; best_j = j; best_k = k; 
                }
            }
        }

        bool do_duel = (best_duel_val < best_val);

        if (t_now < inst.d) do_duel = false;

        if (!do_duel) {
            int arm = best_i;

            int r01 = sample_classic_reward(inst, arm, rng);
            Obs ob;
            ob.is_duel = false;
            ob.i = arm;
            ob.r01 = r01;
            ob.feat = inst.x[arm];
            data.push_back(ob);

            Nc[arm] += 1;
            t_c += 1;
        } else {
            int j = best_j, k = best_k;
            int y01 = sample_duel_outcome(inst, j, k, rng);
            Obs ob;
            ob.is_duel = true;
            ob.j = j; ob.k = k;
            ob.y01 = y01;
            ob.feat = inst.x[j] - inst.x[k];
            data.push_back(ob);

            Nd[j][k] += 1;
        }
    }

    out.stop_time = t;
    out.pred_best = predicted_best_arm(inst, theta_hat);
    out.correct = (out.pred_best == out.true_best);
    return out;
}


inline RunSummary run_rand(const Instance& inst, const HybridConfig& cfg, RNG& rng) {
    RunSummary out;
    out.true_best = inst.true_best_arm();

    std::vector<int> Nc(inst.K, 0);

    std::vector<Obs> data;
    data.reserve(cfg.max_steps);

    Vec theta_hat(inst.d, 0.0);

    std::vector<double> W_arm(inst.K, 0.0);

    int t = 0;
    
    for (int t_c = 0; t <= cfg.max_steps; ++t) {

        if (t % 500 == 0) {
            std::cout << "Step: " << t << "\n";
        }

        theta_hat = constrained_mle_logistic(data, inst.d, inst.S, cfg.zeta_c, cfg.zeta_d, cfg.mle_cfg);

        Mat Hc = hessian_classic_only(data, theta_hat, inst.d, cfg.zeta_c);
        Mat Hd = hessian_duel_only(data, theta_hat, inst.d, cfg.zeta_d);
        Mat A  = info_matrix_A(Hc, Hd, inst.S, cfg.Rs_c, cfg.Rs_d) + Mat(inst.d, cfg.lambda);

        double Lt = lipschitz_Lt_bernoulli(t_c, t-1 - t_c, inst.S);
        double beta = beta_t(cfg.delta, inst.d, inst.S, Lt);

        if (t > inst.d && stop_condition(inst, theta_hat, A, beta)) {
            break;
        }

        int arm = rand() % inst.K;

        int r01 = sample_classic_reward(inst, arm, rng);
        Obs ob;
        ob.is_duel = false;
        ob.i = arm;
        ob.r01 = r01;
        ob.feat = inst.x[arm];
        data.push_back(ob);

        Nc[arm] += 1;
        t_c += 1;
    }

    out.stop_time = t;
    out.pred_best = predicted_best_arm(inst, theta_hat);
    out.correct = (out.pred_best == out.true_best);
    return out;
}
