#include <iostream>
#include <string>
#include <unordered_map>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <filesystem>
#include <cmath>

#include "rng.h"
#include "instance.h"
#include "hybrid_bai.h"
#include "baseline_glgape.h"
#include "env.h"
#include "baseline_rageglm.h"

static bool has(const std::unordered_map<std::string,std::string>& mp, const std::string& k) {
    return mp.find(k)!=mp.end();
}
static std::string get(const std::unordered_map<std::string,std::string>& mp, const std::string& k, const std::string& def) {
    auto it=mp.find(k);
    return (it==mp.end())?def:it->second;
}
static int geti(const std::unordered_map<std::string,std::string>& mp, const std::string& k, int def) {
    return has(mp,k)?std::stoi(mp.at(k)):def;
}
static double getd(const std::unordered_map<std::string,std::string>& mp, const std::string& k, double def) {
    return has(mp,k) ? std::stod(mp.at(k)) : def;
}

static void usage() {
    std::cerr << "Pls refer to README for usage.\n";
}

static std::string fmt_S(double S) {
    double r = std::round(S);
    if (std::fabs(S - r) < 1e-12) return std::to_string((long long)r);

    std::ostringstream oss;
    oss << std::setprecision(12) << std::fixed << S;
    std::string s = oss.str();
    while (!s.empty() && s.back() == '0') s.pop_back();
    if (!s.empty() && s.back() == '.') s.pop_back();
    for (char& c : s) if (c == '.') c = 'p';
    return s;
}

static std::string build_outfile(
    const std::string& algo, int d, double S, int K,
    bool reward_only, bool duel_only, double delta
) {
    std::string prefix;
    if (algo == "hybrid") {
        // Both if neither-only flag is set; otherwise Reward_Only
        if (!reward_only && !duel_only) prefix = "Hybrid_Both";
        else if (reward_only) prefix = "Reward_Only";
        else prefix = "Dueling_Only";
    } else {
        prefix = algo;
    }

    std::ostringstream oss;
    oss << prefix << "_" << d << "_" << fmt_S(S) << "_" << K << "_" << delta << ".txt";
    return oss.str();
}

auto init = [](Instance &inst) -> double {
    double max_m = -1.;
    for (int i = 0; i < inst.K; ++i) {
        inst.dots[i] = dot(inst.x[i], inst.theta_star);
        inst.means[i] = sigmoid(inst.dots[i]);
        // std::cout << max_m << " " << inst.means[i] << std::endl;
        if (inst.dots[i] > max_m) {
            max_m = inst.dots[i];
            inst.best = i;
        }
    }
    double duel_bound = 0.;
    for (int j = 0; j < inst.K; ++j) {
        for (int k = 0; k < inst.K; ++k) {
            inst.g[j][k] = inst.x[j] - inst.x[k];
            inst.gap_dots[j][k] = dot(inst.g[j][k], inst.theta_star);
            duel_bound = std::max(duel_bound, std::sqrt(dot(inst.g[j][k], inst.g[j][k])));
            inst.gaps[j][k] = sigmoid(inst.gap_dots[j][k]);
        }
    }
    // std::cout << ">>> Max g: " << duel_bound << std::endl;
    return duel_bound;
};


static std::string fmt_num(double x) {
    std::ostringstream oss;
    oss << std::setprecision(12) << std::fixed << x;
    std::string s = oss.str();
    while (!s.empty() && s.back() == '0') s.pop_back();
    if (!s.empty() && s.back() == '.') s.pop_back();
    for (char& c : s) if (c == '.') c = 'p';
    return s;
}


int main(int argc, char** argv) {
    std::unordered_map<std::string,std::string> mp;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("--", 0) == 0) {
            std::string val = "1";
            if (i + 1 < argc && std::string(argv[i+1]).rfind("--", 0) != 0) {
                val = std::string(argv[i+1]);
                ++i;
            }
            mp[key] = val;
        }
    }

    std::string mode = get(mp, "--mode", "");
    if (mode.empty()) { 
        usage(); 
        return 1; 
    }

    uint64_t seed = (uint64_t)std::stoull(get(mp, "--seed", "1"));
    RNG rng(seed);



    if (mode == "gen") {
        int K = geti(mp, "--K", 10);
        int d = geti(mp, "--d", 2);
        double S = getd(mp, "--S", 2.0);
        std::string out = get(mp, "--out", "instance.txt");

        Instance inst = generate_synthetic_instance(K, d, S, rng);
        save_instance(inst, out);

        std::cout << "Generated instance to: " << out << "\n";
        return 0;
    }

    if (mode == "cost") {
        // Generate data instead of --load (because you want multiple instances)
        int K = geti(mp, "--K", 10);
        int d = geti(mp, "--d", 2);
        double S = getd(mp, "--S", 2.0);

        double delta = getd(mp, "--delta", 0.2);
        int max_steps = geti(mp, "--max_steps", 200000);
        const int inst_per_group = 10;                       // fixed: 10 instances per group
        const int runs_per_instance = 1;                     // fixed: each instance run once

        // Ratios from 1:5 to 5:1 (9 settings)
        const std::vector<std::pair<int,int>> ratio_list = {
            {1,5},{1,4},{1,3},{1,2},{1,1},{2,1},{3,1},{4,1},{5,1}
        };

        std::filesystem::create_directories("../output");

        // Base config (cc/cd will change per ratio; duel_bound depends on instance)
        HybridConfig cfg_base;
        cfg_base.delta     = delta;
        cfg_base.max_steps = max_steps;

        cfg_base.Rs_c   = getd(mp, "--Rs_c", 1.0);
        cfg_base.Rs_d   = getd(mp, "--Rs_d", 1.0);
        cfg_base.zeta_c = getd(mp, "--zeta_c", 1.0);
        cfg_base.zeta_d = getd(mp, "--zeta_d", 1.0);
        cfg_base.lambda = getd(mp, "--lambda", 1e-6);

        for (size_t si = 0; si < ratio_list.size(); ++si) {
            int a = ratio_list[si].first;
            int b = ratio_list[si].second;

            // Normalize to cc+cd=2 while preserving ratio a:b
            double cc = 2.0 * (double)a / (double)(a + b);
            double cd = 2.0 * (double)b / (double)(a + b);

            HybridConfig cfg = cfg_base;
            cfg.cc = cc;
            cfg.cd = cd;

            // One output file per ratio setting
            std::string outname =
                "Cost_ratio" + std::to_string(a) + "to" + std::to_string(b)+ ".txt";
            std::string outpath = std::string("../output/") + outname;

            std::ofstream out(outpath, std::ios::out | std::ios::trunc);
            if (!out) throw std::runtime_error("cannot open for writing: " + outpath);

            out << "Seed = " << seed << ", Delta = " << delta << "\n";
            out << "K = " << K << ", d = " << d << ", S = " << fmt_S(S)
                << ", max_steps = " << max_steps
                << ", inst_per_group = " << inst_per_group
                << ", runs_per_instance = " << runs_per_instance
                << ", ratio = " << a << ":" << b
                << ", cc = " << cc << ", cd = " << cd
                << ", (cc+cd=" << (cc+cd) << ")\n";

            out << "Setting, Instance, InstanceSeed, Method, StopTime, Cost_c, Cost_d, TotalCost, Right\n";

            // Collect stats separately for two methods
            std::vector<double> tot_with_c, tot_with_d, tot_no_c, tot_no_d;
            std::vector<double> st_with, st_no;
            int succ_with = 0, succ_no = 0;
            long long n_with = 0, n_no = 0;

            // Each group has 10 instances; each instance run once
            for (int j = 0; j < inst_per_group; ++j) {

                // Instance seed: depends on setting, group, instance index
                uint64_t inst_seed =
                    seed
                    + 1000003ull * (uint64_t)si
                    + 131ull     * (uint64_t)j
                    + 17ull;

                RNG inst_rng(inst_seed);
                Instance inst = generate_synthetic_instance(K, d, S, inst_rng);
                inst.reallocate();
                double duel_bound = init(inst);

                cfg.duel_bound = duel_bound;
                cfg.get_sc_sd(inst.S);

                // 1) With-cost run (uses run_cost)
                {
                    RNG rrng(inst_seed ^ 0x369dea0f31a53f85ull);
                    RunSummary rs = run_cost(inst, cfg, rrng);

                    // total cost computed from returned (c_c, c_d)
                    double total_cost = cfg.cc * rs.c_c + cfg.cd * rs.c_d;

                    out << (si + 1) << ", " << (j + 1) << ", " << inst_seed
                        << ", WithCost"
                        << ", " << rs.stop_time
                        << ", " << rs.c_c
                        << ", " << rs.c_d
                        << ", " << total_cost
                        << ", " << (rs.correct ? 1 : 0) << "\n";

                    tot_with_c.push_back(rs.c_c);
                    tot_with_d.push_back(rs.c_d);
                    st_with.push_back((double)rs.stop_time);
                    succ_with += (rs.correct ? 1 : 0);
                    ++n_with;
                }

                // 2) No-cost run (uses run_one), but still compute cost at the end
                {
                    RNG rrng(inst_seed ^ 0x369dea0f31a53f85ull);
                    RunSummary rs = run_one(inst, cfg, rrng);

                    double total_cost = rs.c_c + rs.c_d;

                    out << (si + 1) << ", " << (j + 1) << ", " << inst_seed
                        << ", NoCost"
                        << ", " << rs.stop_time
                        << ", " << rs.c_c
                        << ", " << rs.c_d
                        << ", " << total_cost
                        << ", " << (rs.correct ? 1 : 0) << "\n";

                    tot_no_c.push_back(rs.c_c);
                    tot_no_d.push_back(rs.c_d);
                    st_no.push_back((double)rs.stop_time);
                    succ_no += (rs.correct ? 1 : 0);
                    ++n_no;
                }
            }

            // Summary (Average / StdDev)
            MeanStd ms_tot_with_c = mean_std(tot_with_c);
            MeanStd ms_tot_with_d = mean_std(tot_with_d);
            MeanStd ms_tot_no_c   = mean_std(tot_no_c);
            MeanStd ms_tot_no_d   = mean_std(tot_no_d);
            MeanStd ms_st_with  = mean_std(st_with);
            MeanStd ms_st_no    = mean_std(st_no);

            out << "Summary: WithCost"
                << ", Avg_C_Cost = " << ms_tot_with_c.mean
                << ", Std_C_Cost = " << ms_tot_with_c.stdev
                << ", Avg_D_Cost = " << ms_tot_with_d.mean
                << ", Std_D_Cost = " << ms_tot_with_d.stdev
                << ", AvgStopTime = " << ms_st_with.mean
                << ", StdStopTime = " << ms_st_with.stdev
                << ", SuccessRate = " << (n_with ? (double)succ_with / (double)n_with : 0.0)
                << "\n";

            out << "Summary: NoCost"
                << ", Avg_C_Cost = " << ms_tot_no_c.mean
                << ", Std_C_Cost = " << ms_tot_no_c.stdev
                << ", Avg_D_Cost = " << ms_tot_no_d.mean
                << ", Std_D_Cost = " << ms_tot_no_d.stdev
                << ", AvgStopTime = " << ms_st_no.mean
                << ", StdStopTime = " << ms_st_no.stdev
                << ", SuccessRate = " << (n_no ? (double)succ_no / (double)n_no : 0.0)
                << "\n";

            out.close();
            std::cout << "Wrote: " << outname << "\n";
        }

        return 0;
    }




    if (mode == "run") {
        std::string path = get(mp, "--load", "");
        if (path.empty()) { std::cerr << "--load is required in run mode\n"; return 1; }

        Instance inst = load_instance(path);
        std::string algo = get(mp, "--algo", "hybrid");
        int runs = geti(mp, "--runs", 1);

        inst.reallocate();
        double duel_bound = init(inst);
        std::cout << "Duel Bound: " << duel_bound << std::endl;
        
        // Optional console debug (keep if you want)
        for (int i = 0; i < inst.K; ++i) {
            std::cout << "Arm " << i << ": x_i^T theta* = " << inst.dots[i] << "\n";
        }
        std::cout << "i^star: " << inst.true_best_arm() << "\n";

        if (algo == "hybrid") {
            HybridConfig cfg;
            cfg.duel_bound = duel_bound;
            cfg.delta = getd(mp, "--delta", 0.2);
            cfg.max_steps = geti(mp, "--max_steps", 200000);

            cfg.Rs_c = getd(mp, "--Rs_c", 1.0);
            cfg.Rs_d = getd(mp, "--Rs_d", 1.0);
            cfg.zeta_c = getd(mp, "--zeta_c", 1.0);
            cfg.zeta_d = getd(mp, "--zeta_d", 1.0);
            cfg.lambda = getd(mp, "--lambda", 1e-6);

            cfg.reward_only = (geti(mp, "--reward_only", 0) != 0);
            cfg.duel_only = (geti(mp, "--duel_only", 0) != 0);
            cfg.get_sc_sd(inst.S);

            long long sumT = 0;
            int succ = 0;

            std::filesystem::create_directories("../output");
            std::string outname = build_outfile(algo, inst.d, inst.S, inst.K, cfg.reward_only, cfg.duel_only, cfg.delta);
            std::string outpath = std::string("../output/") + outname;

            std::cout << outname << std::endl;

            std::ofstream out(outpath, std::ios::out | std::ios::trunc);
            if (!out) throw std::runtime_error("cannot open for writing: " + outpath);

            out << "S = " 
                << fmt_S(inst.S) 
                << ", K = " 
                << inst.K 
                << ", d = " 
                << inst.d 
                << "\n"
                << "Round, Stop Time, Right\n";

            for (int r = 0; r < runs; ++r) {
                std::cout << "Run " << (r + 1) << "/" << runs << "\n";
                RNG rrng(seed + 10007ull * (uint64_t)r + 17ull);
                RunSummary rs = run_one(inst, cfg, rrng);
                sumT += rs.stop_time;
                succ += (rs.correct ? 1 : 0);
                out << (r + 1) << ", " << rs.stop_time << ", " << (rs.correct ? 1 : 0) << "\n";
            }

            out << "Algo = hybrid"
                << ", Runs = " << runs
                << ", Avg stop time = " << (double)sumT / (double)runs
                << ", Success rate = " << (double)succ / (double)runs
                << "\n";

            out.close();

            return 0;
        }

        else if (algo == "random") {
            HybridConfig cfg;
            cfg.delta           = getd(mp, "--delta",           0.2);
            cfg.max_steps       = geti(mp, "--max_steps",       200000);

            cfg.Rs_c            = getd(mp, "--Rs_c",            1.0);
            cfg.Rs_d            = getd(mp, "--Rs_d",            1.0);
            cfg.zeta_c          = getd(mp, "--zeta_c",          1.0);
            cfg.zeta_d          = getd(mp, "--zeta_d",          1.0);
            cfg.lambda          = getd(mp, "--lambda",          1e-6);
            bool duel_do        = getd(mp, "--duel",            0);
            cfg.get_sc_sd(inst.S);

            cfg.reward_only     = true;

            long long sumT = 0;
            int succ = 0;

            std::filesystem::create_directories("../output");
            std::string outname = build_outfile(algo, inst.d, inst.S, inst.K, cfg.reward_only, cfg.duel_only, cfg.delta);
            if (duel_do) outname = "Hy_" + outname;

            std::string outpath = std::string("../output/") + outname;

            std::ofstream out(outpath, std::ios::out | std::ios::trunc);
            if (!out) throw std::runtime_error("cannot open for writing: " + outpath);

            out << "S = " 
                << fmt_S(inst.S) 
                << ", K = " 
                << inst.K 
                << ", d = " 
                << inst.d 
                << "\n"
                << "Round, Stop Time, Right\n";

            for (int r = 0; r < runs; ++r) {
                std::cout << "Run " << (r + 1) << "/" << runs << "\n";
                RNG rrng(seed + 10007ull * (uint64_t)r + 17ull);
                RunSummary rs = duel_do ? run_rand_hybrid(inst, cfg, rrng) : run_rand(inst, cfg, rrng);
                sumT += rs.stop_time;
                succ += (rs.correct ? 1 : 0);
                out << (r + 1) << ", " << rs.stop_time << ", " << (rs.correct ? 1 : 0) << "\n";
            }

            out << "Algo = random"
                << ", Runs = " << runs
                << ", Avg stop time = " << (double)sumT / (double)runs
                << ", Success rate = " << (double)succ / (double)runs
                << "\n";

            out.close();

            return 0;
        }


        else if (algo == "glgape") {
            GLGapEConfig cfg;
            cfg.delta       = getd(mp, "--delta", 0.2);
            cfg.eps         = getd(mp, "--eps", 0.1);

            cfg.E           = geti(mp, "--E", -1);
            cfg.max_steps   = geti(mp, "--max_steps",       200000);

            cfg.c_mu = mu_prime(double(inst.S - 1));
            std::cout << "Low = " << cfg.c_mu << std::endl;

            cfg.downscale_C = (geti(mp, "--downscale_C", 0) != 0);
            cfg.C_scale     = getd(mp, "--C_scale", 1.0);

            long long sumT  = 0;
            int succ = 0;

            std::filesystem::create_directories("../output");
            std::string outname = build_outfile(algo, inst.d, inst.S, inst.K, false, false, cfg.delta);
            std::string outpath = std::string("../output/") + outname;

            std::ofstream out(outpath, std::ios::out | std::ios::trunc);
            if (!out) throw std::runtime_error("cannot open for writing: " + outpath);

            out << "S = " << fmt_S(inst.S) << ", K = " << inst.K << ", d = " << inst.d << "\n";
            out << "Round, Stop Time, Right\n";

            for (int r = 0; r < runs; ++r) {
                std::cout << "Run " << (r + 1) << "/" << runs << "\n";
                RNG rrng(seed + 10007ull * (uint64_t)r + 17ull);
                GLGapEResult rs = run_glgape_baseline(inst, cfg, rrng);
                sumT += rs.stop_t;
                succ += (rs.correct ? 1 : 0);

                out << (r + 1) << ", " << rs.stop_t << ", " << (rs.correct ? 1 : 0) << "\n";
            }

            // Optional console summary
            out << "Algo = glgape"
                << ", Runs = " << runs
                << ", Avg stop time = " << (double)sumT / (double)runs
                << ", Success rate = " << (double)succ / (double)runs
                << "\n";

            out.close();

            return 0;
        }

        
        else if (algo == "rageglm") {
            RAGEGLMConfig cfg;
            cfg.delta       = getd(mp, "--delta", 0.2);
            cfg.max_steps   = geti(mp, "--max_steps", 500000);
            cfg.eps_round   = getd(mp, "--eps_round", 0.10);
            cfg.burnin_n    = geti(mp, "--burnin_n", -1);
            cfg.include_dueling_pairs_as_actions = geti(mp, "--duel", 0);

            long long sumT  = 0;
            int succ = 0;

            std::filesystem::create_directories("../output");
            std::string outname = build_outfile(algo, inst.d, inst.S, inst.K, false, false, cfg.delta);
            if (cfg.include_dueling_pairs_as_actions) outname = "Both" + outname;
            std::string outpath = std::string("../output/") + outname;

            std::ofstream out(outpath, std::ios::out | std::ios::trunc);
            if (!out) throw std::runtime_error("cannot open for writing: " + outpath);

            out << "S = " << fmt_S(inst.S) << ", K = " << inst.K << ", d = " << inst.d << "\n";
            out << "Round, Stop Time, Right\n";

            for (int r = 0; r < runs; ++r) {
                std::cout << "Run " << (r + 1) << "/" << runs << "\n";
                RNG rrng(seed + 10007ull * (uint64_t)r + 17ull);
                RAGEGLMResult rs = run_rageglm_baseline(inst, cfg, rrng);
                sumT += rs.stop_t;
                succ += (rs.correct ? 1 : 0);
                out << (r + 1) << ", " << rs.stop_t << ", " << (rs.correct ? 1 : 0) << "\n";
            }

            out << "Algo = rageglm"
                << ", Runs = " << runs
                << ", Avg stop time = " << (double)sumT / (double)runs
                << ", Success rate = " << (double)succ / (double)runs
                << "\n";

            out.close();
            return 0;
        }

        std::cerr << "Unknown --algo: " << algo << "\n";
        usage();
        return 1;
    }

    else if (mode == "batch") {
        // Inputs you requested
        int d = geti(mp, "--d", 2);
        int K = geti(mp, "--K", d+1);
        double S = getd(mp, "--S", 2.0);

        double delta = getd(mp, "--delta", 0.2);
        int max_steps = geti(mp, "--max_steps", 200000);
        int runs = geti(mp, "--runs", 1);
        std::string algo = get(mp, "--algo", "hybrid");
        // Store stop times for Average / StdDev
        std::vector<long long> st_rage;   st_rage.reserve(runs);
        std::vector<long long> st_rand;   st_rand.reserve(runs);
        std::vector<long long> st_rets;   st_rets.reserve(runs);
        std::vector<long long> st_hyts;   st_hyts.reserve(runs);

        // (optional) store success counts if you also want success rate
        int succ_rage = 0, succ_rand = 0, succ_rets = 0, succ_hyts = 0;


        std::filesystem::create_directories("../output");

        auto outpath_of = [&](const std::string& tag) {
            std::ostringstream oss;
            oss << tag << "_" << d << "_" << fmt_S(S) << "_" << K << "_" << delta << ".txt";
            return std::string("../output/") + oss.str();
        };

        auto init_file = [&](const std::string& tag) {
            std::string path = outpath_of(tag);
            std::ofstream out(path, std::ios::out | std::ios::trunc);
            if (!out) throw std::runtime_error("cannot open for writing: " + path);

            // REQUIRED header: seed + delta
            out << "Seed = " << seed << ", Delta = " << delta << "\n";
            out << "K = " << K << ", d = " << d << ", S = " << fmt_S(S)
                << ", max_steps = " << max_steps << ", runs = " << runs << "\n";
            out << "Run, InstanceSeed, Stop Time, Right\n";
            out.close();
        };

        // 4 output files
        init_file("RAGEGLM_NoDuel");
        init_file("Random_WithDuel");
        init_file("ReTS_GLB"); // hybrid + reward_only=1
        init_file("HyTS_GLB"); // hybrid + reward_only=0 (Both)

        // ---- Config templates (instance-dependent values filled each run) ----

        // ReTS-GLB / HyTS-GLB share HybridConfig
        HybridConfig hy_cfg;
        hy_cfg.delta     = delta;
        hy_cfg.max_steps = max_steps;

        hy_cfg.Rs_c   = getd(mp, "--Rs_c", 1.0);
        hy_cfg.Rs_d   = getd(mp, "--Rs_d", 1.0);
        hy_cfg.zeta_c = getd(mp, "--zeta_c", 1.0);
        hy_cfg.zeta_d = getd(mp, "--zeta_d", 1.0);
        hy_cfg.lambda = getd(mp, "--lambda", 1e-6);

        // Random (With Duel) uses HybridConfig too (your run_rand_hybrid signature)
        HybridConfig rnd_cfg = hy_cfg;
        rnd_cfg.reward_only = true;

        // Rage-GLM (Without Duel)
        RAGEGLMConfig rg_cfg;
        rg_cfg.delta     = delta;
        rg_cfg.max_steps = max_steps;
        rg_cfg.eps_round = getd(mp, "--eps_round", 0.10);
        rg_cfg.burnin_n  = geti(mp, "--burnin_n", -1);
        rg_cfg.include_dueling_pairs_as_actions = 0; // WITHOUT Duel (key requirement)

        printf("runs = %d\n", runs);
        
        // ---- Main loop: each run generates fresh instance, then runs 4 algos ----
        for (int r = 0; r < runs; ++r) {
            std::cout << "Batch Run " << (r + 1) << "/" << runs << "\n";

            uint64_t inst_seed = seed + 10007ull * (uint64_t)r + 17ull;
            RNG inst_rng(inst_seed);
            Instance inst = generate_synthetic_instance(K, d, S, inst_rng);
            inst.reallocate();
            
            double duel_bound = init(inst);

            // instance-dependent fields
            hy_cfg.duel_bound  = duel_bound;
            rnd_cfg.duel_bound = duel_bound;

            hy_cfg.get_sc_sd(inst.S);
            rnd_cfg.get_sc_sd(inst.S);

            // 1) Rage-GLM (Without Duel)
            {
                std::string path = outpath_of("RAGEGLM_NoDuel");
                std::ofstream out(path, std::ios::out | std::ios::app);
                if (!out) throw std::runtime_error("cannot open for appending: " + path);

                RNG rrng(inst_seed ^ 0x369dea0f31a53f85ull);
                RAGEGLMResult rs = run_rageglm_baseline(inst, rg_cfg, rrng);

                out << (r + 1) << ", " << inst_seed << ", " << rs.stop_t << ", " << (rs.correct ? 1 : 0) << "\n";
                out.close();
                printf("RaGe-GLM (Reward): Round = %d\n", rs.stop_t);
                st_rage.push_back(rs.stop_t);
                succ_rage += (rs.correct ? 1 : 0);

            }

            // 2) Random (With Duel)  -> force run_rand_hybrid
            {
                std::string path = outpath_of("Random_WithDuel");
                std::ofstream out(path, std::ios::out | std::ios::app);
                if (!out) throw std::runtime_error("cannot open for appending: " + path);

                RNG rrng(inst_seed ^ 0xbf58476d1ce4e5b9ull);
                RunSummary rs = run_rand_hybrid(inst, rnd_cfg, rrng);

                out << (r + 1) << ", " << inst_seed << ", " << rs.stop_time << ", " << (rs.correct ? 1 : 0) << "\n";
                out.close();
                printf("Random (Hybrid): Round = %d\n", rs.stop_time);
                st_rand.push_back(rs.stop_time);
                succ_rand += (rs.correct ? 1 : 0);

            }


            // 3) ReTS-GLB  -> hybrid with reward_only = 1
            {
                std::string path = outpath_of("ReTS_GLB");
                std::ofstream out(path, std::ios::out | std::ios::app);
                if (!out) throw std::runtime_error("cannot open for appending: " + path);

                HybridConfig cfg = hy_cfg;
                cfg.reward_only = true;   // key
                cfg.duel_only   = false;

                RNG rrng(inst_seed ^ 0x94d049bb133111ebull);
                RunSummary rs = run_one(inst, cfg, rrng);

                out << (r + 1) << ", " << inst_seed << ", " << rs.stop_time << ", " << (rs.correct ? 1 : 0) << "\n";
                out.close();

                printf("ReTS-GLB: Round = %d\n", rs.stop_time);

                st_rets.push_back(rs.stop_time);
                succ_rets += (rs.correct ? 1 : 0);

            }

            // 4) HyTS-GLB  -> hybrid Both (reward_only = 0, duel_only = 0)
            {
                std::string path = outpath_of("HyTS_GLB");
                std::ofstream out(path, std::ios::out | std::ios::app);
                if (!out) throw std::runtime_error("cannot open for appending: " + path);

                HybridConfig cfg = hy_cfg;
                cfg.reward_only = false;
                cfg.duel_only   = false;

                RNG rrng(inst_seed ^ 0x9e3779b97f4a7c15ull);
                RunSummary rs = run_one(inst, cfg, rrng);

                out << (r + 1) << ", " << inst_seed << ", " << rs.stop_time << ", " << (rs.correct ? 1 : 0) << "\n";
                out.close();

                printf("HyTS-GLB: Round = %d\n", rs.stop_time);

                st_hyts.push_back(rs.stop_time);
                succ_hyts += (rs.correct ? 1 : 0);

            }
        }
        // Append Average / StdDev to each output file
        {
            MeanStd ms = mean_std(st_rage);
            std::string path = outpath_of("RAGEGLM_NoDuel");
            std::ofstream out(path, std::ios::out | std::ios::app);
            if (!out) throw std::runtime_error("cannot open for appending: " + path);
            out << "Average stop time = " << ms.mean
                << ", StdDev = " << ms.stdev
                << ", Success rate = " << (runs ? (double)succ_rage / (double)runs : 0.0)
                << "\n";
        }
        {
            MeanStd ms = mean_std(st_rand);
            std::string path = outpath_of("Random_WithDuel");
            std::ofstream out(path, std::ios::out | std::ios::app);
            if (!out) throw std::runtime_error("cannot open for appending: " + path);
            out << "Average stop time = " << ms.mean
                << ", StdDev = " << ms.stdev
                << ", Success rate = " << (runs ? (double)succ_rand / (double)runs : 0.0)
                << "\n";
        }
        {
            MeanStd ms = mean_std(st_rets);
            std::string path = outpath_of("ReTS_GLB");
            std::ofstream out(path, std::ios::out | std::ios::app);
            if (!out) throw std::runtime_error("cannot open for appending: " + path);
            out << "Average stop time = " << ms.mean
                << ", StdDev = " << ms.stdev
                << ", Success rate = " << (runs ? (double)succ_rets / (double)runs : 0.0)
                << "\n";
        }
        {
            MeanStd ms = mean_std(st_hyts);
            std::string path = outpath_of("HyTS_GLB");
            std::ofstream out(path, std::ios::out | std::ios::app);
            if (!out) throw std::runtime_error("cannot open for appending: " + path);
            out << "Average stop time = " << ms.mean
                << ", StdDev = " << ms.stdev
                << ", Success rate = " << (runs ? (double)succ_hyts / (double)runs : 0.0)
                << "\n";
        }


        std::cout << "Batch done. Outputs are in ../output/\n";
        return 0;
    }




    std::cerr << "Unknown --mode: " << mode << "\n";
    usage();
    return 1;
}
