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
    std::cerr <<
R"(Usage:
  # Generate instance
  glb_bai --mode gen --out instance.txt --K 10 --d 5 --S 2.0 --seed 1

  # Run Hybrid (default)
  glb_bai --mode run --algo hybrid --load instance.txt --delta 0.2 --max_steps 200000 --seed 2

  # Run GLGapE baseline
  glb_bai --mode run --algo glgape --load instance.txt --delta 0.2 --eps 0.1 --seed 2

  # Multiple trials
  glb_bai --mode run --algo glgape --load instance.txt --runs 50 --delta 0.2 --eps 0.1 --seed 123

Common options:
  --mode gen|run
  --seed, --runs

Instance options (gen):
  --K, --d, --S, --out

Hybrid options (run, --algo hybrid):
  --delta, --max_steps,
  --Rs_c, --Rs_d, --zeta_c, --zeta_d, --lambda
  --reward_only (0/1), --duel_only (0/1)


Hybrid options (run, --algo random):
  --delta, --max_steps,
  --Rs_c, --Rs_d, --zeta_c, --zeta_d, --lambda

GLGapE options (run, --algo glgape):
  --delta, --eps
  --E
  --downscale_C (0/1), --C_scale
)";
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
    if (algo == "glgape") {
        prefix = "Baseline";
    } else if (algo == "hybrid") {
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

auto init = [](Instance &inst) {
    double max_m = -1.;
    for (int i = 0; i < inst.K; ++i) {
        inst.dots[i] = dot(inst.x[i], inst.theta_star);
        inst.means[i] = sigmoid(inst.dots[i]);
        std::cout << max_m << " " << inst.means[i] << std::endl;
        if (inst.dots[i] > max_m) {
            max_m = inst.dots[i];
            inst.best = i;
            std::cout << "Best: " << inst.best << std::endl;
        }
    }
    for (int j = 0; j < inst.K; ++j) {
        for (int k = j+1; k < inst.K; ++k) {
            inst.g[j][k] = inst.x[j] - inst.x[k];
            inst.gap_dots[j][k] = dot(inst.g[j][k], inst.theta_star);
            inst.gaps[j][k] = sigmoid(inst.gap_dots[j][k]);
        }
    }
};


int main(int argc, char** argv) {
    std::unordered_map<std::string,std::string> mp;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("--",0) == 0) {
            std::string val="1";
            if (i + 1 < argc && std::string(argv[i+1]).rfind("--", 0) != 0) {
                val = std::string(argv[i+1]);
                ++i;
            }
            mp[key] = val;
        }
    }

    std::string mode = get(mp, "--mode", "");
    if (mode.empty()) { usage(); return 1; }

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

        
    if (mode == "run") {
        std::string path = get(mp, "--load", "");
        if (path.empty()) { std::cerr << "--load is required in run mode\n"; return 1; }

        Instance inst = load_instance(path);
        std::string algo = get(mp, "--algo", "hybrid");
        int runs = geti(mp, "--runs", 1);

        // Optional console debug (keep if you want)
        for (int i = 0; i < inst.K; ++i) {
            double dotp = dot(inst.x[i], inst.theta_star);
            std::cout << "Arm " << i << ": x_i^T theta* = " << dotp << "\n";
        }
        inst.reallocate();
        init(inst);
        std::cout << "i^star: " << inst.true_best_arm() << "\n";
        
        if (algo == "hybrid") {
            HybridConfig cfg;
            cfg.delta = getd(mp, "--delta", 0.2);
            cfg.max_steps = geti(mp, "--max_steps", 200000);

            cfg.Rs_c = getd(mp, "--Rs_c", 1.0);
            cfg.Rs_d = getd(mp, "--Rs_d", 1.0);
            cfg.zeta_c = getd(mp, "--zeta_c", 1.0);
            cfg.zeta_d = getd(mp, "--zeta_d", 1.0);
            cfg.lambda = getd(mp, "--lambda", 1e-6);

            cfg.reward_only = (geti(mp, "--reward_only", 0) != 0);
            cfg.duel_only = (geti(mp, "--duel_only", 0) != 0);

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


            // Optional console summary
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

            cfg.reward_only     = true;

            long long sumT = 0;
            int succ = 0;

            std::filesystem::create_directories("../output");
            std::string outname = build_outfile(algo, inst.d, inst.S, inst.K, cfg.reward_only, cfg.duel_only, cfg.delta);

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
                RunSummary rs = run_rand(inst, cfg, rrng);
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

        std::cerr << "Unknown --algo: " << algo << "\n";
        usage();
        return 1;
    }

    std::cerr << "Unknown --mode: " << mode << "\n";
    usage();
    return 1;
}
