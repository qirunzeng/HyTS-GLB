# Hybrid GLM Best-Arm Identification

This codebase runs hybrid reward/dueling best-arm identification experiments for logistic generalized linear bandits.

## Build

```bash
mkdir -p build
cd build
cmake ..
cmake --build . --config Release
```

The executable is `build/main` on Unix-like systems and `build/Release/main.exe` or `build/main.exe` on Windows, depending on the generator.

If CMake is not available, a direct compile also works:

```bash
g++ -std=c++17 -O2 -Isrc src/main.cpp -o main
```

## Quick Runs

Run the dimension-scaling experiment on freshly generated shared instances:

```bash
./main --mode batch --K 5 --d 4 --S 2.0 --seed 123 --delta 0.05 --max_steps 2000000 --runs 10
```

Run the same batch of algorithms on the deterministic basis-plus-rotated instance, where `K=d+1`:

```bash
./main --mode batch_basis --d 4 --S 5.0 --seed 123 --delta 0.05 --max_steps 2000000 --runs 10
```

For a faster exploratory run of the same experiment, refresh the HyTS MLE/design every 10 steps:

```bash
./main --mode batch --K 5 --d 4 --S 5.0 --seed 123 --delta 0.05 --max_steps 2000000 --runs 10 --update_period 10
```

Run the cost-aware experiment:

```bash
./main --mode cost --K 3 --d 2 --S 5.0 --seed 123 --delta 0.05 --max_steps 20000000
```

Generate one instance and run a single algorithm on it:

```bash
./main --mode gen --K 5 --d 4 --S 5.0 --seed 123 --out instance.txt
./main --mode run --load instance.txt --algo hybrid --delta 0.05 --max_steps 2000000 --runs 10
```

Generate the deterministic basis-plus-rotated instance with `K=d+1`, arms `x_i=e_i`, `x_{d+1}=(cos 0.1, sin 0.1, 0,...)`, and `theta_star=(S-1)e_1`. This mode requires `d >= 2`.

```bash
./main --mode gen_basis --d 4 --S 5.0 --out basis_instance.txt
./main --mode run --load basis_instance.txt --algo hybrid --delta 0.05 --max_steps 2000000 --runs 10
```

## Usage Examples

Small smoke test:

```bash
./main --mode batch --K 3 --d 2 --S 5.0 --seed 1 --delta 0.05 --max_steps 2000 --runs 1
```

Small smoke test on the deterministic basis-plus-rotated instance:

```bash
./main --mode batch_basis --d 2 --S 5.0 --seed 1 --delta 0.05 --max_steps 2000 --runs 1
```

Run HyTS-GLB only on a saved instance:

```bash
./main --mode gen --K 5 --d 4 --S 5.0 --seed 123 --out instance.txt
./main --mode run --load instance.txt --algo hybrid --delta 0.05 --max_steps 2000000 --runs 10
```

Run HyTS-GLB on the deterministic basis-plus-rotated instance:

```bash
./main --mode gen_basis --d 12 --S 5.0 --out basis_d12.txt
./main --mode run --load basis_d12.txt --algo hybrid --delta 0.05 --max_steps 2000000 --runs 10
```

Run reward-only HyTS, corresponding to ReTS-GLB:

```bash
./main --mode run --load instance.txt --algo hybrid --reward_only 1 --delta 0.05 --max_steps 2000000 --runs 10
```

Run dueling-only HyTS:

```bash
./main --mode run --load instance.txt --algo hybrid --duel_only 1 --delta 0.05 --max_steps 2000000 --runs 10
```

Run reward-only RAGE-GLM:

```bash
./main --mode run --load instance.txt --algo rageglm --delta 0.05 --max_steps 2000000 --runs 10
```

Run dueling-only RAGE-GLM by treating arm pairs as measurement actions:

```bash
./main --mode run --load instance.txt --algo rageglm --duel 1 --delta 0.05 --max_steps 2000000 --runs 10
```

Run random hybrid sampling:

```bash
./main --mode run --load instance.txt --algo random --duel 1 --delta 0.05 --max_steps 2000000 --runs 10
```

Run cost-aware HyTS across the built-in cost ratios:

```bash
./main --mode cost --K 3 --d 2 --S 5.0 --seed 123 --delta 0.05 --max_steps 20000000
```

## Algorithms

`--mode batch` and `--mode batch_basis` run:

* `RAGEGLM_NoDuel`: reward-only RAGE-GLM baseline.
* `RAGEGLM_NoReward`: dueling-only RAGE-GLM baseline, treating pairs as measurement actions.
* `Random_WithDuel`: random hybrid reward/dueling sampling.
* `ReTS_GLB`: reward-only track-and-stop variant.
* `DuelTS_GLB`: dueling-only track-and-stop variant.
* `HyTS_GLB`: proposed hybrid track-and-stop method.

`batch` generates one fresh synthetic instance per run. `batch_basis` uses the deterministic basis-plus-rotated instance with `K=d+1` in every run; its output filenames are prefixed with `Basis_`.

The RAGE-GLM baseline is implemented in the theorem-style fixed-design form: after burn-in, each elimination round computes its MLE using only that round's samples, not cumulative samples from previous rounds.

## Main Parameters

* `--mode`: command mode. Common values are `gen`, `gen_basis`, `run`, `batch`, `batch_basis`, and `cost`.
* `--K`: number of arms.
* `--d`: feature dimension.
* `--S`: radius constraint for theta, `||theta|| <= S`.
* `--delta`: confidence level.
* `--max_steps`: hard cap on time steps.
* `--runs`: number of independent runs in `run` or `batch` mode.
* `--seed`: random seed.
* `--update_period`: HyTS optimization refresh period. Default is `1`; larger values reuse the previous MLE/design for more steps and can speed up exploratory runs.
* `--duel`: for `--algo rageglm`, set to `1` to include dueling pairs as measurement actions.
* `--reward_only` / `--duel_only`: for `--algo hybrid`, restrict HyTS to one feedback type.

## Instance File Format

`instance.txt` is plain text:

* First line: `K d S`
* Second line: `theta_star` with `d` numbers
* Next `K` lines: one arm feature vector `x_i` per line

All vectors are stored in Euclidean coordinates.

## Output

Experiment results are written to `../output/` relative to the executable working directory.
