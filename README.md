## 目录结构

```
GLB-BAI/
  CMakeLists.txt
  readme.md
  src/
    main.cpp
    lin_alg.h
    rng.h
    instance.h
    glm_logistic.h
    mle.h
    hybrid_bai.h
```

---

## CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.12)
project(main_hybrid CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# For macOS clang/gcc
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(main
    src/main.cpp
)

target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Reasonable warnings
if (MSVC)
  target_compile_options(main PRIVATE /W4)
else()
  target_compile_options(main PRIVATE -Wall -Wextra -Wpedantic)
endif()
```

---

## readme.md

````markdown
# Hybrid GLM Best-Arm Identification (classic + dueling)

This project provides:
- A fast synthetic instance generator (dataset generator).
- A runnable Hybrid GLM-BAI algorithm (classic Bernoulli/logistic reward + dueling Bradley-Terry logistic comparisons).
- Outputs: stopping time, predicted best arm, correctness (vs true best arm), and an approximate `T^*(theta*)`.

## Build (cmake .. && make)

```bash
mkdir -p build
cd build
cmake ..
make
````

You should see `main` in `build/`.

## Quick run (should finish fast)

## # Batch: generate runs instances, and run 4 algos each time
```bash  
./main --mode batch --K 5 --d 4 --S 5.0 --seed 123 --delta 0.05 --max_steps 2000000 --runs 10
```

> For Cost Analysis:


```bash
./main --mode --K 3 --d 2 --S 5.0 cost --load instance.txt --delta 0.05 --max_steps 20000000 --runs 10 --seed 123
```





### 1) Generate an instance

```bash
./main --mode gen --out instance.txt --K 10 --d 4 --S 5.0 --seed 1
```

### 2) Run the algorithm (single trial)

```bash
./main --mode run --load instance.txt --delta 0.05 --max_steps 20000000 --seed 123
```

### 3) Run multiple trials (average stopping time + success rate)

> Under Hybrid Case.

```bash
./main --mode run --load instance.txt --delta 0.05 --max_steps 20000000 --runs 10 --seed 123
```

> For Reward Only.

```bash
./main --mode run --reward_only --load instance.txt --delta 0.05 --max_steps 20000000 --runs 10 --seed 123
```

### 4) Baseline:

```bash
./main --mode run --algo glgape --load instance.txt  --max_steps 20000000 --delta 0.05 --eps 0.0 --seed 123 --runs 10
```


```bash
./main --mode run --algo rageglm --load instance.txt  --max_steps 20000000 --delta 0.05 --seed 123 --runs 10
```

```bash
./main --mode run --algo rageglm --duel 1 --load instance.txt  --max_steps 20000000 --delta 0.05 --seed 123 --runs 10
```


```bash
./main --mode run --algo random --load instance.txt --delta 0.05 --max_steps 20000000 --runs 10 --seed 123
```

```bash
./main --mode run --algo random --duel 1 --load instance.txt --delta 0.05 --max_steps 20000000 --runs 10 --seed 123
```

## Main parameters

* `--K`: number of arms
* `--d`: dimension
* `--S`: radius constraint for theta ($\|\theta\| \leq S$)
* `--delta`: confidence level
* `--max_steps`: hard cap on time steps (for fast experiments)
* `--duel_prob`: probability of querying dueling feedback at each step (0..1)
* `--Rs_c`, `--Rs_d`: self-concordance constants (default 1.0)
* `--zeta_c`, `--zeta_d`: scaling constants in loss/Hessian (default 1.0)
* `--lambda`: ridge regularization added to A_t (default 1e-6)
* `--tstar_samples`: random search samples for approximating $T^*(\theta^*)$

## Instance file format

`instance.txt` is plain text:

* First line: `K d S`
* Second line: theta* (d numbers)
* Next K lines: x_i (d numbers per line)

All vectors are stored in Euclidean coordinates.
