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
project(glb_bai_hybrid CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# For macOS clang/gcc
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(glb_bai
    src/main.cpp
)

target_include_directories(glb_bai PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Reasonable warnings
if (MSVC)
  target_compile_options(glb_bai PRIVATE /W4)
else()
  target_compile_options(glb_bai PRIVATE -Wall -Wextra -Wpedantic)
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

You should see `glb_bai` in `build/`.

## Quick run (should finish fast)

### 1) Generate an instance

```bash
./glb_bai --mode gen --out instance.txt --K 10 --d 2 --S 2.0 --seed 1
```

### 2) Run the algorithm (single trial)

```bash
./glb_bai --mode run --load instance.txt --delta 0.05 --max_steps 20000 --seed 2
```

### 3) Run multiple trials (average stopping time + success rate)

```bash
./glb_bai --mode run --load instance.txt --delta 0.05 --max_steps 20000 --runs 10 --seed 123
```

### 4) Baseline:

```bash
./glb_bai --mode run --algo glgape --load instance.txt  --max_steps 20000 --delta 0.05 --eps 0.0 --seed 2 --runs 10
```


```bash
./glb_bai --mode run --algo random --load instance.txt --delta 0.05 --max_steps 20000 --runs 10 --seed 123
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

## Notes on speed

To get a version that finishes in ~1 minute:

* Use moderate K,d (e.g., K=10..30, d=5..20).
* Set `--max_steps` around 5e3 to 2e4.
* Use `--runs` modestly (10..50).
* If the algorithm stops too slowly, increase `--lambda` slightly (e.g., 1e-4) for numerical stability and/or reduce `--S`.
