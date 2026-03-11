# Hybrid GLM Best-Arm Identification (classic + dueling)

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

## Main parameters

* `--K`: number of arms
* `--d`: dimension
* `--S`: radius constraint for theta ($\|\theta\| \leq S$)
* `--delta`: confidence level
* `--max_steps`: hard cap on time steps (for fast experiments)

## Instance file format

`instance.txt` is plain text:

* First line: `K d S`
* Second line: theta* (d numbers)
* Next K lines: x_i (d numbers per line)

All vectors are stored in Euclidean coordinates.
