# Data flow test results for os/is, k=1/4/8

## Process

PeCyclesPerMac patches tested with output stationary and input stationary (weight stationary as control).

`project/dataflow_tests.py` runs the different SCALE-Sim application tests and writes results to `project/results/dataflow_tests.csv`.

This test file has to have an upstream repository (no patches: scalesim-project/scale-sim-v3) to test the patched against. You can just go to directory containing this scale-sim-v3 and `git clone https://github.com/scalesim-project/scale-sim-v3.git scale-sim-upstream`. Then to run, in our project scale-sim-v3 folder with the venv activated, run `python project/dataflow_tests.py`.


## Results

| dataflow | k | upstream_total | upstream_compute | total_cycles | compute_cycles | stalls | util | ratio_compute | ratio_total | real_util | k1_matches_upstream |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ws | 1 | 15498 | 11199 | 15498 | 11199 | 0 | 73.14938833824449 | 1.0 | 1.0 | 73.15 | True |
| ws | 4 | 15498 | 11199 | 49098 | 44799 | 0 | 18.28612245809058 | 4.0 | 3.168 | 73.14 | True |
| ws | 8 | 15498 | 11199 | 93898 | 89599 | 0 | 9.14295918481233 | 8.001 | 6.059 | 73.14 | True |
| os | 1 | 14473 | 10175 | 14473 | 10175 | 0 | 80.51105651105651 | 1.0 | 1.0 | 80.51 | True |
| os | 4 | 14473 | 10175 | 45001 | 40703 | 0 | 20.126280618136256 | 4.0 | 3.109 | 80.51 | True |
| os | 8 | 14473 | 10175 | 85705 | 81407 | 0 | 10.063016693896104 | 8.001 | 5.922 | 80.5 | True |
| is | 1 | 18506 | 14207 | 18506 | 14207 | 0 | 57.66171605546562 | 1.0 | 1.0 | 57.66 | True |
| is | 4 | 18506 | 14207 | 61130 | 56831 | 0 | 14.41466805088772 | 4.0 | 3.303 | 57.66 | True |
| is | 8 | 18506 | 14207 | 117962 | 113663 | 0 | 7.207270615767664 | 8.0 | 6.374 | 57.66 | True |

`compute_cycles` is amount of cycles spent computing, no prefetch

`ratio_compute` = compute cycles / compute cycles @ k = 1

`ratio_total` = total cycles / total cycles @ k = 1

`util` is overall utilization percentage from the SCALE-Sim report (actual MACs / 100*(number of number of PEs * compute cycles)) (not adjusted for k)

`real_util` = util * k (real PE busy time)

`k1` matched upstream cycles for all data flows.

Note zero stalls because memory bandwidth is unlimited. The test uses a small GEMM on 32x32 array.

real_util will not change with k as we find, and prefetch stays constant.

We see that os and is operate correctly at k=1 to match unmodified SCALE-Sim, and the compute cycles scale correctly when k is changed to 4, 8.