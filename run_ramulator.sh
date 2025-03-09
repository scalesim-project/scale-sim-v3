#!/bin/bash
topology="topologies/models/${1}.csv"
time python3 scalesim/scale.py -c ./configs/google.cfg -t ${topology} > ${1}_${2}_orig_out
time python3 scripts/dram_sim.py -topology $1 -run_name GoogleTPU_v1_os
time python3 scripts/dram_latency.py -topology $1
time python3 scalesim/scale.py -c ./configs/google_ramulator_${2}.cfg -t ${topology} > ${1}_${2}_stall_out
cp ${1}_${2}_orig_out ./results/dram_results/stall_cycles/${1}_${2}_orig_out
cp ${1}_${2}_stall_out ./results/dram_results/stall_cycles/${1}_${2}_stall_out
