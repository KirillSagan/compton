#!/bin/sh
declare -a names=("long_CSR_03mm")

for name in ${names[@]}; do

export name
export charge_min_nC=0.0
export charge_max_nC=0.45
export n_scan=16
for i in {0..15}
do

export i
chmod +x run_one_intensity.sh
sbatch run_one_intensity.sh

done

done
