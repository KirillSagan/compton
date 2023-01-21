#!/bin/sh
declare -a names=("long_CSR_wo_wakes")

for name in ${names[@]}; do

export name
export charge_min_nC=0.0
export charge_max_nC=0.15
export n_scan=16
for i in {0..0}
do

export i
chmod +x run_one_intensity.sh
sbatch run_one_intensity.sh

done

done
