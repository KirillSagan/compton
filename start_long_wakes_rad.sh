#!/bin/sh
declare -a names=("long_wakes_rad_1e5")

for name in "${names[@]}"
do

export name
export charge_min_nC=0.0
export charge_max_nC=15.0
export n_scan=16
for i in {0..0}
do

export i
chmod +x run_one_intensity.sh
sbatch run_one_intensity.sh

done

done
