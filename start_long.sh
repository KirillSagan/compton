#!/bin/sh
declare -a names=("inj_on_axis")

for name in "${names[@]}"
do

export name
export charge_min_nC=0.0
export charge_max_nC=0.0
export n_scan=1
for i in {0..0}
do

export i
chmod +x run_one_intensity.sh
sbatch run_one_intensity.sh

done

done
