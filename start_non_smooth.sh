#!/bin/sh
declare -a names=("non_smooth_optics_natural_chromoticity")

for name in "${names[@]}"
do

export name
export charge_min_nC=0.0
export charge_max_nC=3.0
export n_scan=16
for i in {0..$n_scan}
do

export i
chmod +x run_one_intensity.sh
sbatch run_one_intensity.sh

done

done
