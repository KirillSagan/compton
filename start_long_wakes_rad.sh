#!/bin/sh
declare -a names=("long_1e5" "long_1e5_smooth" "long_1e6" "long_5sigma_1e6_precise" "long_10sigma_1e6_precise")

for name in ${names[@]}; do

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
