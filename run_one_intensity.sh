#!/bin/sh
#SBATCH -D /s/ls4/users/kssagan/compton
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -t 72:00:00
#SBATCH -p hpc5-gpu-3d
#SBATCH --gres=gpu:4
#SBATCH -n 16
export LD_LIBRARY_PATH=/s/ls4/sw/gcc/7.3.0_v2/lib64:/s/ls4/sw/gcc/7.3.0_v2/lib:/s/ls4/sw/binutils/2.30/lib
export PATH=/s/ls4/sw/gcc/7.3.0_v2/bin::/s/ls4/sw/binutils/2.30/bin
export MPLCONFIGDIR=/d/ls4/users/kssagan/.config/matplotlib
export PYCUDA_CACHE_DIR=/s/ls4/users/kssagan/cache
module load cuda 
source /s/ls4/users/kssagan/envs/env_pht/bin/activate 
python "/s/ls4/users/kssagan/compton/run_${name}.py" $i $charge_min_nC $charge_max_nC $n_scan 
