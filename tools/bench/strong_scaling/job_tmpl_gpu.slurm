#!/bin/bash

# This is meant to be used on Jean Zay

#SBATCH -J dyablo
#SBATCH -N <nb_nodes> -n <nb_mpi> -c <nb_threads> --gres=gpu:<nb_gpus>
#SBATCH -A sah@gpu --qos=qos_gpu-dev
#SBATCH --exclusive
#SBATCH -o log.out


module purge
module load cmake/3.18.0 gcc/8.3.1 openmpi/4.0.2-cuda hdf5/1.12.0-mpi-cuda libxml2 cuda/10.2

srun ./test_solver --kokkos-num-devices=<nb_gpus> blast.ini