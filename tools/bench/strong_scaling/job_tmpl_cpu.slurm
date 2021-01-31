#!/bin/bash

# This is meant to be used on Jean Zay

#SBATCH -J dyablo
#SBATCH -N <nb_nodes> -n <nb_mpi> -c <nb_threads>
#SBATCH -A sah@cpu --qos=qos_cpu-dev
#SBATCH -t 0:30:00
#SBATCH --exclusive
#SBATCH -o log.out

module purge
module load cmake/3.18.0 gcc/9.1.0 hwloc/1.11.11 openmpi/3.1.5 hdf5/1.10.5-mpi libxml2

OMP_NUM_THREADS=<nb_threads> OMP_PROC_BIND=true srun ./test_solver blast.ini
