module purge
module load cmake/3.18.0 gcc/8.3.1 openmpi/4.0.2-cuda hdf5/1.12.0-mpi-cuda libxml2 cuda/10.2
cmake -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH="VOLTA70" -DCMAKE_BUILD_TYPE=Release -DKokkos_HWLOC_DIR=/gpfslocalsup/spack_soft/hwloc/2.2.0/gcc-8.3.1-3fvj365djauc5fgbcnhc5lxd35qqrkix/  ..

make -j 5