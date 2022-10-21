set -ex

if [ $# -ge 1 ]
then
    Kokkos_ARCH=$1
else
    Kokkos_ARCH='PASCAL61'
fi

mkdir build
cd build

cmake -DDYABLO_ENABLE_UNIT_TESTING=ON -DDYABLO_HIDE_BITPIT_COMPILATION=OFF -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH=${Kokkos_ARCH} ..
make -j `nproc`
