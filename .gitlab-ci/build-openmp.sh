set -ex

mkdir build
cd build

cmake -DDYABLO_HIDE_BITPIT_COMPILATION=OFF -DDYABLO_ENABLE_UNIT_TESTING=ON ..
make -j `nproc`
