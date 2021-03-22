set -ex

mkdir build
cd build

cmake -DDYABLO_ENABLE_UNIT_TESTING=ON ..
make -j `nproc`
