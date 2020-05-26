# What is dyablo ?

Dyablo stands for DYnamics adaptive mesh refinement CFD applications with PABLO.

It is a just a attemp to rewrite again an AMR (Adaptive Mesh Refinement) miniapp (with both shared and distributed parallelism in mind). We do not write from scratch but try to couple some of the best state-of-the-art tools.

- AMR is delegated to C++ library [BitPit/PABLO](https://github.com/optimad/bitpit). BitPit only uses MPI (distributed memory parallelism). 
- numerical scheme is built on top of the PABLO mesh, in a decoupled manner, using [kokkos](https://github.com/kokkos/kokkos) for shared memory parallelism.

Why did we chose BitPit/PABLO for this test ? What are the difference with [p4est](http://www.p4est.org/) used in [Canop](https://gitlab.maisondelasimulation.fr/canoPdev/canoP) ?
- [p4est](http://www.p4est.org/) is written in C, about 40 kSLOC; cell-based AMR; manage of forest of octrees, i.e. the physical domain is made of a coarse mesh (p4est connectivty), and each cell of this coarse mesh serves as a root to an octree. Some of the core algorithms used in p4est are very complex due to the management of a forest of octree in distributed memory
- [BitPit/PABLO](https://github.com/optimad/bitpit) also implements cell-based AMR but on a single cubic box; this is a major difference with [p4est](http://www.p4est.org/); it is written in C++, the core code is pleasant to read. It terms of size, BitPit/PABLO is less than half of p4est SLOC.


# What is the plan ?

1. Re-implement Hydro/Euler à la Ramses (as done in [Canop](https://gitlab.maisondelasimulation.fr/canoPdev/canoP), i.e. one cell per leaf of the octree); mesh is managed by PABLO, computational kernels are written in Kokkos. As the computational kernels need to access cell connectivity, the first milestone will be to target MPI + Kokkos/OpenMP only, so that we will enable the use of CPU-only routine (from PABLO) in Kokkos kernels.
2. Test the performance of Euler/Pablo in MPI + Kokkos/OpenMP on a large cluster (e.g. skylake) and compare with Ramses and CanoP to evaluate the cost of mesh management.
3. If 1 and 2 are OK, evaluate how much work is needed to Kokkossify PABLO itself (or a sub-part). To start with, we could consider keeping most of PABLO on CPU, but at each mesh modification (refine + coarsen) export mesh connectivity in a Kokkos::View + hashmap to be used in the computational kernels (either OpenMP, or CUDA).

dyablo is not correlated to [khamr](https://gitlab.maisondelasimulation.fr/pkestene/khamr) yet, will use some of its ideas (e.g Kokkos HashMap implementation).


Performance portability means, we will be using the [Kokkos library](https://github.com/kokkos/kokkos), a C++ parallel programing model and library for performance portability.

# How to build ?

## Get the sources

Make sure to clone this repository recursively, this will also download kokkos source as a git submodule.

```bash
git clone --recurse-submodules git@gitlab.maisondelasimulation.fr:pkestene/dyablo.git
```

Kokkos and BitPit/PABLO are built as part of dyablo with the cmake build system.

## prerequisites

- [kokkos](https://github.com/kokkos/kokkos) : preconfigured as a git submodule
- [BitPit/PABLO](https://github.com/optimad/bitpit) : a local copy (slightly refactored** is included in dyablo

## build dyablo

## superbuild : build bitpit/PABLO and dyablo together

The super-build pattern build both dyablo and its depency (bitpit) using cmake command [ExternalProject_Add](https://cmake.org/cmake/help/latest/module/ExternalProject.html).

We removed the local modified copy of [BitPit/PABLO](https://github.com/optimad/bitpit) introduced in December 2018; we use instead the following archive [bitpit-1.7.0-devel-dyablo.tar.gz](https://github.com/pkestene/bitpit/archive/bitpit-1.7.0-devel-dyablo.tar.gz). BitPit source code archive is downloaded and built as part of dyablo (using the cmake super-build pattern). 

To build bitpit and dyablo (for Kokkos/OpenMP backend which is the default)

```bash
mkdir build_openmp; cd build_openmp
ccmake ..
make
```

The same for Kokkos/CUDA (e.g. for latest Turing CUDA architecture):
```bash
mkdir build_cuda; cd build_cuda
# use nvcc_wrapper compiler from kokkos source directory
export CXX=/full/absolute/path/to/nvcc_wrapper
ccmake -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH=TURING75 ..
make
```

## build only dyablo for Kokkos/OpenMP (when bitpit is already installed)

To build dyablo for kokkos/OpenMP backend, assuming bitpit is already installed :

```bash
mkdir build_openmp; cd build_openmp
ccmake -DBITPIT_DIR=/home/pkestene/local/bitpit-1.7.0-devel-dyablo/lib/cmake/bitpit-1.7 ..
make
```

BITPIT_DIR should to your bitpit install subdirectory where the BITPITConfig.cmake resides.

## build only bitpit

1. get [bitpit sources](https://github.com/pkestene/bitpit/archive/bitpit-1.7.0-devel-dyablo.tar.gz)
2. ```shell
      tar zxf bitpit-1.7.0-devel-dyablo.tar.gz
      cd bitpit-1.7.0-devel-dyablo
      mkdir build; cd build
      ccmake -DCMAKE_INSTALL_PREFIX=/home/pkestene/local/bitpit-1.7.0-devel \
          -DCMAKE_BUILD_TYPE=Release \
          -DENABLE_MPI=ON \
          -DBITPIT_MODULE_LA=OFF \
          -DBITPIT_MODULE_CG=OFF \
          -DBITPIT_MODULE_DISCRETIZATION=OFF \
          -DBITPIT_MODULE_LEVELSET=OFF \
          -DBITPIT_MODULE_PATCHKERNEL=OFF \
          -DBITPIT_MODULE_POD=OFF \
          -DBITPIT_MODULE_RBF=OFF \
          -DBITPIT_MODULE_SA=OFF \
          -DBITPIT_MODULE_SURFUNSTRUCTURED=OFF \
          -DBITPIT_MODULE_VOLCARTESIAN=OFF \
          -DBITPIT_MODULE_VOLOCTREE=OFF \
          -DBITPIT_MODULE_VOLUNSTRUCTURED=OFF \ 
          ..
      make
      make install
      ```

This installed version of bitpit can be used to build dyablo (see above).

# More information

For now, just visit the wiki page https://gitlab.maisondelasimulation.fr/pkestene/dyablo/wikis/home

## build and run unit tests

```shell
# configure cmake for building unit tests
# make sur to have boost installed (with libs)
ccmake -DDYABLO_ENABLE_UNIT_TESTING=ON .. 
make
make dyablo-test
```

## build documentation

### doxygen

```shell
# re-run cmake with additionnal options
cd build
ccmake -DDYABLO_BUILD_DOC=ON -DDYABLO_DOC=doxygen
make
make dyablo-doc
```

This will generate the html doxygen page in `doc/doxygen/html`

### [mkdocs](https://www.mkdocs.org/)

MkDocs is an alternative to sphinx, but relying on markdown instead of ResST. Here the markdown sources are partly generated by [doxybook2](https://github.com/matusnovak/doxybook2) which takes the XML output of doxygen and converts that into markdown sources, directly integrated into mkdocs sources.

```shell
# generate mkdocs sources
cd build
ccmake -DDYABLO_BUILD_DOC=ON -DDYABLO_DOC=mkdocs
make
make dyablo-doc
```

This will generate the markdown sources for the mkdocs static webpage.

```shell
cd doc/mkdocs

# preview of the webpage
mkdocs serve
# open url localhost:8000

# if you want to build the html sources (before deployement)
mkdocs build

# this will create directory `site` that can directly be uploaded to
# a web server
```

# Other references

- AMR on FPGA : FP-AMR: A Reconfigurable Fabric Framework for Adaptive Mesh Refinement Applications,  Tianqi Wang ; Tong Geng ; Xi Jin ; Martin Herbordt , https://ieeexplore.ieee.org/document/8735523

## Other codes (same as in [khamr](https://gitlab.maisondelasimulation.fr/pkestene/khamr/) Readme)

In no particular order, interesting codes implementing parallel AMR algorithms, either standalone, either
by using external libraries (e.g. HPX, ...):

- https://github.com/lanl/CLAMR
- https://github.com/GEM3D/Hash_vs_RB
- https://github.com/STEllAR-GROUP/octotiger
- https://github.com/wahibium/Daino
- https://bitbucket.org/amunteam/amun-code
- https://github.com/dattv/QTAdaptive
- https://github.com/jamesliu516/amrdg2d
- https://github.com/trevor-vincent/d4est

About Morton:
- http://ashtl.sourceforge.net/
