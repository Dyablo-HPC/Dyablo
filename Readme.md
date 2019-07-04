# What is EulerPablo ?

It is a just a attemp to rewrite again an AMR (Adaptive Mesh Refinement) miniapp (with both shared and distributed parallelism in mind). We do not write from scratch but try to couple some of the best state-of-the-art tools.

- AMR is delegated to C++ [BitPit/PABLO](https://github.com/optimad/bitpit) library. BitPit only uses MPI (distributed memory parallelism). 
- numerical scheme is built on top of the PABLO mesh, in a decoupled manner, using [kokkos](https://github.com/kokkos/kokkos) for shared memory parallelism.

Why did we chose BitPit/PABLO for this test ? What are the difference with [p4est](http://www.p4est.org/) used in [Canop](https://gitlab.maisondelasimulation.fr/canoPdev/canoP) ?
- [p4est](http://www.p4est.org/) is written in C, about 40 kSLOC; cell-based AMR; manage of forest of octrees, i.e. the physical domain is made of a coarse mesh (p4est connectivty), and each cell of this coarse mesh serves as a root to an octree. Some of the core algorithms used in p4est are very complex due to the management of a forest of octree in distributed memory
- [BitPit/PABLO](https://github.com/optimad/bitpit) also implements cell-based AMR but on a single cubic box; this is a major difference with [p4est](http://www.p4est.org/); it is written in C++, the core code is pleasant to read. It terms of size, BitPit/PABLO is less than half of p4est SLOC.


# What is the plan ?

1. Re-implement Hydro/Euler à la Ramses (as done in [Canop](https://gitlab.maisondelasimulation.fr/canoPdev/canoP), i.e. on cell per leaf of the octree); mesh is managed by PABLO, computational kernels are written in Kokkos. As the computational kernels need to access cell connectivity, the first milestone will be to target MPI + Kokkos/OpenMP only, so that we will enable the use of CPU-only routine (from PABLO) in Kokkos kernels.
2. Test the performance of Euler/Pablo in MPI + Kokkos/OpenMP on a large cluster (e.g. skylake) and compare with Ramses and CanoP to evaluate the cost of mesh management.
3. If 1 and 2 are OK, evaluate how much work is needed to Kokkossify PABLO itself (or a sub-part). To start with, we could consider keeping most of PABLO on CPU, but at each mesh modification (refine + coarsen) export mesh connectivity in a Kokkos::View + hashmap to be used in the computational kernels (either OpenMP, or CUDA).

EulerPablo is not correlated to [khamr](https://gitlab.maisondelasimulation.fr/pkestene/khamr) yet, will use some of its ideas (e.g Kokkos HashMap implementation).


Performance portability means, we will be using the [Kokkos library](https://github.com/kokkos/kokkos), a C++ parallel programing model and library for performance portability.

# How to build ?

## Get the sources

Make sure to clone this repository recursively, this will also download kokkos source as a git submodule.

```bash
git clone --recurse-submodules git@gitlab.maisondelasimulation.fr:pkestene/EulerPablo.git
```

Kokkos and BitPit/PABLO are built as part of EulerPablo with the cmake build system.

## prerequisites

- [kokkos](https://github.com/kokkos/kokkos) : preconfigured as a git submodule
- [BitPit/PABLO](https://github.com/optimad/bitpit) : a local copy (slightly refactored) is included in EulerPablo

## build EulerPablo

### build for Kokkos/OpenMP

To build kokkos/OpenMP backend

```bash
mkdir build_openmp; cd build_openmp
ccmake -DKOKKOS_ENABLE_OPENMP=ON ..
make
```

Optionally, you can (recommended) activate HWLOC support by turning ON the flag KOKKOS_ENABLE_HWLOC.


### build for Kokkos/Cuda

Obviously, you need to have Nvidia/CUDA driver and toolkit installed on your platform.
Then you need to

 1. tell cmake to use kokkos compiler wrapper for cuda:
 
    ```shell
    export CXX=/complete/path/to/kokos/bin/nvcc_wrapper
    ```
    
 2. activate CUDA backend in the ccmake interface. 
    * Just turn on KOKKOS_ENABLE_CUDA 
    * select cuda architecture, e.g. set KOKKOS_ARCH to Kepler37 (for Nvidia K80 boards)
    
    ```bash
    # example build for cuda
    mkdir build_cuda; cd build_cuda
    ccmake -DKOKKOS_ENABLE_CUDA=ON -DKOKKOS_ARCK=Kepler37 -DKOKKOS_ENABLE_CUDA_LAMBDA=ON -DKOKKOS_ENABLE_HWLOC=ON ..
    make
    ```

# More information

For now, just visit the wiki page https://gitlab.maisondelasimulation.fr/pkestene/EulerPablo/wikis/home

# Other references

## Codes (same as in khamr Readme)

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