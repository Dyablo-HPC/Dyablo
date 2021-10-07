# What is dyablo ?

Dyablo stands for DYnamics adaptive mesh refinement CFD applications with PABLO.

It is a just a attemp to rewrite an AMR (Adaptive Mesh Refinement) code (with both shared and distributed parallelism in mind). We do not write from scratch but try to couple some of the best state-of-the-art tools.

- AMR is delegated to C++ library [BitPit/PABLO](https://github.com/optimad/bitpit). BitPit only uses MPI (distributed memory parallelism). 
- numerical scheme is built on top of the PABLO mesh, in a decoupled manner, using [kokkos](https://github.com/kokkos/kokkos) for shared memory parallelism.

Why PABLO, what are the alternatives ?
- [p4est](http://www.p4est.org/) (used in [Canop](https://gitlab.maisondelasimulation.fr/canoPdev/canoP)) is written in C, about 40 kSLOC; cell-based AMR; manage of forest of octrees, i.e. the physical domain is made of a coarse mesh (p4est connectivty), and each cell of this coarse mesh serves as a root to an octree. Some of the core algorithms used in p4est are very complex due to the management of a forest of octree in distributed memory
- [BitPit/PABLO](https://github.com/optimad/bitpit) also implements cell-based AMR but on a single cubic box; this is a major difference with [p4est](http://www.p4est.org/); it is written in C++, the core code is pleasant to read. It terms of size, BitPit/PABLO is less than half of p4est SLOC.
- Implementing our own AMR backend (more details on the [Dyablo wiki](https://gitlab.maisondelasimulation.fr/pkestene/dyablo/-/wikis/About-AMR-GPU-implementation)). This will allow us to fully utilize Kokkos acceleration for the AMR algorithms, but it needs a lot of work to be implemented correctly. 

dyablo is not correlated to [khamr](https://gitlab.maisondelasimulation.fr/pkestene/khamr) yet, will use some of its ideas (e.g Kokkos HashMap implementation).

Performance portability means, we will be using the [Kokkos library](https://github.com/kokkos/kokkos), a C++ parallel programing model and library for performance portability.

# How to build ?

## Get the sources

Dyablo includes Kokkos and PABLO as git submodules. Make sure to clone this repository recursively, this will also download Kokkos and PABLO from github.

```bash
git clone --recurse-submodules git@gitlab.maisondelasimulation.fr:pkestene/dyablo.git
```

or if you already cloned the repository without --recurse-submodules :

```bash
git submodule update --init
```

For the latest version of Dyablo, we recommend that you use the `dev` branch. 

NOTE : If you don' have access to github from the machine (e.g at TGCC), you will need to populate the `external/` folder manually with [kokkos](https://github.com/kokkos/kokkos), [bitpit](https://github.com/pkestene/bitpit.git) and [backward-cpp](https://github.com/bombela/backward-cpp.git)
 
## build dyablo

### Dependencies

The CMake superbuild should automatically find dependencies and warn you if any dependency is missing.

You will need a recent C++ compiler capable of compiling Kokkos. We regularly compile Dyablo using :
* `g++` (7.5, 9.1, 11.1, ...)
* `icc` (19, 20)

Other dependencies include :
* MPI (OpenMPI)
* HDF5 (parallel)
* libxml2

To compile for GPU, a CUDA installation is needed, preferably newer than CUDA 11.0. Versions before CUDA 10 may need a custom version of PABLO to compile, you can find it here : [PABLO](git@github.com:pkestene/bitpit.git), branches ending with *-dyablo. Dyablo supports both CUDA-Aware and non CUDA-Aware MPI implementations when compiling for GPU, make sure that cuda-aware support has been correctly detected in the CMake logs.

Build commands for some HPC clusters are available [here](https://gitlab.maisondelasimulation.fr/pkestene/dyablo/-/wikis/Compilation-instructions-for-super-computers) to help you find modules that work well with each others.

### Superbuild : build bitpit/PABLO, Kokkos and dyablo alltogether

The top-level `CMakeLists.txt` uses the the super-build pattern to build Dyablo and its depencies (here bitpit and Kokkos) using cmake command [ExternalProject_Add](https://cmake.org/cmake/help/latest/module/ExternalProject.html). Using the superbuild is the recommended way to compile Dyablo because it ensures that the Kokkos compilaton configuration (Architecture, enabled backends, etc...) is compatible with how Dyablo is configured.

To build bitpit, Kokkos and dyablo (for Kokkos/OpenMP backend which is the default)

```bash
mkdir build_openmp; cd build_openmp
cmake ..
make
```

The same for Kokkos/CUDA (e.g. for latest Turing CUDA architecture):
```bash
mkdir build_cuda; cd build_cuda
cmake -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH=TURING75 ..
make
```

See The [Kokkos documentation](https://github.com/kokkos/kokkos/wiki/Compiling#table-43-architecture-variables) to find the correct architecture to target your GPU or CPU. You may target multiple architectures : for example to compile for a machine with Intel Skylake CPU + V100 GPUs, you might want to set `-DKokkos_ARCH="SKX;VOLTA70"`

NOTE : You don't have to use the nvcc_wrapper for Kokkos as the CXX compiler like in a normal Kokkos project, the superbuilt takes care of that for you.

NOTE : The use of the ExternalProject command in cmake may interfere with IDEs automatic cmake configuration to setup autocomplete or IntelliSense features. Using the `build/dyablo` (created during `make`) as the cmake binary path for your project may help your IDE to detect the source/include path.

### Use an external Kokkos or PABLO installation.

The `core` directory is an independant cmake project on its own. You can create a build directory there and compile it like any normal cmake project. See the [Kokkos documentation for general instructions on how to build a Kokkos project](https://github.com/kokkos/kokkos/wiki/Compiling). You will also need to have a PABLO installation that cmake can detect. 


## Run Dyablo

The main executable `test_solver` is in `build/dyablo/test/solver/`. The directory also contains some *.ini files that can passed to the executable on the command line. For instance, to run a 2d Sedov-blast on the block-AMR backend, the command will be `./test_solver test_blast_2D_block.ini`

Beware, when recompiling, the .ini files may be reset to their original state.

run for instance `./test_solver test_blast_3d_block.ini` to run the 3D block-base blast test case. This executable accepts [Kokkos command-line parameters](https://github.com/kokkos/kokkos/wiki/Initialization).

For the best performance, you should follow the global advice for any Kokkos program :
* Configure OpenMP to bind threads by setting the environment variable OMP_PROC_BIND=true
* Use the Kokkos command line arguments to correcty bind GPUs : e.g. `--num-devices=4` when there are 4 GPUs/node
* Of course, learn to use your job manager (slurm) efficiently for hybrid codes (`-c`, `--hint`, etc...).

### Config parameters (.ini file)

Unfortunately no documentation about the content of the .ini file is available yet, you will need to read the code (sorry). For now you can `grep -r configMap.get core/src` to see what variables are fetched from the .ini.

### Output and visualization

If you used the XDMF+HDF5 output backend, Dyablo should produce *.h5 and *.xmf files that contain unstructured mesh data. You can open *_main.xmf with [paraview](https://www.paraview.org/) to display the output of your simulation.

More IO implementations will be added later.


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

## Build documentation

### Requirements

- [doxygen](https://www.doxygen.nl/)
- (optional, but recommended) [mkdocs](https://www.mkdocs.org/) for building a static webpage with documentation, written in markdown
   ```shell
   # we recommend using miniconda for installing python packages
   conda install -c conda-forge mkdocs
   ```
   there is an additionnal useful package, [markdown_katex](https://pypi.org/project/markdown-katex/) for integrating latex equation in markdown; currently there is no conda package, so you must install it with `pip`. 
- (optional, but recommended) [doxybook2](https://github.com/matusnovak/doxybook2) which provides some glue code to integrate doxygen into mkdocs, the resulting webpage is better than a regular doxygen documentation. We recommend installing [doxybook2](https://github.com/matusnovak/doxybook2) using a binary release package from [https://github.com/matusnovak/doxybook2/tags](https://github.com/matusnovak/doxybook2/tags). Make sure to have doxybook2 executable in your PATH environment variable.
  ```shell
  export PATH=${DOXYBOOK2_INSTALL_ROOT}/bin:$PATH
  ```

### [doxygen](https://www.doxygen.nl/)

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
# from the build directory
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
