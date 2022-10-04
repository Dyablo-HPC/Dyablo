# What is dyablo ?

Dyablo is a framework to develop Computational Fluid Dynamics (CFD) simulations using Adaptive Mesh Refinement (AMR) on large scale supercomputers. 

It's an attempt to modernize de software stack, initially for numerical simulations for astrophysics. Dyablo is written in C++ with performance portability in mind and uses an MPI+Kokkos hybrid approach to parallelism. 

The MPI Library is used for distributed parallelism and compute kernels using shared-memory parallelism use the Kokkos performance portability library. MPI is used to distribute the AMR mesh across multiple compute nodes, while Kokkos allows us to write a single code that can be executed on multithreaded CPUs, GPUs and other parallel architectures supported by Kokkos. 

Dyablo is also build with modularity and ease of use in mind to allow physicists to easily add new kernels written with abstract interfaces to access and modify the AMR mesh. 

Modularity is also key to use state-of-the-art libraries interchangeably, reuse existing work and allow compatibility with external tools for vizualization or post-processing for instance. In Dyablo, the AMR mesh can be managed by the PABLO external library or by the custom implementation in written in Kokkos. This modularity enables us to plug in other external libraries to manage the AMR mesh or to perform IO, vizualization or post-processing operations. For now vizualisation outputs are handled by the HDF5 library but other backends can be integrated to Dyablo through plug-ins.

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

For the latest version of Dyablo, we recommend that you use the `dev` (default) branch . 

NOTE : If you don' have access to github from the machine (e.g at TGCC), you will need to populate the `external/` folder manually with [kokkos](https://github.com/kokkos/kokkos), [bitpit](https://github.com/pkestene/bitpit.git) and [backward-cpp](https://github.com/bombela/backward-cpp.git). If you want to build unit-tests you also need [gtest](https://github.com/google/googletest).
 
## build dyablo

### Dependencies

The CMake superbuild should automatically find dependencies and warn you if any dependency is missing. CMake version > 3.16 is needed to compile Kokkos.

You will need a recent C++ compiler compatible with C++17 and capable of compiling Kokkos. Recommended compiler versions for Dyablo are :
* `g++` > 8.2
* `icc` > 19.0.5
* `clang` > 8.0
* `nvcc` > 11.2
* ...

Other dependencies include :
* MPI
* HDF5 (parallel)
* libxml2

#### Dependencies for GPU

To compile for GPU, a CUDA installation is needed, preferably newer than CUDA 11.2. Dyablo supports both CUDA-Aware and non CUDA-Aware MPI implementations when compiling for GPU, make sure that cuda-aware support has been correctly detected in the CMake logs.

Kokkos automatically detects and sets the CUDA compiler when Kokkos_ENABLE_CUDA is ON :
* When the C++ compiler is not compatible with CUDA, Kokkos uses NVCC to compile device code. NVCC version must be >= 11.2
* When the C++ compiler is compatible with CUDA (e.g clang), Kokkos uses this compiler
For more details, see the [Kokkos documentation](https://github.com/kokkos/kokkos/wiki/Compiling)

### Superbuild : build bitpit/PABLO, Kokkos and dyablo alltogether

The top-level `CMakeLists.txt` uses the the super-build pattern to build Dyablo and its depencies (here PABLO and Kokkos) using cmake command [ExternalProject_Add](https://cmake.org/cmake/help/latest/module/ExternalProject.html). Using the superbuild is the recommended way to compile Dyablo because it ensures that the Kokkos compilation configuration (Architecture, enabled backends, etc...) is compatible with how Dyablo is configured.

To build bitpit, Kokkos and dyablo (for Kokkos/OpenMP backend which is the default)

```bash
mkdir build_openmp; cd build_openmp
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

The same for Kokkos/CUDA:
```bash
mkdir build_cuda; cd build_cuda
cmake -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_CUDA=ON  ..
make
```

Build commands for some linux distributions and supercomputers can be found [here](https://gitlab.maisondelasimulation.fr/pkestene/dyablo/-/wikis/Compilation-instructions-for-super-computers) to help you find the right packages or modules as well as the command line to use to compile Dyablo.

Configuration options for Dyablo ( `cmake -D` arguments ) include :
- `-DCMAKE_BUILD_TYPE=<buildtype>` : `Release` (recommended for performance), `Debug` (enables asserts and debug symbols), `RelWithDebInfo`
- `-DKokkos_ENABLE_CUDA=ON/OFF` : enable CUDA
- `-DKokkos_ARCH="<arch1>,..."` : target specific architectures (i.e. set architecture-specific optimization flags) listed in the [Kokkos documentation](https://github.com/kokkos/kokkos/wiki/Compiling#table-43-architecture-variables). You may target multiple architectures : for example to compile for a machine with Intel Skylake CPU + V100 GPUs, you might want to set `-DKokkos_ARCH="SKX;VOLTA70"`. CUDA architecture is auto-detected if `Kokkos_ARCH` is not present.
- `-DDYABLO_ENABLE_UNIT_TESTING=ON/OFF` : enable/disable unit tests. If enabled, run `make dyablo-test` from your build directory to run all tests

Compilation may take a long time, we recommend you use parallel compilation with `make -j <number of cores>`.

## Run Dyablo

The main executable `test_solver` is in `build/dyablo/test/solver/`. The directory also contains some *.ini files that can passed to the executable on the command line. For instance, to run a 2d Sedov-blast on the block-AMR backend, the command will be `./test_solver test_blast_2D_block.ini`

Beware, when recompiling, the .ini files may be reset to their original state.

run for instance `./test_solver test_blast_3D_block.ini` to run the 3D block-base blast test case. This executable accepts [Kokkos command-line parameters](https://github.com/kokkos/kokkos/wiki/Initialization).

For the best performance, you should follow the global advice for any Kokkos program :
* Configure OpenMP to bind threads by setting the environment variable OMP_PROC_BIND=true
* Use the Kokkos command line arguments to correcty bind GPUs : e.g. `--num-devices=4` when there are 4 GPUs/node
* Of course, learn to use your job manager (slurm) efficiently for hybrid codes (`-c`, `--hint`, etc...).


### Config parameters (.ini file)

Dyablo uses `.ini` files for configuration. The format is described [here](https://en.wikipedia.org/wiki/INI_file). `key=value` lists are arranged into `[sections]` : 

```ini
[mesh]
# 2D or 3D
ndim=2 

xmin=0.0
xmax=1.0
ymin=0.0
ymax=1.0

[amr]
level_min=5
level_max=7

[...]
```

Dyablo generates a `last.ini` file listing the values used for all the variables needed for the last simulation. Values in `last.ini` are annotated with :
- `Default` : the value was not listed in the original `.ini` file and the value shown in `last.ini` is the defaut value chosen by Dyablo
- `Unused` : The value was listed in the original `.ini` but was not used by Dyablo

Unfortunately no documentation about the content of the .ini file is available yet, you will need to read the code (sorry). For now you can `grep -r configMap.get core/src` to see what variables are fetched from the .ini.

### Output and visualization

If you used the XDMF+HDF5 output backend, Dyablo should produce *.h5 and *.xmf files that contain unstructured mesh data. You can open *_main.xmf with [paraview](https://www.paraview.org/) to display the output of your simulation.


# More information

For now, just visit the wiki page https://gitlab.maisondelasimulation.fr/pkestene/dyablo/wikis/home



<!---
** Documentation is not working yet **
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
-->

