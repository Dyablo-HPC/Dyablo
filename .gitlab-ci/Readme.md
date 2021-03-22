# Dyablo ci

These scripts are meant to be used for gitlab-ci

## Gitlab runner docker configuration

To build and run openmp test cases, runner must have tag `docker`

For CUDA testcases , runner must have tag 'docker-cuda' and be configured to support GPUs :
* nvidia-docker2 must be installed
* config.toml must be configured to use GPUs over docker
```toml
[[runners]]
  url = "https://gitlab.maisondelasimulation.fr/"
  executor = "docker"
  [...]
  [runners.docker]
    runtime = "nvidia"
    [...]
```

## Add a testcase to the pipeline

Optional test cases can be added to the ci pipeline by adding jobs in the `.gitlab-ci.yml` file. New testcases should be configured for both openmp and CUDA. To add a new testcase, two jobs should be added following this template : 

```yml
run::<test_name>::openmp:
  <<: *run_definition
  <<: *openmp_definition
  script: 
    - .gitlab-ci/run_testcase.sh <testcase input file>.ini
    - .gitlab-ci/render.sh <testcase visualization file>.pvsm

run::<test_name>::cuda:
  <<: *run_definition
  <<: *cuda_definition
  script: 
    - .gitlab-ci/run_testcase.sh <testcase input file>.ini
    - .gitlab-ci/render.sh <testcase visualization file>.pvsm
```

`<testcase input file>.ini` and `<testcase visualization file>.pvsm` are files in `build/dyablo/test/solver` that are respectively the input file to dyablo's `test_solver` executable and the paraview state file used to create de visualization image at the end of the simulation. The paraview state file can be generated when visualizing the simulation output and selecting `file > save state`. To make it compatible with the ./render.sh script, edit the .pvsm file to convert every path to the visualization files (.vtk, .xmf) to local paths (`/home/...build/dyablo/test/solver/test_..._main.xmf` -> `./test_..._main.xmf`).
