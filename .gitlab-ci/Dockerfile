#Dockerfile for dyablo-ci:11.0-visu

FROM nvidia/cuda:11.0-devel
#FROM nvidia/cuda:10.2-devel
#FROM nvidia/cuda:9.2-devel

RUN export DEBIAN_FRONTEND=noninteractive; apt update && apt install -y git cmake libhwloc-dev libxml2-dev libopenmpi-dev libhdf5-openmpi-dev libboost-test-dev

#only for :cuda-11.0-visu tag
RUN export DEBIAN_FRONTEND=noninteractive; apt install -y python3-paraview
RUN export DEBIAN_FRONTEND=noninteractive; apt install -y xvfb

#See https://github.com/open-mpi/ompi/issues/4948
ENV OMPI_MCA_btl="^vader"


RUN useradd -ms /bin/bash gitlab
USER gitlab
WORKDIR /home/gitlab