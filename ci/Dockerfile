#Dockerfile for dyablo-ci:11.0-visu

FROM nvidia/cuda:11.0-devel

RUN export DEBIAN_FRONTEND=noninteractive; apt update && apt install -y git cmake libhwloc-dev libxml2-dev libopenmpi-dev libhdf5-openmpi-dev libboost-test-dev
RUN export DEBIAN_FRONTEND=noninteractive; apt install -y python3-paraview
