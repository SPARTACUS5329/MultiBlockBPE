#!/bin/bash

# load required modules
module load cuda
module load gcc-native/12.3
module unload cray-mpich
module load mpich

# cmake requirements
mkdir -p build
cd ./build
cmake ..
make
cd ../assets
cd ..