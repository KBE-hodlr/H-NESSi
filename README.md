# H-NESSi: The Hierarchical Non-Equilibrium Systems Simulation package

[![DOI](https://zenodo.org/badge/1180109148.svg)](https://doi.org/10.5281/zenodo.18990387)
[![arXiv](https://img.shields.io/badge/arXiv-2604.05319-b31b1b.svg)](https://arxiv.org/abs/2604.05319)

H-NESSi is an open-source C++ software package for solving the Kadanoff-Baym equations (KBE) of nonequilibrium Green's function (NEGF) theory. 

By combining high-order time-stepping schemes with hierarchical off-diagonal low-rank (HODLR) compression techniques and the discrete Lehmann representation (DLR), H-NESSi bypasses the cubic time scaling and quadratic memory growth of conventional two-time formulations. This enables long-time and large-system nonequilibrium simulations of correlated quantum materials. This repository contains the set of classes implementing the basic HODLR extension of the NESSi library, along with example programs and unit tests.

## Documentation

Full API documentation, tutorials, and examples can be found at:  
**[https://kbe-hodlr.github.io/H-NESSi/](https://kbe-hodlr.github.io/H-NESSi/)**

## Building from Source

### Prerequisites
Before compiling, ensure you have the following dependencies installed and accessible on your system:
* [FFTW](https://www.fftw.org/)
* [libdlr](https://github.com/jasonkaye/libdlr)
* [NESSi (optional)](https://github.com/nessi-cntr/nessi)

### Compilation
We recommend using a `build.sh` script to pass the necessary path parameters to CMake. 

1. Create a `build` directory and navigate into it:
   ```bash
   mkdir build
   cd build
2. Create a `build.sh` script. Below is an example configuration—you will need to update the `/path/to/...` variables to match your local installation directories:

   ```bash
   #!/usr/bin/env bash
   
   cmake \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_INSTALL_PREFIX="/path/to/install/location/" \
     -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -g -std=c++11" \
     -DCMAKE_PREFIX_PATH=${FFTW_DIR} \
     -DLIBDLR_LIB=/path/to/libdlr/build/lib/ \
     -DLIBDLR_HEADER=/path/to/libdlr/src/ \
     -DINCLUDE_NESSI=1 \
     -DNESSI_INCLUDE_DIR=/path/to/libcntr/installation/include/ \
     -DNESSI_LIB=/path/to/libcntr/installation/lib/ \
     -DBUILD_TEST=1 \
     ..
   ```

3. Make the script executable and run it:
   ```bash
   chmod +x ../build.sh
   ./../build.sh
   ```

4. Compile and install the code:
   ```bash
   make
   make install
   ```

## Citation

If you use H-NESSi in your research, please consider citing our paper:

> **H-NESSi: The Hierarchical Non-Equilibrium Systems Simulation package** \
>  Thomas Blommel, Jeremija Kovačević, Jason Kaye, Emanuel Gull, Jakša Vučičević, Denis Golež \
> *[Journal / arXiv link](https://arxiv.org/abs/2604.05319)*

## License

This project is licensed under the MIT License.
