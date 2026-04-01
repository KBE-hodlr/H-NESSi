# TODO:
finalize paper \
polish README include link to paper when on arxiv \
documentation can be found at https://kbe-hodlr.github.io/H-NESSi/ \

# HODLR [hierarchical off-diagonal low-rank structure] implementation for the Greens function
[![DOI](https://zenodo.org/badge/1180109148.svg)](https://doi.org/10.5281/zenodo.18990387)

The set of classes implementing the basic hodlr extension 
of the nessi library with example programs and test.

# To compile
We recommend using a build script build.sh to pass parameters to cmake.  We provide an example script below

# Example build.sh
#!/usr/bin/env bash
cmake                               \
  -DCMAKE_BUILD_TYPE=Release        \
  -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -g -std=c++11"  \
  -DCMAKE_PREFIX_PATH=${FFTW_DIR}     \
  -DLIBDLR_LIB=/path/to/libdlr/build/lib/ \
  -DLIBDLR_HEADER=/path/to/libdlr/src/ \
  -DINCLUDE_NESSI=1 \
  -DNESSI_INCLUDE_DIR=/path/to/libcntr/installation/include/ \
  -DNESSI_LIB=~/path/to/libcntr/installation/lib/ \
  -DBUILD_TEST=1 \
..


