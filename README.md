# HODLR [hierarchical off-diagonal low-rank structure] implementation for the Greens function
The set of classes implementing the basic hodlr extension 
of the nessi library with example programs and test.

# To compile
To compile and test the code, we use cmake.  Below is an example of how one can build this package, starting from the root directory.
We recommend using a build script build.sh to pass parameters to cmake.  We provide an example script below

> mkdir build
> cd build
> sh ../build.sh
> make
> make test

# Example build.sh
#!/usr/bin/env bash
cmake                               \
  -DCMAKE_BUILD_TYPE=Release        \
  -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -g -std=c++11"  \
  -DCMAKE_PREFIX_PATH=${FFTW_DIR}     \
  -DLIBDLR_LIB=~/path/to/libdlr/build/lib/ \
  -DLIBDLR_HEADER=~/path/to/libdlr/src/ \
  -DINCLUDE_NESSI=1 \
  -DNESSI_INCLUDE_DIR=~/path/to/libcntr/installation/include/ \
  -DNESSI_LIB=~/path/to/libcntr/installation/lib/ \
  -DBUILD_TEST=1 \
..


