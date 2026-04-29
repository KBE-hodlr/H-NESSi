cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 20) 
project("h_nessi")

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(COMMON_LIBS "")

# ~~ Add OpenMP ~~
message(STATUS "Building with OpenMP")
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    
# ~~ Add MPI ~~
message(STATUS "Building with OpenMPI")
find_package(MPI REQUIRED)
set(CMAKE_C_COMPILE_FLAGS "${CMAKE_C_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS}")
set(CMAKE_CXX_COMPILE_FLAGS
        "${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS}")
set(CMAKE_C_LINK_FLAGS "${CMAKE_C_LINK_FLAGS} ${MPI_LINK_FLAGS}")
set(CMAKE_CXX_LINK_FLAGS
        "${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS}")
include_directories(${MPI_INCLUDE_PATH})
list(APPEND COMMON_LIBS ${MPI_LIBRARIES})

# ~~ Add Eigen ~~
find_package(Eigen3 REQUIRED NO_MODULE)
if(Eigen3_VERSION VERSION_LESS "3.4")
    message(FATAL_ERROR "Eigen3 version 3.4 or higher required. Found ${Eigen3_VERSION}")
endif()

include_directories(${EIGEN3_INCLUDE_DIR})
message(STATUS "Eigen dir: ${EIGEN3_INCLUDE_DIR}")
add_compile_definitions(EIGEN_DONT_PARALLELIZE)

# ~~ Add FFTW ~~
find_path(FFTW3_INCLUDE_DIR NAMES fftw3.h 
          PATHS "${FFTW_DIR}/include" $ENV{FFTW_DIR}/include 
          NO_DEFAULT_PATH)
find_path(FFTW3_INCLUDE_DIR NAMES fftw3.h) 

find_library(FFTW3_LIB NAMES fftw3 
             PATHS "${FFTW_DIR}/lib" $ENV{FFTW_DIR}/lib 
             NO_DEFAULT_PATH)
find_library(FFTW3_LIB NAMES fftw3)

find_library(FFTW3_OMP_LIB NAMES fftw3_omp 
             PATHS "${FFTW_DIR}/lib" $ENV{FFTW_DIR}/lib 
             NO_DEFAULT_PATH)
find_library(FFTW3_OMP_LIB NAMES fftw3_omp)

if(NOT FFTW3_INCLUDE_DIR OR NOT FFTW3_LIB OR NOT FFTW3_OMP_LIB)
    message(FATAL_ERROR "Could not find FFTW3 or its OpenMP components. Please set FFTW_DIR.")
else()
    message(STATUS "Found FFTW3 Headers: ${FFTW3_INCLUDE_DIR}")
    message(STATUS "Found FFTW3 Library: ${FFTW3_LIB}")
    message(STATUS "Found FFTW3 OMP Library: ${FFTW3_OMP_LIB}")
endif()

# ~~ Add HDF5 ~~
find_package(HDF5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIR})
link_directories(${HDF5_DIR})
message(STATUS "HDF5 dir: ${HDF5_INCLUDE_DIR}")

# ~~ Manually add Libdlr ~~
include_directories(${LIBDLR_HEADER})
link_directories(${LIBDLR_LIB})

# Default CPM cache location (persistent across build dirs)
if(NOT DEFINED CPM_SOURCE_CACHE)
  set(CPM_SOURCE_CACHE
      "${CMAKE_SOURCE_DIR}/.cpm-cache"
      CACHE PATH "CPM source cache directory"
  )
endif()
include(CPM)

# ~~ fetch HighFive from GitHub ~~
set(_old_log_level ${CMAKE_MESSAGE_LOG_LEVEL})
set(CMAKE_MESSAGE_LOG_LEVEL ERROR)
CPMAddPackage(
  NAME HighFive
  GITHUB_REPOSITORY BlueBrain/HighFive
  GIT_TAG master
  OPTIONS
  "HIGHFIVE_USE_BOOST OFF"
  "HIGHFIVE_USE_EIGEN ON"
  "HIGHFIVE_UNIT_TESTS OFF"
  "HIGHFIVE_EXAMPLES OFF"
)
set(CMAKE_MESSAGE_LOG_LEVEL ${_old_log_level})

# ~~ Paths and Subdirs ~~
# UPDATED: We point this to your new include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ~~ Option to compile with Nessi ~~
option(INCLUDE_NESSI "Compile functions that use nessi objects" OFF)
if(INCLUDE_NESSI)
    message(STATUS "Including nessi headers")
    include_directories(${NESSI_INCLUDE_DIR})
    link_directories(${NESSI_LIB})
    add_definitions("-DCNTR_USE_HDF5")
    add_definitions("-DCNTR_USE_OMP")
    add_definitions("-DCNTR_USE_MPI")
endif(INCLUDE_NESSI)

# ~~ Build tests ~~
option(BUILD_TEST "Build test suite" OFF)
if(BUILD_TEST)
    message(STATUS "Building tests with googletest")
    CPMAddPackage(
      NAME googletest
      GITHUB_REPOSITORY google/googletest
      VERSION 1.14.0    # pick a known release
      OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt ON"
    )
    enable_testing()
    add_subdirectory(tests)
endif(BUILD_TEST)

# UPDATED: We build the library from the new src/ directory
add_subdirectory(src)

add_subdirectory(SC_example)
add_subdirectory(mpi_example)
