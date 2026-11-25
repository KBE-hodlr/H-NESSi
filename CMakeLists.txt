cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 20) 
project("h_nessi")
set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# ~~ Build type RelWithAsserts (decision D-6): optimised (-O2 -g) with the tier-1 checks (assert, HNESSI_ASSERT) on,
#    i.e. without NDEBUG. Eigen's own internal asserts are switched off (EIGEN_NO_DEBUG), they slow hot loops. ~~
set(CMAKE_CXX_FLAGS_RELWITHASSERTS "-O2 -g -DEIGEN_NO_DEBUG" CACHE STRING
    "C++ flags for the RelWithAsserts build type (no NDEBUG: tier-1 asserts on)" FORCE)
set(CMAKE_C_FLAGS_RELWITHASSERTS "-O2 -g" CACHE STRING "C flags for the RelWithAsserts build type" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_RELWITHASSERTS "" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_RELWITHASSERTS "" CACHE STRING "" FORCE)
mark_as_advanced(CMAKE_CXX_FLAGS_RELWITHASSERTS CMAKE_C_FLAGS_RELWITHASSERTS
                 CMAKE_EXE_LINKER_FLAGS_RELWITHASSERTS CMAKE_SHARED_LINKER_FLAGS_RELWITHASSERTS)
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
  "HIGHFIVE_USE_INSTALL OFF"
)
set(CMAKE_MESSAGE_LOG_LEVEL ${_old_log_level})

# ~~ Paths and Subdirs ~~
# UPDATED: We point this to your new include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ~~ Option to compile with Nessi ~~
# INCLUDE_NESSI=ON compiles the NESSi interoperability functions (distance_norm2_* with a cntr::herm_matrix<double>,
# herm_matrix_hodlr.hpp) and the `nessi` tests.  It needs an installed NESSi libcntr built with -Domp=ON -Dhdf5=ON
# -Dmpi=ON (the CNTR_USE_* macros below must match the libcntr build).  Give either NESSI_ROOT (install prefix) or
# NESSI_INCLUDE_DIR (directory that contains cntr/) and NESSI_LIB (directory that contains libcntr).
# The macros and the library are attached to the h_nessi target as PUBLIC usage requirements (src/CMakeLists.txt),
# so the library, the examples, the tests and downstream code all see the same INCLUDE_NESSI definition.
option(INCLUDE_NESSI "Compile functions that use nessi objects" OFF)
if(INCLUDE_NESSI)
    set(NESSI_ROOT "" CACHE PATH "NESSi (libcntr) install prefix")
    find_path(NESSI_CNTR_INCLUDE_DIR NAMES cntr/cntr.hpp
              HINTS ${NESSI_INCLUDE_DIR} ${NESSI_ROOT}/include NO_DEFAULT_PATH)
    find_path(NESSI_CNTR_INCLUDE_DIR NAMES cntr/cntr.hpp)
    find_library(NESSI_CNTR_LIBRARY NAMES cntr HINTS ${NESSI_LIB} ${NESSI_ROOT}/lib NO_DEFAULT_PATH)
    find_library(NESSI_CNTR_LIBRARY NAMES cntr)
    find_library(NESSI_CNTR_HDF5_LIBRARY NAMES cntr_hdf5 HINTS ${NESSI_LIB} ${NESSI_ROOT}/lib NO_DEFAULT_PATH)
    find_library(NESSI_CNTR_HDF5_LIBRARY NAMES cntr_hdf5)
    if(NOT NESSI_CNTR_INCLUDE_DIR OR NOT NESSI_CNTR_LIBRARY OR NOT NESSI_CNTR_HDF5_LIBRARY)
        message(FATAL_ERROR "INCLUDE_NESSI=ON but NESSi was not found (cntr/cntr.hpp: ${NESSI_CNTR_INCLUDE_DIR}, "
                            "libcntr: ${NESSI_CNTR_LIBRARY}, libcntr_hdf5: ${NESSI_CNTR_HDF5_LIBRARY}). "
                            "Set NESSI_ROOT, or NESSI_INCLUDE_DIR and NESSI_LIB.")
    endif()
    message(STATUS "Including NESSi: headers ${NESSI_CNTR_INCLUDE_DIR}, library ${NESSI_CNTR_LIBRARY}")
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

# ==============================================================================
# INSTALLATION SETUP
# ==============================================================================

# 1. Install your project's public headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/h_nessi
    DESTINATION include
)

# 2. Package HighFive's headers so downstream users can link to h_nessi
if(HighFive_SOURCE_DIR)
    message(STATUS "Packaging HighFive headers for installation...")
    install(DIRECTORY ${HighFive_SOURCE_DIR}/include/highfive
        DESTINATION include
    )
endif()
