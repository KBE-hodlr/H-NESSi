add_library(h_nessi SHARED
  herm_matrix_hodlr.cpp
  herm_matrix_hodlr_distance.cpp
  herm_matrix_hodlr_getit.cpp
  herm_matrix_hodlr_getreal.cpp
  herm_matrix_hodlr_ptrmap.cpp
  herm_matrix_hodlr_set.cpp
  herm_matrix_hodlr_update.cpp
  herm_matrix_hodlr_write.cpp
  utils.cpp
  function.cpp
  block.cpp
  blocks_keldysh.cpp
  dyson.cpp
  dyson_diss.cpp
  dyson_free.cpp
  dyson_it_integrals.cpp
  dyson_mat.cpp
  dyson_start.cpp
  dyson_start_diss.cpp
  dyson_timestep.cpp
  dyson_timestep_diss.cpp
  integration.cpp
  read_inputfile.cpp
  dlr.cpp
  mpi_comm.cpp
  mpi_comm_utils.cpp
  mpi_comm_vecload.cpp
)

# Tell the h_nessi target where to find its headers, both during build and after installation
target_include_directories(h_nessi PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link your dependencies
target_link_libraries(h_nessi ${HDF5_LIBRARIES} HighFive dlr_c ${COMMON_LIBS} fftw3_omp)

if(INCLUDE_NESSI)
  target_link_libraries(h_nessi cntr)
endif()

# Grab the headers from the new directory structure (pointing to the project root -> include -> h_nessi)
file(GLOB h_nessi_HEADERS 
    "${CMAKE_SOURCE_DIR}/include/h_nessi/*.hpp"
    "${CMAKE_SOURCE_DIR}/include/h_nessi/*.h"
)
