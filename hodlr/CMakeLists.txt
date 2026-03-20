include_directories(
  SYSTEM
  )

add_library(hodlr SHARED
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
  dyson_free.cpp
  dyson_it_integrals.cpp
  dyson_mat.cpp
  dyson_start.cpp
  dyson_timestep.cpp
  integration.cpp
  read_inputfile.cpp
  dlr.cpp
  mpi_comm.cpp
  mpi_comm_utils.cpp
  mpi_comm_vecload.cpp
  )

target_link_libraries(hodlr ${HDF5_LIBRARIES} HighFive dlr_c ${COMMON_LIBS} fftw3_omp)
if(INCLUDE_NESSI)
  target_link_libraries(hodlr cntr)
endif()

file(GLOB hodlr_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
set_target_properties(hodlr PROPERTIES PUBLIC_HEADER "${hodlr_HEADERS}")

install(TARGETS hodlr LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION include/hodlr )


