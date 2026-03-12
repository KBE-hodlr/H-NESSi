include_directories(
  SYSTEM
  )

add_library(hodlr SHARED
  herm_matrix_hodlr.cpp
  utils.cpp
  function.cpp
  block.cpp
  blocks_keldysh.cpp
  dyson.cpp
  integration.cpp
  read_inputfile.cpp
  dlr.cpp
  mpi_comm.cpp
  mpi_comm_utils.cpp
  )

target_link_libraries(hodlr ${HDF5_LIBRARIES} HighFive dlr_c ${COMMON_LIBS} fftw3_omp)
if(INCLUDE_NESSI)
  target_link_libraries(hodlr cntr)
endif()

file(GLOB hodlr_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
set_target_properties(hodlr PROPERTIES PUBLIC_HEADER "${hodlr_HEADERS}")

install(TARGETS hodlr LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION include/hodlr )


