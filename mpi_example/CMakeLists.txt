set(mpi_EXECUTABLES Main.x)
set(EXE_Main.x_SOURCES Main.cpp lattice.cpp kpoint.cpp observables.cpp Hubb_2B.cpp)

# ~~ Paths and Subdirs ~~
# Keeps the local directory included so Main.x can find local headers like lattice.hpp
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

foreach(exe ${mpi_EXECUTABLES})
    # declare the target...
    add_executable(${exe} ${EXE_${exe}_SOURCES})
    
    target_include_directories(h_nessi PUBLIC ${FFTW3_INCLUDE_DIR})
    # Linking h_nessi automatically gives this executable access to include/h_nessi/
    target_link_libraries(${exe} h_nessi ${COMMON_LIBS})

endforeach()

