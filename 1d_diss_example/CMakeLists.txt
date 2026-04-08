set(diss_EXECUTABLES Main_diss.x)
set(EXE_Main_diss.x_SOURCES Main.cpp Hubb_diss.cpp)

# ~~ Paths and Subdirs ~~
# Keeps the local directory included so Main_diss.x can find local headers like lattice.hpp
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

foreach(exe ${diss_EXECUTABLES})
    # declare the target...
    add_executable(${exe} ${EXE_${exe}_SOURCES})
    
    # Linking h_nessi automatically gives this executable access to include/h_nessi/
    target_link_libraries(${exe} h_nessi ${COMMON_LIBS} fftw3)

endforeach()
