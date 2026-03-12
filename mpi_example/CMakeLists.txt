set (mpi_EXECUTABLES
   Main.x
   test_Main.x
)

set( EXE_Main.x_SOURCES Main.cpp lattice.cpp kpoint.cpp selfene.cpp observables.cpp )
set( EXE_test_Main.x_SOURCES test_Main.cpp lattice.cpp kpoint.cpp selfene.cpp observables.cpp )


# ~~ Paths and Subdirs ~~
include_directories(${CMAKE_INCLUDE_PATH})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ~~ Libraries ~~
link_directories(${CMAKE_LIBRARY_PATH})

foreach( exe ${mpi_EXECUTABLES} )
    # declare the target...
    add_executable( ${exe} ${EXE_${exe}_SOURCES} )
    target_link_libraries(${exe} openblas dlr hodlr ${COMMON_LIBS} fftw3)
    if(INCLUDE_NESSI)
        target_link_libraries(${exe} cntr)
    endif(INCLUDE_NESSI)
endforeach()
