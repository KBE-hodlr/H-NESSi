include_directories(
  SYSTEM
)

set (programs
	SC.x
)
set( EXE_SC.x_SOURCES SC.cpp SC_gf2.cpp )


foreach(exe ${programs})
  STRING(REGEX REPLACE ".cpp" ".ex" executable_file "${source_file}")  
  add_executable(${exe} ${EXE_${exe}_SOURCES})
  target_link_libraries(${exe} hodlr ${COMMON_LIBS} fftw3_omp)
  if(INCLUDE_NESSI)
    target_link_libraries(${exe} cntr)
  endif(INCLUDE_NESSI)
  install(TARGETS ${exe} DESTINATION bin)
endforeach(exe)
