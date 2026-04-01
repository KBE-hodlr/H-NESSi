set(programs SC.x)
set(EXE_SC.x_SOURCES SC.cpp SC_gf2.cpp)

foreach(exe ${programs})
    # Declare the executable
    add_executable(${exe} ${EXE_${exe}_SOURCES})
    
    # Link the target. This automatically pulls in the include/h_nessi paths!
    target_link_libraries(${exe} h_nessi ${COMMON_LIBS})

endforeach()

