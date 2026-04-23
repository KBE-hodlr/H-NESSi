set(programs Nonint_diss_tests.x)
set(EXE_Nonint_diss_tests.x_SOURCES Nonint_diss_tests.cpp)

foreach(exe ${programs})
    # Declare the executable
    add_executable(${exe} ${EXE_${exe}_SOURCES})

    # Link the target. This automatically pulls in the include/h_nessi paths!
    target_link_libraries(${exe} h_nessi ${COMMON_LIBS})

endforeach()
