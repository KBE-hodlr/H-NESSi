# Usage: cmake -DBUILD_DIR=<dir> -DTARGET=<target> -DEXE=<path> -P build_and_run.cmake
# Builds a target that is excluded from ALL (a known link failure) and runs it; fails if either step fails.
execute_process(COMMAND ${CMAKE_COMMAND} --build ${BUILD_DIR} --target ${TARGET}
                RESULT_VARIABLE rb OUTPUT_VARIABLE ob ERROR_VARIABLE eb)
if(NOT rb EQUAL 0)
  string(REGEX MATCHALL "[^\n]*(multiple definition|undefined reference|error)[^\n]*" errs "${ob}${eb}")
  list(SUBLIST errs 0 6 errs)
  string(REPLACE ";" "\n" errs "${errs}")
  message(FATAL_ERROR "build of ${TARGET} failed:\n${errs}")
endif()
execute_process(COMMAND ${EXE} RESULT_VARIABLE rr OUTPUT_VARIABLE orun ERROR_VARIABLE erun)
if(NOT rr EQUAL 0)
  message(FATAL_ERROR "${TARGET} built but failed at run time (${rr}):\n${orun}${erun}")
endif()
message(STATUS "${TARGET}: built and ran OK")
