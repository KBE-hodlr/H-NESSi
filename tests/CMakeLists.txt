# H-NESSi regression suite (WP0).  See tests/README.md for labels, known-bug mapping and how to run.
#
# Naming convention inside the gtest sources (it drives the ctest registration below):
#   *.KnownBug_B<id>_*       test of the CORRECT behaviour that currently fails because of BUGS.md entry B<id>.
#                            Registered with WILL_FAIL and LABELS "known_bug;B<id>".  When the bug is fixed the
#                            test shows up as a failure ("unexpected pass"): remove WILL_FAIL (hn_known_bugs list).
#   *.KnownBugDebug_B<id>_*  as above, but the symptom is a failed assert(); WILL_FAIL only in builds with
#                            asserts enabled (no -DNDEBUG), a plain test otherwise.
#   *.Slow_*                 long convergence sweeps; LABELS "slow".
#   everything else          LABELS "unit".
# Every known-bug test body runs in a forked child (hn_test::run_isolated), so crashes and failed asserts
# are reported as ordinary failures and WILL_FAIL works.

include(GoogleTest)

# Are asserts active in this build?  (Release/RelWithDebInfo/MinSizeRel add -DNDEBUG.)
string(TOUPPER "${CMAKE_BUILD_TYPE}" _hn_bt)
set(_hn_flags "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_hn_bt}}")
if(_hn_flags MATCHES "-DNDEBUG")
  set(HN_ASSERTS_ON OFF)
else()
  set(HN_ASSERTS_ON ON)
endif()
message(STATUS "H-NESSi tests: asserts active = ${HN_ASSERTS_ON}")

set(HN_TEST_ENV "OMP_NUM_THREADS=2")

# gtest_discover_tests flattens list-valued PROPERTIES (a LABELS list would be split into bogus properties),
# so labels are attached by a ctest include file (hn_labels.cmake, written at the end of this file) that runs
# after the discovered-test files and loops over the TEST_LIST variables they define.
set_property(GLOBAL PROPERTY HN_LABEL_SCRIPT "")
function(_hn_discover name filter var labels)
  set(_wd ${CMAKE_CURRENT_BINARY_DIR}/work)
  gtest_discover_tests(${name}
    TEST_FILTER "${filter}"
    TEST_LIST ${var}
    WORKING_DIRECTORY ${_wd}
    PROPERTIES ${ARGN}
    DISCOVERY_TIMEOUT 120)
  set_property(GLOBAL APPEND_STRING PROPERTY HN_LABEL_SCRIPT
    "foreach(t IN LISTS ${var})\n  set_tests_properties(\"\${t}\" PROPERTIES LABELS \"${labels}\")\nendforeach()\n")
endfunction()

# hn_add_test_exe(<name> MODULE <label> SOURCES ... [ENV ...] [BUGS id...] [DEBUG_BUGS id...])
function(hn_add_test_exe name)
  cmake_parse_arguments(A "" "MODULE" "SOURCES;ENV;BUGS;DEBUG_BUGS" ${ARGN})
  if(NOT A_ENV)
    set(A_ENV ${HN_TEST_ENV})
  endif()
  add_executable(${name} ${A_SOURCES})
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${name} PRIVATE HighFive GTest::gtest_main h_nessi ${COMMON_LIBS})
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/work)
  # ordinary tests
  _hn_discover(${name} "-*KnownBug*:*Slow_*" HN_${name}_unit "unit;${A_MODULE}"
    ENVIRONMENT "${A_ENV}" TIMEOUT 600)
  # slow sweeps
  _hn_discover(${name} "*Slow_*" HN_${name}_slow "slow;${A_MODULE}"
    ENVIRONMENT "${A_ENV}" TIMEOUT 3600)
  # known bugs: expected to fail
  foreach(b ${A_BUGS})
    _hn_discover(${name} "*KnownBug_B${b}_*" HN_${name}_B${b} "known_bug;B${b};${A_MODULE}"
      WILL_FAIL TRUE ENVIRONMENT "${A_ENV}" TIMEOUT 600)
  endforeach()
  # known bugs whose symptom is a failed assert(): expected to fail only when asserts are active
  foreach(b ${A_DEBUG_BUGS})
    if(HN_ASSERTS_ON)
      _hn_discover(${name} "*KnownBugDebug_B${b}_*" HN_${name}_DB${b} "known_bug;B${b};${A_MODULE}"
        WILL_FAIL TRUE ENVIRONMENT "${A_ENV}" TIMEOUT 600)
    else()
      _hn_discover(${name} "*KnownBugDebug_B${b}_*" HN_${name}_DB${b} "unit;B${b};${A_MODULE}"
        ENVIRONMENT "${A_ENV}" TIMEOUT 600)
    endif()
  endforeach()
endfunction()

# ------------------------------------------------------------------ gtest executables
hn_add_test_exe(test_integration MODULE integration SOURCES test_integration.cpp)
hn_add_test_exe(test_dlr_matsubara MODULE dlr SOURCES test_dlr_matsubara.cpp BUGS 3)
hn_add_test_exe(test_bosonic MODULE bosonic SOURCES test_bosonic.cpp)
hn_add_test_exe(test_free_extrap MODULE free_extrap SOURCES test_free_extrap.cpp)
hn_add_test_exe(test_hodlr_storage MODULE storage SOURCES test_hodlr_storage.cpp BUGS 9 22 23 26 DEBUG_BUGS 22)
hn_add_test_exe(test_timestep MODULE timestep SOURCES test_timestep.cpp BUGS 14 28)
hn_add_test_exe(test_bootstrap MODULE bootstrap SOURCES test_bootstrap.cpp)

hn_add_test_exe(test_mpi_comm MODULE mpi_comm SOURCES test_mpi_comm.cpp ENV "OMP_NUM_THREADS=4" BUGS 20 21)
hn_add_test_exe(test_read_inputfile MODULE read_inputfile SOURCES test_read_inputfile.cpp BUGS 15)

# ------------------------------------------------------------------ helper tool: HDF5 dataset flattener
add_executable(h5flat tools/h5flat.cpp)
target_link_libraries(h5flat PRIVATE ${HDF5_LIBRARIES})

# ------------------------------------------------------------------ link-failure known bugs (B6, B19)
# These targets cannot link today, so they are excluded from ALL and built by the ctest itself
# (cmake --build ... --target X, then run X).  WILL_FAIL: a fix shows up as an unexpected pass.
add_executable(test_read_inputfile_2tu EXCLUDE_FROM_ALL link/ri_tu1.cpp link/ri_tu2.cpp)
target_link_libraries(test_read_inputfile_2tu PRIVATE h_nessi ${COMMON_LIBS})
add_executable(test_b6_operator_assign EXCLUDE_FROM_ALL link/b6_assign.cpp)
target_link_libraries(test_b6_operator_assign PRIVATE h_nessi ${COMMON_LIBS})
foreach(pair "B19;test_read_inputfile_2tu;KnownBug_B19_ReadInputfileTwoTULink"
             "B6;test_b6_operator_assign;KnownBug_B6_OperatorAssignLinkAndDeepCopy")
  list(GET pair 0 bug)
  list(GET pair 1 tgt)
  list(GET pair 2 tname)
  add_test(NAME ${tname}
    COMMAND ${CMAKE_COMMAND} -DBUILD_DIR=${CMAKE_BINARY_DIR} -DTARGET=${tgt}
            -DEXE=${CMAKE_CURRENT_BINARY_DIR}/${tgt} -P ${CMAKE_CURRENT_SOURCE_DIR}/link/build_and_run.cmake)
  set_tests_properties(${tname} PROPERTIES WILL_FAIL TRUE LABELS "known_bug;${bug};link"
                       RESOURCE_LOCK hn_build_tree TIMEOUT 900)
endforeach()

# ------------------------------------------------------------------ example tests
# B16: mpi_example cannot restart from a checkpoint (tiny L=4, Nt=64 input; 1 MPI rank, singleton launch).
add_test(NAME KnownBug_B16_MpiExampleRestart
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/examples/mpi_restart.sh $<TARGET_FILE:Main.x> $<TARGET_FILE:h5flat>
          ${CMAKE_CURRENT_BINARY_DIR}/work/mpi_restart ${CMAKE_CURRENT_SOURCE_DIR}/data/mpi_tiny.inp)
set_tests_properties(KnownBug_B16_MpiExampleRestart PROPERTIES WILL_FAIL TRUE LABELS "known_bug;B16;example"
                     ENVIRONMENT "OMP_NUM_THREADS=2" TIMEOUT 900)

# ------------------------------------------------------------------ labels for discovered tests (must be last)
get_property(_hn_label_script GLOBAL PROPERTY HN_LABEL_SCRIPT)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/hn_labels.cmake "${_hn_label_script}")
set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES ${CMAKE_CURRENT_BINARY_DIR}/hn_labels.cmake)
