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
hn_add_test_exe(test_dlr_matsubara MODULE dlr SOURCES test_dlr_matsubara.cpp)
hn_add_test_exe(test_bosonic MODULE bosonic SOURCES test_bosonic.cpp)
hn_add_test_exe(test_free_extrap MODULE free_extrap SOURCES test_free_extrap.cpp)
hn_add_test_exe(test_hodlr_storage MODULE storage SOURCES test_hodlr_storage.cpp BUGS 9 26)
hn_add_test_exe(test_timestep MODULE timestep SOURCES test_timestep.cpp BUGS 28)
hn_add_test_exe(test_bootstrap MODULE bootstrap SOURCES test_bootstrap.cpp)

hn_add_test_exe(test_mpi_comm MODULE mpi_comm SOURCES test_mpi_comm.cpp ENV "OMP_NUM_THREADS=4")
hn_add_test_exe(test_read_inputfile MODULE read_inputfile SOURCES test_read_inputfile.cpp)
hn_add_test_exe(test_checks MODULE checks SOURCES test_checks.cpp)
hn_add_test_exe(test_checks_wp6b MODULE checks SOURCES test_checks_wp6b.cpp)
hn_add_test_exe(test_tv_contractions MODULE tv SOURCES test_tv_contractions.cpp)

# NESSi interoperability (WP-N, B8): only with -DINCLUDE_NESSI=ON; labels "unit;nessi" (ctest -L nessi).
if(INCLUDE_NESSI)
  hn_add_test_exe(test_nessi_interop MODULE nessi SOURCES test_nessi_interop.cpp)
endif()

# ------------------------------------------------------------------ helper tool: HDF5 dataset flattener
add_executable(h5flat tools/h5flat.cpp)
target_link_libraries(h5flat PRIVATE ${HDF5_LIBRARIES})

# ------------------------------------------------------------------ link tests (B6 fixed in WP4; B19 fixed in WP6a)
# These targets are excluded from ALL and built by the ctest itself (cmake --build ... --target X, then run X).
# Both are ordinary tests now: B6 (operator=) links since WP4, B19 (read_inputfile.hpp in two translation units)
# since WP6a.
add_executable(test_read_inputfile_2tu EXCLUDE_FROM_ALL link/ri_tu1.cpp link/ri_tu2.cpp)
target_link_libraries(test_read_inputfile_2tu PRIVATE h_nessi ${COMMON_LIBS})
add_executable(test_b6_operator_assign EXCLUDE_FROM_ALL link/b6_assign.cpp)
target_link_libraries(test_b6_operator_assign PRIVATE h_nessi ${COMMON_LIBS})
foreach(pair "B19;test_read_inputfile_2tu;ReadInputfileTwoTULink_B19;FALSE;unit"
             "B6;test_b6_operator_assign;OperatorAssignLinkAndDeepCopy_B6;FALSE;unit")
  list(GET pair 0 bug)
  list(GET pair 1 tgt)
  list(GET pair 2 tname)
  list(GET pair 3 willfail)
  list(GET pair 4 kind)
  add_test(NAME ${tname}
    COMMAND ${CMAKE_COMMAND} -DBUILD_DIR=${CMAKE_BINARY_DIR} -DTARGET=${tgt}
            -DEXE=${CMAKE_CURRENT_BINARY_DIR}/${tgt} -P ${CMAKE_CURRENT_SOURCE_DIR}/link/build_and_run.cmake)
  set_tests_properties(${tname} PROPERTIES WILL_FAIL ${willfail} LABELS "${kind};${bug};link"
                       RESOURCE_LOCK hn_build_tree TIMEOUT 900)
endforeach()

# ------------------------------------------------------------------ example tests
# Tiny inputs (mpi: L=4, Nt=64, 1 MPI rank, singleton launch; SC: nt=64, nlevel=3). Fixed in WP8: B16, B17, B24.
# In a build with -fsanitize=address in CMAKE_CXX_FLAGS (build-asan) the same tests are AddressSanitizer runs and also
# carry the label "asan"; every script fails on any "Sanitizer" line in the program output.
string(FIND "${CMAKE_CXX_FLAGS}" "-fsanitize=address" _hn_asan_pos)
if(_hn_asan_pos EQUAL -1)
  set(_hn_ex_labels "example")
else()
  set(_hn_ex_labels "example;asan")
endif()
set(_hn_ex ${CMAKE_CURRENT_SOURCE_DIR}/examples)
set(_hn_work ${CMAKE_CURRENT_BINARY_DIR}/work)

# B16: restart from a checkpoint reproduces the uninterrupted run bitwise (obs.h5 incl. jxt_iter).
add_test(NAME MpiExampleRestart_B16
  COMMAND bash ${_hn_ex}/mpi_restart.sh $<TARGET_FILE:Main.x> $<TARGET_FILE:h5flat>
          ${_hn_work}/mpi_restart ${CMAKE_CURRENT_SOURCE_DIR}/data/mpi_tiny.inp)
# B17: StepMaxIter > BootMaxIter; jxt_iter gets max(BootMaxIter,StepMaxIter)+1 rows.
add_test(NAME MpiExampleIterRows_B17
  COMMAND bash ${_hn_ex}/mpi_iter_rows.sh $<TARGET_FILE:Main.x> $<TARGET_FILE:h5flat>
          ${_hn_work}/mpi_iter_rows ${CMAKE_CURRENT_SOURCE_DIR}/data/mpi_tiny.inp)
# B24: nt larger than the phi dataset is refused with a message.
add_test(NAME ScExamplePhiLength_B24
  COMMAND bash ${_hn_ex}/sc_phi_length.sh $<TARGET_FILE:SC.x> ${_hn_work}/sc_phi_length
          ${CMAKE_CURRENT_SOURCE_DIR}/data/sc_tiny.inp ${PROJECT_SOURCE_DIR}/SC_example/phi_input.h5)
# Plain runs of both examples (in build-asan: clean AddressSanitizer runs of the shipped code paths).
add_test(NAME ScExampleRun
  COMMAND bash ${_hn_ex}/sc_run.sh $<TARGET_FILE:SC.x> $<TARGET_FILE:h5flat> ${_hn_work}/sc_run
          ${CMAKE_CURRENT_SOURCE_DIR}/data/sc_tiny.inp ${PROJECT_SOURCE_DIR}/SC_example/phi_input.h5)
# Energy balance of SC_example for a contained pulse: E_total(t) - E_total(0) = E^j(t) (conventions of Q10/D-7, WP8).
add_test(NAME ScExampleEnergyBalance
  COMMAND bash ${_hn_ex}/sc_energy_balance.sh $<TARGET_FILE:SC.x> $<TARGET_FILE:h5flat> ${_hn_work}/sc_energy_balance
          ${CMAKE_CURRENT_SOURCE_DIR}/data/sc_pulse.inp ${CMAKE_CURRENT_SOURCE_DIR}/data/sc_pulse.h5)
set_tests_properties(ScExampleEnergyBalance PROPERTIES LABELS "${_hn_ex_labels};Q10")
# Energy balance of mpi_example for a contained smooth pulse (__pulse= smooth): same conventions (WP8b).
add_test(NAME MpiExampleEnergyBalance
  COMMAND bash ${_hn_ex}/mpi_energy_balance.sh $<TARGET_FILE:Main.x> $<TARGET_FILE:h5flat> ${_hn_work}/mpi_energy_balance
          ${CMAKE_CURRENT_SOURCE_DIR}/data/mpi_pulse.inp)
set_tests_properties(MpiExampleEnergyBalance PROPERTIES LABELS "${_hn_ex_labels};Q10")
set_tests_properties(MpiExampleRestart_B16 PROPERTIES LABELS "${_hn_ex_labels};B16")
set_tests_properties(MpiExampleIterRows_B17 PROPERTIES LABELS "${_hn_ex_labels};B17")
set_tests_properties(ScExamplePhiLength_B24 PROPERTIES LABELS "${_hn_ex_labels};B24")
set_tests_properties(ScExampleRun PROPERTIES LABELS "${_hn_ex_labels}")
set_tests_properties(MpiExampleRestart_B16 MpiExampleIterRows_B17 ScExamplePhiLength_B24 ScExampleRun
                     ScExampleEnergyBalance MpiExampleEnergyBalance PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2" TIMEOUT 900)

# ------------------------------------------------------------------ labels for discovered tests (must be last)
get_property(_hn_label_script GLOBAL PROPERTY HN_LABEL_SCRIPT)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/hn_labels.cmake "${_hn_label_script}")
set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES ${CMAKE_CURRENT_BINARY_DIR}/hn_labels.cmake)
