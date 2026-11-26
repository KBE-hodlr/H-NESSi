# H-NESSi regression suite (WP0).  See tests/README.md for labels, the bug -> test map and how to run.
#
# Naming convention inside the gtest sources (it drives the ctest registration below):
#   *.Slow_*                 long convergence sweeps; LABELS "slow;<module>".
#   everything else          LABELS "unit;<module>".
# Regression tests of a fixed BUGS.md entry B<id> also carry the label B<id> (ctest -L B<id>): see hn_bug_tests below.

include(GoogleTest)

set(HN_TEST_ENV "OMP_NUM_THREADS=2")

# gtest_discover_tests flattens list-valued PROPERTIES (a LABELS list would be split into bogus properties),
# so labels are attached by a ctest include file (hn_labels.cmake, written at the end of this file) that runs
# after the discovered-test files and loops over the TEST_LIST variables they define.
set_property(GLOBAL PROPERTY HN_LABEL_SCRIPT "")
set_property(GLOBAL PROPERTY HN_BUG_SCRIPT "")
function(_hn_discover name filter var labels)
  set(_wd ${CMAKE_CURRENT_BINARY_DIR}/work)
  gtest_discover_tests(${name}
    TEST_FILTER "${filter}"
    TEST_LIST ${var}
    WORKING_DIRECTORY ${_wd}
    PROPERTIES ${ARGN}
    DISCOVERY_TIMEOUT 120)
  set_property(GLOBAL APPEND_STRING PROPERTY HN_LABEL_SCRIPT "_hn_label(${var} \"${labels}\")\n")
endfunction()

# hn_add_test_exe(<name> MODULE <label> SOURCES ... [ENV ...] [LABELS label...])
# LABELS: extra labels for every test of the executable (e.g. a bug ID that the whole executable covers).
function(hn_add_test_exe name)
  cmake_parse_arguments(A "" "MODULE" "SOURCES;ENV;LABELS" ${ARGN})
  if(NOT A_ENV)
    set(A_ENV ${HN_TEST_ENV})
  endif()
  set(_lab "${A_MODULE}")
  foreach(l ${A_LABELS})
    string(APPEND _lab ";${l}")
  endforeach()
  add_executable(${name} ${A_SOURCES})
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${name} PRIVATE HighFive GTest::gtest_main h_nessi ${COMMON_LIBS})
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/work)
  # ordinary tests
  _hn_discover(${name} "-*Slow_*" HN_${name}_unit "unit;${_lab}"
    ENVIRONMENT "${A_ENV}" TIMEOUT 600)
  # slow sweeps
  _hn_discover(${name} "*Slow_*" HN_${name}_slow "slow;${_lab}"
    ENVIRONMENT "${A_ENV}" TIMEOUT 3600)
endfunction()

# hn_bug_tests(B<id> <pattern>...): every discovered gtest test whose full name matches one of the patterns gets the
# label B<id>. Patterns are gtest names with '*' as wildcard (whole-name match; parameterised tests are named
# <Prefix>/<Suite>.<Test>/<param>). This is the bug -> test map of tests/README.md; keep the two consistent.
# Link and example tests get their bug label directly in set_tests_properties below.
function(hn_bug_tests id)
  set(_re "")
  foreach(p ${ARGN})
    string(REPLACE "." "[.]" p "${p}")
    string(REPLACE "*" ".*" p "${p}")
    if(_re)
      string(APPEND _re "|")
    endif()
    string(APPEND _re "${p}")
  endforeach()
  set_property(GLOBAL APPEND_STRING PROPERTY HN_BUG_SCRIPT
    "list(APPEND _hn_bug_ids ${id})\nset(_hn_bug_re_${id} \"^(${_re})$\")\n")
endfunction()

# ------------------------------------------------------------------ gtest executables
hn_add_test_exe(test_integration MODULE integration SOURCES test_integration.cpp)
hn_add_test_exe(test_dlr_matsubara MODULE dlr SOURCES test_dlr_matsubara.cpp)
hn_add_test_exe(test_bosonic MODULE bosonic SOURCES test_bosonic.cpp)
hn_add_test_exe(test_free_extrap MODULE free_extrap SOURCES test_free_extrap.cpp)
hn_add_test_exe(test_hodlr_storage MODULE storage SOURCES test_hodlr_storage.cpp)
hn_add_test_exe(test_timestep MODULE timestep SOURCES test_timestep.cpp)
hn_add_test_exe(test_bootstrap MODULE bootstrap SOURCES test_bootstrap.cpp)

hn_add_test_exe(test_mpi_comm MODULE mpi_comm SOURCES test_mpi_comm.cpp ENV "OMP_NUM_THREADS=4")
hn_add_test_exe(test_read_inputfile MODULE read_inputfile SOURCES test_read_inputfile.cpp)
hn_add_test_exe(test_checks MODULE checks SOURCES test_checks.cpp)
hn_add_test_exe(test_checks_wp6b MODULE checks SOURCES test_checks_wp6b.cpp)
hn_add_test_exe(test_tv_contractions MODULE tv SOURCES test_tv_contractions.cpp)

# NESSi interoperability (WP-N, B8): only with -DINCLUDE_NESSI=ON; labels "unit;nessi;B8" (ctest -L nessi).
if(INCLUDE_NESSI)
  hn_add_test_exe(test_nessi_interop MODULE nessi SOURCES test_nessi_interop.cpp LABELS B8)
endif()

# ------------------------------------------------------------------ bug-ID labels of the gtest tests (tests/README.md)
hn_bug_tests(B1 DlrBoson.TvItConvVsQuadrature ResolventBoson.TvItConv EndToEndBoson.MatStartTimestepVsExact
                GMConvTensor.DlrXiOrRankMismatchThrows HodlrCheckpoint.FormatVersion*)
hn_bug_tests(B2 DlrBoson.GammaIntegralMixedTerm ResolventBoson.GammaIntegralEquilibrium ResolventBoson.GammaIntegralVsMat
                GammaIntegralBoson.VsQuadrature GammaIntegralBoson.EquilibriumVsMat
                GammaIntegralBoson.EquilibriumVsMatSingleOrbital)
hn_bug_tests(B3 DlrArray.GetMatTauArray DlrArray.GetTvTauArray *DlrArrayP.MatTauArrayEqualsPointwise/*
                *DlrArrayP.TvTauArrayEqualsPointwise/*)
hn_bug_tests(B4 DysonMat.FixHamSecondCall *DysonMatFixHam.ReusesSavedG0/*)
hn_bug_tests(B6 HodlrAssign.DeepCopyIndependent HodlrAssign.CopyContinuesDysonIdentically)
hn_bug_tests(B9 HodlrCheckpoint.TimingZeroWhenMissing_B9)
hn_bug_tests(B10 ResolventBoson.GreenFromHdm GreenFromHdm.BosonExact)
hn_bug_tests(B12 Bootstrap.NttiHorizSlotZero Bootstrap.NttiDiagSlotZero Bootstrap.TwoLegDiagSlotZero
                 BootstrapRetNtti.OrderSlotZero Bootstrap.Slow_*SlotZeroSweep)
hn_bug_tests(B13 Bootstrap.NttiImpTp0 BootstrapRetNtti.OrderImpTp0TwoOrbitals BootstrapRetNtti.OrderImpTp0TwoOrbitalsSlotZero
                 Bootstrap.Slow_NttiImpTp0SlotZeroSweep)
hn_bug_tests(B14 LesNobc.AccuracyAndBoundary_B14 LesNobc.MatchesDefaultTimestep_B14)
hn_bug_tests(B15 ReadInputfile.EmptyBoolThrows_B15 ReadInputfile.ErrorsAreRuntimeErrorsWithContext)
hn_bug_tests(B20 MpiCommChecks.TooFewComponentsHighLevel MpiCommChecks.TooManyLambdas
                 MpiCommChecksDeathTest.NospawnCheckInsideRegionAborts)
hn_bug_tests(B21 MpiCommNospawn.TeamSmallerThanMaxThreads MpiCommNospawn.CalledOutsideParallel
                 MpiCommNospawn.LambdaTeamSmallerThanMaxThreads MpiCommNospawn.LambdaCalledOutsideParallel)
hn_bug_tests(B22 HodlrCopy.GetTimestepCurrWindow HodlrCopy.GetTimestepCurrSourceUnchangedAndContinues
                 HodlrCopy.SetTimestepCurrSourceUnchanged HodlrCopy.GetTimestepCurrMatsubara HodlrCopyChecks.WindowAndGeometryThrow)
hn_bug_tests(B23 HodlrGeometry.NlvlZeroRejected_B23 CtorChecks.HermMatrixHodlrGeometry)
hn_bug_tests(B25 Bootstrap2leg.FunctionOverloadForwardsImpTp0)
hn_bug_tests(B26 HodlrMemory.MemoryReport_B26)
hn_bug_tests(B28 DysonTiming.Column5Accumulated_B28)
hn_bug_tests(B31 BootstrapRetNtti.ImpTp0ErrorEstimateTwoOrbitals)
hn_bug_tests(B32 DysonMemory.CountsAllBuffers)
hn_bug_tests(B33 HodlrCheckpoint.FreshGMConvTensorZero_B33)
hn_bug_tests(B34 Bootstrap.*Diag* BootstrapRhoDiagonal.ClosedSystem* BootstrapRhoDiagonal.Idempotent*
                 BootstrapRhoDiagonal.NonContractingThrows)
hn_bug_tests(B37 ExtrapolateState.BubbleWriteIntoRowTBeforeExtrapolate)

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
# WP9: the same with two chained mid-run stops (__stop_tstp= 20, then 40; Nt = 64); also checks timings_file.h5 rows.
add_test(NAME MpiExampleRestartMidRun
  COMMAND bash ${_hn_ex}/mpi_restart.sh $<TARGET_FILE:Main.x> $<TARGET_FILE:h5flat>
          ${_hn_work}/mpi_restart_midrun ${CMAKE_CURRENT_SOURCE_DIR}/data/mpi_tiny.inp 20 40)
# B35: restart on ranks without k-points (L = 2: 4 k-points, 6 ranks). Needs a multi-rank launcher, so it is only
# registered when HN_EXAMPLE_MPI_LAUNCH is set (e.g. -DHN_EXAMPLE_MPI_LAUNCH="srun -n 6 -c 2" inside an allocation).
set(HN_EXAMPLE_MPI_LAUNCH "" CACHE STRING "launcher prefix for multi-rank example tests (empty: not registered)")
if(HN_EXAMPLE_MPI_LAUNCH)
  add_test(NAME MpiExampleRestartManyRanks_B35
    COMMAND bash ${_hn_ex}/mpi_restart.sh $<TARGET_FILE:Main.x> $<TARGET_FILE:h5flat>
            ${_hn_work}/mpi_restart_b35 ${CMAKE_CURRENT_SOURCE_DIR}/data/mpi_L2.inp 10 20)
  set_tests_properties(MpiExampleRestartManyRanks_B35 PROPERTIES LABELS "${_hn_ex_labels};B35;mpi"
                       ENVIRONMENT "OMP_NUM_THREADS=2;MPI_LAUNCH=${HN_EXAMPLE_MPI_LAUNCH}" TIMEOUT 900)
endif()
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
# "slow": Debug ~180 s (Release ~5 s); excluded by the recommended Debug run `ctest -LE slow`.
set_tests_properties(ScExampleEnergyBalance PROPERTIES LABELS "${_hn_ex_labels};Q10;slow")
# Energy balance of mpi_example for a contained smooth pulse (__pulse= smooth): same conventions (WP8b).
add_test(NAME MpiExampleEnergyBalance
  COMMAND bash ${_hn_ex}/mpi_energy_balance.sh $<TARGET_FILE:Main.x> $<TARGET_FILE:h5flat> ${_hn_work}/mpi_energy_balance
          ${CMAKE_CURRENT_SOURCE_DIR}/data/mpi_pulse.inp)
# "slow": Debug ~105 s (Release ~2 s; StepMaxErr/BootMaxErr 1e-20 since WP9, was 1e-24: 537 instead of 733
# corrector iterations, balance residual unchanged at 3.5e-7).
set_tests_properties(MpiExampleEnergyBalance PROPERTIES LABELS "${_hn_ex_labels};Q10;slow")
set_tests_properties(MpiExampleRestart_B16 PROPERTIES LABELS "${_hn_ex_labels};B16")
set_tests_properties(MpiExampleRestartMidRun PROPERTIES LABELS "${_hn_ex_labels};B16")
set_tests_properties(MpiExampleIterRows_B17 PROPERTIES LABELS "${_hn_ex_labels};B17")
set_tests_properties(ScExamplePhiLength_B24 PROPERTIES LABELS "${_hn_ex_labels};B24")
set_tests_properties(ScExampleRun PROPERTIES LABELS "${_hn_ex_labels}")
set_tests_properties(MpiExampleRestart_B16 MpiExampleRestartMidRun MpiExampleIterRows_B17 ScExamplePhiLength_B24 ScExampleRun
                     ScExampleEnergyBalance MpiExampleEnergyBalance PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2" TIMEOUT 900)

# ------------------------------------------------------------------ labels for discovered tests (must be last)
get_property(_hn_bug_script GLOBAL PROPERTY HN_BUG_SCRIPT)
get_property(_hn_label_script GLOBAL PROPERTY HN_LABEL_SCRIPT)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/hn_labels.cmake "# generated by tests/CMakeLists.txt\nset(_hn_bug_ids \"\")\n"
     "${_hn_bug_script}")
file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/hn_labels.cmake [=[
function(_hn_label var labels)
  foreach(t IN LISTS ${var})
    set(_l "${labels}")
    foreach(b IN LISTS _hn_bug_ids)
      if(t MATCHES "${_hn_bug_re_${b}}")
        list(APPEND _l ${b})
      endif()
    endforeach()
    set_tests_properties("${t}" PROPERTIES LABELS "${_l}")
  endforeach()
endfunction()
]=] "${_hn_label_script}")
set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES ${CMAKE_CURRENT_BINARY_DIR}/hn_labels.cmake)
