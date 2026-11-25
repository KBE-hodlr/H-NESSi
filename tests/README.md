# H-NESSi regression suite

Built when the project is configured with `-DBUILD_TEST=1` (see `../build.sh` for Release and `../build-debug.sh`
for Debug). googletest is fetched by CPM. Every test checks the **correct** physics: exact references, dense
re-implementations of the documented discrete equations, or convergence orders. Where the current code is known to
be wrong (`agent_docs/BUGS.md`), the test is registered as an expected failure.

## How to run

```bash
# Release (asserts off)
cd H-NESSi/build && make -j16 && cd tests && ctest -j16
# Debug (-O0 -g, asserts on)
mkdir -p H-NESSi/build-debug && cd H-NESSi/build-debug && ../build-debug.sh && make -j16 && cd tests && ctest -j16

ctest -LE slow             # everything except the long convergence sweeps (recommended for Debug)
ctest -L slow              # only the sweeps
ctest -L known_bug         # only the expected failures
ctest -L B12               # everything that exercises one bug
ctest -L unit -L timestep  # one module: integration dlr bosonic free_extrap storage timestep bootstrap
                           #             mpi_comm read_inputfile link example
ctest -R KnownBug_B12 -V   # show why an expected failure fails
```

Run the gtest executables through ctest. Each test then gets its own process. If you run a binary directly, use
`--gtest_filter`. Known-bug bodies `fork()`, and libgomp is not fork-safe once an earlier test has started its
thread pool.

Timings (Perlmutter login node, `-j16`, all 217 tests including `slow`): Release about 6 s wall, Debug (-O0)
about 90 s wall. The `slow` sweeps (21 tests) dominate in Debug at roughly 70 s each.

## Labels and naming convention

| gtest name pattern | ctest labels | properties |
|---|---|---|
| `*.KnownBug_B<id>_*` | `known_bug;B<id>;<module>` | `WILL_FAIL` |
| `*.KnownBugDebug_B<id>_*` | `known_bug;B<id>;<module>` if asserts are on, else `unit;B<id>;<module>` | `WILL_FAIL` only with asserts |
| `*.Slow_*` | `slow;<module>` | timeout 3600 s |
| everything else | `unit;<module>` | |

* **Expected failures.** Every known-bug test body runs in a forked child (`common/test_util.hpp`,
  `hn_test::run_isolated`). A segfault or a failed `assert` therefore becomes an ordinary test failure. This matters
  because `WILL_FAIL` does not invert crashes. It also means a crashing test cannot take down other tests.
* **When a bug is fixed**, its tests turn into `Failed` (an "unexpected pass"). To flip them, remove the bug id from
  the `BUGS`/`DEBUG_BUGS` list of the executable in `CMakeLists.txt`, and rename `KnownBug_B<id>_` to a normal name.
  For B6, B16 and B19, delete `WILL_FAIL`.
* **Labels.** `gtest_discover_tests` flattens list-valued properties, so labels are attached by the generated ctest
  include file `hn_labels.cmake`.
* **Link-failure tests** (B6, B19). The targets `test_b6_operator_assign` and `test_read_inputfile_2tu` are
  `EXCLUDE_FROM_ALL`, so the default build does not break. The ctest builds the target with `cmake --build`
  (`link/build_and_run.cmake`) and then runs it. The test holds `RESOURCE_LOCK hn_build_tree`.
* **`h5flat`** (`tools/h5flat.cpp`) flattens HDF5 files to text. The example test and
  `agent_scratch/baseline/compare.py` use it.

## Executables

| executable | ported from | contents |
|---|---|---|
| `test_integration` | `agent_scratch/02/t.cpp` | polynomial exactness of every weight table (k = 0..6); `bd_weights` with k+2 terms; Gregory symmetry; convergence orders |
| `test_dlr_matsubara` | `03/test_dlr.cpp`, `51/t_mat.cpp` | DLR maps, `get_mat_tau`, `density_matrix`, reversed/tv accessors, `tv_it_conv`, `les_it_int`, `gamma_integral(_mat)` against quadrature; `dyson_mat` against the exact hybridised G^M (μ shift, mixing, return value); `dyson_mat` leaves Σ and both conv tensors untouched, `fixHam=true` reuses the saved G0; `dyson::memory_usage_bytes` counts every heap buffer with its real size (B32); B1 B2 B3 B4 (B4 fixed in WP2) |
| `test_bosonic` | `51/t_resolvent.cpp` | exact resolvent identity G_ε∗G_η = (G_ε−G_η)/(ε−η) for both ξ; B1 B2 B10 and the Q2 relation |
| `test_free_extrap` | `23/ta,tb,tc.cpp` | `green_from_H` exact (real and complex H, both ξ); `green_from_H_dm`; `extrapolate` order k+1 (and the `_2leg` variant, no-op on a second call); `gamma_integral` against a dense quadrature, order k+1; B2 B10 |
| `test_hodlr_storage` | old `tests/linalg.cpp`, `52/s9.cpp`, `52/s1.cpp` | `mgs`, incremental TSVD; fill/archive/read-back round trips for 8 geometries; zero-first-row blocks (S9); checkpoint round trip; B9 B22 B23 B26 |
| `test_timestep` | `20/rettv.cpp`, `21/les.cpp` | R/⌉ step against dense discrete equations (5 geometries), against svdtol, order k+1 (k = 1..5); < step against the dense system, the 8-region decomposition of `les_lesadv_int_0_tstp` and raw storage, both rho modes, 2-leg, closed-system N conservation, order k+1; B14 B28 |
| `test_bootstrap` | `22/start_test.cpp`, `51/t_start*.cpp` | TTI and non-TTI bootstrap (horiz/diag, nobc, 2leg) against a dense reference that uses the **correct** continuation and the right-acting imp_tp0 system, plus order k+1 against the exact impurity+bath model; the imp_tp0 error estimate; B12 B13 B25 B31 (all fixed in WP1) |
| `test_mpi_comm` | `31/commtest.cpp` | `mpi_comm` routing with P simulated ranks: MPI symbols interposed, std::thread per rank, 1–5 ranks, spawn/nospawn, vector/reference_wrapper, high-level and lambda APIs; B20 B21 (`OMP_NUM_THREADS=4`) |
| `test_read_inputfile` | new | `find_param` for int/double/bool/cplx/vector; missing values throw; B15 |
| `test_read_inputfile_2tu` (link) | `52/ri1+ri2.cpp` | B19 |
| `test_b6_operator_assign` (link) | new | B6 |
| `KnownBug_B16_MpiExampleRestart` (example) | `agent_docs/32` §6.3 | runs `mpi_example` (L=4, Nt=64) uninterrupted, then stop (`__time_limit=0`) followed by restart; `obs.h5` must agree to 1e-10 |

## BUG ID → test map

Status is for the code as of WP0 (CLAUDE_dev = upstream 5c9706b), updated for the bugs fixed since (WP1: B12, B13, B25, B31; WP2: B4).
XF means expected failure.

| BUGS id | test(s) | Release | Debug | symptom seen by the test |
|---|---|---|---|---|
| B1 | `DlrBoson.KnownBug_B1_TvItConvVsQuadrature`, `ResolventBoson.KnownBug_B1_TvItConv` | XF | XF | bosonic `tv_it_conv` = −reference (rel. err 2.0) |
| B2 | `DlrBoson.KnownBug_B2_GammaIntegralMixedTerm`, `ResolventBoson.KnownBug_B2_GammaIntegralEquilibrium`, `ResolventBoson.KnownBug_B2_GammaIntegralVsMat`, `GammaIntegralBoson.KnownBug_B2_VsQuadrature`, `GammaIntegralBoson.KnownBug_B2_EquilibriumVsMat` | XF | XF | wrong sign of the bosonic mixed term (error 2.87; relation `gamma_integral = ξ·gamma_integral_mat` violated) |
| B3 | `DlrArray.KnownBug_B3_GetMatTauArray`, `DlrArray.KnownBug_B3_GetTvTauArray` | XF | XF | child SIGSEGV |
| B4 | `DysonMat.FixHamSecondCall`, `DysonMatFixHam.ReusesSavedG0/*` | pass | pass | **fixed in WP2** (was: rel. difference 1.10) |
| B6 | `KnownBug_B6_OperatorAssignLinkAndDeepCopy` | XF | XF | undefined reference to `operator=` |
| B8 | — | | | not testable without a NESSi build (see below) |
| B9 | `HodlrCheckpoint.KnownBug_B9_TimingZeroWhenMissing` | XF | XF | uninitialised timing (malloc perturbation) |
| B10 | `ResolventBoson.KnownBug_B10_GreenFromHdm`, `GreenFromHdm.KnownBug_B10_BosonExact` | XF | XF | bosonic G^< error 0.13 / 0.11 (decision D-1: `rho` is the density matrix) |
| B12 | `Bootstrap.NttiHorizSlotZero`, `…NttiDiagSlotZero`, `…TwoLegDiagSlotZero`, `BootstrapRetNtti.OrderSlotZero`, `Bootstrap.Slow_*SlotZeroSweep` | pass | pass | **fixed in WP1** (was: library ≠ correct reference (≈1e-3); order 2 instead of k+1) |
| B13 | `Bootstrap.NttiImpTp0`, `BootstrapRetNtti.OrderImpTp0TwoOrbitals`, `…OrderImpTp0TwoOrbitalsSlotZero`, `Bootstrap.Slow_NttiImpTp0SlotZeroSweep` | pass | pass | **fixed in WP1** (was: column-0 order 3 for k ≥ 3, N_o = 2) |
| B14 | `LesNobc.KnownBug_B14_AccuracyAndBoundary` | XF | XF | `_nobc` G^< error 0.33; boundary mismatch 5.7e-3 |
| B15 | `ReadInputfile.KnownBug_B15_EmptyBoolThrows` | XF | XF | child SIGSEGV instead of an exception |
| B16 | `KnownBug_B16_MpiExampleRestart` | XF | XF | restart run exits with 139 (segfault) |
| B17 | — | | | not testable (see below) |
| B19 | `KnownBug_B19_ReadInputfileTwoTULink` | XF | XF | multiple definition of `uncomment_line` … |
| B20 | `MpiCommChecks.KnownBug_B20_TooFewComponentsHighLevel`, `MpiCommChecks.KnownBug_B20_TooManyLambdas` | XF | XF | no exception; heap corruption (child SIGABRT) |
| B21 | `MpiCommNospawn.KnownBug_B21_TeamSmallerThanMaxThreads`, `MpiCommNospawn.KnownBug_B21_CalledOutsideParallel` | XF | XF | stale/unpacked buffer entries |
| B22 | `HodlrCopy.KnownBug_B22_GetTimestepCurrWindow` | XF | XF | source tstpmk 8 → 0; destination window wrong |
| B22 (assert part) | `HodlrCopy.KnownBugDebug_B22_GetTimestepCurrMatsubara` | pass | XF | Debug: `assert(tstp>=tstpmk_)` fires before the `tstp==-1` branch |
| B23 | `HodlrGeometry.KnownBug_B23_NlvlZeroNoCrash` | XF | XF | child SIGSEGV in `init_shape` |
| B24 | — | | | not testable (see below) |
| B25 | `Bootstrap2leg.FunctionOverloadForwardsImpTp0` | pass | pass | **fixed in WP1** (was: `function&` and pointer overloads differ when `imp_tp0=true`) |
| B26 | `HodlrMemory.KnownBug_B26_MemoryReport` | XF | XF | reported memory ≠ true block storage |
| B27 | — | | | not testable (see below) |
| B28 | `DysonTiming.KnownBug_B28_Column5Accumulated` | XF | XF | dyson timing column 5 is always 0 |
| B29, B30 | — | | | notebook plot / source comments only (B30 fixed in WP2) |
| B31 | `BootstrapRetNtti.ImpTp0ErrorEstimateTwoOrbitals` (added in WP1) | pass | pass | **fixed in WP1** (was: Debug Eigen assert at `dyson_start.cpp:197`; wrong error in Release) |
| B32 | `DysonMemory.CountsAllBuffers` (added in WP2) | pass | pass | **fixed in WP2** (was: `DNTauTmp*_`, `iden_`, `bound_` not counted; `M_` counted as k²·es) |

The B14 test also checks the correct boundary sign through its dense reference: the conjugate KBE with
−`les_it_int`.

### Not made into tests

* **B8** (`INCLUDE_NESSI` block does not compile; the CMake option never defines the macro). Building it needs
  NESSi (`cntr`) headers and a library, and none is installed yet (decision D-5, WP-N).
* **B17** (`mpi_example` `jxt_iter_*` indexed past BootMaxIter+1). This is a silent heap overrun with no
  observable symptom unless the example is built with ASan or `_GLIBCXX_ASSERTIONS`, and the examples must not be
  modified or rebuilt differently in WP0.
* **B24** (`SC_example` reads `phi_input.h5` past 32768 entries). This is a silent out-of-bounds read in Release.
  Testing it would need nt > 32768, which is a very long run, and the read does not reliably crash.
* **B27** (`herm_matrix_hodlr::timing(·,5)` measures only the last segment). The value is a wall-clock duration,
  and there is no deterministic reference to compare it against.
* **B29** (notebook) and **B30** (comments) have no runtime behaviour.

## Release vs Debug

Both builds pass 100% (217 tests after WP2). Release has 25 expected failures and Debug has 26. The difference is:

* `HodlrCopy.KnownBugDebug_B22_GetTimestepCurrMatsubara` passes in Release. In Debug it is an expected failure:
  `assert(tstp>=tstpmk_)` at `herm_matrix_hodlr.cpp:339` fires before the `tstp==-1` branch (B22).
* Until WP1, the B13 and B25 tests also aborted in Debug on an Eigen assertion at `dyson_start.cpp:197` (BUGS B31,
  fixed in WP1 together with B13 and B25).
* All other known-bug tests fail the same way in both builds, and no ordinary test asserts in Debug.
