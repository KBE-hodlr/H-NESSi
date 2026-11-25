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
# RelWithAsserts (since WP6a: -O2 -g, tier-1 asserts on, EIGEN_NO_DEBUG): as build.sh with -DCMAKE_BUILD_TYPE=RelWithAsserts

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

Timings (Perlmutter login node, `-j16`, all 231 tests including `slow`): Release about 6 s wall, Debug (-O0)
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
  For B19, delete `WILL_FAIL` (B6 was flipped this way in WP4: `WILL_FAIL FALSE`, label `unit`; B16 in WP8).
* **Labels.** `gtest_discover_tests` flattens list-valued properties, so labels are attached by the generated ctest
  include file `hn_labels.cmake`.
* **Link tests** (B6, fixed in WP4, and B19, fixed in WP6a). The targets `test_b6_operator_assign` and `test_read_inputfile_2tu` are
  `EXCLUDE_FROM_ALL`, so the default build does not break. The ctest builds the target with `cmake --build`
  (`link/build_and_run.cmake`) and then runs it. The test holds `RESOURCE_LOCK hn_build_tree`.
* **`h5flat`** (`tools/h5flat.cpp`) flattens HDF5 files to text. The example test and
  `agent_scratch/baseline/compare.py` use it.

## Executables

| executable | ported from | contents |
|---|---|---|
| `test_integration` | `agent_scratch/02/t.cpp` | polynomial exactness of every weight table (k = 0..6); `bd_weights` with k+2 terms; Gregory symmetry; convergence orders |
| `test_dlr_matsubara` | `03/test_dlr.cpp`, `51/t_mat.cpp` | DLR maps, `get_mat_tau`, `density_matrix`, reversed/tv accessors, `tv_it_conv`, `les_it_int`, `gamma_integral(_mat)` against quadrature; `dyson_mat` against the exact hybridised G^M (μ shift, mixing, return value); `dyson_mat` leaves Σ and both conv tensors untouched, `fixHam=true` reuses the saved G0; `dyson::memory_usage_bytes` counts every heap buffer with its real size (B32); the `initGMConvTensor` dlr ξ/r assert (death test, Debug only, skipped in Release); B1 B2 B3 B4 (B1, B2 fixed in WP3; B4 fixed in WP2; B3 fixed in WP-N: `get_*_tau_array` = pointwise accessors) |
| `test_bosonic` | `51/t_resolvent.cpp`, new (WP3) | exact resolvent identity G_ε∗G_η = (G_ε−G_η)/(ε−η) for both ξ; B1 B2 B10 (fixed in WP3) and the Q2 relation; end-to-end system+bath run (dyson_mat → initGMConvTensor → dyson_start → dyson_timestep, drive after the bootstrap) vs the exact propagator, both ξ, order k+1 for k = 2, 4 in R, <, ⌉ |
| `test_free_extrap` | `23/ta,tb,tc.cpp` | `green_from_H` exact (real and complex H, both ξ); `green_from_H_dm`; `extrapolate` order k+1 (and the `_2leg` variant, no-op on a second call); `gamma_integral` against a dense quadrature, order k+1; `gamma_integral(0)` = ξ·`gamma_integral_mat` (N_o = 1, 2); `green_from_H_dm` G^⌉(t,0) = G^<(t,0); B2 B10 (fixed in WP3) |
| `test_hodlr_storage` | old `tests/linalg.cpp`, `52/s9.cpp`, `52/s1.cpp` | `mgs`, incremental TSVD; fill/archive/read-back round trips for 8 geometries; zero-first-row blocks (S9); checkpoint round trip; checkpoint `format_version` (v3 without `tv_trans`, WP6c; a v2 file with `tv_trans` still loads and its `tv_trans` is ignored; a newer version is refused; new bosonic file silent, missing version warns on stderr, fermionic silent, not in `write_to_hdf5` output); copy semantics (WP4): `get/set_timestep_curr` leave the source bitwise unchanged (full checkpoint dump compared), give the destination the source window, tstp = −1 with different windows, window/geometry asserts (death test, Debug only); `operator=`/copy ctor deep copy, and a copy taken mid-run (dyson_mat → dyson_start → dyson_timestep) continues bitwise like the original; B9 B22 (fixed in WP4) B23 B26 B6 (fixed in WP4) |
| `test_timestep` | `20/rettv.cpp`, `21/les.cpp` | R/⌉ step against dense discrete equations (5 geometries), against svdtol, order k+1 (k = 1..5); < step against the dense system, the 8-region decomposition of `les_lesadv_int_0_tstp` and raw storage, both rho modes, 2-leg, closed-system N conservation, order k+1; B14 (fixed in WP7: `_nobc` vs default `dyson_timestep`, both ξ) B28 |
| `test_bootstrap` | `22/start_test.cpp`, `51/t_start*.cpp` | TTI and non-TTI bootstrap (horiz/diag, nobc, 2leg) against a dense reference that uses the **correct** continuation and the right-acting imp_tp0 system, plus order k+1 against the exact impurity+bath model; the imp_tp0 error estimate; B12 B13 B25 B31 (all fixed in WP1); B34 (fixed in WP7): RHO_DIAGONAL reference with G^<(m,m) known in the row solves, iterated to convergence, fixed-point check of the diagonal; `BootstrapRhoDiagonal.*` closed system (Σ = 0, both ξ): diagonal-equation residual, anti-Hermiticity, N conservation over t ≤ k, order k+1 |
| `test_mpi_comm` | `31/commtest.cpp` | `mpi_comm` routing with P simulated ranks: MPI symbols interposed (incl. `MPI_Query_thread`), std::thread per rank, 1–5 ranks, spawn/nospawn, vector/reference_wrapper, high-level and lambda APIs; `_nospawn` in a `num_threads(2)` team and outside a region; trailing barrier of the set (H24); `my_first_t`/`my_Nt` (H30); thread-level warning (H28); B20 B21 (fixed in WP5; `OMP_NUM_THREADS=4`). WP6b: shim also interposes `MPI_Initialized`, `MPI_Finalized`, `MPI_Allreduce` (Debug collective check), `MPI_Abort`; `MpiCommChecksWP6b*` (constructor MP-1/2, thread level MP-3, ti/objects/dlr MP-6/8/9, get→set pairing ST-9, Im Σ^M warning MP-13, terminate handler → MPI_Abort, tier-1 maps); the in-region count check now aborts (death test, WP6a side finding 2) |
| `test_read_inputfile` | new | `find_param` for int/double/bool/cplx/vector; missing values throw; every error is a `std::runtime_error` with flag, token and file; B15 (fixed in WP6a) |
| `test_nessi_interop` | new (WP-N) | **only with `-DINCLUDE_NESSI=ON`**, label `nessi` (`ctest -L nessi`). The five `distance_norm2_*` overloads for `cntr::herm_matrix<double>` (doc 05): exact `green_from_H` in both libraries (roundoff for R/<, DLR error for tv/M; ξ = ±1, N_o = 1, 2), shared sign conventions, complex h (D1: mat distance = Im G^M), single-entry perturbations seen only by the matching function, checked preconditions, and full driven dyson runs (Σ = 0 and a fixed bath Σ, both ξ) in both libraries with order-k+1 convergence of the distance; B8 |
| `test_checks` | new (WP6a) | the check infrastructure (`checks.hpp`): exception types and message format, tier-1 behaviour, OpenMP policy (death tests: abort inside a user region; capture/rethrow in library regions); constructor validation (herm_matrix_hodlr HM-C*, Integrator KMAX, dlr_info DL-*, dyson DY-C*); dyson entry checks (geometry bundle, step synchronisation DY-T1, bootstrap window DY-S1, `green_from_H*` tmax/window H11, dlr pairing, extrapolate); finiteness (L9, DY-P1); ST-2 (stale convolution tensor, checkpoints incl. pre-v2 bosonic); ST-3 (saved G0); D-11 Hermiticity of h (throw before modifying, in-place symmetrisation + diagnostic, opt-out, warn once, complex h(−1), RHO_HORIZONTAL mean-field run); concurrent `dlr_info` construction (H3) |
| `test_checks_wp6b` | new (WP6b) | the remaining checks of doc 61: herm_matrix_hodlr (`set_tstpmk`, `time2block` failure path, DLR evaluators τ/row/output sizes, `density_matrix`, `update_blocks` HM-U1/U2, checkpoint dataset sizes HM-C11, output writers HM-W2–W5, distances HM-D*; tier-1 accessor death tests); D-11 predictor/corrector Hermiticity (`HermiticityCorrector.*`); component order ST-4; single-thread ownership ST-10 (dyson, dlr_info); `eval_point` τ range; dyson checkpoint parameters ST-11; bosonic spectrum, complex h with G^M, density-matrix validation (DY-M10/F7/F9/F10); bootstrap TTI tolerance, G^<(0,0), singular systems (DY-S4/S7/S10); `gamma_integral`, `tv_ret_int`, `les_lesadv_int_0_tstp` (DY-G*, DY-I5/I6); ST-6 (`extrapolate` after a write into row T); Integrator weights IN-2 and tier-1 accessors; `function` FN-*; `Bubble1/2` BU-*; `read_inputfile` RI-3/4/6 |
| `test_tv_contractions` | new (WP6c) | label `tv`. ISSUES H27 (`tv_trans` removed): the OLD kernels (matrix products on an explicitly built orbital-transposed copy of tv, test-only reference) against the new Eigen Tensor readers `les_it_int` (both overloads, incl. swapped arguments), `tv_it_conv` (both overloads), `get_vt` (all overloads), `Bubble1`, for N_o ∈ {1,2,3,4,8,16}, both ξ, r ∈ {19, 35, 42, 52, ~64}, T up to 3000, non-symmetric complex data (≤ 1e-14 normwise relative; prints the max); writes through `map_tv`/`tvptr`/`set_tstp_zero`/Bubble1 output are seen by every reader (the former stale-cache hazard); `get_memory_usage` counts tv once |
| `test_read_inputfile_2tu` (link) | `52/ri1+ri2.cpp` | B19 |
| `test_b6_operator_assign` (link) | new | B6 (fixed in WP4; ctest `OperatorAssignLinkAndDeepCopy_B6`, ordinary test) |
| `MpiExampleRestart_B16` (example; formerly `KnownBug_B16_MpiExampleRestart`) | `agent_docs/32` §6.3 | `examples/mpi_restart.sh`: runs `mpi_example` (L=4, Nt=64) uninterrupted, then stop (`__time_limit=0`) followed by restart; `obs.h5` (incl. `jxt_iter`) must agree **bitwise** |
| `MpiExampleIterRows_B17` (example, WP8) | new | `examples/mpi_iter_rows.sh`: BootMaxIter=2, StepMaxIter=25, StepMaxErr=0, Nt=12; exit 0, no sanitizer report, `jxt_iter` is 26×12 |
| `ScExamplePhiLength_B24` (example, WP8) | new | `examples/sc_phi_length.sh`: `SC.x` with nt=32769 and the shipped `phi_input.h5` (32768 entries) stops before allocating, with exit ≠ 0 and the message, no output file |
| `ScExampleRun` (example, WP8) | new | `examples/sc_run.sh`: `SC.x` on `data/sc_tiny.inp` (nt=64, nlevel=3); exit 0, no sanitizer report |
| `ScExampleEnergyBalance` (example, WP8, label `Q10`) | new | `examples/sc_energy_balance.sh`: `SC.x` with a contained pulse (`data/sc_pulse.h5` from `tools/mkpulse.sh`, h=0.03, nt=300, `data/sc_pulse.inp`); max\|E_total(t)−E_total(0)−E^j(t)\| < 1e-5 (observed 5.3e-7, O(h⁶); wrong conventions give ≥ 1.5e-2) and E^j(end) > 0.1 |

**AddressSanitizer build (WP8).** Configure a build with `-DCMAKE_CXX_FLAGS="-g -fsanitize=address -fno-omit-frame-pointer"`
(same for `CMAKE_C_FLAGS`) and `-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address -DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=address`
(works with the Cray `CC` wrapper / g++). The five example tests then carry the extra label `asan` (`ctest -L asan`) and
fail on any `Sanitizer` line in the program output. The whole suite also runs in that build.

## BUG ID → test map

Status is for the code as of WP0 (CLAUDE_dev = upstream 5c9706b), updated for the bugs fixed since (WP1: B12, B13, B25, B31; WP2: B4; WP3: B1, B2, B10; WP4: B22, B6; WP5: B20, B21; WP6a: B15, B19, B23).
XF means expected failure.

| BUGS id | test(s) | Release | Debug | symptom seen by the test |
|---|---|---|---|---|
| B1 | `DlrBoson.TvItConvVsQuadrature`, `ResolventBoson.TvItConv`, `EndToEndBoson.MatStartTimestepVsExact`, `GMConvTensor.DlrXiOrRankMismatchThrows`, `HodlrCheckpoint.FormatVersion*` | pass | pass | **fixed in WP3** (was: bosonic `tv_it_conv` = −reference, rel. err 2.0). The ξ/r check is tier 2 since WP6a (was a Debug-only death test) |
| B2 | `DlrBoson.GammaIntegralMixedTerm`, `ResolventBoson.GammaIntegralEquilibrium`, `ResolventBoson.GammaIntegralVsMat`, `GammaIntegralBoson.VsQuadrature`, `GammaIntegralBoson.EquilibriumVsMat`, `GammaIntegralBoson.EquilibriumVsMatSingleOrbital` | pass | pass | **fixed in WP3** (was: wrong sign of the bosonic mixed term, error 2.87; relation `gamma_integral = ξ·gamma_integral_mat` violated) |
| B3 | `DlrArray.GetMatTauArray`, `DlrArray.GetTvTauArray` (formerly KnownBug_B3), `DlrArrayP.MatTauArrayEqualsPointwise/*`, `DlrArrayP.TvTauArrayEqualsPointwise/*` | pass | pass | **fixed in WP-N** (was: child SIGSEGV) |
| B4 | `DysonMat.FixHamSecondCall`, `DysonMatFixHam.ReusesSavedG0/*` | pass | pass | **fixed in WP2** (was: rel. difference 1.10) |
| B6 | `OperatorAssignLinkAndDeepCopy_B6` (link), `HodlrAssign.DeepCopyIndependent`, `HodlrAssign.CopyContinuesDysonIdentically` | pass | pass | **fixed in WP4** (was: undefined reference to `operator=`) |
| B8 | `test_nessi_interop` (`NessiEq.*`, `NessiConventions.*`, `NessiEqComplexH.*`, `NessiPerturb.*`, `NessiPreconditions.*`, `NessiDynamics.*`; label `nessi`, built only with `-DINCLUDE_NESSI=ON`) | pass | pass | **fixed in WP-N** (was: block compiled out; did not compile with the macro) |
| B9 | `HodlrCheckpoint.KnownBug_B9_TimingZeroWhenMissing` | XF | XF | uninitialised timing (malloc perturbation) |
| B10 | `ResolventBoson.GreenFromHdm`, `GreenFromHdm.BosonExact` | pass | pass | **fixed in WP3** (was: bosonic G^< error 0.13 / 0.11; decision D-1: `rho` is the density matrix) |
| B12 | `Bootstrap.NttiHorizSlotZero`, `…NttiDiagSlotZero`, `…TwoLegDiagSlotZero`, `BootstrapRetNtti.OrderSlotZero`, `Bootstrap.Slow_*SlotZeroSweep` | pass | pass | **fixed in WP1** (was: library ≠ correct reference (≈1e-3); order 2 instead of k+1) |
| B13 | `Bootstrap.NttiImpTp0`, `BootstrapRetNtti.OrderImpTp0TwoOrbitals`, `…OrderImpTp0TwoOrbitalsSlotZero`, `Bootstrap.Slow_NttiImpTp0SlotZeroSweep` | pass | pass | **fixed in WP1** (was: column-0 order 3 for k ≥ 3, N_o = 2) |
| B14 | `LesNobc.AccuracyAndBoundary_B14` (formerly KnownBug_B14), `LesNobc.MatchesDefaultTimestep_B14` | pass | pass | **fixed in WP7** (was: `_nobc` G^< error 0.33; boundary mismatch 5.7e-3) |
| B34 | `Bootstrap.*Diag*` (ntti, nobc: fixed-point check), `BootstrapRhoDiagonal.ClosedSystem*` (added in WP7) | pass | pass | **fixed in WP7** (was: converged bootstrap diagonal violates its own diagonal equation, residual ≈1e-5 at k = 3) |
| B15 | `ReadInputfile.EmptyBoolThrows_B15` (formerly KnownBug_B15), `ReadInputfile.ErrorsAreRuntimeErrorsWithContext` | pass | pass | **fixed in WP6a** (was: child SIGSEGV instead of an exception) |
| B16 | `MpiExampleRestart_B16` | pass | pass | **fixed in WP8** (was: restart run exits with 139 (segfault)) |
| B17 | `MpiExampleIterRows_B17` (meaningful as a memory check only in the ASan build, label `asan`) | pass | pass | **fixed in WP8** (was: silent heap overrun; ASan heap-buffer-overflow at `Main.cpp`) |
| B19 | `ReadInputfileTwoTULink_B19` (link, formerly KnownBug_B19) | pass | pass | **fixed in WP6a** (was: multiple definition of `uncomment_line` …) |
| B20 | `MpiCommChecks.TooFewComponentsHighLevel`, `MpiCommChecks.TooManyLambdas` (formerly KnownBug_B20), `MpiCommChecks.NospawnEveryThreadThrowsInsideRegion` | pass | pass | **fixed in WP5** (was: no exception; heap corruption, child SIGABRT) |
| B21 | `MpiCommNospawn.TeamSmallerThanMaxThreads`, `MpiCommNospawn.CalledOutsideParallel` (formerly KnownBug_B21), `MpiCommNospawn.LambdaTeamSmallerThanMaxThreads`, `MpiCommNospawn.LambdaCalledOutsideParallel` | pass | pass | **fixed in WP5** (was: stale/unpacked buffer entries) |
| B22 | `HodlrCopy.GetTimestepCurrWindow`, `HodlrCopy.GetTimestepCurrSourceUnchangedAndContinues`, `HodlrCopy.SetTimestepCurrSourceUnchanged`, `HodlrCopyDeathTest.WindowAndGeometryAsserts` | pass | pass | **fixed in WP4** (was: source tstpmk 8 → 0; destination window wrong). The death test is skipped in Release (assert off) |
| B22 (assert part) | `HodlrCopy.GetTimestepCurrMatsubara` | pass | pass | **fixed in WP4** (was: Debug `assert(tstp>=tstpmk_)` fired before the `tstp==-1` branch) |
| B23 | `HodlrGeometry.NlvlZeroRejected_B23` (formerly KnownBug_B23), `CtorChecks.HermMatrixHodlrGeometry` | pass | pass | **fixed in WP6a** (was: child SIGSEGV in `init_shape`; now `h_nessi::precondition_error`) |
| B24 | `ScExamplePhiLength_B24` | pass | pass | **fixed in WP8** (was: silent out-of-bounds read of `phi`; ASan heap-buffer-overflow at `SC.cpp:100`) |
| B25 | `Bootstrap2leg.FunctionOverloadForwardsImpTp0` | pass | pass | **fixed in WP1** (was: `function&` and pointer overloads differ when `imp_tp0=true`) |
| B26 | `HodlrMemory.KnownBug_B26_MemoryReport` | XF | XF | reported memory ≠ true block storage |
| B27 | — | | | not testable (see below) |
| B28 | `DysonTiming.KnownBug_B28_Column5Accumulated` | XF | XF | dyson timing column 5 is always 0 |
| B29, B30 | — | | | notebook plot / source comments only (B30 fixed in WP2, B29 in WP8) |
| B31 | `BootstrapRetNtti.ImpTp0ErrorEstimateTwoOrbitals` (added in WP1) | pass | pass | **fixed in WP1** (was: Debug Eigen assert at `dyson_start.cpp:197`; wrong error in Release) |
| B32 | `DysonMemory.CountsAllBuffers` (added in WP2) | pass | pass | **fixed in WP2** (was: `DNTauTmp*_`, `iden_`, `bound_` not counted; `M_` counted as k²·es) |

The B14 test also checks the correct boundary sign through its dense reference: the conjugate KBE with
−`les_it_int`.

### Not made into tests

* **B17, B24**: not testable in WP0 (silent out-of-bounds accesses); since WP8 tested as above, as memory checks in
  the ASan build.
* **B27** (`herm_matrix_hodlr::timing(·,5)` measures only the last segment). The value is a wall-clock duration,
  and there is no deterministic reference to compare it against.
* **B29** (notebook) and **B30** (comments) have no runtime behaviour.

## Release vs Debug vs RelWithAsserts

After WP-N + WP6a: Release, Debug and RelWithAsserts each pass 100% (277 tests including `slow`; 5 expected failures:
B9 B14 B16 B26 B28); with `-DINCLUDE_NESSI=ON`, 290 tests. The tier-2 checks are identical in all builds; the death tests of tier-1 asserts
(`HodlrCopyDeathTest`, `ChecksMacrosDeathTest.Tier1AssertAbortsOnlyWithoutNDEBUG` in its abort branch) run only
with asserts on. Before WP6a:

Both builds pass 100% (246 tests after WP-N; 259 with `-DINCLUDE_NESSI=ON`, which adds the 13 `nessi` tests). Release has
8 expected failures and skips the Debug-only death tests (`GMConvTensorDeathTest`, `HodlrCopyDeathTest`); Debug has 8. The differences are:

* Until WP4, `HodlrCopy.KnownBugDebug_B22_GetTimestepCurrMatsubara` (now `HodlrCopy.GetTimestepCurrMatsubara`) was an
  expected failure in Debug only: `assert(tstp>=tstpmk_)` fired before the `tstp==-1` branch (B22, fixed in WP4).
* Until WP1, the B13 and B25 tests also aborted in Debug on an Eigen assertion at `dyson_start.cpp:197` (BUGS B31,
  fixed in WP1 together with B13 and B25).
* All other known-bug tests fail the same way in both builds, and no ordinary test asserts in Debug.
