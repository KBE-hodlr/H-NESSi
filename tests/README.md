# H-NESSi regression suite

Built when the project is configured with `-DBUILD_TEST=ON` (`../build.sh.example`; on Perlmutter `../build.sh` for
Release and `../build-debug.sh` for Debug). googletest is fetched by CPM. Every test checks the **correct** physics:
exact references, dense re-implementations of the documented discrete equations, or convergence orders. Every bug of
`agent_docs/BUGS.md` is fixed on this branch and has a regression test (map below), which carries the bug ID as a ctest
label. There are no expected-failure tests.

## How to run

```bash
cd build/tests
ctest -j16                 # everything
ctest -j16 -LE slow        # skip the long sweeps and energy-balance runs (recommended for Debug)
ctest -L slow              # only those
ctest -L bootstrap         # one module: integration dlr bosonic free_extrap storage timestep bootstrap mpi_comm
                           #             read_inputfile checks tv nessi link example
ctest -L unit              # all gtest tests except the slow ones
ctest -L '^B14$'           # the regression tests of one bug (map below); -L is a regex, so plain `-L B1` would
                           # also select B10..B19
ctest -L Q10               # the energy-balance tests of ISSUES Q10
```

Run the gtest executables through ctest. Each test then gets its own process. If you run a binary directly, use
`--gtest_filter`: some test bodies `fork()` (`hn_test::run_isolated`), and libgomp is not fork-safe once an earlier
test has started its thread pool.

Build types (top-level `README.md`, "Compilation"):
* **Release** (`-O3 -DNDEBUG`): the tier-2 checks (always-on `precondition_error`/`state_error`/`numerical_error`
  throws) are tested; the death tests of the tier-1 asserts are skipped (`GTEST_SKIP`, shown as passed by ctest):
  `HodlrChecksDeathTest.Tier1Accessors`, `IntegratorChecksDeathTest.Tier1Accessors`,
  `FunctionChecksDeathTest.Tier1Accessors`, `MpiCommChecksWP6bDeathTest.Tier1Maps`, and the abort branch of
  `ChecksMacrosDeathTest.Tier1AssertAbortsOnlyWithoutNDEBUG`.
* **Debug** (`../build-debug.sh`) or **RelWithAsserts** (`-O2 -g`, asserts on, `EIGEN_NO_DEBUG`): the same tests plus
  those death tests. The OpenMP-policy death tests (a failed check inside a user parallel region aborts;
  `ChecksOpenMPDeathTest.*`, `MpiCommChecksDeathTest.*`, `MpiCommChecksWP6bDeathTest.TerminateHandlerCallsMpiAbort`)
  run in every build.
* **`-DINCLUDE_NESSI=ON`** (with `-DNESSI_ROOT=…`): adds `test_nessi_interop`, labels `unit;nessi;B8` (`ctest -L nessi`).
* **`-DHN_EXAMPLE_MPI_LAUNCH="srun -n 6 -c 2"`** (inside an allocation): registers the multi-rank restart test
  `MpiExampleRestartManyRanks_B35` (labels `example;B35;mpi`). Without it the test does not exist.
* **AddressSanitizer** build: see the end of the Executables section (label `asan`).

Size and timings (CLAUDE_dev after WP12, Perlmutter CPU node, `ctest -j12`, wall): 344 tests in the default build (316
`unit` incl. the 2 link tests, 23 `slow`, 7 `example`); Release 22 s, RelWithAsserts 16 s, Debug 190 s (the `slow` sweeps
and the energy-balance runs dominate); `-DINCLUDE_NESSI=ON` adds 13 `nessi` tests (357, 17 s in Release).
WP13 adds the 6 `unit` tests of `test_svd` (350 tests; `ctest -j32`: Release 19 s, Debug 183 s).
P1 (Dyson-solver optimisation) adds `DysonP1.LuCacheBitwiseVsFreshObject` and `DysonP1.ProfileOffBitwiseAndLevelTiming` to
`test_timestep` (352 tests).

## Labels and naming convention

| gtest name pattern / test | ctest labels | properties |
|---|---|---|
| `*.Slow_*` | `slow;<module>` (+ `B<id>`) | timeout 3600 s |
| every other gtest test | `unit;<module>` (+ `B<id>`) | timeout 600 s, `OMP_NUM_THREADS=2` (`test_mpi_comm`: 4) |
| link tests `OperatorAssignLinkAndDeepCopy_B6`, `ReadInputfileTwoTULink_B19` | `unit;B<id>;link` | `RESOURCE_LOCK hn_build_tree` |
| example tests (7, see below) | `example;<bug ID or Q10>` (+ `slow` for the two energy-balance tests, + `asan` in an ASan build) | timeout 900 s, `OMP_NUM_THREADS=2` |

* **Labels of gtest tests.** `gtest_discover_tests` flattens list-valued properties, so labels are attached by the
  generated ctest include file `hn_labels.cmake`. Every gtest test carries `unit` or `slow` and its module. The
  regression tests of a bug also carry `B<id>`: the `hn_bug_tests(B<id> <pattern>…)` calls in `CMakeLists.txt` (gtest
  names, `*` as wildcard) are the machine-readable form of the map below, so update both when adding a test for a bug.
  `hn_add_test_exe(… LABELS …)` adds labels to every test of an executable (`test_nessi_interop`: `B8`).
* **Bugs found later.** The WP0 expected-failure machinery (`KnownBug_B<id>_*` tests registered with `WILL_FAIL`) was
  removed in WP12b once every bug was fixed. A regression test for a new bug is written against the correct behaviour
  and added with the fix; a body that may crash or fail an `assert()` can run in a forked child
  (`hn_test::run_isolated`, `common/test_util.hpp`), so that it fails as an ordinary test.
* **Link tests** (B6, fixed in WP4; B19, fixed in WP6a). The targets `test_b6_operator_assign` and
  `test_read_inputfile_2tu` are `EXCLUDE_FROM_ALL`; the ctest builds the target with `cmake --build`
  (`link/build_and_run.cmake`) and then runs it.
* **`h5flat`** (`tools/h5flat.cpp`) flattens HDF5 files to text. The example tests and
  `agent_scratch/baseline/compare.py` use it.

## Executables

Module labels in parentheses.

| executable | ported from | contents |
|---|---|---|
| `test_integration` (integration) | `agent_scratch/02/t.cpp` | polynomial exactness of every weight table (k = 0..6); `bd_weights` with k+2 terms; Gregory symmetry; convergence orders |
| `test_dlr_matsubara` (dlr) | `03/test_dlr.cpp`, `51/t_mat.cpp` | DLR maps, `get_mat_tau`, `density_matrix`, reversed/tv accessors, `tv_it_conv`, `les_it_int`, `gamma_integral(_mat)` against quadrature; `dyson_mat` against the exact hybridised G^M (μ shift, mixing, return value); `dyson_mat` leaves Σ and both conv tensors untouched, `fixHam=true` reuses the saved G0; `dyson::memory_usage_bytes` counts every heap buffer with its real size (B32); the `initGMConvTensor` dlr ξ/r check (tier 2, throws since WP6a); B1 B2 B3 B4 (B1, B2 fixed in WP3; B4 fixed in WP2; B3 fixed in WP-N: `get_*_tau_array` = pointwise accessors) |
| `test_bosonic` (bosonic) | `51/t_resolvent.cpp`, new (WP3) | exact resolvent identity G_ε∗G_η = (G_ε−G_η)/(ε−η) for both ξ; B1 B2 B10 (fixed in WP3) and the Q2 relation; end-to-end system+bath run (dyson_mat → initGMConvTensor → dyson_start → dyson_timestep, drive after the bootstrap) vs the exact propagator, both ξ, order k+1 for k = 2, 4 in R, <, ⌉ |
| `test_free_extrap` (free_extrap) | `23/ta,tb,tc.cpp` | `green_from_H` exact (real and complex H, both ξ); `green_from_H_dm`; `extrapolate` order k+1 (and the `_2leg` variant, no-op on a second call); `gamma_integral` against a dense quadrature, order k+1; `gamma_integral(0)` = ξ·`gamma_integral_mat` (N_o = 1, 2); `green_from_H_dm` G^⌉(t,0) = G^<(t,0); B2 B10 (fixed in WP3) |
| `test_hodlr_storage` (storage) | old `tests/linalg.cpp`, `52/s9.cpp`, `52/s1.cpp` | `mgs`, incremental TSVD; fill/archive/read-back round trips for 8 geometries; zero-first-row blocks (S9); checkpoint round trip; checkpoint `format_version` (v3 without `tv_trans`, WP6c; a v2 file with `tv_trans` still loads and its `tv_trans` is ignored; a newer version is refused; new bosonic file silent, missing version warns on stderr, fermionic silent, not in `write_to_hdf5` output); copy semantics (WP4): `get/set_timestep_curr` leave the source bitwise unchanged (full checkpoint dump compared), give the destination the source window, tstp = −1 with different windows, window/geometry checks (tier 2, throw); the window copies carry `can_extrap` (WP9a); `operator=`/copy ctor deep copy, and a copy taken mid-run (dyson_mat → dyson_start → dyson_timestep) continues bitwise like the original; `h5_join` labels; B6 B22 (WP4), B23 (WP6a), B9 B26 B33 (WP9a) |
| `test_svd` (linalg) | `agent_scratch/svd/bdc_repro.cpp`, WP13 | `arrow_svd` (real arrow-matrix SVD of `updatetsvd`) and `svd()`/`tsvd()` against JacobiSVD: the 18×18 matrix of the SVD study (`data/b38_K_bad.bin`), a 68×68 real arrow matrix on which Eigen 3.4.0's real BDCSVD fails (`data/b38_A_real_bad_n68.bin`, JacobiSVD fallback), 1210 random arrow matrices n = 1..121 with degenerate entries, 60 random complex matrices up to 120; `updatetsvd` on a block whose arrow matrix is K_bad; B38 (WP13) |
| `test_timestep` (timestep) | `20/rettv.cpp`, `21/les.cpp` | R/⌉ step against dense discrete equations (5 geometries), against svdtol, order k+1 (k = 1..5); < step against the dense system, the 8-region decomposition of `les_lesadv_int_0_tstp` and raw storage, both rho modes, 2-leg, closed-system N conservation, order k+1; B14 (fixed in WP7: `_nobc` vs default `dyson_timestep`, both ξ); B28 (fixed in WP9a: timing column 5); P1: the content-keyed LU cache gives bitwise the G of a fresh object per call (also after rewriting an archived h(t')), profile on/off bitwise, per-level block timing sums and HDF5/checkpoint round trip |
| `test_bootstrap` (bootstrap) | `22/start_test.cpp`, `51/t_start*.cpp` | TTI and non-TTI bootstrap (horiz/diag, nobc, 2leg) against a dense reference that uses the **correct** continuation and the right-acting imp_tp0 system, plus order k+1 against the exact impurity+bath model; the imp_tp0 error estimate; B12 B13 B25 B31 (all fixed in WP1); B34 (fixed in WP7, 2leg and idempotence in WP7b): RHO_DIAGONAL reference with G^<(m,m) known in the row solves, iterated to convergence and compared with one library call, fixed-point check of the diagonal, second call err ≈ 0; `BootstrapRhoDiagonal.*` closed system (Σ = 0, both ξ, ntti/nobc/2leg): diagonal-equation residual, anti-Hermiticity, N conservation over t ≤ k, order k+1; `Idempotent*` (closed and bath Σ, both ξ, zero / green_from_H start); `NonContractingThrows` |
| `test_mpi_comm` (mpi_comm) | `31/commtest.cpp` | `mpi_comm` routing with P simulated ranks: MPI symbols interposed (incl. `MPI_Query_thread`), std::thread per rank, 1–5 ranks, spawn/nospawn, vector/reference_wrapper, high-level and lambda APIs; `_nospawn` in a `num_threads(2)` team and outside a region; trailing barrier of the set (H24); `my_first_t`/`my_Nt` (H30); thread-level warning (H28); B20 B21 (fixed in WP5; `OMP_NUM_THREADS=4`). WP6b: shim also interposes `MPI_Initialized`, `MPI_Finalized`, `MPI_Allreduce` (Debug collective check), `MPI_Abort`; `MpiCommChecksWP6b*` (constructor MP-1/2, thread level MP-3, ti/objects/dlr MP-6/8/9, get→set pairing ST-9, Im Σ^M warning MP-13, terminate handler → MPI_Abort, tier-1 maps); the in-region count check now aborts (death test, WP6a side finding 2) |
| `test_read_inputfile` (read_inputfile) | new | `find_param` for int/double/bool/cplx/vector; missing values throw; every error is a `std::runtime_error` with flag, token and file; B15 (fixed in WP6a) |
| `test_nessi_interop` (nessi) | new (WP-N) | **only with `-DINCLUDE_NESSI=ON`**, label `nessi` (`ctest -L nessi`). The five `distance_norm2_*` overloads for `cntr::herm_matrix<double>` (doc 05): exact `green_from_H` in both libraries (roundoff for R/<, DLR error for tv/M; ξ = ±1, N_o = 1, 2), shared sign conventions, complex h (D1: mat distance = Im G^M), single-entry perturbations seen only by the matching function, checked preconditions (incl. a tv row above `tstp()`, refused by `distance_norm2_tv` itself, WP12), and full driven dyson runs (Σ = 0 and a fixed bath Σ, both ξ) in both libraries with order-k+1 convergence of the distance; B8 |
| `test_checks` (checks) | new (WP6a) | the check infrastructure (`checks.hpp`): exception types and message format, tier-1 behaviour, OpenMP policy (death tests: abort inside a user region; capture/rethrow in library regions); constructor validation (herm_matrix_hodlr HM-C*, Integrator KMAX, dlr_info DL-*, dyson DY-C*); dyson entry checks (geometry bundle, step synchronisation DY-T1, bootstrap window DY-S1, `green_from_H*` tmax/window H11, dlr pairing, extrapolate); finiteness (L9, DY-P1); ST-2 (stale convolution tensor, checkpoints incl. pre-v2 bosonic); ST-3 (saved G0); D-11 Hermiticity of h (throw before modifying, in-place symmetrisation + diagnostic, opt-out, warn once, complex h(−1), RHO_HORIZONTAL mean-field run); concurrent `dlr_info` construction (H3) |
| `test_checks_wp6b` (checks) | new (WP6b) | the remaining checks of doc 61: herm_matrix_hodlr (`set_tstpmk`, `time2block` failure path, DLR evaluators τ/row/output sizes, `density_matrix`, `update_blocks` HM-U1/U2, checkpoint dataset sizes HM-C11, output writers HM-W2–W5, distances HM-D*; tier-1 accessor death tests); D-11 predictor/corrector Hermiticity (`HermiticityCorrector.*`); component order ST-4; single-thread ownership ST-10 (dyson, dlr_info); `eval_point` τ range; dyson checkpoint parameters ST-11; bosonic spectrum, complex h with G^M, density-matrix validation (DY-M10/F7/F9/F10); bootstrap TTI tolerance, G^<(0,0), singular systems (DY-S4/S7/S10); `gamma_integral`, `tv_ret_int`, `les_lesadv_int_0_tstp` (DY-G*, DY-I5/I6); ST-6 (`extrapolate` after a write into row T, also by `Bubble1`/`Bubble2`, WP12); Integrator weights IN-2 and tier-1 accessors; `function` FN-*; `Bubble1/2` BU-*; `read_inputfile` RI-3/4/6 |
| `test_tv_contractions` (tv) | new (WP6c) | label `tv`. ISSUES H27 (`tv_trans` removed): the OLD kernels (matrix products on an explicitly built orbital-transposed copy of tv, test-only reference) against the new Eigen Tensor readers `les_it_int` (both overloads, incl. swapped arguments), `tv_it_conv` (both overloads), `get_vt` (all overloads), `Bubble1`, for N_o ∈ {1,2,3,4,8,16}, both ξ, r ∈ {19, 35, 42, 52, ~64}, T up to 3000, non-symmetric complex data (≤ 1e-14 normwise relative; prints the max); writes through `map_tv`/`tvptr`/`set_tstp_zero`/Bubble1 output are seen by every reader (the former stale-cache hazard); `get_memory_usage` counts tv once |
| `test_read_inputfile_2tu` (link) | `52/ri1+ri2.cpp` | B19 |
| `test_b6_operator_assign` (link) | new | B6 (fixed in WP4; ctest `OperatorAssignLinkAndDeepCopy_B6`, ordinary test) |
| `MpiExampleRestart_B16` (example; formerly `KnownBug_B16_MpiExampleRestart`) | `agent_docs/32` §6.3 | `examples/mpi_restart.sh`: runs `mpi_example` (L=4, Nt=64) uninterrupted, then stop (`__time_limit=0`) followed by restart; `obs.h5` (incl. `jxt_iter`; since WP8b also `current_x`, `efield`, `energy_*`) must agree **bitwise** |
| `MpiExampleIterRows_B17` (example, WP8) | new | `examples/mpi_iter_rows.sh`: BootMaxIter=2, StepMaxIter=25, StepMaxErr=0, Nt=12; exit 0, no sanitizer report, `jxt_iter` is 26×12 |
| `ScExamplePhiLength_B24` (example, WP8) | new | `examples/sc_phi_length.sh`: `SC.x` with nt=32769 and the shipped `phi_input.h5` (32768 entries) stops before allocating, with exit ≠ 0 and the message, no output file |
| `ScExampleRun` (example, WP8) | new | `examples/sc_run.sh`: `SC.x` on `data/sc_tiny.inp` (nt=64, nlevel=3); exit 0, no sanitizer report |
| `ScExampleEnergyBalance` (example, WP8, labels `Q10;slow`) | new | `examples/sc_energy_balance.sh`: `SC.x` with a contained pulse (`data/sc_pulse.h5` from `tools/mkpulse.sh`, h=0.03, nt=300, `data/sc_pulse.inp`); max\|E_total(t)−E_total(0)−E^j(t)\| < 1e-5 (observed 5.3e-7, O(h⁶); wrong conventions give ≥ 1.5e-2) and E^j(end) > 0.1 |
| `MpiExampleEnergyBalance` (example, WP8b, labels `Q10;slow`) | new | `examples/mpi_energy_balance.sh`: `Main.x` (1 rank) with `data/mpi_pulse.inp` (L=4, `__pulse= smooth`: A_x = 0.8 sin²(π(t−0.2)/4) sin(π(t−0.2)), h=0.04, Nt=151); max\|E_total(t)−E_total(0)−E^j(t)\| < 1e-5 (observed 3.5e-7; wrong factors or sign give ≥ 5e-4) and max\|E^j\| > 0.01; StepMaxErr/BootMaxErr 1e-20 (WP9; were 1e-24) |
| `MpiExampleRestartMidRun` (example, WP9, label `B16`) | new | `examples/mpi_restart.sh … 20 40`: as `MpiExampleRestart_B16`, but split by two chained mid-run stops (`__stop_tstp= 20`, then `40`); `obs.h5` bitwise, and `timings_file.h5` has no zero row for t > k (timings restored on restart) |
| `MpiExampleRestartManyRanks_B35` (example, WP9, labels `B35;mpi`) | new | only registered with `-DHN_EXAMPLE_MPI_LAUNCH="srun -n 6 …"` (needs an allocation): `examples/mpi_restart.sh` with `data/mpi_L2.inp` (L=2: 4 k-points on 6 ranks, Nt=32), stops at 10 and 20; bitwise |

**AddressSanitizer build (WP8).** Configure a build with `-DCMAKE_CXX_FLAGS="-g -fsanitize=address -fno-omit-frame-pointer"`
(same for `CMAKE_C_FLAGS`) and `-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address -DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=address`
(works with the Cray `CC` wrapper / g++). The five example tests then carry the extra label `asan` (`ctest -L asan`) and
fail on any `Sanitizer` line in the program output. The whole suite also runs in that build.

## BUG ID → test map

All entries are fixed (WP and commit in `agent_docs/BUGS.md`); every listed test passes in Release, Debug and
RelWithAsserts (and the `nessi` tests with `-DINCLUDE_NESSI=ON`). Every listed test carries the label `B<id>`
(`ctest -L '^B<id>$'`).

| BUGS id | test(s) | fixed in | symptom before the fix |
|---|---|---|---|
| B1 | `DlrBoson.TvItConvVsQuadrature`, `ResolventBoson.TvItConv`, `EndToEndBoson.MatStartTimestepVsExact`, `GMConvTensor.DlrXiOrRankMismatchThrows`, `HodlrCheckpoint.FormatVersion*` | WP3 | bosonic `tv_it_conv` = −reference (rel. err 2.0) |
| B2 | `DlrBoson.GammaIntegralMixedTerm`, `ResolventBoson.GammaIntegralEquilibrium`, `ResolventBoson.GammaIntegralVsMat`, `GammaIntegralBoson.VsQuadrature`, `GammaIntegralBoson.EquilibriumVsMat`, `GammaIntegralBoson.EquilibriumVsMatSingleOrbital` | WP3 | wrong sign of the bosonic mixed term (error 2.87); `gamma_integral = ξ·gamma_integral_mat` violated |
| B3 | `DlrArray.GetMatTauArray`, `DlrArray.GetTvTauArray`, `DlrArrayP.MatTauArrayEqualsPointwise/*`, `DlrArrayP.TvTauArrayEqualsPointwise/*` | WP-N | SIGSEGV |
| B4 | `DysonMat.FixHamSecondCall`, `DysonMatFixHam.ReusesSavedG0/*` | WP2 | rel. difference 1.10 |
| B6 | `OperatorAssignLinkAndDeepCopy_B6` (link), `HodlrAssign.DeepCopyIndependent`, `HodlrAssign.CopyContinuesDysonIdentically` | WP4 | undefined reference to `operator=` |
| B8 | `test_nessi_interop` (`NessiEq.*`, `NessiConventions.*`, `NessiEqComplexH.*`, `NessiPerturb.*`, `NessiPreconditions.*`, `NessiDynamics.*`; label `nessi`, only with `-DINCLUDE_NESSI=ON`) | WP-N | block compiled out; did not compile with the macro |
| B9 | `HodlrCheckpoint.TimingZeroWhenMissing_B9` | WP9a | uninitialised timing (malloc perturbation) |
| B10 | `ResolventBoson.GreenFromHdm`, `GreenFromHdm.BosonExact` | WP3 | bosonic G^< error 0.13 / 0.11 |
| B12 | `Bootstrap.NttiHorizSlotZero`, `…NttiDiagSlotZero`, `…TwoLegDiagSlotZero`, `BootstrapRetNtti.OrderSlotZero`, `Bootstrap.Slow_*SlotZeroSweep` | WP1 | library ≠ correct reference (≈1e-3); order 2 instead of k+1 |
| B13 | `Bootstrap.NttiImpTp0`, `BootstrapRetNtti.OrderImpTp0TwoOrbitals`, `…OrderImpTp0TwoOrbitalsSlotZero`, `Bootstrap.Slow_NttiImpTp0SlotZeroSweep` | WP1 | column-0 order 3 for k ≥ 3, N_o = 2 |
| B14 | `LesNobc.AccuracyAndBoundary_B14`, `LesNobc.MatchesDefaultTimestep_B14` | WP7 | `_nobc` G^< error 0.33; boundary mismatch 5.7e-3 |
| B15 | `ReadInputfile.EmptyBoolThrows_B15`, `ReadInputfile.ErrorsAreRuntimeErrorsWithContext` | WP6a | SIGSEGV instead of an exception |
| B16 | `MpiExampleRestart_B16`, `MpiExampleRestartMidRun` | WP8 (mid-run stops: WP9) | restart run exits with 139 (segfault) |
| B17 | `MpiExampleIterRows_B17` (a memory check only in the ASan build, label `asan`) | WP8 | silent heap overrun |
| B19 | `ReadInputfileTwoTULink_B19` (link) | WP6a | multiple definition of `uncomment_line` … |
| B20 | `MpiCommChecks.TooFewComponentsHighLevel`, `MpiCommChecks.TooManyLambdas`, `MpiCommChecksDeathTest.NospawnCheckInsideRegionAborts` | WP5 | no exception; heap corruption |
| B21 | `MpiCommNospawn.TeamSmallerThanMaxThreads`, `MpiCommNospawn.CalledOutsideParallel`, `MpiCommNospawn.LambdaTeamSmallerThanMaxThreads`, `MpiCommNospawn.LambdaCalledOutsideParallel` | WP5 | stale/unpacked buffer entries |
| B22 | `HodlrCopy.GetTimestepCurrWindow`, `HodlrCopy.GetTimestepCurrSourceUnchangedAndContinues`, `HodlrCopy.SetTimestepCurrSourceUnchanged`, `HodlrCopy.GetTimestepCurrMatsubara`, `HodlrCopyChecks.WindowAndGeometryThrow` | WP4 | source tstpmk 8 → 0; destination window wrong; Debug assert before the `tstp==-1` branch |
| B23 | `HodlrGeometry.NlvlZeroRejected_B23`, `CtorChecks.HermMatrixHodlrGeometry` | WP6a | SIGSEGV in `init_shape` |
| B24 | `ScExamplePhiLength_B24` | WP8 | silent out-of-bounds read of `phi` |
| B25 | `Bootstrap2leg.FunctionOverloadForwardsImpTp0` | WP1 | `function&` and pointer overloads differ when `imp_tp0=true` |
| B26 | `HodlrMemory.MemoryReport_B26` | WP9a | reported memory ≠ true block storage |
| B27 | — (wall-clock duration, no deterministic reference) | WP9a | `herm_matrix_hodlr` timing column 5 measured only the last segment |
| B28 | `DysonTiming.Column5Accumulated_B28` | WP9a | dyson timing column 5 always 0 |
| B29, B30 | — (notebook plot, source comments) | WP8, WP2 | — |
| B31 | `BootstrapRetNtti.ImpTp0ErrorEstimateTwoOrbitals` | WP1 | Debug Eigen assert; wrong error in Release |
| B32 | `DysonMemory.CountsAllBuffers` | WP2 | buffers not counted / counted with the wrong size |
| B33 | `HodlrCheckpoint.FreshGMConvTensorZero_B33` | WP9a | uninitialised `GMConvTensMatrix_` in a fresh object's checkpoint |
| B34 | `Bootstrap.*Diag*` (fixed-point check), `BootstrapRhoDiagonal.ClosedSystem*`, `BootstrapRhoDiagonal.Idempotent*`, `BootstrapRhoDiagonal.NonContractingThrows` | WP7, WP7b | converged bootstrap diagonal violated its own diagonal equation (residual ≈1e-5 at k = 3) |
| B35 | `MpiExampleRestartManyRanks_B35` (only with `-DHN_EXAMPLE_MPI_LAUNCH`) | WP9b | restart segfaulted on ranks without k-points |
| B36 | — (bitwise for the default `__Ainitx= 0`; energy bookkeeping checked by `MpiExampleEnergyBalance`) | WP8b | velocity omitted `Ainitx` |
| B37 | `ExtrapolateState.BubbleWriteIntoRowTBeforeExtrapolate` | WP12 | `extrapolate` after `Bubble1/2` into row `tstp()` replaced the bubble's G^< by a prediction (Δ 0.02) |
| B38 | `SvdB38.*` (`test_svd`): `KBadPublicSvdTsvd`, `UpdatetsvdOnKBad` (fail on 4b5bf4b), `KBadArrowSvd`, `RealBdcFailureFallsBackToJacobi`, `ArrowSvdRandomVsJacobi`, `PublicSvdRandomComplex` | WP13 | Eigen 3.4.0 BDCSVD: `svd()` residual 1.4e-3 on K_bad; `updatetsvd` block error 1.4e-3 vs bound 4.3e-9 |

ISSUES entries with tests: Q10/D-7 energy conventions (`ScExampleEnergyBalance`, `MpiExampleEnergyBalance`, label
`Q10`); H27 (`test_tv_contractions`); H3 (`DlrInfoThreads.ConcurrentConstructionMatchesSerial`); the checks of doc 61
(`test_checks`, `test_checks_wp6b`).

The B14 test also checks the correct boundary sign through its dense reference: the conjugate KBE with
−`les_it_int`.
