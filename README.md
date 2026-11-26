# H-NESSi: The Hierarchical Non-Equilibrium Systems Simulation package

[![DOI](https://zenodo.org/badge/1180109148.svg)](https://doi.org/10.5281/zenodo.18990387)
[![Paper](https://img.shields.io/badge/Paper-ScienceDirect-blue.svg)](https://www.sciencedirect.com/science/article/pii/S0010465526002389)

H-NESSi is an open-source C++ software package for solving the Kadanoff-Baym equations (KBE) of nonequilibrium Green's function (NEGF) theory. 

By combining high-order time-stepping schemes with hierarchical off-diagonal low-rank (HODLR) compression techniques and the discrete Lehmann representation (DLR), H-NESSi bypasses the cubic time scaling and quadratic memory growth of conventional two-time formulations. This enables long-time and large-system nonequilibrium simulations of correlated quantum materials. This repository contains the set of classes implementing the basic HODLR extension of the NESSi library, along with example programs and unit tests.

## Documentation

Full API documentation, tutorials, and examples can be found at:  
**[https://kbe-hodlr.github.io/H-NESSi/](https://kbe-hodlr.github.io/H-NESSi/)**

## Building from Source

### Prerequisites
* A **C++20** compiler (the project sets `CMAKE_CXX_STANDARD 20`) and CMake ≥ 3.13
* MPI and OpenMP
* HDF5
* [FFTW3](https://www.fftw.org/), including the `fftw3_omp` library
* [Eigen](https://eigen.tuxfamily.org/) ≥ 3.4
* [libdlr](https://github.com/jasonkaye/libdlr), built separately (`LIBDLR_LIB` must contain `libdlr_c`, `LIBDLR_HEADER` must contain `dlr_c/dlr_c.h`)
* [NESSi (optional)](https://github.com/nessi-cntr/nessi), only for the interoperability functions (see below)

[HighFive](https://github.com/BlueBrain/HighFive) and, with `-DBUILD_TEST=ON`, googletest are downloaded by CPM during the first configure. Set `-DCPM_SOURCE_CACHE=<dir>` (default `<source>/.cpm-cache`) to share the download between build trees or to build offline once the cache is filled.

**BLAS/LAPACK and libdlr.** libdlr is the only BLAS/LAPACK user, and H-NESSi calls it from inside OpenMP regions, so the BLAS it binds to must be **serial**. H-NESSi links `HNESSI_BLAS_LIBRARIES` ahead of libdlr's own dependencies; the default is the serial Cray libsci of the loaded programming environment on Cray PE, and libdlr's own BLAS elsewhere. If libdlr was built against the reference BLAS (e.g. `/usr/lib64/libblas.so.3`) and you do not override it, the DLR operations run on that reference BLAS; build libdlr against an optimised serial BLAS, or set `HNESSI_BLAS_LIBRARIES`, if this matters for performance.

### Compilation
A template build script is provided as `build.sh.example`:

```bash
cp build.sh.example build.sh      # build.sh is ignored by git
# edit build.sh: replace the <...> placeholders (compilers, FFTW_DIR, LIBDLR_LIB, LIBDLR_HEADER)
./build.sh [BuildType] [extra cmake arguments]     # configures and builds in ./build (override with BUILD_DIR=...)
```

It runs, in essence:

```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=<cc> -DCMAKE_CXX_COMPILER=<CC> \
      -DFFTW_DIR=/path/to/fftw \
      -DLIBDLR_LIB=/path/to/libdlr/build/lib -DLIBDLR_HEADER=/path/to/libdlr/src \
      -DCPM_SOURCE_CACHE=/path/to/.cpm-cache \
      -DBUILD_TEST=ON \
      /path/to/H-NESSi
make -j8
make install          # optional; set -DCMAKE_INSTALL_PREFIX
```

Build types:

| `CMAKE_BUILD_TYPE` | Flags | Checks |
|---|---|---|
| `Release` (default in `build.sh.example`) | `-O3 -DNDEBUG` (CMake default) | always-on argument/state checks only |
| `RelWithAsserts` | `-O2 -g -DEIGEN_NO_DEBUG` | additionally the hot-path asserts (`assert`, `HNESSI_ASSERT`); recommended for testing new programs |
| `Debug` | `-g` (CMake default; unoptimised) | all asserts, including Eigen's |
| `RelWithDebInfo` | CMake default | as Release |

Other options: `-DBUILD_TEST=ON` (regression suite), `-DINCLUDE_NESSI=ON` (NESSi interoperability, below), `-DHNESSI_BLAS_LIBRARIES=<serial BLAS/LAPACK>`.

A single configure gives the final link line; a second `cmake .` is not needed.

### Linking your own programs
Link through the **`h_nessi` CMake target**, as `SC_example/` and `mpi_example/` do (add your program to the H-NESSi CMake project, e.g. with `add_subdirectory`, and `target_link_libraries(myprog h_nessi)`). The target carries the include paths, the `INCLUDE_NESSI`/`CNTR_USE_*` definitions, and a deterministic link line: MPI, HDF5, FFTW, OpenMP without the threaded libsci/FFTW that the Cray compiler wrapper adds, and the serial BLAS ahead of libdlr. The install step does not export a CMake package configuration.
Programs linked by hand (`CC -fopenmp … -lh_nessi -ldlr_c -ldlr`) can end up with both the serial and the threaded libsci, which prints `CRAYBLAS_WARNING: linked against multiple cray-libsci libraries`, and libdlr may then bind to the threaded one. If you link by hand, put the serial BLAS (e.g. Cray's `-lsci_gnu`) before `-ldlr_c -ldlr`.

### Testing
Configure with `-DBUILD_TEST=ON`, build, then run the suite from the build tree:

```bash
cd build/tests
ctest -j16                 # everything (Release: well under a minute with -j16)
ctest -j16 -LE slow        # skip the long convergence sweeps and energy-balance runs (recommended for Debug)
ctest -L bootstrap         # one module: integration dlr bosonic free_extrap storage timestep bootstrap mpi_comm
                           #             read_inputfile checks tv link example
ctest -L B12               # every test that exercises one (fixed) bug of agent_docs/BUGS.md
```

* Tests check exact references, dense re-implementations of the discrete equations, and convergence orders. Every bug fixed on this branch has a regression test labelled with its ID (`tests/README.md` has the map).
* Label `slow`: the `*.Slow_*` convergence sweeps and the example energy-balance tests `ScExampleEnergyBalance` (about 200 s in Debug) and `MpiExampleEnergyBalance` (about 110 s in Debug).
* Build the suite in Debug (`./build.sh Debug` with `BUILD_DIR=build-debug`) or RelWithAsserts to run the death tests of the asserts; they are skipped in Release.
* `-DINCLUDE_NESSI=ON` adds the `nessi` tests (`ctest -L nessi`). `-DHN_EXAMPLE_MPI_LAUNCH="srun -n 6 -c 2"` (inside an allocation) registers the multi-rank restart test `MpiExampleRestartManyRanks_B35`.
* An AddressSanitizer build of the examples is described in `tests/README.md` (label `asan`).
* The HDF5 dump tool `tests/tools/h5flat` (used by the example tests) links `${HDF5_LIBRARIES}`. That variable is empty when the compiler wrapper adds HDF5 itself (Cray `CC`), which works; with a plain compiler, FindHDF5 must find the HDF5 C library.

## NESSi interoperability
With `-DINCLUDE_NESSI=ON`, H-NESSi compiles the five `distance_norm2_*` functions that compare a `herm_matrix_hodlr` with a NESSi `cntr::herm_matrix<double>` (absolute Frobenius norms; the mixed and Matsubara components are evaluated on NESSi's equidistant τ grid through the DLR).
* Point CMake to an installed NESSi (libcntr) with `-DNESSI_ROOT=<install prefix>`, or with `-DNESSI_INCLUDE_DIR=<dir containing cntr/>` and `-DNESSI_LIB=<dir containing libcntr>`.
* libcntr must have been built with `-Domp=ON -Dhdf5=ON -Dmpi=ON`. Otherwise set the matching `-DNESSI_CNTR_USE_{HDF5,OMP,MPI}=OFF`: the `CNTR_USE_*` macros must match the libcntr build, and libcntr installs no configuration header from which to detect them.
* **Include order:** include the H-NESSi headers **before** `cntr/cntr.hpp`. NESSi defines global macros (`II`, `cdmatrix`, `GREEN`, …) that would otherwise clash; `herm_matrix_hodlr.hpp` only forward-declares `cntr::herm_matrix`.

## Using the library: behaviour to be aware of
**Errors.** Invalid arguments and calls in the wrong order are reported by exceptions derived from standard ones: `h_nessi::precondition_error` (a `std::invalid_argument`; bad argument, geometry mismatch, non-Hermitian h), `h_nessi::state_error` (a `std::logic_error`; e.g. `Sigma.update_blocks` not called once per time step, a stale convolution tensor, `fixHam=true` without a saved G0) and `h_nessi::numerical_error` (a `std::runtime_error`; non-finite results, a non-converging RHO_DIAGONAL bootstrap). Messages name the routine, the violated condition and the values involved (`include/h_nessi/checks.hpp`). These checks are always on and cheap relative to the call they guard; the per-element accessor bounds are asserts, active in Debug and RelWithAsserts only. `read_inputfile.hpp` throws `std::runtime_error` naming the flag and the file.
**OpenMP policy.** An exception cannot leave an OpenMP parallel region. A failed check **outside** a parallel region throws. Inside a parallel region **you** own (for example solver calls inside your `#pragma omp parallel`, as in `mpi_example`), it prints the message with MPI rank, thread and source location and aborts the job (`MPI_Abort` when MPI is initialised). Inside H-NESSi's own parallel regions (the `mpi_comm` `_spawn` routines), exceptions are collected and rethrown on the calling thread. Catch H-NESSi exceptions only around code whose parallel regions you own.

**Hermiticity of h(t).** h(t) must be Hermitian. Every `dyson_*` and `green_from_H*` call replaces each slice of h it reads by its Hermitian part (h+h†)/2, **in place in your `function` object or buffer**. It warns once if the relative anti-Hermitian part ‖h−h†‖/‖h‖ exceeds 1e-4 and throws above 1e-2. Mean-field h built from a G^<(t,t) that is not exactly anti-Hermitian (e.g. RHO_HORIZONTAL) therefore stays consistent. During a time step the tolerances apply to the h(T) of the corrector iterations, not to the extrapolated predictor; switch drives on smoothly. Tolerances: `dyson::set_hermiticity_tolerances(warn, err)`; check without modifying h: `set_hermitian_symmetrization(false)`; diagnostics: `last_antihermiticity()`, `antihermiticity_per_slice()`. A complex h(−1) is rejected where G^M is computed, because G^M is stored as real.

**RHO_DIAGONAL (the default `rho_version` of `dyson`).** G^<(t,t) is obtained from the density-matrix equation, which is a fixed point. In the time step, your corrector loop iterates it, even at fixed Σ. The non-TTI bootstraps (`dyson_start_ntti`, `_ntti_nobc`, `_2leg`) iterate rows and diagonal to the fixed point inside each call (relative tolerance 1e-14), so a call with fixed Σ is idempotent; your bootstrap loop is needed only for self-consistency in Σ. The returned error is the change of G between input and output. RHO_DIAGONAL has a time-step limit: in closed driven test models the bootstrap converges up to h·spread(h(t)) ≈ 0.9 (k = 3) / 0.6 (k = 5), and the time step up to about 1.7. Beyond the bootstrap limit the call throws `h_nessi::numerical_error`; beyond the time-step limit your corrector loop does not converge. RHO_HORIZONTAL has no such limit (paper §3.2 discusses the trade-off).

**Other contracts.** Call `initGMConvTensor` after the last change of G^M (the last `dyson_mat`) and before real-time steps; the solvers refuse a stale tensor. Call `update_blocks` k+1 more times after the last step before `write_to_hdf5` (it warns otherwise). The solver error returned by `dyson_*` is an unnormalised sum of per-entry Frobenius norms (it grows with T and N_o). Only square orbital matrices are supported, and the integration order is 1 ≤ k ≤ 6. Checkpoints are written in format version 3; versions 1 and 2 still load, but a bosonic checkpoint written before version 2 needs `initGMConvTensor` again.

**`read_inputfile.hpp`.** The header is header-only and can be included from several translation units. It no longer brings `using namespace h_nessi` or a global `strtok` overload into your code: add `using namespace h_nessi;` or qualify names (`h_nessi::cplx`, …) if your code relied on it.

## Examples
`SC_example/` (a quantum dot coupled to leads, TTI bootstrap) and `mpi_example/` (a 2D Hubbard lattice in 2nd Born, MPI over k-points). Each `README.txt` gives the command line and describes the outputs. Energies follow Galitskii–Migdal and are checked by energy conservation (ctests `ScExampleEnergyBalance`, `MpiExampleEnergyBalance`): the interaction terms (HF and 2nd-order Σ) carry ½, the one-body terms (hybridisation/hopping, band energy) carry 1, traces run over both spins, and `energy_total(t) − energy_total(0) = energy_deposited(t)` up to the discretisation error.
* `SC_example` writes `current`, `efield`, `energy_deposited`, `energy_hf`, `energy_corr`, `energy_hyb` and `energy_total`. Each has a `description` attribute. Earlier versions wrote `current` with the opposite sign, `energy_kin` (= `energy_hyb`) and `energy_pot` (= 2·`energy_corr`).
* `mpi_example` writes `current_x`, `energy_band`, `energy_hf`, `energy_corr`, `energy_total`, `efield` and `energy_deposited` in addition to the earlier `jxtnorm`, `jytnorm`, `Navgt` and `Ekint`. `Ekint` uses the bare ε_k and is kept for compatibility; use `energy_band` instead. The vector potential is A = +A_max (so `jxtnorm` = −σ of paper Eq. 30); `__pulse= smooth` selects a contained pulse. Restart from a checkpoint works on any number of ranks, and `__stop_tstp=` stops after a given step.

## Changes in this branch
Relative to upstream `5c9706b`. Every fix has a regression test; the full lists are in `agent_docs/BUGS.md` and `agent_docs/ISSUES.md`.
* **Results:** non-TTI bootstrap restored to order k+1 (was O(h²)); bosonic signs in `initGMConvTensor`, `gamma_integral` and `green_from_H_dm`; `dyson_mat(fixHam=true)`; `dyson_mat` no longer overwrites Σ^⌉(0) or the convolution tensor; consistent, idempotent RHO_DIAGONAL bootstrap; `_nobc` boundary sign; `get_timestep_curr` no longer corrupts its source.
* **Robustness:** always-on argument and state checks with `h_nessi` exceptions and an OpenMP error policy; h(t) Hermitian symmetrisation; the `RelWithAsserts` build type; `dlr_info` parameter validation and thread-safe construction; checkpoint version and parameter checks.
* **Memory/performance:** the orbital-transposed `tv_trans` cache is removed (Eigen Tensor contractions instead; −26 % G+Σ memory in `mpi_example`); exact memory reports.
* **MPI:** `_nospawn` routines are correct in any team size and outside parallel regions, end with a barrier, and check component counts; `MPI_Init_thread(FUNNELED)`.
* **API:** `herm_matrix_hodlr` copy/assignment work; vector DLR evaluation (`get_mat_tau_array`, `get_tv_tau_array`) fixed; NESSi interoperability compiles and is tested; `read_inputfile.hpp` is header-only; `dipole_step` is declared but not implemented.
* **Build:** `build.sh.example`, a deterministic link line on the first configure, and the regression suite (`tests/`).
* **Examples:** energy bookkeeping by energy conservation, restart fixes, bounds checks, and warnings when iteration caps are reached.

## Citation

If you use H-NESSi in your research, please cite our paper.
> **H-NESSi: The hierarchical non-equilibrium systems simulation package** \
> Thomas Blommel, Jeremija Kovačević, Jason Kaye, Emanuel Gull, Jakša Vučičević, Denis Golež \
> *Computer Physics Communications*, Volume 327, October 2026, 110256 \
> DOI: [10.1016/j.cpc.2026.110256](https://doi.org/10.1016/j.cpc.2026.110256) | [ScienceDirect Link](https://www.sciencedirect.com/science/article/pii/S0010465526002389)

## License

This project is licensed under the MIT License.
