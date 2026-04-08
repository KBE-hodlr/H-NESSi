This example implements a 1D Second-Born self-energy solver adapted from a previous 2D example.

Required `mpi_comm` k-ordering for this solver:
- `k` must run from `0` to `L-1` (inclusive) and be stored contiguously in the communicator's mapping functions (e.g., `map_mat(k,t)`).
- Each `k` index maps directly to the FFT input index: `in_fft[k]` corresponds to momentum `k`, and after `fftw_plan_dft_1d(L, ...)` a backward transform produces real-space sample at index `r=k`.

Notes:
- There is no symmetry-adaptation in this version; all `L` k-points and `L` real-space points are explicitly stored and processed.
- If your `mpi_comm` currently provides a different ordering, adapt it to the contiguous `0..L-1` ordering before using this solver.

