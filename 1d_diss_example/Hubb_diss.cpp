#include "Hubb_diss.hpp"
#include <iostream>

// Constructor
Hubb_diss::Hubb_diss(function &U, int L, int nthreads) 
    : U_(U), L_(L), Nk_(L), nthreads_(nthreads) 
{

    ellG_ = 0;

    // Create optimal FFTW plans using temporary dummy arrays
    fftw_complex* in_dummy = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * L_);
    fftw_complex* out_dummy = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * L_);

    // Backward: k -> r (1D)
    Gk_to_Gr = fftw_plan_dft_1d(L_, in_dummy, out_dummy, FFTW_BACKWARD, FFTW_ESTIMATE);
    // Forward: r -> k
    Sigmar_to_Sigmak = fftw_plan_dft_1d(L_, in_dummy, out_dummy, FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_free(in_dummy);
    fftw_free(out_dummy);

    // Pre-allocate thread-local buffers to avoid malloc locks during OMP loops
    // need 2 per thread, as we need mat/revmat, les/ret, tv/revtv
    in_thread_vec.resize(2 * nthreads_);
    out_thread_vec.resize(2 * nthreads_);

    for (int tid = 0; tid < 2 * nthreads_; tid++) {
        in_thread_vec[tid] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * L_);
        out_thread_vec[tid] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * L_);
    }
}

// Destructor
Hubb_diss::~Hubb_diss() {
    fftw_destroy_plan(Gk_to_Gr);
    fftw_destroy_plan(Sigmar_to_Sigmak);

    for (int tid = 0; tid < 2*nthreads_; tid++) {
        fftw_free(in_thread_vec[tid]);
        fftw_free(out_thread_vec[tid]);
    }
    fftw_cleanup();
}

// Orchestrator
void Hubb_diss::Sigma_spawn(int tstp, 
                          mpi_comm &comm, 
                          std::vector<std::reference_wrapper<h_nessi::herm_matrix_hodlr>> &Grefs, 
                          std::vector<std::reference_wrapper<h_nessi::herm_matrix_hodlr>> &Srefs, 
                          h_nessi::dlr_info &dlr) 
{
    // 1. Fetch remote G data into comm buffers
    comm.mpi_get_and_comm_spawn(tstp, Grefs, dlr);

    // 2. Compute self-energy
//    if (tstp == -1) {
//        Sigma_Mat_spawn(comm);
//    } else {
        Sigma_Real_spawn(tstp, comm);
//    }

    // 3. Sync and write back S results
    comm.mpi_comm_and_set_spawn(tstp, Srefs);
}

// Matsubara Implementation
/*
void Hubb_diss::Sigma_Mat_spawn(mpi_comm &comm) 
{
    int init_tau = comm.my_first_tau;
    int last_tau = init_tau + comm.my_Ntau;

    #pragma omp parallel
    {
        // identify thread local fftw buffers
        int tid = omp_get_thread_num();
        fftw_complex* in_fft = in_thread_vec[2*tid];
        fftw_complex* out_fft = out_thread_vec[2*tid];
        fftw_complex* in_fft_2 = in_thread_vec[2*tid+1];
        fftw_complex* out_fft_2 = out_thread_vec[2*tid+1];

        // each mpi process must do assigned tau points and all k points
        #pragma omp for schedule(static)
        for (int tau = init_tau; tau < last_tau; tau++) {

            // We first FT G_k to G_r (1D)
            // Read G^M and G^Mrev into in_fft using contiguous k_points [0..L_)
            for(int ki = 0; ki < Nk_; ki++) {
                auto G_val = comm.map_mat(ki, tau)(0,0);
                in_fft[ki][0] = G_val.real();
                in_fft[ki][1] = G_val.imag();
                auto G_valrev = comm.map_mat_rev(ki, tau)(0,0);
                in_fft_2[ki][0] = G_valrev.real();
                in_fft_2[ki][1] = G_valrev.imag();
            }

            // Do the fft for both
            fftw_execute_dft(Gk_to_Gr, in_fft, out_fft);
            fftw_execute_dft(Gk_to_Gr, in_fft_2, out_fft_2);

            // FFT normalization factor
            double norm = 1.0 / L_;

            // Evaluate Sigma^M_r(\tau) = -U^2 G^M_r(\tau) G^M_r(\tau) G^M_{-r}(-\tau)
            //                             U^2 G^M_r(\tau) G^M_r(\tau) G^M_{-r}(\beta-\tau)
            for(int ri = 0; ri < L_; ri++) {
                int iri = (L_ - ri) % L_; // index for -r
                // Convert data in fftw buffers into std::complex
                std::complex<double> G_r(out_fft[ri][0] * norm, out_fft[ri][1] * norm);
                std::complex<double> Grev_minusr(out_fft_2[iri][0] * norm, out_fft_2[iri][1] * norm);
                std::complex<double> Sigma = U_ * U_ * G_r * G_r * Grev_minusr;
                in_fft[ri][0] = Sigma.real();
                in_fft[ri][1] = Sigma.imag();
            }
            
            // 4. Transform Sigma(r) -> Sigma(k)
            fftw_execute_dft(Sigmar_to_Sigmak, in_fft, out_fft);

            // 5. Map the full k-space grid back to your k-adapted storage (contiguous)
            for(int ki = 0; ki < Nk_; ki++) {
                std::complex<double> Sigma_val(out_fft[ki][0], out_fft[ki][1]);
                comm.map_mat(ki, tau)(0,0) = Sigma_val; 
            }
        }
    }
}
*/

void Hubb_diss::Sigma_Real_spawn(int tstp, mpi_comm &comm) {
//    Sigma_Real_tv_spawn(tstp, comm);
    Sigma_Real_lesret_spawn(tstp, comm);

  MPI_Allreduce(
    MPI_IN_PLACE,          // Tells MPI to read from and write to the recvbuf
    &ellG_,          // Pointer to the underlying contiguous array
    1,          // Total number of elements (L * L)
    MPI_DOUBLE_COMPLEX,    // MPI datatype matching std::complex<double>
    MPI_SUM,               // Summing zeroes with the non-zero gives the non-zero
    MPI_COMM_WORLD         // Replace with your specific communicator if needed
  );

}

void Hubb_diss::Sigma_Real_lesret_spawn(int tstp, mpi_comm &comm) {
    int init_t = comm.my_first_t;
    int last_t = init_t + comm.my_Nt;

    // Each mpi process sets ellG to zero
    ellG_ = 0;

    #pragma omp parallel
    {
        // identify thread local fftw buffers
        int tid = omp_get_thread_num();
        fftw_complex* in_fft = in_thread_vec[2*tid];
        fftw_complex* out_fft = out_thread_vec[2*tid];
        fftw_complex* in_fft_2 = in_thread_vec[2*tid+1];
        fftw_complex* out_fft_2 = out_thread_vec[2*tid+1];

        // each mpi process must do assigned t points and all k points
        #pragma omp for schedule(static)
        for (int t = init_t; t < last_t; t++) {

            std::complex<double> U2 = U_(t,0,0) * U_(tstp,0,0);

            // We first FT G_k to G_r (1D)
            for(int ki = 0; ki < Nk_; ki++) {
                auto G_les = comm.map_les(ki, t)(0,0);
                in_fft[ki][0] = G_les.real();
                in_fft[ki][1] = G_les.imag();
                auto G_ret = comm.map_ret(ki, t)(0,0);
                in_fft_2[ki][0] = G_ret.real();
                in_fft_2[ki][1] = G_ret.imag();
            }

            // Do the fft for both
            fftw_execute_dft(Gk_to_Gr, in_fft, out_fft);
            fftw_execute_dft(Gk_to_Gr, in_fft_2, out_fft_2);

            // FFT normalization factor
            double norm = 1.0 / L_;

            // we have G^<(t,t) so do the HF eval
            if(t == tstp) {
              // ellG_{in}(t) = U(t) \rho_{ii}(t)\delta_{in}
              // \rho_{ii}(t) = -iG^<_{r=0}(t,t) * Iden
              // Fourier transform gives \ellG_{k} = -iG^<_{r=0}(t,t) * U(t)
              int ri = 0;
              ellG_ = -cplx(0.,1.) * cplx(out_fft[ri][0] * norm, out_fft[ri][1] * norm) * U_(tstp,0,0);
            }

            // Evaluate Sigma^<(t,T) =   U^2 G^<_r(t,T) G^<_r(t,T) G^>_{-r}(T,t)
            // Evaluate Sigma^>(T,t) = - U^2 G^>_r(T,t) G^>_r(T,t) G^<_{-r}(t,T)
            //                       +  2U^2 G^<_r(T,t) G^<_r(T,t) G^<_{-r}(t,T)
            //                       = - U^2 G^>_r(T,t) G^>_r(T,t) G^<_{-r}(t,T)
            //                       +  2U^2 G^<_{-r}(t,T)*G^<_{-r}(t,T)*G^<_{-r}(t,T)
            for(int ri = 0; ri < L_; ri++) {
                int iri = (L_ - ri) % L_; // index for -r
                // Convert data in fftw buffers into std::complex
                // G^>_r(T,t) = G^R_r(T,t) + G^<_r(T,t)
                //            = G^R_r(T,t) - G^<_{-r}(t,T)*
                std::complex<double> G_les_r(out_fft[ri][0] * norm, out_fft[ri][1] * norm);
                std::complex<double> G_les_mr(out_fft[iri][0] * norm, out_fft[iri][1] * norm);
                std::complex<double> G_ret_r(out_fft_2[ri][0] * norm, out_fft_2[ri][1] * norm);
                std::complex<double> G_ret_mr(out_fft_2[iri][0] * norm, out_fft_2[iri][1] * norm);
                std::complex<double> G_gtr_r = G_ret_r - std::conj(G_les_mr);
                std::complex<double> G_gtr_mr = G_ret_mr - std::conj(G_les_r);

                std::complex<double> Sigma_les_r = U2 * G_les_r * G_les_r * G_gtr_mr;
                std::complex<double> Sigma_les_mr = U2 * G_les_mr * G_les_mr * G_gtr_r;
                std::complex<double> Sigma_gtr_r = -U2 * G_gtr_r * G_gtr_r * G_les_mr;
                Sigma_gtr_r                     += 2*U2 * std::conj(G_les_mr) * std::conj(G_les_mr) * G_les_mr;

                std::complex<double> Sigma_ret_r = Sigma_gtr_r + std::conj(Sigma_les_mr);
                
                // 3. Assign the real and imaginary parts back into target fftw_complex array
                in_fft[ri][0] = Sigma_les_r.real();
                in_fft[ri][1] = Sigma_les_r.imag();
                in_fft_2[ri][0] = Sigma_ret_r.real();
                in_fft_2[ri][1] = Sigma_ret_r.imag();
            }

            // 4. Transform Sigma(r) -> Sigma(k)
            fftw_execute_dft(Sigmar_to_Sigmak, in_fft, out_fft);
            fftw_execute_dft(Sigmar_to_Sigmak, in_fft_2, out_fft_2);

            // 5. Map the full k-space grid back to your k-adapted storage (contiguous)
            for(int ki = 0; ki < Nk_; ki++) {
                std::complex<double> Sigma_les_val(out_fft[ki][0], out_fft[ki][1]);
                std::complex<double> Sigma_ret_val(out_fft_2[ki][0], out_fft_2[ki][1]);
        
                // Save to communicator
                comm.map_les(ki, t)(0,0) = Sigma_les_val; 
                comm.map_ret(ki, t)(0,0) = Sigma_ret_val; 
            }
        }
    }
}

/*
void Hubb_diss::Sigma_Real_tv_spawn(int tstp, mpi_comm &comm) {
    int init_tau = comm.my_first_tau;
    int last_tau = init_tau + comm.my_Ntau;

    #pragma omp parallel
    {
        // identify thread local fftw buffers
        int tid = omp_get_thread_num();
        int base = 2 * tid;
        fftw_complex* in_fft = in_thread_vec[2*tid];
        fftw_complex* out_fft = out_thread_vec[2*tid];
        fftw_complex* in_fft_2 = in_thread_vec[2*tid+1];
        fftw_complex* out_fft_2 = out_thread_vec[2*tid+1];

        // each mpi process must do assigned tau points and all k points
        #pragma omp for schedule(static)
        for (int tau = init_tau; tau < last_tau; tau++) {

            // We first FT G_k to G_r (1D)
            for(int ki = 0; ki < Nk_; ki++) {
                auto G_val = comm.map_tv(ki, tau)(0,0);
                in_fft[ki][0] = G_val.real();
                in_fft[ki][1] = G_val.imag();
                auto G_valrev = comm.map_tv_rev(ki, tau)(0,0);
                in_fft_2[ki][0] = G_valrev.real();
                in_fft_2[ki][1] = G_valrev.imag();
            }

            // Do the fft for both
            fftw_execute_dft(Gk_to_Gr, in_fft, out_fft);
            fftw_execute_dft(Gk_to_Gr, in_fft_2, out_fft_2);

            // FFT normalization factor
            double norm = 1.0 / L_;

            // Evaluate Sigma^\rceil_r(\tau) = U^2 G^\rceil_r(\tau) G_\rceil_r(\tau) G^\lceil_{-r}(\tau)
            //                               = -\xi  U^2 G^\rceil_r(\tau) G_\rceil_r(\tau) G^\rceil_{r}(\beta-\tau)*
            for(int ri = 0; ri < L_; ri++) {
                // Convert data in fftw buffers into std::complex
                std::complex<double> G_r(out_fft[ri][0] * norm, out_fft[ri][1] * norm);
                std::complex<double> Grev_r(out_fft_2[ri][0] * norm, out_fft_2[ri][1] * norm);
                std::complex<double> Sigma = U_ * U_ * G_r * G_r * std::conj(Grev_r);

                // 3. Assign the real and imaginary parts back into target fftw_complex array
                in_fft[ri][0] = Sigma.real();
                in_fft[ri][1] = Sigma.imag();
            }
            
            // 4. Transform Sigma(r) -> Sigma(k)
            fftw_execute_dft(Sigmar_to_Sigmak, in_fft, out_fft);

            // 5. Map the full k-space grid back to your k-adapted storage (contiguous)
            for(int ki = 0; ki < Nk_; ki++) {
              std::complex<double> Sigma_val(out_fft[ki][0], out_fft[ki][1]);
              comm.map_tv(ki, tau)(0,0) = Sigma_val; 
            }
        }
    }
}
*/

// Orchestrator
void Hubb_diss::Sigma_nospawn(int tstp, 
                          mpi_comm &comm, 
                          std::vector<std::reference_wrapper<h_nessi::herm_matrix_hodlr>> &Grefs, 
                          std::vector<std::reference_wrapper<h_nessi::herm_matrix_hodlr>> &Srefs, 
                          h_nessi::dlr_info &dlr) 
{

    #pragma omp barrier

    // 1. Fetch remote G data into comm buffers
    comm.mpi_get_and_comm_nospawn(tstp, Grefs, dlr);

    #pragma omp barrier

    // 2. Compute self-energy
//    if (tstp == -1) {
//        Sigma_Mat_nospawn(comm);
//    } else {
        Sigma_Real_nospawn(tstp, comm);
//    }

    #pragma omp barrier

    // 3. Sync and write back S results
    comm.mpi_comm_and_set_nospawn(tstp, Srefs);

    #pragma omp barrier

    if(omp_get_thread_num() == 0 ) {
      MPI_Allreduce(
        MPI_IN_PLACE,          // Tells MPI to read from and write to the recvbuf
        &ellG_,          // Pointer to the underlying contiguous array
        1,          // Total number of elements (L * L)
        MPI_DOUBLE_COMPLEX,    // MPI datatype matching std::complex<double>
        MPI_SUM,               // Summing zeroes with the non-zero gives the non-zero
        MPI_COMM_WORLD         // Replace with your specific communicator if needed
      );
    }

    #pragma omp barrier
}

/*
void Hubb_diss::Sigma_Mat_nospawn(mpi_comm &comm)
{
    // Fetch OpenMP thread details (assuming we are already in a parallel region)
    int thread_id = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    int init_tau = comm.my_first_tau;
    int my_Ntau = comm.my_Ntau;

    // Identify thread local fftw buffers
    fftw_complex* in_fft = in_thread_vec[2 * thread_id];
    fftw_complex* out_fft = out_thread_vec[2 * thread_id];
    fftw_complex* in_fft_2 = in_thread_vec[2 * thread_id + 1];
    fftw_complex* out_fft_2 = out_thread_vec[2 * thread_id + 1];

    // Get relative local indices by passing 0 as the starting offset
    std::array<int,3> indxs = get_my_index(thread_id, nthreads, my_Ntau, 0);
    
    // Shift the relative bounds by the MPI process's starting tau
    int th_my_init_tau = indxs[0] + init_tau;
    int th_my_end_tau  = indxs[1] + init_tau;
    int th_my_Ntau     = indxs[2];

    if (th_my_Ntau != 0) {
        // Iterate over the thread's assigned chunk using tau directly
        for (int tau = th_my_init_tau; tau <= th_my_end_tau; tau++) {

            // We first FT G_k to G_r (1D)
            // Read G^M and G^Mrev into in_fft using contiguous k_points [0..L_)
            for(int ki = 0; ki < Nk_; ki++) {
                auto G_val = comm.map_mat(ki, tau)(0,0);
                in_fft[ki][0] = G_val.real();
                in_fft[ki][1] = G_val.imag();
                auto G_valrev = comm.map_mat_rev(ki, tau)(0,0);
                in_fft_2[ki][0] = G_valrev.real();
                in_fft_2[ki][1] = G_valrev.imag();
            }

            // Do the fft for both
            fftw_execute_dft(Gk_to_Gr, in_fft, out_fft);
            fftw_execute_dft(Gk_to_Gr, in_fft_2, out_fft_2);

            // FFT normalization factor
            double norm = 1.0 / L_;

            // Evaluate Sigma^M_r(\tau) = -U^2 G^M_r(\tau) G^M_r(\tau) G^M_{-r}(-\tau)
            //                             U^2 G^M_r(\tau) G^M_r(\tau) G^M_{-r}(\beta-\tau)
            for(int ri = 0; ri < L_; ri++) {
                int iri = (L_ - ri) % L_; // index for -r
                // Convert data in fftw buffers into std::complex
                std::complex<double> G_r(out_fft[ri][0] * norm, out_fft[ri][1] * norm);
                std::complex<double> Grev_minusr(out_fft_2[iri][0] * norm, out_fft_2[iri][1] * norm);
                std::complex<double> Sigma = U_ * U_ * G_r * G_r * Grev_minusr;
                in_fft[ri][0] = Sigma.real();
                in_fft[ri][1] = Sigma.imag();
            }

            // 4. Transform Sigma(r) -> Sigma(k)
            fftw_execute_dft(Sigmar_to_Sigmak, in_fft, out_fft);

            // 5. Map the full k-space grid back to your k-adapted storage (contiguous)
            for(int ki = 0; ki < Nk_; ki++) {
                std::complex<double> Sigma_val(out_fft[ki][0], out_fft[ki][1]);
                comm.map_mat(ki, tau)(0,0) = Sigma_val; 
            }
        }
    }
}
*/

void Hubb_diss::Sigma_Real_nospawn(int tstp, mpi_comm &comm) {
//    Sigma_Real_tv_nospawn(tstp, comm);
//    #pragma omp barrier
    Sigma_Real_lesret_nospawn(tstp, comm);
}

/*
void Hubb_diss::Sigma_Real_tv_nospawn(int tstp, mpi_comm &comm) {
    // Fetch OpenMP thread details (assuming we are already in a parallel region)
    int thread_id = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    int init_tau = comm.my_first_tau;
    int my_Ntau = comm.my_Ntau;

    // identify thread local fftw buffers
    fftw_complex* in_fft = in_thread_vec[2 * thread_id];
    fftw_complex* out_fft = out_thread_vec[2 * thread_id];
    fftw_complex* in_fft_2 = in_thread_vec[2 * thread_id + 1];
    fftw_complex* out_fft_2 = out_thread_vec[2 * thread_id + 1];

    // Get relative local indices by passing 0 as the starting offset
    std::array<int,3> indxs = get_my_index(thread_id, nthreads, my_Ntau, 0);
    
    // Shift the relative bounds by the MPI process's starting tau
    int th_my_init_tau = indxs[0] + init_tau;
    int th_my_end_tau  = indxs[1] + init_tau;
    int th_my_Ntau     = indxs[2];

    if (th_my_Ntau != 0) {
        // Iterate over the thread's assigned chunk using tau directly
        for (int tau = th_my_init_tau; tau <= th_my_end_tau; tau++) {

            // We first FT G_k to G_r (1D)
            for(int ki = 0; ki < Nk_; ki++) {
                auto G_val = comm.map_tv(ki, tau)(0,0);
                in_fft[ki][0] = G_val.real();
                in_fft[ki][1] = G_val.imag();
                auto G_valrev = comm.map_tv_rev(ki, tau)(0,0);
                in_fft_2[ki][0] = G_valrev.real();
                in_fft_2[ki][1] = G_valrev.imag();
            }

            // Do the fft for both
            fftw_execute_dft(Gk_to_Gr, in_fft, out_fft);
            fftw_execute_dft(Gk_to_Gr, in_fft_2, out_fft_2);

            // FFT normalization factor
            double norm = 1.0 / L_;

            // Evaluate Sigma^\rceil_r(\tau) = U^2 G^\rceil_r(\tau) G_\rceil_r(\tau) G^\lceil_{-r}(\tau)
            //                               = -\xi  U^2 G^\rceil_r(\tau) G_\rceil_r(\tau) G^\rceil_{r}(\beta-\tau)*
            for(int ri = 0; ri < L_; ri++) {
                // Convert data in fftw buffers into std::complex
                std::complex<double> G_r(out_fft[ri][0] * norm, out_fft[ri][1] * norm);
                std::complex<double> Grev_r(out_fft_2[ri][0] * norm, out_fft_2[ri][1] * norm);
                std::complex<double> Sigma = U_ * U_ * G_r * G_r * std::conj(Grev_r);

                // 3. Assign the real and imaginary parts back into target fftw_complex array
                in_fft[ri][0] = Sigma.real();
                in_fft[ri][1] = Sigma.imag();
            }
            
            // 4. Transform Sigma(r) -> Sigma(k)
            fftw_execute_dft(Sigmar_to_Sigmak, in_fft, out_fft);

            // 5. Map the full k-space grid back to your k-adapted storage (contiguous)
            for(int ki = 0; ki < Nk_; ki++) {
              std::complex<double> Sigma_val(out_fft[ki][0], out_fft[ki][1]);
              comm.map_tv(ki, tau)(0,0) = Sigma_val;
            }
        }
    }
}
*/


void Hubb_diss::Sigma_Real_lesret_nospawn(int tstp, mpi_comm &comm) {
    // Fetch OpenMP thread details (assuming we are already in a parallel region)
    int thread_id = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    int init_t = comm.my_first_t;
    int my_Nt = comm.my_Nt;

    // if mpi process is not assigned any time points, we just set ellG_=0
    if(my_Nt == 0) ellG_ = 0;

    // identify thread local fftw buffers
    fftw_complex* in_fft = in_thread_vec[2 * thread_id];
    fftw_complex* out_fft = out_thread_vec[2 * thread_id];
    fftw_complex* in_fft_2 = in_thread_vec[2 * thread_id + 1];
    fftw_complex* out_fft_2 = out_thread_vec[2 * thread_id + 1];

    // Get relative local indices by passing 0 as the starting offset
    std::array<int,3> indxs = get_my_index(thread_id, nthreads, my_Nt, 0);
    
    // Shift the relative bounds by the MPI process's starting t
    int th_my_init_t = indxs[0] + init_t;
    int th_my_end_t  = indxs[1] + init_t;
    int th_my_Nt     = indxs[2];

    if (th_my_Nt != 0) {
        // Iterate over the thread's assigned chunk using t directly
        for (int t = th_my_init_t; t <= th_my_end_t; t++) {

            std::complex<double> U2 = U_(t,0,0) * U_(tstp,0,0);

            // We first FT G_k to G_r (1D)
            for(int ki = 0; ki < Nk_; ki++) {
                auto G_les = comm.map_les(ki, t)(0,0);
                in_fft[ki][0] = G_les.real();
                in_fft[ki][1] = G_les.imag();
                auto G_ret = comm.map_ret(ki, t)(0,0);
                in_fft_2[ki][0] = G_ret.real();
                in_fft_2[ki][1] = G_ret.imag();
            }

            // Do the fft for both
            fftw_execute_dft(Gk_to_Gr, in_fft, out_fft);
            fftw_execute_dft(Gk_to_Gr, in_fft_2, out_fft_2);

            // Each mpi process sets ellG to zero
            // Here, only the thread assigned to the last time point is responsible
            if(t == init_t+my_Nt) ellG_ = 0;

            // we have G^<(t,t) so do the HF eval
            if(t == tstp) {
              // ellG_{in}(t) = U(t) \rho_{ii}(t)\delta_{in}
              // \rho_{ii}(t) = -iG^<_{r=0}(t,t) * Iden
              int ri = 0;
              ellG_ = -cplx(0.,1.) * cplx(out_fft[ri][0] * norm, out_fft[ri][1] * norm) * U_(tstp,0,0);
            }

            // FFT normalization factor
            double norm = 1.0 / L_;

            // Evaluate Sigma^<(t,T) =   U^2 G^<_r(t,T) G^<_r(t,T) G^>_{-r}(T,t)
            // Evaluate Sigma^>(T,t) = - U^2 G^>_r(T,t) G^>_r(T,t) G^<_{-r}(t,T)
            //                       +  2U^2 G^<_r(T,t) G^<_r(T,t) G^<_{-r}(t,T)
            //                       = - U^2 G^>_r(T,t) G^>_r(T,t) G^<_{-r}(t,T)
            //                       +  2U^2 G^<_{-r}(t,T)*G^<_{-r}(t,T)*G^<_{-r}(t,T)
            for(int ri = 0; ri < L_; ri++) {
                int iri = (L_ - ri) % L_; // index for -r
                // Convert data in fftw buffers into std::complex
                // G^>_r(T,t) = G^R_r(T,t) + G^<_r(T,t)
                //            = G^R_r(T,t) - G^<_{-r}(t,T)*
                std::complex<double> G_les_r(out_fft[ri][0] * norm, out_fft[ri][1] * norm);
                std::complex<double> G_les_mr(out_fft[iri][0] * norm, out_fft[iri][1] * norm);
                std::complex<double> G_ret_r(out_fft_2[ri][0] * norm, out_fft_2[ri][1] * norm);
                std::complex<double> G_ret_mr(out_fft_2[iri][0] * norm, out_fft_2[iri][1] * norm);
                std::complex<double> G_gtr_r = G_ret_r - std::conj(G_les_mr);
                std::complex<double> G_gtr_mr = G_ret_mr - std::conj(G_les_r);

                std::complex<double> Sigma_les_r = U2 * G_les_r * G_les_r * G_gtr_mr;
                std::complex<double> Sigma_les_mr = U2 * G_les_mr * G_les_mr * G_gtr_r;
                std::complex<double> Sigma_gtr_r = -U2 * G_gtr_r * G_gtr_r * G_les_mr;
                Sigma_gtr_r                     += 2*U2 * std::conj(G_les_mr) * std::conj(G_les_mr) * G_les_mr;

                std::complex<double> Sigma_ret_r = Sigma_gtr_r + std::conj(Sigma_les_mr);
                
                // 3. Assign the real and imaginary parts back into target fftw_complex array
                in_fft[ri][0] = Sigma_les_r.real();
                in_fft[ri][1] = Sigma_les_r.imag();
                in_fft_2[ri][0] = Sigma_ret_r.real();
                in_fft_2[ri][1] = Sigma_ret_r.imag();
            }

            // 4. Transform Sigma(r) -> Sigma(k)
            fftw_execute_dft(Sigmar_to_Sigmak, in_fft, out_fft);
            fftw_execute_dft(Sigmar_to_Sigmak, in_fft_2, out_fft_2);

            // 5. Map the full k-space grid back to your k-adapted storage (contiguous)
            for(int ki = 0; ki < Nk_; ki++) {
                std::complex<double> Sigma_les_val(out_fft[ki][0], out_fft[ki][1]);
                std::complex<double> Sigma_ret_val(out_fft_2[ki][0], out_fft_2[ki][1]);
        
                // Save to communicator
                comm.map_les(ki, t)(0,0) = Sigma_les_val;
                comm.map_ret(ki, t)(0,0) = Sigma_ret_val;
            }
        }
    }
}
