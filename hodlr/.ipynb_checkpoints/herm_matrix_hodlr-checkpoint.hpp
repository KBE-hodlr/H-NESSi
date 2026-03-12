#pragma once

#include "block.hpp"
#include "blocks_keldysh.hpp"
#include <hdf5.h>
#include <chrono>
#include "cntr/hdf5/hdf5_interface.hpp"
#include "integration.hpp"

extern "C"
{
  #include "dlr_c/dlr_c.h"
}

namespace hodlr {

template <typename T> class function;
template <typename T> class herm_matrix_timestep;

/** \brief <b> Class `herm_matrix_hodlr` for two-time contour objects \f$ C(t,t') \f$
 * compressed with the hodlr structure and with hermitian symmetry .</b>
 *
 * <!-- ====== DOCUMENTATION ====== -->
 *
 *  \par Purpose
 * <!-- ========= -->
 *
 *  This class contains the data structures for representing bosonic (\f$\eta = +1 \f$)
 *  or fermionic (\f$\eta = -1 \f$) or two-time contour
 *  functions \f$ C(t,t') \f$ in the hierarchical off-diagonal low-rank structure.
 *  The class `herm_matrix_hodlr` stores TODO
 *
 * The contour function \f$ C(t,t') \f$ can be of scalar type or matrix-valued.
 * Square-matrix-type contour functions are fully supported, while non-square-
 * matrix contour functions are under development.
 *
 *  If `nt = 0`, only the Matsubara component is stored.
 *
 */
class herm_matrix_hodlr{
  public:

    /* construction, destruction */
    herm_matrix_hodlr();
    ~herm_matrix_hodlr();
    herm_matrix_hodlr(int nt,int r,int nlvl,double svdtol,int size1,int size2,int sig,int k,int shape=0);
    herm_matrix_hodlr(GREEN &G,int nlvl,double svdtol, int r = 1,int shape=0);
    herm_matrix_hodlr &operator=(herm_matrix_hodlr &g);
    int getbkl_indices(void);
    int getbkl_indices_h(void);
    void init_shape(int nt,int nlvl);
    void init_shape_h(int nt,int nlvl);
    int time2direct_col(int t1,int t2);
    int time2block(int t1,int t2);

    int nlvl(void) const {return nlvl_;};
    int nt(void) const {return nt_;};
    int ntau(void) const {return ntau_;};
    int r(void) const {return r_;};
    int size(void) const {return size1_;};
    int size1(void) const {return size1_;};
    int size2(void) const {return size2_;};
    int shape(void) const {return shape_;};
    int sig(void) const {return sig_;};
    int nbox(void) const {return nbox_;};
    int ndir(void) const {return ndir_;};
    int tstpmk(void) const {return tstpmk_;};
    int maxdir(void) const {return maxdir_;};
    void set_tstpmk(int tstpmk) {tstpmk_=tstpmk;};
    int k(void) const {return k_;};
    
    ret_blocks &ret(void) {return ret_;};
    les_blocks &les(void)  {return les_;};
    tv_blocks &tv(void)  {return tv_;};
    double *mat(void) const {return mat_;};
    double *GMConvTens(void) {return GMConvTensMatrix_.data();};
    DMatrix &GMConvTensMatrix(void) {return GMConvTensMatrix_;};

    void set_tstp_zero(int tstp);
    
    std::vector<cplx> &curr_timestep_ret(void)  {return curr_timestep_ret_;}
    
    cplx *curr_timestep_ret_ptr(int t, int tp) {
      // std::cout << "Curr " << t << " " << tp << " " << tstpmk_ << std::endl;
      assert(t <= tstpmk_+k_);
      // assert(t >= tstpmk_);
      int index = t%(k_+1);
      // std::cout << "Curr " << t << " " << index << std::endl;
      return curr_timestep_ret_.data() + index * (nt_+1) * size1_ * size2_ + tp * size1_ * size2_;
    }

    int curr_timestep_ret_offset(int t, int tp) {
      assert(t <= tstpmk_+k_);
      assert(t >= tstpmk_);
      int index = t%(k_+1);
      return index * (nt_+1) * size1_ * size2_ + tp * size1_ * size2_;
    }


    cplx *curr_timestep_ret_ptr(void) {
      return curr_timestep_ret_.data() + tstpmk_%(k_+1) * (nt_+1) * size1_ * size2_;
    }
    cplx *curr_timestep_ret_ptr_raw(void) {
      return curr_timestep_ret_.data();
    }
    std::vector<cplx> &curr_timestep_les(void)  {return curr_timestep_les_;}

    cplx *curr_timestep_les_ptr(int t, int tp) {
      assert(tp <= tstpmk_+k_);
      // assert(tp >= tstpmk_);
      int index = tp%(k_+1);
      return curr_timestep_les_.data() + index * (nt_ +1) * size1_ * size2_ + t * size1_ * size2_;
    }

    int curr_timestep_les_offset(int t, int tp) {
      assert(tp <= tstpmk_+k_);
      assert(tp >= tstpmk_);
      int index = tp%(k_+1);
      return index * (nt_ +1) * size1_ * size2_ + t * size1_ * size2_;
    }

    cplx *curr_timestep_les_ptr(void) {
      return curr_timestep_les_.data() + tstpmk_%(k_+1) * (nt_+1) * size1_ * size2_;
    }
    cplx *curr_timestep_les_ptr_raw(void) {
      return curr_timestep_les_.data();
    }



    // SVD stuff
    double mgs(const ZMatrix &V,const ZRowVector &b,const ZRowVector &x_,const ZRowVector &Vb_);
    void updatetsvd(hodlr::block &B,const ZRowVector &row,int row_cur);
    void update_blocks(Integration::Integrator &I);


    double svdtol(void) const {return svdtol_;};
    std::vector<int> &blkr1(void)  {return blkr1_;};
    std::vector<int> &blkr2(void)  {return blkr2_;};
    std::vector<int> &blkc1(void)  {return blkc1_;};
    std::vector<int> &blkc2(void)  {return blkc2_;};
    std::vector<int> &blklevel(void)  {return blklevel_;};
    std::vector<int> &blkdirheight(void)  {return blkdirheight_;};
    std::vector<int> &blkndirstart(void)  {return blkndirstart_;};
    int built_blocks(void) const {return built_blocks_;};
    int blkr1(int blk) {if(blk==-1) return 0;
                        else return blkr1_[blk];};
    int blkr2(int blk) {return blkr2_[blk];};
    int blkc1(int blk) {return blkc1_[blk];};
    int blkc2(int blk) {return blkc2_[blk];};
    std::vector<int> &c1_dir(void)  {return c1_dir_;};
    std::vector<int> &r2_dir(void)  {return r2_dir_;};
    int c1_dir(int tstp) {return c1_dir_[tstp];};
    std::vector<int> &mapdirc_r(void)  { return mapdirc_r_;};
    std::vector<int> &mapdirc_c(void)  { return mapdirc_c_;};
    std::vector<int> &mapdirr_c(void)  { return mapdirr_c_;};
    std::vector<int> &mapdirr_r(void)  { return mapdirr_r_;};
    int r2_dir(int tstp) {return r2_dir_[tstp];};
    int ntri(int tstp) {return ntri_[tstp];};
    int tstpmk(void) {return tstpmk_;};
    int blklen(int blkindex) {return blklen_[blkindex];};
    std::vector<int> &blklen(void)  {return blklen_;};
    std::vector<int> &ntri(void)  {return ntri_;};
    void print_memory_usage(void);
    void set_memory_usage(void);
    
    // Get
    void get_ret(int t1,int t2,cplx *M);
    void get_les(int t1,int t2,cplx *M, bool print=false);
    void get_tv(int t1,int t2,cplx *M);
    void get_tv_trans(int t1,int t2,cplx *M);
    void get_ret(int t1,int t2,ZMatrix &M);
    void get_les(int t1,int t2,ZMatrix &M);
    void get_tv(int t1,int t2,ZMatrix &M);
    void get_tv_trans(int t1,int t2,ZMatrix &M);
    void get_mat(int i,DMatrix &M);
    void get_mat_reversed(double *dest, double *dlrit2itr);
    void get_tv_reversed(int tstp, cplx *dest, double *dlrit2itr);
    void get_vt(int tstp, cplx *dest, double *dlrit2itr);
    void get_mat_tau(double tau,double beta,double *it2cf,int *it2cfp, double *dlrrf, double *M);
    void get_mat_tau(double tau,double beta,double *it2cf,int *it2cfp, double *dlrrf, cplx *M);
    void get_mat_tau(double tau,double beta,double *it2cf,int *it2cfp, double *dlrrf, DMatrix &M);
    void get_tv_tau(int tstp, double tau,double beta,double *it2cf,int *it2cfp, double *dlrrf, double *dlrit, ZMatrix &M);
    void get_tv_tau(int tstp, double tau,double beta,double *it2cf,int *it2cfp, double *dlrrf, double *dlrit, cplx *M);
    void get_ret_curr(int t,int tp,cplx *M);
    void get_les_curr(int t,int tp,cplx *M);
    void get_ret_curr(int t,int tp,ZMatrix &M);
    void get_les_curr(int t,int tp,ZMatrix &M);
    void density_matrix(int tstp,double *it2cf,int *it2cfp, double *dlrrf, DMatrix &M);
    void density_matrix(int tstp,double *it2cf,int *it2cfp, double *dlrrf, cplx *res);
    // Set 
    void set_mat(int i,DMatrix &M);
    void set_tv(int t1, int t2, ZMatrix &M);
    void set_tv(int t1, int t2, cplx* M);

    // Get/set for current timesteps
    void get_timestep_curr(int tstp,herm_matrix_hodlr &G);
    void get_timestep_curr(herm_matrix_hodlr &G);
    void set_timestep_curr(int tstp,herm_matrix_hodlr &G);
    void set_timestep_curr(herm_matrix_hodlr &G);

    double *matptr(int i);
    cplx *tvptr(int t, int tau);
    cplx *tvptr_trans(int t, int tau);
    int tv_offset(int t, int tau) {assert(t <= nt_ && tau < r_); return t*r_*size1_*size2_ + tau*size1_*size2_;}

    std::complex<double>* retptr_col(int t, int t1); // Directly stored data in col order

    void get_timestep_curr(int tstp, cntr::herm_matrix_timestep<double> &G);
    void set_timestep_curr(int tstp, cntr::herm_matrix_timestep<double> &G);
    void set_timestep_curr(int tstp, cntr::herm_matrix<double> &G);

    // Algebraic operations on timesteps
    void incr_timestep_curr(int tstp,herm_matrix_hodlr &G,cplx alpha);
    void incr_timestep_curr(herm_matrix_hodlr &G,cplx alpha);
    void smul_curr(int tstp,cplx alpha);
    void smul_curr(cplx alpha);

    // Mixed component
    void initGMConvTensor(double *it2itr, double *it2cf, int *it2cfp, double *phi);

    // Write to hdf5
    void write_to_hdf5(hid_t group_id,bool storeSVD);
    void write_to_hdf5(const char *filename, bool storeSVD);

    void write_to_hdf5(h5e::File &out, std::string label, bool inc_data = true, bool inc_geometry = true);
    void write_rho_to_hdf5(h5e::File &out, double *it2cf, int *it2cfp, double *dlrrf, std::string label = "rho");
    void write_GL0_to_hdf5(h5e::File &out, std::string label);
    void write_GR0_to_hdf5(h5e::File &out, std::string label);
    void write_rank_to_hdf5(h5e::File &out, std::string label);
    void write_mat_to_hdf5(h5e::File &out);
    void write_mat_to_hdf5(h5e::File &out, std::string label);

    void write_curr_to_hdf5(hid_t group_id,int tstp);
    void write_curr_to_hdf5(std::string &filename,int tstp);

// // MPI UTILS
// #if CNTR_USE_MPI == 1
//     void Reduce_timestep(int tstp, int root);
//     void Bcast_timestep(int tstp, int root);
//     void Send_timestep(int tstp, int dest, int tag);
//     void Recv_timestep(int tstp, int root, int tag);
// #endif
//   private:
//     int les_offset(int t, int t1) const;
//     int ret_offset(int t, int t1) const;
//     int tv_offset(int t, int tau) const;
//     int mat_offset(int tau) const;

    int len_les_dir_square() { return len_les_dir_square_; }
    int t_to_dirlvl(int t) { return t_to_dirlvl_[t]; }
    std::vector<cplx> &les_dir_square() { return les_dir_square_; }
    std::vector<int> &les_dir_square_first_index() { return les_dir_square_first_index_; }
    std::vector<cplx> &les_left_edge() { return les_left_edge_; }


  private:
    bool LU_built_;
    DMatrix it2cf_tmp_;
    Eigen::PartialPivLU<DMatrix> LU_;
    double *Gijc_it_;
    double *res_;
// ######################### GEOMETRY ################################
    /// @private
    /** \brief <b> First rows in blocks. [Jason blki1] </b> */
    std::vector<int> t_to_dirlvl_; 
    /// @private
    /** \brief <b> First rows in blocks. [Jason blki1] </b> */
    std::vector<int> blkr1_; 
    /// @private
    /** \brief <b> Last rows in blocks [Jason blki2] </b> */
    std::vector<int> blkr2_; 
    /// @private
    /** \brief <b> First columns in blocks [Jason blkj1] </b> */
    std::vector<int> blkc1_; 
    /// @private
    /** \brief <b> Last columns in blocks [Jason blkj2] </b> */
    std::vector<int> blkc2_; 
    /// @private
    /** \brief <b> Levels of blocks </b> */
    std::vector<int> blklevel_; 
    /// @private
    /** \brief <b> Height of directly stored values at the of the block </b> */
    std::vector<int> blkdirheight_; 
    /// @private
    /** \brief <b> Levels of blocks </b> */
    std::vector<int> blkndirstart_;
    // /// @private
    // /** \brief <b> Column index of first directly-stored entry in each row. Jason's j1dir  </b> */
    std::vector<int> c1_dir_;
    // /// @private
    // /** \brief <b> Row index of last directly-stored entry in each column   </b> */
    std::vector<int> r2_dir_;
    std::vector<int> mapdirc_r_;
    std::vector<int> mapdirc_c_;
    std::vector<int> mapdirr_c_;
    std::vector<int> mapdirr_r_;


    // /// @private
    // /** \brief <b>  Number of directly stored indices in row  </b> */
    std::vector<int> ntri_;
    // /// @private
    // /** \brief <b>  rows in each block  </b> */
    std::vector<int> blklen_;
    // /// @private
    // /** \brief <b>  rows in each block  </b> */
    std::vector<int> les_dir_square_first_index_;
    int len_les_dir_square_;
// ######################### GEOMETRY ################################

// #########################   DATA   ################################
    /// @private
    /** \brief <b> Retarded blocks  </b> */
    ret_blocks ret_;
    /// @private
    /** \brief <b> Lesser blocks   </b> */
    les_blocks les_;
    /// @private
    /** \brief <b> tv blocks   </b> */
    tv_blocks tv_;
    /// @private
    /** \brief <b> mat blocks   </b> */
    double *mat_;

    /// @private
    /** \brief <b> Convolution tensor for mixed component </b> */
    DMatrix GMConvTensMatrix_;

    /** \brief <b> ret directly stored slice of the last k timesteps   </b> */
    std::vector<cplx> curr_timestep_ret_;
    /** \brief <b> les directly stored slice of the last k timesteps   </b> */
    std::vector<cplx> curr_timestep_les_;

    /// @private
    /** \brief <b> lesser component in squares along diagonal </b> */
    std::vector<cplx> les_dir_square_;
    /// @private
    /** \brief <b> Stores les gf along left edge <\b> */
    std::vector<cplx> les_left_edge_;
// #########################   DATA   ################################


//  ################### SCALARS #####################################
    /** \brief <b> integration order </b> */
    int k_;
    /** \brief <b> counts the number of blocks that have been initialized </b> */
    int built_blocks_;
    /** \brief <b> current timestepk</b> */
    // int tstp_;
    /** \brief <b> current timestep minus k</b> */
    int tstpmk_;
    /// @private
    /** \brief <b> Maximum number of directly-stored entries in a row.</b> */
    int maxdir_;
    /// @private
    /** \brief <b> Number of levels in hodlr.</b> */
    int nlvl_; 
    /// @private
    /** \brief <b> Number of boxes.</b> */
    int nbox_; 
    // /// @private
    /** \brief <b> SVD tolerance.</b> */
    double svdtol_; //HUGO: This info is doubled among box and herm_matrix_hodlr. But it is natural to both of them? any way out ?
    /// @private
    /** \brief <b> Maximum number of time steps.</b> */
    int nt_; 
    /// @private
    /** \brief <b> Number of initially proposed time points on the Matsubara axis.</b> */
    int ntau_;
    /// @private
    /** \brief <b> Number of the DLR time grids on the Matsubara axis.</b> */
    int r_;
    // /// @private
    // /** \brief <b> Number of directly stored enties.</b> */
    int ndir_;
    /// @private
    /** \brief <b> Shape of hierarchical structure.</b> */
    /** \brief <b> hodlr = 0, h=1 </b> */
    int shape_;
    /// @private
    /** \brief <b> Compressed memory usage ret</b> */
    double cr_;
    /// @private
    /** \brief <b> full memory usage ret</b> */
    double fr_;
    /// @private
    /** \brief <b> Compressed memory usage les</b> */
    double cl_;
    /// @private
    /** \brief <b> full memory usage les</b> */
    double fl_;
    /// @private
    /** \brief <b> Column dimension in the orbital space.</b> */
    int size1_; 
    /// @private
    /** \brief <b> Row dimension in the orbital space.</b> */
    int size2_; 
    /// @private
    /** \brief <b> Bose = +1, Fermi =-1. </b> */
    int sig_; // Bose = +1, Fermi =-1
};




double distance_norm2(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2, double *it2cf, int *it2cfp, double *dlrrf);
double distance_norm2_curr(int tstp, herm_matrix_hodlr &g1, herm_matrix_hodlr &g2);
double distance_norm2_curr(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix_timestep<double> &g2,double *it2cf, int *it2cfp, double *dlrrf,bool dlr);
double distance_norm2_curr(herm_matrix_hodlr &g1, herm_matrix_hodlr &g2);
// MAT
double distance_norm2_mat(herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2,double *it2cf, int *it2cfp, double *dlrrf);
double distance_norm2_mat(herm_matrix_hodlr &g1, cntr::herm_matrix_timestep<double> &g2,double *it2cf, int *it2cfp, double *dlrrf,bool dlr);
// RET
double distance_norm2_curr_ret(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix_timestep<double> &g2);
double distance_norm2_curr_ret(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2);
double distance_norm2_ret(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2);
double distance_norm2_curr_ret(int tstp, herm_matrix_hodlr &g1, herm_matrix_hodlr &g2);
// LES
double distance_norm2_curr_les(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix_timestep<double> &g2);
double distance_norm2_curr_les(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2);
double distance_norm2_les(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2);
double distance_norm2_curr_les(int tstp, herm_matrix_hodlr &g1, herm_matrix_hodlr &g2);
// TV
double distance_norm2_tv(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2,double *it2cf, int *it2cfp, double *dlrrf);
double distance_norm2_tv(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix_timestep<double> &g2,double *it2cf, int *it2cfp, double *dlrrf,bool dlr);
double distance_norm2_curr_tv(int tstp, herm_matrix_hodlr &g1, herm_matrix_hodlr &g2);

// BUBBLE1:  C_{c1,c2}(t1,t2) = ii * A_{a1,a2}(t1,t2) * B_{b2,b1}(t2,t1)
// void Bubble1_curr(int tstp, herm_matrix_hodlr &C, int c1, int c2, herm_matrix_hodlr &A, herm_matrix_hodlr &Acc, int a1, int a2, herm_matrix_hodlr &B,
//              herm_matrix_hodlr &Bcc, int b1, int b2,double *it2itr);


// void Bubble2_curr(int tstp, herm_matrix_hodlr &C, int c1, int c2, herm_matrix_hodlr &A, herm_matrix_hodlr &Acc, int a1, int a2, herm_matrix_hodlr &B,
//              herm_matrix_hodlr &Bcc, int b1, int b2,double *it2itr);


// With pointers

void Bubble1_curr(int tstp, herm_matrix_hodlr &C, int c1, int c2, herm_matrix_hodlr &A, herm_matrix_hodlr &Acc, int a1, int a2, herm_matrix_hodlr &B,
             herm_matrix_hodlr &Bcc, int b1, int b2,double *it2itr);

void Bubble1_curr(int tstp, cntr::herm_matrix_timestep<double> &C, int c1, int c2, herm_matrix_hodlr &A, herm_matrix_hodlr &Acc, int a1, int a2, herm_matrix_hodlr &B,
             herm_matrix_hodlr &Bcc, int b1, int b2,double *it2itr);

void Bubble1_curr(int tstp, herm_matrix_hodlr &C, int c1, int c2, herm_matrix_timestep<double> &A, herm_matrix_timestep<double> &Acc, int a1, int a2, herm_matrix_hodlr &B,
             herm_matrix_hodlr &Bcc, int b1, int b2,double *it2itr);

void get_bubble_1_mat(double *cmat,int sizec1,int sizec2,int c1, int c2,
                      double *amat, int sizea1,int sizea2, int a1, int a2,
                      double *bmat, int sizeb1,int sizeb2, int b1, int b2,int signb,double *Bm_reverse);

void get_bubble_1_mat(std::complex<double> *cmat,int sizec1,int sizec2,int c1, int c2,
                      double *amat, int sizea1,int sizea2, int a1, int a2,
                      double *bmat, int sizeb1,int sizeb2, int b1, int b2,int signb,double *Bm_reverse);

void get_bubble_1_mat(double *cmat,int sizec1,int sizec2,int c1, int c2,
                      std::complex<double> *amat, int sizea1,int sizea2, int a1, int a2,
                      double *bmat, int sizeb1,int sizeb2, int b1, int b2,int signb,double *Bm_reverse);

void get_bubble_1_timestep(int tstp, std::complex<double> *cret, std::complex<double> *ctv, std::complex<double> *cles,int sizec1,int sizec2,int c1, int c2,
                  std::complex<double> *aret, std::complex<double> *atv, std::complex<double> *ales, std::complex<double> *accret, std::complex<double> *acctv, std::complex<double> *accles,int sizea1,int sizea2, int a1, int a2,
                std::complex<double> *bret, std::complex<double> *btv, std::complex<double> *bles, std::complex<double> *bccret, std::complex<double> *bcctv, std::complex<double> *bccles,int sizeb1,int sizeb2, int b1, int b2,int r,std::complex<double> *Btv_reverse);

//   BUBBLE2:  C_{c1,c2}(t1,t2) = ii * A_{a1,a2}(t1,t2) * B_{b1,b2}(t1,t2)
void Bubble2_curr(int tstp, herm_matrix_hodlr &C, int c1, int c2, herm_matrix_hodlr &A, herm_matrix_hodlr &Acc, int a1, int a2, herm_matrix_hodlr &B,
             herm_matrix_hodlr &Bcc, int b1, int b2);

void Bubble2_curr(int tstp, cntr::herm_matrix_timestep<double> &C, int c1, int c2, herm_matrix_hodlr &A, herm_matrix_hodlr &Acc, int a1, int a2, herm_matrix_hodlr &B,
             herm_matrix_hodlr &Bcc, int b1, int b2);

void Bubble2_curr(int tstp, herm_matrix_hodlr &C, int c1, int c2, cntr::herm_matrix_timestep<double> &A, cntr::herm_matrix_timestep<double> &Acc, int a1, int a2, herm_matrix_hodlr &B,
             herm_matrix_hodlr &Bcc, int b1, int b2);

void Bubble2_curr(int tstp, herm_matrix_hodlr &C, int c1, int c2, herm_matrix_hodlr &A, herm_matrix_hodlr &Acc, int a1, int a2, cntr::herm_matrix_timestep<double> &B,
             cntr::herm_matrix_timestep<double> &Bcc, int b1, int b2);

void get_bubble_2_timestep(int tstp, std::complex<double> *cret, std::complex<double> *ctv, std::complex<double> *cles,int sizec1,int sizec2,int c1, int c2,
                  std::complex<double> *aret, std::complex<double> *atv, std::complex<double> *ales, std::complex<double> *accret, std::complex<double> *acctv, std::complex<double> *accles,int sizea1,int sizea2, int a1, int a2,
                std::complex<double> *bret, std::complex<double> *btv, std::complex<double> *bles, std::complex<double> *bccret, std::complex<double> *bcctv, std::complex<double> *bccles,int sizeb1,int sizeb2, int b1, int b2,int r);

void get_bubble_2_mat(double *cmat,int sizec1,int sizec2,int c1, int c2,
                      double *amat, int sizea1,int sizea2, int a1, int a2,
                      double *bmat, int sizeb1,int sizeb2, int b1, int b2);

void get_bubble_2_mat(std::complex<double> *cmat,int sizec1,int sizec2,int c1, int c2,
                      double *amat, int sizea1,int sizea2, int a1, int a2,
                      double *bmat, int sizeb1,int sizeb2, int b1, int b2);

void get_bubble_2_mat(double *cmat,int sizec1,int sizec2,int c1, int c2,
                      std::complex<double> *amat, int sizea1,int sizea2, int a1, int a2,
                      double *bmat, int sizeb1,int sizeb2, int b1, int b2);

void get_bubble_2_mat(double *cmat,int sizec1,int sizec2,int c1, int c2,
                      double *amat, int sizea1,int sizea2, int a1, int a2,
                      std::complex<double> *bmat, int sizeb1,int sizeb2, int b1, int b2);

void get_bubble_2_timestep(int tstp, std::complex<double> *cret, std::complex<double> *ctv, std::complex<double> *cles,int sizec1,int sizec2,int c1, int c2,
                  std::complex<double> *aret, std::complex<double> *atv, std::complex<double> *ales, std::complex<double> *accret, std::complex<double> *acctv, std::complex<double> *accles,int sizea1,int sizea2, int a1, int a2,
                std::complex<double> *bret, std::complex<double> *btv, std::complex<double> *bles, std::complex<double> *bccret, std::complex<double> *bcctv, std::complex<double> *bccles,int sizeb1,int sizeb2, int b1, int b2, int r);



// Classes for H structure

// Binary tree

// Node
class node
{
  public:
    node();
    ~node();
    node(int min,int max);

    int min(void){return min_;};
    int max(void){return max_;};
    node *left(void){return left_child_;};
    node *right(void){return right_child_;};
    node *insert(node *root,int min,int max);
    // void getNodesOnLevel(int level, node *root);
    void inOrder(node *root);
    int admissible(node *t,node *s);
    void qtree(node *t,node *s,int level,std::vector<int> &blkr1,std::vector<int> &blkr2,std::vector<int> &blkc1,std::vector<int> &blkc2,std::vector<int> &blklevel);
    void print(node *root,int nlevel);
    void getNodesOnLevel(int level, node *root,std::vector<int> &min,std::vector<int> &max);
    // void qtreeINT(int s_min,int s_max,int t_min,int t_max);
    // int admissibleINT(int s_min,int s_max,int t_min,int t_max);
    int min_; // Lower boundary
    int max_; // Higher boundary
    node* left_child_;
    node* right_child_;

};





}  // namespace hodlr
