#include <vector>
#include <iostream>
#include <string>

#include "h_nessi/herm_matrix_hodlr.hpp"
#include "h_nessi/checks.hpp"

namespace h_nessi {

void herm_matrix_hodlr::update_blocks(Integration::Integrator &I) {
  // HM-U1 (tier 2, state): an extra (k+2)-th final call would index ntri_[nt], timing(nt,.) (ISSUES H1f)
  HNESSI_REQUIRE_STATE(nt_ >= 1 && 0 <= tstpmk_ && tstpmk_ < nt_, "herm_matrix_hodlr::update_blocks",
                       HNESSI_KV(tstpmk_), HNESSI_KV(nt_),
                       detail::note{"every row is already archived: update_blocks is called once per time step and k+1 "
                                    "times after the last step"});
  // HM-U2 (tier 2): the left-edge pre-weighting uses I.omega(tbar), tbar <= k (doc 12 §4.1)
  HNESSI_REQUIRE(I.k() == k_, "herm_matrix_hodlr::update_blocks", HNESSI_KV(I.k()), HNESSI_KV(k_));
  std::chrono::time_point<std::chrono::steady_clock> start, end;
  std::chrono::duration<double, std::micro> dur;

  start = std::chrono::steady_clock::now();
  // First look at the timestep that you are putting into the svds and see what blocks need to be updated
  std::vector<int> blocks_list(0);
  for( int b = 0; b < nbox_; b++) {
    if(blkr1_[b] <= tstpmk_ && blkr2_[b] >= tstpmk_) {
      blocks_list.push_back(b);
    }
  }

  // Put directly stored data into triangles (row and column)
  int ndir = ntri_[tstpmk_];
  // HM-U6 (tier 1): row tstpmk of the yellow triangle and its square row index ndir-1
  HNESSI_ASSERT(ndir >= 1 && c1_dir_[tstpmk_] == tstpmk_-ndir+1, "herm_matrix_hodlr::update_blocks", HNESSI_KV(ndir),
                HNESSI_KV(tstpmk_), HNESSI_KV(c1_dir_[tstpmk_]));
  cplx *dataptr = curr_timestep_ret_ptr(tstpmk_, tstpmk_);
  for(int i = 0; i < ndir; i++) {
    ZMatrixMap(retptr_col(tstpmk_, tstpmk_-i), size1_, size2_) = ZMatrixMap(dataptr - i*size1_*size2_, size1_, size2_);
  }
  end = std::chrono::steady_clock::now();
  dur = end-start;
  timing(tstpmk_, 0) = dur.count();

  // Put data into ret_corr container
  start = std::chrono::steady_clock::now();
  for(int t2 = std::max(tstpmk_-k_,0); t2 < c1_dir_[tstpmk_]; t2++) {
    ZMatrixMap(retptr_corr(tstpmk_, t2), size1_, size2_) = ZMatrixMap(curr_timestep_ret_ptr(tstpmk_, t2), size1_, size2_);
  }
  end = std::chrono::steady_clock::now();
  dur = end-start;
  timing(tstpmk_, 1) = dur.count();

  // Put data into ret_left_edge conainer
  start = std::chrono::steady_clock::now();
  for(int t2 = 0; t2 <= k_; t2++) {
    ZMatrixMap(ret_left_edge_.data() + (tstpmk_*(k_+1) + t2)*size1_*size2_, size1_, size2_) = ZMatrixMap(curr_timestep_ret_ptr(tstpmk_, t2), size1_, size2_);
  }
  end = std::chrono::steady_clock::now();
  dur = end-start;
  timing(tstpmk_, 2) = dur.count();

  // Loop over blocks and update svds.  Column 5 times the whole loop (BUGS B27: `start` is reused for the per-block
  // columns 6-8 inside the loop, so column 5 used to measure only the last segment); columns 6-8 accumulate.
  const auto loop_start = std::chrono::steady_clock::now();
  for(int b = 0; b < blocks_list.size(); b++) {
    start = std::chrono::steady_clock::now();
    int bindex = blocks_list[b];
    ZMatrix y(1,blkc2_[bindex]-blkc1_[bindex]+1);
    // HM-U5 (tier 1): the block has archived exactly the rows blkr1..tstpmk-1
    HNESSI_ASSERT(blklen_[bindex] == tstpmk_ - blkr1_[bindex], "herm_matrix_hodlr::update_blocks", HNESSI_KV(bindex),
                  HNESSI_KV(blklen_[bindex]), HNESSI_KV(tstpmk_), HNESSI_KV(blkr1_[bindex]));
    end = std::chrono::steady_clock::now();
    dur = end-start;
    timing(tstpmk_, 6) += dur.count();
    if(blklen_[bindex] == 0) { // no data yet, init the block
      start = std::chrono::steady_clock::now();
      built_blocks_ += 1;
      for(int i = 0; i < size1_; i++) {
        for(int j = 0; j < size2_; j++) {
          y = Eigen::Map<Eigen::MatrixXcd, 0, Eigen::InnerStride<> >(curr_timestep_ret_ptr(tstpmk_,0) + blkc1(bindex)*size1_*size2_ + i*size2_ + j, 1, blkc2(bindex)-blkc1(bindex)+1, Eigen::InnerStride<>(size1_*size2_));
          // simply fill U, S, V from trivial SVD of y
          if(y.norm() > 1e-15) {
            ret_.data().blocks()[bindex][i][j].U() = ZMatrix::Ones(1,1);
            ret_.data().blocks()[bindex][i][j].S() = y.norm() * DMatrix::Ones(1,1);
            ret_.data().blocks()[bindex][i][j].V() = y.adjoint()/y.norm();
            ret_.data().blocks()[bindex][i][j].set_rank(1);
          }
          else {
            ret_.data().blocks()[bindex][i][j].U() = ZMatrix::Ones(1,1);
            ret_.data().blocks()[bindex][i][j].S() = 0. * DMatrix::Ones(1,1);
            ret_.data().blocks()[bindex][i][j].V() = ZMatrix::Zero(blkc2(bindex)-blkc1(bindex)+1,1);
            ret_.data().blocks()[bindex][i][j].set_rank(1);
          }
        }
      }
      end = std::chrono::steady_clock::now();
      dur = end-start;
      timing(tstpmk_, 7) += dur.count();
    }
    else { // update the tsvd
      start = std::chrono::steady_clock::now();
      for(int i = 0; i < size1_; i++) {
        for(int j = 0; j < size2_; j++) {
          y = Eigen::Map<Eigen::MatrixXcd, 0, Eigen::InnerStride<> >(curr_timestep_ret_ptr(tstpmk_,0) + blkc1(bindex)*size1_*size2_ + i*size2_ + j, 1, blkc2(bindex)-blkc1(bindex)+1, Eigen::InnerStride<>(size1_*size2_));
          updatetsvd(ret_.data().blocks()[bindex][i][j],y,blklen_[bindex]);
        }
      }
      end = std::chrono::steady_clock::now();
      dur = end-start;
      timing(tstpmk_, 8) += dur.count();
    }
    if(blklen_[bindex] == 0) { // no data yet, init the block
      start = std::chrono::steady_clock::now();
      for(int i = 0; i < size1_; i++) {
        for(int j = 0; j < size2_; j++) {
          y = Eigen::Map<Eigen::MatrixXcd, 0, Eigen::InnerStride<> >(curr_timestep_les_ptr(0,tstpmk_) + blkc1(bindex)*size1_*size2_ + i*size2_ + j, 1, blkc2(bindex)-blkc1(bindex)+1, Eigen::InnerStride<>(size1_*size2_));
          // simply fill U, S, V from trivial SVD of y
          if(y.norm() > 1e-15) {
            les_.data().blocks()[bindex][i][j].U() = ZMatrix::Ones(1,1);
            les_.data().blocks()[bindex][i][j].S() = y.norm() * DMatrix::Ones(1,1);
            les_.data().blocks()[bindex][i][j].V() = y.adjoint()/y.norm();
            les_.data().blocks()[bindex][i][j].set_rank(1);
          }
          else {
            les_.data().blocks()[bindex][i][j].U() = ZMatrix::Ones(1,1);
            les_.data().blocks()[bindex][i][j].S() = 0. * DMatrix::Ones(1,1);
            les_.data().blocks()[bindex][i][j].V() = ZMatrix::Zero(blkc2(bindex)-blkc1(bindex)+1,1);
            les_.data().blocks()[bindex][i][j].set_rank(1);
          }
        }
      }
      end = std::chrono::steady_clock::now();
      dur = end-start;
      timing(tstpmk_, 7) += dur.count();
    }
    else { // update the tsvd
      start = std::chrono::steady_clock::now();
      for(int i = 0; i < size1_; i++) {
        for(int j = 0; j < size2_; j++) {
          y = Eigen::Map<Eigen::MatrixXcd, 0, Eigen::InnerStride<> >(curr_timestep_les_ptr(0,tstpmk_) + blkc1(bindex)*size1_*size2_ + i*size2_ + j, 1, blkc2(bindex)-blkc1(bindex)+1, Eigen::InnerStride<>(size1_*size2_));
          updatetsvd(les_.data().blocks()[bindex][i][j],y,blklen_[bindex]);
        }
      }
      end = std::chrono::steady_clock::now();
      dur = end-start;
      timing(tstpmk_, 8) += dur.count();
    }
    // block now has one more row
    blklen_[bindex] = blklen_[bindex]+1;
  }
  end = std::chrono::steady_clock::now();
  dur = end-loop_start;
  timing(tstpmk_, 5) = dur.count();
  // HM-U4 (tier 1): segment index of row tstpmk used for les_dir_square_ below
  HNESSI_ASSERT(built_blocks_ == t_to_dirlvl_[tstpmk_], "herm_matrix_hodlr::update_blocks", HNESSI_KV(built_blocks_),
                HNESSI_KV(t_to_dirlvl_[tstpmk_]), HNESSI_KV(tstpmk_));

  int nao_ = size1_;
  int es_ = nao_*nao_;
  int row = ndir-1;
  start = std::chrono::steady_clock::now();
  for(int d = 0; d < ndir; d++) {
    for(int i = 0; i < nao_; i++) {
      for(int j = 0; j < nao_; j++) {
        dataptr = curr_timestep_les_ptr(tstpmk_-ndir+1+d, tstpmk_);
        les_dir_square_[i*nao_*len_les_dir_square_ + j*len_les_dir_square_ + les_dir_square_first_index_[built_blocks_] + d*blkdirheight_[built_blocks_] + row] = -std::conj(dataptr[j*nao_+i]);
        les_dir_square_[i*nao_*len_les_dir_square_ + j*len_les_dir_square_ + les_dir_square_first_index_[built_blocks_] + row*blkdirheight_[built_blocks_] + d] = dataptr[i*nao_+j];
      }
    }
  }
  end = std::chrono::steady_clock::now();
  dur = end-start;
  timing(tstpmk_, 3) = dur.count();

  start = std::chrono::steady_clock::now();
  int min = std::min(std::min(tstpmk_, k_), c1_dir_[tstpmk_]-1);
  for(int tbar = 0; tbar <= min; tbar++) {
    for(int i = 0; i < nao_; i++) {
      for(int k = 0; k < nao_; k++) {
        les_left_edge_[i * (k_+1) * nt_ * nao_ + tstpmk_ * nao_ * (k_+1) + k * (k_+1) + tbar] = std::conj(curr_timestep_les_ptr(tbar, tstpmk_)[k*nao_+i]) * (1-I.omega(tbar));
      }
    }
  }
  for(int tbar = min + 1; tbar <= k_; tbar++) {
    for(int i = 0; i < nao_; i++) {
      for(int k = 0; k < nao_; k++) {
        les_left_edge_[i * (k_+1) * nt_ * nao_ + tstpmk_ * nao_ * (k_+1) + k * (k_+1) + tbar] = 0.;
      }
    }
  }
  end = std::chrono::steady_clock::now();
  dur = end-start;
  timing(tstpmk_, 4) = dur.count();
  can_extrap_ = true;
  tstpmk_++;
}

/*
// Modified Gram-Schmidt orthogonalization of b agains subspace V
// Golub Sec. 5.2.8
double herm_matrix_hodlr::mgs(const ZMatrix &V,const ZRowVector &b,const ZRowVector &x_,const ZRowVector &Vb_){
  int rank=V.cols();
  ZRowVector& Vb=const_cast< ZRowVector& >(Vb_); //https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html another option is v
  ZRowVector& x=const_cast< ZRowVector& >(x_);
  x=b;
  for(int i=0;i<rank;i++){
    Vb(i)=V.col(i).dot(x);
    x-=Vb(i)*V.col(i);
  }
  double nrm=x.norm();
  x=x/nrm;
  std::cout << nrm << std::endl<<std::endl;
  std::cout << x.transpose() << std::endl<<std::endl;
  return nrm;
}
*/

double herm_matrix_hodlr::mgs(const ZMatrix &V, const ZRowVector &b, const ZRowVector &x_, const ZRowVector &Vb_) {
    int rank = V.cols();
    // HM-U11 (tier 1): x_ and Vb_ are written through const_cast and must be real (sized) objects
    HNESSI_ASSERT(V.rows() == b.size() && Vb_.size() >= rank, "herm_matrix_hodlr::mgs", HNESSI_KV(V.rows()),
                  HNESSI_KV(b.size()), HNESSI_KV(Vb_.size()), HNESSI_KV(rank));
    ZRowVector& Vb = const_cast<ZRowVector&>(Vb_);
    ZRowVector& x = const_cast<ZRowVector&>(x_);

    // 1. Initialize
    x = b;
    double initial_norm = x.norm();
    
    // 2. First Pass: Standard Modified Gram-Schmidt
    for (int i = 0; i < rank; i++) {
        // Compute projection onto i-th basis vector
        std::complex<double> dot_val = V.col(i).dot(x); 
        Vb(i) = dot_val;
        x -= Vb(i) * V.col(i);
    }
    double nrm = x.norm();

    // 3. Re-orthogonalization (Kahan's "Twice is Enough")
    // If the norm dropped significantly (e.g. by factor of 0.5), we lost precision.
    // We must re-project the residual 'x' against V to clean up remaining noise.
    if (nrm < 0.5 * initial_norm) {
        for (int i = 0; i < rank; i++) {
            std::complex<double> correction = V.col(i).dot(x);
            // IMPT: Accumulate the correction into Vb! 
            // The total projection is (First Pass + Second Pass)
            Vb(i) += correction; 
            x -= correction * V.col(i);
        }
        nrm = x.norm();
    }

    // 4. Normalize (with safety check for linear dependence)
    // If nrm is near machine epsilon, b is in the span of V. 
    // We cannot produce a normalized q.
    if (nrm > 1e-15) { 
        x = x / nrm;
    } else {
        // b is linearly dependent on V. beta is effectively 0.
        // We set nrm to 0 and x (q) to 0. 
        // Note: In Eq (15), if beta=0, the 'q' term is multiplied by 0, 
        // so mathematically this is safe for the update structure.
        nrm = 0.0;
        x.setZero(); 
    }

    // Debugging output (optional)
    // std::cout << "Beta: " << nrm << std::endl;
    
    return nrm;
}


void herm_matrix_hodlr::updatetsvd(block &B,const ZRowVector &row,int row_cur){
  std::chrono::time_point<std::chrono::steady_clock> start, end;
  std::chrono::duration<double, std::micro> dur;

  start = std::chrono::steady_clock::now();
  int cols=B.cols();
  int rank=B.epsrank();
  // HM-U8 (tier 1): the block holds row_cur rows of rank `rank`
  HNESSI_ASSERT(row.size() == cols && B.U().rows() == row_cur && B.S().size() == rank && B.U().cols() == rank &&
                B.V().cols() == rank && B.V().rows() == cols && rank <= std::min(row_cur, cols),
                "herm_matrix_hodlr::updatetsvd", HNESSI_KV(row.size()), HNESSI_KV(cols), HNESSI_KV(B.U().rows()),
                HNESSI_KV(row_cur), HNESSI_KV(B.S().size()), HNESSI_KV(B.U().cols()), HNESSI_KV(B.V().cols()),
                HNESSI_KV(rank));
  double nrm;
  ZRowVector Vrow(rank);
  ZMatrix uu(rank+1,rank+1); //Temporal U matrix
  DColVector su(rank+1); // Updated singular values
  ZMatrix uout(row_cur,rank+1),vout(cols,rank+1); //Final update
  ZRowVector q(cols);
  end = std::chrono::steady_clock::now();
  dur = end-start;
  timing(tstpmk_, 9) += dur.count();

  start = std::chrono::steady_clock::now();
  nrm=mgs(B.V(),row.adjoint(),q,Vrow);
  end = std::chrono::steady_clock::now();
  dur = end-start;
  timing(tstpmk_, 10) += dur.count();

  // Construct middle matrix [uu] in Eq. 15 of Low rank compression in the numerical solution of the nonequilibrium Dyson equation
  start = std::chrono::steady_clock::now();
  uu.setZero();
  for(int i=0;i<rank;i++){
    uu(i,i)=B.S()(i);
  }
  uu.block(rank,0,1,rank)=Vrow.conjugate();
  uu(rank,rank)=nrm;
  end = std::chrono::steady_clock::now();
  dur = end-start;
  timing(tstpmk_, 11) += dur.count();

  // Compute SVD [TODO: can be done faster as in Ref. 65]
  start = std::chrono::steady_clock::now();
  ZMatrix unew,vnew;
  svd(uu,unew,vnew,su);
  end = std::chrono::steady_clock::now();
  dur = end-start;
  timing(tstpmk_, 12) += dur.count();

  // Eigen::BDCSVD<Eigen::MatrixXcd> svd(uu,Eigen::ComputeThinU | Eigen::ComputeThinV);
  // ZMatrix unew=svd.matrixU(); 
  // ZMatrix vnew=svd.matrixV();
  // su= svd.singularValues().array();
  // Update block
  if(su(rank)<svdtol_){ //if rank has not increased

    start = std::chrono::steady_clock::now();
    ZMatrix ufin(row_cur+1,rank+1);
    end = std::chrono::steady_clock::now();
    dur = end-start;
    timing(tstpmk_, 13) += dur.count();

    // Set up U
    start = std::chrono::steady_clock::now();
    uout=B.U().block(0,0,row_cur,rank)*unew.block(0,0,rank,rank+1);
    end = std::chrono::steady_clock::now();
    dur = end-start;
    timing(tstpmk_, 14) += dur.count();

    start = std::chrono::steady_clock::now();
    ufin.block(0,0,row_cur,rank+1)=uout.block(0,0,row_cur,rank+1);
    end = std::chrono::steady_clock::now();
    dur = end-start;
    timing(tstpmk_, 15) += dur.count();

    start = std::chrono::steady_clock::now();
    ufin.block(row_cur,0,1,rank+1)=unew.block(rank,0,1,rank+1);
    end = std::chrono::steady_clock::now();
    dur = end-start;
    timing(tstpmk_, 16) += dur.count();

    start = std::chrono::steady_clock::now();
    B.set_U(ufin.block(0,0,row_cur+1,rank));
    end = std::chrono::steady_clock::now();
    dur = end-start;
    timing(tstpmk_, 17) += dur.count();

    // Set up V
    start = std::chrono::steady_clock::now();
    vout.block(0,0,cols,rank)=B.V().block(0,0,cols,rank);
    end = std::chrono::steady_clock::now();
    dur = end-start;
    timing(tstpmk_, 18) += dur.count();

    start = std::chrono::steady_clock::now();
    vout.block(0,rank,cols,1)=q.transpose();
    end = std::chrono::steady_clock::now();
    dur = end-start;
    timing(tstpmk_, 19) += dur.count();

    start = std::chrono::steady_clock::now();
    vout=vout*vnew.block(0,0,rank+1,rank+1);
    end = std::chrono::steady_clock::now();
    dur = end-start;
    timing(tstpmk_, 20) += dur.count();

    start = std::chrono::steady_clock::now();
    B.set_V(vout.block(0,0,cols,rank));
    B.set_S(su.head(rank));
    end = std::chrono::steady_clock::now();
    dur = end-start;
    timing(tstpmk_, 21) += dur.count();

  }else{ //if rank has increased

    start = std::chrono::steady_clock::now();
    ZMatrix ufin(row_cur+1,rank+1);
    B.set_rank(rank+1);
    end = std::chrono::steady_clock::now();
    dur = end-start;
    timing(tstpmk_, 13) += dur.count();

    // Set up U+
    start = std::chrono::steady_clock::now();
    uout=B.U().block(0,0,row_cur,rank)*unew.block(0,0,rank,rank+1);
    end = std::chrono::steady_clock::now();
    dur = end-start;
    timing(tstpmk_, 14) += dur.count();

    start = std::chrono::steady_clock::now();
    ufin.block(0,0,row_cur,rank+1)=uout.block(0,0,row_cur,rank+1);
    end = std::chrono::steady_clock::now();
    dur = end-start;
    timing(tstpmk_, 15) += dur.count();

    start = std::chrono::steady_clock::now();
    ufin.block(row_cur,0,1,rank+1)=unew.block(rank,0,1,rank+1);
    end = std::chrono::steady_clock::now();
    dur = end-start;
    timing(tstpmk_, 16) += dur.count();

    start = std::chrono::steady_clock::now();
    B.set_U(ufin);
    end = std::chrono::steady_clock::now();
    dur = end-start;
    timing(tstpmk_, 17) += dur.count();

    // Set up V+
    start = std::chrono::steady_clock::now();
    vout.block(0,0,cols,rank)=B.V().block(0,0,cols,rank);
    end = std::chrono::steady_clock::now();
    dur = end-start;
    timing(tstpmk_, 18) += dur.count();

    start = std::chrono::steady_clock::now();
    vout.block(0,rank,cols,1)=q.transpose();
    end = std::chrono::steady_clock::now();
    dur = end-start;
    timing(tstpmk_, 19) += dur.count();

    start = std::chrono::steady_clock::now();
    vout=vout*vnew.block(0,0,rank+1,rank+1);
    end = std::chrono::steady_clock::now();
    dur = end-start;
    timing(tstpmk_, 20) += dur.count();

    start = std::chrono::steady_clock::now();
    B.set_V(vout);
    B.set_S(su);
    end = std::chrono::steady_clock::now();
    dur = end-start;
    timing(tstpmk_, 21) += dur.count();

  }
#ifndef NDEBUG
  // HM-U9 (tier 1): new rank <= min(rows, cols); singular values finite, >= 0, non-increasing
  HNESSI_ASSERT(B.epsrank() <= std::min(row_cur+1, cols) && B.S().size() == B.epsrank(), "herm_matrix_hodlr::updatetsvd",
                HNESSI_KV(B.epsrank()), HNESSI_KV(row_cur), HNESSI_KV(cols));
  for(int i = 0; i < B.S().size(); i++)
    HNESSI_ASSERT(std::isfinite(B.S()(i)) && B.S()(i) >= 0 && (i == 0 || B.S()(i) <= B.S()(i-1)),
                  "herm_matrix_hodlr::updatetsvd", HNESSI_KV(i), HNESSI_KV(B.S()(i)));
#endif
}
} // namespace
