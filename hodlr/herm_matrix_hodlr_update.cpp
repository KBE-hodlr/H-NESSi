#include <vector>
#include <iostream>
#include <string>

#include "herm_matrix_hodlr.hpp"

namespace hodlr {

void herm_matrix_hodlr::update_blocks(Integration::Integrator &I) {
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double, std::micro> dur;

  start = std::chrono::system_clock::now();
  // First look at the timestep that you are putting into the svds and see what blocks need to be updated
  std::vector<int> blocks_list(0);
  for( int b = 0; b < nbox_; b++) {
    if(blkr1_[b] <= tstpmk_ && blkr2_[b] >= tstpmk_) {
      blocks_list.push_back(b);
    }
  }

  // Put directly stored data into triangles (row and column)
  int ndir = ntri_[tstpmk_];
  cplx *dataptr = curr_timestep_ret_ptr(tstpmk_, tstpmk_);
  for(int i = 0; i < ndir; i++) {
    ZMatrixMap(retptr_col(tstpmk_, tstpmk_-i), size1_, size2_) = ZMatrixMap(dataptr - i*size1_*size2_, size1_, size2_);
  }
  end = std::chrono::system_clock::now();
  dur = end-start;
  timing(tstpmk_, 0) = dur.count();

  // Put data into ret_corr container
  start = std::chrono::system_clock::now();
  for(int t2 = std::max(tstpmk_-k_,0); t2 < c1_dir_[tstpmk_]; t2++) {
    ZMatrixMap(retptr_corr(tstpmk_, t2), size1_, size2_) = ZMatrixMap(curr_timestep_ret_ptr(tstpmk_, t2), size1_, size2_);
  }
  end = std::chrono::system_clock::now();
  dur = end-start;
  timing(tstpmk_, 1) = dur.count();

  // Put data into ret_left_edge conainer
  start = std::chrono::system_clock::now();
  for(int t2 = 0; t2 <= k_; t2++) {
    ZMatrixMap(ret_left_edge_.data() + (tstpmk_*(k_+1) + t2)*size1_*size2_, size1_, size2_) = ZMatrixMap(curr_timestep_ret_ptr(tstpmk_, t2), size1_, size2_);
  }
  end = std::chrono::system_clock::now();
  dur = end-start;
  timing(tstpmk_, 2) = dur.count();

  // Loop over blocks and update svds
  for(int b = 0; b < blocks_list.size(); b++) {
    start = std::chrono::system_clock::now();
    int bindex = blocks_list[b];
    ZMatrix y(1,blkc2_[bindex]-blkc1_[bindex]+1);
    end = std::chrono::system_clock::now();
    dur = end-start;
    timing(tstpmk_, 6) += dur.count();
    if(blklen_[bindex] == 0) { // no data yet, init the block
      start = std::chrono::system_clock::now();
      built_blocks_ += 1;
      for(int i = 0; i < size1_; i++) {
        for(int j = 0; j < size2_; j++) {
          y = Eigen::Map<Eigen::MatrixXcd, 0, Eigen::InnerStride<> >(curr_timestep_ret_ptr(tstpmk_,0) + blkc1(bindex)*size1_*size2_ + i*size1_ + j, 1, blkc2(bindex)-blkc1(bindex)+1, Eigen::InnerStride<>(size1_*size2_));
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
      end = std::chrono::system_clock::now();
      dur = end-start;
      timing(tstpmk_, 7) += dur.count();
    }
    else { // update the tsvd
      start = std::chrono::system_clock::now();
      for(int i = 0; i < size1_; i++) {
        for(int j = 0; j < size2_; j++) {
          y = Eigen::Map<Eigen::MatrixXcd, 0, Eigen::InnerStride<> >(curr_timestep_ret_ptr(tstpmk_,0) + blkc1(bindex)*size1_*size2_ + i*size1_ + j, 1, blkc2(bindex)-blkc1(bindex)+1, Eigen::InnerStride<>(size1_*size2_));
          updatetsvd(ret_.data().blocks()[bindex][i][j],y,blklen_[bindex]);
        }
      }
      end = std::chrono::system_clock::now();
      dur = end-start;
      timing(tstpmk_, 8) += dur.count();
    }
    if(blklen_[bindex] == 0) { // no data yet, init the block
      start = std::chrono::system_clock::now();
      for(int i = 0; i < size1_; i++) {
        for(int j = 0; j < size2_; j++) {
          y = Eigen::Map<Eigen::MatrixXcd, 0, Eigen::InnerStride<> >(curr_timestep_les_ptr(0,tstpmk_) + blkc1(bindex)*size1_*size2_ + i*size1_ + j, 1, blkc2(bindex)-blkc1(bindex)+1, Eigen::InnerStride<>(size1_*size2_));
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
      end = std::chrono::system_clock::now();
      dur = end-start;
      timing(tstpmk_, 7) += dur.count();
    }
    else { // update the tsvd
      start = std::chrono::system_clock::now();
      for(int i = 0; i < size1_; i++) {
        for(int j = 0; j < size2_; j++) {
          y = Eigen::Map<Eigen::MatrixXcd, 0, Eigen::InnerStride<> >(curr_timestep_les_ptr(0,tstpmk_) + blkc1(bindex)*size1_*size2_ + i*size1_ + j, 1, blkc2(bindex)-blkc1(bindex)+1, Eigen::InnerStride<>(size1_*size2_));
          updatetsvd(les_.data().blocks()[bindex][i][j],y,blklen_[bindex], bindex==8191 and i==0 and j==0 and false);
        }
      }
      end = std::chrono::system_clock::now();
      dur = end-start;
      timing(tstpmk_, 8) += dur.count();
    }
    // block now has one more row
    blklen_[bindex] = blklen_[bindex]+1;
  }
  end = std::chrono::system_clock::now();
  dur = end-start;
  timing(tstpmk_, 5) = dur.count();

  int nao_ = size1_;
  int es_ = nao_*nao_;
  int row = ndir-1;
  start = std::chrono::system_clock::now();
  for(int d = 0; d < ndir; d++) {
    for(int i = 0; i < nao_; i++) {
      for(int j = 0; j < nao_; j++) {
        dataptr = curr_timestep_les_ptr(tstpmk_-ndir+1+d, tstpmk_);
        les_dir_square_[i*nao_*len_les_dir_square_ + j*len_les_dir_square_ + les_dir_square_first_index_[built_blocks_] + d*blkdirheight_[built_blocks_] + row] = -std::conj(dataptr[j*nao_+i]);
        les_dir_square_[i*nao_*len_les_dir_square_ + j*len_les_dir_square_ + les_dir_square_first_index_[built_blocks_] + row*blkdirheight_[built_blocks_] + d] = dataptr[i*nao_+j];
      }
    }
  }
  end = std::chrono::system_clock::now();
  dur = end-start;
  timing(tstpmk_, 3) = dur.count();

  start = std::chrono::system_clock::now();
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
  end = std::chrono::system_clock::now();
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


void herm_matrix_hodlr::updatetsvd(hodlr::block &B,const ZRowVector &row,int row_cur, bool print){
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double, std::micro> dur;

  start = std::chrono::system_clock::now();
  int cols=B.cols();
  int rank=B.epsrank();
  double nrm;
  ZRowVector Vrow(rank);
  ZMatrix uu(rank+1,rank+1); //Temporal U matrix
  DColVector su(rank+1); // Updated singular values
  ZMatrix uout(row_cur,rank+1),vout(cols,rank+1); //Final update
  ZRowVector q(cols);
  end = std::chrono::system_clock::now();
  dur = end-start;
  timing(tstpmk_, 9) += dur.count();

  if(print) {
    std::cout << "row   " << row .norm() << std::endl;
  }

  start = std::chrono::system_clock::now();
  nrm=mgs(B.V(),row.adjoint(),q,Vrow);
  end = std::chrono::system_clock::now();
  dur = end-start;
  timing(tstpmk_, 10) += dur.count();

  if(print) {
    std::cout << "q     " << q   .norm() << std::endl;
    std::cout << "nrm   " << nrm         << std::endl;
    std::cout << "Vrow  " << Vrow.norm() << std::endl;
  }

  // Construct middle matrix [uu] in Eq. 15 of Low rank compression in the numerical solution of the nonequilibrium Dyson equation
  start = std::chrono::system_clock::now();
  uu.setZero();
  for(int i=0;i<rank;i++){
    uu(i,i)=B.S()(i);
  }
  uu.block(rank,0,1,rank)=Vrow.conjugate();
  uu(rank,rank)=nrm;
  end = std::chrono::system_clock::now();
  dur = end-start;
  timing(tstpmk_, 11) += dur.count();

  if(print) {
    std::cout << "uu    " << uu  .norm() << std::endl;
  }

  // Compute SVD [TODO: can be done faster as in Ref. 65]
  start = std::chrono::system_clock::now();
  ZMatrix unew,vnew;
  hodlr::svd(uu,unew,vnew,su);
  end = std::chrono::system_clock::now();
  dur = end-start;
  timing(tstpmk_, 12) += dur.count();

  if(print) {
    std::cout << "unew  " << unew.norm() << std::endl;
    std::cout << "vnew  " << vnew.norm() << std::endl;
    std::cout << "su    " << su  .norm() << std::endl;
    std::cout << "inc?  " << !(su(rank)<svdtol_) << std::endl;
    std::cout << "sv(r) " << su(rank) << std::endl;
    std::cout << "svtol " << svdtol_ << std::endl;
  }

  // Eigen::BDCSVD<Eigen::MatrixXcd> svd(uu,Eigen::ComputeThinU | Eigen::ComputeThinV);
  // ZMatrix unew=svd.matrixU(); 
  // ZMatrix vnew=svd.matrixV();
  // su= svd.singularValues().array();
  // Update block
  if(su(rank)<svdtol_){ //if rank has not increased

    start = std::chrono::system_clock::now();
    ZMatrix ufin(row_cur+1,rank+1);
    end = std::chrono::system_clock::now();
    dur = end-start;
    timing(tstpmk_, 13) += dur.count();

    // Set up U
  if(print) {
    std::cout << "==================" << std::endl;
    std::cout << "==   U update   ==" << std::endl;
    std::cout << "==================" << std::endl;
  }
    start = std::chrono::system_clock::now();
    uout=B.U().block(0,0,row_cur,rank)*unew.block(0,0,rank,rank+1);
    end = std::chrono::system_clock::now();
    dur = end-start;
    timing(tstpmk_, 14) += dur.count();

  if(print) {
    std::cout << "uout    " << uout.norm() << std::endl;
  }

    start = std::chrono::system_clock::now();
    ufin.block(0,0,row_cur,rank+1)=uout.block(0,0,row_cur,rank+1);
    end = std::chrono::system_clock::now();
    dur = end-start;
    timing(tstpmk_, 15) += dur.count();

  if(print) {
    std::cout << "ufin1   " << ufin.block(0,0,row_cur,rank+1).norm() << std::endl;
  }

    start = std::chrono::system_clock::now();
    ufin.block(row_cur,0,1,rank+1)=unew.block(rank,0,1,rank+1);
    end = std::chrono::system_clock::now();
    dur = end-start;
    timing(tstpmk_, 16) += dur.count();

    start = std::chrono::system_clock::now();
    B.set_U(ufin.block(0,0,row_cur+1,rank));
    end = std::chrono::system_clock::now();
    dur = end-start;
    timing(tstpmk_, 17) += dur.count();

  if(print) {
    std::cout << "ufin2   " << ufin.block(0,0,row_cur+1,rank).norm() << std::endl;
  }


    // Set up V
  if(print) {
    std::cout << "==================" << std::endl;
    std::cout << "==   V update   ==" << std::endl;
    std::cout << "==================" << std::endl;
  }
    start = std::chrono::system_clock::now();
    vout.block(0,0,cols,rank)=B.V().block(0,0,cols,rank);
    end = std::chrono::system_clock::now();
    dur = end-start;
    timing(tstpmk_, 18) += dur.count();

  if(print) {
    std::cout << "vout    " << vout.block(0,0,cols,rank).norm() << std::endl;
  }

    start = std::chrono::system_clock::now();
    vout.block(0,rank,cols,1)=q.transpose();
    end = std::chrono::system_clock::now();
    dur = end-start;
    timing(tstpmk_, 19) += dur.count();

  if(print) {
    std::cout << "vout2   " << vout.norm() << std::endl;
  }

    start = std::chrono::system_clock::now();
    vout=vout*vnew.block(0,0,rank+1,rank+1);
    end = std::chrono::system_clock::now();
    dur = end-start;
    timing(tstpmk_, 20) += dur.count();

  if(print) {
    std::cout << "vout3   " << vout.norm() << std::endl;
  }
  if(print) {
    std::cout << "vout4   " << vout.block(0,0,cols,rank).norm() << std::endl;
  }

    start = std::chrono::system_clock::now();
    B.set_V(vout.block(0,0,cols,rank));
    B.set_S(su.head(rank));
    end = std::chrono::system_clock::now();
    dur = end-start;
    timing(tstpmk_, 21) += dur.count();

  }else{ //if rank has increased

    start = std::chrono::system_clock::now();
    ZMatrix ufin(row_cur+1,rank+1);
    B.set_rank(rank+1);
    end = std::chrono::system_clock::now();
    dur = end-start;
    timing(tstpmk_, 13) += dur.count();

    // Set up U+
    start = std::chrono::system_clock::now();
    uout=B.U().block(0,0,row_cur,rank)*unew.block(0,0,rank,rank+1);
    end = std::chrono::system_clock::now();
    dur = end-start;
    timing(tstpmk_, 14) += dur.count();

    start = std::chrono::system_clock::now();
    ufin.block(0,0,row_cur,rank+1)=uout.block(0,0,row_cur,rank+1);
    end = std::chrono::system_clock::now();
    dur = end-start;
    timing(tstpmk_, 15) += dur.count();

    start = std::chrono::system_clock::now();
    ufin.block(row_cur,0,1,rank+1)=unew.block(rank,0,1,rank+1);
    end = std::chrono::system_clock::now();
    dur = end-start;
    timing(tstpmk_, 16) += dur.count();

    start = std::chrono::system_clock::now();
    B.set_U(ufin);
    end = std::chrono::system_clock::now();
    dur = end-start;
    timing(tstpmk_, 17) += dur.count();

    // Set up V+
    start = std::chrono::system_clock::now();
    vout.block(0,0,cols,rank)=B.V().block(0,0,cols,rank);
    end = std::chrono::system_clock::now();
    dur = end-start;
    timing(tstpmk_, 18) += dur.count();

    start = std::chrono::system_clock::now();
    vout.block(0,rank,cols,1)=q.transpose();
    end = std::chrono::system_clock::now();
    dur = end-start;
    timing(tstpmk_, 19) += dur.count();

    start = std::chrono::system_clock::now();
    vout=vout*vnew.block(0,0,rank+1,rank+1);
    end = std::chrono::system_clock::now();
    dur = end-start;
    timing(tstpmk_, 20) += dur.count();

    start = std::chrono::system_clock::now();
    B.set_V(vout);
    B.set_S(su);
    end = std::chrono::system_clock::now();
    dur = end-start;
    timing(tstpmk_, 21) += dur.count();

  }
}
} // namespace
