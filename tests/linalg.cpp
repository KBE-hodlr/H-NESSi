#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstring>
#include <iostream>
#include <sys/stat.h>

#include "hodlr/herm_matrix_hodlr.hpp"
#include "hodlr/dyson.hpp"
#include "hodlr/utils.hpp"

#define CPLX std::complex<double>
using namespace std;

// Test linear algebra tools used for the update of TSVD

TEST(LinearAlgebra, BlockUpdates) {

    // Test modified Gram-Schmidt routine on SVD of random matrix
    {
      double err = 0.0;
      int rows=15;
      int cols=10;
      double epsilon=1e-4,norm=0.0;
      // aux variables
      int ntau = 10;
      int r=ntau;
      int nlvl = 3;
      int size=1;
      int sig=-1;
      int kt=5;

      hodlr::ZMatrix S(10,10);
      
      // Set S matix
      S.setZero();
      int rank=0;
      for(int i=0;i<10;i++){
        S(i,i)=std::exp(-i*3.0);
        if(std::exp(-i*3.0)<epsilon){
          break;
        }
        rank++;
      }
      S.conservativeResize(rank,rank);
      hodlr::ZMatrix R(rows,cols),M(rows,cols),Unew(rows,rank),Vnew(cols,rank);
      hodlr::ZMatrix U,V;
      hodlr::DColVector Stmp;
      // Use U,V from a random matrix
      R=hodlr::ZMatrix::Random(rows,cols);
      hodlr::svd(R,U,V,Stmp);
      // std::cout << " ------------------ " << std::endl;
      Vnew=V.block(0,0,cols,rank);
      Unew=U.block(0,0,rows,rank);

      // Construct matrix M
      M=Unew*S*Vnew.adjoint();

      // Check rank of M
      hodlr::ZMatrix M1,V1,U1;
      hodlr::DColVector S1;
      hodlr::svd(M,U1,V1,S1);
      int epsrank=(S1.array() >= epsilon).count();

      M1=U1*S1.asDiagonal()*V1.adjoint();
      ASSERT_TRUE(epsrank ==rank);
      ASSERT_TRUE((M-M1).norm()<epsilon);
      // Construct random vector
      hodlr::ZRowVector x(cols),y(cols),Vb(cols);
      x=hodlr::ZRowVector::Random(cols);

      // Modified Gram-Schmidt
      hodlr::herm_matrix_hodlr G(cols,r,nlvl,epsilon,size,size,sig,kt);
      norm=G.mgs(Vnew,x,y,Vb);
      // Check orthogonality
      for(int i=0;i<epsrank;i++){
        err+=std::abs(Vnew.col(i).dot(y));
      }
      ASSERT_TRUE(err < epsilon);
    }

    // Update single line in TSVD
    {
      double epsilon=1e-4; 
      double err = 0.0;
      int rows=15;
      int cols=10;
      int size=1;
      int rank;
      int epsrank;
      // aux variables
      int ntau = 10;
      int r=ntau;
      int nlvl = 3;
      int sig=-1;
      int kt=5;


      hodlr::ZMatrix S(10,10); 
      
      // Set S matix
      S.setZero();
      rank=0;
      for(int i=0;i<10;i++){
        S(i,i)=std::exp(-i*3.0);
        if(std::exp(-i*3.0)<epsilon){
          break;
        }
        rank++;
      }
      S.conservativeResize(rank,rank);
      hodlr::ZMatrix R(rows,cols),U1(rows,rank),V1(cols,rank),U(rows,rank),V(cols,rank),M(rows,cols),Mfin(rows+1,cols);
      hodlr::DColVector Stmp;
      // Use U,V from a random matrix
      R=hodlr::ZMatrix::Random(rows,cols);
      hodlr::svd(R,U1,V1,Stmp);

      V=V1.block(0,0,cols,rank);
      U=U1.block(0,0,rows,rank);
      // Construct matrix m
      M=U*S*V.adjoint();
      // Vector to add
      hodlr::ZRowVector x(cols),x11(cols);
      x=hodlr::ZRowVector::Random(cols);
      // Construct block
      hodlr::block B(M,epsilon);
      B.resize(rows+1,cols,B.epsrank());
      ASSERT_TRUE(B.epsrank() == rank);

      hodlr::herm_matrix_hodlr G(rows,r,nlvl,epsilon,size,size,sig,kt);
      G.updatetsvd(B,x,rows);
      Mfin.block(0,0,rows,cols)=M;
      Mfin.block(rows,0,1,cols)=x;

      hodlr::ZMatrix Mupd;
      Mupd=B.U()*B.S().asDiagonal()*B.V().adjoint();
      double errF=(Mfin-Mupd).norm();
      ASSERT_TRUE(errF < epsilon);      
    }

    // Update whole svd
    {
      int rows=50;
      int cols=50;
      double epsilon=1e-5;
      // aux variables
      int ntau = 10;
      int r=ntau;
      int nlvl = 3;
      int size=1;
      int sig=-1;
      int kt=5;

      int rank;

      // Matrix S is fixed to rank 5
      int ssize=10;
      hodlr::ZMatrix S(ssize,ssize);
      S.setZero();
      for(int i=0;i<ssize;i++){
        S(i,i)=std::exp(-i*3.0);
      }
      S.conservativeResize(ssize,ssize);

      hodlr::ZMatrix R(rows,cols),U1(rows,ssize),V1(cols,ssize),U(rows,ssize),V(cols,ssize),M(rows,cols);
      hodlr::DColVector Stmp;
      R=hodlr::ZMatrix::Random(rows,cols);
      hodlr::svd(R,U1,V1,Stmp);
      V=V1.block(0,0,cols,ssize);
      U=U1.block(0,0,rows,ssize);
      M=U*S*V.adjoint();
      
      //Init dyson, block and set the first row
      // hodlr::dyson dyson(rows,ntau,1,5,1,epsilon,0.,0.,0.,0);
      
      hodlr::block b(M.row(0),epsilon);
      b.resize(rows,cols,b.epsrank());

      hodlr::herm_matrix_hodlr G(rows,r,nlvl,epsilon,size,size,sig,kt);
      // update rows
      for(int i=1;i<rows;i++){
        G.updatetsvd(b,M.row(i),i);
      }

      hodlr::DColVector S1,S2;
      hodlr::svd(M,S1);

      ASSERT_TRUE(b.epsrank()==(S1.array() >= epsilon).count()); // Check that the rank has been reproduced
      ASSERT_TRUE((M-(b.U()*b.S().asDiagonal()*b.V().adjoint())).norm() < epsilon); // Check that difference is below epsilon
    }
  
}
