#include <vector>
#include <iostream>
#include <string>

#include "herm_matrix_hodlr.hpp"


namespace hodlr {

/* #######################################################################################
#
#   herm_matrix_hodlr
#
########################################################################################*/
herm_matrix_hodlr::herm_matrix_hodlr() {
    blkr1_=std::vector<int>();
    blkr2_=std::vector<int>();
    blkc1_=std::vector<int>();
    blkc2_=std::vector<int>();
    blklevel_=std::vector<int>();
    blkdirheight_=std::vector<int>();
    blkndirstart_=std::vector<int>();
    c1_dir_=std::vector<int>();
    blklen_=std::vector<int>();
    built_blocks_ = 0;

    ret_=hodlr::ret_blocks();
    les_=hodlr::les_blocks();
    tv_=hodlr::tv_blocks();
    mat_=0;
    GMConvTensMatrix_ = DMatrix();

    nlvl_=0;
    nbox_=0;
    svdtol_=0.0;
  
    curr_timestep_ret_ = std::vector<cplx>(0);
    curr_timestep_les_ = std::vector<cplx>(0);
    les_dir_square_ = std::vector<cplx>(0);
    les_dir_square_first_index_ = std::vector<int>(0);
    les_left_edge_ = std::vector<cplx>(0);

    tstpmk_ = 0;
    k_ = 0;
    ntau_ = 0;
    r_ = 0;
    nt_ = 0;
    ndir_=0;
    size1_ = 0;
    size2_ = 0;
    shape_ = 0;
    sig_ = -1;
}

herm_matrix_hodlr::~herm_matrix_hodlr() {
    // delete [] ret_;
    // delete [] les_;
    // delete [] tv_;
    delete [] mat_;
}

// TODO: 
herm_matrix_hodlr::herm_matrix_hodlr(int nt,int r,int nlvl,double svdtol,int size1,int size2,int sig,int k,int shape) : 
it2cf_tmp_(r,r),
LU_(r),
LU_built_(false)
{
    assert(nt >= 0 && nlvl >= 0 && svdtol > 0);
    nt_=nt;
    nlvl_=nlvl;
    svdtol_=svdtol;
    size1_=size1;
    size2_=size2;
    ntau_=r;
    r_=r;
    sig_=sig;
    k_ = k;
    tstpmk_ = 0;
    built_blocks_ = 0;
    shape_=shape;
 
    mat_ = new double[r_ * size1_ * size2_];
    GMConvTensMatrix_ = DMatrix(r_ , r_ * size1_ * size2_);

    curr_timestep_ret_ = std::vector<cplx>((k_+1) * (nt_ + 1)  * size1_ * size2_);
    curr_timestep_les_ = std::vector<cplx>((k_+1) * (nt_ + 1) * size1_ * size2_);

    if(shape_==0){
        init_shape(nt,nlvl);    
    }else if(shape_==1){
        init_shape_h(nt,nlvl);
    }
    std::vector<int> nrows(nbox_),ncols(nbox_);
    for(int i=0;i<nbox_;i++){
        nrows[i]=blkr2_[i]-blkr1_[i]+1;
        ncols[i]=blkc2_[i]-blkc1_[i]+1;
    }

    // Init Keldysh components
    // Mat
    for(int i=0;i<r_;i++){
        DMatrixMap(matptr(i), size1, size2) = DMatrix::Zero(size1,size2);
    }

    // real time
    ret_=hodlr::ret_blocks(nbox_,ndir_,nrows,ncols,svdtol,size1_,size2_,sig_);
    les_=hodlr::les_blocks(nbox_,ndir_,nrows,ncols,svdtol,size1_,size2_,sig_);
    tv_ =hodlr::tv_blocks(size1_, size2_, nt_, r_, sig_, svdtol);

    // for LU eval
    Gijc_it_ = new double[r_ * size1_ * size2_];
    res_ = new double[size1_ * size2_];
}

herm_matrix_hodlr::herm_matrix_hodlr(GREEN &G,int nlvl,double svdtol, int r,int shape){
    assert(r>0);
    nt_=G.nt();
    ntau_=G.ntau();
    nlvl_=nlvl;
    r_=r;
    svdtol_=svdtol;
    size1_=G.size1();
    size2_=G.size2();
    sig_=G.sig();
    k_ = 5;
    tstpmk_ = nt_-k_;
    shape_=shape;

    // Set up the matsubara component
    mat_ = new double[r_ * size1_ * size2_];
    curr_timestep_ret_ = std::vector<cplx>((k_+1) * (nt_+1) * size1_ * size2_,cplx(0.0,0.0));
    curr_timestep_les_ = std::vector<cplx>((k_+1) * (nt_+1) * size1_ * size2_,cplx(0.0,0.0));
    built_blocks_ = 0;
    GMConvTensMatrix_ = DMatrix();

    if(shape==0){
        init_shape(nt_,nlvl);    
    }else if(shape==1){
        init_shape_h(nt_,nlvl);
    }
    // Init Keldysh components
    std::vector<int> nrows(nbox_),ncols(nbox_);
    blklen_ = std::vector<int>(nbox_,0);
    for(int i=0;i<nbox_;i++){
        nrows[i]=blkr2_[i]-blkr1_[i]+1;
        ncols[i]=blkc2_[i]-blkc1_[i]+1;
        blklen_[i]=blkc2_[i]-blkc1_[i]+1;
    }
    
    blkdirheight_ = std::vector<int>(nbox_+1,0);
    if(shape==0){
        blkdirheight_[0] = blkr1_[0];
        for(int b = 1; b < nbox_; b++) {
            blkdirheight_[b] = blkr1_[b] - blkr1_[b-1];
        }
        blkdirheight_[nbox_] = nt_ - blkr1_[nbox_-1];    
    }else{
        blkdirheight_[0] = blkr1_[0];
        blkdirheight_[1] = blkr2_[0] - blkr1_[0]+1;
        for(int b = 2; b < nbox_-1; b++) {
            if(this->blklevel()[b-1]>1){
                blkdirheight_[b] = 0;
            }else{
                if(this->blklevel()[b-2]==1 && this->blklevel()[b]==1){
                    blkdirheight_[b] = 0;
                }else{
                    blkdirheight_[b] = blkr2_[b-1] - blkr1_[b-1]+1 ;
                }
            }
        }
        blkdirheight_[nbox_-1] =  0; //Prelast block is always 0
        blkdirheight_[nbox_] =  nt_ - blkr1_[nbox_-1];
    }

    blkndirstart_ = std::vector<int>(nbox_+1,0);
    if(shape_==0){
        for(int b = 1; b <= nbox_; b++) {
            blkndirstart_[b] = blkndirstart_[b-1] + blkdirheight_[b-1]*(blkdirheight_[b-1]+1)/2;
        }    
    }else if(shape_==1){
        for(int b = 1; b <= nbox_; b++) {
            blkndirstart_[b] = blkndirstart_[b-1] + blkdirheight_[b-1]*(blkdirheight_[b-1]+1)/2;
        }
    }else{
        abort();
    }
    
    ret_=hodlr::ret_blocks(nbox_,ndir_,nrows,ncols,svdtol,size1_,size2_,sig_);
    les_=hodlr::les_blocks(nbox_,ndir_,nrows,ncols,svdtol,size1_,size2_,sig_);
    tv_ =hodlr::tv_blocks(size1_, size2_, nt_, r_, sig_, svdtol);

    ret_.set(G,blkr1_,blkr2_,blkc1_,blkc2_,c1_dir_,r2_dir_,ntri_,svdtol);
    les_.set(G,blkr1_,blkr2_,blkc1_,blkc2_,c1_dir_,r2_dir_,ntri_,svdtol);

    // Set last kt+1 timestep
    for(int i=tstpmk_;i<=tstpmk_+k_;i++){
        set_timestep_curr(i,G);  
    }

  cplx *dataptr;
  for(int t = 0; t < nt_; t++) {
    int ndir = ntri_[t];
    if(t > 0) if(ntri_[t] < ntri_[t-1]) built_blocks_++;
    int nao_ = size1_;
    int es_ = nao_*nao_;
    int row = ndir-1;
    for(int d = 0; d < ndir; d++) {
      for(int i = 0; i < nao_; i++) {
        for(int j = 0; j < nao_; j++) {
          dataptr = G.lesptr(tstpmk_-ndir+1+d, tstpmk_);
          les_dir_square_[i*nao_*len_les_dir_square_ + j*len_les_dir_square_ + les_dir_square_first_index_[built_blocks_] + d*blkdirheight_[built_blocks_] + row] = -std::conj(dataptr[j*nao_+i]);
          les_dir_square_[i*nao_*len_les_dir_square_ + j*len_les_dir_square_ + les_dir_square_first_index_[built_blocks_] + row*blkdirheight_[built_blocks_] + d] = dataptr[i*nao_+j];
        }
      }
    }
  }
  set_memory_usage();
}

void herm_matrix_hodlr::init_shape(int nt,int nlvl){
    // Fix the geometry of hierarchical structure
    nbox_=(int)round(pow(2,nlvl)-1);
    blkr1_=std::vector<int>(nbox_);
    blkr2_=std::vector<int>(nbox_);
    blkc1_=std::vector<int>(nbox_);
    blkc2_=std::vector<int>(nbox_);
    blklevel_=std::vector<int>(nbox_);
    c1_dir_=std::vector<int>(nt_,0); // Column index of first directly-stored entry in each row
    r2_dir_=std::vector<int>(nt_,nt_-1); // Row index of last directly-stored entry in each column
    ntri_=std::vector<int>(nt_,0);

    maxdir_=getbkl_indices(); // Return maximum number of directly-stored entries in a row

    ntri_[0]=1;c1_dir_[0]=0; // c1_dir==j1dir

    std::vector<int> diagi1(nbox_+1); // First column of directly stored element
    std::vector<int> diagi2(nbox_+1); // Last row of directly stored element
    diagi1[0]=0; 
    for(int i=0;i<nbox_;i++){
        diagi1[i+1]=blkr1_[i];
    }
    for(int i=0;i<nbox_;i++){
        diagi2[i]=blkr1_[i]-1;
    }
    diagi2[nbox_]=nt_;

    int maxdiag=0,tmp=0;
    for(int i=0; i<diagi2.size(); ++i){
        tmp=diagi2[i]-diagi1[i];
        if(tmp>maxdiag){
            maxdiag=tmp;
        }
    }
    for(int i=0;i<nbox_;i++){
        for(int l=blkr1_[i];l<=blkr2_[i];l++){
            c1_dir_[l]=diagi1[i+1]; 
        }
    }

    for(int i=0;i<nbox_;i++){
        for(int l=blkc1_[i];l<=blkc2_[i];l++){
            r2_dir_[l]=std::min(r2_dir_[l],diagi2[i]); 
        }
    }

    for(int i=0;i<nt_;i++){
        ntri_[i]=i+1-c1_dir_[i];
    }

    //Set map for directly stored entries in each row
    int i=0;
    mapdirr_c_ = std::vector<int>(ndir_);
    mapdirr_r_ = std::vector<int>(ndir_);
    for(int t=0;t<nt;t++){
        for(int t1=c1_dir_[t];t1<c1_dir_[t]+ntri_[t];t1++){
            mapdirr_r_[i]=t;
            mapdirr_c_[i]=t1;
            i++;
        }
    }

    i=0;
    mapdirc_r_ = std::vector<int>(ndir_);
    mapdirc_c_ = std::vector<int>(ndir_);
    for(int t=0;t<nt;t++){
        for(int t1=t;t1<=r2_dir_[t];t1++){ //From diagonal on
            mapdirc_c_[i]=t;
            mapdirc_r_[i]=t1;
            i++;
        }
    }

    blklen_ = std::vector<int>(nbox_,0);
    blkdirheight_ = std::vector<int>(nbox_+1,0);
    
    if(shape_==0){
        blkdirheight_[0] = blkr1_[0];
        for(int b = 1; b < nbox_; b++) {
            blkdirheight_[b] = blkr1_[b] - blkr1_[b-1];
        }
        blkdirheight_[nbox_] = nt_ - blkr1_[nbox_-1];    
    }else if(shape_==1){
        blkdirheight_[0] = blkr1_[0];
        blkdirheight_[1] = blkr2_[0] - blkr1_[0]+1;
        for(int b = 2; b < nbox_-1; b++) {
            if(this->blklevel()[b-1]>1){
                blkdirheight_[b] = 0;
            }else{
                if(this->blklevel()[b-2]==1 && this->blklevel()[b]==1){
                    blkdirheight_[b] = 0;
                }else{
                    blkdirheight_[b] = blkr2_[b-1] - blkr1_[b-1]+1 ;
                }
            }
        }
        blkdirheight_[nbox_-1] =  0; //Prelast block is always 0
        blkdirheight_[nbox_] =  nt_ - blkr1_[nbox_-1];
    }else{
        abort();
    }
    

    blkndirstart_ = std::vector<int>(nbox_+1,0);
    for(int b = 1; b <= nbox_; b++) {
      blkndirstart_[b] = blkndirstart_[b-1] + blkdirheight_[b-1]*(blkdirheight_[b-1]+1)/2;
    }

    les_dir_square_first_index_ = std::vector<int>(nbox_+1);
    t_to_dirlvl_ = std::vector<int>(nt_);
    les_dir_square_first_index_[0] = 0;
    int len = 0;
    for(int b = 0; b < nbox_+1; b++) {
      les_dir_square_first_index_[b] = len;     
      len += blkdirheight_[b] *blkdirheight_[b]; 
    }
    len_les_dir_square_ = len;
    les_dir_square_ = std::vector<cplx>(len * size1_ * size2_);
    les_left_edge_ = std::vector<cplx>(size1_ * size2_ * nt_ * (k_+1));

    int lvl = 0;
    int next_height = blkdirheight_[lvl];
    for(int t = 0; t < nt_; t++) {
      if(t == next_height) {
        lvl++;
        next_height += blkdirheight_[lvl];
      }
      t_to_dirlvl_[t] = lvl;
    }
}

int herm_matrix_hodlr::getbkl_indices(){
    int nbox=(int)round(pow(2,nlvl_))-1;
    int l;
    std::vector<int> is((nbox+1)*2);
    //  First row indices of all blocks
    // is(i+1)=i*nt/(2^{nlvl})
    for(int i=0;i<=nbox+1;i++){
        is[i]=(int)round(nt_*i/double(nbox+1)); 
        if(i>=1 && i<nbox+1) blkr1_[i-1]=is[i];
        // std::cout << "index " << i << " " << is[i] << " " << blkr1_[i] << std::endl;
    }
    // std::cout << "is ";
    // for(int i=0; i<is.size(); ++i) std::cout << is[i] << ' ';
    // std::cout << std::endl;
    // std::cout << "blkr1 ";
    // for(int i=0; i<blkr1_.size(); ++i) std::cout << blkr1_[i] << ' ';
    // std::cout << std::endl;

    // Levels of all blocks [closest to diagonal=1, \ldots]
    std::fill(blklevel_.begin(), blklevel_.end(), 1);
    for(int l=1;l<nlvl_;l++){
        // std::cout << int(pow(2,l)) << " " << int(pow(2,nlvl_)) << " " << int(pow(2,l)) << std::endl;
        for(int j=(int)round(pow(2,l));j<(int)round(pow(2,nlvl_));j=j+(int)round(pow(2,l))){
            blklevel_[j-1]=l+1;
            // std::cout << "Index: " << j << " " << l+1 << std::endl; 
        }
    }
    // std::cout << "blklevel ";
    // for(int i=0; i<blklevel_.size(); ++i) std::cout << blklevel_[i] << ' ';
    // std::cout << std::endl;
    // First column indices of all blocks
    std::vector<int> il(nlvl_,1);
    std::fill(blkc1_.begin(), blkc1_.end(), 0);
    for(int i=0;i<nbox;i++){
        l=blklevel_[i];
        // std::cout <<" PreValues " << i << " " << l << " " << blkc1_[i]<< " " << il[l-1] <<  " " << (il[l-1]-1)*(int)round(pow(2,l))  << std::endl;
        blkc1_[i]=is[(il[l-1]-1)*(int)round(pow(2,l))];
        // std::cout <<" Values " << i << " " << l << " " << blkc1_[i]<< " " << il[l-1] << " " << (il[l-1]-1)*(int)round(pow(2,l))  << std::endl;
        il[l-1]=il[l-1]+1;        
    }

    // std::cout << "blkc1 ";
    // for(int i=0; i<blkc1_.size(); ++i) std::cout << blkc1_[i] << ' ';
    // std::cout << std::endl;
    
    // std::cout << "il ";
    // for(int i=0; i<il.size(); ++i) std::cout << il[i] << ' ';
    // std::cout << std::endl;

    // Last row indices of all blocks
    std::fill(blkr2_.begin(), blkr2_.end(), 0);
    for(int i=0;i<nbox;i++){
        l=blklevel_[i];
        // std::cout << "inside " << i <<" " << l << " " << (int)round(pow(2,l-1)) << " " << is[i+(int)round(pow(2,l-1))+1] << std::endl;
        blkr2_[i]=is[i+(int)round(pow(2,l-1))+1]-1;
        if(blkr2_[i]==is[(int)round(pow(2,nlvl_))]-1){
            blkr2_[i]=blkr2_[i];
        }
    }
    // std::cout << "blkr2_ ";
    // for(int i=0; i<blkr2_.size(); ++i) std::cout << blkr2_[i] << ' ';
    // std::cout << std::endl;
    //  Last column indices of all blocks
    for(int i=0; i<blkc2_.size(); ++i) blkc2_[i]=blkr1_[i]-1;
    
    // Maximum possible number of directly-stored entries in a row
    // for(int i=0; i<blkc2_.size(); ++i) std::cout << blkc2_[i] << ' ';
    // std::cout << std::endl;
    int val1 = 0; // size1 of triangular region
    int val2 = 0; // size2 of triangular region
    for(int i=0;i<nbox;i++){
        if(blklevel_[i]==1){ //only the blocks closest to the diagonal
            val1=std::max(blkr1_[i]-blkc1_[i]+1,val1);
        }
    }
    // Column index of first directly-stored entry in each row
    for(int i=0;i<nbox;i++){
        // std::cout << "outer " << i << " " << blkr1_[i] << ' '<< blkr2_[i]  << std::endl;
        for(int j=blkr1_[i];j<=blkr2_[i];j++){
            // std::cout << i  << " " << j << " " << c1_dir_[j] << " " << blkc2_[i] << std::endl;
            if(c1_dir_[j]<blkc2_[i]+1) c1_dir_[j]=blkc2_[i]+1;
        }
    }
    // std::cout << "c1_dir_ ";
    // for(int i=0; i<c1_dir_.size(); ++i) std::cout << i << " " << c1_dir_[i] << " " << std::endl;

    ndir_=0;
    // Indices of directly stored entries
    for(int i=0;i<nt_;i++){
        for(int j=c1_dir_[i];j<=i;j++){
            ndir_+=1;
            // std::cout << i << " " << j << " " << ndir_ << std::endl;
        }
    }
    // std::cout << "Ndir " << ndir_ << std::endl;
    

    return std::max(val1,val2);
}


// Imaginary time convolution \int_0^\beta d\tau' \Sigma_{ik}(T,\tau') G_{kj}(\tau-\tau') in dlr basis is
// \phi_{abc} G_{c,kj} \Sigma_{b,ik}
// The first part is the same for each integral, so we construct
// C_{ajbk} = -\phi_{abc} G_{-c,kj}
void herm_matrix_hodlr::initGMConvTensor(double *it2itr,double *it2cf,int *it2cfp,double *phi){
    GMConvTensMatrix_.resize(r_*size1_*size2_, r_);

    double *G1 = new double[r_*size1_*size2_];
    double *G2 = new double[r_*size1_*size2_];

    int es = size1_*size2_;
    int one = 1;

    // A_{cjk} = G_{ckj}
    for(int i = 0; i<r_; i++) {
        DMatrixMap(G1+i*es, size1_, size2_).noalias() = DMatrixMap(mat_+i*es, size1_, size2_).transpose();
    }

    // B_{jkc} = -A_{cjk}
    DMatrixMap(G2, es, r_).noalias() = - DMatrixMap(G1, r_, es).transpose();

    // C_{jkc} = B_{jk(ntau-c-1)}
    for(int i = 0; i<es; i++) {
        c_dlr_it2itr(&r_, &one,it2itr, G2+i*r_, G1+i*r_);
        c_dlr_it2cf(&r_,&one,it2cf, it2cfp, G1+i*r_, G2+i*r_);
    }

    // D_{jkba} = C_{jkc} Phi_{cba}
    DMatrixMap(GMConvTensMatrix_.data(), es, r_*r_).noalias() = DMatrixMap(G2, es, r_) * DMatrixMap(phi, r_, r_*r_);

    // E_{ajkb} = D_{jkba}
    GMConvTensMatrix_.transposeInPlace();

    // C_{ajbk} = E_{ajkb}
    for( int i = 0; i < r_*size1_; i++) {
        DMatrixMap(GMConvTensMatrix_.data() + i*r_*size1_, r_, size2_) = DMatrixMap(GMConvTensMatrix_.data() + i*r_*size1_, size2_, r_).transpose().eval();
    }

    delete [] G1;
    delete [] G2;
}

// TODO: use t1,t2 for time indices and ij for orbital
void herm_matrix_hodlr::get_ret(int t1, int t2, cplx *M){
    if(t1 >= t2){
        if( t1 >= tstpmk_) {
          ZMatrixMap(M, size1_, size2_).noalias() = ZMatrixMap(curr_timestep_ret_ptr(t1,t2), size1_, size2_);
        }
        else if(t2>=c1_dir_[t1]){ // direct get
            ret_.get_direct_col(time2direct_col(t1,t2),M);
        } else {               // compressed get
            int b=time2block(t1,t2);
            ret_.get_compress(t1-blkr1_[b],t2-blkc1_[b],b,M);
        }
    }else{
        if( t2 >= tstpmk_) {
          ZMatrixMap(M, size1_, size2_).noalias() = ZMatrixMap(curr_timestep_ret_ptr(t2,t1), size1_, size2_);
        }
        else if(t1>=c1_dir_[t2]){ // direct get
            ret_.get_direct_col(time2direct_col(t2,t1),M);

        }else{               // compressed get
            int b=time2block(t2,t1);
            ret_.get_compress(t2-blkr1_[b],t1-blkc1_[b],b,M);
        }
        ZMatrixMap MMap(M, size1_, size2_);
        MMap = MMap.adjoint().eval();
        MMap *= -1.0;
    }
}

void herm_matrix_hodlr::get_ret(int t1, int t2, ZMatrix &M){
    assert(t1<=nt_ && t2<=nt_);
    M.resize(size1_,size2_);
    get_ret(t1, t2, M.data());
}

void herm_matrix_hodlr::set_tstp_zero(int tstp) {
    assert( tstp == -1 || (tstp >= tstpmk_ && tstp <= tstpmk_+k_));
    if(tstp == -1) {
      memset(mat_, 0, sizeof(double) * r_ * size1_ * size2_);
    }
    else {
      memset(curr_timestep_les_ptr(0,tstp), 0, sizeof(cplx) * (tstp+1) * size1_ * size2_);
      memset(curr_timestep_ret_ptr(tstp,0), 0, sizeof(cplx) * (tstp+1) * size1_ * size2_);
      memset(tvptr(tstp,0), 0, sizeof(cplx) * r_ * size1_ * size2_);
    }
}

// TODO: use t1,t2 for time indices and ij for orbital
void herm_matrix_hodlr::get_les(int t1, int t2, cplx* M, bool print){
    if(t2 >= t1){
        if( t2 >= tstpmk_) {
          ZMatrixMap(M, size1_, size2_).noalias() = ZMatrixMap(curr_timestep_les_ptr(t1,t2), size1_, size2_);
        }
        else if(t1>=c1_dir_[t2]){ // direct get
            int dirlvl = t_to_dirlvl_[t2];
            int first_index = les_dir_square_first_index_[dirlvl];
            int row = t2 - blkr1(dirlvl-1);
            int col = t1 - c1_dir_[t2];
            int width = blkdirheight_[dirlvl];
            int index = first_index + row*width + col;
            ZMatrixMap(M, size1_, size2_).noalias() = Eigen::Map<Eigen::Matrix<cplx, -1, -1, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(les_dir_square_.data() + index, size1_, size2_, Eigen::InnerStride<>(len_les_dir_square_));
        }else{               // compressed get
            int b=time2block(t2,t1);
            les_.get_compress(t2-blkr1_[b],t1-blkc1_[b],b,M,print);
        }
    }else{
        if( t1 >= tstpmk_) {
          ZMatrixMap(M, size1_, size2_) = ZMatrixMap(curr_timestep_les_ptr(t2,t1), size1_, size2_);
        }
        else if(t2>=c1_dir_[t1]){ // direct get
            int dirlvl = t_to_dirlvl_[t1];
            int first_index = les_dir_square_first_index_[dirlvl];
            int row = t1 - blkr1(dirlvl-1);
            int col = t2 - c1_dir_[t1];
            int width = blkdirheight_[dirlvl];
            int index = first_index + row*width + col;
            ZMatrixMap(M, size1_, size2_).noalias() = Eigen::Map<Eigen::Matrix<cplx, -1, -1, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(les_dir_square_.data() + index, size1_, size2_, Eigen::InnerStride<>(len_les_dir_square_));
        }else{               // compressed get
            int b=time2block(t1,t2);
            les_.get_compress(t1-blkr1_[b],t2-blkc1_[b],b,M);
        }
        ZMatrixMap MMap(M, size1_, size2_);
        MMap = MMap.adjoint().eval();
        MMap *= -1.0;
    }
}

void herm_matrix_hodlr::get_les(int t1, int t2, ZMatrix &M){
    assert(t1<=nt_ && t2<=nt_);
    M.resize(size1_,size2_);

    get_les(t1, t2, M.data());
}

void herm_matrix_hodlr::get_tv(int t1, int t2, cplx* M){
    assert(t1<=nt_ && t2<r_);
    memcpy(M,tvptr(t1,t2),size1_*size2_*sizeof(cplx));
}

void herm_matrix_hodlr::get_tv(int t1, int t2, ZMatrix &M){
    assert(t1<=nt_ && t2<r_);
    M.resize(size1_,size2_);
    get_tv(t1, t2, M.data());
}

void herm_matrix_hodlr::get_tv_trans(int t1, int t2, cplx* M){
    assert(t1<=nt_ && t2<r_);
    memcpy(M,tvptr_trans(t1,t2),size1_*size2_*sizeof(cplx));
}

void herm_matrix_hodlr::get_tv_trans(int t1, int t2, ZMatrix &M){
    assert(t1<=nt_ && t2<r_);
    M.resize(size1_,size2_);
    get_tv_trans(t1, t2, M.data());
}

void herm_matrix_hodlr::set_tv(int t1, int t2, cplx* M){
    assert(t1<=nt_ && t2<r_);
    memcpy(tvptr(t1,t2),M,size1_*size2_*sizeof(cplx));
    ZMatrixMap(tvptr_trans(t1,t2), size2_, size1_) = ZMatrixMap(tvptr(t1,t2), size1_, size2_).transpose();
}

void herm_matrix_hodlr::set_tv(int t1, int t2, ZMatrix &M){
    assert(t1<=nt_ && t2<r_);
    set_tv(t1, t2, M.data());
}

// return t1-th element in dlr tau
void herm_matrix_hodlr::get_mat(int t1, DMatrix &M) {
  assert(t1 < r_ && t1 >=0);
  memcpy(M.data(), mat_ + t1*size1_*size2_, size1_*size2_*sizeof(double));
}

// return (r,size1,size2) array where data[n,i,j] = G^M[beta-tau[n],i,j]
// tau[n] is the n^{th} dlr node on the imaginary time axis
void herm_matrix_hodlr::get_mat_reversed(double *M, double *dlrit2itr) {
  DMatrixMap(M, r_, size1_*size2_) = DMatrixMap(dlrit2itr, r_, r_).transpose() * DMatrixMap(mat_, r_, size1_*size2_);
}

// return (r,size1,size2) array where data[n,i,j] = G^tv[beta-tau[n],i,j]
// tau[n] is the n^{th} dlr node on the imaginary time axis
void herm_matrix_hodlr::get_tv_reversed(int tstp, cplx *M, double *dlrit2itr) {
  ZMatrixMap(M, r_, size1_*size2_) = DMatrixMap(dlrit2itr, r_, r_).transpose() * ZMatrixMap(tvptr(tstp, 0), r_, size1_*size2_);
}

// return (r,size1,size2) array where data[n,i,j] = G^vt[tau[n],i,j]
// tau[n] is the n^{th} dlr node on the imaginary time axis
void herm_matrix_hodlr::get_vt(int tstp, cplx *M, double *dlrit2itr) {
  ZMatrixMap(M, r_, size1_*size2_) = -sig_ * (DMatrixMap(dlrit2itr, r_, r_).transpose() * ZMatrixMap(tvptr_trans(tstp, 0), r_, size1_*size2_)).conjugate();
}

// return value at arbitrary tau point
void herm_matrix_hodlr::get_mat_tau(double tau,double beta,double *it2cf,int *it2cfp, double *dlrrf, DMatrix &M){
  get_mat_tau(tau, beta, it2cf, it2cfp, dlrrf, M.data());
}

void herm_matrix_hodlr::get_mat_tau(double tau,double beta,double *it2cf,int *it2cfp, double *dlrrf, double *M){
  assert(0 <= tau && tau<=beta);

  int es = size1_*size2_;
  int one = 1;

  double *Gij_it = new double[r_ * es];
  double *Gijc_it = new double[r_ * es];
  DMatrixMap(Gij_it, es, r_).noalias() = DMatrixMap(this->matptr(0), r_, es).transpose();

  double tau01 = tau/beta;
  double taurel;

  c_dlr_it2cf(&r_, &size1_, it2cf, it2cfp, Gij_it, Gijc_it);
  c_abs2rel(&one, &tau01, &taurel);
  c_dlr_it_eval(&r_, &size1_, dlrrf, Gijc_it, &taurel, M);

  delete[] Gij_it;
  delete[] Gijc_it;
}

void herm_matrix_hodlr::get_mat_tau(double tau,double beta,double *it2cf,int *it2cfp, double *dlrrf, cplx *M){
  assert(0 <= tau && tau<=beta);

  int es = size1_*size2_;
  int one = 1;

  double *Gij_it = new double[r_ * es];
  double *Gijc_it = new double[r_ * es];
  double *res = new double[es];
  DMatrixMap(Gij_it, es, r_).noalias() = DMatrixMap(this->matptr(0), r_, es).transpose();

  double tau01 = tau/beta;
  double taurel;

  c_dlr_it2cf(&r_, &size1_, it2cf, it2cfp, Gij_it, Gijc_it);
  c_abs2rel(&one, &tau01, &taurel);
  c_dlr_it_eval(&r_, &size1_, dlrrf, Gijc_it, &taurel, res);
  
  ZMatrixMap(M, size1_, size2_).noalias() = DMatrixMap(res, size1_, size2_);

  delete[] Gij_it;
  delete[] Gijc_it;
  delete[] res;
}


void herm_matrix_hodlr::density_matrix(int tstp,double *it2cf,int *it2cfp, double *dlrrf,DMatrix &M){
  assert(M.rows() == size1_ && M.cols() == size2_);
  assert(tstp >= -1 && tstp <= nt_);
    if (tstp == -1) {
        get_mat_tau(1.,1.,it2cf,it2cfp,dlrrf,M);
        M *= (-1.0);
    } else {
        ZMatrix tmpM(size1_,size2_);
        get_les(tstp, tstp, tmpM);
        tmpM *= std::complex<double>(0.0, 1.0 * sig_);
        M=tmpM.real();
    }
}

void herm_matrix_hodlr::density_matrix(int tstp,double *it2cf,int *it2cfp, double *dlrrf, cplx *res){
    if (tstp == -1) {
        get_mat_tau(1.,1.,it2cf,it2cfp,dlrrf,res);
        ZMatrixMap(res, size1_, size1_) *= -1.;
    } else {
        get_les(tstp, tstp, res);
        ZMatrixMap(res, size1_, size1_) *= std::complex<double>(0.0, 1.0 * sig_);
    }
}

// return value at arbitrary tau point
void herm_matrix_hodlr::get_tv_tau(int tstp, double tau, double beta, double *it2cf, int *it2cfp, double *dlrrf, double *dlrit, ZMatrix &M){
  get_tv_tau(tstp, tau, beta, it2cf, it2cfp, dlrrf, dlrit, M.data());
}

void herm_matrix_hodlr::get_tv_tau(int tstp, double tau, double beta, double *it2cf, int *it2cfp, double *dlrrf, double *dlrit, cplx *M){
  assert(0 <= tau && tau<=beta);

  int es = size1_*size2_;
  int one = 1;
  ZMatrixMap(M, size1_, size2_).setZero();
  
  double tau01 = tau/beta;
  double taurel;
  c_abs2rel(&one, &tau01, &taurel);

  if(!LU_built_) {
    LU_built_ = true;
    c_dlr_cf2it_init(&r_, dlrrf, dlrit, it2cf_tmp_.data());
    LU_.compute(it2cf_tmp_.transpose());
  }


/*
  double *Gij_it = new double[r_ * es];
  double *Gijc_it = new double[r_ * es];
  double *res = new double[es];

  DMatrixMap(Gij_it, es, r_).noalias() = ZMatrixMap(this->tvptr(tstp,0), r_, es).transpose().real();
  c_dlr_it2cf(&r_, &size1_, it2cf, it2cfp, Gij_it, Gijc_it);
  c_dlr_it_eval(&r_, &size1_, dlrrf, Gijc_it, &taurel, res);
  ZMatrixMap(M, size1_, size2_) += DMatrixMap(res, size1_, size2_);

  DMatrixMap(Gij_it, es, r_).noalias() = ZMatrixMap(this->tvptr(tstp,0), r_, es).transpose().imag();
  c_dlr_it2cf(&r_, &size1_, it2cf, it2cfp, Gij_it, Gijc_it);
  c_dlr_it_eval(&r_, &size1_, dlrrf, Gijc_it, &taurel, res);
  ZMatrixMap(M, size1_, size2_) += cplx(0.,1.) * DMatrixMap(res, size1_, size2_);

  delete[] Gij_it;
  delete[] Gijc_it;
  delete[] res;
*/


  DMatrixMap(Gijc_it_, es, r_).noalias() = LU_.solve(ZMatrixMap(this->tvptr(tstp,0), r_, es).real()).transpose();
  c_dlr_it_eval(&r_, &size1_, dlrrf, Gijc_it_, &taurel, res_);
//  std::cout << "eval err_r: " << (ZMatrixMap(M, size1_, size2_).real() - DMatrixMap(res_, size1_, size2_)).norm() << std::endl;
  ZMatrixMap(M, size1_, size2_) += DMatrixMap(res_, size1_, size2_);
  DMatrixMap(Gijc_it_, es, r_).noalias() = LU_.solve(ZMatrixMap(this->tvptr(tstp,0), r_, es).imag()).transpose();
  c_dlr_it_eval(&r_, &size1_, dlrrf, Gijc_it_, &taurel, res_);
//  std::cout << "eval err_i: " << (ZMatrixMap(M, size1_, size2_).imag() - DMatrixMap(res_, size1_, size2_)).norm() << std::endl;
  ZMatrixMap(M, size1_, size2_) += cplx(0.,1.) * DMatrixMap(res_, size1_, size2_);

}

void herm_matrix_hodlr::set_mat(int t1, DMatrix &M) {
  assert(t1 < r_ && t1 >=0);
  memcpy(mat_ + t1*size1_*size2_,M.data(),size1_*size2_*sizeof(double));
}

void herm_matrix_hodlr::get_ret_curr(int t1, int t2, cplx *M){
    assert(t1 <= tstpmk_+k_);
    assert(t1 >= tstpmk_);
    if(t1>=t2){
        memcpy(M,curr_timestep_ret_ptr(t1,t2),size1_*size2_*sizeof(cplx));
    }else{
        memcpy(M,curr_timestep_ret_ptr(t2,t1),size1_*size2_*sizeof(cplx));
        ZMatrixMap MMap(M, size1_, size2_);
        MMap = MMap.adjoint().eval();
        MMap *= -1.0;
    }
}

void herm_matrix_hodlr::get_ret_curr(int t1, int t2, ZMatrix &M){
    assert(t1 <= tstpmk_+k_);
    assert(t1 >= tstpmk_);
    M.resize(size1_,size2_);
    get_ret_curr(t1, t2, M.data());
}


void herm_matrix_hodlr::get_les_curr(int t1, int t2, cplx *M){
    assert(t2 <= tstpmk_+k_);
    assert(t2 >= tstpmk_);
    if(t1<=t2){
        memcpy(M,curr_timestep_les_ptr(t1,t2),size1_*size2_*sizeof(cplx));    
    }else{
        memcpy(M,curr_timestep_les_ptr(t2,t1),size1_*size2_*sizeof(cplx)); 
        ZMatrixMap MMap(M, size1_, size2_);
        MMap = MMap.adjoint().eval();
        MMap *= -1.0;
    }
}

void herm_matrix_hodlr::get_les_curr(int t1, int t2, ZMatrix &M){
    assert(t2 <= tstpmk_+k_);
    assert(t2 >= tstpmk_);
    M.resize(size1_,size2_);
    get_les_curr(t1, t2, M.data());
}


double* herm_matrix_hodlr::matptr(int i){
  assert(i < r_ && i >=0);
  return mat_ + i * size1_ * size2_;
}

cplx* herm_matrix_hodlr::tvptr(int t, int tau) {
  assert(t <= nt_ && tau < r_);
  return tv_.data() + t*r_*size1_*size2_ + tau*size1_*size2_;
}

cplx* herm_matrix_hodlr::tvptr_trans(int t, int tau) {
  assert(t <= nt_ && tau < r_);
  return tv_.data_trans() + t*r_*size1_*size2_ + tau*size1_*size2_;
}

std::complex<double>* herm_matrix_hodlr::retptr_col(int t1, int t2){
  return ret_.drtricol()+time2direct_col(t1,t2)*size1_*size2_;
}

int herm_matrix_hodlr::time2direct_col(int t1,int t2){
    int bb = t_to_dirlvl_[t1];
    int is = blkndirstart_[bb];
    int bh = blkdirheight_[bb];
    int ttop = bb==0?0:blkr1_[bb-1];    
    int trel = t1-ttop;
    int tprel = t2-ttop;
    int ret = is + tprel*bh + trel - (tprel*(tprel+1))/2;

    return ret;
}


// which block does (t1,t2) pair correspond
// TODO: how is this done faster ?
int herm_matrix_hodlr::time2block(int t1,int t2){
    assert(t2<c1_dir_[t1]);
    for(int i=0;i<nbox_;i++){
        if(blkc1_[i]<=t2 && t2<=blkc2_[i] && blkr1_[i]<=t1 && t1<=blkr2_[i]) return i;
    }
    assert(("Couldnt determine the block",1==2));
    return -1;
}

void herm_matrix_hodlr::get_timestep_curr(int tstp,herm_matrix_hodlr &G){
    assert(tstp>=tstpmk_ && tstp <= tstpmk_+k_);
    if(tstp==-1){
        memcpy(G.mat(),mat_,r_ * size1_ * size2_ * sizeof(double));
    }else{ // set last kt timesteps together
        memcpy(G.curr_timestep_ret_ptr(tstp,0),this->curr_timestep_ret_ptr(tstp,0),(nt_ +1) * size1_ * size2_* sizeof(cplx));
        memcpy(G.curr_timestep_les_ptr(0,tstp),this->curr_timestep_les_ptr(0,tstp),(nt_ +1) * size1_ * size2_ * sizeof(cplx));
        memcpy(G.tvptr(tstp,0),this->tvptr(tstp,0),r_ * size1_ * size2_ * sizeof(cplx));
        memcpy(G.tvptr_trans(tstp,0),this->tvptr_trans(tstp,0),r_ * size1_ * size2_ * sizeof(cplx));
    }
}

// Copy last kt timesteps
void herm_matrix_hodlr::get_timestep_curr(herm_matrix_hodlr &G){
    G.curr_timestep_ret()=curr_timestep_ret_;
    G.curr_timestep_les()=curr_timestep_les_;
    for(int i=tstpmk_;i<tstpmk_+k_;i++){
        memcpy(G.tvptr(i,0),this->tvptr(i,0),r_ * size1_ * size2_ * sizeof(cplx));
        memcpy(G.tvptr_trans(i,0),this->tvptr_trans(i,0),r_ * size1_ * size2_ * sizeof(cplx));
    }
}

void herm_matrix_hodlr::get_timestep_curr(int tstp,cntr::herm_matrix_timestep<double> &G){
    std::cout << "get_timestep_curr " << tstp << " " << tstpmk_ << " " << k_ << std::endl;
    assert(tstp <= tstpmk_+k_);
    assert(tstp==G.tstp());
    
    if(tstp==-1){
        memcpy(G.matptr(0),mat_,r_ * size1_ * size2_ * sizeof(double));
    }else{ // set last kt timesteps together
        memcpy(G.retptr(0),this->curr_timestep_ret_ptr(tstp,0),(nt_ +1) * size1_ * size2_* sizeof(cplx));
        memcpy(G.lesptr(0),this->curr_timestep_les_ptr(0,tstp),(nt_ +1) * size1_ * size2_ * sizeof(cplx));
        memcpy(G.tvptr(0),this->tvptr(tstp,0),r_ * size1_ * size2_ * sizeof(cplx));
    }
}

// set is only a reasonable operation for directly stored timesteps
void herm_matrix_hodlr::set_timestep_curr(int tstp,herm_matrix_hodlr &G){
    assert(tstp >= -1 && tstp <= nt_ && tstpmk_==G.tstpmk());
    if(tstp==-1){
        memcpy(mat_,G.mat(),r_ * size1_ * size2_ * sizeof(double));
    }else{
        assert(tstp>=tstpmk_ && tstp <= tstpmk_+k_);
        memcpy(this->curr_timestep_ret_ptr(tstp,0),G.curr_timestep_ret_ptr(tstp,0),(nt_ + 1) * size1_ * size2_ * sizeof(cplx));
        memcpy(this->curr_timestep_les_ptr(0,tstp),G.curr_timestep_les_ptr(0,tstp),(nt_ + 1) * size1_ * size2_ * sizeof(cplx));
        memcpy(this->tvptr(tstp,0),G.tvptr(tstp,0),r_ * size1_ * size2_ * sizeof(cplx));
        memcpy(this->tvptr_trans(tstp,0),G.tvptr_trans(tstp,0),r_ * size1_ * size2_ * sizeof(cplx));
    }
}

// Copy last kt timesteps
void herm_matrix_hodlr::set_timestep_curr(herm_matrix_hodlr &G){
    tstpmk_=G.tstpmk();
    for(int i=tstpmk_;i<=tstpmk_+k_;i++){
        set_timestep_curr(i,G);
    }
}

// set is only a reasonable operation for directly stored timesteps
// TODO: There a routine in libdlr to get from an equidistant grid to dlr grid 
void herm_matrix_hodlr::set_timestep_curr(int tstp, cntr::herm_matrix<double> &G){
    assert(tstp >= -1 && tstp <= nt_);
    assert((tstp >= tstpmk_ && tstp <= tstpmk_+k_) || tstp == -1);

    memcpy(this->curr_timestep_ret_ptr(tstp,0),G.retptr(tstp,0),(nt_+1) * size1_ * size2_ * sizeof(cplx));
    memcpy(this->curr_timestep_les_ptr(0,tstp),G.lesptr(0,tstp),(nt_+1) * size1_ * size2_ * sizeof(cplx));
}

// set is only a reasonable operation for directly stored timesteps
// TODO: There a routine in libdlr to get from an equidistant grid to dlr grid 
void herm_matrix_hodlr::set_timestep_curr(int tstp, cntr::herm_matrix_timestep<double> &G){
    assert(tstp >= -1 && tstp <= nt_);
    assert((tstp >= tstpmk_ && tstp <= tstpmk_+k_) || tstp == -1);

    memcpy(this->curr_timestep_ret_ptr(tstp,0),G.retptr(0),(nt_+1) * size1_ * size2_ * sizeof(cplx));
    memcpy(this->curr_timestep_les_ptr(0,tstp),G.lesptr(0),(nt_+1) * size1_ * size2_ * sizeof(cplx));
}


void herm_matrix_hodlr::incr_timestep_curr(int tstp,herm_matrix_hodlr &G,cplx alpha){
    assert(tstp >= -1 && tstp <= nt_);
    
    cplx *x1,*x2;
    double *xM1,*xM2;
    if(tstp==-1){
        int len= r_ * size1_ * size2_;
        xM1=this->matptr(0);
        xM2=G.matptr(0);
        for(int i=0;i<len;i++){
            xM1[i]+=std::real(alpha) * xM2[i];
        }
    }else{
        assert(tstp>=tstpmk_ && tstp <= tstpmk_+k_);
        // std::cout << "incr R" << std::endl;
        // Ret
        int len= (tstp + 1) * size1_ * size2_;
        x1=this->curr_timestep_ret_ptr(tstp,0);
        x2=G.curr_timestep_ret_ptr(tstp,0);
        for(int i=0;i<len;i++){
            x1[i]+=alpha * x2[i];
        }
        // std::cout << "incr L" << std::endl;
        // std::cout << "------------------" << std::endl;
        // Les
        x1=this->curr_timestep_les_ptr(0,tstp);
        x2=G.curr_timestep_les_ptr(0,tstp);
        for(int i=0;i<len;i++){
            x1[i]+=alpha * x2[i];
        }
        // std::cout << "incr M" << std::endl;
        // Mixed component
        len= r_  * size1_ * size2_;
        x1=this->tvptr(tstp,0);
        x2=G.tvptr(tstp,0);
        // std::cout << "incr M1 " << len << std::endl;
        for(int i=0;i<len;i++){
            // std::cout << "incr M2 " << i << std::endl;
            x1[i]+=alpha * x2[i];
        }
        // std::cout << "incr MT" << std::endl;
        // Mixed component transpose
        len= r_  * size1_ * size2_;
        x1=this->tvptr_trans(tstp,0);
        x2=G.tvptr_trans(tstp,0);
        for(int i=0;i<len;i++){
            x1[i]+=alpha * x2[i];
        }
    }
}

void herm_matrix_hodlr::incr_timestep_curr(herm_matrix_hodlr &G,cplx alpha){
    for(int t=tstpmk_;t<=tstpmk_+k_;t++){
        incr_timestep_curr(t,G,alpha);
    }
}

void herm_matrix_hodlr::smul_curr(int tstp,cplx alpha){
    assert(tstp >= -1 && tstp <= nt_);
    cplx *x;
    double *xM;
    if(tstp==-1){
        int len= r_ * size1_ * size2_;
        xM=this->matptr(0);
        for(int i=0;i<len;i++){
            xM[i]*=std::real(alpha);
        }
    }else{
        assert(tstp>=tstpmk_ && tstp <= tstpmk_+k_);
        // Ret
        int len= (tstp + 1) * size1_ * size2_;
        x=this->curr_timestep_ret_ptr(tstp,0);
        for(int i=0;i<len;i++){
            x[i]*=alpha;
        }
        // Les
        x=this->curr_timestep_les_ptr(0,tstp);
        for(int i=0;i<len;i++){
            x[i]*=alpha;
        }
        // Mixed component
        len= r_  * size1_ * size2_;
        x=this->tvptr(tstp,0);
        for(int i=0;i<len;i++){
            x[i]*=alpha;
        }
        // Mixed component transpose
        len= r_  * size1_ * size2_;
        x=this->tvptr_trans(tstp,0);
        for(int i=0;i<len;i++){
            x[i]*=alpha;
        }
    }
}

void herm_matrix_hodlr::smul_curr(cplx alpha){
    for(int t=tstpmk_;t<=tstpmk_+k_;t++){
        smul_curr(t,alpha);
    }
}

void herm_matrix_hodlr::set_memory_usage(void){
    int ret=0;
    int nrows=0,ncols=0,ranks=0;

    // Retarded component
    // compressed represenation
    for(int b=0;b<nbox_;b++){
        nrows=blkr2_[b]-blkr1_[b];
        ncols=blkc2_[b]-blkc1_[b];
        ranks=0;
        for(int i=0;i<size1_;i++){
            for(int j=0;j<size2_;j++){
                ranks+=ret_.data().blocks()[b][i][j].epsrank();
                // std::cout << " rank " << i << " " << j << " " <<  ret_.data().blocks()[b][i][j].epsrank() << std::endl;
            }
        }
        ret+=ranks*(nrows+ncols+1);
        // std::cout << "Ret usage: "  << b << " " << nrows << " " << ncols << " " << ranks << std::endl;
    }
    for(int i=0;i<nt_;i++){
        ret+=ntri_[i];
        // std::cout << "Ret dir " <<  i << " " << ntri_[i] << std::endl;
    }
    
    
    cr_=ret*8.0*2.0/(1024.0*1024.0);
    fr_=size1_*size2_*nt_*(nt_+1)*8.0*2.0/(2.0*1024.0*1024.0);
    std::cout << "Ret " << ret << " " << cr_ << std::endl;

    // Lesser component
    // compressed represenation
    int les=0;
    for(int b=0;b<nbox_;b++){
        nrows=blkr2_[b]-blkr1_[b];
        ncols=blkc2_[b]-blkc1_[b];
        ranks=0;
        for(int i=0;i<size1_;i++){
            for(int j=0;j<size2_;j++){
                ranks+=les_.data().blocks()[b][i][j].epsrank();
                // std::cout << " rank " << i << " " << j << " " <<  ret_.data().blocks()[b][i][j].epsrank() << std::endl;
            }
        }
        les+=ranks*(nrows+ncols+1);
        // std::cout << "Ret usage: "  << b << " " << nrows << " " << ncols << " " << ranks << std::endl;
    }
    for(int i=0;i<nt_;i++){
        les+=ntri_[i];
        // std::cout << "Ret dir " <<  i << " " << ntri_[i] << std::endl;
    }
    // std::cout << "Ret " << ret << " " << nt_*(nt_+1)/2.0 << std::endl;
    
    cl_=les*8.0*2.0/(1024.0*1024.0);
    fl_=size1_*size2_*nt_*(nt_+1)*8.0*2.0/(2.0*1024.0*1024.0);
    std::cout << "les " << les << " " << cl_ << std::endl;
}

void herm_matrix_hodlr::print_memory_usage(void){
    int ret=0;
    int nrows=0,ncols=0,ranks=0;

    // Retarded component
    // compressed represenation
    for(int b=0;b<nbox_;b++){
        nrows=blkr2_[b]-blkr1_[b];
        ncols=blkc2_[b]-blkc1_[b];
        ranks=0;
        for(int i=0;i<size1_;i++){
            for(int j=0;j<size2_;j++){
                ranks+=ret_.data().blocks()[b][i][j].epsrank();
                // std::cout << " rank " << i << " " << j << " " <<  ret_.data().blocks()[b][i][j].epsrank() << std::endl;
            }
        }
        ret+=ranks*(nrows+ncols+1);
        // std::cout << "Ret usage: "  << b << " " << nrows << " " << ncols << " " << ranks << std::endl;
    }
    for(int i=0;i<nt_;i++){
        ret+=ntri_[i];
        // std::cout << "Ret dir " <<  i << " " << ntri_[i] << std::endl;
    }
    // std::cout << "Ret " << ret << " " << nt_*(nt_+1)/2.0 << std::endl;
    double C=ret*8.0*2.0/(1024.0*1024.0);
    double F=size1_*size2_*nt_*(nt_+1)*8.0*2.0/(2.0*1024.0*1024.0);
    cr_=C;
    fr_=F;
    std::cout << "Estimated memory usage for RET compressed [Mb]/full [Mb]/ratio : " << C << " / " << F << " " << C/F << std::endl;

    // Lesser component
    // compressed represenation
    int les=0;
    for(int b=0;b<nbox_;b++){
        nrows=blkr2_[b]-blkr1_[b];
        ncols=blkc2_[b]-blkc1_[b];
        ranks=0;
        for(int i=0;i<size1_;i++){
            for(int j=0;j<size2_;j++){
                ranks+=les_.data().blocks()[b][i][j].epsrank();
                // std::cout << " rank " << i << " " << j << " " <<  ret_.data().blocks()[b][i][j].epsrank() << std::endl;
            }
        }
        les+=ranks*(nrows+ncols+1);
        // std::cout << "Ret usage: "  << b << " " << nrows << " " << ncols << " " << ranks << std::endl;
    }
    for(int i=0;i<nt_;i++){
        les+=ntri_[i];
        // std::cout << "Ret dir " <<  i << " " << ntri_[i] << std::endl;
    }
    // std::cout << "Ret " << ret << " " << nt_*(nt_+1)/2.0 << std::endl;
    double CL=les*8.0*2.0/(1024.0*1024.0);
    double FL=size1_*size2_*nt_*(nt_+1)*8.0*2.0/(2.0*1024.0*1024.0);
    cl_=CL;
    fl_=FL;
    std::cout << "Estimated memory usage for LES compressed [Mb]/full [Mb]/ratio : " << CL << " / " << FL << " " << CL/FL << std::endl;
}

void herm_matrix_hodlr::write_to_hdf5(hid_t group_id,bool storeSVD) {
    set_memory_usage();
    hid_t  sub_group_id = create_group(group_id, "parm");
    hid_t keldysh_id;
    // Store parameters
    store_int_attribute_to_hid(sub_group_id, std::string("ntau"), ntau_);
    store_int_attribute_to_hid(sub_group_id, std::string("r"), r_);
    store_int_attribute_to_hid(sub_group_id, std::string("nt"), nt_);
    store_int_attribute_to_hid(sub_group_id, std::string("sig"), sig_);
    store_int_attribute_to_hid(sub_group_id, std::string("size1"), size1_);
    store_int_attribute_to_hid(sub_group_id, std::string("size2"), size2_);
    store_int_attribute_to_hid(sub_group_id, std::string("ndir"),ndir_);
    store_int_attribute_to_hid(sub_group_id, std::string("nbox"),nbox_);
    store_int_attribute_to_hid(sub_group_id, std::string("nlvl"),nlvl_);
    store_int_attribute_to_hid(sub_group_id, std::string("maxdir"),maxdir_);
    store_int_attribute_to_hid(sub_group_id, std::string("k"),k_);
    // store_int_attribute_to_hid(group_id, std::string("ndir"),ndir_); TODO - what is the problem ?
    store_int_attribute_to_hid(sub_group_id, std::string("tstpmk"),tstpmk_);
    store_int_attribute_to_hid(sub_group_id, std::string("built_blocks"),built_blocks_);
    close_group(sub_group_id);

    // Geometry of hodlr
    sub_group_id = create_group(group_id, "geometry");

    hsize_t len_shape = 3, shape[3];
    shape[0]=blkr1_.size();
    shape[1]=1;shape[2]=1;

    shape[0]=blkr1_.size();
    store_array_to_hid(sub_group_id,std::string("blkr1"),blkr1_.data(),shape,len_shape,H5T_NATIVE_INT);
    store_array_to_hid(sub_group_id,std::string("blkr2"),blkr2_.data(),shape,len_shape,H5T_NATIVE_INT);
    store_array_to_hid(sub_group_id,std::string("blkc1"),blkc1_.data(),shape,len_shape,H5T_NATIVE_INT);
    store_array_to_hid(sub_group_id,std::string("blkc2"),blkc2_.data(),shape,len_shape,H5T_NATIVE_INT);

    shape[0]=blklevel_.size();
    store_array_to_hid(sub_group_id,std::string("blklevel"),blklevel_.data(),shape,len_shape,H5T_NATIVE_INT);
    shape[0]=blkdirheight_.size();
    store_array_to_hid(sub_group_id,std::string("blkdirheight"),blkdirheight_.data(),shape,len_shape,H5T_NATIVE_INT);
    shape[0]=blkndirstart_.size();
    store_array_to_hid(sub_group_id,std::string("blkndirstart"),blkndirstart_.data(),shape,len_shape,H5T_NATIVE_INT);
    shape[0]=c1_dir_.size();
    store_array_to_hid(sub_group_id,std::string("c1_dir"),c1_dir_.data(),shape,len_shape,H5T_NATIVE_INT);
    shape[0]=r2_dir_.size();
    store_array_to_hid(sub_group_id,std::string("r2_dir"),r2_dir_.data(),shape,len_shape,H5T_NATIVE_INT);
    shape[0]=mapdirc_r_.size();
    store_array_to_hid(sub_group_id,std::string("mapdirc_r"),mapdirc_r_.data(),shape,len_shape,H5T_NATIVE_INT);
    shape[0]=mapdirc_c_.size();
    store_array_to_hid(sub_group_id,std::string("mapdirc_c"),mapdirc_c_.data(),shape,len_shape,H5T_NATIVE_INT);
    shape[0]=mapdirr_c_.size();
    store_array_to_hid(sub_group_id,std::string("mapdirr_c"),mapdirr_c_.data(),shape,len_shape,H5T_NATIVE_INT);
    shape[0]=mapdirr_r_.size();
    store_array_to_hid(sub_group_id,std::string("mapdirr_r"),mapdirr_r_.data(),shape,len_shape,H5T_NATIVE_INT);
    shape[0]=ntri_.size();
    store_array_to_hid(sub_group_id,std::string("ntri"),ntri_.data(),shape,len_shape,H5T_NATIVE_INT);
    shape[0]=blklen_.size();
    store_array_to_hid(sub_group_id,std::string("blklen"),blklen_.data(),shape,len_shape,H5T_NATIVE_INT);
    close_group(sub_group_id);

    // Store data
    // Ret
    assert(size1_==ret_.data().size1() && size2_==ret_.data().size2() && nbox_==ret_.data().nbox());
    sub_group_id = create_group(group_id, "ret");

    std::cout << "values " << cr_ <<" " <<fr_<< std::endl;
    store_double_attribute_to_hid(sub_group_id,"cr",cr_);
    store_double_attribute_to_hid(sub_group_id,"fr",fr_);

    // shape[0]=ret_.ndir();
    // shape[1]=size1_;shape[2]=size2_;
    // store_cplx_array_to_hid(sub_group_id,"tricol",ret_.drtricol(),shape,len_shape);
    

    for(int b=0;b<nbox_;b++){
        for(int i=0;i<size1_;i++){
            for(int j=0;j<size2_;j++){
                keldysh_id = create_group(sub_group_id,std::to_string(b)+"_"+std::to_string(i)+"_"+std::to_string(j));
                store_int_attribute_to_hid(keldysh_id,"epsrank",ret_.data().blocks()[b][i][j].epsrank());
                store_int_attribute_to_hid(keldysh_id,"rows",ret_.data().blocks()[b][i][j].rows());
                store_int_attribute_to_hid(keldysh_id,"cols",ret_.data().blocks()[b][i][j].cols());
                if(storeSVD){
                    // std::cout << "inside " << storeSVD << std::endl;
                    int cols=ret_.data().blocks()[b][i][j].cols();
                    int rows=ret_.data().blocks()[b][i][j].rows();
                    int epsrank=ret_.data().blocks()[b][i][j].epsrank();

                    hsize_t len_shapeR = 2, shapeR[2];
                    shapeR[0]=rows; shapeR[1]=epsrank;
                    store_cplx_array_to_hid(keldysh_id,"U",ret_.data().blocks()[b][i][j].U().data(),shapeR,len_shapeR);

                    shapeR[0]=cols; shapeR[1]=epsrank;
                    store_cplx_array_to_hid(keldysh_id,"V",ret_.data().blocks()[b][i][j].V().data(),shapeR,len_shapeR);

                    hsize_t len_shapeR1 = 1,shapeR1[1];
                    shapeR1[0]=epsrank;
                    store_array_to_hid(keldysh_id,"S",ret_.data().blocks()[b][i][j].S().data(),shapeR1,len_shapeR1,H5T_NATIVE_DOUBLE);
                }
                close_group(keldysh_id);   
            }
        }
    }

    close_group(sub_group_id);

    // Les
    assert(size1_==les_.data().size1() && size2_==les_.data().size2() && nbox_==les_.data().nbox());
    sub_group_id = create_group(group_id, "les");

    std::cout << "values " << cl_ <<" " <<fl_<< std::endl;
    store_double_attribute_to_hid(sub_group_id,"cl",cl_);
    store_double_attribute_to_hid(sub_group_id,"fl",fl_);

    shape[0]=len_les_dir_square_;
    shape[1]=size1_;shape[2]=size2_;
    store_cplx_array_to_hid(sub_group_id,"les_dir_square",les_dir_square_.data(),shape,len_shape);

    // les
    sub_group_id = create_group(group_id, "les");

    shape[0]=size1_;
    shape[1]=size2_;
    shape[2]=len_les_dir_square_;
    store_cplx_array_to_hid(sub_group_id,"les_square",les_dir_square_.data(),shape,len_shape);

    for(int b=0;b<nbox_;b++){
        for(int i=0;i<size1_;i++){
            for(int j=0;j<size2_;j++){
                keldysh_id = create_group(sub_group_id, std::to_string(b)+"_"+std::to_string(i)+"_"+std::to_string(j));
                store_int_attribute_to_hid(keldysh_id,"epsrank",les_.data().blocks()[b][i][j].epsrank());
                store_int_attribute_to_hid(keldysh_id,"rows",les_.data().blocks()[b][i][j].rows());
                store_int_attribute_to_hid(keldysh_id,"cols",les_.data().blocks()[b][i][j].cols());

                if(storeSVD){
                    int cols=les_.data().blocks()[b][i][j].cols();
                    int rows=les_.data().blocks()[b][i][j].rows();
                    int epsrank=les_.data().blocks()[b][i][j].epsrank();

                    hsize_t len_shapeR = 2, shapeR[2];
                    shapeR[0]=rows; shapeR[1]=epsrank;
                    store_cplx_array_to_hid(keldysh_id,"U",les_.data().blocks()[b][i][j].U().data(),shapeR,len_shapeR);

                    shapeR[0]=cols; shapeR[1]=epsrank;
                    store_cplx_array_to_hid(keldysh_id,"V",les_.data().blocks()[b][i][j].V().data(),shapeR,len_shapeR);

                    hsize_t len_shapeR1 = 1,shapeR1[1];
                    shapeR1[0]=epsrank;
                    store_array_to_hid(keldysh_id,"S",les_.data().blocks()[b][i][j].S().data(),shapeR1,len_shapeR1,H5T_NATIVE_DOUBLE);
                }
                close_group(keldysh_id);   
            }
        }
    }
    close_group(sub_group_id);

    // tv
    sub_group_id = create_group(group_id, "tv");
    hsize_t len_shapeTV = 4, shapeTV[4];
    shapeTV[0]=nt_; shapeTV[1]=ntau_;shapeTV[2]=size1_;shapeTV[3]=size2_;
    store_cplx_array_to_hid(sub_group_id,"data",tv_.data(),shapeTV,len_shapeTV);

    // Mat
    sub_group_id = create_group(group_id, "mat");
    hsize_t len_shapeM = 3, shapeM[3];
    shapeM[0]=r_; shapeM[1]=size1_;shapeM[2]=size2_;
    store_array_to_hid(sub_group_id,"data",mat_,shapeM,len_shapeM,H5T_NATIVE_DOUBLE);

    close_group(sub_group_id);

}

void herm_matrix_hodlr::write_to_hdf5(const char *filename,bool storeSVD) {
    hid_t file_id = open_hdf5_file(filename);
    this->write_to_hdf5(file_id,storeSVD);
    close_hdf5_file(file_id);
}

void herm_matrix_hodlr::write_rho_to_hdf5(h5e::File &out, double *it2cf,int *it2cfp, double *dlrrf, std::string label) {
    ZMatrix rho_t((nt_+1)*size1_*size1_, 1);
    for(int i = -1; i < nt_; i++) {
      density_matrix(i, it2cf, it2cfp, dlrrf, rho_t.data() + (i+1)*size1_*size1_);
    }
    
    h5e::dump(out, label, rho_t);
}

void herm_matrix_hodlr::write_GR0_to_hdf5(h5e::File &out, std::string label) {
    ZMatrix GR_t(nt_*size1_*size1_, 1);
    for(int i = 0; i < nt_; i++) {
      get_ret(i, 0, GR_t.data() + i*size1_*size1_);
    }
    
    h5e::dump(out, label + "/R0", GR_t);
}

void herm_matrix_hodlr::write_GL0_to_hdf5(h5e::File &out, std::string label) {
    ZMatrix GL_t(nt_*size1_*size1_, 1);
    for(int i = 0; i < nt_; i++) {
      get_les(0, i, GL_t.data() + i*size1_*size1_);
    }
    
    h5e::dump(out, label + "/L0", GL_t);
}

void herm_matrix_hodlr::write_rank_to_hdf5(h5e::File &out, std::string label) {
    DColVector ranksR(nlvl_*size1_*size1_);
    DColVector ranksL(nlvl_*size1_*size1_);

    for(int b = 0, i = 0; b < nlvl_; i+=std::pow(2,b), b++) {
      for(int j = 0; j < size1_; j++) {
        for(int k = 0; k < size1_; k++) {
          std::cout << b << " " << i << " " << j << " " << k << std::endl;
          ranksR(b*size1_*size1_ + j*size1_ + k) = ret_.data().blocks()[i][j][k].epsrank();
          ranksL(b*size1_*size1_ + j*size1_ + k) = les_.data().blocks()[i][j][k].epsrank();
        }
      }
    }
    
    h5e::dump(out, label + "/ranksR", ranksR);
    h5e::dump(out, label + "/ranksL", ranksL);
}


void herm_matrix_hodlr::write_to_hdf5(h5e::File &out, std::string label, bool inc_data, bool inc_geometry) {
  h5e::dump(out, label + std::string("nt"), nt_);
  h5e::dump(out, label + std::string("size1"), size1_);
  h5e::dump(out, label + std::string("size2"), size2_);
  h5e::dump(out, label + std::string("ndir"), ndir_);
  h5e::dump(out, label + std::string("nbox"), nbox_);
  h5e::dump(out, label + std::string("nlvl"), nlvl_);
  h5e::dump(out, label + std::string("maxdir"), maxdir_);
  h5e::dump(out, label + std::string("k"), k_);
  h5e::dump(out, label + std::string("tstpmk"), tstpmk_);
  h5e::dump(out, label + std::string("built_blocks"), built_blocks_);

  if(inc_geometry) {
    h5e::dump(out, "geometry/blkr1", IMatrixMap(blkr1_.data(), blkr1_.size(), 1));
    h5e::dump(out, "geometry/blkr2", IMatrixMap(blkr2_.data(), blkr1_.size(), 1));
    h5e::dump(out, "geometry/blkc1", IMatrixMap(blkc1_.data(), blkr1_.size(), 1));
    h5e::dump(out, "geometry/blkc2", IMatrixMap(blkc2_.data(), blkr1_.size(), 1));
    h5e::dump(out, "geometry/blklevel", IMatrixMap(blklevel_.data(), blklevel_.size(), 1));
    h5e::dump(out, "geometry/blkdirheight", IMatrixMap(blkdirheight_.data(), blkdirheight_.size(), 1));
    h5e::dump(out, "geometry/blkndirstart", IMatrixMap(blkndirstart_.data(), blkndirstart_.size(), 1));
    h5e::dump(out, "geometry/c1_dir", IMatrixMap(c1_dir_.data(), c1_dir_.size(), 1));
    h5e::dump(out, "geometry/r2_dir", IMatrixMap(r2_dir_.data(), r2_dir_.size(), 1));
    h5e::dump(out, "geometry/mapdirc_r", IMatrixMap(mapdirc_r_.data(), mapdirc_r_.size(), 1));
    h5e::dump(out, "geometry/mapdirc_c", IMatrixMap(mapdirc_c_.data(), mapdirc_c_.size(), 1));
    h5e::dump(out, "geometry/mapdirr_c", IMatrixMap(mapdirr_c_.data(), mapdirr_c_.size(), 1));
    h5e::dump(out, "geometry/mapdirr_r", IMatrixMap(mapdirr_r_.data(), mapdirr_r_.size(), 1));
    h5e::dump(out, "geometry/ntri", IMatrixMap(ntri_.data(), ntri_.size(), 1));
    h5e::dump(out, "geometry/blklen", IMatrixMap(blklen_.data(), blklen_.size(), 1));
  }

  for(int b = 0; b < nbox_; b++) {
    for(int i = 0; i < size1_; i++) {
      for(int j = 0; j < size2_; j++) {
        std::string prelabel = label + "ret/" + std::to_string(b)+"_"+std::to_string(i)+"_"+std::to_string(j)+"/";
        h5e::dump(out, prelabel + "epsrank", ret_.data().blocks()[b][i][j].epsrank());
        h5e::dump(out, prelabel + "rows", ret_.data().blocks()[b][i][j].rows());
        h5e::dump(out, prelabel + "cols", ret_.data().blocks()[b][i][j].cols());
        if(inc_data) {
          h5e::dump(out, prelabel + "U", ret_.data().blocks()[b][i][j].U());
          h5e::dump(out, prelabel + "V", ret_.data().blocks()[b][i][j].V());
          h5e::dump(out, prelabel + "S", ret_.data().blocks()[b][i][j].S());
        }
      }
    }
  }

  for(int b = 0; b < nbox_; b++) {
    for(int i = 0; i < size1_; i++) {
      for(int j = 0; j < size2_; j++) {
        std::string prelabel = label + "les/"+std::to_string(b)+"_"+std::to_string(i)+"_"+std::to_string(j)+"/";
        h5e::dump(out, prelabel + "epsrank", les_.data().blocks()[b][i][j].epsrank());
        h5e::dump(out, prelabel + "rows", les_.data().blocks()[b][i][j].rows());
        h5e::dump(out, prelabel + "cols", les_.data().blocks()[b][i][j].cols());
        if(inc_data) {
          h5e::dump(out, prelabel + "U", les_.data().blocks()[b][i][j].U());
          h5e::dump(out, prelabel + "V", les_.data().blocks()[b][i][j].V());
          h5e::dump(out, prelabel + "S", les_.data().blocks()[b][i][j].S());
        }
      }
    }
  }
  if(inc_data) {
    h5e::dump(out, label + "les/dir", ZMatrixMap(les_dir_square_.data(),size1_*size2_*len_les_dir_square_,1));
    h5e::dump(out, label + "ret/drtricol", ZMatrixMap(ret_.drtricol(), ndir_*size1_*size2_,1));
  }
}


void herm_matrix_hodlr::write_mat_to_hdf5(h5e::File &out) {
    h5e::dump(out, "mat", DMatrixMap(mat_, r_*size1_*size1_, 1));
}
void herm_matrix_hodlr::write_mat_to_hdf5(h5e::File &out, std::string label) {
    h5e::dump(out, label+"/mat", DMatrixMap(mat_, r_*size1_*size1_, 1));
}

void herm_matrix_hodlr::write_curr_to_hdf5(hid_t group_id,int tstp) {
    hid_t sub_group_id = create_group(group_id, "t"+std::to_string(tstp));
    // Retarded
    hsize_t len_shape = 3, shape[3];

    shape[0]=nt_+1;
    shape[1]=size1_;shape[2]=size2_;
    store_cplx_array_to_hid(sub_group_id,"ret",curr_timestep_ret_ptr(tstp,0),shape,len_shape);

    // Lesser
    shape[0]=nt_+1;
    shape[1]=size1_;shape[2]=size2_;
    store_cplx_array_to_hid(sub_group_id,"les",curr_timestep_les_ptr(0,tstp),shape,len_shape);

}


void herm_matrix_hodlr::write_curr_to_hdf5(std::string &filename,int tstp){
    hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR,H5P_DEFAULT);
    this->write_curr_to_hdf5(file_id,tstp);
    close_hdf5_file(file_id);
}

// Routines to check the diffence between currently stored timesteps in hodlr and herm_matrix

double distance_norm2_curr_tv(int tstp, herm_matrix_hodlr &g1, herm_matrix_hodlr &g2) {
    int size1 = g1.size1(), size2 = g1.size2(), ntau = g2.ntau(), r = g1.r(), es = size1*size2, i;
    double err = 0.0;
    assert(g1.size1() == g2.size1());

    for (int i = 0; i < r; i++) {
        ZMatrixMap M1(g1.tvptr(tstp,i), size1, size2);
        ZMatrixMap M2(g2.tvptr(tstp,i), size1, size2);
        err += (M1-M2).squaredNorm();
    }
    return std::sqrt(err);
}

double distance_norm2_curr_les(int tstp, herm_matrix_hodlr &g1, herm_matrix_hodlr &g2) {
    int size1 = g1.size1(), size2 = g1.size2(), ntau = g2.ntau(), r = g1.r(), es = size1*size2, i;
    double err = 0.0;
    assert(g1.size1() == g2.size1());

    for (int i = 0; i <= tstp; i++) {
        ZMatrixMap M1(g1.curr_timestep_les_ptr(i,tstp), size1, size2);
        ZMatrixMap M2(g2.curr_timestep_les_ptr(i,tstp), size1, size2);
        err += (M1-M2).squaredNorm();
    
    }
    return std::sqrt(err);
}


double distance_norm2_curr_ret(int tstp, herm_matrix_hodlr &g1, herm_matrix_hodlr &g2) {
    int size1 = g1.size1(), size2 = g1.size2(), ntau = g2.ntau(), r = g1.r(), es = size1*size2, i;
    double err = 0.0;
    assert(g1.size1() == g2.size1());

    for (int i = 0; i <= tstp; i++) {
        ZMatrixMap M1(g1.curr_timestep_ret_ptr(tstp,i), size1, size2);
        ZMatrixMap M2(g2.curr_timestep_ret_ptr(tstp,i), size1, size2);
        err += (M1-M2).squaredNorm();
    
    }
    return std::sqrt(err);
}


double distance_norm2_curr(int tstp, herm_matrix_hodlr &g1, herm_matrix_hodlr &g2) {
    int size1 = g1.size1(), size2 = g1.size2(), ntau = g2.ntau(), r = g1.r(), es = size1*size2, i;
    double err = 0.0;
    assert(g1.size1() == g2.size1());
    if (tstp == -1) {
        for (int i = 0; i < r; i++) {
            // assert("Mat not yet implemented");
            DMatrixMap M1(g1.matptr(i), size1, size2);
            DMatrixMap M2(g2.matptr(i), size1, size2);
            err += (M1-M2).squaredNorm();
        }
    } else {
        for (int i = 0; i <= tstp; i++) {
            ZMatrixMap M1(g1.curr_timestep_ret_ptr(tstp,i), size1, size2);
            ZMatrixMap M2(g2.curr_timestep_ret_ptr(tstp,i), size1, size2);
            err += (M1-M2).squaredNorm();
        }

        for (int i = 0; i <= tstp; i++) {
            ZMatrixMap M1(g1.curr_timestep_les_ptr(i,tstp), size1, size2);
            ZMatrixMap M2(g2.curr_timestep_les_ptr(i,tstp), size1, size2);
            err += (M1-M2).squaredNorm();
        }

        for (int i = 0; i < r; i++) {
            ZMatrixMap M1(g1.tvptr(tstp,i), size1, size2);
            ZMatrixMap M2(g2.tvptr(tstp,i), size1, size2);
            err += (M1-M2).squaredNorm();
        }
    }
    return std::sqrt(err);
}

double distance_norm2_curr(herm_matrix_hodlr &g1, herm_matrix_hodlr &g2){
    assert(g1.tstpmk()==g2.tstpmk() && g1.size1()==g2.size1() && g1.size2()==g2.size2());
    double err;
    for(int t=g1.tstpmk();t<=g1.tstpmk()+g1.k();t++){
        err=distance_norm2_curr(t,g1,g2);
    }
    return err;
}

double distance_norm2(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2, double *it2cf, int *it2cfp, double *dlrrf){
    int size1 = g1.size1(), size2 = g1.size2(), ntau = g2.ntau()+1, r = g1.r(), es = size1*size2;
    double err = 0.0;
    int one = 1;

    assert(g1.size1() == g2.size1());
    
    if (tstp == -1) {
      // Get equidistant tau grid
      double *eqpts_rel = new double[ntau];
      c_eqpts_rel(&ntau, eqpts_rel);

      // Get DLR coefficients
      double *Gij_it = new double[r*es];
      double *Gijc_it = new double[r*es];
      double *res = new double[es];
      DMatrixMap(Gij_it, es, r).noalias() = DMatrixMap(g1.matptr(0), r, es).transpose();
      c_dlr_it2cf(&r, &size1, it2cf, it2cfp, Gij_it, Gijc_it);

      // Measure distance
      for(int tau = 0; tau < ntau; tau++) {
        c_dlr_it_eval(&r, &size1, dlrrf, Gijc_it, eqpts_rel+tau, res);
        err += (DColVectorMap(res, es) - ZColVectorMap(g2.matptr(tau), es)).squaredNorm();
      }
      
      delete [] eqpts_rel;
      delete [] Gij_it;
      delete [] Gijc_it;
      delete [] res;
    }
    else {
      //Ret and Lesser
      for (int i = 0; i <= tstp; i++) {
        ZMatrix M1(size1,size2),M2(size1,size2);
        g1.get_ret(tstp,i,M1);
        g2.get_ret(tstp,i,M2);
        err += (M1-M2).squaredNorm();

        g1.get_les(i,tstp,M1);
        g2.get_les(i,tstp,M2);
        err += (M1-M2).squaredNorm();
      }

      // Get equidistant tau grid
      double *eqpts_rel = new double[ntau];
      c_eqpts_rel(&ntau, eqpts_rel);

      // Get DLR coefficients
      double *Gij_it = new double[r*es];
      double *Gijc_it = new double[r*es];
      double *res = new double[es];
      
      DMatrixMap(Gij_it, es, r).noalias() = ZMatrixMap(g1.tvptr(tstp, 0), r, es).transpose().real();
      c_dlr_it2cf(&r, &size1, it2cf, it2cfp, Gij_it, Gijc_it);

      // Measure distance
      for(int tau = 0; tau < ntau; tau++) {
        c_dlr_it_eval(&r, &size1, dlrrf, Gijc_it, eqpts_rel+tau, res);
        err += (DColVectorMap(res, es) - ZColVectorMap(g2.tvptr(tstp,tau), es).real()).squaredNorm();
      }
      
      DMatrixMap(Gij_it, es, r).noalias() = ZMatrixMap(g1.tvptr(tstp, 0), r, es).transpose().imag();
      c_dlr_it2cf(&r, &size1, it2cf, it2cfp, Gij_it, Gijc_it);

      // Measure distance
      for(int tau = 0; tau < ntau; tau++) {
        c_dlr_it_eval(&r, &size1, dlrrf, Gijc_it, eqpts_rel+tau, res);
        err += (DColVectorMap(res, es) - ZColVectorMap(g2.tvptr(tstp,tau), es).imag()).squaredNorm();
      }

      delete [] eqpts_rel;
      delete [] Gij_it;
      delete [] Gijc_it;
      delete [] res;
    }

    return std::sqrt(err);
}


/// @private
template <int SIZE1,int SIZE2>
double distance_norm2_curr_ret_dispatch(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix_timestep<double> &g2) {
    int size1 = g1.size1();
    int size2 = g1.size2();
    assert(tstp>=-1 && g1.nt()>=tstp && tstp<= g1.tstpmk()+g1.k() && g1.size1() == g2.size1() && tstp==g2.tstp());

    double err = 0.0;
    for (int i = 0; i <= tstp; i++) {
        ZMatrixMap M1(g1.curr_timestep_ret_ptr(tstp,i), size1, size2);
        ZMatrixMap M2(g2.retptr(i), size1, size2);
        err += (M1-M2).squaredNorm();
    }
    return std::sqrt(err);
}

/// @private
template <int SIZE1,int SIZE2>
double distance_norm2_curr_ret_dispatch(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2) {
    int size1 = g1.size1();
    int size2 = g1.size2();
    assert(tstp>=-1 && g1.nt()>=tstp && g1.nt()>=tstp && g1.size1() == g2.size1());
    assert(tstp >= g1.tstpmk() && tstp<=g1.tstpmk()+g1.k());


    double err = 0.0;
    for (int i = 0; i <= tstp; i++) {
        ZMatrixMap M1(g1.curr_timestep_ret_ptr(tstp,i), size1, size2);
        ZMatrixMap M2(g2.retptr(tstp,i), size1, size2);
        // std::cout << "Val: " << i << " " << M1 << " " << M2 << std::endl;
        err += (M1-M2).squaredNorm();
    }
    return std::sqrt(err);
}


double distance_norm2_curr_ret(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix_timestep<double> &g2){
    assert(g1.size1() == g2.size1());
    assert(tstp >= g1.tstpmk() && tstp<=g1.tstpmk()+g1.k());
    assert(g1.nt() >= tstp);
    assert(g2.tstp()==tstp);

    if(g1.size1()==1 && g1.size2()==1){
        return distance_norm2_curr_ret_dispatch<1,1>(tstp, g1, g2);    
    }else{
        return distance_norm2_curr_ret_dispatch<LARGESIZE,LARGESIZE>(tstp, g1, g2);
    }
}

double distance_norm2_curr_ret(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2){
    // std::cout << "Tstp " << tstp <<" " << g1.tstpmk() << " " << g1.tstpmk()+g1.k() << std::endl;
    assert(g1.size1() == g2.size1());
    assert(tstp >= g1.tstpmk() && tstp<=g1.tstpmk()+g1.k());
    assert(g1.nt() >= tstp);

    if(g1.size1()==1 && g1.size2()==1){
        return distance_norm2_curr_ret_dispatch<1,1>(tstp, g1, g2);    
    }else{
        return distance_norm2_curr_ret_dispatch<LARGESIZE,LARGESIZE>(tstp, g1, g2);
    }
}



template <int SIZE1,int SIZE2>
double distance_norm2_curr_les_dispatch(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2) {
    int size1 = g1.size1();
    int size2 = g1.size2();
    assert(tstp>=-1 && g1.nt()>=tstp && g1.nt()>=tstp && g1.size1() == g2.size1());
    assert(tstp >= g1.tstpmk() && tstp<=g1.tstpmk()+g1.k());


    double err = 0.0;
    for (int i = 0; i <= tstp; i++) {
        ZMatrixMap M1(g1.curr_timestep_les_ptr(i,tstp), size1, size2);
        ZMatrixMap M2(g2.lesptr(i,tstp), size1, size2);
        // std::cout << "Val: " << i << " " << M1 << " " << M2 << std::endl;
        err += (M1-M2).squaredNorm();
    }
    return std::sqrt(err);
}


double distance_norm2_curr_les(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2){
    assert(g1.size1() == g2.size1());
    assert(tstp >= g1.tstpmk() && tstp<=g1.tstpmk()+g1.k());
    assert(g1.nt() >= tstp);
    // std::cout << "TSTP " << tstp << std::endl;

    if(g1.size1()==1 && g1.size2()==1){
        return distance_norm2_curr_les_dispatch<1,1>(tstp, g1, g2);    
    }else{
        return distance_norm2_curr_les_dispatch<LARGESIZE,LARGESIZE>(tstp, g1, g2);
    }
}

template <int SIZE1,int SIZE2>
double distance_norm2_curr_les_dispatch(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix_timestep<double> &g2) {
    int size1 = g1.size1();
    int size2 = g1.size2();
    assert(tstp>=-1 && g1.nt()>=tstp && tstp <= g1.tstpmk()+g1.k()  && g1.size1() == g2.size1() && tstp==g2.tstp());


    double err = 0.0;
    for (int i = 0; i <= tstp; i++) {
        ZMatrixMap M1(g1.curr_timestep_les_ptr(i,tstp), size1, size2);
        ZMatrixMap M2(g2.lesptr(i), size1, size2);
        // std::cout << "Val: " << i << " " << M1 << " " << M2 << std::endl;
        err += (M1-M2).squaredNorm();
    }
    return std::sqrt(err);
}

double distance_norm2_curr_les(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix_timestep<double> &g2){
    assert(g1.size1() == g2.size1());
    assert(tstp >= g1.tstpmk() && tstp<=g1.tstpmk()+g1.k());
    assert(g1.nt() >= tstp);
    assert(g2.tstp()==tstp);

    if(g1.size1()==1 && g1.size2()==1){
        return distance_norm2_curr_les_dispatch<1,1>(tstp, g1, g2);    
    }else{
        return distance_norm2_curr_les_dispatch<LARGESIZE,LARGESIZE>(tstp, g1, g2);
    }
}


// Routines to check the diffence between compressed hodlr and herm_matrix

template <int SIZE1,int SIZE2>
double distance_norm2_ret_dispatch(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2) {
    int size1 = g1.size1();
    int size2 = g1.size2();
    assert(tstp>=-1 && g1.nt()>=tstp && g1.nt()>=tstp && g1.size1() == g2.size1());
    double err = 0.0;
    hodlr::ZMatrix M1(size1,size2);
    for (int i = 0; i <= tstp; i++) {
        g1.get_ret(tstp,i,M1);
        ZMatrixMap M2(g2.retptr(tstp,i), size1, size2);
        err += (M1-M2).squaredNorm();
    }
    return std::sqrt(err);
}


double distance_norm2_ret(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2){
    assert(g1.size1() == g2.size1());
    assert(g1.nt() >= tstp);
    assert(g2.nt() >= tstp);

    if(g1.size1()==1 && g1.size2()==1){
        return distance_norm2_ret_dispatch<1,1>(tstp, g1, g2);    
    }else{
        return distance_norm2_ret_dispatch<LARGESIZE,LARGESIZE>(tstp, g1, g2);
    }
}


template <int SIZE1,int SIZE2>
double distance_norm2_les_dispatch(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2) {
    int size1 = g1.size1();
    int size2 = g1.size2();
    assert(tstp>=-1 && g1.nt()>=tstp && g1.nt()>=tstp && g1.size1() == g2.size1());

    double err = 0.0;
    hodlr::ZMatrix M1(size1,size2);
    for (int i = 0; i <= tstp; i++) {
        g1.get_les(i,tstp,M1);
        ZMatrixMap M2(g2.lesptr(i,tstp), size1, size2);
        err += (M1-M2).squaredNorm();
    }
    return std::sqrt(err);
}


double distance_norm2_les(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2){
    assert(g1.size1() == g2.size1());
    assert(g1.nt() >= tstp);
    assert(g2.nt() >= tstp);

    if(g1.size1()==1 && g1.size2()==1){
        return distance_norm2_les_dispatch<1,1>(tstp, g1, g2);    
    }else{
        return distance_norm2_les_dispatch<LARGESIZE,LARGESIZE>(tstp, g1, g2);
    }
}


template <int SIZE1,int SIZE2>
double distance_norm2_tv_dispatch(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2,double *it2cf, int *it2cfp, double *dlrrf,int r) {
    assert(g1.size1() == g2.size1());
    assert(g1.nt() >= tstp);
    assert(g2.nt() >= tstp);
    assert(tstp>=0);

    int size1 = g1.size1(), size2 = g1.size2(), ntau = g2.ntau()+1, es = size1*size2, i;
    double err = 0.0;

    // Get equidistant tau grid
    double *eqpts_rel = new double[ntau];
    c_eqpts_rel(&ntau, eqpts_rel);

    double *Gij_it = new double[r*es];
    double *Gijc_it = new double[r*es];
    double *res = new double[es];
    
    // Get DLR coefficients
    DMatrixMap(Gij_it, es, r).noalias() = ZMatrixMap(g1.tvptr(tstp, 0), r, es).transpose().real();
    c_dlr_it2cf(&r, &size1, it2cf, it2cfp, Gij_it, Gijc_it);

    // Measure distance
    for(int tau = 0; tau < ntau; tau++) {
      c_dlr_it_eval(&r, &size1, dlrrf, Gijc_it, eqpts_rel+tau, res);
      err += (DColVectorMap(res, es) - ZColVectorMap(g2.tvptr(tstp,tau), es).real()).squaredNorm();
    }
    
    // Get DLR coefficients
    DMatrixMap(Gij_it, es, r).noalias() = ZMatrixMap(g1.tvptr(tstp, 0), r, es).transpose().imag();
    c_dlr_it2cf(&r, &size1, it2cf, it2cfp, Gij_it, Gijc_it);

    // Measure distance
    for(int tau = 0; tau < ntau; tau++) {
      c_dlr_it_eval(&r, &size1, dlrrf, Gijc_it, eqpts_rel+tau, res);
      err += (DColVectorMap(res, es) - ZColVectorMap(g2.tvptr(tstp,tau), es).imag()).squaredNorm();
    }

    delete [] eqpts_rel;
    delete [] Gij_it;
    delete [] Gijc_it;
    delete [] res;

    return std::sqrt(err);
}


double distance_norm2_tv(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2,double *it2cf, int *it2cfp, double *dlrrf){
    assert(g1.size1() == g2.size1());
    assert(g1.nt() >= tstp);
    assert(g2.nt() >= tstp);

    if(g1.size1()==1 && g1.size2()==1){
        return distance_norm2_tv_dispatch<1,1>(tstp, g1, g2,it2cf,it2cfp,dlrrf,g1.r());    
    }else{
        return distance_norm2_tv_dispatch<LARGESIZE,LARGESIZE>(tstp, g1, g2,it2cf,it2cfp,dlrrf,g1.r());
    }
}


template <int SIZE1,int SIZE2>
double distance_norm2_tv_dispatch(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix_timestep<double> &g2,double *it2cf, int *it2cfp, double *dlrrf,int r,bool dlr) {
    assert(g1.size1() == g2.size1());
    assert(g1.nt() >= tstp);
    assert(g2.tstp() == tstp);
    assert(tstp>=0);

    int size1 = g1.size1(), size2 = g1.size2(), ntau = g2.ntau()+1, es = size1*size2, i;
    double err = 0.0;

    if(dlr){
        for(int i = 0; i < r; i++) {
            ZMatrixMap M1(g1.tvptr(tstp,i), size1, size2);
            ZMatrixMap M2(g2.tvptr(i), size1, size2);
            err += (M1-M2).squaredNorm();
        }
    }
    else {
      // Get equidistant tau grid
      double *eqpts_rel = new double[ntau];
      c_eqpts_rel(&ntau, eqpts_rel);
  
      double *Gij_it = new double[r*es];
      double *Gijc_it = new double[r*es];
      double *res = new double[es];
      
      // Get DLR coefficients
      DMatrixMap(Gij_it, es, r).noalias() = ZMatrixMap(g1.tvptr(tstp, 0), r, es).transpose().real();
      c_dlr_it2cf(&r, &size1, it2cf, it2cfp, Gij_it, Gijc_it);
  
      // Measure distance
      for(int tau = 0; tau < ntau; tau++) {
        c_dlr_it_eval(&r, &size1, dlrrf, Gijc_it, eqpts_rel+tau, res);
        err += (DColVectorMap(res, es) - ZColVectorMap(g2.tvptr(tau), es).real()).squaredNorm();
      }
      
      // Get DLR coefficients
      DMatrixMap(Gij_it, es, r).noalias() = ZMatrixMap(g1.tvptr(tstp, 0), r, es).transpose().imag();
      c_dlr_it2cf(&r, &size1, it2cf, it2cfp, Gij_it, Gijc_it);
  
      // Measure distance
      for(int tau = 0; tau < ntau; tau++) {
        c_dlr_it_eval(&r, &size1, dlrrf, Gijc_it, eqpts_rel+tau, res);
        err += (DColVectorMap(res, es) - ZColVectorMap(g2.tvptr(tau), es).imag()).squaredNorm();
      }
  
      delete [] eqpts_rel;
      delete [] Gij_it;
      delete [] Gijc_it;
      delete [] res;
    }

    return std::sqrt(err);
}


double distance_norm2_tv(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix_timestep<double> &g2,double *it2cf, int *it2cfp, double *dlrrf,bool dlr){
    assert(g1.size1() == g2.size1());
    assert(g1.nt() >= tstp);
    assert(g2.tstp() == tstp);

    if(g1.size1()==1 && g1.size2()==1){
        return distance_norm2_tv_dispatch<1,1>(tstp, g1, g2,it2cf,it2cfp,dlrrf,g1.r(),dlr);    
    }else{
        return distance_norm2_tv_dispatch<LARGESIZE,LARGESIZE>(tstp, g1, g2,it2cf,it2cfp,dlrrf,g1.r(),dlr);
    }
}


template <int SIZE1,int SIZE2>
double distance_norm2_mat_dispatch(herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2,double *it2cf, int *it2cfp, double *dlrrf,int r){
    assert(g1.size1() == g2.size1());
    // r = g1.ntau()

    int size1 = g1.size1(), size2 = g1.size2(), ntau = g2.ntau()+1, es = size1*size2;
    double err = 0.0;
      
    // Get equidistant tau grid
    double *eqpts_rel = new double[ntau];
    c_eqpts_rel(&ntau, eqpts_rel);

    // Get DLR coefficients
    double *Gij_it = new double[r*es];
    double *Gijc_it = new double[r*es];
    double *res = new double[es];
    DMatrixMap(Gij_it, es, r).noalias() = DMatrixMap(g1.matptr(0), r, es).transpose();
    c_dlr_it2cf(&r, &size1, it2cf, it2cfp, Gij_it, Gijc_it);

    // Measure distance
    for(int tau = 0; tau < ntau; tau++) {
      c_dlr_it_eval(&r, &size1, dlrrf, Gijc_it, eqpts_rel+tau, res);
      err += (DColVectorMap(res, es) - ZColVectorMap(g2.matptr(tau), es)).squaredNorm();
    }
    
    delete [] eqpts_rel;
    delete [] Gij_it;
    delete [] Gijc_it;
    delete [] res;
    
    return std::sqrt(err);
}

double distance_norm2_mat(herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2,double *it2cf, int *it2cfp, double *dlrrf){
    assert(g1.size1() == g2.size1());

    if(g1.size1()==1 && g1.size2()==1){
        return distance_norm2_mat_dispatch<1,1>(g1, g2,it2cf,it2cfp,dlrrf,g1.r());
    }else{
        return distance_norm2_mat_dispatch<LARGESIZE,LARGESIZE>(g1, g2,it2cf,it2cfp,dlrrf,g1.r());
    }
}

template <int SIZE1,int SIZE2>
double distance_norm2_mat_dispatch(herm_matrix_hodlr &g1, cntr::herm_matrix_timestep<double> &g2,double *it2cf, int *it2cfp, double *dlrrf,int r,bool dlr) {
  assert(g1.size1() == g2.size1());
  assert(g2.tstp() == -1);

  // r = g1.ntau()
  int size1 = g1.size1(), size2 = g1.size2(), ntau = g2.ntau()+1, es = size1*size2;
  double err = 0.0;
  if(dlr){
    for(int i = 0; i < r; i++) {
      DMatrixMap M1(g1.matptr(i), size1, size2);
      ZMatrixMap M2(g2.matptr(i), size1, size2);
      err += (M1-M2).squaredNorm();
    }
  }
  else {
    // Get equidistant tau grid
    double *eqpts_rel = new double[ntau];
    c_eqpts_rel(&ntau, eqpts_rel);

    // Get DLR coefficients
    double *Gij_it = new double[r*es];
    double *Gijc_it = new double[r*es];
    double *res = new double[es];
    DMatrixMap(Gij_it, es, r).noalias() = DMatrixMap(g1.matptr(0), r, es).transpose();
    c_dlr_it2cf(&r, &size1, it2cf, it2cfp, Gij_it, Gijc_it);

    // Measure distance
    for(int tau = 0; tau < ntau; tau++) {
      c_dlr_it_eval(&r, &size1, dlrrf, Gijc_it, eqpts_rel+tau, res);
      err += (DColVectorMap(res, es) - ZColVectorMap(g2.matptr(tau), es)).squaredNorm();
    }
    
    delete [] eqpts_rel;
    delete [] Gij_it;
    delete [] Gijc_it;
    delete [] res;
  }

  return std::sqrt(err);
}

double distance_norm2_mat(herm_matrix_hodlr &g1, cntr::herm_matrix_timestep<double> &g2,double *it2cf, int *it2cfp, double *dlrrf,bool dlr){
    assert(g1.size1() == g2.size1());

    if(g1.size1()==1 && g1.size2()==1){
        return distance_norm2_mat_dispatch<1,1>(g1, g2,it2cf,it2cfp,dlrrf,g1.r(),dlr);    
    }else{
        return distance_norm2_mat_dispatch<LARGESIZE,LARGESIZE>(g1, g2,it2cf,it2cfp,dlrrf,g1.r(),dlr);
    }
}

double distance_norm2_curr(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix_timestep<double> &g2,double *it2cf, int *it2cfp, double *dlrrf,bool dlr){
    assert(g1.size1() == g2.size1());
    assert(tstp == g2.tstp() && tstp<=g1.tstpmk()+g1.k());
    assert(g1.nt() >= tstp);
    assert(dlr==0 || dlr==1);

    double err=0.0;
    if(tstp==-1){
        err=distance_norm2_mat(g1,g2,it2cf,it2cfp,dlrrf,dlr);        
    }else{
        err+=distance_norm2_curr_ret(tstp,g1,g2);
        err+=distance_norm2_curr_les(tstp,g1,g2);
        err+=distance_norm2_tv(tstp,g1,g2,it2cf,it2cfp,dlrrf,dlr);    
    }
    return err;
}

void herm_matrix_hodlr::update_blocks(Integration::Integrator &I) {
  std::ofstream out;
//  out.open("/pauli-storage/tblommel/hodlr_data/timings_blocks_1e-18.dat", std::ofstream::app);
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> dur;

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
//  out << tstpmk_ << " " << dur.count();

  // Loop over blocks and update svds
  start = std::chrono::system_clock::now();
  for(int b = 0; b < blocks_list.size(); b++) {
    int bindex = blocks_list[b];
    ZMatrix y(1,blkc2_[bindex]-blkc1_[bindex]+1);
    if(blklen_[bindex] == 0) { // no data yet, init the block
      built_blocks_ += 1;
      for(int i = 0; i < size1_; i++) {
        for(int j = 0; j < size2_; j++) {
          y = Eigen::Map<Eigen::MatrixXcd, 0, Eigen::InnerStride<> >(curr_timestep_ret_ptr() + blkc1(bindex)*size1_*size2_ + i*size1_ + j, 1, blkc2(bindex)-blkc1(bindex)+1, Eigen::InnerStride<>(size1_*size2_));
          ret_.data().blocks()[bindex][i][j] = hodlr::block(y, svdtol_);
          ret_.data().blocks()[bindex][i][j].resize(blkr2(bindex)-blkr1(bindex)+1, blkc2(bindex)-blkc1(bindex)+1, ret_.data().blocks()[bindex][i][j].epsrank());
        }
      }
    }
    else { // update the tsvd
      for(int i = 0; i < size1_; i++) {
        for(int j = 0; j < size2_; j++) {
          y = Eigen::Map<Eigen::MatrixXcd, 0, Eigen::InnerStride<> >(curr_timestep_ret_ptr() + blkc1(bindex)*size1_*size2_ + i*size1_ + j, 1, blkc2(bindex)-blkc1(bindex)+1, Eigen::InnerStride<>(size1_*size2_));
          updatetsvd(ret_.data().blocks()[bindex][i][j],y,blklen_[bindex]);
        }
      }
    }
    if(blklen_[bindex] == 0) { // no data yet, init the block
      for(int i = 0; i < size1_; i++) {
        for(int j = 0; j < size2_; j++) {
          y = Eigen::Map<Eigen::MatrixXcd, 0, Eigen::InnerStride<> >(curr_timestep_les_ptr() + blkc1(bindex)*size1_*size2_ + i*size1_ + j, 1, blkc2(bindex)-blkc1(bindex)+1, Eigen::InnerStride<>(size1_*size2_));
          les_.data().blocks()[bindex][i][j] = hodlr::block(y, svdtol_);
          les_.data().blocks()[bindex][i][j].resize(blkr2(bindex)-blkr1(bindex)+1, blkc2(bindex)-blkc1(bindex)+1, ret_.data().blocks()[bindex][i][j].epsrank());
        }
      }
    }
    else { // update the tsvd
      for(int i = 0; i < size1_; i++) {
        for(int j = 0; j < size2_; j++) {
          y = Eigen::Map<Eigen::MatrixXcd, 0, Eigen::InnerStride<> >(curr_timestep_les_ptr() + blkc1(bindex)*size1_*size2_ + i*size1_ + j, 1, blkc2(bindex)-blkc1(bindex)+1, Eigen::InnerStride<>(size1_*size2_));
          updatetsvd(les_.data().blocks()[bindex][i][j],y,blklen_[bindex]);
        }
      }
    }
    // block now has one more row
    blklen_[bindex] = blklen_[bindex]+1;
  }
  end = std::chrono::system_clock::now();
  dur = end-start;
//  out << " " << dur.count();

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
//  out << " " << dur.count() << std::endl;
//  out.close();
  tstpmk_++;
}

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
  return nrm;
}

void herm_matrix_hodlr::updatetsvd(hodlr::block &B,const ZRowVector &row,int row_cur){
  int cols=B.cols();
  int rank=B.epsrank();
  double nrm;
  ZRowVector Vrow(rank);
  ZMatrix uu(rank+1,rank+1); //Temporal U matrix
  DColVector su(rank+1); // Updated singular values
  ZMatrix uout(row_cur,rank+1),vout(cols,rank+1); //Final update
  ZRowVector q(cols);

  nrm=mgs(B.V(),row.adjoint(),q,Vrow);
  // Construct middle matrix [uu] in Eq. 15 of Low rank compression in the numerical solution of the nonequilibrium Dyson equation
  uu.setZero();
  for(int i=0;i<rank;i++){
    uu(i,i)=B.S()(i);
  }
  uu.block(rank,0,1,rank)=Vrow.conjugate();
  uu(rank,rank)=nrm;
  // Compute SVD [TODO: can be done faster as in Ref. 65]
  ZMatrix unew,vnew;
  hodlr::svd(uu,unew,vnew,su);
  // Eigen::BDCSVD<Eigen::MatrixXcd> svd(uu,Eigen::ComputeThinU | Eigen::ComputeThinV);
  // ZMatrix unew=svd.matrixU(); 
  // ZMatrix vnew=svd.matrixV();
  // su= svd.singularValues().array();
  // Update block
  if(su(rank)<svdtol_){ //if rank has not increased
    ZMatrix ufin(row_cur+1,rank+1);
    // Set up U
    uout=B.U().block(0,0,row_cur,rank)*unew.block(0,0,rank,rank+1);
    ufin.block(0,0,row_cur,rank+1)=uout.block(0,0,row_cur,rank+1);
    ufin.block(row_cur,0,1,rank+1)=unew.block(rank,0,1,rank+1);
    B.set_U(ufin.block(0,0,row_cur+1,rank));
    // Set up V
    vout.block(0,0,cols,rank)=B.V().block(0,0,cols,rank);
    vout.block(0,rank,cols,1)=q.transpose();
    vout=vout*vnew.block(0,0,rank+1,rank+1);
    B.set_V(vout.block(0,0,cols,rank));
    B.set_S(su.head(rank));
  }else{ //if rank has increased
    ZMatrix ufin(row_cur+1,rank+1);
    B.resize_rank(rank+1);
    // Set up U+
    uout=B.U().block(0,0,row_cur,rank)*unew.block(0,0,rank,rank+1);
    ufin.block(0,0,row_cur,rank+1)=uout.block(0,0,row_cur,rank+1);
    ufin.block(row_cur,0,1,rank+1)=unew.block(rank,0,1,rank+1);
    B.set_U(ufin);
    // Set up V+
    vout.block(0,0,cols,rank)=B.V().block(0,0,cols,rank);
    vout.block(0,rank,cols,1)=q.transpose();
    vout=vout*vnew.block(0,0,rank+1,rank+1);
    B.set_V(vout);
    B.set_S(su);
  }
}

// //  BUBBLE 1 :
// //  C(t,t') = ii * A(t,t') * B(t',t)
void Bubble1_curr(int tstp, herm_matrix_hodlr &C, int c1, int c2, herm_matrix_hodlr &A, herm_matrix_hodlr &Acc, int a1, int a2, herm_matrix_hodlr &B,
             herm_matrix_hodlr &Bcc, int b1, int b2,double *it2itr){

    assert(C.r()==A.r() && C.r()==B.r());
    assert(C.size1()==A.size1() && C.size1()==B.size1());
    assert(C.size2()==A.size2() && C.size2()==B.size2());
    assert(C.ntau()==A.ntau() && C.ntau()==B.ntau());
    assert(a1 <= A.size1());
    assert(a2 <= A.size2());
    assert(b1 <= B.size1());
    assert(b2 <= B.size2());
    assert(c1 <= C.size1());
    assert(c2 <= C.size2());

    int size1 = A.size1(), size2 = A.size2(), ntau = A.ntau(), r = A.r();
    DMatrixMap it2(it2itr, r, r);

    if(tstp==-1){
        // Matsubara reversed G(\beta-tau_i)
        double *Bm_reverse;
        Bm_reverse = new double[r * size1 * size2];
        DMatrixMap(Bm_reverse,r,size1 * size2 ) = it2.transpose() * DMatrixMap(B.matptr(0), r, size1*size2);

        for(int i=0;i<r;i++){
            get_bubble_1_mat(C.matptr(i),C.size1(),C.size2(),c1,c2,
                      A.matptr(i), A.size1(),A.size2(),a1,a2,
                      B.matptr(i), B.size1(), B.size2(),b1,b2,B.sig(),Bm_reverse + i * size1 * size2);
        }
        delete Bm_reverse;
    }else{
        cplx *Btv_reverse;
        Btv_reverse = new cplx[r * size1 * size2];
        ZMatrixMap(Btv_reverse, r, size1 * size2) = (-1.0) * B.sig() * it2.transpose() * ZMatrixMap(B.tvptr(tstp,0), r, size1*size2).conjugate();

        for (int m = 0; m <= tstp; m++) {
            get_bubble_1_timestep(tstp, C.curr_timestep_ret_ptr(tstp,m),C.tvptr(tstp,0),C.curr_timestep_les_ptr(m,tstp),C.size1(),C.size2(),c1,c2,
                  A.curr_timestep_ret_ptr(tstp,m),A.tvptr(tstp,0),A.curr_timestep_les_ptr(m,tstp), Acc.curr_timestep_ret_ptr(tstp,m),Acc.tvptr(tstp,0),Acc.curr_timestep_les_ptr(m,tstp),A.size1(),A.size2(),a1,a2,
                  B.curr_timestep_ret_ptr(tstp,m),B.tvptr(tstp,0),B.curr_timestep_les_ptr(m,tstp), Bcc.curr_timestep_ret_ptr(tstp,m),Bcc.tvptr(tstp,0),Bcc.curr_timestep_les_ptr(m,tstp),B.size1(),B.size2(),b1,b2,A.r(),Btv_reverse);
            
        }
        // Set transposed tv
        for(int i=0;i<r;i++){
            ZMatrixMap(C.tvptr_trans(tstp,i), size2, size1) = ZMatrixMap(C.tvptr(tstp,i), size1, size2).transpose();
        }
        delete Btv_reverse;
    }
}

//  BUBBLE 1 :
//  C(t,t') = ii * A(t,t') * B(t',t)
void Bubble1_curr(int tstp, cntr::herm_matrix_timestep<double> &C, int c1, int c2, herm_matrix_hodlr &A, herm_matrix_hodlr &Acc, int a1, int a2, herm_matrix_hodlr &B,
             herm_matrix_hodlr &Bcc, int b1, int b2,double *it2itr){

    assert(C.ntau()==A.r() && C.ntau()==B.r());
    assert(tstp==C.tstp());
    assert(C.size1()==A.size1() && C.size1()==B.size1());
    assert(C.size2()==A.size2() && C.size2()==B.size2());
    assert(C.ntau()==A.ntau() && C.ntau()==B.ntau());
    assert(a1 <= A.size1());
    assert(a2 <= A.size2());
    assert(b1 <= B.size1());
    assert(b2 <= B.size2());
    assert(c1 <= C.size1());
    assert(c2 <= C.size2());

    int size1 = A.size1(), size2 = A.size2(), ntau = A.ntau(), r = A.r();
    DMatrixMap it2(it2itr, r, r);

    if(tstp==-1){
        // Matsubara reversed G(\beta-tau_i)
        double *Bm_reverse;
        Bm_reverse = new double[r * size1 * size2];
        DMatrixMap(Bm_reverse,r,size1 * size2 ) = it2.transpose() * DMatrixMap(B.matptr(0), r, size1*size2);

        for(int i=0;i<r;i++){
            get_bubble_1_mat(C.matptr(i),C.size1(),C.size2(),c1,c2,
                      A.matptr(i), A.size1(),A.size2(),a1,a2,
                      B.matptr(i), B.size1(), B.size2(),b1,b2,B.sig(),Bm_reverse + i * size1 * size2);
        }
        delete Bm_reverse;
    }else{
        cplx *Btv_reverse;
        Btv_reverse = new cplx[r * size1 * size2];
        ZMatrixMap(Btv_reverse, r, size1 * size2) = (-1.0) * B.sig() * it2.transpose() * ZMatrixMap(B.tvptr(tstp,0), r, size1*size2).conjugate();

        for (int m = 0; m <= tstp; m++) {
            get_bubble_1_timestep(tstp, C.retptr(m),C.tvptr(0),C.lesptr(m),C.size1(),C.size2(),c1,c2,
                  A.curr_timestep_ret_ptr(tstp,m),A.tvptr(tstp,0),A.curr_timestep_les_ptr(m,tstp), Acc.curr_timestep_ret_ptr(tstp,m),Acc.tvptr(tstp,0),Acc.curr_timestep_les_ptr(m,tstp),A.size1(),A.size2(),a1,a2,
                  B.curr_timestep_ret_ptr(tstp,m),B.tvptr(tstp,0),B.curr_timestep_les_ptr(m,tstp), Bcc.curr_timestep_ret_ptr(tstp,m),Bcc.tvptr(tstp,0),Bcc.curr_timestep_les_ptr(m,tstp),B.size1(),B.size2(),b1,b2,A.r(),Btv_reverse);
            // TODO: Set transposed tv once we have herm_matrix_timestep_hodlr
            // ZMatrixMap(C.tvptr_trans(tstp,m), size2, size1) = ZMatrixMap(C.tvptr(tstp,m), size1, size2).transpose();
        }
        delete Btv_reverse;
    }
}


// //  BUBBLE 1 :
// //  C(t,t') = ii * A(t,t') * B(t',t)
void Bubble1_curr(int tstp, herm_matrix_hodlr &C, int c1, int c2, cntr::herm_matrix_timestep<double> &A, cntr::herm_matrix_timestep<double> &Acc, int a1, int a2, herm_matrix_hodlr &B,
             herm_matrix_hodlr &Bcc, int b1, int b2,double *it2itr){

    assert(C.r()==A.ntau() && C.r()==B.r());
    assert(tstp==A.tstp());
    assert(C.size1()==A.size1() && C.size1()==B.size1());
    assert(C.size2()==A.size2() && C.size2()==B.size2());
    assert(C.ntau()==A.ntau() && C.ntau()==B.ntau());
    assert(a1 <= A.size1());
    assert(a2 <= A.size2());
    assert(b1 <= B.size1());
    assert(b2 <= B.size2());
    assert(c1 <= C.size1());
    assert(c2 <= C.size2());

    int size1 = A.size1(), size2 = A.size2(), ntau = B.ntau(), r = B.r();
    DMatrixMap it2(it2itr, r, r);

    if(tstp==-1){
        // Matsubara reversed G(\beta-tau_i)
        double *Bm_reverse;
        Bm_reverse = new double[r * size1 * size2];
        DMatrixMap(Bm_reverse,r,size1 * size2 ) = it2.transpose() * DMatrixMap(B.matptr(0), r, size1*size2);

        for(int i=0;i<r;i++){
            get_bubble_1_mat(C.matptr(i),C.size1(),C.size2(),c1,c2,
                      A.matptr(i), A.size1(),A.size2(),a1,a2,
                      B.matptr(i), B.size1(), B.size2(),b1,b2,B.sig(),Bm_reverse + i * size1 * size2);
        }
        delete Bm_reverse;
    }else{
        cplx *Btv_reverse;
        Btv_reverse = new cplx[r * size1 * size2];
        ZMatrixMap(Btv_reverse, r, size1 * size2) = (-1.0) * B.sig() * it2.transpose() * ZMatrixMap(B.tvptr(tstp,0), r, size1*size2).conjugate();

        for (int m = 0; m <= tstp; m++) {
            get_bubble_1_timestep(tstp, C.curr_timestep_ret_ptr(tstp,m),C.tvptr(tstp,0),C.curr_timestep_les_ptr(m,tstp),C.size1(),C.size2(),c1,c2,
                  A.retptr(m),A.tvptr(0),A.lesptr(m), Acc.retptr(m),Acc.tvptr(0),Acc.lesptr(m),A.size1(),A.size2(),a1,a2,
                  B.curr_timestep_ret_ptr(tstp,m),B.tvptr(tstp,0),B.curr_timestep_les_ptr(m,tstp), Bcc.curr_timestep_ret_ptr(tstp,m),Bcc.tvptr(tstp,0),Bcc.curr_timestep_les_ptr(m,tstp),B.size1(),B.size2(),b1,b2,B.r(),Btv_reverse);
        }
        // Set transposed tv
        for(int i=0;i<r;i++){
            ZMatrixMap(C.tvptr_trans(tstp,i), size2, size1) = ZMatrixMap(C.tvptr(tstp,i), size1, size2).transpose();
        }
        delete Btv_reverse;
    }
}

// Matsubara:
// C_{c1,c2}(tau) = - A_{a1,a2}(tau) * B_{b2,b1}(-tau)
//                = - A_{a1,a2}(tau) * B_{b2,b1}(beta-tau) (no cc needed !!)
void get_bubble_1_mat(double *cmat,int sizec1,int sizec2,int c1, int c2,
                      double *amat, int sizea1,int sizea2, int a1, int a2,
                      double *bmat, int sizeb1,int sizeb2, int b1, int b2,int signb,double *Bm_reverse){

    // C_{c1,c2}(tau) = - sig * A_{a1,a2}(tau) * B_{b2,b1}(beta-tau)
    DMatrixMap AM(amat , sizea1, sizea2);
    DMatrixMap BMR(Bm_reverse , sizeb1, sizeb2);
    DMatrixMap CM(cmat , sizec1, sizec2);
    CM(c1,c2)= (-1.0) * signb * AM(a1,a2)*BMR(b1,b2); // TODO I'm not certain about if the last one should be transpose/ check 
}

// Matsubara:
// C_{c1,c2}(tau) = - A_{a1,a2}(tau) * B_{b2,b1}(-tau)
//                = - A_{a1,a2}(tau) * B_{b2,b1}(beta-tau) (no cc needed !!)
void get_bubble_1_mat(std::complex<double> *cmat,int sizec1,int sizec2,int c1, int c2,
                      double *amat, int sizea1,int sizea2, int a1, int a2,
                      double *bmat, int sizeb1,int sizeb2, int b1, int b2,int signb,double *Bm_reverse){

    // C_{c1,c2}(tau) = - sig * A_{a1,a2}(tau) * B_{b2,b1}(beta-tau)
    DMatrixMap AM(amat , sizea1, sizea2);
    DMatrixMap BMR(Bm_reverse , sizeb1, sizeb2);
    ZMatrixMap CM(cmat , sizec1, sizec2);
    CM(c1,c2)= (-1.0) * signb * AM(a1,a2)*BMR(b1,b2); // TODO I'm not certain about if the last one should be transpose/ check 

}

// Matsubara:
// C_{c1,c2}(tau) = - A_{a1,a2}(tau) * B_{b2,b1}(-tau)
//                = - A_{a1,a2}(tau) * B_{b2,b1}(beta-tau) (no cc needed !!)
void get_bubble_1_mat(double *cmat,int sizec1,int sizec2,int c1, int c2,
                      std::complex<double> *amat, int sizea1,int sizea2, int a1, int a2,
                      double *bmat, int sizeb1,int sizeb2, int b1, int b2,int signb,double *Bm_reverse){

    // C_{c1,c2}(tau) = - sig * A_{a1,a2}(tau) * B_{b2,b1}(beta-tau)
    ZMatrixMap AM(amat, sizea1, sizea2);
    DMatrixMap BMR(Bm_reverse, sizeb1, sizeb2);
    DMatrixMap CM(cmat, sizec1, sizec2);
    CM(c1,c2)= (-1.0) * signb * std::real(AM(a1,a2))*BMR(b1,b2); // TODO I'm not certain about if the last one should be transpose/ check 

}

void get_bubble_1_timestep(int tstp, std::complex<double> *cret, std::complex<double> *ctv, std::complex<double> *cles,int sizec1,int sizec2,int c1, int c2,
                  std::complex<double> *aret, std::complex<double> *atv, std::complex<double> *ales, std::complex<double> *accret, std::complex<double> *acctv, std::complex<double> *accles,int sizea1,int sizea2, int a1, int a2,
                std::complex<double> *bret, std::complex<double> *btv, std::complex<double> *bles, std::complex<double> *bccret, std::complex<double> *bcctv, std::complex<double> *bccles,int sizeb1,int sizeb2, int b1, int b2,int r,std::complex<double> *Btv_reverse){

    assert(a1 <= sizea1);
    assert(a2 <= sizea2);
    assert(b1 <= sizeb1);
    assert(b2 <= sizeb2);
    assert(c1 <= sizec1);
    assert(c2 <= sizec2);
    assert(sizea1==sizeb1 && sizea1==sizec1);
    assert(sizea2==sizeb2 && sizea2==sizec2);


    // temp variables
    cplx *a_arr,*b_arr,*c_arr,*c_arr1;
    a_arr = new cplx[sizea1 * sizea2];
    b_arr = new cplx[sizeb1 * sizeb2];
    c_arr = new cplx[sizec1 * sizec2];
    c_arr1 = new cplx[sizec1 * sizec2];

    // std::cout << "Lesser " << tstp << " " << A.tstpmk() << " " << B.tstpmk() << " " << Bcc.tstpmk() << " " << C.tstpmk() << std::endl;
    // Lesser C^<_{c1,c2}(m,tstp) = ii * A^<_{a1,a2}(m,tstp)*B^>_{b2,b1}(tstp,m)
    ZMatrixMap ALes(ales , sizea1, sizea2);
    ZMatrixMap BLes(bles , sizeb1, sizeb2);
    ZMatrixMap BLesCC(bccles , sizeb1, sizeb2);
    ZMatrixMap ALesCC(accles , sizea1, sizea2);
    ZMatrixMap ARet(aret , sizea1, sizea2);
    ZMatrixMap BRet(bret , sizeb1, sizeb2);
    ZMatrixMap BRetCC(bccret , sizeb1, sizeb2);
    ZMatrixMap CLes(cles, sizec1, sizec2);

    // std::cout << "LesserB " << tstp << std::endl;
    // B^>_{b2,b1}(tstp,m) = B^R_{b2,b1}(tstp,m) - [B^{<,cc}(m,tstp)]^{\dagger};
    ZMatrixMap BGtr(b_arr, sizeb1, sizeb2);
    BGtr=BRet-BLesCC.adjoint();
    CLes(c1,c2)=std::complex<double>(0.0,1.0) * ALes(a1,a2)*BGtr(b2,b1);

    // Retarded is evaluated as C^R(tstp,m)=C^>(tstp,m)-C^<(tstp,m)

    // C^>_{c1,c2}(tstp,m) = ii * A^>_{a1,a2}(tstp,m)*[B^{cc,<}_{b1,b2}(m,tstp)]^{\dagger}
    // A^>_{a1,a2}(tstp,m) = A^R_{a1,a2}(tstp,m) - [A^{cc,<}_{a1,a2}(m,tstp)]^{\dagger}
    ZMatrixMap AGtr(a_arr, sizea1, sizea2);
    AGtr=ARet-ALesCC.adjoint();
    ZMatrixMap CGtr(c_arr, sizec1, sizec2);
    CGtr(c1,c2)=std::complex<double>(0.0,1.0) * AGtr(a1,a2) * BLes(b2,b1);
            
            
    // C^<_{12}(tstp,m) = ii * A^<_{12}(tstp,m)*B^>_{21}(m,tstp)
    ZMatrixMap CLes_tt1(c_arr1, sizec1, sizec2);
    CLes_tt1(c1,c2)=std::complex<double>(0.0,-1.0) * std::conj(ALesCC(a1,a2)) * (-std::conj(BRetCC(b1,b2)) + BLes(b2,b1));
    // std::cout << "CGTR " << tstp << " " << m << " " << CGtr(c1,c2) << std::endl;
    // std::cout << "CLES_tt1 " << tstp << " " << m << " " << CLes_tt1(c1,c2) << std::endl;
    ZMatrixMap CRet(cret, sizec1, sizec2);
    CRet(c1,c2)=CGtr(c1,c2)-CLes_tt1(c1,c2);
    // std::cout << "CRer " << tstp << " " << m << " " << CRet(c1,c2) << std::endl;
    // std::cout << "RetB " << tstp << std::endl;
    
    // std::cout << "mixed " << std::endl;
    for (int m = 0; m < r; m++) {
        ZMatrixMap ATV(atv + m * sizea1 *sizea2 , sizea1, sizea2);
        ZMatrixMap BTV(Btv_reverse +  m * sizeb1 * sizeb2, sizeb1, sizeb2);
        ZMatrixMap CTV(ctv + m * sizec1 *sizec2 , sizec1, sizec2);
        CTV(c1,c2)=std::complex<double>(0.0,1.0) * ATV(a1,a2) * BTV(b1,b2); // TODO I'm not certain about if the last one should be transpose/ check 
    }
    delete a_arr;
    delete b_arr;
    delete c_arr;
    delete c_arr1;
}


//  BUBBLE 2 :
//  C(t,t') = ii * A(t,t') * B(t,t')
void Bubble2_curr(int tstp, herm_matrix_hodlr &C, int c1, int c2, herm_matrix_hodlr &A, herm_matrix_hodlr &Acc, int a1, int a2, herm_matrix_hodlr &B,
             herm_matrix_hodlr &Bcc, int b1, int b2){

    assert(C.r()==A.r() && C.r()==B.r());
    assert(C.size1()==A.size1() && C.size1()==B.size1());
    assert(C.size2()==A.size2() && C.size2()==B.size2());
    assert(C.ntau()==A.ntau() && C.ntau()==B.ntau());
    assert(a1 <= A.size1());
    assert(a2 <= A.size2());
    assert(b1 <= B.size1());
    assert(b2 <= B.size2());
    assert(c1 <= C.size1());
    assert(c2 <= C.size2());

    int size1 = A.size1(), size2 = A.size2(), ntau = B.ntau(), r = B.r();
    if(tstp==-1){
        for(int i=0;i<r;i++){
            get_bubble_2_mat(C.matptr(i),C.size1(),C.size2(),c1,c2,
                      A.matptr(i), A.size1(),A.size2(),a1,a2,
                      B.matptr(i), B.size1(), B.size2(),b1,b2);
        }
    }else{
        for (int m = 0; m <= tstp; m++) {
            get_bubble_2_timestep(tstp, C.curr_timestep_ret_ptr(tstp,m), C.tvptr(tstp,0),C.curr_timestep_les_ptr(m,tstp),C.size1(),C.size2(),c1,c2,
                  A.curr_timestep_ret_ptr(tstp,m), A.tvptr(tstp,0),A.curr_timestep_les_ptr(m,tstp), Acc.curr_timestep_ret_ptr(tstp,m), Acc.tvptr(tstp,0),Acc.curr_timestep_les_ptr(m,tstp),A.size1(),A.size2(),a1,a2,
                  B.curr_timestep_ret_ptr(tstp,m), B.tvptr(tstp,0),B.curr_timestep_les_ptr(m,tstp), Bcc.curr_timestep_ret_ptr(tstp,m), Bcc.tvptr(tstp,0),Bcc.curr_timestep_les_ptr(m,tstp),B.size1(),B.size2(),b1,b2,B.r());
        }
        // Set transposed tv
        for(int i=0;i<r;i++){
            ZMatrixMap(C.tvptr_trans(tstp,i), size2, size1) = ZMatrixMap(C.tvptr(tstp,i), size1, size2).transpose();
        }
    }

}

void Bubble2_curr(int tstp, cntr::herm_matrix_timestep<double> &C, int c1, int c2, herm_matrix_hodlr &A, herm_matrix_hodlr &Acc, int a1, int a2, herm_matrix_hodlr &B,
             herm_matrix_hodlr &Bcc, int b1, int b2){

    assert(C.ntau()==A.r() && C.ntau()==B.r());
    assert(C.size1()==A.size1() && C.size1()==B.size1());
    assert(C.size2()==A.size2() && C.size2()==B.size2());
    assert(a1 <= A.size1());
    assert(a2 <= A.size2());
    assert(b1 <= B.size1());
    assert(b2 <= B.size2());
    assert(c1 <= C.size1());
    assert(c2 <= C.size2());

    int size1 = A.size1(), size2 = A.size2(), ntau = B.ntau(), r = B.r();

    if(tstp==-1){
        for(int i=0;i<r;i++){
            get_bubble_2_mat(C.matptr(i),C.size1(),C.size2(),c1,c2,
                      A.matptr(i), A.size1(),A.size2(),a1,a2,
                      B.matptr(i), B.size1(), B.size2(),b1,b2);
        }
    }else{
        for (int m = 0; m <= tstp; m++) {
            get_bubble_2_timestep(tstp, C.retptr(m), C.tvptr(0),C.lesptr(m),C.size1(),C.size2(),c1,c2,
                  A.curr_timestep_ret_ptr(tstp,m), A.tvptr(tstp,0),A.curr_timestep_les_ptr(m,tstp), Acc.curr_timestep_ret_ptr(tstp,m), Acc.tvptr(tstp,0),Acc.curr_timestep_les_ptr(m,tstp),A.size1(),A.size2(),a1,a2,
                  B.curr_timestep_ret_ptr(tstp,m), B.tvptr(tstp,0),B.curr_timestep_les_ptr(m,tstp), Bcc.curr_timestep_ret_ptr(tstp,m), Bcc.tvptr(tstp,0),Bcc.curr_timestep_les_ptr(m,tstp),B.size1(),B.size2(),b1,b2,B.r());
            // Set transposed tv
            // ZMatrixMap(C.tvptr_trans(tstp,m), size2, size1) = ZMatrixMap(C.tvptr(tstp,m), size1, size2).transpose();
        }
    }
}

void Bubble2_curr(int tstp, herm_matrix_hodlr &C, int c1, int c2, cntr::herm_matrix_timestep<double> &A, cntr::herm_matrix_timestep<double> &Acc, int a1, int a2, herm_matrix_hodlr &B,
             herm_matrix_hodlr &Bcc, int b1, int b2){

    assert(C.r()==A.ntau() && C.r()==B.r());
    assert(C.size1()==A.size1() && C.size1()==B.size1());
    assert(C.size2()==A.size2() && C.size2()==B.size2());
    assert(a1 <= A.size1());
    assert(a2 <= A.size2());
    assert(b1 <= B.size1());
    assert(b2 <= B.size2());
    assert(c1 <= C.size1());
    assert(c2 <= C.size2());

    int size1 = B.size1(), size2 = B.size2(), ntau = B.ntau(), r = B.r();

    if(tstp==-1){
        for(int i=0;i<r;i++){
            get_bubble_2_mat(C.matptr(i),C.size1(),C.size2(),c1,c2,
                      A.matptr(i), A.size1(),A.size2(),a1,a2,
                      B.matptr(i), B.size1(), B.size2(),b1,b2);
        }
    }else{
        for (int m = 0; m <= tstp; m++) {
            get_bubble_2_timestep(tstp, C.curr_timestep_ret_ptr(tstp,m), C.tvptr(tstp,0),C.curr_timestep_les_ptr(m,tstp),C.size1(),C.size2(),c1,c2,
                  A.retptr(m), A.tvptr(0),A.lesptr(m), Acc.retptr(m), Acc.tvptr(0),Acc.lesptr(m),A.size1(),A.size2(),a1,a2,
                  B.curr_timestep_ret_ptr(tstp,m), B.tvptr(tstp,0),B.curr_timestep_les_ptr(m,tstp), Bcc.curr_timestep_ret_ptr(tstp,m), Bcc.tvptr(tstp,0),Bcc.curr_timestep_les_ptr(m,tstp),B.size1(),B.size2(),b1,b2,B.r());
        }
        for(int i=0;i<r;i++){
            ZMatrixMap(C.tvptr_trans(tstp,i), size2, size1) = ZMatrixMap(C.tvptr(tstp,i), size1, size2).transpose();
        }
    }
}


void Bubble2_curr(int tstp, herm_matrix_hodlr &C, int c1, int c2, herm_matrix_hodlr &A, herm_matrix_hodlr &Acc, int a1, int a2, cntr::herm_matrix_timestep<double> &B,
             cntr::herm_matrix_timestep<double> &Bcc, int b1, int b2){

    assert(C.r()==A.r() && C.r()==B.ntau());
    assert(C.size1()==A.size1() && C.size1()==B.size1());
    assert(C.size2()==A.size2() && C.size2()==B.size2());
    assert(a1 <= A.size1());
    assert(a2 <= A.size2());
    assert(b1 <= B.size1());
    assert(b2 <= B.size2());
    assert(c1 <= C.size1());
    assert(c2 <= C.size2());

    int size1 = B.size1(), size2 = B.size2(), ntau = A.ntau(), r = A.r();

    if(tstp==-1){
        for(int i=0;i<r;i++){
            get_bubble_2_mat(C.matptr(i),C.size1(),C.size2(),c1,c2,
                      A.matptr(i), A.size1(),A.size2(),a1,a2,
                      B.matptr(i), B.size1(), B.size2(),b1,b2);
        }
    }else{
        for (int m = 0; m <= tstp; m++) {
            get_bubble_2_timestep(tstp, C.curr_timestep_ret_ptr(tstp,m), C.tvptr(tstp,0),C.curr_timestep_les_ptr(m,tstp),C.size1(),C.size2(),c1,c2,
                  A.curr_timestep_ret_ptr(tstp,m), A.tvptr(tstp,0),A.curr_timestep_les_ptr(m,tstp), Acc.curr_timestep_ret_ptr(tstp,m), Acc.tvptr(tstp,0),Acc.curr_timestep_les_ptr(m,tstp),A.size1(),A.size2(),a1,a2,
                  B.retptr(m), B.tvptr(0),B.lesptr(m), Bcc.retptr(m), Bcc.tvptr(0),Bcc.lesptr(m),B.size1(),B.size2(),b1,b2,r);
        }
        for(int i=0;i<r;i++){
            ZMatrixMap(C.tvptr_trans(tstp,i), size2, size1) = ZMatrixMap(C.tvptr(tstp,i), size1, size2).transpose();
        }
    }
}

// Matsubara:
// C_{c1,c2}(tau) = - A_{a1,a2}(tau) * B_{b1,b2}(tau)
void get_bubble_2_mat(double *cmat,int sizec1,int sizec2,int c1, int c2,
                      double *amat, int sizea1,int sizea2, int a1, int a2,
                      double *bmat, int sizeb1,int sizeb2, int b1, int b2){
        DMatrixMap AM(amat, sizea1, sizea2);
        DMatrixMap BM(bmat, sizeb1, sizeb2);
        DMatrixMap CM(cmat, sizec1, sizec2);
        CM(c1,c2)=(-1.0) * AM(a1,a2)*BM(b1,b2);
}

// TODO: When libdlr has support for complex remove 
// Matsubara:
// C_{c1,c2}(tau) = - A_{a1,a2}(tau) * B_{b1,b2}(tau)
void get_bubble_2_mat(std::complex<double> *cmat,int sizec1,int sizec2,int c1, int c2,
                      double *amat, int sizea1,int sizea2, int a1, int a2,
                      double *bmat, int sizeb1,int sizeb2, int b1, int b2){


        DMatrixMap AM(amat, sizea1, sizea2);
        DMatrixMap BM(bmat, sizeb1, sizeb2);
        ZMatrixMap CM(cmat, sizec1, sizec2);
        CM(c1,c2)=(-1.0) * AM(a1,a2)*BM(b1,b2);
    
}



void get_bubble_2_mat(double *cmat,int sizec1,int sizec2,int c1, int c2,
                      std::complex<double> *amat, int sizea1,int sizea2, int a1, int a2,
                      double *bmat, int sizeb1,int sizeb2, int b1, int b2){


        ZMatrixMap AM(amat, sizea1, sizea2);
        DMatrixMap BM(bmat, sizeb1, sizeb2);
        DMatrixMap CM(cmat, sizec1, sizec2);
        CM(c1,c2)=(-1.0) * std::real(AM(a1,a2))*BM(b1,b2);
    
}

void get_bubble_2_mat(double *cmat,int sizec1,int sizec2,int c1, int c2,
                      double *amat, int sizea1,int sizea2, int a1, int a2,
                      std::complex<double> *bmat, int sizeb1,int sizeb2, int b1, int b2){


        DMatrixMap AM(amat, sizea1, sizea2);
        ZMatrixMap BM(bmat, sizeb1, sizeb2);
        DMatrixMap CM(cmat, sizec1, sizec2);
        CM(c1,c2)=(-1.0) * AM(a1,a2)*std::real(BM(b1,b2));
    
}

void get_bubble_2_timestep(int tstp, std::complex<double> *cret, std::complex<double> *ctv, std::complex<double> *cles,int sizec1,int sizec2,int c1, int c2,
                  std::complex<double> *aret, std::complex<double> *atv, std::complex<double> *ales, std::complex<double> *accret, std::complex<double> *acctv, std::complex<double> *accles,int sizea1,int sizea2, int a1, int a2,
                std::complex<double> *bret, std::complex<double> *btv, std::complex<double> *bles, std::complex<double> *bccret, std::complex<double> *bcctv, std::complex<double> *bccles,int sizeb1,int sizeb2, int b1, int b2, int r){

    assert(a1 <= sizea1);
    assert(a2 <= sizea2);
    assert(b1 <= sizeb1);
    assert(b2 <= sizeb2);
    assert(c1 <= sizec1);
    assert(c2 <= sizec2);
    assert(sizea1==sizeb1 && sizea1==sizec1);
    assert(sizea2==sizeb2 && sizea2==sizec2);

    // temp variables
    cplx *a_arr,*b_arr,*c_arr,*c_arr1;
    a_arr = new cplx[sizea1 * sizea2];
    b_arr = new cplx[sizeb1 * sizeb2];
    c_arr = new cplx[sizec1 * sizec2];
    c_arr1 = new cplx[sizec1 * sizec2];


    // Lesser C^<_{c1,c2}(m,tstp) = ii * A^<_{a1,a2}(m,tstp)*B^<_{b1,b2}(m,tstp)
    ZMatrixMap ALes(ales, sizea1, sizea2);
    ZMatrixMap BLes(bles, sizeb1, sizeb2);
    ZMatrixMap CLes(cles, sizec1, sizec2);
    CLes(c1,c2)=std::complex<double>(0.0,1.0) * ALes(a1,a2)*BLes(b1,b2);
            
    // Retarded is evaluated as C^R=C^>-C^<
    // A^>_{a1,a2}(tstp,m) = A^R_{a1,a2}(tstp,m) - A^{cc,<,*}_{a2,a1}(tstp,m)
    ZMatrixMap ARet(aret, sizea1, sizea2);
    ZMatrixMap ALesCC(accles, sizea1, sizea2);
    ZMatrixMap AGtr(a_arr, sizea1, sizea2);
    AGtr=ARet-ALesCC.adjoint();
    // Retarded B^>_{b1,b2}(tstp,m) = B^R_{b1,b2}(tstp,m) - B^{cc,<}_{b2,b1}(tstp,m)
    ZMatrixMap BRet(bret, sizeb1, sizeb2);
    ZMatrixMap BLesCC(bccles, sizeb1, sizeb2);
    ZMatrixMap BGtr(b_arr, sizeb1, sizeb2);
    BGtr=BRet-BLesCC.adjoint();
    ZMatrixMap CGtr(c_arr, sizec1, sizec2);
    // C^>_{c1,c2}(tstp,m) = ii * A^>_{a1,a2}(tstp,m)*B^>_{b2,b1}(m,tstp)
    CGtr(c1,c2)=std::complex<double>(0.0,1.0) * AGtr(a1,a2)*BGtr(b1,b2);
    // C^<_{c1,c2}(tstp,m) = ii * A^{cc,<,*}_{a2,a1}(tstp,m)*B^{cc,<,*}_{b2,b1}(tstp,m)
    ZMatrixMap CLes_tt1(c_arr1, sizec1, sizec2);
    CLes_tt1(c1,c2)=std::complex<double>(0.0,1.0) * std::conj(ALesCC(a2,a1)) * std::conj(BLesCC(b2,b1)); 
    // C^R=C^>-C^<
    ZMatrixMap CRet(cret, sizec1, sizec2);
    CRet(c1,c2)=CGtr(c1,c2)-CLes_tt1(c1,c2);
    
    // Mixed component
        
    for (int m = 0; m < r; m++) {
        ZMatrixMap ATV(atv + m * sizea1 *sizea2, sizea1, sizea2);
        ZMatrixMap BTV(btv +  m * sizeb1 * sizeb2 , sizeb1, sizeb2);
        ZMatrixMap CTV(ctv + m * sizec1 *sizec2, sizec1, sizec2);

        CTV(c1,c2)=std::complex<double>(0.0,1.0) * ATV(a1,a2) * BTV(b1,b2);
    }
    
    delete a_arr;
    delete b_arr;
    delete c_arr;
    delete c_arr1;
}

} // namespace hodlr
