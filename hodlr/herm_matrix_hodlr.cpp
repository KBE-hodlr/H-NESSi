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
    can_extrap_ = false;

    ret_=hodlr::ret_blocks();
    les_=hodlr::les_blocks();
    tv_=hodlr::tv_blocks();
    mat_.clear();
    GMConvTensMatrix_ = DMatrix();

    nlvl_=0;
    nbox_=0;
    svdtol_=0.0;
  
    curr_timestep_ret_ = std::vector<cplx>(0);
    curr_timestep_les_ = std::vector<cplx>(0);
    les_dir_square_ = std::vector<cplx>(0);
    les_dir_square_first_index_ = std::vector<int>(0);
    les_left_edge_ = std::vector<cplx>(0);
    ret_left_edge_ = std::vector<cplx>(0);
    ret_corr_below_tri_ = std::vector<cplx>(0);

    tstpmk_ = 0;
    k_ = 0;
    ntau_ = 0;
    r_ = 0;
    nt_ = 0;
    ndir_=0;
    size1_ = 0;
    size2_ = 0;
    sig_ = -1;

    timing = DMatrix::Zero(nt_, 22);
}

// destructor is defaulted in header

herm_matrix_hodlr::herm_matrix_hodlr(int nt,int r,int nlvl,double svdtol,int size1,int size2,int sig,int k)
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
    can_extrap_ = false;
 
    mat_.resize(r_ * size1_ * size2_);
    GMConvTensMatrix_ = DMatrix(r_ , r_ * size1_ * size2_);

    curr_timestep_ret_ = std::vector<cplx>((k_+1) * (nt_ + 1)  * size1_ * size2_);
    curr_timestep_les_ = std::vector<cplx>((k_+1) * (nt_ + 1) * size1_ * size2_);

    init_shape(nt,nlvl);

    std::vector<int> nrows(nbox_),ncols(nbox_);
    for(int i=0;i<nbox_;i++){
        nrows[i]=blkr2_[i]-blkr1_[i]+1;
        ncols[i]=blkc2_[i]-blkc1_[i]+1;
    }

    // Mat
    for(int i=0;i<r_;i++){
        DMatrixMap(matptr(i), size1, size2) = DMatrix::Zero(size1,size2);
    }

    // real time
    ret_=hodlr::ret_blocks(nbox_,ndir_,nrows,ncols,svdtol,size1_,size2_);
    les_=hodlr::les_blocks(nbox_,nrows,ncols,svdtol,size1_,size2_);
    tv_ =hodlr::tv_blocks(size1_, size2_, nt_, r_, svdtol);

    // timing
    timing = DMatrix::Zero(nt_, 22);

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

    blklen_ = std::vector<int>(nbox_,0);
    blkdirheight_ = std::vector<int>(nbox_+1,0);
    
    blkdirheight_[0] = blkr1_[0];
    for(int b = 1; b < nbox_; b++) {
        blkdirheight_[b] = blkr1_[b] - blkr1_[b-1];
    }
    blkdirheight_[nbox_] = nt_ - blkr1_[nbox_-1];
    

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
    ret_left_edge_ = std::vector<cplx>(size1_ * size2_ * nt_ * (k_+1));

    ret_corr_index_t_ = std::vector<int>(nt_);
    int nrcbt = 0;
    for(int t = 0; t < nt_; t++) {
      ret_corr_index_t_[t] = nrcbt;
      nrcbt += std::max(0,k_+1-(r2_dir_[t]-t+1));
    }
    ret_corr_below_tri_ = std::vector<cplx>(size1_ * size2_ * nrcbt);

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
    }

    // Levels of all blocks [closest to diagonal=1, \ldots]
    std::fill(blklevel_.begin(), blklevel_.end(), 1);
    for(int l=1;l<nlvl_;l++){
        for(int j=(int)round(pow(2,l));j<(int)round(pow(2,nlvl_));j=j+(int)round(pow(2,l))){
            blklevel_[j-1]=l+1;
        }
    }
    // First column indices of all blocks
    std::vector<int> il(nlvl_,1);
    std::fill(blkc1_.begin(), blkc1_.end(), 0);
    for(int i=0;i<nbox;i++){
        l=blklevel_[i];
        blkc1_[i]=is[(il[l-1]-1)*(int)round(pow(2,l))];
        il[l-1]=il[l-1]+1;        
    }

    // Last row indices of all blocks
    std::fill(blkr2_.begin(), blkr2_.end(), 0);
    for(int i=0;i<nbox;i++){
        l=blklevel_[i];
        blkr2_[i]=is[i+(int)round(pow(2,l-1))+1]-1;
        if(blkr2_[i]==is[(int)round(pow(2,nlvl_))]-1){
            blkr2_[i]=blkr2_[i];
        }
    }
    //  Last column indices of all blocks
    for(int i=0; i<blkc2_.size(); ++i) blkc2_[i]=blkr1_[i]-1;
    
    // Maximum possible number of directly-stored entries in a row
    int val1 = 0; // size1 of triangular region
    int val2 = 0; // size2 of triangular region
    for(int i=0;i<nbox;i++){
        if(blklevel_[i]==1){ //only the blocks closest to the diagonal
            val1=std::max(blkr1_[i]-blkc1_[i]+1,val1);
        }
    }
    // Column index of first directly-stored entry in each row
    for(int i=0;i<nbox;i++){
        for(int j=blkr1_[i];j<=blkr2_[i];j++){
            if(c1_dir_[j]<blkc2_[i]+1) c1_dir_[j]=blkc2_[i]+1;
        }
    }

    ndir_=0;
    // Indices of directly stored entries
    for(int i=0;i<nt_;i++){
        for(int j=c1_dir_[i];j<=i;j++){
            ndir_+=1;
        }
    }
    

    return std::max(val1,val2);
}


// Imaginary time convolution \int_0^\beta d\tau' \Sigma_{ik}(T,\tau') G_{kj}(\tau-\tau') in dlr basis is
// \phi_{abc} G_{c,kj} \Sigma_{b,ik}
// The first part is the same for each integral, so we construct
// C_{ajbk} = -\phi_{abc} G_{-c,kj}
void herm_matrix_hodlr::initGMConvTensor(dlr_info &dlr){
    GMConvTensMatrix_.resize(r_*size1_*size2_, r_);

    std::vector<double> G1(r_ * size1_ * size2_);
    std::vector<double> G2(r_ * size1_ * size2_);

    int es = size1_*size2_;
    int one = 1;

    // A_{cjk} = G_{ckj}
    for(int i = 0; i<r_; i++) {
        DMatrixMap(G1.data()+i*es, size1_, size2_).noalias() = DMatrixMap(mat_.data()+i*es, size1_, size2_).transpose();
    }

    // B_{jkc} = -A_{cjk}
    DMatrixMap(G2.data(), es, r_).noalias() = - DMatrixMap(G1.data(), r_, es).transpose();

    // C_{jkc} = B_{jk(ntau-c-1)}
    for(int i = 0; i<es; i++) {
        c_dlr_it2itr(&r_, &one, dlr.it2itr(), G2.data()+i*r_, G1.data()+i*r_);
        // This calls some fortran code that is not threadsafe
        // this needs protection if the same dlr object is being used to update two different G
        // such as in the case of large numbers of k-points
        #pragma omp critical
        c_dlr_it2cf(&r_,&one, dlr.it2cf(), dlr.it2cfp(), G1.data()+i*r_, G2.data()+i*r_);
    }

    // D_{jkba} = C_{jkc} Phi_{cba}
    DMatrixMap(GMConvTensMatrix_.data(), es, r_*r_).noalias() = DMatrixMap(G2.data(), es, r_) * DMatrixMap(dlr.phi(), r_, r_*r_);

    // E_{ajkb} = D_{jkba}
    GMConvTensMatrix_.transposeInPlace();

    // C_{ajbk} = E_{ajkb}
    for( int i = 0; i < r_*size1_; i++) {
        DMatrixMap(GMConvTensMatrix_.data() + i*r_*size1_, r_, size2_) = DMatrixMap(GMConvTensMatrix_.data() + i*r_*size1_, size2_, r_).transpose().eval();
    }

    // G1 and G2 are managed by std::vector and will be freed automatically
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
        memcpy(G.mat(), mat_.data(), r_ * size1_ * size2_ * sizeof(double));
    }else{ // set last kt timesteps together
        memcpy(G.curr_timestep_ret_ptr(tstp,0),this->curr_timestep_ret_ptr(tstp,0),(nt_ +1) * size1_ * size2_* sizeof(cplx));
        memcpy(G.curr_timestep_les_ptr(0,tstp),this->curr_timestep_les_ptr(0,tstp),(nt_ +1) * size1_ * size2_ * sizeof(cplx));
        memcpy(G.tvptr(tstp,0),this->tvptr(tstp,0),r_ * size1_ * size2_ * sizeof(cplx));
        memcpy(G.tvptr_trans(tstp,0),this->tvptr_trans(tstp,0),r_ * size1_ * size2_ * sizeof(cplx));
    }
}

// Copy last kt timesteps
void herm_matrix_hodlr::get_timestep_curr(herm_matrix_hodlr &G){
    tstpmk_=G.tstpmk();
    for(int i=tstpmk_;i<=tstpmk_+k_;i++){
        get_timestep_curr(i,G);
    }
}

// set is only a reasonable operation for directly stored timesteps
void herm_matrix_hodlr::set_timestep_curr(int tstp,herm_matrix_hodlr &G){
    assert(tstp >= -1 && tstp <= nt_ && tstpmk_==G.tstpmk());
    if(tstp==-1){
        memcpy(mat_.data(), G.mat(), r_ * size1_ * size2_ * sizeof(double));
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
        // Ret
        int len= (tstp + 1) * size1_ * size2_;
        x1=this->curr_timestep_ret_ptr(tstp,0);
        x2=G.curr_timestep_ret_ptr(tstp,0);
        for(int i=0;i<len;i++){
            x1[i]+=alpha * x2[i];
        }
        // Les
        x1=this->curr_timestep_les_ptr(0,tstp);
        x2=G.curr_timestep_les_ptr(0,tstp);
        for(int i=0;i<len;i++){
            x1[i]+=alpha * x2[i];
        }
        // Mixed component
        len= r_  * size1_ * size2_;
        x1=this->tvptr(tstp,0);
        x2=G.tvptr(tstp,0);
        for(int i=0;i<len;i++){
            x1[i]+=alpha * x2[i];
        }
        // Mixed component transpose
        len= r_  * size1_ * size2_;
        x1=this->tvptr_trans(tstp,0);
        x2=G.tvptr_trans(tstp,0);
        for(int i=0;i<len;i++){
            x1[i]+=alpha * x2[i];
        }
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

double herm_matrix_hodlr::get_memory_usage(bool print_memory_usage) {
    
    double D = (double)sizeof(double)/(1e9);
    double Z = 2*D;
    double I = (double)sizeof(int)/(1e9);

    double total = 0;
    double tim = 0;
    double geometry = 0;
    double matdata = 0;
    double gmct = 0;
    double currtstp = 0;
    double retdirect = 0;
    double lesdirect = 0;
    double leftcorr = 0;
    double tvdata = 0;
    double retdata = 0;
    double lesdata = 0;

    tim = D * timing.rows() * timing.cols();

    geometry += I * t_to_dirlvl_.size();
    geometry += I * blkr1_.size();
    geometry += I * blkr2_.size();
    geometry += I * blkc1_.size();
    geometry += I * blkc2_.size();
    geometry += I * blklevel_.size();
    geometry += I * blkdirheight_.size();
    geometry += I * blkndirstart_.size();
    geometry += I * c1_dir_.size();
    geometry += I * r2_dir_.size();
    geometry += I * ntri_.size();
    geometry += I * blklen_.size();
    geometry += I * les_dir_square_first_index_.size();
    geometry += I * ret_corr_index_t_.size();

    matdata = D * r_ * size1_ * size2_;

    gmct = D * GMConvTensMatrix_.rows() * GMConvTensMatrix_.cols();

    currtstp += Z * curr_timestep_ret_.size();
    currtstp += Z * curr_timestep_les_.size();

    retdirect += Z * ret_corr_below_tri_.size();
    retdirect += Z * ret_.ndir() * size1_ * size2_;

    lesdirect = Z * les_dir_square_.size();
  
    leftcorr += Z * ret_left_edge_.size();
    leftcorr += Z * les_left_edge_.size();

    tvdata += Z * tv_.data_vector().size();
    tvdata += Z * tv_.data_trans_vector().size();

    for(int b = 0; b < built_blocks_; b++) {
      for(int i = 0; i < size1_; i++) {
        for(int j = 0; j < size1_; j++) {
          retdata += Z * ret_.data().blocks()[b][i][j].U().rows() * ret_.data().blocks()[b][i][j].U().cols();
          retdata += Z * ret_.data().blocks()[b][i][j].V().rows() * ret_.data().blocks()[b][i][j].V().cols();
          lesdata += Z * les_.data().blocks()[b][i][j].U().rows() * ret_.data().blocks()[b][i][j].U().cols();
          lesdata += Z * les_.data().blocks()[b][i][j].V().rows() * ret_.data().blocks()[b][i][j].V().cols();
        }
      }
    }
  
    // ret and les
    double nocompression = 2 * Z * nt_* (nt_+1) / 2. * size1_ * size2_;
    // mixed
    nocompression += Z * nt_ * r_ * size1_ * size2_;

    const int total_width = 10;
    const int precision = 2;

    total = tim + geometry + gmct + currtstp + retdirect + lesdirect + leftcorr + matdata + tvdata + retdata + lesdata;

    if(print_memory_usage){ 
        std::cout << "==========================================" << std::endl;
        std::cout << "==  Memory usage for herm_matrix_hodlr  ==" << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << "==                                      ==" << std::endl;
        printf(      "==  timing             :%*.*f GB   ==", total_width, precision, tim); std::cout << std::endl;
        printf(      "==  geometry           :%*.*f GB   ==", total_width, precision, geometry); std::cout << std::endl;
        printf(      "==  gmct               :%*.*f GB   ==", total_width, precision, gmct); std::cout << std::endl;
        printf(      "==  currtstp           :%*.*f GB   ==", total_width, precision, currtstp); std::cout << std::endl;
        printf(      "==  retdir             :%*.*f GB   ==", total_width, precision, retdirect); std::cout << std::endl;
        printf(      "==  lesdir             :%*.*f GB   ==", total_width, precision, lesdirect); std::cout << std::endl;
        printf(      "==  leftcorr           :%*.*f GB   ==", total_width, precision, leftcorr); std::cout << std::endl;
        printf(      "==  mat                :%*.*f GB   ==", total_width, precision, matdata); std::cout << std::endl;
        printf(      "==  tv                 :%*.*f GB   ==", total_width, precision, tvdata); std::cout << std::endl;
        printf(      "==  ret                :%*.*f GB   ==", total_width, precision, retdata); std::cout << std::endl;
        printf(      "==  les                :%*.*f GB   ==", total_width, precision, lesdata); std::cout << std::endl;
        std::cout << "==                                      ==" << std::endl;
        printf(      "==  total              :%*.*f GB   ==", total_width, precision, total); std::cout << std::endl;
        printf(      "==  no compression     :%*.*f GB   ==", total_width, precision, nocompression); std::cout << std::endl;
        printf(      "==  compression factor :%*.*f      ==", total_width, precision, nocompression/total); std::cout << std::endl;
        std::cout << "==                                      ==" << std::endl;
        std::cout << "==========================================" << std::endl << std::endl;
    }

    return total;
}

} // namespace hodlr
