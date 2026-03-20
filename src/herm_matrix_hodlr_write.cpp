#include <vector>
#include <iostream>
#include <string>

#include "h_nessi/herm_matrix_hodlr.hpp"

namespace h_nessi {

herm_matrix_hodlr::herm_matrix_hodlr(h5e::File &in, std::string label) :
  r_(in.getDataSet(label + std::string("/r")).read<int>())
{
  nt_ = in.getDataSet(label + std::string("/nt")).read<int>();
  nlvl_ = in.getDataSet(label + std::string("/nlvl")).read<int>();
  svdtol_ = in.getDataSet(label + std::string("/svdtol")).read<double>();
  size1_ = in.getDataSet(label + std::string("/size1")).read<int>();
  size2_ = in.getDataSet(label + std::string("/size2")).read<int>();
  ntau_ = in.getDataSet(label + std::string("/r")).read<int>();
  sig_ = in.getDataSet(label + std::string("/sig")).read<int>();
  k_ = in.getDataSet(label + std::string("/k")).read<int>();
  tstpmk_ = in.getDataSet(label + std::string("/tstpmk")).read<int>();
  built_blocks_ = in.getDataSet(label + std::string("/built_blocks")).read<int>();
  can_extrap_ = in.getDataSet(label + std::string("/can_extrap")).read<bool>();
  
  mat_.resize(r_ * size1_ * size2_);
  DMatrixMap(mat_.data(), r_, size1_*size2_) = in.getDataSet(label + std::string("/mat")).read<DMatrix>();
  GMConvTensMatrix_ = in.getDataSet(label + std::string("/GMConvTensMatrix")).read<DMatrix>();

  curr_timestep_ret_ = in.getDataSet(label + std::string("/curr_timestep_ret")).read<std::vector<cplx>>();
  curr_timestep_les_ = in.getDataSet(label + std::string("/curr_timestep_les")).read<std::vector<cplx>>();

  init_shape(nt_,nlvl_);
  blklen_ = in.getDataSet(label + std::string("/blklen")).read<std::vector<int>>();

  std::vector<int> nrows(nbox_),ncols(nbox_);
  for(int i=0;i<nbox_;i++){
    nrows[i]=blkr2_[i]-blkr1_[i]+1;
    ncols[i]=blkc2_[i]-blkc1_[i]+1;
  }

  // real time
  ret_=ret_blocks(nbox_,ndir_,nrows,ncols,svdtol_,size1_,size2_);
  les_=les_blocks(nbox_,nrows,ncols,svdtol_,size1_,size2_);
  tv_ =tv_blocks(size1_, size2_, nt_, r_, svdtol_);

  // Retarded block data
  for(int b = 0; b < built_blocks_; b++) {
    for(int i = 0; i < size1_; i++) {
      for(int j = 0; j < size2_; j++) {
        std::string prelabel = label + "/ret/" + std::to_string(b)+"_"+std::to_string(i)+"_"+std::to_string(j)+"/";
        double epsrank = in.getDataSet(prelabel + "epsrank").read<int>();
        ret_.data().blocks()[b][i][j].set_epsrank(epsrank); 
        ret_.data().blocks()[b][i][j].U()       = in.getDataSet(prelabel + "U"      ).read<ZMatrix>();
        ret_.data().blocks()[b][i][j].V()       = in.getDataSet(prelabel + "V"      ).read<ZMatrix>();
        auto a = in.getDataSet(prelabel + "S").read<std::vector<double>>();
        ret_.data().blocks()[b][i][j].S() = Eigen::Map<Eigen::VectorXd>(a.data(), epsrank);
      }
    }
  }

  // Retarded diagonal data
  ZMatrixMap(ret_.dirtricol(), ndir_*size1_*size2_, 1) = in.getDataSet(label + std::string("/ret/dirtricol")).read<ZMatrix>(); 

  // Lesser diagonal data
  ret_corr_below_tri_ = in.getDataSet(label + std::string("/ret/ret_corr_below_tri")).read<std::vector<cplx>>();

  // Retarded data on left edge
  ret_left_edge_ = in.getDataSet(label + std::string("/ret_left_edge")).read<std::vector<cplx>>();

  // Lesser block data
  for(int b = 0; b < built_blocks_; b++) {
    for(int i = 0; i < size1_; i++) {
      for(int j = 0; j < size2_; j++) {
        std::string prelabel = label + "/les/" + std::to_string(b)+"_"+std::to_string(i)+"_"+std::to_string(j)+"/";
        double epsrank = in.getDataSet(prelabel + "epsrank").read<int>();
        les_.data().blocks()[b][i][j].set_epsrank(epsrank); 
        les_.data().blocks()[b][i][j].U()       = in.getDataSet(prelabel + "U"      ).read<ZMatrix>();
        les_.data().blocks()[b][i][j].V()       = in.getDataSet(prelabel + "V"      ).read<ZMatrix>();
        auto a = in.getDataSet(prelabel + "S").read<std::vector<double>>();
        les_.data().blocks()[b][i][j].S() = Eigen::Map<Eigen::VectorXd>(a.data(), epsrank);
      }
    }
  }

  // Lesser diagonal data
  les_dir_square_ = in.getDataSet(label + std::string("/les_dir_square")).read<std::vector<cplx>>();

  // Lesser data on left edge
  les_left_edge_ = in.getDataSet(label + std::string("/les_left_edge")).read<std::vector<cplx>>();

  // TV data
  tv_.data_vector() = in.getDataSet(label + std::string("/tv")).read<std::vector<cplx>>();
  tv_.data_trans_vector() = in.getDataSet(label + std::string("/tv_trans")).read<std::vector<cplx>>();

  // timing
  try {
    timing = in.getDataSet(label + std::string("/timing")).read<DMatrix>();
  }
  catch (const HighFive::DataSetException& e) {
    timing = DMatrix(nt_, 22);
  }
}

void herm_matrix_hodlr::write_checkpoint_hdf5(h5e::File &out, std::string label) {
  // scalar parameters
  h5e::dump(out, label + std::string("/nt"), nt_);
  h5e::dump(out, label + std::string("/r"), r_);
  h5e::dump(out, label + std::string("/nlvl"), nlvl_);
  h5e::dump(out, label + std::string("/svdtol"), svdtol_);
  h5e::dump(out, label + std::string("/size1"), size1_);
  h5e::dump(out, label + std::string("/size2"), size2_);
  h5e::dump(out, label + std::string("/sig"), sig_);
  h5e::dump(out, label + std::string("/k"), k_);
  h5e::dump(out, label + std::string("/tstpmk"), tstpmk_);
  h5e::dump(out, label + std::string("/built_blocks"), built_blocks_);
  h5e::dump(out, label + std::string("/can_extrap"), can_extrap_);

  // Matsubara data
  h5e::dump(out, label + std::string("/mat"), DMatrixMap(mat_.data(), r_, size1_*size2_));
  h5e::dump(out, label + std::string("/GMConvTensMatrix"), GMConvTensMatrix_);

  // Current timestep data
  h5e::dump(out, label + std::string("/curr_timestep_ret"), curr_timestep_ret_);
  h5e::dump(out, label + std::string("/curr_timestep_les"), curr_timestep_les_);

  // Number of rows each block has built
  h5e::dump(out, label + std::string("/blklen"), blklen_);

  // Retarded block data
  for(int b = 0; b < built_blocks_; b++) {
    for(int i = 0; i < size1_; i++) {
      for(int j = 0; j < size2_; j++) {
        std::string prelabel = label + "/ret/" + std::to_string(b)+"_"+std::to_string(i)+"_"+std::to_string(j)+"/";
        h5e::dump(out, prelabel + "epsrank", ret_.data().blocks()[b][i][j].epsrank());
        h5e::dump(out, prelabel + "rows", ret_.data().blocks()[b][i][j].rows());
        h5e::dump(out, prelabel + "cols", ret_.data().blocks()[b][i][j].cols());
        h5e::dump(out, prelabel + "U", ret_.data().blocks()[b][i][j].U());
        h5e::dump(out, prelabel + "V", ret_.data().blocks()[b][i][j].V());
        h5e::dump(out, prelabel + "S", ret_.data().blocks()[b][i][j].S());
      }
    }
  }
  
  // Retarded diagonal data
  h5e::dump(out, label + std::string("/ret/dirtricol"), ZMatrixMap(ret_.dirtricol(), ndir_*size1_*size2_, 1));

  // Retarded data k below diagonal
  h5e::dump(out, label + std::string("/ret/ret_corr_below_tri"), ret_corr_below_tri_);

  // Retarded data on left edge
  h5e::dump(out, label + std::string("/ret_left_edge"), ret_left_edge_);

  // Les block data
  for(int b = 0; b < built_blocks_; b++) {
    for(int i = 0; i < size1_; i++) {
      for(int j = 0; j < size2_; j++) {
        std::string prelabel = label + "/les/"+std::to_string(b)+"_"+std::to_string(i)+"_"+std::to_string(j)+"/";
        h5e::dump(out, prelabel + "epsrank", les_.data().blocks()[b][i][j].epsrank());
        h5e::dump(out, prelabel + "rows", les_.data().blocks()[b][i][j].rows());
        h5e::dump(out, prelabel + "cols", les_.data().blocks()[b][i][j].cols());
        h5e::dump(out, prelabel + "U", les_.data().blocks()[b][i][j].U());
        h5e::dump(out, prelabel + "V", les_.data().blocks()[b][i][j].V());
        h5e::dump(out, prelabel + "S", les_.data().blocks()[b][i][j].S());
      }
    }
  }

  // Lesser diagonal data
  h5e::dump(out, label + std::string("/les_dir_square"), les_dir_square_);

  // Lesser data on left edge
  h5e::dump(out, label + std::string("/les_left_edge"), les_left_edge_);

  // TV data
  h5e::dump(out, label + std::string("/tv"), tv_.data_vector());
  h5e::dump(out, label + std::string("/tv_trans"), tv_.data_trans_vector());

  // timing
  h5e::dump(out, label + std::string("/timing"), timing);
}

void herm_matrix_hodlr::write_to_hdf5(h5e::File &out, std::string label) {
  h5e::dump(out, label + std::string("nt"), nt_);
  h5e::dump(out, label + std::string("r"), r_);
  h5e::dump(out, label + std::string("size1"), size1_);
  h5e::dump(out, label + std::string("size2"), size2_);
  h5e::dump(out, label + std::string("ndir"), ndir_);
  h5e::dump(out, label + std::string("nbox"), nbox_);
  h5e::dump(out, label + std::string("nlvl"), nlvl_);
  h5e::dump(out, label + std::string("maxdir"), maxdir_);
  h5e::dump(out, label + std::string("k"), k_);
  h5e::dump(out, label + std::string("tstpmk"), tstpmk_);
  h5e::dump(out, label + std::string("built_blocks"), built_blocks_);

  if (!out.exist("geometry/blkr1")) {
    h5e::dump(out, "geometry/blkr1", blkr1_,h5e::DumpMode::Overwrite);
    h5e::dump(out, "geometry/blkr2", blkr2_,h5e::DumpMode::Overwrite);
    h5e::dump(out, "geometry/blkc1", blkc1_,h5e::DumpMode::Overwrite);
    h5e::dump(out, "geometry/blkc2", blkc2_,h5e::DumpMode::Overwrite);
    h5e::dump(out, "geometry/blklevel", blklevel_,h5e::DumpMode::Overwrite);
    h5e::dump(out, "geometry/blkdirheight", blkdirheight_,h5e::DumpMode::Overwrite);
    h5e::dump(out, "geometry/blkndirstart", blkndirstart_,h5e::DumpMode::Overwrite);
    h5e::dump(out, "geometry/c1_dir", c1_dir_,h5e::DumpMode::Overwrite);
    h5e::dump(out, "geometry/r2_dir", r2_dir_,h5e::DumpMode::Overwrite);
    h5e::dump(out, "geometry/ntri", ntri_,h5e::DumpMode::Overwrite);
    h5e::dump(out, "geometry/blklen", blklen_,h5e::DumpMode::Overwrite);
  }

  write_mat_to_hdf5(out, label);

  for(int b = 0; b < nbox_; b++) {
    for(int i = 0; i < size1_; i++) {
      for(int j = 0; j < size2_; j++) {
        std::string prelabel = label + "ret/" + std::to_string(b)+"_"+std::to_string(i)+"_"+std::to_string(j)+"/";
        h5e::dump(out, prelabel + "epsrank", ret_.data().blocks()[b][i][j].epsrank());
        h5e::dump(out, prelabel + "rows", ret_.data().blocks()[b][i][j].rows());
        h5e::dump(out, prelabel + "cols", ret_.data().blocks()[b][i][j].cols());
        h5e::dump(out, prelabel + "U", ret_.data().blocks()[b][i][j].U());
        h5e::dump(out, prelabel + "V", ret_.data().blocks()[b][i][j].V());
        h5e::dump(out, prelabel + "S", ret_.data().blocks()[b][i][j].S());
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
        h5e::dump(out, prelabel + "U", les_.data().blocks()[b][i][j].U());
        h5e::dump(out, prelabel + "V", les_.data().blocks()[b][i][j].V());
        h5e::dump(out, prelabel + "S", les_.data().blocks()[b][i][j].S());
      }
    }
  }

  // les_dir data is stored as (i,j,t,t') but since each square along diagonal is not, in general, the same size
  // we can only print it out as (i,j,z)
  std::array<size_t, 3> les_dir_shape = {
    static_cast<size_t>(size1_),
    static_cast<size_t>(size2_),
    static_cast<size_t>(len_les_dir_square_)
  };
  auto dset_les = out.createDataSet<cplx>(label + std::string("les/dir"), HighFive::DataSpace(les_dir_shape));
  dset_les.write_raw(les_dir_square_.data());

  // ret_dir is (z,i,j), where z starts at diagonal and goes downwards
  std::array<size_t, 3> ret_dir_shape = {
    static_cast<size_t>(ndir_),
    static_cast<size_t>(size1_),
    static_cast<size_t>(size2_)
  };
  auto dset_ret = out.createDataSet<cplx>(label + std::string("ret/dirtricol"), HighFive::DataSpace(ret_dir_shape));
  dset_ret.write_raw(ret_.dirtricol());

  // TV data
  std::array<size_t, 4> shape = {
    static_cast<size_t>(nt_),
    static_cast<size_t>(r_),
    static_cast<size_t>(size1_),
    static_cast<size_t>(size2_)
  };
  auto dset = out.createDataSet<cplx>(label + std::string("/tv"), HighFive::DataSpace(shape));
  dset.write_raw(tv_.data_vector().data()); 

  // timing
  h5e::dump(out, label + std::string("/timing"), timing);
}

void herm_matrix_hodlr::write_rho_to_hdf5(h5e::File &out, std::string label, dlr_info &dlr) {
  ZMatrix rho_t((nt_+1)*size1_*size2_, 1);
  ZMatrix rho(size1_, size2_);
  for(int i = -1; i < nt_; i++) {
    density_matrix(i, dlr, rho);
    ZMatrixMap(rho_t.data() + (i+1)*size1_*size2_, size1_, size2_) = rho;
  }
  std::array<size_t, 3> shape = {
    static_cast<size_t>(nt_+1),
    static_cast<size_t>(size1_),
    static_cast<size_t>(size2_)
  };
  auto dset = out.createDataSet<cplx>(label, HighFive::DataSpace(shape));
  dset.write_raw(rho_t.data());
}

void herm_matrix_hodlr::write_rank_to_hdf5(h5e::File &out, std::string label) {
  IColVector ranksR(nlvl_*size1_*size2_);
  IColVector ranksL(nlvl_*size1_*size2_);

  for(int b = 0, i = 0; b < nlvl_; i+=std::pow(2,b), b++) {
    for(int j = 0; j < size1_; j++) {
      for(int k = 0; k < size2_; k++) {
        ranksR(b*size1_*size2_ + j*size1_ + k) = ret_.data().blocks()[i][j][k].epsrank();
        ranksL(b*size1_*size2_ + j*size1_ + k) = les_.data().blocks()[i][j][k].epsrank();
      }
    }
  }

  std::array<size_t, 3> shape = {
    static_cast<size_t>(nlvl_),
    static_cast<size_t>(size1_),
    static_cast<size_t>(size2_)
  };
  auto dset = out.createDataSet<int>(label + "/ranksR", HighFive::DataSpace(shape));
  dset.write_raw(ranksR.data());
  dset = out.createDataSet<int>(label + "/ranksL", HighFive::DataSpace(shape));
  dset.write_raw(ranksL.data());
}

void herm_matrix_hodlr::write_mat_to_hdf5(h5e::File &out, std::string label) {
  std::array<size_t, 3> shape = {
    static_cast<size_t>(r_),
    static_cast<size_t>(size1_),
    static_cast<size_t>(size2_)
  };
  auto dset = out.createDataSet<double>(label + "/mat", HighFive::DataSpace(shape));
  dset.write_raw(mat_.data());
}

void herm_matrix_hodlr::write_curr_to_hdf5(h5e::File &out, std::string label) {
  h5e::dump(out, label + std::string("curr_timestep_ret"), curr_timestep_ret_);
  h5e::dump(out, label + std::string("curr_timestep_les"), curr_timestep_les_);
}

void herm_matrix_hodlr::write_GR0_to_hdf5(h5e::File &out, std::string label) {
    ZMatrix GR_t(nt_*size1_*size1_, 1);
    for(int i = 0; i < nt_; i++) {
      get_ret(i, 0, GR_t.data() + i*size1_*size1_);
    }
    
  std::array<size_t, 3> shape = {
    static_cast<size_t>(nt_),
    static_cast<size_t>(size1_),
    static_cast<size_t>(size2_)
  };
  auto dset = out.createDataSet<cplx>(label + "/R0", HighFive::DataSpace(shape));
  dset.write_raw(GR_t.data());
}

void herm_matrix_hodlr::write_GL0_to_hdf5(h5e::File &out, std::string label) {
    ZMatrix GL_t(nt_*size1_*size1_, 1);
    for(int i = 0; i < nt_; i++) {
      get_les(0, i, GL_t.data() + i*size1_*size1_);
    }
    
  std::array<size_t, 3> shape = {
    static_cast<size_t>(nt_),
    static_cast<size_t>(size1_),
    static_cast<size_t>(size2_)
  };
  auto dset = out.createDataSet<cplx>(label + "/L0", HighFive::DataSpace(shape));
  dset.write_raw(GL_t.data());
}
} // namespace
