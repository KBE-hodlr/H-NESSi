#include <vector>
#include <iostream>
#include <string>

#include "h_nessi/herm_matrix_hodlr.hpp"
#include "h_nessi/checks.hpp"

namespace h_nessi {

namespace {
// HM-C11 (tier 2): a checkpoint dataset must have the size implied by the stored geometry
void require_dataset_size(const char *where, const std::string &name, long long got, long long expected) {
  HNESSI_REQUIRE(got == expected, where, HNESSI_KV(name), HNESSI_KV(got), HNESSI_KV(expected),
                 detail::note{"the checkpoint dataset does not match the stored geometry (corrupt or foreign file)"});
}
}  // namespace

herm_matrix_hodlr::herm_matrix_hodlr(h5e::File &in, std::string label) :
  r_(in.getDataSet(h5_join({label, "r"})).read<int>())
{
  nt_ = in.getDataSet(h5_join({label, "nt"})).read<int>();
  nlvl_ = in.getDataSet(h5_join({label, "nlvl"})).read<int>();
  svdtol_ = in.getDataSet(h5_join({label, "svdtol"})).read<double>();
  size1_ = in.getDataSet(h5_join({label, "size1"})).read<int>();
  size2_ = in.getDataSet(h5_join({label, "size2"})).read<int>();
  ntau_ = in.getDataSet(h5_join({label, "r"})).read<int>();
  sig_ = in.getDataSet(h5_join({label, "sig"})).read<int>();
  k_ = in.getDataSet(h5_join({label, "k"})).read<int>();
  tstpmk_ = in.getDataSet(h5_join({label, "tstpmk"})).read<int>();
  built_blocks_ = in.getDataSet(h5_join({label, "built_blocks"})).read<int>();
  can_extrap_ = in.getDataSet(h5_join({label, "can_extrap"})).read<bool>();
  // HM-C11 (scalar part, tier 2): the stored geometry must be one the constructor accepts
  check_geometry_("herm_matrix_hodlr::herm_matrix_hodlr(h5e::File&,label)", nt_, r_, nlvl_, svdtol_, size1_, size2_,
                  sig_, k_);
  const char *where = "herm_matrix_hodlr::herm_matrix_hodlr(h5e::File&,label)";
  HNESSI_REQUIRE(0 <= tstpmk_ && tstpmk_ <= nt_, where, HNESSI_KV(tstpmk_), HNESSI_KV(nt_));
  const long long es = (long long)size1_ * size2_;

  mat_.resize(r_ * size1_ * size2_);
  {
    DMatrix m = in.getDataSet(h5_join({label, "mat"})).read<DMatrix>();
    HNESSI_REQUIRE(m.rows() == r_ && m.cols() == es, where, HNESSI_KV(m.rows()), HNESSI_KV(m.cols()), HNESSI_KV(r_),
                   HNESSI_KV(es), detail::note{"dataset mat must be r x size1*size2"});
    DMatrixMap(mat_.data(), r_, size1_*size2_) = m;
  }
  GMConvTensMatrix_ = in.getDataSet(h5_join({label, "GMConvTensMatrix"})).read<DMatrix>();
  require_dataset_size(where, "GMConvTensMatrix", GMConvTensMatrix_.size(), (long long)r_ * r_ * es);

  // Checkpoints without format_version (or older than 2) were written before the bosonic sign fix of
  // initGMConvTensor: for bosons the stored GMConvTensMatrix has the wrong sign.
  int format_version = 1;
  if(in.exist(h5_join({label, "format_version"}))) {
    format_version = in.getDataSet(h5_join({label, "format_version"})).read<int>();
  }
  // HM-C11: versions 1..checkpoint_format_version are readable (3: no tv_trans dataset; older files carry one, which
  // is ignored since WP6c, ISSUES H27)
  HNESSI_REQUIRE(1 <= format_version && format_version <= checkpoint_format_version, where, HNESSI_KV(format_version),
                 HNESSI_KV(checkpoint_format_version),
                 detail::note{"unknown checkpoint format (written by a newer H-NESSi?)"});
  if(format_version < 2 && sig_ == 1) {
    std::cerr << "h_nessi warning: bosonic checkpoint '" << label << "' has format_version " << format_version
              << " < 2. Its GMConvTensMatrix was written before the bosonic-sign fix and is wrong;"
              << " rebuild it with initGMConvTensor(dlr) before time stepping." << std::endl;
  }
  // ST-2: the stored tensor is accepted for the stored G^M unless it is a pre-v2 bosonic tensor (wrong sign) or the
  // writer recorded it as not built for its G^M (gmconv_valid = 0, written since WP6); the solvers refuse an invalid
  // tensor until initGMConvTensor is called.
  bool stored_valid = true;
  if(in.exist(h5_join({label, "gmconv_valid"}))) {
    stored_valid = in.getDataSet(h5_join({label, "gmconv_valid"})).read<int>() != 0;
  }
  gmconv_valid_ = stored_valid && !(format_version < 2 && sig_ == 1);
  gmconv_hash_ = mat_hash();

  curr_timestep_ret_ = in.getDataSet(h5_join({label, "curr_timestep_ret"})).read<std::vector<cplx>>();
  curr_timestep_les_ = in.getDataSet(h5_join({label, "curr_timestep_les"})).read<std::vector<cplx>>();
  require_dataset_size(where, "curr_timestep_ret", curr_timestep_ret_.size(), (long long)(k_+1) * (nt_+1) * es);
  require_dataset_size(where, "curr_timestep_les", curr_timestep_les_.size(), (long long)(k_+1) * (nt_+1) * es);

  init_shape(nt_,nlvl_);
  // init_shape sized every direct store from the geometry: the datasets read below must have the same sizes
  const long long len_les_dir = les_dir_square_.size(), len_edge = les_left_edge_.size(),
                  len_corr = ret_corr_below_tri_.size();
  blklen_ = in.getDataSet(h5_join({label, "blklen"})).read<std::vector<int>>();
  require_dataset_size(where, "blklen", blklen_.size(), nbox_);
  HNESSI_REQUIRE(0 <= built_blocks_ && built_blocks_ <= nbox_, where, HNESSI_KV(built_blocks_), HNESSI_KV(nbox_));
  for(int b = 0; b < nbox_; b++)
    HNESSI_REQUIRE(0 <= blklen_[b] && blklen_[b] <= blkr2_[b]-blkr1_[b]+1 && (b < built_blocks_ || blklen_[b] == 0),
                   where, HNESSI_KV(b), HNESSI_KV(blklen_[b]), HNESSI_KV(built_blocks_));

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
        std::string prelabel = h5_join({label, "ret", std::to_string(b)+"_"+std::to_string(i)+"_"+std::to_string(j)}) + "/";
        int epsrank = in.getDataSet(prelabel + "epsrank").read<int>();
        block &B = ret_.data().blocks()[b][i][j];
        B.set_epsrank(epsrank);
        B.U()       = in.getDataSet(prelabel + "U"      ).read<ZMatrix>();
        B.V()       = in.getDataSet(prelabel + "V"      ).read<ZMatrix>();
        auto a = in.getDataSet(prelabel + "S").read<std::vector<double>>();
        // HM-C11 per block: 0 <= epsrank <= min(rows built, cols), U is rows x epsrank, V is cols x epsrank
        HNESSI_REQUIRE(0 <= epsrank && epsrank <= std::min(blklen_[b], B.cols()) && B.U().rows() == blklen_[b] &&
                       B.U().cols() == epsrank && B.V().rows() == B.cols() && B.V().cols() == epsrank &&
                       (long long)a.size() >= epsrank, where, HNESSI_KV(prelabel), HNESSI_KV(epsrank),
                       HNESSI_KV(blklen_[b]), HNESSI_KV(B.cols()), HNESSI_KV(B.U().rows()), HNESSI_KV(B.U().cols()),
                       HNESSI_KV(B.V().rows()), HNESSI_KV(B.V().cols()), HNESSI_KV(a.size()));
        B.S() = Eigen::Map<Eigen::VectorXd>(a.data(), epsrank);
      }
    }
  }

  // Retarded diagonal data
  {
    ZMatrix d = in.getDataSet(h5_join({label, "ret", "dirtricol"})).read<ZMatrix>();
    require_dataset_size(where, "ret/dirtricol", d.size(), (long long)ndir_ * es);
    ZMatrixMap(ret_.dirtricol(), ndir_*size1_*size2_, 1) = ZMatrixMap(d.data(), ndir_*size1_*size2_, 1);
  }

  // Lesser diagonal data
  ret_corr_below_tri_ = in.getDataSet(h5_join({label, "ret", "ret_corr_below_tri"})).read<std::vector<cplx>>();
  require_dataset_size(where, "ret/ret_corr_below_tri", ret_corr_below_tri_.size(), len_corr);

  // Retarded data on left edge
  ret_left_edge_ = in.getDataSet(h5_join({label, "ret_left_edge"})).read<std::vector<cplx>>();
  require_dataset_size(where, "ret_left_edge", ret_left_edge_.size(), len_edge);

  // Lesser block data
  for(int b = 0; b < built_blocks_; b++) {
    for(int i = 0; i < size1_; i++) {
      for(int j = 0; j < size2_; j++) {
        std::string prelabel = h5_join({label, "les", std::to_string(b)+"_"+std::to_string(i)+"_"+std::to_string(j)}) + "/";
        int epsrank = in.getDataSet(prelabel + "epsrank").read<int>();
        block &B = les_.data().blocks()[b][i][j];
        B.set_epsrank(epsrank);
        B.U()       = in.getDataSet(prelabel + "U"      ).read<ZMatrix>();
        B.V()       = in.getDataSet(prelabel + "V"      ).read<ZMatrix>();
        auto a = in.getDataSet(prelabel + "S").read<std::vector<double>>();
        // HM-C11 per block: 0 <= epsrank <= min(rows built, cols), U is rows x epsrank, V is cols x epsrank
        HNESSI_REQUIRE(0 <= epsrank && epsrank <= std::min(blklen_[b], B.cols()) && B.U().rows() == blklen_[b] &&
                       B.U().cols() == epsrank && B.V().rows() == B.cols() && B.V().cols() == epsrank &&
                       (long long)a.size() >= epsrank, where, HNESSI_KV(prelabel), HNESSI_KV(epsrank),
                       HNESSI_KV(blklen_[b]), HNESSI_KV(B.cols()), HNESSI_KV(B.U().rows()), HNESSI_KV(B.U().cols()),
                       HNESSI_KV(B.V().rows()), HNESSI_KV(B.V().cols()), HNESSI_KV(a.size()));
        B.S() = Eigen::Map<Eigen::VectorXd>(a.data(), epsrank);
      }
    }
  }

  // Lesser diagonal data
  les_dir_square_ = in.getDataSet(h5_join({label, "les_dir_square"})).read<std::vector<cplx>>();
  require_dataset_size(where, "les_dir_square", les_dir_square_.size(), len_les_dir);

  // Lesser data on left edge
  les_left_edge_ = in.getDataSet(h5_join({label, "les_left_edge"})).read<std::vector<cplx>>();
  require_dataset_size(where, "les_left_edge", les_left_edge_.size(), len_edge);

  // TV data
  // (format_version <= 2 files also hold the orbital-transposed copy "tv_trans"; it is not read: WP6c, ISSUES H27)
  tv_.data_vector() = in.getDataSet(h5_join({label, "tv"})).read<std::vector<cplx>>();
  require_dataset_size(where, "tv", tv_.data_vector().size(), (long long)nt_ * r_ * es);

  // timing
  try {
    timing = in.getDataSet(h5_join({label, "timing"})).read<DMatrix>();
  }
  catch (const HighFive::DataSetException& e) {
    timing = DMatrix::Zero(nt_, 22);   // BUGS B9: was uninitialised
  }
  HNESSI_REQUIRE(timing.rows() == nt_ && timing.cols() == 22, where, HNESSI_KV(timing.rows()), HNESSI_KV(timing.cols()),
                 HNESSI_KV(nt_));
}

void herm_matrix_hodlr::write_checkpoint_hdf5(h5e::File &out, std::string label) {
  // checkpoint format version (2: GMConvTensMatrix built with the xi factor, bosonic sign fix; 3: no tv_trans)
  h5e::dump(out, h5_join({label, "format_version"}), checkpoint_format_version);

  // scalar parameters
  h5e::dump(out, h5_join({label, "nt"}), nt_);
  h5e::dump(out, h5_join({label, "r"}), r_);
  h5e::dump(out, h5_join({label, "nlvl"}), nlvl_);
  h5e::dump(out, h5_join({label, "svdtol"}), svdtol_);
  h5e::dump(out, h5_join({label, "size1"}), size1_);
  h5e::dump(out, h5_join({label, "size2"}), size2_);
  h5e::dump(out, h5_join({label, "sig"}), sig_);
  h5e::dump(out, h5_join({label, "k"}), k_);
  h5e::dump(out, h5_join({label, "tstpmk"}), tstpmk_);
  h5e::dump(out, h5_join({label, "built_blocks"}), built_blocks_);
  h5e::dump(out, h5_join({label, "can_extrap"}), can_extrap_);

  // Matsubara data
  h5e::dump(out, h5_join({label, "mat"}), DMatrixMap(mat_.data(), r_, size1_*size2_));
  h5e::dump(out, h5_join({label, "GMConvTensMatrix"}), GMConvTensMatrix_);
  h5e::dump(out, h5_join({label, "gmconv_valid"}), int(gmconv_current() ? 1 : 0));

  // Current timestep data
  h5e::dump(out, h5_join({label, "curr_timestep_ret"}), curr_timestep_ret_);
  h5e::dump(out, h5_join({label, "curr_timestep_les"}), curr_timestep_les_);

  // Number of rows each block has built
  h5e::dump(out, h5_join({label, "blklen"}), blklen_);

  // Retarded block data
  for(int b = 0; b < built_blocks_; b++) {
    for(int i = 0; i < size1_; i++) {
      for(int j = 0; j < size2_; j++) {
        std::string prelabel = h5_join({label, "ret", std::to_string(b)+"_"+std::to_string(i)+"_"+std::to_string(j)}) + "/";
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
  h5e::dump(out, h5_join({label, "ret", "dirtricol"}), ZMatrixMap(ret_.dirtricol(), ndir_*size1_*size2_, 1));

  // Retarded data k below diagonal
  h5e::dump(out, h5_join({label, "ret", "ret_corr_below_tri"}), ret_corr_below_tri_);

  // Retarded data on left edge
  h5e::dump(out, h5_join({label, "ret_left_edge"}), ret_left_edge_);

  // Les block data
  for(int b = 0; b < built_blocks_; b++) {
    for(int i = 0; i < size1_; i++) {
      for(int j = 0; j < size2_; j++) {
        std::string prelabel = h5_join({label, "les", std::to_string(b)+"_"+std::to_string(i)+"_"+std::to_string(j)}) + "/";
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
  h5e::dump(out, h5_join({label, "les_dir_square"}), les_dir_square_);

  // Lesser data on left edge
  h5e::dump(out, h5_join({label, "les_left_edge"}), les_left_edge_);

  // TV data
  h5e::dump(out, h5_join({label, "tv"}), tv_.data_vector());

  // timing
  h5e::dump(out, h5_join({label, "timing"}), timing);
}

void herm_matrix_hodlr::write_to_hdf5(h5e::File &out, std::string label, std::string geometry_label) {
  // HM-W2 (Q-8: warning): rows >= tstpmk are still in the circular buffers and are not written by this function
  if(tstpmk_ != nt_) {
    detail::warn("h_nessi::herm_matrix_hodlr::write_to_hdf5('" + label + "'): tstpmk=" + std::to_string(tstpmk_) +
                 " < nt=" + std::to_string(nt_) + ": the final k+1 update_blocks calls have not been made, rows >= " +
                 std::to_string(tstpmk_) + " are missing from the compressed output");
  }
  // HM-W3 (tier 2): an existing geometry group must describe this geometry (the readers use it for every object
  // written with this geometry_label)
  if (out.exist(h5_join({"geometry", geometry_label, "blkr1"}))) {
    std::vector<int> b1 = out.getDataSet(h5_join({"geometry", geometry_label, "blkr1"})).read<std::vector<int>>();
    HNESSI_REQUIRE(b1 == blkr1_, "herm_matrix_hodlr::write_to_hdf5", HNESSI_KV(geometry_label), HNESSI_KV(b1.size()),
                   HNESSI_KV(nbox_), detail::note{"geometry/<geometry_label> already holds a different geometry (other "
                                                  "nt or nlvl): pass a different geometry_label"});
  }
  h5e::dump(out, h5_join({label, "nt"}), nt_);
  h5e::dump(out, h5_join({label, "r"}), r_);
  h5e::dump(out, h5_join({label, "size1"}), size1_);
  h5e::dump(out, h5_join({label, "size2"}), size2_);
  h5e::dump(out, h5_join({label, "ndir"}), ndir_);
  h5e::dump(out, h5_join({label, "nbox"}), nbox_);
  h5e::dump(out, h5_join({label, "nlvl"}), nlvl_);
  h5e::dump(out, h5_join({label, "maxdir"}), maxdir_);
  h5e::dump(out, h5_join({label, "k"}), k_);
  h5e::dump(out, h5_join({label, "tstpmk"}), tstpmk_);
  h5e::dump(out, h5_join({label, "built_blocks"}), built_blocks_);

  if (!out.exist(  h5_join({"geometry", geometry_label, "blkr1"}))) {
    h5e::dump(out, h5_join({"geometry", geometry_label, "blkr1"}), blkr1_,h5e::DumpMode::Overwrite);
    h5e::dump(out, h5_join({"geometry", geometry_label, "blkr2"}), blkr2_,h5e::DumpMode::Overwrite);
    h5e::dump(out, h5_join({"geometry", geometry_label, "blkc1"}), blkc1_,h5e::DumpMode::Overwrite);
    h5e::dump(out, h5_join({"geometry", geometry_label, "blkc2"}), blkc2_,h5e::DumpMode::Overwrite);
    h5e::dump(out, h5_join({"geometry", geometry_label, "blklevel"}), blklevel_,h5e::DumpMode::Overwrite);
    h5e::dump(out, h5_join({"geometry", geometry_label, "blkdirheight"}), blkdirheight_,h5e::DumpMode::Overwrite);
    h5e::dump(out, h5_join({"geometry", geometry_label, "blkndirstart"}), blkndirstart_,h5e::DumpMode::Overwrite);
    h5e::dump(out, h5_join({"geometry", geometry_label, "c1_dir"}), c1_dir_,h5e::DumpMode::Overwrite);
    h5e::dump(out, h5_join({"geometry", geometry_label, "r2_dir"}), r2_dir_,h5e::DumpMode::Overwrite);
    h5e::dump(out, h5_join({"geometry", geometry_label, "ntri"}), ntri_,h5e::DumpMode::Overwrite);
    h5e::dump(out, h5_join({"geometry", geometry_label, "blklen"}), blklen_,h5e::DumpMode::Overwrite);
  }

  write_mat_to_hdf5(out, label);

  for(int b = 0; b < nbox_; b++) {
    for(int i = 0; i < size1_; i++) {
      for(int j = 0; j < size2_; j++) {
        std::string prelabel = h5_join({label, "ret", std::to_string(b)+"_"+std::to_string(i)+"_"+std::to_string(j)}) + "/";
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
        std::string prelabel = h5_join({label, "les", std::to_string(b)+"_"+std::to_string(i)+"_"+std::to_string(j)}) + "/";
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
  auto dset_les = out.createDataSet<cplx>(h5_join({label, "les", "dir"}), HighFive::DataSpace(les_dir_shape));
  dset_les.write_raw(les_dir_square_.data());

  // ret_dir is (z,i,j), where z starts at diagonal and goes downwards
  std::array<size_t, 3> ret_dir_shape = {
    static_cast<size_t>(ndir_),
    static_cast<size_t>(size1_),
    static_cast<size_t>(size2_)
  };
  auto dset_ret = out.createDataSet<cplx>(h5_join({label, "ret", "dirtricol"}), HighFive::DataSpace(ret_dir_shape));
  dset_ret.write_raw(ret_.dirtricol());

  // TV data
  std::array<size_t, 4> shape = {
    static_cast<size_t>(nt_),
    static_cast<size_t>(r_),
    static_cast<size_t>(size1_),
    static_cast<size_t>(size2_)
  };
  auto dset = out.createDataSet<cplx>(h5_join({label, "tv"}), HighFive::DataSpace(shape));
  dset.write_raw(tv_.data_vector().data()); 

  // timing
  h5e::dump(out, h5_join({label, "timing"}), timing);
}

// HM-W4, HM-W5 (tier 2): writers that read every row 0..nt-1
#define HNESSI_REQUIRE_ALL_ROWS_(where)                                                                             \
  HNESSI_REQUIRE(tstpmk_ + k_ >= nt_ - 1, where, HNESSI_KV(tstpmk_), HNESSI_KV(k_), HNESSI_KV(nt_),                 \
                 detail::note{"all rows 0..nt-1 must be computed (tstp() >= nt-1)"})

void herm_matrix_hodlr::write_rho_to_hdf5(h5e::File &out, std::string label, dlr_info &dlr) {
  HNESSI_REQUIRE_ALL_ROWS_("herm_matrix_hodlr::write_rho_to_hdf5");
  check_dlr_("herm_matrix_hodlr::write_rho_to_hdf5", dlr);
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
        ranksR(b*size1_*size2_ + j*size2_ + k) = ret_.data().blocks()[i][j][k].epsrank();
        ranksL(b*size1_*size2_ + j*size2_ + k) = les_.data().blocks()[i][j][k].epsrank();
      }
    }
  }

  std::array<size_t, 3> shape = {
    static_cast<size_t>(nlvl_),
    static_cast<size_t>(size1_),
    static_cast<size_t>(size2_)
  };
  auto dset = out.createDataSet<int>(h5_join({label, "ranksR"}), HighFive::DataSpace(shape));
  dset.write_raw(ranksR.data());
  dset = out.createDataSet<int>(h5_join({label, "ranksL"}), HighFive::DataSpace(shape));
  dset.write_raw(ranksL.data());
}

void herm_matrix_hodlr::write_mat_to_hdf5(h5e::File &out, std::string label) {
  std::array<size_t, 3> shape = {
    static_cast<size_t>(r_),
    static_cast<size_t>(size1_),
    static_cast<size_t>(size2_)
  };
  auto dset = out.createDataSet<double>(h5_join({label, "mat"}), HighFive::DataSpace(shape));
  dset.write_raw(mat_.data());
}

void herm_matrix_hodlr::write_curr_to_hdf5(h5e::File &out, std::string label) {
  h5e::dump(out, h5_join({label, "curr_timestep_ret"}), curr_timestep_ret_);
  h5e::dump(out, h5_join({label, "curr_timestep_les"}), curr_timestep_les_);
}

void herm_matrix_hodlr::write_GR0_to_hdf5(h5e::File &out, std::string label) {
    HNESSI_REQUIRE_ALL_ROWS_("herm_matrix_hodlr::write_GR0_to_hdf5");
    ZMatrix GR_t(nt_*size1_*size1_, 1);
    for(int i = 0; i < nt_; i++) {
      get_ret(i, 0, GR_t.data() + i*size1_*size1_);
    }
    
  std::array<size_t, 3> shape = {
    static_cast<size_t>(nt_),
    static_cast<size_t>(size1_),
    static_cast<size_t>(size2_)
  };
  auto dset = out.createDataSet<cplx>(h5_join({label, "R0"}), HighFive::DataSpace(shape));
  dset.write_raw(GR_t.data());
}

void herm_matrix_hodlr::write_GL0_to_hdf5(h5e::File &out, std::string label) {
    HNESSI_REQUIRE_ALL_ROWS_("herm_matrix_hodlr::write_GL0_to_hdf5");
    ZMatrix GL_t(nt_*size1_*size1_, 1);
    for(int i = 0; i < nt_; i++) {
      get_les(0, i, GL_t.data() + i*size1_*size1_);
    }
    
  std::array<size_t, 3> shape = {
    static_cast<size_t>(nt_),
    static_cast<size_t>(size1_),
    static_cast<size_t>(size2_)
  };
  auto dset = out.createDataSet<cplx>(h5_join({label, "L0"}), HighFive::DataSpace(shape));
  dset.write_raw(GL_t.data());
}
} // namespace
