#include "h_nessi/function.hpp"
#include <algorithm>

using namespace h_nessi;

function::function() {
  data_.resize(1);
  nt_ = 0;
  size1_ = 0;
  size2_ = 0;
  element_size_ = 0;
}

function::function(int nt, int size1, int size2) {
  // FN-1 (tier 2): the element index i*size1+j assumes square matrices (ISSUES D2)
  HNESSI_REQUIRE(nt >= 0 && size1 >= 1 && size1 == size2, "function::function", HNESSI_KV(nt), HNESSI_KV(size1),
                 HNESSI_KV(size2), detail::note{"nt >= 0 and square orbital matrices (size1 == size2 >= 1)"});
  nt_ = nt;
  size1_ = size1;
  size2_ = size2;
  element_size_ = size1*size2;
  data_.resize((nt+1)*element_size_);
  std::fill(data_.begin(), data_.end(), cplx(0.,0.));
}

function &function::operator=(const function &g) {
    int len;
    if (this == &g)
        return *this;
    nt_ = g.nt_;
    size1_ = g.size1_;
    size2_ = g.size2_;
    element_size_ = size1_ * size2_;
    len = (nt_ + 1) * size1_ * size2_;
    if (len > 0) {
        data_.assign(g.data_.begin(), g.data_.end());
    } else {
        data_.clear();
    }
    return *this;
}

function::function(const function &g) {
    nt_ = g.nt_;
    size1_ = g.size1_;
    size2_ = g.size2_;
    element_size_ = size1_ * size2_;
    data_ = g.data_;
}

function::function(function &&g) noexcept 
    : data_(std::move(g.data_)), 
      nt_(g.nt_), 
      size1_(g.size1_), 
      size2_(g.size2_), 
      element_size_(g.element_size_) { 
    g.nt_ = -2; 
    g.size1_ = 0; 
    g.size2_ = 0; 
    g.element_size_ = 0; 
}


function::function(h5e::File &in, std::string label) {
  nt_ = in.getDataSet(h5_join({label, "nt"})).read<int>();
  size1_ = in.getDataSet(h5_join({label, "size1"})).read<int>();
  size2_ = in.getDataSet(h5_join({label, "size2"})).read<int>();
  element_size_ = in.getDataSet(h5_join({label, "element_size"})).read<int>();
  // FN-6 (tier 2): stored sizes consistent (FN-1) and the data length matches
  const char *where = "function::function(h5e::File&,label)";
  HNESSI_REQUIRE(nt_ >= 0 && size1_ >= 1 && size1_ == size2_ && element_size_ == size1_*size2_, where, HNESSI_KV(nt_),
                 HNESSI_KV(size1_), HNESSI_KV(size2_), HNESSI_KV(element_size_));
  ZMatrix d = in.getDataSet(h5_join({label, "data"})).read<ZMatrix>();
  HNESSI_REQUIRE(d.size() == (Eigen::Index)(nt_+1)*element_size_, where, HNESSI_KV(d.size()), HNESSI_KV(nt_),
                 HNESSI_KV(element_size_), detail::note{"dataset data must hold (nt+1)*size1*size2 values"});
  data_.resize((nt_+1)*size1_*size2_);
  ZMatrixMap(data_.data(), (nt_+1)*element_size_, 1) = ZMatrixMap(d.data(), (nt_+1)*element_size_, 1);
}


void function::write_checkpoint_hdf5(h5e::File &out, std::string label) {
  h5e::dump(out, h5_join({label, "nt"}), nt_);
  h5e::dump(out, h5_join({label, "size1"}), size1_);
  h5e::dump(out, h5_join({label, "size2"}), size2_);
  h5e::dump(out, h5_join({label, "element_size"}), element_size_);
  h5e::dump(out, h5_join({label, "data"}), ZMatrixMap(data_.data(), (nt_+1)*element_size_, 1));
}

void function::write_to_hdf5(h5e::File &out, std::string label) {
  std::array<size_t, 3> shape = {
    static_cast<size_t>(nt_+1),
    static_cast<size_t>(size1_),
    static_cast<size_t>(size2_)
  };
  auto dset = out.createDataSet<cplx>(h5_join({label, "data"}), HighFive::DataSpace(shape));
  dset.write_raw(data_.data());
}

void function::set_value(int t, int i, int j, cplx v) {
  (*this)(t, i, j) = v;
}

void function::set_value(int t, ZMatrix &M) {
  HNESSI_ASSERT(M.rows() == size1_ && M.cols() == size2_, "function::set_value", HNESSI_KV(M.rows()), HNESSI_KV(M.cols()),
                HNESSI_KV(size1_));   // FN-4 (tier 1)
  ZMatrixMap(ptr(t), size1_, size2_) = M;
}

void function::get_value(int t, ZMatrix &M) {
  M = ZMatrixMap(ptr(t), size1_, size2_);
}

void function::set_constant(ZMatrix &M) {
  HNESSI_REQUIRE(M.rows() == size1_ && M.cols() == size2_, "function::set_constant", HNESSI_KV(M.rows()),
                 HNESSI_KV(M.cols()), HNESSI_KV(size1_));   // FN-5 (tier 2)
  for(int t = -1; t < nt_; t++) {
    ZMatrixMap(ptr(t), size1_, size2_) = M;
  }
}

void function::set_constant(cplx A) {
  for(int t = -1; t < nt_; t++) {
    ZMatrixMap(ptr(t), size1_, size2_) = A*ZMatrix::Identity(size1_,size2_);
  }
}

void function::set_constant(double A) {
  for(int t = -1; t < nt_; t++) {
    ZMatrixMap(ptr(t), size1_, size2_) = A*ZMatrix::Identity(size1_,size2_);
  }
}

void function::set_zero(void) {
  std::fill(data_.begin(), data_.end(), cplx(0.,0.));
}

cplx &function::operator()(int t, int i, int j) {
  HNESSI_ASSERT(0 <= i && i < size1_ && 0 <= j && j < size2_, "function::operator()", HNESSI_KV(i), HNESSI_KV(j),
                HNESSI_KV(size1_));   // FN-3 (tier 1)
  return ptr(t)[i*size1_ + j];
}
