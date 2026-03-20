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
  nt_ = in.getDataSet(label + std::string("/nt")).read<int>();
  size1_ = in.getDataSet(label + std::string("/size1")).read<int>();
  size2_ = in.getDataSet(label + std::string("/size2")).read<int>();
  element_size_ = in.getDataSet(label + std::string("/element_size")).read<int>();
  data_.resize((nt_+1)*size1_*size2_);
  ZMatrixMap(data_.data(), (nt_+1)*element_size_, 1) = in.getDataSet(label + std::string("/data")).read<ZMatrix>();
}


void function::write_checkpoint_hdf5(h5e::File &out, std::string label) {
  h5e::dump(out, label + std::string("/nt"), nt_);
  h5e::dump(out, label + std::string("/size1"), size1_);
  h5e::dump(out, label + std::string("/size2"), size2_);
  h5e::dump(out, label + std::string("/element_size"), element_size_);
  h5e::dump(out, label + std::string("/data"), ZMatrixMap(data_.data(), (nt_+1)*element_size_, 1));
}

void function::write_to_hdf5(h5e::File &out, std::string label) {
  std::array<size_t, 3> shape = {
    static_cast<size_t>(nt_+1),
    static_cast<size_t>(size1_),
    static_cast<size_t>(size2_)
  };
  auto dset = out.createDataSet<cplx>(label + std::string("/data"), HighFive::DataSpace(shape));
  dset.write_raw(data_.data());
}

void function::set_value(int t, int i, int j, cplx v) {
  ptr(t)[i*size1_ + j] = v;
}

void function::set_value(int t, ZMatrix &M) {
  ZMatrixMap(ptr(t), size1_, size2_) = M;
}

void function::get_value(int t, ZMatrix &M) {
  M = ZMatrixMap(ptr(t), size1_, size2_);
}

void function::set_constant(ZMatrix &M) {
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
  return ptr(t)[i*size1_ + j];
}
