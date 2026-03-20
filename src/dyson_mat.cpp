#include "h_nessi/dyson.hpp"
#include <iomanip>

using namespace std::chrono;

namespace h_nessi {

double dyson::dyson_mat(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, bool fixHam, double alpha){
    assert(xi_==G.sig() && nao_==G.size1() && nao_==G.size2());
    double sig = double(xi_);

    if(!fixHam) {
      dyson::green_from_H_mat(DNTauTmp2_.data(),mu,H.get_map(-1));
      DMatrixMap(DNTauTmp_.data(), es_, ntau_).noalias() = DMatrixMap(DNTauTmp2_.data(), ntau_, es_).transpose();
      c_dlr_convmat(&r_,&nao_,dlr_.it2cf(),dlr_.it2cfp(),dlr_.phi(),DNTauTmp_.data(),G.GMConvTens());
    }

    double *tmp = reinterpret_cast<double*>(Sigma.tvptr(0,0));
    DMatrixMap(tmp, es_, ntau_).noalias() = DMatrixMap(Sigma.mat(), ntau_, es_).transpose();

    c_dyson_it(&r_, &nao_, dlr_.it2cf(), dlr_.it2cfp(), dlr_.phi(), DNTauTmp_.data(), G.GMConvTens(), tmp, DNTauTmp2_.data());
    DMatrixMap(DNTauTmp_.data(), ntau_, es_).noalias() = DMatrixMap(DNTauTmp2_.data(), es_, ntau_).transpose();

    double ret = (DMatrixMap(DNTauTmp_.data(), 1, r_*es_) - DMatrixMap(G.mat(), 1, r_*es_)).norm();

    DMatrixMap(G.mat(), 1, r_*es_) = (1-alpha) * DMatrixMap(G.mat(), 1, r_*es_) + alpha * DMatrixMap(DNTauTmp_.data(), 1, r_*es_);

    return ret;
}


double dyson::dyson_mat(herm_matrix_hodlr &G, double mu, DMatrix &H, herm_matrix_hodlr &Sigma, bool fixHam, double alpha){
    assert(xi_==G.sig() && nao_==G.size1() && nao_==G.size2());
    double sig = double(xi_);

    if(!fixHam) {
      dyson::green_from_H_mat(DNTauTmp2_.data(),mu,H);
      DMatrixMap(DNTauTmp_.data(), es_, ntau_).noalias() = DMatrixMap(DNTauTmp2_.data(), ntau_, es_).transpose();
      c_dlr_convmat(&r_,&nao_,dlr_.it2cf(),dlr_.it2cfp(),dlr_.phi(),DNTauTmp_.data(),G.GMConvTens());
    }

    double *tmp = reinterpret_cast<double*>(Sigma.tvptr(0,0));
    DMatrixMap(tmp, es_, ntau_).noalias() = DMatrixMap(Sigma.mat(), ntau_, es_).transpose();

    c_dyson_it(&r_, &nao_, dlr_.it2cf(), dlr_.it2cfp(), dlr_.phi(), DNTauTmp_.data(), G.GMConvTens(), tmp, DNTauTmp2_.data());
    DMatrixMap(DNTauTmp_.data(), ntau_, es_).noalias() = DMatrixMap(DNTauTmp2_.data(), es_, ntau_).transpose();

    double ret = (DMatrixMap(DNTauTmp_.data(), 1, r_*es_) - DMatrixMap(G.mat(), 1, r_*es_)).norm();
    DMatrixMap(G.mat(), 1, r_*es_) = (1-alpha) * DMatrixMap(G.mat(), 1, r_*es_) + alpha * DMatrixMap(DNTauTmp_.data(), 1, r_*es_);

    return ret;
}

} // namespace
