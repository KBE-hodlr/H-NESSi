void herm_matrix_hodlr::get_mat_tau(double tau, dlr_info &dlr, DMatrix &M){
  dlr.eval_point(tau, matptr(0), M.data());
}

void herm_matrix_hodlr::get_mat_tau(double tau, dlr_info &dlr, double *M){
  dlr.eval_point(tau, matptr(0), M);
}

void herm_matrix_hodlr::get_mat_tau(double tau, dlr_info &dlr, ZMatrix &M){
  dlr.eval_point(tau, matptr(0), M.data());
}

void herm_matrix_hodlr::get_mat_tau(double tau, dlr_info &dlr, cplx *M){
  dlr.eval_point(tau, matptr(0), M);
}

void herm_matrix_hodlr::get_tv_tau(int tstp, double tau, dlr_info &dlr, ZMatrix &M){
  dlr.eval_point(tau, tvptr(tstp, 0), M.data());
}

void herm_matrix_hodlr::get_tv_tau(int tstp, double tau, dlr_info &dlr, cplx *M){
  dlr.eval_point(tau, tvptr(tstp, 0), M);
}





void herm_matrix_hodlr::get_mat_tau_array(DColVector taus, dlr_info &dlr, DMatrix &M){
  dlr.eval_point(taus, matptr(0), M.data());
}

void herm_matrix_hodlr::get_mat_tau_array(DColVector taus, dlr_info &dlr, double *M){
  dlr.eval_point(taus, matptr(0), M);
}

void herm_matrix_hodlr::get_mat_tau_array(DColVector taus, dlr_info &dlr, ZMatrix &M){
  dlr.eval_point(taus, matptr(0), M.data());
}

void herm_matrix_hodlr::get_mat_tau_array(DColVector taus, dlr_info &dlr, cplx *M){
  dlr.eval_point(taus, matptr(0), M);
}

void herm_matrix_hodlr::get_tv_tau_array(int tstp, DColVector taus, dlr_info &dlr, ZMatrix &M){
  dlr.eval_point(taus, tvptr(tstp,0), M.data());
}

void herm_matrix_hodlr::get_tv_tau_array(int tstp, DColVector taus, dlr_info &dlr, cplx *M){
  dlr.eval_point(taus, tvptr(tstp,0), M);
}

