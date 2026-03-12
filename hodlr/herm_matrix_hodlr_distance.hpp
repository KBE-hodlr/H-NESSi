double distance_norm2_curr_les(int tstp, herm_matrix_hodlr &g1, herm_matrix_hodlr &g2) {
  assert(g1.size1() == g2.size1());
  assert(tstp >= g1.tstpmk() && tstp >= g2.tstpmk());
  assert(tstp <= (g1.tstpmk()+g1.k()) && tstp <= (g2.tstpmk()+g2.k()));

  int size1 = g1.size1(), size2 = g1.size2(), es = size1*size2;

  return (ZMatrixMap(g1.curr_timestep_les_ptr(0, tstp), (tstp+1)*es, 1) - ZMatrixMap(g2.curr_timestep_les_ptr(0,tstp), (tstp+1)*es, 1)).norm();
}

double distance_norm2_curr_ret(int tstp, herm_matrix_hodlr &g1, herm_matrix_hodlr &g2) {
  assert(g1.size1() == g2.size1());
  assert(tstp >= g1.tstpmk() && tstp >= g2.tstpmk());
  assert(tstp <= (g1.tstpmk()+g1.k()) && tstp <= (g2.tstpmk()+g2.k()));

  int size1 = g1.size1(), size2 = g1.size2(), es = size1*size2;

  return (ZMatrixMap(g1.curr_timestep_ret_ptr(tstp,0), (tstp+1)*es, 1) - ZMatrixMap(g2.curr_timestep_ret_ptr(tstp,0), (tstp+1)*es, 1)).norm();
}

double distance_norm2_mat(herm_matrix_hodlr &g1, herm_matrix_hodlr &g2) {
  assert(g1.size1() == g2.size1());
  assert(g1.r() == g2.r());

  int size1 = g1.size1(), size2 = g1.size2(), r = g1.r(), es = size1*size2;

  return (DMatrixMap(g1.matptr(0), r*es, 1) - DMatrixMap(g2.matptr(0), r*es, 1)).norm();
}

double distance_norm2_tv(int tstp, herm_matrix_hodlr &g1, herm_matrix_hodlr &g2) {
  assert(g1.size1() == g2.size1());
  assert(g1.r() == g2.r());

  int size1 = g1.size1(), size2 = g1.size2(), r = g1.r(), es = size1*size2;

  return (ZMatrixMap(g1.tvptr(tstp,0), r*es, 1) - ZMatrixMap(g2.tvptr(tstp,0), r*es, 1)).norm();
}

double distance_norm2_curr(int tstp, herm_matrix_hodlr &g1, herm_matrix_hodlr &g2) {
  if(tstp == -1) return distance_norm2_mat(g1,g2);
  else {
    double err_R, err_L, err_TV;
    err_R = std::pow(distance_norm2_curr_ret(tstp, g1, g2),2);
    err_L = std::pow(distance_norm2_curr_les(tstp, g1, g2),2);
    err_TV = std::pow(distance_norm2_tv(tstp, g1, g2),2);

    return std::sqrt(err_R + err_L + err_TV);
  }
}


#ifdef INCLUDE_NESSI
double distance_norm2_curr(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix &g2, dlr_info &dlr) {
  if(tstp == -1) return distance_norm2_mat(g1,g2,dlr);
  else {
    double err_R, err_L, err_TV;
    err_R = std::pow(distance_norm2_curr_ret(tstp, g1, g2),2);
    err_L = std::pow(distance_norm2_curr_les(tstp, g1, g2),2);
    err_TV = std::pow(distance_norm2_tv(tstp, g1, g2, dlr),2);

    return std::sqrt(err_R + err_L + err_TV);
  }
}

double distance_norm2_curr_ret(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix &g2) {
  assert(g1.size1() == g2.size1());
  assert(tstp >= g1.tstpmk());
  assert(tstp <= (g1.tstpmk()+g1.k()));

  int size1 = g1.size1(), size2 = g1.size2(), es = size1*size2;

  return (ZMatrixMap(g1.curr_timestep_ret_ptr(tstp,0), (tstp+1)*es, 1) - ZMatrixMap(g2.retptr(tstp,0), (tstp+1)*es, 1)).norm();
}

double distance_norm2_curr_les(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix &g2) {
  assert(g1.size1() == g2.size1());
  assert(tstp >= g1.tstpmk());
  assert(tstp <= (g1.tstpmk()+g1.k()));

  int size1 = g1.size1(), size2 = g1.size2(), es = size1*size2;

  return (ZMatrixMap(g1.curr_timestep_les_ptr(0,tstp), (tstp+1)*es, 1) - ZMatrixMap(g2.retptr(0,tstp), (tstp+1)*es, 1)).norm();
}

double distance_norm2_mat(herm_matrix_hodlr &g1, cntr::herm_matrix &g2, dlr_info &dlr) {
  int ntau = g2.ntau();

  // first get nessi equidistant grid
  double beta = dlr.beta();
  DColVector nessi_pts(ntau+1);
  for(int i = 0; i <= ntau; i++) {
    nessi_pts(i) = i*beta/ntau;
  }

  // get mat evaluated on this grid
  DMatrix nessi_vals(ntau+1, es);
  g1.get_mat_tau_array(nessi_pts, dlr_info &dlr, nessi_vals);
  
  return (nessi_vals-ZMatrixMap(g2.matptr(0), ntau+1, es)).norm();
}

double distance_norm2_tv(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix &g2, dlr_info &dlr) {
  int ntau = g2.ntau();

  // first get nessi equidistant grid
  double beta = dlr.beta();
  DColVector nessi_pts(ntau+1);
  for(int i = 0; i <= ntau; i++) {
    nessi_pts(i) = i*beta/ntau;
  }

  // get mat evaluated on this grid
  ZMatrix nessi_vals(ntau+1, es);
  g1.get_tv_tau_array(tstp, nessi_pts, dlr, nessi_vals);
  
  return (nessi_vals-ZMatrixMap(g2.tvptr(tstp,0), ntau+1, es)).norm();
}

#endif


