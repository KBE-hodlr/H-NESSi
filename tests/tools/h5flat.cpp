// h5flat: flatten numeric HDF5 datasets to text (used by the example tests and agent_scratch/baseline/compare.py).
//   h5flat dump FILE                 -> for every numeric dataset: "DATASET <path> <ncomp> <n> <dims>" then n*ncomp values
//                                       (%.17g, one per line; complex = compound of 2 floats -> re, im interleaved)
//   h5flat cmp A B TOL DS [DS ...]   -> exit 0 iff every listed dataset exists in both with equal shape and
//                                       max|a-b| <= TOL*max(1,max|a|); prints max abs/rel differences.
#include <hdf5.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct DS { int ncomp = 1; std::vector<double> v; std::vector<hsize_t> dims; };

static bool read_ds(hid_t file, const char *path, DS &out) {
  hid_t d = H5Dopen2(file, path, H5P_DEFAULT);
  if (d < 0) return false;
  hid_t t = H5Dget_type(d), s = H5Dget_space(d);
  hssize_t n = H5Sget_simple_extent_npoints(s);
  int nd = H5Sget_simple_extent_ndims(s);
  out.dims.assign(nd > 0 ? nd : 0, 0);
  if (nd > 0) H5Sget_simple_extent_dims(s, out.dims.data(), nullptr);
  H5T_class_t c = H5Tget_class(t);
  bool ok = false;
  if (c == H5T_FLOAT || c == H5T_INTEGER) {
    out.ncomp = 1;
    out.v.resize(n);
    ok = H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.v.data()) >= 0;
  } else if (c == H5T_COMPOUND && H5Tget_nmembers(t) == 2 && H5Tget_member_class(t, 0) == H5T_FLOAT &&
             H5Tget_member_class(t, 1) == H5T_FLOAT) {
    char *n0 = H5Tget_member_name(t, 0), *n1 = H5Tget_member_name(t, 1);
    hid_t m = H5Tcreate(H5T_COMPOUND, 2 * sizeof(double));
    H5Tinsert(m, n0, 0, H5T_NATIVE_DOUBLE);
    H5Tinsert(m, n1, sizeof(double), H5T_NATIVE_DOUBLE);
    out.ncomp = 2;
    out.v.resize(2 * n);
    ok = H5Dread(d, m, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.v.data()) >= 0;
    H5Tclose(m);
    H5free_memory(n0);
    H5free_memory(n1);
  }
  H5Sclose(s);
  H5Tclose(t);
  H5Dclose(d);
  return ok;
}

static herr_t visit(hid_t obj, const char *name, const H5O_info2_t *info, void *op) {
  if (info->type == H5O_TYPE_DATASET) ((std::vector<std::string> *)op)->push_back(std::string("/") + name);
  (void)obj;
  return 0;
}

int main(int argc, char **argv) {
  H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
  if (argc >= 3 && !strcmp(argv[1], "dump")) {
    hid_t f = H5Fopen(argv[2], H5F_ACC_RDONLY, H5P_DEFAULT);
    if (f < 0) { fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }
    std::vector<std::string> names;
    H5Ovisit3(f, H5_INDEX_NAME, H5_ITER_INC, visit, &names, H5O_INFO_BASIC);
    for (auto &p : names) {
      DS d;
      if (!read_ds(f, p.c_str(), d)) { printf("SKIP %s\n", p.c_str()); continue; }
      std::string dims;
      for (size_t q = 0; q < d.dims.size(); q++) dims += (q ? "x" : "") + std::to_string(d.dims[q]);
      if (dims.empty()) dims = "scalar";
      printf("DATASET %s %d %zu %s\n", p.c_str(), d.ncomp, d.v.size() / d.ncomp, dims.c_str());
      for (double x : d.v) printf("%.17g\n", x);
    }
    H5Fclose(f);
    return 0;
  }
  if (argc >= 6 && !strcmp(argv[1], "cmp")) {
    hid_t fa = H5Fopen(argv[2], H5F_ACC_RDONLY, H5P_DEFAULT), fb = H5Fopen(argv[3], H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fa < 0 || fb < 0) { fprintf(stderr, "cannot open input files\n"); return 2; }
    double tol = atof(argv[4]);
    int bad = 0;
    for (int i = 5; i < argc; i++) {
      DS a, b;
      if (!read_ds(fa, argv[i], a) || !read_ds(fb, argv[i], b) || a.v.size() != b.v.size()) {
        printf("%s: missing or shape mismatch\n", argv[i]);
        bad++;
        continue;
      }
      double ma = 0, mabs = 0;
      for (size_t j = 0; j < a.v.size(); j++) { ma = std::max(ma, std::fabs(a.v[j])); mabs = std::max(mabs, std::fabs(a.v[j] - b.v[j])); }
      bool ok = mabs <= tol * std::max(1.0, ma);
      printf("%s: max_abs=%.3e max_rel=%.3e %s\n", argv[i], mabs, ma > 0 ? mabs / ma : 0., ok ? "OK" : "DIFFER");
      if (!ok) bad++;
    }
    return bad ? 1 : 0;
  }
  fprintf(stderr, "usage: h5flat dump FILE | h5flat cmp A B TOL DS...\n");
  return 2;
}
