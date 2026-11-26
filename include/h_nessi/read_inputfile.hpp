/*
 * Adapted for inclusion from the 'nessi' Library by Thomas Blommel (2026).
 * Original repository: https://github.com/nessi-cntr/nessi
 * Licensed under the Mozilla Public License, v. 2.0.
*/

/*
 * Input-file reader (NESSi syntax, doc 30 §5).  Header-only: every non-template function is `inline`, so the header
 * can be included in any number of translation units (B19).  All errors are reported by throwing
 * std::runtime_error with the flag / token / file that caused them (no `int` or `const char*` throws, B19/LX-25).
 * The functions live in the global namespace (as in NESSi); this header does not inject `using namespace h_nessi`.
 */

#ifndef READ_INPUTFILE_H_
#define READ_INPUTFILE_H_

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////
// ROUTINES FOR INTERPRETING THE INPUT FILE (LIKE F0R LAYER)
// ... interpreting input file:
#define INPUT_LINELEN 1024

namespace h_nessi {
namespace input_detail {
// split line into words separated by characters in delim; delim characters in line are replaced by \0
inline int split_words(char *line, std::vector<char *> &words, const char *delim) {
  words.resize(0);
  char *save = nullptr;
  char *pch = strtok_r(line, delim, &save);
  while (pch != NULL) {
    words.push_back(pch);
    pch = strtok_r(NULL, delim, &save);
  }
  return (int)words.size();
}
inline std::runtime_error error(const std::string &where, const std::string &what) {
  return std::runtime_error("h_nessi::read_inputfile::" + where + ": " + what);
}
}  // namespace input_detail
}  // namespace h_nessi

// cut the line at the first '#'
inline void uncomment_line(char *line, int len) {
  (void)len;
  char *p = strchr(line, '#');
  if (p) *p = '\0';
}

// read n consecutive double numbers from file:
inline void read_vector_from_file(char *file, std::vector<double> &data, int n) {
  data.resize(n, 0);
  std::ifstream in(file, std::ios::in);
  if (!in.good()) throw h_nessi::input_detail::error("read_vector_from_file", std::string("cannot open file '") + file + "'");
  for (int j = 0; j < n; j++) {
    in >> data[j];
    if (in.fail())
      throw h_nessi::input_detail::error("read_vector_from_file", std::string("error reading entry ") + std::to_string(j) +
                                                                     " of " + std::to_string(n) + " from file '" + file + "'");
  }
}

// read first line in file containing the string flag
inline char *find_line_with_flag(char *file, const char *flag, char *line) {
  std::ifstream in(file, std::ios::in);
  if (!in.good()) throw h_nessi::input_detail::error("find_line_with_flag", std::string("cannot open file '") + file + "'");
  while (in.getline(line, INPUT_LINELEN)) {
    uncomment_line(line, INPUT_LINELEN);
    char *pch = strstr(line, flag);
    if (pch) return pch;
  }
  throw h_nessi::input_detail::error("find_line_with_flag",
                                     std::string("flag '") + flag + "' not found in file '" + file +
                                         "' (or a line is longer than " + std::to_string(INPUT_LINELEN) + " characters)");
}

// find line containing consecutive fields "flag" "i1",
// and split rest of the line in words[0] ... words[n2-1]
template <typename T>
int find_line_with_fields(char *file, char *line, std::vector<char *> &words, const char *flag, T i1) {
  std::ifstream in(file, std::ios::in);
  if (!in.good()) throw h_nessi::input_detail::error("find_line_with_fields", std::string("cannot open file '") + file + "'");
  std::vector<char *> words1;
  while (in.getline(line, INPUT_LINELEN)) {
    uncomment_line(line, INPUT_LINELEN);
    char *pch = strstr(line, flag);
    if (!pch) continue;
    int n = h_nessi::input_detail::split_words(pch, words1, " \t");
    if (n >= 2) {
      T i2;
      std::stringstream(words1[1]) >> i2;
      if (i2 == i1) {
        int n2 = n - 2;
        words.resize(n2);
        for (int l = 0; l < n2; l++) words[l] = words1[l + 2];
        return n2;
      }
    }
  }
  std::ostringstream s;
  s << "no line with the fields '" << flag << " " << i1 << "' in file '" << file << "'";
  throw h_nessi::input_detail::error("find_line_with_fields", s.str());
}

template <typename T1, typename T2>
int find_line_with_fields(char *file, char *line, std::vector<char *> &words, const char *flag, T1 i1, T2 j1) {
  std::ifstream in(file, std::ios::in);
  if (!in.good()) throw h_nessi::input_detail::error("find_line_with_fields", std::string("cannot open file '") + file + "'");
  std::vector<char *> words1;
  while (in.getline(line, INPUT_LINELEN)) {
    uncomment_line(line, INPUT_LINELEN);
    char *pch = strstr(line, flag);
    if (!pch) continue;
    int n = h_nessi::input_detail::split_words(pch, words1, " \t");
    if (n >= 3) {
      T1 i2;
      T2 j2;
      std::stringstream(words1[1]) >> i2;
      std::stringstream(words1[2]) >> j2;
      if (i2 == i1 && j2 == j1) {
        int n2 = n - 3;
        words.resize(n2);
        for (int l = 0; l < n2; l++) words[l] = words1[l + 3];
        return n2;
      }
    }
  }
  std::ostringstream s;
  s << "no line with the fields '" << flag << " " << i1 << " " << j1 << "' in file '" << file << "'";
  throw h_nessi::input_detail::error("find_line_with_fields", s.str());
}

namespace h_nessi {
namespace input_detail {
inline void require_words(const std::vector<char *> &words, int m, int count, const char *fn) {
  if (m < 0 || (int)words.size() < m + count)
    throw error(fn, "too few values in line (need " + std::to_string(count) + " starting at word " + std::to_string(m) +
                        ", found " + std::to_string(std::max(0, (int)words.size() - m)) + ")");
}
template <class T>
inline void parse_word(const char *word, T &x, const char *fn, const char *type) {
  std::stringstream s(word);
  if (!(s >> x)) throw error(fn, std::string("cannot interpret '") + word + "' as " + type);
  // RI-3: the whole token must be consumed ("1.5" is not an int, "2x" is not a number); trailing whitespace is fine
  s >> std::ws;
  if (!s.eof()) throw error(fn, std::string("cannot interpret '") + word + "' as " + type + " (trailing characters)");
}
// RI-4: counts of the vector readers
inline void require_dim(int dim, const char *fn) {
  if (dim < 0) throw error(fn, "dim = " + std::to_string(dim) + " < 0");
}
inline void require_nt(int nt, const char *fn) {
  if (nt < -1) throw error(fn, "nt = " + std::to_string(nt) + " < -1");
}
}  // namespace input_detail
}  // namespace h_nessi

// READ A PARAMETER FROM A SEQUENCE OF WORDS words[m],words[m+1],...
inline void read_param(std::vector<char *> &words, int m, double &data) {
  h_nessi::input_detail::require_words(words, m, 1, "read_param(double)");
  h_nessi::input_detail::parse_word(words[m], data, "read_param(double)", "double");
}
inline void read_param(std::vector<char *> &words, int m, int &data) {
  h_nessi::input_detail::require_words(words, m, 1, "read_param(int)");
  h_nessi::input_detail::parse_word(words[m], data, "read_param(int)", "int");
}
inline void read_param(std::vector<char *> &words, int m, bool &data) {
  h_nessi::input_detail::require_words(words, m, 1, "read_param(bool)");  // B15: an empty value used to segfault
  std::stringstream s(words[m]);
  if (!(s >> std::boolalpha >> data) || !(s >> std::ws).eof())  // RI-3: the whole token
    throw h_nessi::input_detail::error("read_param(bool)", std::string("cannot interpret '") + words[m] +
                                                               "' as bool (expected true or false)");
}
inline void read_param(std::vector<char *> &words, int m, h_nessi::cplx &data) {
  double real, imag;
  h_nessi::input_detail::require_words(words, m, 2, "read_param(cplx)");
  h_nessi::input_detail::parse_word(words[m], real, "read_param(cplx)", "double (real part)");
  h_nessi::input_detail::parse_word(words[m + 1], imag, "read_param(cplx)", "double (imaginary part)");
  data.real(real);
  data.imag(imag);
}
inline void read_param(std::vector<char *> &words, int m, h_nessi::DColVector &data, int dim) {
  h_nessi::input_detail::require_dim(dim, "read_param(DColVector)");
  data = h_nessi::DColVector(dim);
  h_nessi::input_detail::require_words(words, m, dim, "read_param(DColVector)");
  for (int i1 = 0; i1 < dim; i1++)
    h_nessi::input_detail::parse_word(words[m + i1], data(i1), "read_param(DColVector)", "double");
}
inline void read_param(std::vector<char *> &words, int m, std::vector<double> &data, int dim) {
  h_nessi::input_detail::require_dim(dim, "read_param(std::vector<double>)");
  h_nessi::input_detail::require_words(words, m, dim, "read_param(std::vector<double>)");
  if ((int)data.size() < dim) data.resize(dim);  // RI-2: was written without a size check
  for (int i1 = 0; i1 < dim; i1++)
    h_nessi::input_detail::parse_word(words[m + i1], data[i1], "read_param(std::vector<double>)", "double");
}

// find the first line containing "flag", and read the following number
template <typename T>
void find_param(char *file, const char *flag, T &x) {
  try {
    char line[INPUT_LINELEN], *pch;
    std::vector<char *> words;
    find_line_with_flag(file, flag, line);
    pch = strstr(line, flag);
    h_nessi::input_detail::split_words(pch + strlen(flag), words, " \t");
    read_param(words, 0, x);
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string(e.what()) + " [find_param flag '" + flag + "', file '" + file + "']");
  }
}

inline void find_param(char *file, const char *flag, h_nessi::DColVector &x, int dim) {
  try {
    char line[INPUT_LINELEN], *pch;
    std::vector<char *> words;
    find_line_with_flag(file, flag, line);
    pch = strstr(line, flag);
    h_nessi::input_detail::split_words(pch + strlen(flag), words, " \t");
    read_param(words, 0, x, dim);
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string(e.what()) + " [find_param flag '" + flag + "', file '" + file + "']");
  }
}

inline void find_param(char *file, const char *flag, std::vector<double> &x, int dim) {
  try {
    h_nessi::input_detail::require_dim(dim, "find_param(std::vector<double>)");   // RI-4
    x.resize(dim);
    char line[INPUT_LINELEN], *pch;
    std::vector<char *> words;
    find_line_with_flag(file, flag, line);
    pch = strstr(line, flag);
    h_nessi::input_detail::split_words(pch + strlen(flag), words, " \t");
    read_param(words, 0, x, dim);
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string(e.what()) + " [find_param flag '" + flag + "', file '" + file + "']");
  }
}

// READ a time-dependent function A SEQUENCE OF WORDS words[m],words[m+1],...
// if words[m] is of the form --FILENAME, read funtion from file
// otherwise, a time-independent value is read from words[m], words[m+1], ...
inline void read_param_tvector(std::vector<char *> &words, int m, std::vector<h_nessi::DColVector> &data, int dim, int nt) {
  char *pch;
  h_nessi::input_detail::require_dim(dim, "read_param_tvector(DColVector)");
  h_nessi::input_detail::require_nt(nt, "read_param_tvector(DColVector)");
  data.resize(nt + 2);
  for (int tstp = -1; tstp <= nt; tstp++) data[tstp + 1] = h_nessi::DColVector(dim);
  h_nessi::input_detail::require_words(words, m, 1, "read_param_tvector(DColVector)");
  // test whether the first word is a file
  if ((pch = strstr(words[m], "--"))) {
    std::vector<double> tmp;
    read_vector_from_file(pch + 2, tmp, (nt + 2) * dim);  // read as many numbers
    for (int tstp = -1; tstp <= nt; tstp++)
      for (int i1 = 0; i1 < dim; i1++) data[tstp + 1](i1) = tmp[(tstp + 1) * dim + i1];
  } else {
    h_nessi::DColVector tmp(dim);
    read_param(words, m, tmp, dim);
    for (int tstp = -1; tstp <= nt; tstp++) data[tstp + 1] = tmp;
  }
}
inline void read_param_tvector(std::vector<char *> &words, int m, std::vector<h_nessi::cplx> &data, int nt) {
  char *pch;
  h_nessi::input_detail::require_nt(nt, "read_param_tvector(cplx)");
  data.resize(nt + 2);
  h_nessi::input_detail::require_words(words, m, 1, "read_param_tvector(cplx)");
  // test whether the first word is a file
  if ((pch = strstr(words[m], "--"))) {
    std::vector<double> tmp;
    read_vector_from_file(pch + 2, tmp, (nt + 2) * 2);  // read as many numbers
    for (int tstp = -1; tstp <= nt; tstp++) data[tstp + 1] = h_nessi::cplx(tmp[(tstp + 1) * 2], tmp[(tstp + 1) * 2 + 1]);
  } else {
    h_nessi::cplx tmp;
    read_param(words, m, tmp);
    for (int tstp = -1; tstp <= nt; tstp++) data[tstp + 1] = tmp;
  }
}
inline void read_param_tvector(std::vector<char *> &words, int m, std::vector<double> &data, int nt) {
  char *pch;
  h_nessi::input_detail::require_nt(nt, "read_param_tvector(double)");
  data.resize(nt + 2);
  h_nessi::input_detail::require_words(words, m, 1, "read_param_tvector(double)");
  // test whether the first word is a file
  if ((pch = strstr(words[m], "--"))) {
    std::vector<double> tmp;
    read_vector_from_file(pch + 2, tmp, nt + 2);  // read as many numbers
    for (int tstp = -1; tstp <= nt; tstp++) data[tstp + 1] = tmp[tstp + 1];
  } else {
    double tmp;
    read_param(words, m, tmp);
    for (int tstp = -1; tstp <= nt; tstp++) data[tstp + 1] = tmp;
  }
}
///
inline void read_param_tvector(char *line, std::vector<h_nessi::DColVector> &data, int dim, int nt) {
  std::vector<char *> words;
  h_nessi::input_detail::split_words(line, words, " \t");
  read_param_tvector(words, 0, data, dim, nt);
}
inline void read_param_tvector(char *line, std::vector<h_nessi::cplx> &data, int nt) {
  std::vector<char *> words;
  h_nessi::input_detail::split_words(line, words, " \t");
  read_param_tvector(words, 0, data, nt);
}
inline void read_param_tvector(char *line, std::vector<double> &data, int nt) {
  std::vector<char *> words;
  h_nessi::input_detail::split_words(line, words, " \t");
  read_param_tvector(words, 0, data, nt);
}

inline void find_param_tvector(char *file, const char *flag, std::vector<h_nessi::DColVector> &data, int dim, int nt) {
  try {
    char line[INPUT_LINELEN], *pch;
    find_line_with_flag(file, flag, line);
    pch = strstr(line, flag);
    read_param_tvector(pch + strlen(flag), data, dim, nt);
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string(e.what()) + " [find_param_tvector flag '" + flag + "', file '" + file + "']");
  }
}
inline void find_param_tvector(char *file, const char *flag, std::vector<h_nessi::cplx> &data, int nt) {
  try {
    char line[INPUT_LINELEN], *pch;
    find_line_with_flag(file, flag, line);
    pch = strstr(line, flag);
    read_param_tvector(pch + strlen(flag), data, nt);
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string(e.what()) + " [find_param_tvector flag '" + flag + "', file '" + file + "']");
  }
}
inline void find_param_tvector(char *file, const char *flag, std::vector<double> &data, int nt) {
  try {
    char line[INPUT_LINELEN], *pch;
    find_line_with_flag(file, flag, line);
    pch = strstr(line, flag);
    read_param_tvector(pch + strlen(flag), data, nt);
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string(e.what()) + " [find_param_tvector flag '" + flag + "', file '" + file + "']");
  }
}

#endif
