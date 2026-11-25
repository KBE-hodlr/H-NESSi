// B19: two translation units that both include read_inputfile.hpp must link (and run).
#include "h_nessi/read_inputfile.hpp"
int tu2_value();
int main() { return tu2_value() == 42 ? 0 : 1; }
