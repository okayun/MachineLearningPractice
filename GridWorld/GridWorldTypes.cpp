#include "GridWorldTypes.h"

/**
 * デフォルトコンストラクタ
 */
State::State() {}

/**
 * 引数ありコンストラクタ
 */
State::State(int h_, int w_) : h(h_), w(w_) {}

/**
 * オペレータ
 */
std::ostream& operator << (std::ostream& out, State& s) {
  out << "(" << s.h << ", " << s.w << ")";
  return out;
}