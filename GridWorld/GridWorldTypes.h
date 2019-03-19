#ifndef __GRIDWORLD_TYPES_H__
#define __GRIDWORLD_TYPES_H__

#include <functional>
#include <memory>
#include <ostream>


// 行動
using Action = int;

// 報酬
using Reward = int;

/**
 * ゲームでの状態を表す構造体
 */
struct State {
  int h, w;

  State();
  State(int h_, int w_);
};

/**
 * オペレータを定義
 */
std::ostream& operator << (std::ostream& out, State& s);

// using StatePtr = std::shared_ptr<State>;

#endif