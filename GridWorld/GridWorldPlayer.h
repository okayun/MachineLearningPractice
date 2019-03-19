#ifndef __GRIDWORLD_PLAYER_H__
#define __GRIDWORLD_PLAYER_H__

#include <iostream>
#include <algorithm>
#include <random>
#include <map>
#include <cstring>
#include <cassert>

#include "GridWorldConsts.h"
#include "GridWorldTypes.h"

// 取得する画像のチャネル数
// 1 : グレースケール, 3 : RGB
static constexpr int kImageChannel = 3;
// 学習率 (色々変えてみてください)
static constexpr double kAlpha = 0.1;
// 割引率
static constexpr double kGamma = 1.0;
// ランダムに行動する確率 (色々変えてみてください)
static constexpr double kEpsilon = 0.3;

/**
 * プレイヤを表すクラス
 */
class GridPlayer {
 private:
  /* 行動価値を保存する配列を作成 */
  /* ここを実装 */
  double Q[kHeight][kWidth][kAction] = { { { } } };

  // その他変数
  std::random_device rnd;
  std::mt19937 mt;
  // ランダムに行動を選択する際に使用
  std::uniform_int_distribution<> dist;
  // epsilon-Greedy方策を実装する際に使用
  std::uniform_real_distribution<> epsilon;

 public:
  GridPlayer();
  void Init();
  Action SelectAction(State s_);
  bool IsEnd(State s_);
  void Update(State s_, Action a_, State ns_, Reward r_);
  void Print();
};

#endif
