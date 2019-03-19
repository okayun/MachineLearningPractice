#ifndef __GRIDWORLD_GAME_H__
#define __GRIDWORLD_GAME_H__

#include <algorithm>
#include <functional>
#include <vector>
#include <random>
#include <map>
#include <ostream>
#include <cassert>

#include "opencv2/opencv.hpp"

#include "GridWorldConsts.h"
#include "GridWorldTypes.h"

/**
 *  ゲームを扱うクラス
 */
class GridWorld {
 private:
  // 現在の状態
  State state;
  // 画像をグレースケール化するかどうか
  bool is_grayscale;
  // その他
  std::random_device rnd;
  std::mt19937 mt;
  std::uniform_int_distribution<> dis_h, dis_w;
  std::uniform_int_distribution<>::param_type param_h, param_w;

 public:
  GridWorld();
  void InitState();
  State GetState();
  void UpdateState(Action a);
  Reward GetLastReward();
  bool IsEnd();
  void SetImageChannel(int c = 1);
};

#endif
