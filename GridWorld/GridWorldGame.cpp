#include "GridWorldGame.h"

/**
 * 引数ありコンストラクタ
 */
GridWorld::GridWorld() {
  param_h = std::uniform_int_distribution<>::param_type(0, kHeight - 1);
  param_w = std::uniform_int_distribution<>::param_type(0, kWidth - 1);
}

/**
 * 状態の初期化
 */
void GridWorld::InitState() {
  // 初期位置をランダムに初期化
  mt.seed(rnd());
  dis_h.param(this->param_h);
  dis_w.param(this->param_w);

  int sh = 0, sw = 0;

  // あんまり賢くない気がしている
  do {
    sh = dis_h(mt); // h
    sw = dis_w(mt); // w
  } while ((sh == 0 and sw == 0) or (sh == (kHeight - 1) and sw == (kWidth - 1)));

  this->state.h = sh;
  this->state.w = sw;
}

/**
 * 現在の状態を返す
 */
State GridWorld::GetState() {
  return state;
}

/**
 * AI の行動を受け取ってゲームの状態を更新する
 * action : 0 ... Down
 * action : 1 ... Right
 * action : 2 ... Up
 * action : 3 ... Left
 */
void GridWorld::UpdateState(Action a) {

  // 壁に向かって進もうとした場合, その場に留まる
  for (int i = 0; i < kSkipFrame; ++i) {
    this->state.h = std::max(0, std::min(kHeight - 1, this->state.h + vh[a]));
    this->state.w = std::max(0, std::min(kWidth - 1, this->state.w + vw[a]));
		assert(0 <= this->state.h and this->state.h < kHeight and 0 <= this->state.w and this->state.w < kWidth);
  }
}

/**
 *  最新の報酬を返す
 */
Reward GridWorld::GetLastReward() {
  return -1;
}

/**
 *  ゲームが終了しているかどうかを返す
 */
bool GridWorld::IsEnd() {
  return ((this->state.h == 0 and this->state.w == 0) or ((this->state.h == (kHeight - 1)) and (this->state.w == (kWidth - 1))));
}

/**
 *  画像のチャネル数を指定する
 */
void GridWorld::SetImageChannel(int c) {
  assert(c == 1 or c == 3);
  this->is_grayscale = (c == 1);
}

