#include "GridWorldPlayer.h"

/**
 * デフォルトコンストラクタ
 */
GridPlayer::GridPlayer() {
}

/**
 * 初期化などの処理を行う
 */
void GridPlayer::Init() {
	mt.seed(rnd());
	dist.param(std::uniform_int_distribution<>::param_type(0, kAction - 1));
	epsilon.param(std::uniform_real_distribution<>::param_type(0.0, 1.0));
}

/**
 * 行動を決定する
 */
Action GridPlayer::SelectAction(State s_) {
  /* ここを実装 */
  double e = epsilon(mt);
  Action ret = 0;

  /* epsilon-Greedy方策の実装 */
  /* 確率 kEpsilon でランダムに行動を選択する */
  if (e < kEpsilon) {
    ret = dist(mt);
  }
  /* 確率 1 - kEpsilon で, 現在の状態における, 最も行動価値の高い行動を選択する */
  else {
    double max_q = -1e9;

    for (int i = 0; i < kAction; ++i) {
      if (max_q < Q[s_.h][s_.w][i]) {
        max_q = Q[s_.h][s_.w][i];
        ret = i;
      }
    }
  }
  /* 決定した行動を返す */

  return ret;
}

/**
 * 状態 s_ が終端状態かどうかを判定する
 */
// 適当に使ってください
bool GridPlayer::IsEnd(State s_) {
  return ((s_.h == 0 and s_.w == 0) or (s_.h == (kHeight - 1) and s_.w == (kWidth - 1)));
}

/**
 * 一つの状態遷移を元に, 状態 s_, 行動 a_ の行動価値を更新する
 */
void GridPlayer::Update(State s_, Action a_, State ns_, Reward r_) {
  /* ここを実装 */
  double target = 0;

  /* 次状態が終端なら 0, 非終端なら最大の行動価値を取得 */
  if (IsEnd(ns_)) {
    target = r_;
  }
  else {
    double max_q = -1e9;

    for (int i = 0; i < kAction; ++i) {
      if (max_q < Q[ns_.h][ns_.w][i]) {
        max_q = Q[ns_.h][ns_.w][i];
      }
    }

    target = r_ + kGamma * max_q;
  }

  /* 状態行動対 (s_, a_) の行動価値を更新する */
  Q[s_.h][s_.w][a_] += kAlpha * (target - Q[s_.h][s_.w][a_]);

	return;
}

/**
 * 各状態行動対の価値をすべて出力する
 */
void GridPlayer::Print() {
	for (int h = 0; h < kHeight; ++h) {
    for (int w = 0; w < kWidth; ++w) {

      if (IsEnd(State(h, w))) {
        continue;
      }

      std::cerr << "state : (" << h << ", " << w << ")" << std::endl;

      for (int a = 0; a < kAction; ++a) {
        std::cerr << "Q[" << a << "] = " << Q[h][w][a] << std::endl;
      }

      std::cerr << "-----------------------" << std::endl;
    }
	}
}
