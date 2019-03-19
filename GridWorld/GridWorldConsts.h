#ifndef __GRIDWORLD_CONSTS_H__
#define __GRIDWORLD_CONSTS_H__

// 格子の高さ
static constexpr int kHeight = 4;

// 格子の幅
static constexpr int kWidth = 4;

// 可能な行動の数
static constexpr int kAction = 4;

// スキップフレーム数
static constexpr int kSkipFrame = 1;

// 各行動の移動方向
// 順に 下, 右, 上, 左
static constexpr int vh[kAction] = { 1, 0, -1, 0 };
static constexpr int vw[kAction] = { 0, 1, 0, -1 };

#endif