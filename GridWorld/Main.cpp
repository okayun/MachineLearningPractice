#include <iostream>
#include <algorithm>
#include <memory>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "GridWorld.h"

using std::cin;
using std::cout;
using std::cerr;
using std::endl;

DEFINE_int32(episodes, 1000, "episode's iteration");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  google::LogToStderr();

  // kHeight, kWidth, kSkipFrame は 'GridPlayer.h' で定義している
  std::shared_ptr<GridWorld> grid_world = std::shared_ptr<GridWorld>(new GridWorld());

  // カラー画像を返すように設定
  grid_world->SetImageChannel(kImageChannel);

  // プレイヤ
  std::shared_ptr<GridPlayer> player = std::shared_ptr<GridPlayer>(new GridPlayer());

  player->Init();

  cerr << "エピソード数 : " << FLAGS_episodes << endl;

  // 1. 一定回数エピソードを繰り返す
  for (int ep = 1; ep <= FLAGS_episodes; ++ep) {
    // 2. 初期状態を決定
    grid_world->InitState();

    // ゲームが終了するまで繰り返し
    while (true) {
      /*
      ** 3. 現在の状態を取得する
      */
      //現在の状態を取得
      State state = grid_world->GetState();

      /*
      ** 4. 行動を決定する
      */
      // GridPlayer::SelectAction の中身を実装する
      Action action = player->SelectAction(state);

      /*
      ** 5. 行動を行い, 次の状態と報酬を観測する
      */
      grid_world->UpdateState(action);
      State next_state = grid_world->GetState();
      Reward reward = grid_world->GetLastReward();

      /*
      ** 6. 行動価値を更新する
      */
      // GridPlayer::Update の中身を実装
      player->Update(state, action, next_state, reward);

      /*
      ** 7. 終端状態に到達していたら終了
      */
      if (grid_world->IsEnd()) {
        break;
      }
    }

    // エピソード 100 回ごとに推定途中の行動価値を出力
    if (ep % 100 == 0) {
      player->Print();
    }
  }

  // 最後にすべての行動価値を出力
  player->Print();

  return 0;
}
