#include <iostream>
#include <algorithm>
#include <random>
#include <string>

/**
 * 「強化学習」p97 4×4の格子の世界
 *  価値反復法(Q-Learning)により最適行動価値関数を求める
 *
 *  行動は上下左右への移動のみ
 *  1 度の遷移で -1 の報酬が与えられる
*/

//#define DEBUG

using std::cin;
using std::cout;
using std::cerr;
using std::endl;

const double ALPHA = 0.1; // 学習率
const double GAMMA = 0.9; // 割引率
const double EPSILON = 0.01;
const int DOWN = 0;
const int RIGHT = 1;
const int UP = 2;
const int LEFT = 3;

/**
 *
*/
int dir[4] = { DOWN, RIGHT, UP, LEFT };
std::string dir_s[4] = { "DOWN", "RIGHT", "UP", "LEFT" };
int vx[4] = { 1, 0, -1, 0 }, vy[4] = { 0, 1, 0, -1 };

// 行動価値関数
double Q[4][4][4];

void Q_Learning() {
  std::random_device rnd;
  std::mt19937 mt(rnd());
  std::uniform_real_distribution<> ep(0.0, 1.0);

  // Q の初期化
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        Q[i][j][k] = 0;
      }
    }
  }

#ifdef DEBUG
  cerr << endl;
  cerr << "Previous Q-Value" << endl;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        cerr << "Q[" << i << "][" << j << "][" << dir_s[k] << "] = " << Q[i][j][k] << (k == 3 ? "\n" : " ");
      }
    }
  }
  cerr << endl;
#endif

  std::uniform_int_distribution<> rndm(0, 3);

  for (int episode = 0; episode < 10000000; ++episode) {
    if (episode % 1000 == 0) {
      cout << "EPISODE " << episode << endl;
    }
    int px = rndm(mt), py = rndm(mt);

    //cerr << "start position : " << px << " " << py << endl;

    while (true) {
      if (px == py && (px == 0 || py == 3)) {
        //cout << "EPISODE " << episode << " END" << endl;

#ifdef DEBUG
        cerr << endl;
        for (int i = 0; i < 4; ++i) {
          for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
              cerr << "Q[" << i << "][" << j << "][" << dir_s[k] << "] = " << Q[i][j][k] << (k == 3 ? "\n" : " ");
            }
          }
        }
        cerr << endl;
#endif
        break;
      }

      double maxQ = -1e9;
      int action = UP;
      double epsilon = ep(mt);

      // ε-Greedy

      // ランダムに選択
      if (epsilon < EPSILON) {
        action = rndm(mt);
      }
      // 
      else {

        // 4つの行動の中から, 最もQ値の高い行動を選択する
        for (int i = 0; i < 4; ++i) {
          if (maxQ < Q[px][py][dir[i]]) {
            maxQ = Q[px][py][dir[i]];
            action = dir[i];
          }
        }
      }

      int nx = px + vx[action], ny = py + vy[action];

      if (nx < 0 || nx > 3 || ny < 0 || ny > 3) {
        nx = px, ny = py;
      }

      maxQ = -1e9;

      for (int i = 0; i < 4; ++i) {
        maxQ = std::max(maxQ, Q[nx][ny][dir[i]]);
      }

      Q[px][py][action] += (ALPHA * (-1 + maxQ - Q[px][py][action]));

      px = nx, py = ny;

      //cerr << "px, py = " << px << " " << py << endl;
    }
  }

  cerr << endl;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        cerr << "Q[" << i << "][" << j << "][" << dir_s[dir[k]] << "] = " << Q[i][j][dir[k]] << (k == 3 ? "\n" : " ");
      }
    }
  }
  cerr << endl;
}

void Output() {
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      double maxQ = -1e9;
      int action = UP;

      for (int k = 0; k < 4; ++k) {
        if (Q[i][j][dir[k]] > maxQ) {
          maxQ = Q[i][j][dir[k]];
          action = dir[k];
        }
      }
      cout << dir_s[action] << (j == 3 ? "\n" : " ");
    }
  }
}

int main() {

  Q_Learning();
  Output();

  return 0;
}
