#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <ostream>
#include <random>
#include <tuple>
#include <vector>

/**
 * and, or をロジスティック回帰するためのソースコード (xor は失敗するはず)
 * 勾配の最適化手法は 最急降下法
 * xor も分類したいならソフトマックス関数を使って回帰すること (ただしいろいろ書き換えないといけないので注意)
 */

using std::cin;
using std::cout;
using std::cerr;
using std::endl;

// ステップサイズ
constexpr static double kStepSize = 0.01;
// しきい値
constexpr static double kEPS = 1e-4;
// inputの数 (今回は2) + バイアス
constexpr static int kInputSize = 2 + 1;

/************************************* テンプレート ****************************************/

// std::pair を cout, cerr で出力できるようにする
template <typename T>
std::ostream& operator << (std::ostream& os, std::pair<T, T>& p) {
  os << "(" << p.first << ", " << p.second << ")";

  return os;
}

// std::vector を cout, cerr で出力できるようにする
template <typename T>
std::ostream& operator << (std::ostream& os, std::vector<T>& v) {
  os << "[";
  for (int i = 0; i < int(v.size()); ++i) {
    os << v[i] << (i == int(v.size()) - 1 ? "" : ", ");
  }
  os << "]";

  return os;
}

// ベクトルと定数の掛け算 ( T は int か double を想定しているのでその他の挙動は保証していない)
// c * vec と書けばOK. vec * c だと compile error.
template <typename T, typename U>
std::vector<T> operator * (const T c, const std::vector<U>& v) {
  std::vector<T> ret(v.size());

  for (int i = 0; i < int(v.size()); ++i) {
    ret[i] = v[i] * c;
  }

  return ret;
}

// ベクトル同士の引き算
template <typename T, typename U>
std::vector<T> operator - (const std::vector<T>& a, const std::vector<U>& b) {
  std::vector<T> ret(a.size());

  for (int i = 0; i < int(a.size()); ++i) {
    ret[i] = a[i] - b[i];
  }

  return ret;
}

/************************************* テンプレート ****************************************/

// シグモイド関数
inline double sigmoid(const double& t) {
  return 1.0 / (1.0 + exp(-t));
}

int main() {
  // 入力データ
  std::vector <std::vector<int> > data = { { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 }, { 1, 1, 1 } }; // 3つめはバイアス
  // 入力に対する正解データ
  std::vector<int> and_ = { 0, 0, 0, 1 }, or_ = { 0, 1, 1, 1 }, xor_ = { 0, 1, 1, 0 };
  // xorを分類する場合
  std::vector<int> answer = or_;
  // パラメータ
  std::vector<double> theta(kInputSize);

  std::random_device rnd;
  std::mt19937 mt(rnd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  // パラメータを適当に初期化 (多分0でもOK)
  for (int i = 0; i < kInputSize; ++i) {
    theta[i] = dis(mt);
  }

  // 一定回数繰り返す (収束しなかったらずっとループするので一応上限を設ける)
  for (int iter = 0; iter <= 100000000; ++iter) {
    // 途中経過を出力
    if (iter % 100000 == 0) {
      cout << "iteration = " << iter << endl;
      cout << "theta = " << theta << endl;

      for (int i = 0; i < int(data.size()); ++i) {
        double a = 0;
        for (int j = 0; j < kInputSize; ++j) {
          a += theta[j] * data[i][j];
        }

        cout << "data[" << i << "] = " << data[i] << endl;
        cout << "answer = " << answer[i] << endl;
        cout << "result = " << sigmoid(a) << endl;
        cout << endl;
      }
    }

    // 最急降下法
    // パラメータの勾配を格納するvector
    std::vector<double> grad(kInputSize, 0.0);

    // パラメータの j 番目の勾配を求める
    for (int j = 0; j < kInputSize; ++j) {
      // ここわかりにくいかもしれない
      for (int i = 0; i < int(data.size()); ++i) {
        double a = 0;
        for (int k = 0; k < kInputSize; ++k) {
          a += theta[k] * data[i][k];
        }
        double y = sigmoid(a);

        grad[j] += (y - answer[i]) * data[i][j];
      }
    }

    // 収束具合の確認
    bool is_convergence = true;
    for (int i = 0; i < int(grad.size()); ++i) {
      if (fabs(grad[i]) > kEPS) {
        is_convergence = false;
        break;
      }
    }

    // パラメータが収束していたら終了
    if (is_convergence) {
      break;
    }

    // パラメータ の更新
    grad = kStepSize * grad;
    theta = theta - grad;
  }

  // 出力
  cout << endl;
  cout << "theta = " << theta << endl;
  cout << endl;

  for (int i = 0; i < int(data.size()); ++i) {
    double a = 0;
    for (int j = 0; j < kInputSize; ++j) {
      a += data[i][j] * theta[j];
    }

    cout << "data[" << i << "] = " << data[i] << endl;
    cout << "answer = " << answer[i] << endl;
    cout << "result = " << sigmoid(a) << endl;
    cout << endl;
  }

  return 0;
}
