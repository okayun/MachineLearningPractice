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

using std::cin;
using std::cout;
using std::cerr;
using std::endl;

double learning_rate = 0.01; // 一定の反復回数ごとに小さくするなどしたほうが良い
const double EPS = 1e-5;

const int InputSize = 3; // inputの数 + バイアス

/*****************************************************************************************/

template <typename T>
std::ostream& operator << (std::ostream& os, std::pair<T, T>& p) {
  os << "(" << p.first << ", " << p.second << ")";

  return os;
}

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

// ベクトル同士の掛け算
template <typename T, typename U>
std::vector<T> operator * (const std::vector<T>& a, const std::vector<U>& b) {
  std::vector<T> ret(a.size());

  for (int i = 0; i < a.size(); ++i) {
    ret[i] = a[i] * b[i];
  }

  return ret;
}


// ベクトル同士の足し算
template <typename T, typename U>
std::vector<T> operator + (const std::vector<T>& a, const std::vector<U>& b) {
  std::vector<T> ret(a.size());

  for (int i = 0; i < int(a.size()); ++i) {
    ret[i] = a[i] + b[i];
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

/*****************************************************************************************/

// シグモイド関数
inline double sigmoid(const double& t) {
  return 1.0 / (1.0 + exp(-t));
}

// 勾配の計算
std::vector<double> gradient(const double sig, const double answer, const std::vector<int>& input) {
  double coef = (sig - answer);
  return coef * input;
}

int main() {
  std::vector<int> and_ = {0, 0, 0, 1}, or_ = {0, 1, 1, 1}, xor_ = {0, 1, 1, 0};

  std::vector <std::vector<int> > input = { { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 }, { 1, 1, 1 } }; // 3つめはバイアス
  std::vector<int> answer = or_;
  std::vector<double> theta(3, 0.0);

  std::random_device rnd;
  std::mt19937 mt(rnd());

  int index[4] = {0, 1, 2, 3}; // 順番をシャッフルするための配列

  bool flag;

  for (int iter = 0; iter <= 100000000; ++iter) {
    // check
    if (iter % 100000 == 0) {
      cout << "iteration = " << iter << endl;
      cout << "theta = " << theta << endl;

      for (int i = 0; i < int(input.size()); ++i) {
        double a = 0;
        for (int j = 0; j < InputSize; ++j) {
          a += theta[j] * input[i][j];
        }

        cout << "input = " << input[i] << endl;
        cout << "answer = " << answer[i] << endl;
        cout << "prediction = " << sigmoid(a) << endl;
        cout << endl;
      }
    }

    // 確率的勾配降下法

    std::shuffle(index, index + int(input.size()), mt);

    flag = false;

    // theta の更新
    for (int i = 0; i < int(input.size()); ++i) {
      // sigmoidの値を計算する
      double a = 0;
      for (int j = 0; j < InputSize; ++j) {
        a += theta[j] * input[index[i]][j];
      }
      double y = sigmoid(a);

      // i 番目のデータに対する勾配の計算
      std::vector<double> grad = gradient(y, answer[index[i]], input[index[i]]);

      // 勾配の収束具合の確認
      bool is_convergence = true;
      for (int j = 0; j < int(grad.size()); ++j) {
        if (fabs(grad[j]) > EPS) {
          is_convergence = false;
        }
      }

      // i 番目のデータについて, \thetaの値が収束していたら勾配の更新を行わない
      if (is_convergence) {
        continue;
      }

      // theta の更新
      grad = learning_rate * grad;
      theta = theta - grad;
      flag = true;
    }

    // 一度も theta が更新されなかったら収束とみなす
    if (!flag) {
      break;
    }
  }

  // 出力
  cout << endl;
  cout << "theta = " << theta << endl;
  cout << endl;

  for (int i = 0; i < int(input.size()); ++i) {
    double a = 0;
    for (int j = 0; j < InputSize; ++j) {
      a += input[i][j] * theta[j];
    }

    cout << "input = " << input[i] << endl;
    cout << "answer = " << answer[i] << endl;
    cout << "prediction = " << sigmoid(a) << endl;
    cout << endl;
  }

  return 0;
}
