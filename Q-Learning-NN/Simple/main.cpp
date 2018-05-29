#include "caffe/caffe.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include <algorithm>
#include <fstream>
#include <functional>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <queue>
#include <random>
#include <string>

#include "opencv2/opencv.hpp"

/**
 * 「強化学習」p97 4×4の格子の世界
 *  価値反復法 (の一つである Q-Learning) により最適行動価値関数を求める
 *
 *  行動は上下左右への移動のみ
 *  1 度の遷移で -1 の報酬が与えられる
 *  進めない方向へ移動しようとした場合 ((0, 1) -> (-1, 1) など), その場にとどまる
 *  初期状態を(0, 3) に固定して, (3, 0) の "DOWN" の行動価値を求められるかどうかを見る
*/

//#define DEBUG // アニメーション (っぽいやつ) 付きで学習 (ただし学習速度はかなり遅くなる)

// -snapshot *.solver_state とすれば, その重みから学習を再開できる
DEFINE_string(snapshot, "", "SolverState file to load");

using std::cin;
using std::cout;
using std::cerr;
using std::endl;

/********************* 画像表示用変数 **********************/

const int HEIGHT = 100; // row
const int WIDTH = 100; // col
const int LINEWIDTH = 2; // 枠線の太さ

/**********************************************************/



/******************* 学習に使用する変数 *******************/

const double ALPHA = 0.1; // 学習率
const double GAMMA = 1.0; // 割引率
const double EPSILON = 0.7; // ε

// enum 使えば良さそう
const int DOWN = 0;
const int RIGHT = 1;
const int UP = 2;
const int LEFT = 3;

const int MAX_ACTION = 1000000; // 行動回数の上限
const int ACTION_NUM = 4; // 可能な行動の数
const int INPUT_NUM = 2; // NN への入力の数 (状態を表す数)
const int mini_batch_size = 1;
const int batch_size = 1;

std::string dir_s[4] = {"DOWN", "RIGHT", "UP", "LEFT"}; // 確認出力などに用いる
int vx[4] = {1, 0, -1, 0}, vy[4] = {0, 1, 0, -1}; // 上下左右

int counter; // ループカウンタ

std::uniform_int_distribution<> rndm(0, 3); // ランダム行動のときに使う
std::uniform_real_distribution<> ep(0.0, 1.0); // ε-greedy で使う

// ネットワークに関する変数
boost::shared_ptr<caffe::Solver<float>> solver; // solver.prototxt を読み込む
boost::shared_ptr<caffe::Net<float>> net; // ネットワークを取得する
boost::shared_ptr<caffe::MemoryDataLayer<float>> inputLayer, labelLayer, filterLayer; // 各層を読み込む
boost::shared_ptr<caffe::Blob<float>> Qvalue, loss, filteredQvalue; // 上と同様

float input[16 * INPUT_NUM * 4], label[16 * ACTION_NUM * 4], filter[16 * ACTION_NUM * 4]; // NN にデータを流すときに用いる
double loss_average = 0.0; // 損失関数値の平均値

/**********************************************************/




void Check(int nowx, int nowy);
double getActionQvalue(int x, int y, int action);

/*********************** 画像を表示する関数など ************************/

// 端っこを黒色に塗りつぶす
void drawBorder(cv::Mat& img, int x, int y) {
  // 上
  for (int i = 0; i < img.rows; ++i) {
    int s = (x == 0 ? 2 : 1) * LINEWIDTH;

    for (int j = 0; j < s; ++j) {
      for (int k = 0; k < 3; ++k) {
        // 3 channel の画像の場合
        img.at<cv::Vec3b>(j, i)[k] = 0;
      }
    }
  }

  // 下
  for (int i = 0; i < img.rows; ++i) {
    int s = (x == 3 ? 2 : 1) * LINEWIDTH;

    for (int j = 0; j < s; ++j) {
      for (int k = 0; k < 3; ++k) {
        img.at<cv::Vec3b>(img.cols - (j + 1), i)[k] = 0;
      }
    }
  }

  // 左
  for (int i = 0; i < img.cols; ++i) {
    int s = (y == 0 ? 2 : 1) * LINEWIDTH;

    for (int j = 0; j < s; ++j) {
      for (int k = 0; k < 3; ++k) {
        img.at<cv::Vec3b>(i, j)[k] = 0;
      }
    }
  }

  // 右
  for (int i = 0; i < img.cols; ++i) {
    int s = (y == 3 ? 2 : 1) * LINEWIDTH;

    for (int j = 0; j < s; ++j) {
      for (int k = 0; k < 3; ++k) {
        img.at<cv::Vec3b>(i, img.rows - (j + 1))[k] = 0;
      }
    }
  }  

}

// 全部塗りつぶす
void makeNormalPicture(cv::Mat& img, int x, int y) {
  img = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
  auto color = ((x == y) and (x == 0 or x == 3) ? cv::Scalar(41, 232, 209) : cv::Scalar(255, 255, 255));
  img = color;
  drawBorder(img, x, y);
} 

// 点を打つ
void makePointPicture(cv::Mat& img, int x, int y) {
  img = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
  // 一旦塗りつぶす
  auto color = ((x == y) and (x == 0 or x == 3) ? cv::Scalar(255, 100, 100) : cv::Scalar(255, 255, 255));
  img = color;
  // 画像の中心に赤色の円を描画する
  // cv::Scalar(青, 緑, 赤) に注意
  cv::circle(img, cv::Point(WIDTH / 2, HEIGHT / 2), HEIGHT / 4, cv::Scalar(0, 0, 255), -1, CV_AA);
  drawBorder(img, x, y);
}

// 描画する
void show(int x, int y) {
  cv::Mat imgs[4][4], tmp;

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      if (i == x and j == y) {
        makePointPicture(tmp, i, j);
      }
      else {
        makeNormalPicture(tmp, i, j);
      }

      imgs[i][j] = tmp.clone();
    }
  }

  for (int i = 0; i < 4; ++i) {
    for (int j = 1; j < 4; ++j) {
      cv::hconcat(imgs[i][0], imgs[i][j], imgs[i][0]);
    }
  }

  for (int i = 1; i < 4; ++i) {
    cv::vconcat(imgs[0][0], imgs[i][0], imgs[0][0]);
  }

  cv::imshow("Map", imgs[0][0]);
  cv::waitKey(1); // cv::waitKey(t) で t ms 間表示 (t == 0 ならキーの入力があるまで表示)
}

/**********************************************************************/



/******************* NN の読み込みや学習の本体など *******************/

/*
 * 初期化やネットワークの読み込みなど
**/
void Init() {
	caffe::SolverParameter param;

	caffe::ReadSolverParamsFromTextFileOrDie("solver.prototxt", &param);

	solver = boost::shared_ptr<caffe::Solver<float>>(
		caffe::SolverRegistry<float>::CreateSolver(param));

	net = solver->net();

	// inputLayer : 入力
	// targetLayer : 教師
	// filterLayer : よくわからん
	inputLayer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
		net->layer_by_name("input_layer"));
	labelLayer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
		net->layer_by_name("label_layer"));
	filterLayer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
		net->layer_by_name("filter_layer"));

	// ソルバー読み込み
	if(FLAGS_snapshot.size()) {
		solver->Restore(FLAGS_snapshot.c_str());
	}

	// Brobを登録
	// qvalueあってるかわからん…
	Qvalue = net->blob_by_name("q_values");
	loss = net->blob_by_name("loss");
	filteredQvalue = net->blob_by_name("filtered_q_values");

	assert(solver);
	assert(net);
	assert(inputLayer);
	assert(labelLayer);
	assert(filterLayer);
	assert(Qvalue);
	assert(loss);
}

/*
 * バッチサイズの変更
**/
void setBatchSize(int batchSize) {

	// MemoryDataLayerで参照する配列・バッチサイズを設定
	// input, label, filterは静的配列
	inputLayer->set_batch_size(batchSize);
	labelLayer->set_batch_size(batchSize);
	filterLayer->set_batch_size(batchSize);

	// 参照する配列の変更
	inputLayer->Reset(input, input, batchSize);
	labelLayer->Reset(label, label, batchSize);
	filterLayer->Reset(filter, filter, batchSize);
}

/*
 * 状態 (x, y) における行動を決定する
**/
int selectAction(int x, int y) {
	int ret = 0;

	std::random_device rnd;
	std::mt19937 mt(rnd());

	double maxQ = -1e9;
	double epsilon = ep(mt);

	// バッチサイズ1
	setBatchSize(1);

	for(int i = 0; i < 4; ++i) {
		filter[i] = 1;
	}

	// ランダムに選択
	// ここは大丈夫
	if(epsilon < EPSILON) {
		ret = rndm(mt);
	}
	// 最もQ値の高い行動を選択する
	else {
		// ここらへんでForwardする
		input[0] = x, input[1] = y;
		net->Forward();

		// 4つの行動の中から, 最もQ値の高い行動を選択する
		for(int i = 0; i < 4; ++i) {
			if(maxQ < Qvalue->cpu_data()[i]) {
				maxQ = Qvalue->cpu_data()[i];
				ret = i;
			}
		}
	}

	return ret;
}

/*
 * 状態 nowState で行動 action を選択したときの行動価値を更新する
**/
void Update(std::pair<int, int> nowState, int action) {
	float nextMaxQ = -1e9;

	int nx = std::max(0, std::min(3, nowState.first + vx[action])),
		ny = std::max(0, std::min(3, nowState.second + vy[action]));
  //Check(nowState.first, nowState.second);
	// 次の状態が終端状態のとき
	if(nx == ny && (nx == 0 || nx == 3)) {
		nextMaxQ = 0;
	}
	// 終端状態でないとき
	else {
		setBatchSize(1);

		input[0] = nx, input[1] = ny;
		for(int i = 0; i < 4; ++i) {
			filter[i] = 1;
		}

		net->Forward();

		// 次の状態で最大のQを取ってくる
		for(int i = 0; i < 4; ++i) {
			nextMaxQ = std::max(nextMaxQ, Qvalue->cpu_data()[i]);
		}
	}

	setBatchSize(1);

	input[0] = nowState.first, input[1] = nowState.second;

	// 学習させたいやつだけ状態行動対だけ学習させる
	for(int i = 0; i < 4; ++i) {
		if(i == action) {
			filter[i] = 1;
			label[i] = -1 + GAMMA * nextMaxQ;
		}
		else {
			filter[i] = 0;
			label[i] = 0;
		}
	}

	// 学習
	solver->Step(1);

	loss_average += loss->cpu_data()[0];

	if(solver->iter() % 5000 == 0) {
		// 損失関数値
		// std::ios::appとすると,ファイルに追加で書き込むことができる
		std::ofstream ofs("train" + std::to_string(counter) + ".txt", std::ios::app);
		// 一定回数反復するごとに, その時の反復回数, 損失関数値の平均, (3, 0) の "DOWN" の行動価値をファイルへ出力
		ofs << "iter, loss_average: " << solver->iter() << " " << (loss_average / 5000) << " " << getActionQvalue(3, 0, DOWN) << endl;
		ofs.close();
		loss_average = 0.0;
	}

	// 一定回数ごとにスナップショットを作成
	if(solver->iter() % 500000 == 0) {
		solver->Snapshot();
	}
}

/*
 * 学習など
**/
void Q_Learning() {
	std::random_device rnd;
	std::mt19937 mt(rnd());

	for(int count_action = 0;;) {
		/*
		// 非終端状態の中から初期状態をランダムに選ぶ
		int px = rndm(mt), py = rndm(mt);
		while(px == py && (px == 0 || px == 3)) {
			px = rndm(mt);
			py = rndm(mt);
		}
		*/

		// 初期状態を固定
		int px = 0, py = 3;

		while(true) {
#ifdef DEBUG
      show(px, py);
#endif
			int action_index = 0;

			// ε-Greedy
			action_index = selectAction(px, py);
			count_action++;

			int nx = px + vx[action_index], ny = py + vy[action_index];

			if(nx < 0 || nx > 3 || ny < 0 || ny > 3) {
				nx = px, ny = py;
			}

			/** 3-3 **/
			Update({px, py}, action_index);

			// 最後に現在の状態の更新
			// 3-4
			px = nx, py = ny;

			// 3-5
			if(px == py && (px == 0 || px == 3) || count_action >= MAX_ACTION) {
#ifdef DEBUG
        show(px, py);
#endif
				break;
			}
		}

		if (count_action >= MAX_ACTION) {
			break;
		}
	}

	// 最後にネットワークの重みを保存
	solver->Snapshot();
}

/*
 * 状態 (x, y) の行動 action の行動価値を取得する
**/
double getActionQvalue(int x, int y, int action) {
	setBatchSize(1);

	input[0] = x;
	input[1] = y;

	for (int i = 0; i < 4; ++i) {
		filter[i] = 1;
	}

	net->Forward();

	return Qvalue->cpu_data()[action];
}

/**********************************************************/

int main(int argc, char** argv) {
  // おまじない
	FLAGS_alsologtostderr = 1;
	caffe::GlobalInit(&argc, &argv);

  // 異なる初期重みで複数回学習を行う
	for (counter = 1; counter <= 10; ++counter) {
		Init();
		Q_Learning();
	}

  cv::destroyAllWindows(); // 最後に画面に表示した画像を閉じる

	return 0;
}
