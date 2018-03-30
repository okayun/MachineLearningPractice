#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <queue>
#include <random>
#include <string>

#include "caffe/caffe.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include <gflags/gflags.h>
#include <glog/logging.h>

/** 
 * 「強化学習」p97 4×4の格子の世界
 *  価値反復法(Q-Learning)により最適行動価値関数を求める
 *
 *  行動は上下左右への移動のみ
 *  1 度の遷移で -1 の報酬が与えられる
*/

/**
 * 汚いので後で書き直す
*/

//#define DEBUG

// -snapshot *.solver_state とすれば, その重みから学習を再開できる
DEFINE_string(solver_state, "", "SolverState file to load");

using std::cin;
using std::cout;
using std::cerr;
using std::endl;

class Replay {
public:
  std::pair<int, int> currentPosition, nextPosition;
	float reward;
	int action_index;

	Replay() {}
	Replay(std::pair<int, int> cp, int action_index, std::pair<int, int> np,
		   float reward)
		: currentPosition(cp), action_index(action_index), nextPosition(np),
		  reward(reward) {}
};

const double ALPHA = 0.1; // 学習率
const double GAMMA = 1.0; // 割引率
// const double EPSILON = 0.1;
const int DOWN = 0;
const int RIGHT = 1;
const int UP = 2;
const int LEFT = 3;

/**
 *
*/
int dir[4] = {DOWN, RIGHT, UP, LEFT};
std::string dir_s[4] = {"DOWN", "RIGHT", "UP", "LEFT"};
int vx[4] = {1, 0, -1, 0}, vy[4] = {0, 1, 0, -1};
std::random_device rnd;

double loss_average;

/**
 * network の変数
*/
const int mini_batch_size = 64;
const int batch_size = 1;

boost::shared_ptr<caffe::Solver<float>> solver;
boost::shared_ptr<caffe::Net<float>> net;
boost::shared_ptr<caffe::MemoryDataLayer<float>> inputLayer, labelLayer, filterLayer;
boost::shared_ptr<caffe::Blob<float>> Qvalue, loss, filteredQvalue;
// 2 はネットワークに与える入力データの数, 4 は可能な行動の数
float input[mini_batch_size * 2], label[mini_batch_size * 4],
	filter[mini_batch_size * 4];
// リプレイデータ
std::deque<boost::shared_ptr<Replay>> replay_data;

// solver_stateをどこかで書く
void Init(std::string solver_state) {
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
	if(solver_state.size()) {
		solver->Restore(solver_state.c_str());
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

// バッチサイズの変更 行動選択の前に
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

void AddReplay(std::pair<int, int> cp, int action_index, std::pair<int, int> np, float reward) {
	if(replay_data.size() >= 1000) {
		replay_data.pop_front();
	}

	boost::shared_ptr<Replay> replay =
		boost::shared_ptr<Replay>(new Replay(cp, action_index, np, reward));

	replay_data.push_back(replay);

	return;
}

void Update() {
	if(replay_data.size() < 1000) {
		return;
	}
	// cerr << "learning!!" << endl;
	// 今回はバッチサイズ64
	setBatchSize(mini_batch_size);

	// ランダムにデータを取ってくる
	boost::shared_ptr<Replay> reps[mini_batch_size];
	std::mt19937 mt(rnd());
	for(int i = 0; i < mini_batch_size; ++i) {
		reps[i] = replay_data[mt() % 1000];
	}

	// inputに次の状態のデータをmini_batch_size分入れる
	for(int i = 0; i < mini_batch_size; ++i) {
		if(reps[i]->nextPosition.first == reps[i]->nextPosition.second &&
		   (reps[i]->nextPosition.first == 0 ||
			reps[i]->nextPosition.first == 3)) {
			continue;
		}

		input[i * 2] = reps[i]->nextPosition.first;
		input[i * 2 + 1] = reps[i]->nextPosition.second;
	}

	// またfilterを1にする
	for(int i = 0; i < mini_batch_size * 4; ++i) {
		filter[i] = 1;
	}

	// 次の状態の各行動のQ値を調べる
	net->Forward();

	// 次の状態でForwardして最大のQ値を得る
	float maxQ[mini_batch_size];
	for(int i = 0; i < mini_batch_size; ++i) {
		maxQ[i] = -1e9;

		if(reps[i]->nextPosition.first == reps[i]->nextPosition.second &&
		   (reps[i]->nextPosition.first == 0 ||
			reps[i]->nextPosition.first == 3)) {
			maxQ[i] = 0;
			continue;
		}

		for(int j = 0; j < 4; ++j) {
			if(maxQ[i] < Qvalue->cpu_data()[i * 4 + j]) {
				maxQ[i] = Qvalue->cpu_data()[i * 4 + j];
			}
		}
	}

	// 再びinputにデータをmini_batch_size分入れる
	for(int i = 0; i < mini_batch_size; ++i) {
		input[i * 2] = reps[i]->currentPosition.first;
		input[i * 2 + 1] = reps[i]->currentPosition.second;
	}

	// 教師データをlabelにいれる
	// labelのaction番目には reward + gamma * maxQを, それ以外には 0 を入れる
	// filterもaction番目には 1 を, それ以外には 0 を入れる
	for(int i = 0; i < mini_batch_size; ++i) {
		for(int j = 0; j < 4; ++j) {
			if(reps[i]->action_index == j) {
				label[i * 4 + j] = reps[i]->reward + GAMMA * maxQ[i];
				filter[i * 4 + j] = 1;
			}
			else {
				label[i * 4 + j] = 0;
				filter[i * 4 + j] = 0;
			}
		}
	}

	// 学習
	solver->Step(1);
	cout << "------------------------------------------------------------------"
			"--"
			"--------"
		 << endl;
	cout << "iter = " << solver->iter() << endl;
	cout << endl;
	// チェック
	for(int i = 0; i < mini_batch_size; i++) {
		cout << "batch id = " << i << endl;
		cout << "input : " << reps[i]->currentPosition.first << ", "
			 << reps[i]->currentPosition.second << endl;
		cout << "labelQvalue : ";
		for(int j = 0; j < 4; ++j) {
			cout << label[i * 4 + j] << " \n"[j == 3];
		}
		cout << "outputQvalue : ";
		for(int j = 0; j < 4; ++j) {
			cout << Qvalue->cpu_data()[i * 4 + j] << " \n"[j == 3];
		}
		/*
    cout << "accuracy :";
    for (int j = 0; )
    cout << endl;
    */
	}
	cout << "loss : " << loss->cpu_data()[0] << endl;

	loss_average += loss->cpu_data()[0];

	if(solver->iter() % 1000 == 0) {
		//cout << "loss average = " << loss_average / 1000 << endl;
    // 損失関数値
    std::ofstream ofs("train.txt", std::ios::app); // std::ios::appとすると, ファイルに追加で書き込むことができる
    ofs << "iter, loss_average: " << solver->iter() << " " << loss->cpu_data()[0] << endl;
    ofs.close();
		loss_average = 0.0;
	}
	cout << "--------------------------------------------------------------------------" << endl;

	// 一定回数ごとにスナップショットを作成
	if(solver->iter() % 10000 == 0) {
		solver->Snapshot();
	}

	return;
}

// 本体
void Q_Learning() {
	std::random_device rnd;
	std::mt19937 mt(rnd());
	std::uniform_real_distribution<> ep(0.0, 1.0);

	std::uniform_int_distribution<> rndm(0, 3);
	Init(FLAGS_solver_state);

	for(int episode = 0; episode < 50000; ++episode) {
		cout << "EPISODE " << episode << endl;
		int px = rndm(mt), py = rndm(mt);
		int cnt = 0;
		// ofs << "start : (" << px << " " << py << ")" << endl;

		cerr << "start : (" << px << " " << py << ")" << endl;

		while(true) {
			// game end
			if(px == py && (px == 0 || py == 3)) {
				cerr << "EPISODE " << episode << " END" << endl;
				// ofs << "EPISODE " << episode << " END" << endl;
				// ofs << "end : move_count = " << cnt << endl;
				// ofs << endl;
				break;
			}

			cnt++;

			double maxQ = -1e9;
			int action_index = 0;
			double epsilon = ep(mt);
			double EPSILON = 0.1; //(episode < 50000 ? 0.3 : 0.1);

			// バッチサイズ1
			setBatchSize(1);

			// なんかこうするらしい
			for(int i = 0; i < 4; ++i) {
				filter[i] = 1;
			}

			// ε-Greedy

			// ランダムに選択
			// ここは大丈夫
			if(epsilon < EPSILON) {
				action_index = rndm(mt);
				cout << "random action = " << dir_s[action_index] << endl;
			}
			// 最もQ値の高い行動を選択する
			else {
				// ここらへんでForwardする
				input[0] = px, input[1] = py;
				net->Forward();

				// 4つの行動の中から, 最もQ値の高い行動を選択する
				for(int i = 0; i < 4; ++i) {
					if(maxQ < Qvalue->cpu_data()[i]) {
						maxQ = Qvalue->cpu_data()[i];
						action_index = i;
					}
				}

				cout << "best action = " << dir_s[action_index] << endl;
			}

			int nx = px + vx[action_index], ny = py + vy[action_index];

			if(nx < 0 || nx > 3 || ny < 0 || ny > 3) {
				nx = px, ny = py;
			}

			// データ追加
			AddReplay({px, py}, action_index, {nx, ny}, -1);

			// 更新
			Update();

			// 移動
			cerr << "episode = " << episode << endl;
			cerr << "(" << px << ", " << py << ") ===> (" << nx << ", " << ny << ")"
				 << endl;

			// 最後に現在の状態の更新
			px = nx, py = ny;
		}
	}
}

// for check
void Output() {}

int main() {

	Q_Learning();

	return 0;
}
