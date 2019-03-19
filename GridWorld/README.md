# 格子世界
これは「強化学習」の p97 にある例題で扱っている, 4 x 4 の格子世界をプログラムで簡単に扱えるように再現したものです.  

# 動作環境
C++11と14で動作確認済です. また, OpenCV ライブラリが必要です.

## ファイル説明
ファイルは以下の通り.
- build.sh : コンパイルするためのスクリプト.
- Main.cpp : GridWorldとGridWorldPlayerのやりとりを記述する.
- GridWorldConsts.h : ゲームに必要な定数を定義. 格子のサイズやスキップフレーム数を変更したい場合はここをいじる.
- GridWorldTypes.h : 状態を表す構造体や行動, 報酬などを定義.
- GridWorldTypes.cpp : 上の具体的な中身を記述.
- GridWorldPlayer.h : プレイヤを定義している. 具体的な処理はcppファイルに書く.
- GridWorldPlayer.cpp : 上と同じ.
- GridWorldGame.h : ゲーム本体を定義している. 基本的に弄る必要はないはず.
- GridWorldGame.cpp : 上と同じ.


# 使用方法
1. ここにあるファイルをダウンロード.
2. 必要なライブラリを導入し, プレイヤの処理を実装した後, `sh build.sh`でコンパイル.
3. `./Main`でゲーム実行 (`./Main -episodes n`とすれば, n回エピソードを繰り返す)

# ゲームルール
- 状態は `kHeight` * `kWidth` 個ある
- 終端状態は一番左上のマス (0, 0) と 一番右下のマス (kHeight - 1, kWidth - 1)
- 行動は上下左右いずれかのマスへ1マス移動 (つまり 4 つ)
- 報酬は, 1 回行動する毎に -1
- 格子から外れるような行動を取る場合, そのマスに留まる
- 初期状態はランダムに選択 (初期状態を指定することも可)

# 実装上の注意
- 各行動は整数で表されます.
  - 下 : 0 (h, w) -> (h + 1, w)
  - 右 : 1 (h, w) -> (h, w + 1)
  - 上 : 2 (h, w) -> (h - 1, w)
  - 左 : 3 (h, w) -> (h, w - 1)
- ゲームとプレイヤのやり取りは以下のメソッドを使用することで可能 (一応`GridWorldGame.h`も確認してね(汗))
  - `GridWorld::GridWorld()` : コンストラクタ. `kHeight * kWidth`の格子世界を生成する.
  - `GridWorld::InitState()` : 初期状態をランダムに決定する.
  - `GridWorld::GetState()` : 現在の状態 (座標) を取得する.
  - `GridWorld::UpdateState(a)` : 与えられた行動`a`を行い, 状態を更新する.
  - `GridWorld::GetLastReward()` : 最後の行動に対する報酬を返す.
  - `GridWorld::IsEnd()` : ゲームが終了したかどうかを判定する.
  - `GridWorld::SetImageChannel(c)` : `GetGridImage`で返す画像のチャネル数を`c`に変更する. デフォルトのチャネル数は 1.
- 基本的に `Main.cpp` と `GridWorldPlayer.*` の中身を実装すれば良いはずです. バグや仕様の欠陥などあれば気軽に言ってください.
- (再) 格子世界の大きさやスキップフレーム数を変更したい場合は, `GridWorldConsts.h` の `kHeight`, `kWidth`, `kSkipFrame` を変更してください.
