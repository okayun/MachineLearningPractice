g++ -o Main Main.cpp GridWorldPlayer.cpp GridWorldGame.cpp GridWorldTypes.cpp -std=c++11 -O2 \
-I $HOME/caffe/include \
-L $HOME/caffe/build/lib \
-lcaffe -lglog -lgflags -lboost_system \
-DCPU_ONLY --exec-charset=utf-8 --input-charset=utf-8 -ggdb `pkg-config --cflags opencv` $@ `pkg-config --libs opencv`
