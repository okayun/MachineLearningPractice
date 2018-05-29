g++ main.cpp -std=c++11 -g \
-I $HOME/caffe/include \
-L $HOME/caffe/build/lib \
-lcaffe -lglog -lgflags -lboost_system \
-DCPU_ONLY -lopencv_highgui -lopencv_core -lopencv_imgproc
