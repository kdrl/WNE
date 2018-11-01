OBJS = main.o skipgram.o 
CXX = g++
CXXFLAGS = --std=c++11 -Wall -Wno-sign-compare -Wno-unknown-pragmas -fPIC -fopenmp -O3 -pthread

all: main

main : $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o main

main.o : main.cpp cmdline.h skipgram.h
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

skipgram.o : cheaprand.h skipgram.h skipgram.cpp
	$(CXX) $(CXXFLAGS) -c skipgram.cpp -o skipgram.o

clean:
	rm -f -r ./*.o main
