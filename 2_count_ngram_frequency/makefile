OBJS = main.o lossycounting.o
CXX = g++
CXXFLAGS = --std=c++11 -Wall -Wno-sign-compare -Wno-unknown-pragmas -fPIC -fopenmp -O3 -pthread

all: main

main : $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o main

main.o : main.cpp lossycounting.h cmdline.h
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

lossycounting.o : lossycounting.h lossycounting.cpp
	$(CXX) $(CXXFLAGS) -c lossycounting.cpp -o lossycounting.o

clean:
	rm -f -r ./*.o main
