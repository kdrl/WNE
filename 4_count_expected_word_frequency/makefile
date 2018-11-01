OBJS = main.o counting_word.o 
CXX = g++
CXXFLAGS = --std=c++11 -Wall -Wno-sign-compare -Wno-unknown-pragmas -fPIC -fopenmp -O3 -pthread -lhdf5 -lhdf5_cpp

all: main

main : $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o main

main.o : main.cpp cmdline.h counting_word.h
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

counting_word.o : counting_word.h counting_word.cpp
	$(CXX) $(CXXFLAGS) -c counting_word.cpp -o counting_word.o

clean:
	rm -f -r ./*.o main
