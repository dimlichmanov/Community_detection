all:
	g++ *.cpp -fopenmp -o prog
	./prog 16 32 ur
