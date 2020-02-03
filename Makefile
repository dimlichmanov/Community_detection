all:
	g++ *.cpp -fopenmp -o prog
	./prog 18 32 ur
