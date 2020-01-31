all:
	g++ *.cpp -fopenmp -o prog
	./prog 4 2 rmat
