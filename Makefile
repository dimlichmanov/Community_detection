all:
	 nvcc CSR_GRAPH.cpp device_gather.cu generator.cpp main.cu -Xcompiler -fopenmp -o prog 
	./prog 18 3 ur
