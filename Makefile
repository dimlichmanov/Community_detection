gather:
	/usr/local/cuda-10.1/bin/nvcc CSR_GRAPH.cpp device_gather.cu generator.cpp main.cu -Xcompiler -fopenmp -o prog

lp:
	nvcc CSR_GRAPH.cpp lp.cpp device_gather.cu generator.cpp main.cu -Xcompiler -fopenmp -o prog
