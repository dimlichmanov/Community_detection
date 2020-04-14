gather:
	/usr/local/cuda-10.1/bin/nvcc CSR_GRAPH.cpp device_gather.cu generator.cpp main.cu -Xcompiler -fopenmp -o prog
lp: content
	nvcc -std=c++11 --expt-extended-lambda CSR_GRAPH.cpp device_gather.cu lp.cpp generator.cpp main.cu -Xcompiler -fopenmp -o main
download:
	mkdir moderngpu
content: download
	git clone https://www.github.com/moderngpu/moderngpu ./moderngpu