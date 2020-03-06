

//#include "/usr/local/cuda-10.1/include/cuda_runtime.h"
#include "cuda_runtime.h"
#include "device_gather.h"
#include <iostream>
#include <stdio.h>
#include <assert.h>

#define SAFE_CALL( CallInstruction ) { \
    cudaError_t cuerr = CallInstruction; \
    if(cuerr != cudaSuccess) { \
         printf("CUDA error: %s at call \"" #CallInstruction "\"\n", cudaGetErrorString(cuerr)); \
		 throw "error in CUDA API function, aborting..."; \
    } \
}

#define SAFE_KERNEL_CALL( KernelCallInstruction ){ \
    KernelCallInstruction; \
    cudaError_t cuerr = cudaGetLastError(); \
    if(cuerr != cudaSuccess) { \
        printf("CUDA error in kernel launch: %s at kernel \"" #KernelCallInstruction "\"\n", cudaGetErrorString(cuerr)); \
		throw "error in CUDA kernel launch, aborting..."; \
    } \
    cuerr = cudaDeviceSynchronize(); \
    if(cuerr != cudaSuccess) { \
        printf("CUDA error in kernel execution: %s at kernel \"" #KernelCallInstruction "\"\n", cudaGetErrorString(cuerr)); \
		throw "error in CUDA kernel execution, aborting..."; \
    } \
}



__global__ void device_gather(unsigned int *v_array,unsigned int *e_array,unsigned int *dest_labels ,unsigned int *labels,
                              unsigned long  long edges, unsigned long long vertices) {

    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;

    assert(i < vertices+1);

    int v_begin = v_array[i];
    int v_end = v_array[i+1];


    for (int j = v_begin; j < v_end; ++j) {
        assert(e_array[j] < edges);
        dest_labels[j] = labels[e_array[j]];
    }
}



void CSR_GRAPH::move_to_device(void) {

    SAFE_CALL((cudaMalloc((void**)&dev_v_array,(size_t)sizeof(this->v_array[0])*(vertices_count+1))));

    SAFE_CALL((cudaMalloc((void**)&dev_e_array,(size_t)sizeof(this->e_array[0])*edges_count)));

    if(weighted){
        SAFE_CALL((cudaMalloc((void**)&dev_weigths,(size_t)sizeof(this->e_array[0])*edges_count)));
    }

    SAFE_CALL((cudaMalloc((void**)&dev_labels,(size_t)sizeof(this->v_array[0])*(vertices_count))));

    SAFE_CALL((cudaMalloc((void**)&dev_dest_labels,(size_t)sizeof(this->e_array[0])*edges_count)));


    SAFE_CALL((cudaMemcpy(dev_dest_labels,dest_labels,(size_t)(sizeof(this->e_array[0])*edges_count),cudaMemcpyHostToDevice)));
    SAFE_CALL((cudaMemcpy(dev_v_array,v_array,(size_t)((vertices_count+1)* sizeof(this->v_array[0])),cudaMemcpyHostToDevice)));
    SAFE_CALL((cudaMemcpy(dev_e_array,e_array,(size_t)(sizeof(this->e_array[0])*edges_count),cudaMemcpyHostToDevice)));
    SAFE_CALL((cudaMemcpy(dev_weigths,weigths,(size_t)(sizeof(this->e_array[0])*edges_count),cudaMemcpyHostToDevice)));
    SAFE_CALL((cudaMemcpy(dev_labels,labels,(size_t)(sizeof(this->v_array[0])*(vertices_count)),cudaMemcpyHostToDevice)));

    std::cout<<"moved to device"<<std::endl;

}


void CSR_GRAPH::move_to_host (void) {

    SAFE_CALL((cudaMemcpy(v_array,dev_v_array,(size_t)(vertices_count+1)* sizeof(this->v_array[0]),cudaMemcpyDeviceToHost)));
    SAFE_CALL((cudaMemcpy(e_array,dev_e_array,(size_t)edges_count* sizeof(this->e_array[0]),cudaMemcpyDeviceToHost)));
    SAFE_CALL((cudaMemcpy(weigths,dev_weigths,(size_t)edges_count* sizeof(this->e_array[0]),cudaMemcpyDeviceToHost)));
    SAFE_CALL((cudaMemcpy(labels,dev_labels,(size_t)(vertices_count)* sizeof(this->v_array[0]),cudaMemcpyDeviceToHost)));
    SAFE_CALL((cudaMemcpy(dest_labels,dev_dest_labels,(size_t)edges_count* sizeof(this->e_array[0]),cudaMemcpyDeviceToHost)));

    cudaFree(dev_v_array);
    cudaFree(dev_e_array);
    if(weighted){
        cudaFree(dev_weigths);
    }
    cudaFree(dev_labels);
    cudaFree(dev_dest_labels);
    std::cout<<"moved back"<<std::endl;

}
