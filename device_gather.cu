

//#include "/usr/local/cuda-10.1/include/cuda_runtime.h"
#include "cuda_runtime.h"
#include "device_gather.h"





__global__ void device_gather(unsigned int *v_array,unsigned int *e_array,unsigned int *dest_labels ,unsigned int *labels) {

    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;

    int v_begin = v_array[i];
    int v_end = v_array[i+1];

    for (int j = v_begin; j < v_end; ++j) {
        dest_labels[j] = labels[e_array[j]];
    }
}

void CSR_GRAPH::move_to_device(void) {


    cudaMalloc((unsigned**)&dev_v_array,(size_t)vertices_count);
    cudaMalloc((unsigned**)&dev_e_array,(size_t)edges_count);
    if(weighted){
        cudaMalloc((float**)&dev_weigths,(size_t)edges_count);
    }
    cudaMalloc((unsigned**)&dev_labels,(size_t)edges_count);
    cudaMalloc((unsigned**)&dev_dest_labels,(size_t)edges_count);

    cudaMemcpy(dev_v_array,v_array,(size_t)vertices_count,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_e_array,e_array,(size_t)edges_count,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_weigths,weigths,(size_t)edges_count,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_labels,labels,(size_t)vertices_count,cudaMemcpyHostToDevice);
}


void CSR_GRAPH::move_to_host (void) {

    cudaFree(dev_v_array);
    cudaFree(dev_e_array);
    if(weighted){
        cudaFree(dev_weigths);
    }

    cudaFree(dev_labels);
    cudaFree(dev_dest_labels);

    cudaMemcpy(v_array,dev_v_array,(size_t)vertices_count,cudaMemcpyDeviceToHost);
    cudaMemcpy(e_array,dev_e_array,(size_t)edges_count,cudaMemcpyDeviceToHost);
    cudaMemcpy(weigths,dev_weigths,(size_t)edges_count,cudaMemcpyDeviceToHost);
    cudaMemcpy(labels,dev_labels,(size_t)vertices_count,cudaMemcpyDeviceToHost);
    cudaMemcpy(dest_labels,dev_dest_labels,(size_t)edges_count,cudaMemcpyDeviceToHost);
}
