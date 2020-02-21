//
// Created by dimon on 12.02.20.
//

#include "CSR_GRAPH.h"
#include "cuda_runtime.h"
//#include "/usr/local/cuda-10.1/include/cuda_runtime.h"




#ifndef RMAT_DEVICE_GATHER_H
#define RMAT_DEVICE_GATHER_H
#endif //RMAT_DEVICE_GATHER_H



__global__ void device_gather(unsigned int *v_array,unsigned int *e_array,unsigned int *dest_labels ,unsigned int *labels) ;