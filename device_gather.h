
//#include "/usr/local/cuda-10.1/include/cuda_runtime.h"




#ifndef RMAT_DEVICE_GATHER_H


#define RMAT_DEVICE_GATHER_H

#include "CSR_GRAPH.h"
#include "cuda_runtime.h"


__global__ void gather_warp_per_vertex(  int *v_array,  int *e_array,  int *dest_labels ,  int *labels,   long long v,
          long long t) ;

#endif //RMAT_DEVICE_GATHER_H