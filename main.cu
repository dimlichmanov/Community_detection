#include <omp.h>
#include <iostream>
#include <string.h>
#include <fstream>
#include <algorithm>
#include "stdlib.h"
#include "CSR_GRAPH.h"
#include <stdio.h>
#include <math.h>
#include <vector>

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


#include "CSR_GRAPH.h"
#include "generator.h"
#include "device_gather.h"

//#include "/usr/local/cuda-10.1/include/cuda_runtime.h"
#include "cuda_runtime.h"
//#include "/usr/local/cuda-10.1/include/cuda_profiler_api.h"
#include "cuda_profiler_api.h"

using namespace std;


#ifndef uint32_t
#define uint32_t int
#endif


int main(int argc, char **argv) {
    try {

        cudaEvent_t start,stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int threads = omp_get_max_threads();
        int vertices_index = atoi(argv[1]);
        int density_degree = atoi(argv[2]);
        char *graph_type = argv[3];
        //double begin,end;

        unsigned int vertices_count =  pow(2.0, vertices_index);
        unsigned int edges_count = density_degree * vertices_count;
        unsigned int *src_ids = new unsigned int[edges_count];
        unsigned int *dst_ids = new unsigned int[edges_count];
        float *weights = new float[edges_count];


        if (strcmp(graph_type, "rmat") == 0) {

            R_MAT(src_ids, dst_ids, weights, vertices_count, edges_count, 45, 20, 20, 15, threads, true, true);

        } else {

            uniform_random(src_ids, dst_ids, weights, vertices_count, edges_count, threads, true, true);
        }

        //for (int i = 0; i < edges_count; i++) {
        //    cout << src_ids[i] << "----" << dst_ids[i] << endl;
        //
        //}

       CSR_GRAPH a(vertices_count,edges_count,src_ids,dst_ids,weights, true);


        //a.print_CSR_format();
        //a.print_adj_format();
        //a.adj_distribution(edges_count);


        unsigned int* labels = new unsigned int[vertices_count];
        unsigned int* dest_labels = new unsigned int[edges_count];
        unsigned int* dev_labels;
        unsigned int* dev_dest_labels;
        SAFE_CALL((cudaMalloc((void**)&dev_labels,(size_t)(sizeof(unsigned int))*(vertices_count))));
        SAFE_CALL((cudaMalloc((void**)&dev_dest_labels,(size_t)(sizeof(unsigned int))*edges_count)));

        generate_labels(threads,vertices_count,labels);


        a.move_to_device(dest_labels, labels, dev_dest_labels ,dev_labels);

        SAFE_CALL(cudaEventRecord(start));


        dim3 block(1024,1);
        dim3 grid(vertices_count*32/block.x,1);
        //dim3 block(16,1);
        //dim3 grid(1,1);


        printf("starting...");
        SAFE_KERNEL_CALL((gather_warp_per_vertex<<<grid,block>>> (a.get_dev_v_array(),a.get_dev_e_array(),dev_dest_labels,dev_labels,edges_count,vertices_count)));
        printf("terminating....");
        SAFE_CALL(cudaEventRecord(stop));
        SAFE_CALL(cudaEventSynchronize(stop));
        float time;
        SAFE_CALL(cudaEventElapsedTime(&time,start,stop));
        time*=1000000;
        a.move_to_host(dest_labels, labels, dev_dest_labels ,dev_labels);
        SAFE_CALL(cudaFree(dev_labels));
        SAFE_CALL(cudaFree(dev_dest_labels));

        //if (a.check() == 0){
        //    printf("CORRECT");
        //}

        printf("Bandwidth for 2^%d vertices and 2^%d edges is %f GB/s\n ", vertices_index,vertices_index + (int) log2((double)density_degree) , sizeof(unsigned int)*(2*vertices_count + 2*edges_count)/(time));


        /*begin = omp_get_wtime();
        a.form_label_array(threads);
        end = omp_get_wtime();
        //a.print_label_info(threads);
        printf("Time for 2^%d edges is %f\n ", vertices_index + (int) log2(density_degree) ,end - begin);

        begin = omp_get_wtime();
        a.form_label_array(threads);
        end = omp_get_wtime();
        //a.print_label_info(threads);
        printf("Time for 2^%d edges is %f\n ", vertices_index + (int) log2(density_degree) ,end - begin);
*/

        //a.print_label_info(threads);
        delete[] src_ids;
        delete[] dst_ids;
        delete[] weights;

    }
    catch (const char *error) {
        cout << error << endl;
        getchar();
        return 1;
    }
    catch (...) {
        cout << "unknown error" << endl;
    }

    SAFE_CALL(cudaProfilerStop());
    return 0;
}