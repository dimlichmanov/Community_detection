#include "./moderngpu/kernel_segsort.hxx"
#include "./moderngpu/memory.hxx"
#include "./moderngpu/kernel_scan.hxx"
#include "./moderngpu/kernel_segreduce.hxx"

//#include "./moderngpu/src/moderngpu/kernel_segsort.hxx"
//#include "./moderngpu/src/moderngpu/memory.hxx"
//#include "./moderngpu/src/moderngpu/kernel_segreduce.hxx"
//#include "./moderngpu/src/moderngpu/kernel_scan.hxx"
//
//#include "/usr/local/cuda-10.1/include/cuda_runtime.h"
//#include "/usr/local/cuda-10.1/include/cuda_profiler_api.h"
#include <omp.h>
#include <iostream>
#include <string.h>
#include <fstream>
#include <algorithm>
#include "stdlib.h"
#include "CSR_GRAPH.h"
#include <stdio.h>
#include <math.h>
#include <sstream>
#include <string>
#include <vector>
#include "lp.h"
#include "CSR_GRAPH.h"
#include "generator.h"
#include "device_gather.h"
#include "cuda_runtime.h"
#include "cuda_profiler_api.h"
#include "map"


#define SAFE_CALL(CallInstruction) { \
    cudaError_t cuerr = CallInstruction; \
    if(cuerr != cudaSuccess) { \
         printf("CUDA error: %s at call \"" #CallInstruction "\"\n", cudaGetErrorString(cuerr)); \
         throw "error in CUDA API function, aborting..."; \
    } \
}

#define SAFE_KERNEL_CALL(KernelCallInstruction){ \
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


void debug_info(std::vector<int> &ptr, string info) {
    cout << info << endl;
    for (int i = 0; i < ptr.size(); i++) {
        std::cout << ptr[i] << " ";
    }
    cout << endl;
}

void debug_info(int *ptr, int size_n, string info) {
    cout << info << endl;
    for (int i = 0; i < size_n; i++) {
        std::cout << ptr[i] << " ";
    }
    cout << endl;
}

//void debug_info(int *ptr, int size_n, string info) {
//    cout << info << endl;
//    for (int i = 0; i < size_n; i++) {
//        std::cout << ptr[i] << " ";
//    }
//    cout << endl;
//}


void debug_info(bool *ptr, int size_n, string info) {
    cout << info << endl;
    for (int i = 0; i < size_n; i++) {
        std::cout << ptr[i] << " ";
    }
    cout << endl;
}

void debug_info(short *ptr, int size_n, string info) {
    cout << info << endl;
    for (int i = 0; i < size_n; i++) {
        std::cout << ptr[i] << " ";
    }
    cout << endl;
}

void print_bounds(std::vector<int> &ptr, int edges_count) {
    for (int i = 0; i < ptr.size(); i++) {
        if (i == 0) {
            //std::cout << "[ " << 0 << " , " << ptr[0] - 1 << " ]" << std::endl;
            std::cout << "[ " << ptr[i] << " , " << ptr[i + 1] - 1 << " ]" << std::endl;
            continue;
        }
        if (i == ptr.size() - 1) {
            std::cout << "[ " << ptr[ptr.size() - 1] << " , " << edges_count - 1 << " ]"
                      << std::endl;;
            continue;
        }
        std::cout << "[ " << ptr[i] << " , " << ptr[i + 1] - 1 << " ]" << std::endl;
    }
}


void label_stats(int *labels, int vertices_count) { // Почему то в map много нулей
    std::map<int, int> mp;
    for (int i = 0; i < vertices_count; i++) {
        if (mp.count(labels[i])) {
            mp[labels[i]]++;
        } else {
            mp[labels[i]] = 1;
        }
    }
    std::map<int, int> components;
    for (auto it = mp.begin(); it != mp.end(); it++) {
        if (components.count(it->second)) {
            components[it->second]++;
        } else {
            components[it->second] = 1;
        }
    }
    for (auto it = components.begin(); it != components.end(); it++) {
        if (it->first != 0) {
            cout << "there are " << it->second << " components of size " << it->first << endl;
        }
    }
}


void input(char *filename, bool directed, int *&src_ids, int *&dst_ids, int &vertices_count,
           int &edges_count) {
    int max_vertice = 0;

    std::ifstream infile(filename);
    std::string line;
    int i = 0;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        int a, b;
        if (!(iss >> a >> b)) {
            break;
        } else {
            if (max(a, b) > max_vertice) {
                max_vertice = (unsigned)
                max(a, b);
            }
        }
        i++;
    }
    edges_count = i;
    vertices_count = max_vertice;
    src_ids = new int[edges_count];
    dst_ids = new int[edges_count];

    std::ifstream infile1(filename);
    i = 0;
    while (std::getline(infile1, line)) {
        std::istringstream iss(line);
        int a, b;
        if (!(iss >> a >> b)) {
            break;
        } else {
            src_ids[i] = a;
            dst_ids[i] = b;
            if (!directed) {
                src_ids[i + 1] = b;
                dst_ids[i + 1] = a;
            }
        }
        i++;
    }
}

__global__ void extract_boundaries_initial(int *boundaries, int *v_array, int edges_count) {

    long int i = threadIdx.x + blockIdx.x * blockDim.x;
    long int position = v_array[i];
    if (i != 0) {
        boundaries[position - 1] = 1;
    } else {
        boundaries[edges_count - 1] = 1;
    }
}

__global__ void extract_boundaries_optional(int *boundaries, int *dest_labels, int edges_count) {
    long int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ((boundaries[i] != 1) && (i < edges_count)) {
        if (dest_labels[i] != dest_labels[i + 1]) {
            boundaries[i] = 1;
        }
    }
}

__global__ void count_labels(int *scanned_array, int edges_count, int *S_array) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ((i < edges_count - 1) && (scanned_array[i + 1] != scanned_array[i])) {
        S_array[scanned_array[i]] = i;
    }
}

__global__ void new_boundaries(int *scanned_array, int *v_array, int edges_count, int *S_ptr) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    S_ptr[i] = scanned_array[v_array[i]];
}


__global__ void frequency_count(int *W_array, int *S) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ((i > 0) && (S[i] != 0)) {
        W_array[i] = S[i] - S[i - 1];
    } else {
        W_array[0] = S[0] + 1;
    }
}

__global__ void get_labels(int *I, int *S, int *L, int *labels) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    labels[i] = L[S[I[i]]];
}

__global__ void print_scanned_array(int *scanned, int edges_count) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    printf("%d\n", scanned[i]);
    if (i == edges_count - 1) {
        printf("THE FINAL VALUE is %d\n ", scanned[i]);
    }
}

__global__ void fill_indices(int *I) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    I[i] = i;

}

using namespace std;


int main(int argc, char **argv) {
    try {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float time;
        int threads = omp_get_max_threads();
        int vertices_index;
        int density_degree;
        bool check_flag = false;
        bool test_flag = false;
        char *graph_type;
        bool lp_flag = false;
        bool gather_flag = false;
        char *test_file = NULL;
        for (int i = 1; i < argc; i++) {
            string option(argv[i]);

            if ((option.compare("-scale") == 0) || (option.compare("-s") == 0)) {
                vertices_index = atoi(argv[++i]);
            }

            if ((option.compare("-edges") == 0) || (option.compare("-e") == 0)) {
                density_degree = atoi(argv[++i]);
            }

            if ((option.compare("-check") == 0)) {
                check_flag = true;
            }

            if ((option.compare("-nocheck") == 0)) {
                check_flag = false;
            }
            if ((option.compare("-type") == 0)) {
                graph_type = argv[++i];
            }
            if ((option.compare("-testing") == 0)) {
                test_file = argv[++i];
                test_flag = true;
                cout << "FLAG FOUND" << endl;
            }
            if ((option.compare("-lp")) == 0) {
                lp_flag = true;
            }
            if ((option.compare("-gather")) == 0) {
                gather_flag = true;
            }

        }

        int vertices_count = pow(2.0, vertices_index);
        int edges_count = density_degree * vertices_count;
        int *src_ids = NULL;
        int *dst_ids = NULL;
        float *weights = new float[edges_count];

        if (!test_flag) {
            src_ids = new int[edges_count];
            dst_ids = new int[edges_count];
            cout << "test_flag" << endl;
            if (strcmp(graph_type, "rmat") == 0) {
                cout << "RMAT";
                R_MAT(src_ids, dst_ids, weights, vertices_count, edges_count, 45, 20, 20, 15, threads, true, true);

            } else {
                cout << "UR_GEN" << endl;
                uniform_random(src_ids, dst_ids, weights, vertices_count, edges_count, threads, true, true);
                cout << "Generated_UR" << endl;
            }
        } else {
            cout << test_flag << endl;
            cout << "file_init" << endl;
            input(test_file, true, src_ids, dst_ids, vertices_count, edges_count);
            vertices_count++;
            cout << "vertices:" << vertices_count << endl;
            cout << "edges: " << edges_count << endl;
        }

        cout << endl;
        CSR_GRAPH a(vertices_count, edges_count, src_ids, dst_ids, weights, true);
        //a.save_to_graphviz_file("graph_pred", NULL);
        //a.print_CSR_format();

        int *labels = new int[vertices_count];
        for (int j = 0; j < vertices_count; j++) {
            labels[j] = j;
        }
        int *dest_labels = new int[edges_count];
        int *dev_labels;
        int *dev_dest_labels;
        int *values;
        int *s_ptr_array;
        int *F_mem;
        int *F_scanned;

        if (gather_flag) {

            SAFE_CALL((cudaMalloc((void **) &dev_labels, (size_t) (sizeof(int)) * (vertices_count))));
            SAFE_CALL((cudaMalloc((void **) &dev_dest_labels, (size_t) (sizeof(int)) * edges_count)));
            SAFE_CALL((cudaMalloc((void **) &F_mem, (size_t) (sizeof(int)) * edges_count)));
            SAFE_CALL((cudaMalloc((void **) &values, (size_t) (sizeof(int)) * edges_count)));
            //mgpu::mem_t<int> s_ptr_array(vertices_count, context);

            SAFE_CALL((cudaMalloc((void **) &s_ptr_array, (size_t) (sizeof(int)) * vertices_count)));
            SAFE_CALL((cudaMalloc((void **) &F_scanned, (size_t) (sizeof(int)) * edges_count)));

            a.move_to_device(dest_labels, labels, dev_dest_labels, dev_labels);
            cout << 1 << endl;
            int iter = 0;
            mgpu::standard_context_t context;

//            std::vector<int> ptr; //Bounds as segments
//            for (int k = 0; k < vertices_count; k++) {
//                ptr.push_back(a.get_v_array()[k]);
//            }
//
//            cout<<2<<endl;
//            mgpu::mem_t<int> segs = mgpu::to_mem(ptr, context); //


            mgpu::mem_t<int> out(vertices_count, context);
            //mgpu::mem_t<int> values(edges_count, context);
            mgpu::mem_t<int> I_mem(edges_count, context);
            //mgpu::mem_t<int> F_scanned(edges_count, context);
            {
                dim3 block(1024, 1);
                dim3 grid(edges_count / block.x, 1); //only for test
                SAFE_KERNEL_CALL((fill_indices << < grid, block >> > (I_mem.data())));
            }
            cudaDeviceSynchronize();

            do {

                SAFE_CALL(cudaEventRecord(start));
                {
                    //Change configuration after
                    dim3 block(1024, 1);
                    dim3 grid(32 * vertices_count / block.x, 1); //only for test
                    SAFE_KERNEL_CALL((gather_warp_per_vertex << < grid, block >> >
                                                                        (a.get_dev_v_array(), a.get_dev_e_array(), dev_dest_labels, dev_labels, edges_count, vertices_count)));
                }
                SAFE_CALL(cudaEventRecord(stop));
                SAFE_CALL(cudaEventSynchronize(stop));
                SAFE_CALL(cudaEventElapsedTime(&time, start, stop));
                time *= 1000000;
                cout << "TEPS for gather" << edges_count / time << endl;
                //a.move_to_host(dest_labels, labels, dev_dest_labels, dev_labels);
                if (check_flag) {
                    int *test_dest_labels = new int[edges_count];
                    form_label_array(threads, vertices_count, edges_count, test_dest_labels, a.get_dev_v_array(),
                                     labels,
                                     a.get_e_array());
                    int flag = check(edges_count, dest_labels, test_dest_labels);
                    if (flag == 0) {
                        printf("CORRECT");
                    }
                    delete[] test_dest_labels;
                }


//                printf("GATHER Bandwidth for 2^%d vertices and 2^%d edges is %f GB/s\n ", vertices_index,
//                       vertices_index + (int) log2((double) density_degree),
//                       sizeof(  int) * (2 * vertices_count + 2 * edges_count) / (time));





                SAFE_CALL(cudaEventRecord(start));
                mgpu::segmented_sort(dev_dest_labels, values, edges_count, a.get_dev_v_array(), vertices_count,
                                     mgpu::less_t<int>(), context);
                SAFE_CALL(cudaEventRecord(stop));
                SAFE_CALL(cudaEventSynchronize(stop));
                SAFE_CALL(cudaEventElapsedTime(&time, start, stop));
                time *= 1000000;
                cout << "TEPS for segsort" << edges_count / time << endl;


                SAFE_CALL(cudaEventRecord(start));
                SAFE_CALL(
                        (cudaMemset(F_mem, 0, (size_t) (sizeof(int)) * edges_count))); //was taken from group of memcpy

                {
                    dim3 block(1024, 1);
                    dim3 grid(vertices_count / block.x, 1);

                    SAFE_KERNEL_CALL(
                            (extract_boundaries_initial << < grid, block >> >
                                                                   (F_mem, a.get_dev_v_array(), edges_count))); //fill 1 in bounds
                }
                {
                    dim3 block(1024, 1);
                    dim3 grid(edges_count / block.x, 1);

                    SAFE_KERNEL_CALL(
                            (extract_boundaries_optional << < grid, block >> >
                                                                    (F_mem, dev_dest_labels, edges_count))); //sub(i+1, i)
                }


                mgpu::scan(F_mem, edges_count, F_scanned, context); // may not work because of bool


                int reduced_size = 0;

                int *scanned_data_ptr = F_scanned;

//                {
//                    dim3 block(1024, 1);
//                    dim3 grid(edges_count / block.x, 1);
//
//                    SAFE_KERNEL_CALL((print_scanned_array<< < grid, block >> >(scanned_data_ptr, edges_count)));
//                }


                cudaMemcpy(&reduced_size, scanned_data_ptr + (edges_count - 1), sizeof(int), cudaMemcpyDeviceToHost);

                mgpu::mem_t<int> s_array(reduced_size, context);

                {
                    dim3 block(1024, 1);
                    dim3 grid(edges_count / block.x, 1);
                    SAFE_KERNEL_CALL(
                            (count_labels << < grid, block >> > (F_scanned, edges_count, s_array.data())));
                }


                {
                    dim3 block(1024, 1);
                    dim3 grid(vertices_count / block.x, 1);
                    SAFE_KERNEL_CALL((new_boundaries << < grid, block >> >
                                                                (F_scanned, a.get_dev_v_array(), edges_count, s_ptr_array)));
                }

                mgpu::mem_t<int> w_array(reduced_size, context);
                {
                    dim3 block(1024, 1);
                    dim3 grid(reduced_size / block.x, 1);


                    SAFE_KERNEL_CALL((frequency_count << < grid, block >> > (w_array.data(), s_array.data())));
                }


                int init = 0;


                int *w_ptr = w_array.data();

                auto my_cool_lambda =[w_ptr] MGPU_DEVICE(int
                a, int
                b) ->int{
                        if ( w_ptr[a] > w_ptr[b]){
                            return a;
                        } else{
                            return b;
                        }
                };


                mgpu::segreduce(I_mem.data(), reduced_size, s_ptr_array, vertices_count, out.data(),
                                my_cool_lambda, (int) init, context);

                {
                    dim3 block(1024, 1);
                    dim3 grid(vertices_count / block.x, 1);
                    SAFE_KERNEL_CALL((get_labels << < grid, block >> >
                                                            (out.data(), s_array.data(), dev_dest_labels, dev_labels)));
                }
                SAFE_CALL(cudaEventRecord(stop));
                SAFE_CALL(cudaEventSynchronize(stop));
                SAFE_CALL(cudaEventElapsedTime(&time, start, stop));
                time *= 1000000;
                cout << "TEPS for iteration" << edges_count / time << endl;

//                cudaMemcpy(labels,dev_labels,vertices_count,cudaMemcpyDeviceToHost);
//                std::cout<<"Iteration "<<iter<< " is over"<<endl;
//                debug_info(labels,vertices_count,"Labels after current iteration");

                iter++;
            } while (iter < 4);
            //cudaMemcpy(labels,dev_labels,edges_count,cudaMemcpyDeviceToHost);
            //debug_info(labels,vertices_count, "FINALLY");
            a.get_dev_v_array();
            a.get_dev_e_array();
            a.get_dev_weigths();
            SAFE_CALL(cudaFree(dev_dest_labels));
            SAFE_CALL(cudaFree(dev_labels));
            SAFE_CALL(cudaFree(a.get_dev_v_array()));
            SAFE_CALL(cudaFree(a.get_dev_e_array()));
            SAFE_CALL(cudaFree(a.get_dev_weigths())); //check for unweigthed graph!
            SAFE_CALL(cudaFree(F_mem));


        }


        if (lp_flag) {
            lp(vertices_count, a.get_e_array(), a.get_v_array(), labels);
            //louvain(vertices_count, edges_count, a.get_e_array(), a.get_v_array(), labels,a.get_weights(),true);
            a.save_to_graphviz_file("graph_res", labels);
            label_stats(labels, vertices_count);
            delete[] labels;
        }
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
