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

void debug_info(unsigned int *ptr, int size_n, string info) {
    cout << info << endl;
    for (int i = 0; i < size_n; i++) {
        std::cout << ptr[i] << " ";
    }
    cout << endl;
}


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


void label_stats(unsigned int *labels, unsigned int vertices_count) { // Почему то в map много нулей
    std::map<unsigned int, int> mp;
    for (unsigned int i = 0; i < vertices_count; i++) {
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




void input(char *filename, bool directed, unsigned int *&src_ids, unsigned int *&dst_ids, unsigned int &vertices_count,
           unsigned int &edges_count) {
    unsigned int max_vertice = 0;

    std::ifstream infile(filename);
    std::string line;
    unsigned int i = 0;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        int a, b;
        if (!(iss >> a >> b)) {
            break;
        } else {
            if (max(a, b) > max_vertice) {
                max_vertice = (unsigned) max(a, b);
            }
        }
        i++;
    }
    edges_count = i;
    vertices_count = max_vertice;
    src_ids = new unsigned int[edges_count];
    dst_ids = new unsigned int[edges_count];

    std::ifstream infile1(filename);
    i = 0;
    while (std::getline(infile1, line)) {
        std::istringstream iss(line);
        unsigned int a, b;
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

__global__ void extract_boundaries_initial(short *boundaries, unsigned int *v_array, unsigned int edges_count) {

    unsigned long int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long int position = v_array[i];
    if (i != 0) {
        boundaries[position - 1] = 1;
    } else {
        boundaries[edges_count - 1] = 1;
    }
}

__global__ void extract_boundaries_optional(short *boundaries, unsigned  int *dest_labels, unsigned int edges_count) {
    unsigned long int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ((boundaries[i] != 1) && (i < edges_count)) {
        if (dest_labels[i] != dest_labels[i + 1]) {
            boundaries[i] = 1;
        }
    }
}

__global__ void count_labels(unsigned int *scanned_array, unsigned int edges_count, int *S_array) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ((i < edges_count - 1) && (scanned_array[i + 1] != scanned_array[i])) {
        S_array[scanned_array[i]] = i;
    }
}

__global__ void new_boundaries(unsigned int *scanned_array, unsigned int *v_array, unsigned int edges_count, int *S_ptr) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    S_ptr[i] = scanned_array[v_array[i]];
}

__global__ void frequency_count(int *W_array, int *S) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ((i > 0) && (S[i] != 0)) {
        W_array[i] = S[i] - S[i - 1];
    } else {
        W_array[0] = S[0] + 1;
    }
}

__global__ void get_labels(int *I , int* S, unsigned int *L ,unsigned int* labels){
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    labels[i] = L[S[I[i]]];
}


using namespace std;


int main(int argc, char **argv) {
    try {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
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

        unsigned int vertices_count = pow(2.0, vertices_index);
        unsigned int edges_count = density_degree * vertices_count;
        unsigned int *src_ids = NULL;
        unsigned int *dst_ids = NULL;
        float *weights = new float[edges_count];

        if (!test_flag) {
            src_ids = new unsigned int[edges_count];
            dst_ids = new unsigned int[edges_count];
            cout << "test_flag" << endl;
            if (strcmp(graph_type, "rmat") == 0) {
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

        unsigned int *labels = new unsigned int[vertices_count];
        for (unsigned int j = 0; j < vertices_count; j++) {
            labels[j] = j;
        }
        unsigned int *dest_labels = new unsigned int[edges_count];
        unsigned int *dev_labels;
        unsigned int *dev_dest_labels;
        short *F_mem;

        if (gather_flag) {

            SAFE_CALL((cudaMalloc((void **) &dev_labels, (size_t) (sizeof(unsigned int)) * (vertices_count))));
            SAFE_CALL((cudaMalloc((void **) &dev_dest_labels, (size_t) (sizeof(unsigned int)) * edges_count)));
            SAFE_CALL((cudaMalloc((void **) &F_mem, (size_t) (sizeof(short)) * edges_count)));

            a.move_to_device(dest_labels, labels, dev_dest_labels, dev_labels);



            cout<<1<<endl;
            int iter = 0;
            mgpu::standard_context_t context;
            std::vector<int> ptr; //Bounds as segments
            for (int k = 0; k < vertices_count; k++) {
                ptr.push_back(a.get_v_array()[k]);
            }

            cout<<2<<endl;
            mgpu::mem_t<int> segs = mgpu::to_mem(ptr, context);
            mgpu::mem_t<int> s_ptr_array(vertices_count, context);
            mgpu::mem_t<int> out(vertices_count, context);
            cout<<3<<endl;
            do {
                SAFE_CALL(cudaEventRecord(start));

                {
                    //Change configuration after
                    dim3 block(1024, 1);
                    dim3 grid(32*vertices_count/block.x, 1); //only for test
                    printf("starting...");
                    SAFE_KERNEL_CALL((gather_warp_per_vertex << < grid, block >> >
                                                                        (a.get_dev_v_array(), a.get_dev_e_array(), dev_dest_labels, dev_labels, edges_count, vertices_count)));
                }
                printf("terminating....");
                SAFE_CALL(cudaEventRecord(stop));
                SAFE_CALL(cudaEventSynchronize(stop));
                float time;
                SAFE_CALL(cudaEventElapsedTime(&time, start, stop));
                time *= 1000000;
                a.move_to_host(dest_labels, labels, dev_dest_labels, dev_labels);
                //mgpu::mem_t<int> F_scanned(edges_count, context);


                if (check_flag) {
                    unsigned int *test_dest_labels = new unsigned int[edges_count];
                    form_label_array(threads, vertices_count, edges_count, test_dest_labels, a.get_dev_v_array(),
                                     labels,
                                     a.get_e_array());
                    int flag = check(edges_count, dest_labels, test_dest_labels);
                    if (flag == 0) {
                        printf("CORRECT");
                    }
                    delete[] test_dest_labels;
                }


                printf("GATHER Bandwidth for 2^%d vertices and 2^%d edges is %f GB/s\n ", vertices_index,
                       vertices_index + (int) log2((double) density_degree),
                       sizeof(unsigned int) * (2 * vertices_count + 2 * edges_count) / (time));


                //print_bounds(ptr, edges_count);
                //debug_info(dest_labels,edges_count,"initial gather");


                cout<<4<<endl;
                mgpu::mem_t<int> values(edges_count, context);


                mgpu::segmented_sort(dev_dest_labels, values.data(), edges_count, segs.data(), vertices_count,
                                     mgpu::less_t<int>(), context);

                cout<<5<<endl;
                //cudaMemcpy(dest_labels,dev_dest_labels,edges_count,cudaMemcpyDeviceToHost);

                //debug_info(dest_labels,edges_count, "sorted_gather");

                SAFE_CALL((cudaMemset(F_mem, 0, (size_t) (sizeof(short)) * edges_count))); //was taken from group of memcpy

                {
                    dim3 block(1024, 1);
                    dim3 grid(vertices_count/block.x, 1);

                    SAFE_KERNEL_CALL(
                            (extract_boundaries_initial << < grid, block >> >
                                                                   (F_mem, a.get_dev_v_array(), edges_count))); //fill 1 in bounds
                }
                cout<<6<<endl;
                {
                    dim3 block(1024, 1);
                    dim3 grid(edges_count/block.x, 1);

                    SAFE_KERNEL_CALL(
                            (extract_boundaries_optional << < grid, block >> >
                                                                    (F_mem, dev_dest_labels, edges_count))); //sub(i+1, i)
                }

                //short *F_host = new short[edges_count];
                //cudaMemcpy(F_host, F_mem, (size_t) edges_count * sizeof(short), cudaMemcpyDeviceToHost);

                //debug_info(F_host, edges_count, "neighbors");


                mgpu::mem_t<unsigned int> F_scanned(edges_count, context);

                cout<<7<<endl;
                mgpu::scan(F_mem, edges_count, F_scanned.data(), context); // may not work because of bool

                //std::vector<int> hosted_bounds = from_mem(F_scanned); // gather

                //debug_info(hosted_bounds, "scanned F");
                SAFE_CALL(cudaDeviceSynchronize());
                cout<<8<<endl;
                unsigned int reduced_size;

                SAFE_CALL(cudaMemcpy(&reduced_size, &F_scanned.data()[edges_count - 1 ], sizeof(unsigned int), cudaMemcpyDeviceToHost));


                mgpu::mem_t<int> s_array(reduced_size, context);
                cout<<9<<endl;
                {
                    dim3 block(1024, 1);
                    dim3 grid(edges_count/block.x, 1);
                    SAFE_KERNEL_CALL(
                            (count_labels << < grid, block >> > (F_scanned.data(), edges_count, s_array.data())));
                }


                //std::vector<int> s_host = from_mem(s_array);

                //debug_info(s_host, "S for frequency");

                //mgpu::mem_t<int> s_ptr_array(vertices_count, context);


                cout<<10<<endl;
                {
                    dim3 block(1024, 1);
                    dim3 grid(vertices_count/block.x, 1);
                    SAFE_KERNEL_CALL((new_boundaries << < grid, block >> >
                                                                (F_scanned.data(), a.get_dev_v_array(), edges_count, s_ptr_array.data())));
                }

//                std::vector<int> ptr_host = from_mem(s_ptr_array);
//
//                debug_info(ptr_host, "new bounds");
                cout<<11<<endl;
                mgpu::mem_t<int> w_array(reduced_size, context);
                {
                    dim3 block(1024, 1);
                    dim3 grid(reduced_size/block.x, 1);


                    SAFE_KERNEL_CALL((frequency_count << < grid, block >> > (w_array.data(), s_array.data())));
                }
                cout<<12<<endl;

//                std::vector<int> w_host = from_mem(w_array);
//                std::vector<int> debug_w((size_t) reduced_size);
//                debug_info(w_host, "W_array");



                std::vector<int> I;
                for (int k = 0; k < reduced_size; k++) { //indices for reduce
                    I.push_back(k);
                }

                mgpu::mem_t<int> I_mem = mgpu::to_mem(I, context);

                int init = 0;
                cout<<13<<endl;

//            auto k = [] MGPU_DEVICE(int tid, int cta) {
//
//            };
                int *w_ptr = w_array.data();

                auto my_cool_lambda =[w_ptr] MGPU_DEVICE(int a, int b) ->int{
                        if ( w_ptr[a] > w_ptr[b]){
                            return a;
                        } else{
                            return b;
                        }
                };


                mgpu::segreduce(I_mem.data(), reduced_size, s_ptr_array.data(), vertices_count, out.data(),
                                my_cool_lambda, (int) init, context);

//                std::vector<int> i_host = from_mem(out);
//                debug_info(i_host, "seg_reduce");


                {
                    dim3 block(1024, 1);
                    dim3 grid(vertices_count/block.x, 1);
                    SAFE_KERNEL_CALL((get_labels << < grid, block >> >
                                                            (out.data(), s_array.data(), dev_dest_labels, dev_labels)));
                }

//                cudaMemcpy(labels,dev_labels,vertices_count,cudaMemcpyDeviceToHost);
//                std::cout<<"Iteration "<<iter<< " is over"<<endl;
//                debug_info(labels,vertices_count,"Labels after current iteration");

                iter++;
            } while (iter <4);
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
