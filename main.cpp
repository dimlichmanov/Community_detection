#include <omp.h>
#include <iostream>
#include <string.h>
#include <fstream>
#include <algorithm>
#include "stdlib.h"
#include <stdio.h>
#include <math.h>
#include <vector>

#include "CSR_GRAPH.h"
#include "generator.h"
#include "device_gather.h"

using namespace std;

#ifndef uint32_t
#define uint32_t int
#endif


int main(int argc, char **argv) {
    try {


        int threads = omp_get_max_threads();
        int vertices_index = atoi(argv[1]);
        int density_degree = atoi(argv[2]);
        char *graph_type = argv[3];
        double begin,end;

        int vertices_count =  pow(2.0, vertices_index);
        int edges_count = density_degree * vertices_count;

        int *src_ids = new int[edges_count];
        int *dst_ids = new int[edges_count];
        float *weights = new float[edges_count];


        if (strcmp(graph_type, "rmat") == 0) {
            R_MAT(src_ids, dst_ids, weights, vertices_count, edges_count, 45, 20, 20, 15, threads, true, true);

        } else {
            uniform_random(src_ids, dst_ids, weights, vertices_count, edges_count, threads, true, true);
        }

        /*for (int i = 0; i < edges_count; i++) {
            cout << src_ids[i] << "----" << dst_ids[i] << endl;

        }*/

        CSR_GRAPH a(vertices_count,edges_count,src_ids,dst_ids,weights, true);


        //a.print_CSR_format();
        //a.print_adj_format();
        //a.adj_distribution(edges_count);



        a.generate_labels(threads);


        begin = omp_get_wtime();
        a.form_label_array(threads);
        end = omp_get_wtime();
        //a.print_label_info(threads);
        printf("Bandwidth for 2^%d edges is %f GB/s\n ", vertices_index + (int) log2(density_degree) , sizeof(int)*(vertices_count + 3*edges_count)/((end - begin)*(int)pow(1000,3)));

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
    return 0;
}