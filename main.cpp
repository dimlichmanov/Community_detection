#include <omp.h>
#include <iostream>
#include <string.h>
#include <fstream>
#include <algorithm>

#include <stdio.h>
#include <math.h>
#include <vector>

#include "CSR_GRAPH.h"
#include "generator.h"

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

        for (int i = 0; i < edges_count; i++) {
            cout << src_ids[i] << "----" << dst_ids[i] << endl;

        }

        CSR_GRAPH a(vertices_count,edges_count,src_ids,dst_ids,weights);


        a.print_CSR_format();
        a.print_adj_format();
        a.adj_distribution(edges_count);

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

    cout << "press any key to exit..." << endl;
    //getchar();
    return 0;
}