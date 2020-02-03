#include "CSR_GRAPH.h"
#include <iostream>
#include "math.h"
#include "omp.h"

#include "stdio.h"
#include "stdlib.h"
using namespace std;

void CSR_GRAPH::print_CSR_format(void) {
    for (int i = 0; i < vertices_count; i++) {
        cout << v_array[i] << endl;

    }
    cout << endl;
    for (int i = 0; i < edges_count; i++) {
        cout << e_array[i] << endl;

    }
}

CSR_GRAPH ::~CSR_GRAPH() {
    {
        delete[] v_array;
        delete[] e_array;
        delete[] weigths;
        delete[] labels;
        delete [] dest_labels;
    }
}

CSR_GRAPH:: CSR_GRAPH(int v, int e, int *_src_ids, int *_dst_ids, float *_weigths,bool weighted) : vertices_count(v),
                                                                                  edges_count(e) {
    v_array = new unsigned int[vertices_count + 1];
    unsigned int *v_array_copy   = new unsigned int[vertices_count + 1];
    e_array = new unsigned int[edges_count];
    weigths = new float[edges_count];     //optional

    for (int i = 0; i < edges_count; i++) {
        v_array[_src_ids[i]]++;
        v_array_copy[_src_ids[i]]++;
    }
    unsigned int sum = v_array[0];
    v_array[0] = 0;
    v_array_copy[0] = 0;
    for (int j = 1; j < vertices_count; j++) {
        int temp = v_array[j];
        v_array[j] = sum;
        v_array_copy[j] = sum;
        sum+=temp;
    }
    for (int k = 0; k < edges_count; k++) {
        if(weighted) weigths[v_array_copy[_src_ids[k]]] = _weigths[k];
        e_array[v_array_copy[_src_ids[k]]++] = _dst_ids[k];
        
    }

}

void CSR_GRAPH::adj_distribution(int _edges) {

    int num  = (int)log2(_edges)+1;
    int *borders = new int [num];
    for(int i =0; i<num ;i++){
        borders[i] = 0;
    }

    v_array[vertices_count] = edges_count;
    for (int i = 0; i < vertices_count; i++) {
        int count  = v_array[i+1] - v_array[i];


        count = (int)log2(count+1);
        cout<<i<<" "<<count<<endl;
        borders[count] ++;
    }

    for(int i =0;i<num;i++){
        cout<<borders[i]<<" ";
    }

    delete[] borders;

}

void CSR_GRAPH::print_adj_format(void) {
    v_array[vertices_count] = edges_count;

    for (int i = 0; i < vertices_count; i++) {
        cout << i <<" vertice is connected to";
        for(int j = v_array[i]; j<v_array[i+1];j++){
            cout<<" "<<e_array[j];
        }
        cout<<" vertices"<<endl;
    }
};

void CSR_GRAPH::form_label_array(int _omp_threads) {
    v_array[vertices_count] = edges_count;
    dest_labels = new unsigned int[edges_count];
#pragma omp parallel num_threads(_omp_threads)
    {
#pragma omp for schedule(static)
        for (int i = 0; i < vertices_count; ++i) {
            for (int j = v_array[i]; j <v_array[i+1] ; ++j) {
                dest_labels[j] = labels[e_array[j]];
            }
        }
    }
}

void CSR_GRAPH::print_label_info(int _omp_threads) {
#pragma omp parallel num_threads(_omp_threads)
    {
#pragma omp for schedule(static)
        for (int i = 0; i < vertices_count; i++) {
            printf("vertice %d has label %d\n",i,labels[i]);
        }
#pragma omp barrier
#pragma omp for schedule(static)
        for (int i = 0; i < edges_count; ++i) {
            printf("edge %d has destination label %d\n",i,dest_labels[i]);
        }

    }
}


void CSR_GRAPH::generate_labels(int _omp_threads) {
    unsigned int seed;
    labels = new unsigned int[vertices_count];
#pragma omp parallel num_threads(_omp_threads) private(seed)
    {
        seed = int(time(NULL)) * omp_get_thread_num();
#pragma omp for schedule(static)
        for (int i = 0; i < vertices_count; i++) {
            labels[i] = (unsigned int) rand_r(&seed)%(500);
        }
    }
}
