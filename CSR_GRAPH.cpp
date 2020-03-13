#include "CSR_GRAPH.h"
#include <iostream>
#include "math.h"
#include "omp.h"
#include <vector>
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
        delete[] weights;
        //delete[] labels;
        //delete [] dest_labels;
        //delete []test_dest_labels;
    }
}

typedef std::pair<unsigned int,float> edge;


CSR_GRAPH:: CSR_GRAPH(unsigned int v, unsigned int e, unsigned int *_src_ids, unsigned int *_dst_ids, float *_weigths,bool w) : vertices_count(v),
                                                                                                   edges_count(e),weighted(w) {


    std::vector<std::vector<edge> > graph_info(vertices_count+1);

    //dest_labels = new unsigned int[edges_count];
    //weigths = new float[edges_count];

    e_array = new unsigned int[edges_count];
    v_array = new unsigned int[vertices_count+1];

    for(long long int i = 0; i < edges_count; i++)
    {
        int src_id = _src_ids[i];
        int dst_id = _dst_ids[i];
        float weight = _weigths[i];
        graph_info[src_id].push_back(std::pair<unsigned int,float>(dst_id,weight));
    }

    unsigned int current_edge = 0;
    e_array[0] = 0;
    for(int cur_vertex = 0; cur_vertex < vertices_count; cur_vertex++)
    {
        int src_id = cur_vertex;

        for(int i = 0; i < graph_info[src_id].size(); i++)
        {
            e_array[current_edge] = graph_info[src_id][i].first;
            weights[current_edge] = graph_info[src_id][i].second;
            current_edge++;
        }
        v_array[cur_vertex + 1] = current_edge;
    }
    std::cout<<"no segmentation in constructor"<<endl;
}

void CSR_GRAPH::adj_distribution(int _edges) {

    int num  = (int)log2(_edges)+1;
    int *borders = new int [num];
    for(int i =0; i<num ;i++){
        borders[i] = 0;
    }


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
    //v_array[vertices_count] = edges_count;

    for (int i = 0; i < vertices_count; i++) {
        cout << i <<" vertice is connected to";
        for(int j = v_array[i]; j<v_array[i+1];j++){
            cout<<" "<<e_array[j];
        }
        cout<<" vertices"<<endl;
    }
};



void gather_thread_per_vertex(int _omp_threads, unsigned int edges_count, unsigned int vertices_count, unsigned int *dest_labels,
                              const unsigned int* v_array, const unsigned int *e_array, const unsigned int *labels) {
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



void print_label_info(int _omp_threads,const int *labels, const int *dest_labels, unsigned int vertices_count, unsigned int edges_count) {
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


void generate_labels(int _omp_threads, unsigned int vertices_count, unsigned int *labels) {
    unsigned int seed;
    //labels = new unsigned int[vertices_count];
#pragma omp parallel num_threads(_omp_threads) private(seed)
    {
        seed = int(time(NULL)) * omp_get_thread_num();
#pragma omp for schedule(static)
        for (int i = 0; i < vertices_count; i++) {
            labels[i] = (unsigned int) rand_r(&seed)%(500);
        }
    }
}

long unsigned int check(unsigned int edges_count, unsigned int*test_dest_labels, unsigned int * dest_labels ) {
    for(long unsigned int j=0;j<edges_count;j++){
        if(test_dest_labels[j]!=dest_labels[j]){
            printf("ERROR IN %ld position",j);
            return -1;
        }
    }
    return 0;
}
