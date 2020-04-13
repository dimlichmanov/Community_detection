#include "CSR_GRAPH.h"
#include <iostream>
#include "math.h"
#include "omp.h"
#include <vector>
#include "stdio.h"
#include "stdlib.h"
#include "string"
#include <fstream>

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

CSR_GRAPH::~CSR_GRAPH() {
    {
        delete[] v_array;
        delete[] e_array;
        delete[] weights;
    }
}

typedef std::pair<   int, float> edge;


CSR_GRAPH::CSR_GRAPH(   int v,    int e,    int *_src_ids,    int *_dst_ids, float *_weigths,
                     bool w) : vertices_count(v),
                               edges_count(e), weighted(w) {


    std::vector<std::vector<edge> > graph_info(vertices_count + 1);

    //dest_labels = new    int[edges_count];
    weights = new float[edges_count];

    e_array = new    int[edges_count];
    v_array = new    int[vertices_count + 1];

    for (long long int i = 0; i < edges_count; i++) {
        int src_id = _src_ids[i];
        int dst_id = _dst_ids[i];
        float weight = _weigths[i];
        graph_info[src_id].push_back(std::pair<   int, float>(dst_id, weight));
    }

       int current_edge = 0;
    e_array[0] = 0;
    for (int cur_vertex = 0; cur_vertex < vertices_count; cur_vertex++) {
        int src_id = cur_vertex;

        for (int i = 0; i < graph_info[src_id].size(); i++) {
            e_array[current_edge] = graph_info[src_id][i].first;
            weights[current_edge] = graph_info[src_id][i].second;
            current_edge++;
        }
        v_array[cur_vertex + 1] = current_edge;
    }
    v_array[0] = 0;
    std::cout << "no segmentation in constructor" << endl;
}


void
form_label_array(int _omp_threads,    int vertices_count,    int edges_count,    int *dest_labels,
                    int *v_array,    int *labels,    int *e_array) {
    dest_labels = new    int[edges_count];
#pragma omp parallel num_threads(_omp_threads)
    {
#pragma omp for schedule(static)
        for (int i = 0; i < vertices_count; ++i) {
            for (int j = v_array[i]; j < v_array[i + 1]; ++j) {
                dest_labels[j] = labels[e_array[j]];
            }
        }
    }
}

void CSR_GRAPH::save_to_graphviz_file(string _file_name,    int * labels) {
    ofstream dot_output(_file_name.c_str());
    using namespace std;

    string connection;
    dot_output << "digraph G {" << endl;
    connection = " -> ";

    for (int cur_vertex = 0; cur_vertex < vertices_count; cur_vertex++) {
        int src_id = cur_vertex;
        for (long long edge_pos = v_array[cur_vertex]; edge_pos < v_array[cur_vertex + 1]; edge_pos++) {
            int dst_id = e_array[edge_pos];
            dot_output << src_id << connection << dst_id << endl;
        }
    }

    if(labels!= NULL) {
        for (int i = 0; i < this->vertices_count; i++) {
            dot_output << i << " [label= \"id=" << i << ", value=" << labels[i] << "\"] " << endl;
            //dot_output << i << " [label=" << this->vertex_values[i] << "]"<< endl;
        }
    }
    dot_output << "}";
    dot_output.close();
}

void CSR_GRAPH::adj_distribution(int _edges) {

    int num = (int) log2(_edges) + 1;
    int *borders = new int[num];
    for (int i = 0; i < num; i++) {
        borders[i] = 0;
    }


    for (int i = 0; i < vertices_count; i++) {
        int count = v_array[i + 1] - v_array[i];


        count = (int) log2(count + 1);
        cout << i << " " << count << endl;
        borders[count]++;
    }

    for (int i = 0; i < num; i++) {
        cout << borders[i] << " ";
    }

    delete[] borders;

}

void CSR_GRAPH::print_adj_format(void) {
    //v_array[vertices_count] = edges_count;

    for (int i = 0; i < vertices_count; i++) {
        cout << i << " vertice is connected to";
        for (int j = v_array[i]; j < v_array[i + 1]; j++) {
            cout << " " << e_array[j];
        }
        cout << " vertices" << endl;
    }
};


void gather_thread_per_vertex(int _omp_threads,    int edges_count,    int vertices_count,
                                 int *dest_labels,
                              const    int *v_array, const    int *e_array, const    int *labels) {
    dest_labels = new    int[edges_count];
#pragma omp parallel num_threads(_omp_threads)
    {
#pragma omp for schedule(static)
        for (int i = 0; i < vertices_count; ++i) {
            for (int j = v_array[i]; j < v_array[i + 1]; ++j) {
                dest_labels[j] = labels[e_array[j]];
            }
        }
    }
}


void print_label_info(int _omp_threads, const int *labels, const int *dest_labels,    int vertices_count,
                         int edges_count) {
#pragma omp parallel num_threads(_omp_threads)
    {
#pragma omp for schedule(static)
        for (int i = 0; i < vertices_count; i++) {
            printf("vertice %d has label %d\n", i, labels[i]);
        }
#pragma omp barrier
#pragma omp for schedule(static)
        for (int i = 0; i < edges_count; ++i) {
            printf("edge %d has destination label %d\n", i, dest_labels[i]);
        }

    }
}


void generate_labels(int _omp_threads,    int vertices_count,    int *labels) {
    unsigned int seed;
    //labels = new    int[vertices_count];
#pragma omp parallel num_threads(_omp_threads) private(seed)
    {
        seed = int(time(NULL)) * omp_get_thread_num();
#pragma omp for schedule(static)
        for (int i = 0; i < vertices_count; i++) {
            labels[i] = ( int) rand_r(&seed) % (500);
        }
    }
}

int check(   int edges_count,    int *test_dest_labels,    int *dest_labels) {
    for (long    int j = 0; j < edges_count; j++) {
        if (test_dest_labels[j] != dest_labels[j]) {
            printf("ERROR IN %ld position", j);
            return -1;
        }
    }
    return 0;
}
