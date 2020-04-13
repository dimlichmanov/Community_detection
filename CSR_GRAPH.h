#ifndef RMAT_CSR_GRAPH_H
#define RMAT_CSR_GRAPH_H

#include "string"
#include "fstream"

using namespace std;

class CSR_GRAPH {
    bool weighted;
    int *v_array;
    int *e_array;
    int *dev_v_array;
    int *dev_e_array;
    float *weights;
    float *dev_weigths;
    int vertices_count;
    int edges_count;
public:
    int *get_v_array() { return v_array; }

    int *get_e_array() { return e_array; }

    float *get_weights() { return weights; }

    int *get_dev_v_array() { return dev_v_array; }

    int *get_dev_e_array() { return dev_e_array; }

    float *get_dev_weigths() { return dev_weigths; }

    explicit CSR_GRAPH(int v, int e, int *_src_ids, int *_dst_ids, float *_weigths, bool weighted);

    void print_CSR_format();

    ~CSR_GRAPH();

    void print_adj_format();

    void adj_distribution(int _edges);

    void move_to_device(int *dest_labels, int *labels, int *dev_dest_labels, int *dev_labels);

    void move_to_host(int *dest_labels, int *labels, int *dev_dest_labels, int *dev_labels);

    void save_to_graphviz_file(string _file_name, int *k);
};

void
form_label_array(int _omp_threads, int vertices_count, int edges_count, int *dest_labels,
                 int *v_array, int *labels, int *e_array);

void gather_thread_per_vertex(int _omp_threads, int edges_count, int vertices_count, int *dest_labels,
                              const int *v_array, const int *e_array, const int *labels);

void print_label_info(int _omp_threads, const int *labels, const int *dest_labels, int vertices_count, int edges_count);

void generate_labels(int _omp_threads, int vertices_count, int *labels);

int check(int edges_count, int *test_dest_labels, int *dest_labels);

int label_prop();


#endif //RMAT_CSR_GRAPH_H
