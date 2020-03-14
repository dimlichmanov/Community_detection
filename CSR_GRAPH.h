#ifndef RMAT_CSR_GRAPH_H
#define RMAT_CSR_GRAPH_H

class CSR_GRAPH {
    bool weighted;
    unsigned int *v_array;
    unsigned int *e_array;
    unsigned int *dev_v_array;
    unsigned int *dev_e_array;
    float *weights;
    float * dev_weigths;
    unsigned int vertices_count;
    unsigned int edges_count;
public:
    unsigned int *get_v_array(){return v_array;}
    unsigned int *get_e_array(){return e_array;}
    float* get_weights(){return weights;}
    unsigned int * get_dev_v_array(){ return dev_v_array; }
    unsigned int * get_dev_e_array(){return dev_e_array;}
    float * get_dev_weigths(){return dev_weigths;}

    explicit CSR_GRAPH(unsigned int v, unsigned int e, unsigned int *_src_ids, unsigned int *_dst_ids, float *_weigths,bool weighted);
    void print_CSR_format();
    ~CSR_GRAPH();
    void print_adj_format();
    void adj_distribution(int _edges);

    void move_to_device(unsigned int* dest_labels, unsigned int* labels, unsigned int* dev_dest_labels ,unsigned int* dev_labels);
    void move_to_host(unsigned int* dest_labels, unsigned int* labels, unsigned int* dev_dest_labels ,unsigned int* dev_labels);
};

void
form_label_array(int _omp_threads, unsigned int vertices_count, unsigned int edges_count, unsigned int *dest_labels,
                 unsigned int *v_array, unsigned int *labels, unsigned int *e_array);

void gather_thread_per_vertex(int _omp_threads, unsigned int edges_count, unsigned int vertices_count, unsigned int *dest_labels,
                              const unsigned int* v_array, const unsigned int *e_array, const unsigned int *labels);
void print_label_info(int _omp_threads,const int *labels, const int *dest_labels, unsigned int vertices_count, unsigned int edges_count);

void generate_labels(int _omp_threads, unsigned int vertices_count, unsigned int *labels);

int check(unsigned int edges_count, unsigned int*test_dest_labels, unsigned int * dest_labels );

#endif //RMAT_CSR_GRAPH_H
