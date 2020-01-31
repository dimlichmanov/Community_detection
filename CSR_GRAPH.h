#ifndef RMAT_CSR_GRAPH_H
#define RMAT_CSR_GRAPH_H


class CSR_GRAPH {
    bool weighted;
    unsigned int *v_array;
    unsigned int *e_array;
    float *weigths;
    int vertices_count;
    int edges_count;
    unsigned int *labels;
    unsigned int *dest_labels;
public:
    explicit CSR_GRAPH(int v, int e, int *_src_ids, int *_dst_ids, float *_weigths,bool weighted);
    void print_CSR_format();
    ~CSR_GRAPH();
    void print_adj_format();
    void adj_distribution(int _edges);
    void generate_labels(int threads);
    void form_label_array(int threads);
};


#endif //RMAT_CSR_GRAPH_H
