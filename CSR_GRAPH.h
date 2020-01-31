#ifndef RMAT_CSR_GRAPH_H
#define RMAT_CSR_GRAPH_H


class CSR_GRAPH {
    int *v_array;
    int *e_array;
    float *weigths;
    int vertices_count;
    int edges_count;

public:
    explicit CSR_GRAPH(int v, int e, int *_src_ids, int *_dst_ids, float *_weigths);
    void print_CSR_format();
    ~CSR_GRAPH();
    void print_adj_format();
    void adj_distribution(int _edges);


};


#endif //RMAT_CSR_GRAPH_H
