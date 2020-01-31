#include "CSR_GRAPH.h"
#include <iostream>
#include "math.h"
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
    }
}

CSR_GRAPH:: CSR_GRAPH(int v, int e, int *_src_ids, int *_dst_ids, float *_weigths) : vertices_count(v),
                                                                                  edges_count(e) {
    v_array = new int[vertices_count+1];
    e_array = new int[edges_count];
    weigths = new float[edges_count];
    int position = 0;
    for (int vertice = 0; vertice < vertices_count; vertice++) {
        int count = 0;
        for (int i = 0; i < edges_count; i++) {
            if (_src_ids[i] == vertice) {
                count++;
                e_array[position + count - 1] = _dst_ids[i];
                weigths[position + count - 1] = _weigths[i];
            }
        }
        v_array[vertice] = position;
        position += count;
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
