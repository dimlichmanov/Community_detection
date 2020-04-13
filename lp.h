
#ifndef RMAT_LP_H
#define RMAT_LP_H

#include <string.h>



void lp(size_t vertices_count,   int *e_array,   int *v_array,   int *labels);


void louvain(size_t vertices_count, size_t edges_count,   int *e_array,   int *v_array,   int *labels, float *weights,bool directed);

#endif //RMAT_LP_H
