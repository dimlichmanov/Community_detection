
#ifndef RMAT_LP_H
#define RMAT_LP_H

#include <string.h>



void lp(size_t vertices_count, unsigned int *e_array, unsigned int *v_array, unsigned int *labels);


void louvain(size_t vertices_count, size_t edges_count, unsigned int *e_array, unsigned int *v_array, unsigned int *labels, float *weights);

#endif //RMAT_LP_H
