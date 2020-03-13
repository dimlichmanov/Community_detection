//
// Created by dimon on 30.01.20.
//
#include "omp.h"
#include <iostream>
using namespace std;
#include <stdio.h>
#include <math.h>
#ifndef RMAT_GENERATOR_H
#define RMAT_GENERATOR_H


void
uniform_random(unsigned int *src_ids, unsigned int *dst_ids, float *weights, unsigned int _vertices_count, long _edges_count, int _omp_threads,
               bool _directed, bool _weighted);

void R_MAT(unsigned int *src_ids, unsigned int *dst_ids, float *weights, unsigned int _vertices_count, long _edges_count, int _a_prob, int _b_prob,
           int _c_prob, int _d_prob, int _omp_threads, bool _directed, bool _weighted);

#endif //RMAT_GENERATOR_H
