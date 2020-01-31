#include "generator.h"
#include "stdlib.h"
void
uniform_random(int *src_ids, int *dst_ids, float *weights, int _vertices_count, long _edges_count, int _omp_threads,
               bool _directed, bool _weighted) {
    int n = (int) log2(_vertices_count);

    cout << "using " << _omp_threads << " threads" << endl;

    // generate and add edges to graph
    unsigned int seed = 0;
#pragma omp parallel num_threads(_omp_threads) private(seed)
    {
        seed = int(time(NULL)) * omp_get_thread_num();

#pragma omp for schedule(static)
        for (long long cur_edge = 0; cur_edge < _edges_count; cur_edge++) {
            int from = rand() % _vertices_count;
            int to = rand() % _vertices_count;
            float edge_weight = static_cast <float> (rand_r(&seed)) / static_cast <float> (RAND_MAX);

            if (_directed) {
                src_ids[cur_edge] = from;
                dst_ids[cur_edge] = to;

                if (_weighted)
                    weights[cur_edge] = edge_weight;
            }

            if (!_directed) {
                src_ids[cur_edge] = min(to, from);
                dst_ids[cur_edge] = max(to, from);
                if (_weighted)
                    weights[cur_edge] = edge_weight;
            }
        }
    }
}

void R_MAT(int *src_ids, int *dst_ids, float *weights, int _vertices_count, long _edges_count, int _a_prob, int _b_prob,
           int _c_prob, int _d_prob, int _omp_threads, bool _directed, bool _weighted) {
    int n = (int) log2(_vertices_count);

    cout << "using " << _omp_threads << " threads" << endl;

    // generate and add edges to graph
    unsigned int seed = 0;
#pragma omp parallel num_threads(_omp_threads) private(seed)
    {
        seed = int(time(NULL)) * omp_get_thread_num();

#pragma omp for schedule(static)
        for (long long cur_edge = 0; cur_edge < _edges_count; cur_edge++) {
            int x_middle = _vertices_count / 2, y_middle = _vertices_count / 2;
            for (long long i = 1; i < n; i++) {
                int a_beg = 0, a_end = _a_prob;
                int b_beg = _a_prob, b_end = b_beg + _b_prob;
                int c_beg = _a_prob + _b_prob, c_end = c_beg + _c_prob;
                int d_beg = _a_prob + _b_prob + _c_prob, d_end = d_beg + _d_prob;

                int step = (int) pow(2, n - (i + 1));

                int probability = rand_r(&seed) % 100;
                if (a_beg <= probability && probability < a_end) {
                    x_middle -= step, y_middle -= step;
                } else if (b_beg <= probability && probability < b_end) {
                    x_middle -= step, y_middle += step;
                } else if (c_beg <= probability && probability < c_end) {
                    x_middle += step, y_middle -= step;
                } else if (d_beg <= probability && probability < d_end) {
                    x_middle += step, y_middle += step;
                }
            }
            if (rand_r(&seed) % 2 == 0)
                x_middle--;
            if (rand_r(&seed) % 2 == 0)
                y_middle--;

            int from = x_middle;
            int to = y_middle;
            float edge_weight = static_cast <float> (rand_r(&seed)) / static_cast <float> (RAND_MAX);

            if (_directed) {
                src_ids[cur_edge] = from;
                dst_ids[cur_edge] = to;

                if (_weighted)
                    weights[cur_edge] = edge_weight;
            }

            if (!_directed) {
                src_ids[cur_edge] = min(to, from);
                dst_ids[cur_edge] = max(to, from);
                if (_weighted)
                    weights[cur_edge] = edge_weight;
            }
        }
    }
}