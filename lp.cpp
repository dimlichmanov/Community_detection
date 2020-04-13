#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include "fstream"
#include "lp.h"
#include "map"

using namespace std;


void
louvain(size_t vertices_count, size_t edges_count,   int *e_array,   int *v_array,   int *labels,
        float *weights, bool directed) {
    std::vector<  int> v(vertices_count);
    float modularity = 0;
    float *k_i = new float[vertices_count]; //output weigths for each vertice
    float *k_j = new float[vertices_count]; //input weights for each verice


    for (  int l = 0; l < vertices_count; ++l) {
        k_j[l] = 0;
    }
    for (  int l = 0; l < vertices_count; ++l) {
        labels[l] = l;
        k_i[l] = 0;
        for (  int j = v_array[l]; j < v_array[l + 1]; j++) {
            k_i[l] += weights[j];
            if(e_array[j] == 2){
                cout<<"!!!!!!!!!!!!!!!!!!!!!!"<<l<<endl;
            }
            k_j[e_array[j]] += weights[j];
        }
        v[l] = l;
    }
    for(int j =0; j<vertices_count;j++){
        cout<< j<< " has "<<k_j[j]<<endl;
    }
    float m = 0;
    for (  int i = 0; i < edges_count; i++) {
        m += weights[i];
    }
    cout << "m is " << m << endl;
    if (!directed) {
        m /= 2;
    }
    bool updated;
    int iters = 0;
    do {
        updated = false;
        //std::random_device rd;
        //std::mt19937 g(rd());
        //std::shuffle(v.begin(), v.end(), g);
        for (  int i = 0; i < vertices_count; i++) { //searching shuffled array
            
            float local_modularity;
              int vertice_id = v[i];
            cout << "WE ARE IN " << vertice_id << " vertice" << endl;
              int begin = v_array[vertice_id];
              int end = v_array[vertice_id + 1];
            float weight_i = 0;
            float weight_j = 0;

            weight_i = k_i[vertice_id];
            cout << "k_i is " << weight_i<<endl;

              int decision_label = vertices_count; //unreachable maximum label instead of -1
            for (  int j = begin; j < end; j++) {

//                weight_j = k_j[e_array[j]];
//                cout << "k_j for  "<< << weight_j<<endl;

                if (labels[vertice_id] != labels[e_array[j]]) {
                    float gain_modularity = (weights[j] - weight_i * weight_j / (2 * m)) / (2 * m);

                    if (gain_modularity > 0) {
                        cout << "new gain is " << gain_modularity << endl;
                        modularity += gain_modularity;
                        updated = true;
                        decision_label = labels[e_array[j]];
                        cout << "new label is " << decision_label << endl;

                    }
                }
            }
            if (decision_label != vertices_count) {
                labels[vertice_id] = decision_label;
                cout << "vertice " << vertice_id << " is updated. New label is " << decision_label << endl;
            }
        }
        iters++;
        std::cout << "labels after current iteration: ";
        for (int l = 0; l < vertices_count; ++l) {
            std::cout << labels[l] << " ";
        }
        cout<<endl;
    } while ((updated) && (iters < 10));
    cout << "iterations" << iters << endl;
    delete[] k_i;
}

void lp(size_t vertices_count,   int *e_array,   int *v_array,   int *labels) {
    std::vector<  int> v(vertices_count);
    for (int i = 0; i < vertices_count; i++) {
        v[i] = i;
    }
    bool updated;
    int iters = 0;
    for (  int l = 0; l < vertices_count; ++l) {
        labels[l] = l;
        std::cout << labels[l] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;

    do {
        updated = false;
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(v.begin(), v.end(), g);
        for (  int i = 0; i < vertices_count; i++) {

              int vertice_id = v[i];
            std::cout << "WE ARE IN " << vertice_id << " VERTICE" << endl;
              int begin = v_array[vertice_id];
              int end = v_array[vertice_id + 1];
            std::map<  int, int> mp;
            for (  int j = begin; j < end; j++) {
                if (mp.count(labels[e_array[j]])) {
                    mp[labels[e_array[j]]]++;
                } else {
                    mp[labels[e_array[j]]] = 1;
                }
            }

            int label_frequence = 0;
              int decision_label = -1;

            for (auto it = mp.begin(); it != mp.end(); it++) {
                if (it->second > label_frequence)       //переделать рандомом
                {
                    cout << "Label " << it->first << " is meet" << endl;
                    label_frequence = it->second;
                    decision_label = it->first;
                }
            }

            if (vertice_id < vertices_count) {
                std::cout << "vertice " << vertice_id << " selected label" << decision_label << endl;
                auto it = mp.begin();
                for (int k = 0; it != mp.end(); k++, it++) {
                    std::cout << it->first << "label meets " << it->second << "times" << endl;
                }

            }

            if (decision_label != labels[vertice_id]) {
                labels[vertice_id] = decision_label;
                updated = true;
                std::cout << "label" << vertice_id << "updated" << std::endl;

            }

        }
        std::cout << "labels after current iteration: ";
        for (int l = 0; l < vertices_count; ++l) {
            std::cout << labels[l] << " ";
        }

        std::cout << std::endl;
        iters++;
    } while ((updated) && (iters < 10));
}
