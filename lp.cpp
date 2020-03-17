#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include "lp.h"
#include "map"

int lp(size_t vertices_count, unsigned int *e_array, unsigned int *v_array, unsigned int *labels) {
    std::vector<unsigned int> v(vertices_count); //вектор для генерации рандомности обхода
    bool updated = false;

    do {

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(v.begin(), v.end(), g);

        for(unsigned int i = 0; i < vertices_count;i++){
            unsigned int vertice_id = v[i];
            unsigned int begin  = v_array[vertice_id];
            unsigned int end = v_array[vertice_id + 1];
            std::map<unsigned int, int> mp;
            for(unsigned int j = begin; j < end;j++){
                if (mp.count(labels[e_array[j]])){
                    mp[labels[e_array[j]]]++;
                } else{
                    mp[labels[e_array[j]]] = 1;
                }
            }
            int frequent_label = 0;
            unsigned int local_label = 0;
            auto it = mp.begin();
            for(int k =0; it!= mp.end() ;k++,it++){
                if (mp[k] > frequent_label){
                    frequent_label = mp[k];
                    local_label = it->first;
                }
            }
            if(local_label != labels[vertice_id]){
                labels[vertice_id] = local_label;
                updated = true;
            }
        }
    } while(updated);
}
