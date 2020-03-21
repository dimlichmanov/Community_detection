#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include "fstream"
#include "lp.h"
#include "map"

using namespace std;


void louvain(size_t vertices_count, unsigned int *e_array, unsigned int *v_array, unsigned int *labels, unsigned float *weights, unsigned int *dest_labels){
    std::vector<unsigned int> v(vertices_count);
    unsigned float modularity = 0;
    for(int i =0;i<vertices_count;i++){
        v[i] = i;
    }

    unsigned float m = 0;
    for(unsigned int i =0; i<edges_count;i+=2){
        m+=weights[i];
    }

    do {
        updated = false;
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(v.begin(), v.end(), g);
        for(unsigned int i = 0;i<vertices_count;i++){ //searching shuffled array
            unsigned float local_modularity;
            unsigned int vertice_id = v[i];
            unsigned int begin  = v_array[vertice_id];
            unsigned int end = v_array[vertice_id + 1];
            unsigned float weight_i = 0;
            unsigned float weight_j = 0;

            for(unsigned int j = begin; j < end;j++){
                weight_i+=weights[j];
            } //ki is counted
            unsigned int decision_label = vertices_count; //unreachable maximum label instead of -1
            for(unsigned int j = begin; j < end;j++){
                float max_gain = 0;
                for(unsigned l = v_array[e_array[j]]; l<v_array[e_array[j] + 1];l++) {
                    weight_j+=weights[l];
                }
                float gain_modularity = (weights[j] - weight_i * weight_j/(2*m)) / (2*m); //?
                if(gain_modularity > max_gain){
                    max_gain = gain_modularity;
                    updated = true;
                    decision_label = labels[e_array[j]];
                }

            }
            if(decision_label != vertices_count){
                labels[vertice_id] = decision_label;
            }
        }
    } while((updated)&&(iters<10));
    
}

void lp(size_t vertices_count, unsigned int *e_array, unsigned int *v_array, unsigned int *labels) {
    std::vector<unsigned int> v(vertices_count);
    for(int i =0;i<vertices_count;i++){
        v[i] = i;
    }
    bool updated;
    int iters = 0;
    for (unsigned int l = 0; l < vertices_count; ++l) {
        labels[l] =l;
        std::cout<<labels[l]<<" ";
    }
    std::cout<<std::endl;
    std::cout<<std::endl;
    std::cout<<std::endl;

    do {
        updated = false;
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(v.begin(), v.end(), g);
        for(unsigned int i = 0; i < vertices_count;i++){

            unsigned int vertice_id = v[i];
            std::cout<<"WE ARE IN "<<vertice_id<<" VERTICE"<<endl;
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

            int label_frequence = 0;
            unsigned int decision_label = -1;

            for(auto it = mp.begin(); it!= mp.end() ;it++){
                if (it->second > label_frequence)       //переделать рандомом
                {
                    cout<<"Label "<<it->first<<" is meet"<<endl;
                    label_frequence = it->second;
                    decision_label = it->first;
                }
            }

            if(vertice_id < vertices_count){
                std::cout<<"vertice "<< vertice_id <<" selected label"<<decision_label<<endl;
                auto it = mp.begin();
                for(int k =0; it!= mp.end() ;k++,it++){
                    std::cout<<it->first<<"label meets "<<it->second<<"times"<<endl;
                }

            }

            if(decision_label != labels[vertice_id]){
                labels[vertice_id] = decision_label;
                updated = true;
                std::cout<<"label"<<vertice_id<<"updated"<<std::endl;

            }

        }
        std::cout<<"labels after current iteration: ";
        for (int l = 0; l < vertices_count; ++l) {
            std::cout<<labels[l]<<" ";
        }

        std::cout<<std::endl;
        iters++;
    } while((updated)&&(iters<10));
}
