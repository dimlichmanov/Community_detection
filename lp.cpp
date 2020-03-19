#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include "fstream"
#include "lp.h"
#include "map"

using namespace std;




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
            std::cout<<"we are in "<<vertice_id<<" vertice"<<endl;
            unsigned int begin  = v_array[vertice_id];
            unsigned int end = v_array[vertice_id + 1];
            std::map<unsigned int, int> mp;

            for(unsigned int j = begin; j < end;j++){

                if(vertice_id == 3){

                    cout<<e_array[j]<<endl;
                }

                if (mp.count(labels[e_array[j]])){
                    mp[labels[e_array[j]]]++;
                } else{
                    mp[labels[e_array[j]]] = 1;
                }
            }

            int label_frequence = 0;
            unsigned int decision_label = -1;

            for(auto it = mp.begin(); it!= mp.end() ;it++){
                if (it->second >= label_frequence)
                {
                    cout<<"Label "<<it->first<<" meet"<<endl;
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

        for (int l = 0; l < vertices_count; ++l) {
            std::cout<<labels[l]<<" ";
        }

        std::cout<<std::endl;
        iters++;
    } while((updated)&&(iters<10));
}
