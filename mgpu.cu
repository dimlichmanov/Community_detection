#include <iostream>
#include "./moderngpu/kernel_segsort.hxx"
#include "./moderngpu/memory.hxx"


using  namespace mgpu;

int main() {
    int count = 18;
    int num_segments = div_up(count,6);
    standard_context_t context;


    std::cout<<num_segments<<std::endl;

    mem_t<int> segs = fill_random(0, count - 1,num_segments, true, context);
    std::vector<int> segs_host = from_mem(segs);

    for(int i = 0; i<segs_host.size();i++){
        if(i == 0){
            std::cout<<"[ "<<0<<" , "<<segs_host[0] - 1<<" ]"<<std::endl;
            std::cout<<"[ "<<segs_host[i]<<" , "<< segs_host[i+1] - 1 <<" ]"<<std::endl;
            continue;
        }
        if(i == segs_host.size() - 1){
            std::cout<<"[ "<<segs_host[segs_host.size() - 1 ]<<" , "<< count -1<<" ]"<<std::endl; ;
            continue;
        }
        std::cout<<"[ "<<segs_host[i]<<" , "<< segs_host[i+1] - 1 <<" ]"<<std::endl;
    }

    std::cout<<std::endl;

    mem_t<int> data = fill_random(0, 100, count, false, context);
    std::vector<int> data_host = from_mem(data);

    for(int i = 0; i<data_host.size();i++){
        std::cout<<data_host[i]<<" ";
    }
    std::cout<<std::endl;

    mem_t<int> values(count, context);

//    segmented_sort_indices(data.data(), values.data(), count, segs.data(),
//                           num_segments, less_t<int>(), context);
//
//    std::vector<int> indeces = from_mem(values);
//    std::vector<int> values_host = from_mem(data);
//
//    for(int i = 0; i<indeces.size();i++){
//        std::cout<<indeces[i]<<" -th element is "<< values_host[i]<<std::endl;
//    }
//    std::cout<<std::endl;

    segmented_sort(data.data(), values.data(), count, segs.data(), num_segments, less_t<int>(), context);

    std::vector<int> values_host = from_mem(data);

    for(int i = 0; i<values_host.size();i++){
        std::cout<<i<<" -th element is "<<values_host[i]<<" "<<std::endl;
    }
}