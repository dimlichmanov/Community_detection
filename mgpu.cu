#include <iostream>
#include "./moderngpu/kernel_segsort.hxx"
#include "./moderngpu/memory.hxx"


using  namespace mgpu;

int main() {
    int count = 18;
    int num_segments = div_up(count,6);
    standard_context_t context;

    mem_t<int> segs = fill_random(0, count - 1,num_segments, true, context);
    std::vector<int> segs_host = from_mem(segs);
    for(int i = 0; i<segs_host.size();i++){
        std::cout<<segs_host[i]<<" ";
    }
    std::cout<<std::endl;

    mem_t<int> data = fill_random(0, 100, count, false, context);
    std::vector<int> data_host = from_mem(data);

    for(int i = 0; i<data_host.size();i++){
        std::cout<<data_host[i]<<" ";
    }
    std::cout<<std::endl;

    mem_t<int> values(count, context);

    segmented_sort_indices(data.data(), values.data(), count, segs.data(),
                           num_segments, less_t<int>(), context);

    std::vector<int> indeces = from_mem(values);

    for(int i = 0; i<indeces.size();i++){
        std::cout<<indeces[i]<<" ";
    }
    std::cout<<std::endl;

    for(int i = 0; i<indeces.size();i++){
        std::cout<<data_host[indeces[i]]<<" ";
    }


    
}