# include "pinned_allocator.cuh"
# include <omp.h>
#include <unordered_map>
#include <iostream>
#include <mutex>
#include <atomic>

const int max_thread_num = 1000;
PinnedAllocator* pinned_allocators[max_thread_num] = {nullptr};  // 初始化为 nullptr
bool is_initialized_map[max_thread_num] = {false};               // 初始化为 false
std::mutex map_mutex;
//不知道为什么，退出omp后读取这个值总是有问题
static std::atomic<int> max_thread_id(-1);

int get_omp_id()
{
    if (!omp_in_parallel())
        return 0; //表示初始线程
    return omp_get_thread_num()+1;
}

void initialize() {
    int tid = get_omp_id();
    // bool is_init = false;
    // {
    //     std::lock_guard<std::mutex> lock(map_mutex);
    //     is_init = is_initialized_map[tid];
    // }
    if (is_initialized_map[tid])
        return;
    // #pragma omp critical
    {
        if(tid == 1)
        {
            #ifdef USE_PINNED_MEMORY
            {
                std::cout << "[Pinned Memory Allocator] Thread " << tid << " initializing!" << std::endl;
            }
            #else
            {
                std::cout << "[Pageable Memory Allocator] Thread " << tid << " initializing!" << std::endl;
            }
            #endif
        }
        {
            // std::lock_guard<std::mutex> lock(map_mutex);
            is_initialized_map[tid] = true;
            pinned_allocators[tid] = new PinnedAllocator();
        }
        max_thread_id.store(std::max(max_thread_id.load(), tid), std::memory_order_relaxed);
        //输出现在的max_thread_id用于调试
        // if(tid == 0)
            // std::cout << "[Pinned Memory Allocator] Thread " << tid << " initialized! max_thread_id: " << max_thread_id.load() << std::endl;
    }
}

int pin_alloc(void** ptr, size_t size) 
{
    initialize();
    int tid = get_omp_id();
    #ifdef PINNED_ALLOCATOR_DEBUG
    {
        std::cout << "[Pinned Memory Allocator] pin_alloc"<<std::endl;
    }
    #endif
    int tmp = pinned_allocators[tid]->alloc(ptr, size);
    #ifdef PINNED_ALLOCATOR_DEBUG
    {
        std::cout << "[Pinned Memory Allocator] pin_alloc end"<<std::endl;
        pinned_allocator->showinfo();
    }
    #endif
    return tmp;
}

void pin_free(void* ptr) {
    if(ptr == NULL) return;
    int tid = get_omp_id();
    if (is_initialized_map[tid]==false)
    {
        // std::cout << "[Pinned Memory Allocator] free " << ptr << ", but Thread " << tid << " not initialized!" << std::endl;
        return;
    }
    //get chunk_id
    int chunk_id = GET_CHUNK_ID(ptr);
    #ifdef PINNED_ALLOCATOR_DEBUG
    {
        printf("[Pinned Memory Allocator] free pin - chunk id: %d start\n", chunk_id);
        fflush(stdout);
    }
    #endif
    // if(chunk_id!=0)
    // {
    //     printf("chunk_id: %d\n", chunk_id);
    //     fflush(stdout);
    // }
    
    if(chunk_id >= pinned_allocators[tid]->chunk_cnt)
    {
        printf("chunk_id: %d\n", chunk_id);
        fflush(stdout);
    }
    pinned_allocators[tid]->chunks[chunk_id].free(ptr);
    #ifdef PINNED_ALLOCATOR_DEBUG
    {
        printf("[Pinned Memory Allocator] free pin - chunk id: %d end\n", chunk_id);
        fflush(stdout);
        pinned_allocators[tid]->showinfo();
    }
    #endif
}

void pin_free_all() {
    int tid = get_omp_id();
    if (tid != 0)
    {
        printf("[Pinned Memory Allocator] pin_free_all should not be used in parallel!\n");
        fflush(stdout);
        return;
    }
    // #pragma omp critical
    {
        int max_value = max_thread_id.load();
        #ifdef USE_PINNED_MEMORY
        {
            std::cout << "[Pinned Memory Allocator] pin_free_all, using pinned memory!" << std::endl;
        }
        #else
        {
            std::cout << "[Pinned Memory Allocator] pin_free_all, using pageable memory!" << std::endl;
        }
        #endif
        fflush(stdout);
        //show map
        // std::cout << "[Pinned Memory Allocator] pin_free_all, map size: " << pinned_allocators.size() << std::endl;
        // for (auto it = pinned_allocators.begin(); it != pinned_allocators.end(); ++it)
        // {
        //     std::cout << "[Pinned Memory Allocator] pin_free_all, thread id: " << it->first << std::endl;
        // }
        // std::lock_guard<std::mutex> lock(map_mutex);
        for (int tid = -1; tid <= max_thread_id.load(); tid++)
        {
            if (is_initialized_map[tid])
            {
                size_t total_size = pinned_allocators[tid]->clear();
                delete pinned_allocators[tid];
                is_initialized_map[tid] = false;
                #ifdef PINNED_ALLOCATOR_DEBUG
                {
                    std::cout << "[Pinned Memory Allocator] Thread " << tid << " free!" << std::endl;
                }
                #endif
                std::cout << "[Pinned Memory Allocator] Thread " << tid << " free " << total_size<<std::endl;
            }
        }
    }
    max_thread_id.store(-1, std::memory_order_relaxed);
}