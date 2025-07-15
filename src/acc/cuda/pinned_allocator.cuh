# pragma once
#ifdef _CUDA_ENABLED
# include "helper_cuda.h"
# include <cuda.h>
#endif
# include <omp.h>
# include <vector>
# include <iostream>
# include <iomanip>

#define WSIZE 16
#define DSIZE 32
// maxsize一定得是偶数否则无法对齐
#define MAXSIZE 30
#define ALIGN(size) (((size) + (DSIZE - 1)) & ~(DSIZE - 1))

#define PACK(size, alloc) ((size)|(alloc))
#define GET(p) (*(size_t *)(p))
#define PUT(p,val) ((*(size_t *)(p)) = (val))

#define GET_SIZE(bp) (GET(bp) & ~0x7)
#define GET_CHUNK_ID(bp) (GET(HDRP(bp)+8))
#define SET_CHUNK_ID(bp,id) PUT(HDRP(bp)+8, id)
#define GET_ALLOC(bp) (GET(bp) & 0x1)

#define HDRP(bp) ((char *)(bp)-WSIZE)
#define FTRP(bp) ((char *)(bp)+GET_SIZE(HDRP(bp))-DSIZE)                

#define PREV_BLKP(bp) ((char *)(bp) - GET_SIZE( ((char *)(bp)-DSIZE) ))
#define NEXT_BLKP(bp) ((char *)(bp) + GET_SIZE( ((char *)(bp)-WSIZE) ))

#define GET_FIRST(size) ((size_t *)(long)(GET((char*)start_ptr+WSIZE*size)))  

#define PREV_FREE_BLOCK(bp) ((size_t *)(long)(GET(bp)))
#define NEXT_FREE_BLOCK(bp) ((size_t *)(long)(GET((char *)(bp)+WSIZE)))

// #define PINNED_ALLOCATOR_DEBUG
// #define PINNED_ALLOCATOR_LOG
#define USE_PINNED_MEMORY

struct Chunk
{
    void* start_ptr = nullptr;
    size_t block_size = 0;
    size_t spare_size = 0;
    int chunk_id=0;
    //用隐式空闲列表维护内存

    //插入一个以bp为首的内存块
    void insert_block(void* bp)
    {   
        size_t size=GET_SIZE(HDRP(bp));
        int num=fit_size(size);
        if(GET_FIRST(num)==NULL)
        {
            PUT((char*)start_ptr+num*WSIZE,(size_t)bp);
            PUT(bp,0);
            PUT((char *)bp+WSIZE,0);
        }
        else
        {
            PUT((char *)bp+WSIZE,(size_t)GET_FIRST(num));
            PUT(GET_FIRST(num),(size_t)bp);
            PUT(bp,0);
            PUT((char*)start_ptr+num*WSIZE,(size_t)bp);
        }
        SET_CHUNK_ID(bp, chunk_id);
    }

    //删除一个以前就是空闲的块
    void delete_block(void* bp)
    {
        size_t size=GET_SIZE(HDRP(bp));
        #ifdef PINNED_ALLOCATOR_DEBUG
        {
            printf("delete size: %zu\n", size);
            std::cout<<"delete ["<<(void*)((char*)bp-WSIZE)<<", "<<(void*)((char*)bp+size-WSIZE)<<"]"<<std::endl;
            fflush(stdout);
        }
        #endif
        int num=fit_size(size);
        if(PREV_FREE_BLOCK(bp)==NULL)
            PUT((char*)start_ptr+num*WSIZE,(size_t)NEXT_FREE_BLOCK(bp));
        if(PREV_FREE_BLOCK(bp)!=NULL)
            PUT((char*)PREV_FREE_BLOCK(bp)+WSIZE,(size_t)NEXT_FREE_BLOCK(bp));
        if(NEXT_FREE_BLOCK(bp)!=NULL)
            PUT(NEXT_FREE_BLOCK(bp),(size_t)PREV_FREE_BLOCK(bp));
    }

    void print_hex(void* ptr, size_t size) //调试用
    {
        unsigned char* byte_ptr = reinterpret_cast<unsigned char*>(ptr);
    
        for (size_t i = 0; i < size; ++i) {
            std::cout << std::hex << std::setw(2) << std::setfill('0') 
                      << static_cast<int>(byte_ptr[i]) << " ";
        }
        std::cout << std::dec << std::endl;  // 恢复十进制格式
    }

    void init(void* ptr, size_t size, int chunk_id) {
        #ifdef PINNED_ALLOCATOR_LOG
        {
            std::cout<<"[Pinned Memory Allocator] Allocating pinned memory of size: " << size << std::endl;
        }
        #endif
        if ((size_t)ptr % DSIZE != 0) {
            ptr = (void*)((size_t)ptr + WSIZE);
            if ((size_t)ptr % DSIZE != 0) {
                std::cerr << "[Pinned Memory Allocator] Error: Pointer is not aligned to DSIZE." << std::endl;
                return;
            }
            size -= WSIZE;
        }
        start_ptr = ptr;
        spare_size = size;
        block_size = size;
        this->chunk_id = chunk_id;
        for (int i=0;i<=MAXSIZE;++i)
            PUT((char *)start_ptr+i*WSIZE,0);
        PUT((char *)start_ptr+(MAXSIZE+1)*WSIZE,PACK(DSIZE,1));
        PUT((char *)start_ptr+(MAXSIZE+2)*WSIZE,PACK(DSIZE,1));
        // std::cout<<"first block  "<<(void*)( (char *)start_ptr+MAXSIZE*WSIZE )<<std::endl;
        spare_size -= (MAXSIZE + 4)*WSIZE;
        char* bp = (char *)start_ptr+(4+MAXSIZE)*WSIZE;
        PUT(HDRP(bp),PACK((spare_size-DSIZE),0));
        PUT(FTRP(bp),PACK((spare_size-DSIZE),0));
        PUT(HDRP(NEXT_BLKP(bp)),PACK(0,1));

        coalesce(bp);
        //head_0, head_1, ……, [对齐][block_0][block_0][block_1]
    }

    void* place(void* bp,size_t size)
    {
        size_t block_size=GET_SIZE(HDRP(bp));
        #ifdef PINNED_ALLOCATOR_DEBUG
        {
            printf("place size: %zu\n", size);
            printf("block size: %zu\n", block_size);
            fflush(stdout);
        }
        #endif
        delete_block(bp);
        if((block_size-size)>=(2*DSIZE))
        {
            PUT(HDRP(bp),PACK(block_size-size,0));
            PUT(FTRP(bp),PACK(block_size-size,0));
            //set block_id
            void* old_bp=bp;
            bp=NEXT_BLKP(bp);
            PUT(HDRP(bp),PACK(size,1));
            PUT(FTRP(bp),PACK(size,1));
            SET_CHUNK_ID(bp, chunk_id);
            insert_block(old_bp);
            spare_size -= size;
        }
        else
        {
            PUT(HDRP(bp),PACK(block_size,1));
            PUT(FTRP(bp),PACK(block_size,1));
            spare_size -= block_size;
        }
        return bp;
    }

    int fit_size(size_t size)
    {
        for (int i=4;i<MAXSIZE;++i)
            if(size<=(1<<i)) return i-4;
        return MAXSIZE-4;
    }

    void* coalesce(void *bp)
    {
        size_t prev_alloc = GET_ALLOC(FTRP(PREV_BLKP(bp)));
        size_t next_alloc = GET_ALLOC(HDRP(NEXT_BLKP(bp)));
        size_t size=GET_SIZE(HDRP(bp));

        #ifdef PINNED_ALLOCATOR_DEBUG
        {
            std::cout<<"PREV_BLKP  "<<(void*)FTRP(PREV_BLKP(bp))<<std::endl;
            printf("coalesce size: %zu\n", size);
            printf("prev alloc: %zu\n", prev_alloc);
            printf("next alloc: %zu\n", next_alloc);
            fflush(stdout);
        }
        #endif

        if(prev_alloc && next_alloc)
        {
            insert_block(bp);
            return bp;
        }
        else if(prev_alloc && !next_alloc)
        {
            delete_block(NEXT_BLKP(bp));
            size+=GET_SIZE(HDRP(NEXT_BLKP(bp)));
            PUT(HDRP(bp),PACK(size,0));
            PUT(FTRP(bp),PACK(size,0));
        }   
        else if(!prev_alloc && next_alloc)
        {
            delete_block(PREV_BLKP(bp));
            size+=GET_SIZE(HDRP(PREV_BLKP(bp)));
            PUT(FTRP(bp),PACK(size,0));
            PUT(HDRP(PREV_BLKP(bp)),PACK(size,0));
            bp=PREV_BLKP(bp);
        } 
        else
        {
            #ifdef PINNED_ALLOCATOR_DEBUG
            {
                std::cout<<"NEXT_BLKP "<<(void*)NEXT_BLKP(bp)<<std::endl;
                std::cout<<"PREV_BLKP "<<(void*)PREV_BLKP(bp)<<std::endl;
                fflush(stdout);
            }
            #endif
            delete_block(NEXT_BLKP(bp));
            delete_block(PREV_BLKP(bp));
            size+=GET_SIZE(HDRP(PREV_BLKP(bp)))+GET_SIZE(FTRP(NEXT_BLKP(bp)));
            PUT(HDRP(PREV_BLKP(bp)),PACK(size,0));
            PUT(FTRP(NEXT_BLKP(bp)),PACK(size,0));
            bp=PREV_BLKP(bp);
        }
        #ifdef PINNED_ALLOCATOR_DEBUG
        {
            std::cout<<"coalesce bp  "<<(void*)bp<<std::endl;
            fflush(stdout);
        }
        #endif
        insert_block(bp);
        return bp;
    }

    void* find_fit(size_t size)
    {
        int num=fit_size(size);
        size_t* bp;
        while(num<MAXSIZE)
        {
            bp=GET_FIRST(num);
            while(bp!=NULL)
            {
                if(GET_SIZE(HDRP(bp))>=size) return (void*)bp;
                bp=NEXT_FREE_BLOCK(bp);
            }
            num++;
        }
        return NULL;
    }

    bool alloc(void** ptr, size_t size)
    {
        if (spare_size < size) return false;
        void* bp;
        bp = find_fit(size);
        if (bp == NULL) return false;
        (*ptr) = place(bp, size);
        return true;
    }

    void free(void* ptr)
    {
        if (ptr == nullptr) return;
        size_t size = GET_SIZE(HDRP(ptr));
        #ifdef PINNED_ALLOCATOR_DEBUG
        {
            printf("free size: %zu\n", size);
            fflush(stdout);
        }
        #endif
        spare_size += size;
        PUT(HDRP(ptr), PACK(size, 0));
        PUT(FTRP(ptr), PACK(size, 0));
        coalesce(ptr);
    }

    void show_list()
    {
        std::cout<<"block control:["<<start_ptr<<","<<(void*)( (char*)start_ptr + block_size )<<"]"<<std::endl;
        //访问链表，顺序输出
        for (int i=0;i<MAXSIZE;++i)
        {
            void* now = GET_FIRST(i);
            if (now == NULL) continue;
            printf("size %lld:",1LL<<(i+4));
            while(now!=NULL)
            {
                //[start,end]
                std::cout<<"["<<(void*)((char*)now-WSIZE)<<", "<<(void*)((char*)now+GET_SIZE(HDRP(now))-WSIZE)<<"]--";
                now = NEXT_FREE_BLOCK(now);
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }

    void clear() {
        if (start_ptr) {
            #ifdef USE_PINNED_MEMORY
            {
                checkCudaErrors(cudaFreeHost(start_ptr));
            }
            #else
            {
                free(start_ptr);
            }
            #endif
            start_ptr = nullptr;
        }
    }
};

struct PinnedAllocator
{
    const size_t chunk_size = 256 * 1024 * 1024;
    std::vector<Chunk> chunks;
    int chunk_cnt = 0;

    int alloc(void** ptr, size_t size)
    {
        // size = ALIGN(size+DSIZE);
        size = ALIGN(size+8*DSIZE);
        for (int i = chunk_cnt - 1; i >= 0; --i) {
            if (chunks[i].alloc(ptr, size)) {
                return 0 ;
            }
        }
        AllocNewChunk(size);
        if (chunks[chunk_cnt - 1].alloc(ptr, size)) {
            return 0;
        }
        //刚分配完足够大就失败是不正常的
        std::cerr << "[Pinned Memory Allocator] Error: Allocation failed after allocating a new chunk." << std::endl;
        return -1;//-1表示分配失败了
    }

    void AllocNewChunk(size_t size)
    {
        //实际内容+head*maxsize+序言块+结尾块
        size_t true_size = size + (MAXSIZE + 8)*WSIZE;
        true_size = (true_size+chunk_size-1)/chunk_size*chunk_size;
        if(true_size>=chunk_size)
        {
            #ifdef PINNED_ALLOCATOR_LOG
            {
                std::cout<<"[Pinned Memory Allocator] Allocating " << size << ", new chunk of size: " << true_size << std::endl;
                fflush(stdout);
            }
            #endif
        }
        //如果单块内存的需求>chunk_size,则申请chunk_size的倍数
        void* ptr = nullptr;
        #ifdef USE_PINNED_MEMORY
        {
            cudaError_t err = cudaHostAlloc(&ptr, true_size, cudaHostAllocDefault);
            if (err != cudaSuccess) {
                std::cerr << "[Pinned Memory Allocator] Pinned memory allocation failed: " << cudaGetErrorString(err) << std::endl;
                return;
            }
        }
        #else
        {
            ptr = malloc(true_size);
            if (ptr == nullptr) {
                std::cerr << "[Pageable Memory Allocator] Memory allocation failed." << std::endl;
                return;
            }
        }
        #endif
        
        //show分配的部分
        #ifdef PINNED_ALLOCATOR_LOG
        {
            std::cout<<"[Pinned Memory Allocator] allocate ["<<(void*)ptr<<","<<(void*)((char*)ptr+true_size)<<"]"<<std::endl;
        }
        #endif
        Chunk chunk;
        chunk.init(ptr, true_size, chunk_cnt);
        chunks.push_back(chunk);
        chunk_cnt++;
    }

    void showinfo()
    {
        chunks[0].show_list();
    }

    size_t clear()
    {
        size_t total_size = 0;
        for (int i = chunk_cnt - 1; i >= 0; --i) {
            total_size += chunks[i].block_size;
            chunks[i].clear();
        }
        chunks.clear();
        chunk_cnt = 0;
        return total_size;
    }
};

extern void initialize();
extern int pin_alloc(void** ptr, size_t size);
extern void pin_free(void* ptr);
extern void pin_free_all();



// template <typename T>
// class Allocator {
// public:
//     void* pin_alloc(size_t size)
//     {
//         void* ptr;
//         checkCudaErrors(cudaMallocHost(&ptr, sizeof(T) * size));
//         return ptr;
//     }

//     void pin_free(void* ptr) {
//         checkCudaErrors(cudaFreeHost(ptr));
//     }

// };


// enum ValueType {
//     TYPE_INT,
//     TYPE_FLOAT,
//     TYPE_DOUBLE
// };