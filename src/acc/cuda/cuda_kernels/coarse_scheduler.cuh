#include <iostream>
#include <cuda_runtime.h>

enum class CoarseSchedulerStrategy {
    InterleavedSplitK,
    SplitK,
    Default,
};

// 基础模板类，用于实现不同的策略
template <int kBlockM, int kBlockN, int kBlockK, CoarseSchedulerStrategy Strategy, int kStages = 1>
struct CoarseScheduler {
public:
    CoarseScheduler() = default;
    ~CoarseScheduler() = default;

};


// Default 策略特化
template <int kBlockM, int kBlockN, int kBlockK>
struct CoarseScheduler<kBlockM, kBlockN, kBlockK, CoarseSchedulerStrategy::Default, 1> {
public:
    CoarseScheduler() = delete;
    ~CoarseScheduler() = default;

    __device__ __forceinline__
    CoarseScheduler(const int m, const int n, const int k) {
        worker_num_ = gridDim.x * gridDim.y * gridDim.z;
        worker_idx_ = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
        m_ = m;
        n_ = n;
        k_ = k;
        current_work_linear_index_ = worker_idx_;
        current_work_k_num_ = get_K_block_num();
    }

    __device__ __forceinline__
    int get_M_block_num() const {
        return (m_ + kBlockM - 1) / kBlockM;
    }    
    
    __device__ __forceinline__
    int get_N_block_num() const {
        return (n_ + kBlockN - 1) / kBlockN;
    }
    
    __device__ __forceinline__
    int get_K_block_num() const {
        // default strategy, do not split k
        return (k_ + kBlockK - 1) / kBlockK;
    }

    __device__ __forceinline__
    int get_work_num() const {
        return get_M_block_num() * get_N_block_num() * 1; // default strategy, do not split k
    }

    __device__ __forceinline__
    int get_wave_num() const {
        return (get_work_num() + worker_num_ - 1) / worker_num_;
    }

    __device__ __forceinline__
    double get_wave_efficiency() const {
        return (double)get_work_num() / (get_wave_num() * worker_num_);
    }

    __device__ __forceinline__
    int next_work_linear_index() {
        return current_work_linear_index_ + worker_num_; 
    }

    // exposed function

    // get scheduler strategy
    __device__ __forceinline__
    static CoarseSchedulerStrategy get_strategy() {
        return CoarseSchedulerStrategy::Default;
    }

    __device__ __forceinline__
    bool has_work() {
        // if (threadIdx.x == 0) {
        //     printf("current_work_linear_index_: %d | get_work_num(): %d\n", current_work_linear_index_, get_work_num());
        // }
        return current_work_linear_index_ < get_work_num();
    }

    __device__ __forceinline__
    int advance_to_next_work() {
        current_work_linear_index_ = next_work_linear_index();
        // reset k index
        current_work_k_linear_index_ = -1; // -1 means not started
        current_work_k_num_ = get_K_block_num();

        return current_work_linear_index_;
    }

    __device__ __forceinline__
    int get_current_work_linear_index() {
        return current_work_linear_index_;
    }

    __device__ __forceinline__
    int get_current_work_m_block_offset() {
        int m_block_index = current_work_linear_index_ % get_M_block_num();
        return m_block_index * kBlockM;
    }

    __device__ __forceinline__
    int get_current_work_n_block_offset() {
        int n_block_index = (current_work_linear_index_ / get_M_block_num()) % get_N_block_num();
        return n_block_index * kBlockN;
    }

    __device__ __forceinline__
    bool get_current_work_next_k_block_offset(int& block_k_offset) {
        current_work_k_linear_index_++;
        if (current_work_k_linear_index_ < current_work_k_num_) {
            block_k_offset = current_work_k_linear_index_ * kBlockK;
            return true;
        }
        return false;
    }

    // k cycle: When traversing the k dimension, the increment of k 
    // may not be a simple linear increase. To address this, the concept
    //  of “cycle” is introduced. 
    // The cycle represents time and increases linearly. If there 
    // are n tasks in the k dimension, the number of cycles is n.
    // 
    // The reason for introducing this concept is that during the 
    // traversal of the k dimension, it may be necessary to prefetch 
    // data for subsequent cycles and manage the stages using a 
    // circular array. This helps describe which stage is being 
    // processed at each moment in time.
    __device__ __forceinline__
    int is_first_k_cycle() {
        return current_work_k_linear_index_ == 0;
    }

    __device__ __forceinline__
    int is_last_k_cycle() {
        return current_work_k_linear_index_ == current_work_k_num_ - 1;
    }

    __device__ __forceinline__
    int get_current_work_k_cycle() {
        return current_work_k_linear_index_;
    }

    void print() const {
        std::cout << "Block Num: " << worker_num_ << "\n";
        std::cout << "M: " << m_ << ", N: " << n_ << ", K: " << k_ << "\n";
        std::cout << "Wave Efficiency: " << get_wave_efficiency() << "\n";
    }


    int worker_num_, worker_idx_;
    int m_, n_, k_; // k -> n -> m

    int current_work_linear_index_;

    int current_work_k_linear_index_; // do not split k, equals to k_cycle
    int current_work_k_num_;



    // 其他可能的 Default 策略特定成员
};


// SplitK 策略特化
template <int kBlockM, int kBlockN, int kBlockK>
struct CoarseScheduler<kBlockM, kBlockN, kBlockK, CoarseSchedulerStrategy::SplitK, 1> {
public:
CoarseScheduler() = delete;
~CoarseScheduler() = default;

__device__ __forceinline__
CoarseScheduler(const int m, const int n, const int k) 
    : m_(m), n_(n), k_(k) {
    worker_num_ = gridDim.x * gridDim.y * gridDim.z;
    worker_idx_ = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

    // total_k_block_num = get_K_block_num();
    // split_k_num_ = 3; // split k into 2 parts
    int mn_block_num = get_M_block_num() * get_N_block_num();
    int avg_k_spilt = (worker_num_ +mn_block_num - 1) / mn_block_num;
    split_k_num_ = min(avg_k_spilt,get_K_block_num());
    current_work_linear_index_ = worker_idx_;

    init_k_block_range(current_work_linear_index_);
}

// helper function 
// use work_linear_index to initialize k block range: 
// [current_work_k_start_, current_work_k_start_ + current_work_k_num_)
__device__ __forceinline__
void init_k_block_range(int work_linear_index) {
    int k_idx = work_linear_index / (get_M_block_num() * get_N_block_num());
    int k_blocks_per_split = (get_K_block_num() + split_k_num_ - 1 ) / split_k_num_;
    
    current_work_k_cycle_ = -1;
    current_work_k_start_ = k_idx * k_blocks_per_split;
    current_work_k_num_ = min(get_K_block_num() - current_work_k_start_, k_blocks_per_split);
}


__device__ __forceinline__
int get_M_block_num() const {
    return (m_ + kBlockM - 1) / kBlockM;
}    

__device__ __forceinline__
int get_N_block_num() const {
    return (n_ + kBlockN - 1) / kBlockN;
}

__device__ __forceinline__
int get_K_block_num() const {
    // default strategy, do not split k
    return (k_ + kBlockK - 1) / kBlockK;
}

__device__ __forceinline__
int get_work_num() const {
    return get_M_block_num() * get_N_block_num() * split_k_num_; 
}

__device__ __forceinline__
int get_wave_num() const {
    return (get_work_num() + worker_num_ - 1) / worker_num_;
}

__device__ __forceinline__
double get_wave_efficiency() const {
    return (double)get_work_num() / (get_wave_num() * worker_num_);
}

__device__ __forceinline__
int next_work_linear_index() {
    return current_work_linear_index_ + worker_num_; 
}

// exposed function

// get scheduler strategy
__device__ __forceinline__
static CoarseSchedulerStrategy get_strategy() {
    return CoarseSchedulerStrategy::SplitK;
}
  
__device__ __forceinline__
bool has_work() {
    // if (threadIdx.x == 0) {
    //     printf("current_work_linear_index_: %d | get_work_num(): %d\n", current_work_linear_index_, get_work_num());
    // }
    if (current_work_linear_index_ < get_work_num()) {
        if (threadIdx.x == 0) {
            // printf("bid : %3d current_work_linear_index_: %3d | get_work_num(): %3d, mb : %3d  nb : %3d mk : %3d\n", worker_idx_, current_work_linear_index_, get_work_num(), get_current_work_m_block_offset() / kBlockM, get_current_work_n_block_offset() / kBlockN, current_work_linear_index_ / (get_M_block_num() * get_N_block_num()));
        }
    }
    return current_work_linear_index_ < get_work_num();
}

__device__ __forceinline__
int advance_to_next_work() {
    // update current_work_linear_index_
    current_work_linear_index_ = next_work_linear_index();
    // update k
    init_k_block_range(current_work_linear_index_);
    return current_work_linear_index_;
}

__device__ __forceinline__
int get_current_work_linear_index() {
    return current_work_linear_index_;
}

__device__ __forceinline__
int get_current_work_m_block_offset() {
    int m_block_index = current_work_linear_index_ % get_M_block_num();
    return m_block_index * kBlockM;
}

__device__ __forceinline__
int get_current_work_n_block_offset() {
    int n_block_index = (current_work_linear_index_ / get_M_block_num()) % get_N_block_num();
    return n_block_index * kBlockN;
}

__device__ __forceinline__
int get_k_block_offset_from_k_cycle(int k_cycle) {
    // k_cycle should be in [0, current_work_k_num_)
    assert(k_cycle >= 0 && k_cycle < current_work_k_num_); 
    return (k_cycle + current_work_k_start_) * kBlockK;
}

// Advances the scheduler to the next work item by updating the current work index
// and reinitializing the k dimension range based on the updated index. Returns
// the newly updated current work linear index.
__device__ __forceinline__
bool get_current_work_next_k_block_offset(int& block_k_offset) {
    current_work_k_cycle_++;

    // if (threadIdx.x == 0) {
    //     printf("  bid: %3d, k_cycle : %3d, current_work_k_start_: %d | current_work_k_num_: %d\n", worker_idx_, current_work_k_cycle_, current_work_k_start_, current_work_k_num_);
    // }

    if (current_work_k_cycle_ < current_work_k_num_) {
        block_k_offset = get_k_block_offset_from_k_cycle(current_work_k_cycle_);
        assert(block_k_offset >= 0 && block_k_offset < k_);
        return true;
    }
    return false;
}

// k cycle: When traversing the k dimension, the increment of k 
// may not be a simple linear increase. To address this, the concept
//  of “cycle” is introduced. 
// The cycle represents time and increases linearly. If there 
// are n tasks in the k dimension, the number of cycles is n.
// 
// The reason for introducing this concept is that during the 
// traversal of the k dimension, it may be necessary to prefetch 
// data for subsequent cycles and manage the stages using a 
// circular array. This helps describe which stage is being 
// processed at each moment in time.
__device__ __forceinline__
int is_first_k_cycle() {
    return current_work_k_cycle_ == 0;
}

__device__ __forceinline__
int is_last_k_cycle() {
    return current_work_k_cycle_ == current_work_k_num_ - 1;
}

__device__ __forceinline__
int get_current_work_k_cycle() {
    // if (threadIdx.x == 0) {
    //     // printf("bid: %3d, k_cycle : %3d, current_work_k_start_: %d | current_work_k_num_: %d\n", worker_idx_, current_work_k_cycle_, current_work_k_start_, current_work_k_num_);
    //     // printf("bid %3d   k_cycle : %3d\n", worker_idx_, current_work_k_cycle_);
    // }
    return current_work_k_cycle_;
}

void print() const {
    std::cout << "Block Num: " << worker_num_ << "\n";
    std::cout << "M: " << m_ << ", N: " << n_ << ", K: " << k_ << "\n";
    std::cout << "Wave Efficiency: " << get_wave_efficiency() << "\n";
}


int worker_num_, worker_idx_;
int split_k_num_; // split k_block_num into split_k_num_ parts
const int m_, n_, k_; 

int current_work_linear_index_; // k -> n -> m

// k cycle
int current_work_k_cycle_; // k_cycle will linearly increase to current_work_k_num_
int current_work_k_num_;
int current_work_k_start_;
};



// SplitK 策略特化
template <int kBlockM, int kBlockN, int kBlockK, int kStages>
struct CoarseScheduler<kBlockM, kBlockN, kBlockK, CoarseSchedulerStrategy::SplitK, kStages> {
public:
CoarseScheduler() = delete;
~CoarseScheduler() = default;

__device__ __forceinline__
CoarseScheduler(const int m, const int n, const int k) 
    : m_(m), n_(n), k_(k) {
    worker_num_ = gridDim.x * gridDim.y * gridDim.z;
    worker_idx_ = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

    // total_k_block_num = get_K_block_num();
    // split_k_num_ = 3; // split k into 2 parts
    int mn_block_num = get_M_block_num() * get_N_block_num();
    int avg_k_spilt = (worker_num_ +mn_block_num - 1) / mn_block_num;
    split_k_num_ = min(avg_k_spilt,get_K_block_num());
    current_work_linear_index_ = worker_idx_;

    init_k_block_range(current_work_linear_index_);


    while(current_work_k_num_ == 0 && current_work_linear_index_ < get_work_num()) {
        current_work_linear_index_ = next_work_linear_index();
        init_k_block_range(current_work_linear_index_);
    }
}

// helper function 
// use work_linear_index to initialize k block range: 
// [current_work_k_start_, current_work_k_start_ + current_work_k_num_)
__device__ __forceinline__
void init_k_block_range(int work_linear_index) {
    int k_idx = work_linear_index / (get_M_block_num() * get_N_block_num());
    int k_blocks_per_split = (get_K_block_num() + split_k_num_ - 1 ) / split_k_num_;
    
    // current_work_k_cycle_ = -1;
    current_work_k_cycle_ = -kStages;
    current_work_k_start_ = k_idx * k_blocks_per_split;
    current_work_k_num_ = min(get_K_block_num() - current_work_k_start_, k_blocks_per_split);
}


__device__ __forceinline__
int get_M_block_num() const {
    return (m_ + kBlockM - 1) / kBlockM;
}    

__device__ __forceinline__
int get_N_block_num() const {
    return (n_ + kBlockN - 1) / kBlockN;
}

__device__ __forceinline__
int get_K_block_num() const {
    // default strategy, do not split k
    return (k_ + kBlockK - 1) / kBlockK;
}

__device__ __forceinline__
int get_work_num() const {
    return get_M_block_num() * get_N_block_num() * split_k_num_; 
}

__device__ __forceinline__
int get_wave_num() const {
    return (get_work_num() + worker_num_ - 1) / worker_num_;
}

__device__ __forceinline__
double get_wave_efficiency() const {
    return (double)get_work_num() / (get_wave_num() * worker_num_);
}

__device__ __forceinline__
int next_work_linear_index() {
    return current_work_linear_index_ + worker_num_; 
}

// exposed function

// get scheduler strategy
__device__ __forceinline__
static CoarseSchedulerStrategy get_strategy() {
    return CoarseSchedulerStrategy::SplitK;
}

// get stage number
// static const int kStages = kStages;
  
__device__ __forceinline__
bool has_work() {
    // if (threadIdx.x == 0) {
    //     printf("current_work_linear_index_: %d | get_work_num(): %d\n", current_work_linear_index_, get_work_num());
    // }
    if (current_work_linear_index_ < get_work_num()) {
        if (threadIdx.x == 0) {
            // printf("bid : %3d current_work_linear_index_: %3d | get_work_num(): %3d, mb : %3d  nb : %3d mk : %3d\n", worker_idx_, current_work_linear_index_, get_work_num(), get_current_work_m_block_offset() / kBlockM, get_current_work_n_block_offset() / kBlockN, current_work_linear_index_ / (get_M_block_num() * get_N_block_num()));
        }
    }
    return current_work_linear_index_ < get_work_num();
}

__device__ __forceinline__
int advance_to_next_work() {
    // update current_work_linear_index_
    current_work_linear_index_ = next_work_linear_index();
    // update k
    init_k_block_range(current_work_linear_index_);

    while(current_work_k_num_ == 0 && current_work_linear_index_ < get_work_num()) {
        current_work_linear_index_ = next_work_linear_index();
        init_k_block_range(current_work_linear_index_);
    }

    return current_work_linear_index_;
}

__device__ __forceinline__
int get_current_work_linear_index() {
    return current_work_linear_index_;
}

__device__ __forceinline__
int get_current_work_m_block_offset() {
    int m_block_index = current_work_linear_index_ % get_M_block_num();
    return m_block_index * kBlockM;
}

__device__ __forceinline__
int get_current_work_n_block_offset() {
    int n_block_index = (current_work_linear_index_ / get_M_block_num()) % get_N_block_num();
    return n_block_index * kBlockN;
}

// k cycle: When traversing the k dimension, the increment of k 
// may not be a simple linear increase. To address this, the concept
//  of “cycle” is introduced. 
// The cycle represents time and increases linearly. If there 
// are n tasks in the k dimension with kStages, the number of cycles 
// is n + kStages - 1.
// 
// k cycle is in [1 - kStages, current_work_k_num_)
// 
// The reason for introducing this concept is that during the 
// traversal of the k dimension, it may be necessary to prefetch 
// data for subsequent cycles and manage the stages using a 
// circular array. This helps describe which stage is being 
// processed at each moment in time.
// __device__ __forceinline__
// int is_first_k_cycle() {
//     return current_work_k_cycle_ == 0;
// }

// __device__ __forceinline__
// int is_last_k_cycle() {
//     return current_work_k_cycle_ == current_work_k_num_ - 1;
// }

__device__ __forceinline__
int get_current_work_k_cycle() {
    return current_work_k_cycle_;
}

__device__ __forceinline__
int get_current_work_k_cycle_start() {
    return 1 - kStages;
}

__device__ __forceinline__
int get_current_work_k_cycle_end() {
    return current_work_k_num_;
}

template <int mode>
__device__ __forceinline__
int k_cycle_mod(int k_cycle) {
    assert(k_cycle >= get_current_work_k_cycle_start()
     && k_cycle < get_current_work_k_cycle_end());
    
    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("k_cycle: %3d, k_cycle_mod: %3d\n", k_cycle, (k_cycle - get_current_work_k_cycle_start()) % mode);
    // }
    return (k_cycle - get_current_work_k_cycle_start()) % mode;
}

__device__ __forceinline__
int get_k_block_offset_from_k_cycle(int k_cycle) {
    // k_cycle should be in [0, current_work_k_num_)
    // if(threadIdx.x == 0 && (k_cycle < 0 || k_cycle >= current_work_k_num_)) {
    //     printf("k_cycle: %d, current_work_k_start_: %d, current_work_k_num_: %d bool:%d %d \t worker_idx_: %d, worker_num_: %d split_k_num: %d m%d n%d k%d\n",
    //          k_cycle, current_work_k_start_, current_work_k_num_,k_cycle >= 0, k_cycle < current_work_k_num_,
    //          worker_idx_, worker_num_,split_k_num_,m_,n_,k_);
    //     printf("current_work_linear_index_:%d get_work_num:%d",current_work_linear_index_,get_work_num());
    //     // printf("\n", );
    //     }
    assert(k_cycle >= 0 && k_cycle < current_work_k_num_); 
    return (k_cycle + current_work_k_start_) * kBlockK;
}

// Advances the scheduler to the next k cycle
// and returns true if there is a next k cycle, false otherwise.
__device__ __forceinline__
bool get_current_work_next_k_cycle(int& k_cycle) {
    current_work_k_cycle_++;

    // if (threadIdx.x == 0) {
    //     printf("  bid: %3d, k_cycle : %3d, current_work_k_start_: %d | current_work_k_num_: %d\n", worker_idx_, current_work_k_cycle_, current_work_k_start_, current_work_k_num_);
    // }

    if (current_work_k_cycle_ < current_work_k_num_) {
        k_cycle = current_work_k_cycle_;
        // block_k_offset = get_k_block_offset_from_k_cycle(current_work_k_cycle_);
        // assert(block_k_offset >= 0 && block_k_offset < k_);
        return true;
    }
    return false;
}


void print() const {
    std::cout << "Block Num: " << worker_num_ << "\n";
    std::cout << "M: " << m_ << ", N: " << n_ << ", K: " << k_ << "\n";
    std::cout << "Wave Efficiency: " << get_wave_efficiency() << "\n";
}


int worker_num_, worker_idx_;
int split_k_num_; // split k_block_num into split_k_num_ parts
const int m_, n_, k_; 

int current_work_linear_index_; // k -> n -> m

// k cycle
int current_work_k_cycle_; // k_cycle will linearly increase to current_work_k_num_
int current_work_k_num_;
int current_work_k_start_;
};






// SplitK 策略特化
template <int kBlockM, int kBlockN, int kBlockK, int kStages>
struct CoarseScheduler<kBlockM, kBlockN, kBlockK, CoarseSchedulerStrategy::InterleavedSplitK, kStages> {
public:
CoarseScheduler() = delete;
~CoarseScheduler() = default;

__device__ __forceinline__
CoarseScheduler(const int m, const int n, const int k) 
    : m_(m), n_(n), k_(k) {
    worker_num_ = gridDim.x * gridDim.y * gridDim.z;
    worker_idx_ = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

    // total_k_block_num = get_K_block_num();
    split_k_num_ = 3; // split k into 2 parts
    int mn_block_num = get_M_block_num() * get_N_block_num();
    int avg_k_spilt = (worker_num_ +mn_block_num - 1) / mn_block_num;
    split_k_num_ = min(avg_k_spilt,get_K_block_num());
    current_work_linear_index_ = worker_idx_;

    current_work_linear_index_ = worker_idx_;

    init_k_block_range(current_work_linear_index_);
}

// helper function 
// use work_linear_index to initialize k block range: 
// [current_work_k_start_ + currnet_work_interleaved_k_start_, 
//  current_work_k_start_ + currnet_work_interleaved_k_start_ + currnet_work_interleaved_k_num_)
// [current_work_k_start_ + currnet_work_interleaved_k_start_ + currnet_work_interleaved_k_num_,
//  current_work_k_start_ + currnet_work_k_num_)
// [current_work_k_start_ + currnet_work_k_num_, current_work_k_start_ + currnet_work_interleaved_k_start_) (wrap around)

__device__ __forceinline__
void init_k_block_range(int work_linear_index) {
    int k_idx = work_linear_index / (get_M_block_num() * get_N_block_num());
    int k_blocks_per_split = (get_K_block_num() + split_k_num_ - 1 ) / split_k_num_;
    
    // current_work_k_cycle_ = -1;
    current_work_k_cycle_ = -kStages;
    current_work_k_start_ = k_idx * k_blocks_per_split;
    current_work_k_num_ = min(get_K_block_num() - current_work_k_start_, k_blocks_per_split);


    // current_work_linear_index_ % get_M_block_num();
    
    currnet_work_interleaved_k_num_ = (current_work_k_num_ + get_M_block_num() - 1) / get_M_block_num();
    currnet_work_interleaved_k_start_ = get_M_block_num() * currnet_work_interleaved_k_num_;
    assert(currnet_work_interleaved_k_start_ < current_work_k_num_);

    currnet_work_interleaved_k_num_ = min(current_work_k_num_ - currnet_work_interleaved_k_start_,
                                          currnet_work_interleaved_k_num_);
    

}


__device__ __forceinline__
int get_M_block_num() const {
    return (m_ + kBlockM - 1) / kBlockM;
}    

__device__ __forceinline__
int get_N_block_num() const {
    return (n_ + kBlockN - 1) / kBlockN;
}

__device__ __forceinline__
int get_K_block_num() const {
    // default strategy, do not split k
    return (k_ + kBlockK - 1) / kBlockK;
}

__device__ __forceinline__
int get_work_num() const {
    return get_M_block_num() * get_N_block_num() * split_k_num_; 
}

__device__ __forceinline__
int get_wave_num() const {
    return (get_work_num() + worker_num_ - 1) / worker_num_;
}

__device__ __forceinline__
double get_wave_efficiency() const {
    return (double)get_work_num() / (get_wave_num() * worker_num_);
}

__device__ __forceinline__
int next_work_linear_index() {
    return current_work_linear_index_ + worker_num_; 
}

// exposed function

// get scheduler strategy
__device__ __forceinline__
static CoarseSchedulerStrategy get_strategy() {
    return CoarseSchedulerStrategy::InterleavedSplitK;
}

// get stage number
// static const int kStages = kStages;
  
__device__ __forceinline__
bool has_work() {
    // if (threadIdx.x == 0) {
    //     printf("current_work_linear_index_: %d | get_work_num(): %d\n", current_work_linear_index_, get_work_num());
    // }
    if (current_work_linear_index_ < get_work_num()) {
        if (threadIdx.x == 0) {
            // printf("bid : %3d current_work_linear_index_: %3d | get_work_num(): %3d, mb : %3d  nb : %3d mk : %3d\n", worker_idx_, current_work_linear_index_, get_work_num(), get_current_work_m_block_offset() / kBlockM, get_current_work_n_block_offset() / kBlockN, current_work_linear_index_ / (get_M_block_num() * get_N_block_num()));
        }
    }
    return current_work_linear_index_ < get_work_num();
}

__device__ __forceinline__
int advance_to_next_work() {
    // update current_work_linear_index_
    current_work_linear_index_ = next_work_linear_index();
    // update k
    init_k_block_range(current_work_linear_index_);
    return current_work_linear_index_;
}

__device__ __forceinline__
int get_current_work_linear_index() {
    return current_work_linear_index_;
}

__device__ __forceinline__
int get_current_work_m_block_offset() {
    int m_block_index = current_work_linear_index_ % get_M_block_num();
    return m_block_index * kBlockM;
}

__device__ __forceinline__
int get_current_work_n_block_offset() {
    int n_block_index = (current_work_linear_index_ / get_M_block_num()) % get_N_block_num();
    return n_block_index * kBlockN;
}

// k cycle: When traversing the k dimension, the increment of k 
// may not be a simple linear increase. To address this, the concept
//  of “cycle” is introduced. 
// The cycle represents time and increases linearly. If there 
// are n tasks in the k dimension with kStages, the number of cycles 
// is n + kStages - 1.
// 
// k cycle is in [1 - kStages, current_work_k_num_)
// 
// The reason for introducing this concept is that during the 
// traversal of the k dimension, it may be necessary to prefetch 
// data for subsequent cycles and manage the stages using a 
// circular array. This helps describe which stage is being 
// processed at each moment in time.
// __device__ __forceinline__
// int is_first_k_cycle() {
//     return current_work_k_cycle_ == 0;
// }

// __device__ __forceinline__
// int is_last_k_cycle() {
//     return current_work_k_cycle_ == current_work_k_num_ - 1;
// }

__device__ __forceinline__
int get_current_work_k_cycle() {
    return current_work_k_cycle_;
}

__device__ __forceinline__
int get_current_work_k_cycle_start() {
    return 1 - kStages;
}

__device__ __forceinline__
int get_current_work_k_cycle_end() {
    return current_work_k_num_;
}

template <int mode>
__device__ __forceinline__
int k_cycle_mod(int k_cycle) {
    assert(k_cycle >= get_current_work_k_cycle_start()
     && k_cycle < get_current_work_k_cycle_end());
    
    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("k_cycle: %3d, k_cycle_mod: %3d\n", k_cycle, (k_cycle - get_current_work_k_cycle_start()) % mode);
    // }
    return (k_cycle - get_current_work_k_cycle_start()) % mode;
}

__device__ __forceinline__
int get_k_block_offset_from_k_cycle(int k_cycle) {
    // k_cycle should be in [0, current_work_k_num_)
    assert(k_cycle >= 0 && k_cycle < current_work_k_num_); 

    return (current_work_k_start_ + ((k_cycle + currnet_work_interleaved_k_start_) % current_work_k_num_) ) * kBlockK;
}

// Advances the scheduler to the next k cycle
// and returns true if there is a next k cycle, false otherwise.
__device__ __forceinline__
bool get_current_work_next_k_cycle(int& k_cycle) {
    current_work_k_cycle_++;

    // if (threadIdx.x == 0) {
    //     printf("  bid: %3d, k_cycle : %3d, current_work_k_start_: %d | current_work_k_num_: %d\n", worker_idx_, current_work_k_cycle_, current_work_k_start_, current_work_k_num_);
    // }

    if (current_work_k_cycle_ < current_work_k_num_) {
        k_cycle = current_work_k_cycle_;
        // block_k_offset = get_k_block_offset_from_k_cycle(current_work_k_cycle_);
        // assert(block_k_offset >= 0 && block_k_offset < k_);
        return true;
    }
    return false;
}


void print() const {
    std::cout << "Block Num: " << worker_num_ << "\n";
    std::cout << "M: " << m_ << ", N: " << n_ << ", K: " << k_ << "\n";
    std::cout << "Wave Efficiency: " << get_wave_efficiency() << "\n";
}


int worker_num_, worker_idx_;
int split_k_num_; // split k_block_num into split_k_num_ parts
const int m_, n_, k_; 

int current_work_linear_index_; // k -> n -> m

// k cycle
int current_work_k_cycle_; // k_cycle will linearly increase to current_work_k_num_
int current_work_k_num_;
int current_work_k_start_;

// interleaved k
int currnet_work_interleaved_k_start_;
int currnet_work_interleaved_k_num_;
};