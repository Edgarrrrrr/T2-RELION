#ifndef FINE_SCHEDULER_CUH
#define FINE_SCHEDULER_CUH

#include <cuda_runtime.h>

#include <iostream>

enum class FineSchedulerStrategy {
  SplitK
};

// Basic FineScheduler
template <int kBlockK, FineSchedulerStrategy Strategy, int kStages>
struct FineScheduler {
 public:
  FineScheduler() = default;
  ~FineScheduler() = default;
};

// =======================================================
// SplitK kStages > 1
// =======================================================
template <int kBlockK, int kStages>
struct FineScheduler<kBlockK, FineSchedulerStrategy::SplitK, kStages> {
 public:
  FineScheduler() = delete;
  ~FineScheduler() = default;

  __device__ __forceinline__ FineScheduler(const int block_num, const int k)
      : block_num_(block_num) ,k_(k) {
    worker_num_ = gridDim.x * gridDim.y * gridDim.z;
    worker_idx_ = blockIdx.x + blockIdx.y * gridDim.x +
                  blockIdx.z * gridDim.x * gridDim.y;

    // total_k_block_num = get_K_block_num();
    // split_k_num_ = 3;  // split k into 2 parts

    int avg_k_split = (216 + block_num - 1) / block_num;
    split_k_num_ = min(avg_k_split, get_K_block_num());

    // split_k_num_ = 1;
    current_work_linear_index_ = worker_idx_;

    init_k_block_range(current_work_linear_index_);
    while (current_work_k_num_ == 0 &&
      current_work_linear_index_ < get_work_num()) {
      current_work_linear_index_ = next_work_linear_index();
      init_k_block_range(current_work_linear_index_);
    }
  }

  __host__ __forceinline__ FineScheduler(const int block_num, const int k, 
    const int worker_num, const int worker_idx)
  : worker_num_(worker_num), worker_idx_(worker_idx), block_num_(block_num) ,k_(k) {

  // total_k_block_num = get_K_block_num();
  // split_k_num_ = 3;  // split k into 2 parts

  int avg_k_split = (216 + block_num - 1) / block_num;
  split_k_num_ = min(avg_k_split, get_K_block_num());

  // split_k_num_ = 1;
  // current_work_linear_index_ = worker_idx_;

  // init_k_block_range(current_work_linear_index_);
}

  // helper function
  // use work_linear_index to initialize k block range:
  // [current_work_k_start_, current_work_k_start_ + current_work_k_num_)
  __device__ __forceinline__ void init_k_block_range(int work_linear_index) {
    int k_idx = work_linear_index / block_num_;
    int k_blocks_per_split =
        (get_K_block_num() + split_k_num_ - 1) / split_k_num_;

    // current_work_k_cycle_ = -1;
    current_work_k_cycle_ = -kStages;
    current_work_k_start_ = k_idx * k_blocks_per_split;
    current_work_k_num_ =
        min(get_K_block_num() - current_work_k_start_, k_blocks_per_split);
  }

  __host__ __device__ __forceinline__ int get_K_block_num() const {
    return (k_ + kBlockK - 1) / kBlockK;
  }

  __host__ __device__ __forceinline__ int get_work_num() const {
    return block_num_ * split_k_num_;
  }

  __host__ __device__ __forceinline__ int get_wave_num() const {
    return (get_work_num() + worker_num_ - 1) / worker_num_;
  }
  
  __host__ __device__ __forceinline__ double get_wave_efficiency() const {
    return (double)get_work_num() / (get_wave_num() * worker_num_);
  }

  __device__ __forceinline__ int next_work_linear_index() {
    return current_work_linear_index_ + worker_num_;
  }

  // exposed function

  // get scheduler strategy
  __device__ __forceinline__ static FineSchedulerStrategy get_strategy() {
    return FineSchedulerStrategy::SplitK;
  }

  __device__ __forceinline__ bool has_work() {
    return current_work_linear_index_ < get_work_num();
  }

  __device__ __forceinline__ int advance_to_next_work() {
    // update current_work_linear_index_
    current_work_linear_index_ = next_work_linear_index();
    // update k
    init_k_block_range(current_work_linear_index_);

    // if current_work_k_num_ == 0, we need to advance to the next work.
    while (current_work_k_num_ == 0 &&
           current_work_linear_index_ < get_work_num()) {
      current_work_linear_index_ = next_work_linear_index();
      init_k_block_range(current_work_linear_index_);
    }
    return current_work_linear_index_;
  }

  __device__ __forceinline__ int get_current_work_linear_index() {
    return current_work_linear_index_;
  }

  __device__ __forceinline__ int get_current_work_block_index() {
    return current_work_linear_index_ % block_num_;
  }

  __device__ __forceinline__ int get_current_work_k_split_block() {
    int k_split_block = current_work_linear_index_ /
                      (block_num_);
    return k_split_block;
  }

  // __device__ __forceinline__ int get_current_work_m_block_offset() {
  //   int m_block_index = current_work_linear_index_ % get_M_block_num();
  //   return m_block_index * kBlockM;
  // }

  // __device__ __forceinline__ int get_current_work_n_block_offset() {
  //   int n_block_index =
  //       (current_work_linear_index_ / get_M_block_num()) % get_N_block_num();
  //   return n_block_index * kBlockN;
  // }

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

  __device__ __forceinline__ int get_current_work_k_cycle() {
    return current_work_k_cycle_;
  }

  __device__ __forceinline__ int get_current_work_k_cycle_start() {
    return 1 - kStages;
  }

  __device__ __forceinline__ int get_current_work_k_cycle_end() {
    return current_work_k_num_;
  }

  template <int mode>
  __device__ __forceinline__ int k_cycle_mod(int k_cycle) {
    assert(k_cycle >= get_current_work_k_cycle_start() &&
           k_cycle < get_current_work_k_cycle_end());
    return (k_cycle - get_current_work_k_cycle_start()) % mode;
  }

  __device__ __forceinline__ int get_k_block_offset_from_k_cycle(int k_cycle) {
    // k_cycle should be in [0, current_work_k_num_)
    assert(k_cycle >= 0 && k_cycle < current_work_k_num_);
    return (k_cycle + current_work_k_start_) * kBlockK;
  }

  // Advances the scheduler to the next k cycle
  // and returns true if there is a next k cycle, false otherwise.
  __device__ __forceinline__ bool get_current_work_next_k_cycle(int& k_cycle) {
    current_work_k_cycle_++;

    if (current_work_k_cycle_ < current_work_k_num_) {
      k_cycle = current_work_k_cycle_;
      return true;
    }
    return false;
  }

  __device__ __forceinline__ void print_debug_info() const {
    printf("worker_idx : %3d k_ : %6d current_work_k_num_ : %6d current_work_k_cycle_ : %6d "
           "current_work_k_start_ : %6d"
           "current_work_k_num_ : %6d\n",
           worker_idx_,
           k_, current_work_k_num_, current_work_k_cycle_,
           current_work_k_start_, current_work_k_num_);
  }

  void print() const {
    std::cout << "Block Num: " << block_num_ << "\n";
    std::cout << "Worker Num: " << worker_num_ << "\n";
    std::cout << "Split K Num: " << split_k_num_ << "\n";
    std::cout << "Wave Efficiency: " << get_wave_efficiency() << "\n";
  }

  int worker_num_, worker_idx_;
  int split_k_num_;  // split k_block_num into split_k_num_ parts
  const int k_;
  const int block_num_;

  int current_work_linear_index_;  // k -> n -> m

  // k cycle will linearly increase to current_work_k_num_
  int current_work_k_cycle_;
  int current_work_k_num_;
  int current_work_k_start_;
};

#endif  // FINE_SCHEDULER_CUH