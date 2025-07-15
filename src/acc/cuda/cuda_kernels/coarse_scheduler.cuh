#ifndef COARSE_SCHEDULER_CUH
#define COARSE_SCHEDULER_CUH

#include <cuda_runtime.h>

#include <iostream>

enum class CoarseSchedulerStrategy {
  InterleavedSplitK,
  SplitK,
  Default,
};

// Basic CoarseScheduler
template <int kBlockM, int kBlockN, int kBlockK,
          CoarseSchedulerStrategy Strategy, int kStages = 1>
struct CoarseScheduler {
 public:
  CoarseScheduler() = default;
  ~CoarseScheduler() = default;
};

// =======================================================
//  Default
// =======================================================

template <int kBlockM, int kBlockN, int kBlockK>
struct CoarseScheduler<kBlockM, kBlockN, kBlockK,
                       CoarseSchedulerStrategy::Default, 1> {
 public:
  CoarseScheduler() = delete;
  ~CoarseScheduler() = default;

  __device__ __forceinline__ CoarseScheduler(const int m, const int n,
                                             const int k)
      : m_(m), n_(n), k_(k) {
    worker_num_ = gridDim.x * gridDim.y * gridDim.z;
    worker_idx_ = blockIdx.x + blockIdx.y * gridDim.x +
                  blockIdx.z * gridDim.x * gridDim.y;

    current_work_linear_index_ = worker_idx_;
    current_work_k_num_ = get_K_block_num();
  }

  __host__ CoarseScheduler(const int m, const int n, const int k,
                           int worker_num, int worker_idx)
      : m_(m), n_(n), k_(k), worker_num_(worker_num), worker_idx_(worker_idx) {
    current_work_linear_index_ = worker_idx_;
    current_work_k_num_ = get_K_block_num();
  }

  __host__ __device__ __forceinline__ int get_M_block_num() const {
    return (m_ + kBlockM - 1) / kBlockM;
  }

  __host__ __device__ __forceinline__ int get_N_block_num() const {
    return (n_ + kBlockN - 1) / kBlockN;
  }

  __host__ __device__ __forceinline__ int get_K_block_num() const {
    // default strategy, do not split k
    return (k_ + kBlockK - 1) / kBlockK;
  }

  __host__ __device__ __forceinline__ int get_work_num() const {
    return get_M_block_num() * get_N_block_num() *
           1;  // default strategy, do not split k
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
  __host__ __device__ __forceinline__ static CoarseSchedulerStrategy get_strategy() {
    return CoarseSchedulerStrategy::Default;
  }

  __device__ __forceinline__ bool has_work() {
    return current_work_linear_index_ < get_work_num();
  }

  __device__ __forceinline__ int advance_to_next_work() {
    current_work_linear_index_ = next_work_linear_index();
    // reset k index
    current_work_k_linear_index_ = -1;  // -1 means not started
    current_work_k_num_ = get_K_block_num();

    return current_work_linear_index_;
  }

  __device__ __forceinline__ int get_current_work_linear_index() {
    return current_work_linear_index_;
  }

  __device__ __forceinline__ int get_current_work_m_block_offset() {
    int m_block_index = current_work_linear_index_ % get_M_block_num();
    return m_block_index * kBlockM;
  }

  __device__ __forceinline__ int get_current_work_n_block_offset() {
    int n_block_index =
        (current_work_linear_index_ / get_M_block_num()) % get_N_block_num();
    return n_block_index * kBlockN;
  }

  __device__ __forceinline__ bool get_current_work_next_k_block_offset(
      int& block_k_offset) {
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
  __device__ __forceinline__ int get_current_work_k_cycle() {
    return current_work_k_linear_index_;
  }

  __device__ __forceinline__ int get_current_work_k_cycle_start() { return 0; }

  __device__ __forceinline__ int get_current_work_k_cycle_end() {
    return current_work_k_num_;
  }

  template <int mode>
  __device__ __forceinline__ int k_cycle_mod(int k_cycle) {
    assert(k_cycle >= get_current_work_k_cycle_start() &&
           k_cycle < get_current_work_k_cycle_end());
    return (k_cycle - get_current_work_k_cycle_start()) % mode;
  }

  // __device__ __forceinline__ int is_first_k_cycle() {
  //   return current_work_k_linear_index_ == 0;
  // }

  // __device__ __forceinline__ int is_last_k_cycle() {
  //   return current_work_k_linear_index_ == current_work_k_num_ - 1;
  // }

  void print() const {
    std::cout << "Block Num: " << worker_num_ << "\n";
    std::cout << "M: " << m_ << ", N: " << n_ << ", K: " << k_ << "\n";
    std::cout << "Wave Efficiency: " << get_wave_efficiency() << "\n";
  }

  int worker_num_, worker_idx_;
  int m_, n_, k_;  // k -> n -> m

  int current_work_linear_index_;

  int current_work_k_linear_index_;  // do not split k, equals to k_cycle
  int current_work_k_num_;
};

// =======================================================
// SplitK kStages = 1
// =======================================================
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
template <int kBlockM, int kBlockN, int kBlockK>
struct CoarseScheduler<kBlockM, kBlockN, kBlockK,
                       CoarseSchedulerStrategy::SplitK, 1> {
 public:
  CoarseScheduler() = delete;
  ~CoarseScheduler() = default;

  __device__ __forceinline__ CoarseScheduler(const int m, const int n,
                                             const int k)
      : m_(m), n_(n), k_(k) {
    worker_num_ = gridDim.x * gridDim.y * gridDim.z;
    worker_idx_ = blockIdx.x + blockIdx.y * gridDim.x +
                  blockIdx.z * gridDim.x * gridDim.y;

    // total_k_block_num = get_K_block_num();
    // split_k_num_ = 3;  // split k into 2 parts
    int mn_block_num = get_M_block_num() * get_N_block_num();
    int avg_k_split = (216 + mn_block_num - 1) / mn_block_num;
    split_k_num_ = min(avg_k_split, get_K_block_num());

    current_work_linear_index_ = worker_idx_;

    init_k_block_range(current_work_linear_index_);
    while (current_work_k_num_ == 0 &&
           current_work_linear_index_ < get_work_num()) {
      current_work_linear_index_ = next_work_linear_index();
      init_k_block_range(current_work_linear_index_);
    }
  }

  // helper function
  // use work_linear_index to initialize k block range:
  // [current_work_k_start_, current_work_k_start_ + current_work_k_num_)
  __device__ __forceinline__ void init_k_block_range(int work_linear_index) {
    int k_idx = work_linear_index / (get_M_block_num() * get_N_block_num());
    int k_blocks_per_split =
        (get_K_block_num() + split_k_num_ - 1) / split_k_num_;

    current_work_k_cycle_ = -1;
    current_work_k_start_ = k_idx * k_blocks_per_split;
    current_work_k_num_ =
        min(get_K_block_num() - current_work_k_start_, k_blocks_per_split);
  }

  __device__ __forceinline__ int get_M_block_num() const {
    return (m_ + kBlockM - 1) / kBlockM;
  }

  __device__ __forceinline__ int get_N_block_num() const {
    return (n_ + kBlockN - 1) / kBlockN;
  }

  __device__ __forceinline__ int get_K_block_num() const {
    return (k_ + kBlockK - 1) / kBlockK;
  }

  __device__ __forceinline__ int get_work_num() const {
    return get_M_block_num() * get_N_block_num() * split_k_num_;
  }

  __device__ __forceinline__ int get_wave_num() const {
    return (get_work_num() + worker_num_ - 1) / worker_num_;
  }

  __device__ __forceinline__ double get_wave_efficiency() const {
    return (double)get_work_num() / (get_wave_num() * worker_num_);
  }

  __device__ __forceinline__ int next_work_linear_index() {
    return current_work_linear_index_ + worker_num_;
  }

  // exposed function

  // get scheduler strategy
  __device__ __forceinline__ static CoarseSchedulerStrategy get_strategy() {
    return CoarseSchedulerStrategy::SplitK;
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

  __device__ __forceinline__ int get_current_work_m_block_offset() {
    int m_block_index = current_work_linear_index_ % get_M_block_num();
    return m_block_index * kBlockM;
  }

  __device__ __forceinline__ int get_current_work_n_block_offset() {
    int n_block_index =
        (current_work_linear_index_ / get_M_block_num()) % get_N_block_num();
    return n_block_index * kBlockN;
  }

  __device__ __forceinline__ int get_current_work_k_split_block() {
    int k_split_block = current_work_linear_index_ /
                      (get_M_block_num() * get_N_block_num());
    return k_split_block;
  }

  __device__ __forceinline__ int get_k_block_offset_from_k_cycle(int k_cycle) {
    // k_cycle should be in [0, current_work_k_num_)
    assert(k_cycle >= 0 && k_cycle < current_work_k_num_);
    return (k_cycle + current_work_k_start_) * kBlockK;
  }

  // Advances the scheduler to the next work item by updating the current work
  // index and reinitializing the k dimension range based on the updated index.
  // Returns the newly updated current work linear index.
  __device__ __forceinline__ bool get_current_work_next_k_block_offset(
      int& block_k_offset) {
    current_work_k_cycle_++;

    if (current_work_k_cycle_ < current_work_k_num_) {
      block_k_offset = get_k_block_offset_from_k_cycle(current_work_k_cycle_);
      assert(block_k_offset >= 0 && block_k_offset < k_);
      return true;
    }
    return false;
  }

  __device__ __forceinline__ int get_current_work_k_cycle() {
    return current_work_k_cycle_;
  }

  __device__ __forceinline__ int get_current_work_k_cycle_start() { return 0; }

  __device__ __forceinline__ int get_current_work_k_cycle_end() {
    return current_work_k_num_;
  }

  template <int mode>
  __device__ __forceinline__ int k_cycle_mod(int k_cycle) {
    assert(k_cycle >= get_current_work_k_cycle_start() &&
           k_cycle < get_current_work_k_cycle_end());
    return (k_cycle - get_current_work_k_cycle_start()) % mode;
  }

  void print() const {
    std::cout << "Block Num: " << worker_num_ << "\n";
    std::cout << "M: " << m_ << ", N: " << n_ << ", K: " << k_ << "\n";
    std::cout << "Wave Efficiency: " << get_wave_efficiency() << "\n";
  }

  int worker_num_, worker_idx_;
  int split_k_num_;  // split k_block_num into split_k_num_ parts
  const int m_, n_, k_;

  int current_work_linear_index_;  // k -> n -> m

  // k cycle
  int current_work_k_cycle_;  // k_cycle will linearly increase to
                              // current_work_k_num_
  int current_work_k_num_;
  int current_work_k_start_;
};

// =======================================================
// SplitK kStages > 1
// =======================================================
template <int kBlockM, int kBlockN, int kBlockK, int kStages>
struct CoarseScheduler<kBlockM, kBlockN, kBlockK,
                       CoarseSchedulerStrategy::SplitK, kStages> {
 public:
  CoarseScheduler() = delete;
  ~CoarseScheduler() = default;

  __device__ __forceinline__ CoarseScheduler(const int m, const int n,
                                             const int k)
      : m_(m), n_(n), k_(k) {
    worker_num_ = gridDim.x * gridDim.y * gridDim.z;
    worker_idx_ = blockIdx.x + blockIdx.y * gridDim.x +
                  blockIdx.z * gridDim.x * gridDim.y;

    // total_k_block_num = get_K_block_num();
    // split_k_num_ = 3;  // split k into 2 parts
    int mn_block_num = get_M_block_num() * get_N_block_num();
    int avg_k_split = (216 + mn_block_num - 1) / mn_block_num;
    split_k_num_ = min(avg_k_split, get_K_block_num());
    current_work_linear_index_ = worker_idx_;

    init_k_block_range(current_work_linear_index_);

    while (current_work_k_num_ == 0 &&
           current_work_linear_index_ < get_work_num()) {
      current_work_linear_index_ = next_work_linear_index();
      init_k_block_range(current_work_linear_index_);
    }
  }

  // helper function
  // use work_linear_index to initialize k block range:
  // [current_work_k_start_, current_work_k_start_ + current_work_k_num_)
  __device__ __forceinline__ void init_k_block_range(int work_linear_index) {
    int k_idx = work_linear_index / (get_M_block_num() * get_N_block_num());
    int k_blocks_per_split =
        (get_K_block_num() + split_k_num_ - 1) / split_k_num_;

    // current_work_k_cycle_ = -1;
    current_work_k_cycle_ = -kStages;
    current_work_k_start_ = k_idx * k_blocks_per_split;
    current_work_k_num_ =
        min(get_K_block_num() - current_work_k_start_, k_blocks_per_split);
  }

  __device__ __forceinline__ int get_M_block_num() const {
    return (m_ + kBlockM - 1) / kBlockM;
  }

  __device__ __forceinline__ int get_N_block_num() const {
    return (n_ + kBlockN - 1) / kBlockN;
  }

  __device__ __forceinline__ int get_K_block_num() const {
    // default strategy, do not split k
    return (k_ + kBlockK - 1) / kBlockK;
  }

  __device__ __forceinline__ int get_work_num() const {
    return get_M_block_num() * get_N_block_num() * split_k_num_;
  }

  __device__ __forceinline__ int get_wave_num() const {
    return (get_work_num() + worker_num_ - 1) / worker_num_;
  }

  __device__ __forceinline__ double get_wave_efficiency() const {
    return (double)get_work_num() / (get_wave_num() * worker_num_);
  }

  __device__ __forceinline__ int next_work_linear_index() {
    return current_work_linear_index_ + worker_num_;
  }

  // exposed function

  // get scheduler strategy
  __device__ __forceinline__ static CoarseSchedulerStrategy get_strategy() {
    return CoarseSchedulerStrategy::SplitK;
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

  __device__ __forceinline__ int get_current_work_m_block_offset() {
    int m_block_index = current_work_linear_index_ % get_M_block_num();
    return m_block_index * kBlockM;
  }

  __device__ __forceinline__ int get_current_work_n_block_offset() {
    int n_block_index =
        (current_work_linear_index_ / get_M_block_num()) % get_N_block_num();
    return n_block_index * kBlockN;
  }

  __device__ __forceinline__ int get_current_work_k_split_block() {
    int k_split_block = current_work_linear_index_ /
                      (get_M_block_num() * get_N_block_num());
    return k_split_block;
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
    if (k_cycle < 0 || k_cycle >= current_work_k_num_) {
      printf("k_cycle: %d, current_work_k_num_: %d\n", k_cycle,
             current_work_k_num_);
    }
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

  void print() const {
    std::cout << "Block Num: " << worker_num_ << "\n";
    std::cout << "M: " << m_ << ", N: " << n_ << ", K: " << k_ << "\n";
    std::cout << "Wave Efficiency: " << get_wave_efficiency() << "\n";
  }

  int worker_num_, worker_idx_;
  int split_k_num_;  // split k_block_num into split_k_num_ parts
  const int m_, n_, k_;

  int current_work_linear_index_;  // k -> n -> m

  // k cycle will linearly increase to current_work_k_num_
  int current_work_k_cycle_;
  int current_work_k_num_;
  int current_work_k_start_;
};

// ========================================================
// InterleavedSplitK
// ========================================================
template <int kBlockM, int kBlockN, int kBlockK, int kStages>
struct CoarseScheduler<kBlockM, kBlockN, kBlockK,
                       CoarseSchedulerStrategy::InterleavedSplitK, kStages> {
 public:
  CoarseScheduler() = delete;
  ~CoarseScheduler() = default;

  __device__ __forceinline__ CoarseScheduler(const int m, const int n,
                                             const int k)
      : m_(m), n_(n), k_(k) {
    worker_num_ = gridDim.x * gridDim.y * gridDim.z;
    worker_idx_ = blockIdx.x + blockIdx.y * gridDim.x +
                  blockIdx.z * gridDim.x * gridDim.y;

    // total_k_block_num = get_K_block_num();
    // split_k_num_ = 3;  // split k into 2 parts
    int mn_block_num = get_M_block_num() * get_N_block_num();
    int avg_k_split = (216 + mn_block_num - 1) / mn_block_num;
    split_k_num_ = min(avg_k_split, get_K_block_num());
    current_work_linear_index_ = worker_idx_;

    init_k_block_range(current_work_linear_index_);

    while (current_work_k_num_ == 0 &&
           current_work_linear_index_ < get_work_num()) {
      current_work_linear_index_ = next_work_linear_index();
      init_k_block_range(current_work_linear_index_);
    }
  }

  __host__ CoarseScheduler(const int m, const int n, const int k,
                           int worker_num, int worker_idx)
      : worker_num_(worker_num), worker_idx_(worker_idx), m_(m), n_(n), k_(k) {
    // total_k_block_num = get_K_block_num();
    // split_k_num_ = 3;  // split k into 2 parts
    int mn_block_num = get_M_block_num() * get_N_block_num();
    int avg_k_split = (216 + mn_block_num - 1) / mn_block_num;
    split_k_num_ = min(avg_k_split, get_K_block_num());
    current_work_linear_index_ = worker_idx_;

    init_k_block_range(current_work_linear_index_);

    while (current_work_k_num_ == 0 &&
           current_work_linear_index_ < get_work_num()) {
      current_work_linear_index_ = next_work_linear_index();
      init_k_block_range(current_work_linear_index_);
    }
  }

  // helper function
  // use work_linear_index to initialize k block range:
  // [current_work_k_start_ + current_work_interleaved_k_start_,
  //  current_work_k_start_ + current_work_interleaved_k_start_ +
  //  current_work_interleaved_k_num_)
  // [current_work_k_start_ + current_work_interleaved_k_start_ +
  // current_work_interleaved_k_num_,
  //  current_work_k_start_ + currnet_work_k_num_)
  // [current_work_k_start_ + currnet_work_k_num_, current_work_k_start_ +
  // current_work_interleaved_k_start_) (wrap around)
  __host__ __device__ __forceinline__ void init_k_block_range(
      int work_linear_index) {
    int k_idx = work_linear_index / (get_M_block_num() * get_N_block_num());
    int k_blocks_per_split =
        (get_K_block_num() + split_k_num_ - 1) / split_k_num_;

    // current_work_k_cycle_ = -1;
    current_work_k_cycle_ = -kStages;
    current_work_k_start_ = k_idx * k_blocks_per_split;
    current_work_k_num_ =
        min(get_K_block_num() - current_work_k_start_, k_blocks_per_split);

    current_work_interleaved_k_num_ =
        (current_work_k_num_ + get_N_block_num() - 1) / get_N_block_num();
    current_work_interleaved_k_start_ =
        min(get_current_work_n_block_offset() / kBlockN *
                current_work_interleaved_k_num_,
            current_work_k_num_);
    assert(current_work_interleaved_k_start_ <= current_work_k_num_);

    current_work_interleaved_k_num_ =
        min(current_work_k_num_ - current_work_interleaved_k_start_,
            current_work_interleaved_k_num_);
  }

  __host__ __device__ __forceinline__ int get_M_block_num() const {
    return (m_ + kBlockM - 1) / kBlockM;
  }

  __host__ __device__ __forceinline__ int get_N_block_num() const {
    return (n_ + kBlockN - 1) / kBlockN;
  }

  __host__ __device__ __forceinline__ int get_K_block_num() const {
    return (k_ + kBlockK - 1) / kBlockK;
  }

  __host__ __device__ __forceinline__ int get_work_num() const {
    return get_M_block_num() * get_N_block_num() * split_k_num_;
  }

  __host__ __device__ __forceinline__ int get_wave_num() const {
    return (get_work_num() + worker_num_ - 1) / worker_num_;
  }

  __host__ __device__ __forceinline__ double get_wave_efficiency() const {
    return (double)get_work_num() / (get_wave_num() * worker_num_);
  }

  __host__ __device__ __forceinline__ int next_work_linear_index() {
    return current_work_linear_index_ + worker_num_;
  }

  // exposed function

  // get scheduler strategy
  __host__ __device__ __forceinline__ static CoarseSchedulerStrategy
  get_strategy() {
    return CoarseSchedulerStrategy::InterleavedSplitK;
  }

  __host__ __device__ __forceinline__ size_t get_workspace_size_bytes() {
    // 128Bypes = 1024bits aligned
    size_t bitmap_size = ((get_K_block_num() * sizeof(uint32_t) + 1023) / 1024) * 1024;
    size_t trans_mat_size =
        get_K_block_num() * kBlockM * kBlockK * 2 * sizeof(float);

    return get_M_block_num() * (bitmap_size + trans_mat_size);
  }

  __host__ __device__ __forceinline__ float* get_trans_mat_buf_from_workspace(
      uint32_t* workspace, int k_block_index) {
    assert(k_block_index >= 0 && k_block_index < get_K_block_num());
    size_t bitmap_size = ((get_K_block_num() * sizeof(uint32_t) + 1023) / 1024) * 1024;
    size_t trans_mat_size =
        get_K_block_num() * kBlockM * kBlockK * 2 * sizeof(float);
    size_t trans_mat_size_per_k_block = kBlockM * kBlockK * 2 * sizeof(float);
    size_t m_block_index = get_current_work_m_block_offset() / kBlockM;
    size_t offset_bytes =
        m_block_index * (bitmap_size + trans_mat_size) +
        (bitmap_size + k_block_index * (trans_mat_size_per_k_block));
    size_t workspace_size = get_workspace_size_bytes();

    assert(offset_bytes < get_workspace_size_bytes());
    assert(m_block_index < get_M_block_num());

    return reinterpret_cast<float*>(&workspace[offset_bytes / sizeof(float)]);
  }

  __host__ __device__ __forceinline__ uint32_t* get_bitmap_buf_from_workspace(
      uint32_t* workspace) {
    size_t bitmap_size = ((get_K_block_num() * sizeof(uint32_t) + 1023) / 1024) * 1024;
    size_t trans_mat_size =
        get_K_block_num() * kBlockM * kBlockK * 2 * sizeof(float);
    int m_block_index = get_current_work_m_block_offset() / kBlockM;
    size_t offset_bytes = m_block_index * (bitmap_size + trans_mat_size);
    assert(offset_bytes < get_workspace_size_bytes());
    return &workspace[offset_bytes / sizeof(uint32_t)];
  }

  __host__ void print_workspace_buffer(uint32_t* workspace) {
    size_t K = get_K_block_num();
    size_t M = get_M_block_num();

    // Compute bitmap size (in bytes) and convert to uint32_t count
    size_t bitmap_size_bytes = ((K * sizeof(uint32_t) + 1023) / 1024) * 1024;
    size_t bitmap_uint32_count = bitmap_size_bytes / sizeof(uint32_t);
    printf("bitmap uint32 count: %lu\n", bitmap_uint32_count);

    // Each K block
    // The size of the trans mat buffer (in bytes) for each K block:
    //   kBlockM * (kBlockK * 2) * sizeof(float)
    size_t trans_mat_bytes = kBlockM * kBlockK * 2 * sizeof(float);
    size_t per_k_block_bytes = trans_mat_bytes;
    size_t per_k_block_uint32_count = per_k_block_bytes / sizeof(uint32_t);

    // 每个 M block 内所有 K block 的总大小
    size_t total_k_block_bytes = K * per_k_block_bytes;
    size_t total_k_block_uint32_count = total_k_block_bytes / sizeof(uint32_t);

    // 每个 M block 的总大小（字节和 uint32_t 个数）
    size_t block_size_bytes = bitmap_size_bytes + total_k_block_bytes;
    size_t block_size_uint32_count = block_size_bytes / sizeof(uint32_t);

    // 遍历每个 M block
    for (size_t m = 0; m < M; ++m) {
      printf("M block %zu:\n", m);
      // 计算当前 M block 在 workspace 中的起始偏移（以 uint32_t 为单位）
      size_t m_block_offset = m * block_size_uint32_count;

      // 打印 bitmap 部分（以十六进制打印每个 uint32_t）
      printf("  Bitmap:\n    ");
      for (size_t i = 0; i < bitmap_uint32_count; ++i) {
        printf("0x%08x ", workspace[m_block_offset + i]);
      }
      printf("\n");

      // 遍历当前 M block 内的每个 K block
      for (size_t k = 0; k < K; ++k) {
        // 计算当前 K block 在当前 M block 中的起始偏移
        size_t k_block_offset =
            m_block_offset + bitmap_uint32_count + k * per_k_block_uint32_count;

        // 打印 trans mat 部分
        // 将 workspace 中的数据 reinterpret_cast 为 float*（float 与 uint32_t
        // 均为 4 字节）
        float* trans_mat = (float*)&workspace[k_block_offset];
        printf("  K block %zu: Trans Mat (%zux%zu):\n", k, kBlockM,
               kBlockK * 2);
        for (size_t row = 0; row < kBlockM; ++row) {
          printf("    ");
          for (size_t col = 0; col < kBlockK * 2; ++col) {
            size_t index = row * (kBlockK * 2) + col;
            printf("%14f ", trans_mat[index]);
          }
          printf("\n");
        }
        printf("\n");
      }
      printf("\n");
    }
  }

  __host__ __device__ __forceinline__ bool has_work() {
    return current_work_linear_index_ < get_work_num();
  }

  __host__ __device__ __forceinline__ int advance_to_next_work() {
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

  __host__ __device__ __forceinline__ int get_current_work_linear_index() {
    return current_work_linear_index_;
  }

  __host__ __device__ __forceinline__ int get_current_work_m_block_offset() {
    int m_block_index = current_work_linear_index_ % get_M_block_num();
    return m_block_index * kBlockM;
  }

  __host__ __device__ __forceinline__ int get_current_work_n_block_offset() {
    int n_block_index =
        (current_work_linear_index_ / get_M_block_num()) % get_N_block_num();
    return n_block_index * kBlockN;
  }

  __device__ __forceinline__ int get_current_work_k_split_block() {
    int k_split_block = current_work_linear_index_ /
                      (get_M_block_num() * get_N_block_num());
    return k_split_block;
  }

  __host__ __device__ __forceinline__ int get_current_work_k_cycle() {
    return current_work_k_cycle_;
  }

  __host__ __device__ __forceinline__ int get_current_work_k_cycle_start() {
    return 1 - kStages;
  }

  __host__ __device__ __forceinline__ int get_current_work_k_cycle_end() {
    return current_work_k_num_;
  }

  __host__ __device__ __forceinline__ bool k_cycle_need_store_trans_mat(
      int k_cycle) {
    // assert(k_cycle >= get_current_work_k_cycle_start()
    //  && k_cycle < get_current_work_k_cycle_end());
    return (k_cycle >= 0) && (k_cycle < current_work_interleaved_k_num_);
  }

  template <int mode>
  __host__ __device__ __forceinline__ int k_cycle_mod(int k_cycle) {
    assert(k_cycle >= get_current_work_k_cycle_start() &&
           k_cycle < get_current_work_k_cycle_end());
    return (k_cycle - get_current_work_k_cycle_start()) % mode;
  }

  __host__ __device__ __forceinline__ int get_k_block_offset_from_k_cycle(
      int k_cycle) {
    // k_cycle should be in [0, current_work_k_num_)
    assert(k_cycle >= 0 && k_cycle < current_work_k_num_);

    return (current_work_k_start_ +
            ((k_cycle + current_work_interleaved_k_start_) %
             current_work_k_num_)) *
           kBlockK;
  }

  // Advances the scheduler to the next k cycle
  // and returns true if there is a next k cycle, false otherwise.
  __host__ __device__ __forceinline__ bool get_current_work_next_k_cycle(
      int& k_cycle) {
    current_work_k_cycle_++;
    if (current_work_k_cycle_ < current_work_k_num_) {
      k_cycle = current_work_k_cycle_;
      return true;
    }
    return false;
  }

  __host__ void print() const {
    std::cout << "Block Num: " << worker_num_ << "\n";
    std::cout << "M: " << m_ << ", N: " << n_ << ", K: " << k_ << "\n";
    std::cout << "Wave Efficiency: " << get_wave_efficiency() << "\n";
  }

  int worker_num_, worker_idx_;
  int split_k_num_;  // split k_block_num into split_k_num_ parts
  const int m_, n_, k_;

  int current_work_linear_index_;  // k -> n -> m

  // k cycle
  int current_work_k_cycle_;  // k_cycle will linearly increase to
                              // current_work_k_num_
  int current_work_k_num_;
  int current_work_k_start_;

  // interleaved k
  int current_work_interleaved_k_start_;
  int current_work_interleaved_k_num_;
};

#endif  // COARSE_SCHEDULER_CUH