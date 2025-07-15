#ifndef REG_BITMAP_CUH
#define REG_BITMAP_CUH

#include <assert.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#include "./copy_traits.cuh"

template <size_t N>
struct RegBitmap {
  // Using 32-bit unsigned integer to store N bits, data is expected to store in
  // registers
  uint32_t data[(N + 31) / 32];

  static constexpr size_t reg_num = (N + 31) / 32;

  __device__ __host__ __forceinline__ RegBitmap() {
// Initialize all bits to 0
#pragma unroll
    for (size_t i = 0; i < reg_num; i++) {
      data[i] = 0;
    }
  }

  // Function to set a bit; index is the position of the bit, value is the value
  // to set (0 or 1)
  __device__ __host__ __forceinline__ void set(size_t index,
                                               bool value = true) {
    // Check if the index is within the range
    assert(index < N);
    // Find the corresponding uint32_t array index
    size_t arrayIndex = index / 32;
    // Find the corresponding bit index
    size_t bitIndex = index % 32;
    if (value) {
      data[arrayIndex] |= (1 << bitIndex);  // Set to 1
    } else {
      data[arrayIndex] &= ~(1 << bitIndex);  // Set to 0
    }
  }

  // Function to get a bit; index is the position of the bit
  __device__ __host__ __forceinline__ bool get(size_t index) const {
    assert(index < N);
    size_t arrayIndex = index / 32;
    size_t bitIndex = index % 32;
    return (data[arrayIndex] & (1 << bitIndex)) != 0;
  }

  __device__ __host__ __forceinline__ void clear_all() {
#pragma unroll
    for (size_t i = 0; i < reg_num; i++) {
      data[i] = 0;
    }
  }

  __device__ __host__ __forceinline__ size_t count_set_bits() const {
    size_t count = 0;
#pragma unroll
    for (size_t i = 0; i < reg_num; i++) {
      count += __popc(data[i]);
    }
    return count;
  }

  __device__ __forceinline__ void print_bits() const {
    for (size_t i = 0; i < reg_num; i++) {
      for (size_t j = 0; j < 32; j++) {
        printf("%d", (data[i] & (1 << j)) != 0);
      }
      printf(" ");
    }
    printf("\n");
  }
};

struct DeviceBitmap {
  // Using 32-bit unsigned integer to store bits, data is expected to store in
  // global memory (N + 31) / 32 32-bit unsigned integers are used to store N
  // bits
  uint32_t *data_ptr_;
  size_t bit_num_;
  __device__ __forceinline__ DeviceBitmap(uint32_t *data_ptr, size_t bit_num)
      : data_ptr_(data_ptr), bit_num_(bit_num) {}

  // Function to set a bit; index is the position of the bit, value is the value
  // to set (0 or 1)
  __device__ __forceinline__ void set(size_t index, bool value = true) {
    assert(index < bit_num_);
    // size_t arrayIndex = index / 32;
    // size_t bitIndex = index % 32;
    // if (value) {
    //   atomicOr(&data_ptr_[arrayIndex], (1 << bitIndex));  // Set to 1
    // } else {
    //   atomicAnd(&data_ptr_[arrayIndex], ~(1 << bitIndex));  // Set to 0
    // }

    volatile uint32_t *volatile_data_ptr =
        reinterpret_cast<volatile uint32_t *>(data_ptr_);

    if (value)
      // __stcg(&data_ptr_[index], 0xfefefefe);
      volatile_data_ptr[index] = 0xfefefefe;
      // __stcg(&data_ptr_[index], 0x0);
    else
      // __stcg(&data_ptr_[index], 0x0);
      volatile_data_ptr[index] = 0x0;

      // if ( threadIdx.x == 0) {
      //   printf("bitmap bid : %3d set %3lu bitmap ptr : %p     0 0x%08x 1 0x%08x 2 0x%08x 3 0x%08x 4 0x%08x 5 0x%08x 6 0x%08x 7 0x%08x 8 0x%08x 9 0x%08x 10 0x%08x 11 0x%08x 12 0x%08x 13 0x%08x 14 0x%08x 15 0x%08x 16 0x%08x 17 0x%08x 18 0x%08x 19 0x%08x 20 0x%08x 21 0x%08x 22 0x%08x\n"
      //     , blockIdx.x, index, (void*)volatile_data_ptr, volatile_data_ptr[0], volatile_data_ptr[1], volatile_data_ptr[2], volatile_data_ptr[3], volatile_data_ptr[4], volatile_data_ptr[5], volatile_data_ptr[6], volatile_data_ptr[7], volatile_data_ptr[8], volatile_data_ptr[9], volatile_data_ptr[10], volatile_data_ptr[11], volatile_data_ptr[12], volatile_data_ptr[13], volatile_data_ptr[14], volatile_data_ptr[15], volatile_data_ptr[16], volatile_data_ptr[17], volatile_data_ptr[18], volatile_data_ptr[19], volatile_data_ptr[20], volatile_data_ptr[21], volatile_data_ptr[22]);
      // }
  }

  __device__ __forceinline__ bool get(size_t index) const {
    assert(index < bit_num_);
    // size_t arrayIndex = index / 32;
    // size_t bitIndex = index % 32;

    // volatile uint32_t *volatile_data_ptr =
    //     reinterpret_cast<volatile uint32_t *>(data_ptr_);
    // uint32_t val = volatile_data_ptr[arrayIndex];
    // return (val & (1 << bitIndex)) != 0;
    return __ldcg(&data_ptr_[index]) == 0xfefefefe;
    // return __ldcg(&data_ptr_[index]) == 0;
  }

  __device__ __forceinline__ void get_async(size_t index,
                                            uint32_t &buffer) const {
    assert(index < bit_num_);
    // size_t arrayIndex = index / 32;

    volatile uint32_t *volatile_data_ptr =
        reinterpret_cast<volatile uint32_t *>(data_ptr_);
    // buffer = volatile_data_ptr[arrayIndex];

    // buffer = __ldcg(&data_ptr_[index]);
    buffer = volatile_data_ptr[index];

    // if (threadIdx.x == 0) {
    //   printf("bid %3d get_async %3d bitmap ptr : %p     0 0x%08x 1 0x%08x 2 0x%08x 3 0x%08x 4 0x%08x 5 0x%08x 6 0x%08x 7 0x%08x 8 0x%08x 9 0x%08x 10 0x%08x 11 0x%08x 12 0x%08x 13 0x%08x 14 0x%08x 15 0x%08x 16 0x%08x 17 0x%08x 18 0x%08x 19 0x%08x 20 0x%08x 21 0x%08x 22 0x%08x\n", 
    //     (int)blockIdx.x, (int)index, (void*)volatile_data_ptr, volatile_data_ptr[0], volatile_data_ptr[1], volatile_data_ptr[2], volatile_data_ptr[3], volatile_data_ptr[4], volatile_data_ptr[5], volatile_data_ptr[6], volatile_data_ptr[7], volatile_data_ptr[8], volatile_data_ptr[9], volatile_data_ptr[10], volatile_data_ptr[11], volatile_data_ptr[12], volatile_data_ptr[13], volatile_data_ptr[14], volatile_data_ptr[15], volatile_data_ptr[16], volatile_data_ptr[17], volatile_data_ptr[18], volatile_data_ptr[19], volatile_data_ptr[20], volatile_data_ptr[21], volatile_data_ptr[22]);
    // }

    // copy_traits::copy_async<uint32_t, uint32_t, 128,
    //                         copy_traits::CacheOperator::kCacheAtAllLevel>(
    //     buffer, data_ptr_[index]);
  }

  __device__ __forceinline__ bool get_dump(size_t index,
                                           uint32_t &buffer) const {
    assert(index < bit_num_);
    // size_t bitIndex = index % 32;
    // return (buffer & (1 << bitIndex)) != 0;
    return buffer == 0xfefefefe;
    // return buffer == 0;
  }
};

#endif  // REG_BITMAP_CUH