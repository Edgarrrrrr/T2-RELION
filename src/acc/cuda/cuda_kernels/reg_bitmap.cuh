#ifndef REG_BITMAP_CUH
#define REG_BITMAP_CUH

#include <assert.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

template <size_t N>
struct RegBitmap {
    // Using 32-bit unsigned integer to store N bits, data is expected to store in registers
    uint32_t data[(N + 31) / 32];

    static constexpr size_t reg_num = (N + 31) / 32;

    __device__ __forceinline__
    RegBitmap() {
        // Initialize all bits to 0
        #pragma unroll
        for (size_t i = 0; i < reg_num; i++) {
            data[i] = 0;
        }
    }

    // Function to set a bit; index is the position of the bit, value is the value to set (0 or 1)
    __device__ __forceinline__ 
    void set(size_t index, bool value = true) {
        assert(index < N);                     // Check if the index is within the range
        size_t arrayIndex = index / 32;        // Find the corresponding uint32_t array index
        size_t bitIndex = index % 32;          // Find the corresponding bit index
        if (value) {
            data[arrayIndex] |= (1 << bitIndex); // Set to 1
        } else {
            data[arrayIndex] &= ~(1 << bitIndex); // Set to 0
        }
    }

    // Function to get a bit; index is the position of the bit
    __device__ __forceinline__
    bool get(size_t index) const {
        assert(index < N);              // Check if the index is within the range
        size_t arrayIndex = index / 32; // Find the corresponding uint32_t array index
        size_t bitIndex = index % 32;   // Find the corresponding bit index
        return (data[arrayIndex] & (1 << bitIndex)) != 0; // Return the value of the bit
    }

    __device__ __forceinline__
    void clear_all() {
        #pragma unroll
        for (size_t i = 0; i < reg_num; i++) {
            data[i] = 0;
        }
    }
    
    __device__ __forceinline__
    size_t count_set_bits() const {
        size_t count = 0;
        #pragma unroll
        for (size_t i = 0; i < reg_num; i++) {
            count += __popc(data[i]);
        }
        return count;
    }


    __device__ __forceinline__
    void print_bits() const {
        for (size_t i = 0; i < reg_num; i++) {
            for (size_t j = 0; j < 32; j++) {
                printf("%d", (data[i] & (1 << j)) != 0);
            }
            printf(" ");
        }
        printf("\n");
    }
};

template <size_t N>
struct DeviceBitmap {
    // Using 64-bit unsigned integer to store 64 bits, data is expected to store in global memory
    // (N + 63) / 64 64-bit unsigned integers are used to store N bits
    unsigned long long *data_ptr_;
    static constexpr size_t data_num_ = (N + 63) / 64;

    __device__ __forceinline__
    DeviceBitmap(unsigned long long *data_ptr) : data_ptr_(data_ptr) {
    }

    // Function to set a bit; index is the position of the bit, value is the value to set (0 or 1)
    __device__ __forceinline__
    void set(size_t index, bool value = true) {
        assert(index < N);                     // Check if the index is within the range
        size_t arrayIndex = index / 64;        // Find the corresponding unsigned long long array index
        size_t bitIndex = index % 64;          // Find the corresponding bit index
        if (value) {
            atomicOr(&data_ptr_[arrayIndex], (1ULL << bitIndex)); // Set to 1
            // unsigned long long val = __ldcg(&data_ptr_[arrayIndex]); // 直接从全局内存加载
            // __stcg(&data_ptr_[arrayIndex], 1ULL << bitIndex);
        } else {
            atomicAnd(&data_ptr_[arrayIndex], ~(1ULL << bitIndex)); // Set to 0
        }
    }

    __device__ __forceinline__
    bool get(size_t index) const {
        assert(index < N);
        size_t arrayIndex = index / 64;
        size_t bitIndex = index % 64;
        unsigned long long val = __ldcg(&data_ptr_[arrayIndex]); // 直接从全局内存加载
        return (val & (1ULL << bitIndex)) != 0;
    }

    __device__ __forceinline__
    void clear_all() {
        #pragma unroll
        for (size_t i = 0; i < data_num_; i++) {
            atomicExch(&data_ptr_[i], 0);
        }
    }
};


#endif // REG_BITMAP_CUH