#ifndef COPY_TRAITS_CUH
#define COPY_TRAITS_CUH

#include <assert.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <type_traits>

#include "./mma_utils.cuh"

namespace copy_traits {

template <typename T>
struct always_false : std::false_type {};

enum class CacheOperator {
  kCacheAtAllLevel,    // ca cache in all levels
  kCacheAtGlobalLevel  // cg cache in L2 and below, bypassing the L1 cache
};

template <typename SourceType, typename DestinationType,
          int L2_prefetch_size = 128,
          CacheOperator cache_op = CacheOperator::kCacheAtGlobalLevel>
static __device__ __forceinline__ void copy_async(
    DestinationType& smem_ele, const SourceType& global_ele) {
  static constexpr bool is_same_size =
      sizeof(SourceType) == sizeof(DestinationType);
  static constexpr bool is_acceptable_size = sizeof(SourceType) == 4 ||
                                             sizeof(SourceType) == 8 ||
                                             sizeof(SourceType) == 16;

  static_assert(is_same_size,
                "Source and target data type must have the same size");
  static_assert(is_acceptable_size, "Only support 4, 8, 16 bytes data type");

  SourceType const* global_ptr = &global_ele;
  uint32_t smem_uint_ptr = cast_smem_ptr_to_uint(&smem_ele);

  if constexpr (cache_op == CacheOperator::kCacheAtAllLevel) {
    if constexpr (L2_prefetch_size == 64) {
      asm volatile("cp.async.ca.shared.global.L2::64B [%0], [%1], %2;\n" ::"r"(
                       smem_uint_ptr),
                   "l"(global_ptr), "n"(sizeof(SourceType)));
    } else if constexpr (L2_prefetch_size == 128) {
      asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(
                       smem_uint_ptr),
                   "l"(global_ptr), "n"(sizeof(SourceType)));
    } else if constexpr (L2_prefetch_size == 256) {
      asm volatile("cp.async.ca.shared.global.L2::256B [%0], [%1], %2;\n" ::"r"(
                       smem_uint_ptr),
                   "l"(global_ptr), "n"(sizeof(SourceType)));
    } else {
      static_assert(always_false<SourceType>::value,
                    "Invalid L2 prefetch size");
    }
  } else if constexpr (cache_op == CacheOperator::kCacheAtGlobalLevel) {
    static_assert(sizeof(SourceType) == 16,
                  "kCacheAtGlobalLevel only support 16 bytes data type");
    if constexpr (L2_prefetch_size == 64) {
      asm volatile("cp.async.cg.shared.global.L2::64B [%0], [%1], 16;\n" ::"r"(
                       smem_uint_ptr),
                   "l"(global_ptr));
    } else if constexpr (L2_prefetch_size == 128) {
      asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(
                       smem_uint_ptr),
                   "l"(global_ptr));
    } else if constexpr (L2_prefetch_size == 256) {
      asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;\n" ::"r"(
                       smem_uint_ptr),
                   "l"(global_ptr));
    } else {
      static_assert(always_false<SourceType>::value,
                    "Invalid L2 prefetch size");
    }
  } else {
    static_assert(always_false<SourceType>::value, "Invalid cache operator");
  }
}

// source_size: the size of the data in bytes to be copied from src to dst
// and must be less than cp-size. In such case, remaining bytes in destination
// dst are filled with zeros.
template <typename SourceType, typename DestinationType,
          int L2_prefetch_size = 128,
          CacheOperator cache_op = CacheOperator::kCacheAtGlobalLevel>
static __device__ __forceinline__ void copy_async(DestinationType& smem_ele,
                                                  const SourceType& global_ele,
                                                  const int source_size  //
) {
  static constexpr bool is_same_size =
      sizeof(SourceType) == sizeof(DestinationType);
  static constexpr bool is_acceptable_size = sizeof(SourceType) == 4 ||
                                             sizeof(SourceType) == 8 ||
                                             sizeof(SourceType) == 16;

  static_assert(is_same_size,
                "Source and target data type must have the same size");
  static_assert(is_acceptable_size, "Only support 4, 8, 16 bytes data type");

  assert(source_size <= sizeof(DestinationType));

  SourceType const* global_ptr = &global_ele;
  uint32_t smem_uint_ptr = cast_smem_ptr_to_uint(&smem_ele);

  if constexpr (cache_op == CacheOperator::kCacheAtAllLevel) {
    if constexpr (L2_prefetch_size == 64) {
      asm volatile(
          "cp.async.ca.shared.global.L2::64B [%0], [%1], %2, %3;\n" ::"r"(
              smem_uint_ptr),
          "l"(global_ptr), "n"(sizeof(SourceType)), "r"(source_size));
    } else if constexpr (L2_prefetch_size == 128) {
      asm volatile(
          "cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(
              smem_uint_ptr),
          "l"(global_ptr), "n"(sizeof(SourceType)), "r"(source_size));
    } else if constexpr (L2_prefetch_size == 256) {
      asm volatile(
          "cp.async.ca.shared.global.L2::256B [%0], [%1], %2, %3;\n" ::"r"(
              smem_uint_ptr),
          "l"(global_ptr), "n"(sizeof(SourceType)), "r"(source_size));
    } else {
      static_assert(always_false<SourceType>::value,
                    "Invalid L2 prefetch size");
    }
  } else if constexpr (cache_op == CacheOperator::kCacheAtGlobalLevel) {
    static_assert(sizeof(SourceType) == 16,
                  "kCacheAtGlobalLevel only support 16 bytes data type");
    if constexpr (L2_prefetch_size == 64) {
      asm volatile(
          "cp.async.cg.shared.global.L2::64B [%0], [%1], 16, %2;\n" ::"r"(
              smem_uint_ptr),
          "l"(global_ptr), "r"(source_size));
    } else if constexpr (L2_prefetch_size == 128) {
      asm volatile(
          "cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" ::"r"(
              smem_uint_ptr),
          "l"(global_ptr), "r"(source_size));
    } else if constexpr (L2_prefetch_size == 256) {
      asm volatile(
          "cp.async.cg.shared.global.L2::256B [%0], [%1], 16, %2;\n" ::"r"(
              smem_uint_ptr),
          "l"(global_ptr), "r"(source_size));
    } else {
      static_assert(always_false<SourceType>::value,
                    "Invalid L2 prefetch size");
    }
  } else {
    static_assert(always_false<SourceType>::value, "Invalid cache operator");
  }
}

static __device__ __forceinline__ void copy_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
static __device__ __forceinline__ void copy_wait() {
  if constexpr (N == 0) {
    asm volatile("cp.async.wait_all;\n" ::);
  } else {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
  }
}

}  // namespace copy_traits

#endif  // COPY_TRAITS_CUH