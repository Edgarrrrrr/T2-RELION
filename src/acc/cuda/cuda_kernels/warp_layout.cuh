#ifndef WARP_LAYOUT_CUH
#define WARP_LAYOUT_CUH

#include <assert.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <type_traits>

enum class LayoutMajorType { 
  RowMajor, 
  ColumnMajor 
};

// RowMajor layout: row D0 -> column D1 -> row (32 % D0)
template <int D1, int D0, LayoutMajorType MajorType = LayoutMajorType::RowMajor>
struct WarpLayout {
  static_assert(D0 >= 0 && D1 >= 0, "D0, D1 must be non-negative");
  static_assert(D0 < 32 && D1 < 32, "D0, D1 must be less than 32");
  static_assert(32 % D0 == 0, "32 must be divisible by D0");
  static_assert(32 % D1 == 0, "32 must be divisible by D1");
  static_assert( D0 <= (32 / D1), "D0 must be less than or equal to 32 / D1");

  // Static values for total rows and columns based on layout type
  static constexpr int rows = (MajorType == LayoutMajorType::RowMajor) ? D1 : (32 / D1);
  static constexpr int cols = (MajorType == LayoutMajorType::RowMajor) ? (32 / D1) : D1;
  static constexpr LayoutMajorType major_type = MajorType;
  static constexpr int d1 = D1;
  static constexpr int d0 = D0;

  // Disallow creating an instance of this class
  WarpLayout() = delete;
  WarpLayout(const WarpLayout&) = delete;
  WarpLayout(WarpLayout&&) = delete;
  WarpLayout& operator=(const WarpLayout&) = delete;
  WarpLayout& operator=(WarpLayout&&) = delete;

  __device__ __forceinline__
  static int get_row_idx(int lane_id) {
    if constexpr (MajorType == LayoutMajorType::ColumnMajor) {
      int row_idx = (lane_id % D0) + (lane_id / D0) / D1 * D0;
      assert(row_idx >= 0 && row_idx < (32 / D1));
      return row_idx;
    } else {  // RowMajor      
      int row_idx = (lane_id / D0) % D1;
      assert(row_idx >= 0 && row_idx < (D1));
      return row_idx;
    }
  }

  __device__ __forceinline__
  static int get_col_idx(int lane_id) {
    if constexpr (MajorType == LayoutMajorType::ColumnMajor) {
      int col_idx = (lane_id / D0) % D1;
      assert(col_idx >= 0 && col_idx < D1);
      return col_idx;
    } else {  // RowMajor
      int col_idx = (lane_id % D0) + (lane_id / D0) / D1 * D0;
      assert(col_idx >= 0 && col_idx < (32 / D1));
      return col_idx;
    }
  }

  /**
 * @brief Perform a column-wise reduction within a warp.
 * 
 * In ColumnMajor layout, each thread contributes its value, and only threads in the
 * first row of each column return the final reduced result. Other threads return any value.
 * 
 * Example (D1=4, D0=2):
 * Layout: rows: 8, cols: 4
 * +---+---+---+---+
 * |  0|  2|  4|  6|
 * +---+---+---+---+
 * |  1|  3|  5|  7|
 * +---+---+---+---+
 * |  8| 10| 12| 14|
 * +---+---+---+---+
 * |  9| 11| 13| 15|
 * +---+---+---+---+
 * | 16| 18| 20| 22|
 * +---+---+---+---+
 * | 17| 19| 21| 23|
 * +---+---+---+---+
 * | 24| 26| 28| 30|
 * +---+---+---+---+
 * | 25| 27| 29| 31|
 * +---+---+---+---+
 * t0 return t0 + t1 + t8 + t9 + t16 + t17 + t24 + t25
 * t2 return t2 + t3 + t10 + t11 + t18 + t19 + t26 + t27
 * t4 return t4 + t5 + t12 + t13 + t20 + t21 + t28 + t29
 * t6 return t6 + t7 + t14 + t15 + t22 + t23 + t30 + t31

 * @param val  The value to be reduced.
 * @return     The reduced result in the first row's threads; unspecified otherwise.
 */
  template<typename T>
  __device__ __forceinline__
  static T reduce_by_columns(T val) {
    if constexpr (MajorType == LayoutMajorType::ColumnMajor) {
      #pragma unroll
      for (int i = 1; i < D0; i *= 2) {
        val += __shfl_down_sync(0xffffffff, val, i);
      }
      #pragma unroll
      for (int i = 1; i < (32 / D0 / D1); i *= 2) {
        val += __shfl_down_sync(0xffffffff, val, i * D0 * D1);
      }
      return val;

    } else {  // RowMajor
      #pragma unroll
      for (int i = 1; i < D1; i *= 2) {
        val += __shfl_down_sync(0xffffffff, val, i * D0);
      }
      return val;
    }
  }

  template<typename T>
  __device__ __forceinline__
  static T reduce_by_rows(T val) {
    if constexpr (MajorType == LayoutMajorType::RowMajor) {
      #pragma unroll
      for (int i = 1; i < D0; i *= 2) {
        val += __shfl_down_sync(0xffffffff, val, i);
      }
      #pragma unroll
      for (int i = 1; i < (32 / D0 / D1); i *= 2) {
        val += __shfl_down_sync(0xffffffff, val, i * D0 * D1);
      }
      return val;

    } else {  // ColumnMajor
      #pragma unroll
      for (int i = 1; i < D1; i *= 2) {
        val += __shfl_down_sync(0xffffffff, val, i * D0);
      }
      return val;
    }
  }


};

// Z-order layout: ColumnMajor 8 x 4

#endif // WARP_LAYOUT_CUH