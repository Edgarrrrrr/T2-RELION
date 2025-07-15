#ifndef TRANSLATION_MATRIX_HANDLER_CUH
#define TRANSLATION_MATRIX_HANDLER_CUH

#include "src/acc/acc_projectorkernel_impl.h"

#include "./warp_layout.cuh"
#include "./reg_bitmap.cuh"
#include "./mma_utils.cuh"
#include "./copy_traits.cuh"

#include <assert.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <type_traits>

template <int kTransBlockSize, int kImgBlockSize, int kWarpNum, 
		  int kBlockSize,
		  typename TransMatLayout, 
          typename TransRealMatLayout, typename TransImagMatLayout>
struct TranslationMatrixHandler {

	// linear order
	/**                                  <-1->
	 *    +----------------+   -----   ^ +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
	 *    |       w0       |           1 |  0|  1|  2|  3|  4|  5|  6|  7|  8|  9| 10| 11| 12| 13| 14| 15|
	 *    +----------------+   \       v +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
	 *    |       w1       |    \        | 16| 17| 18| 19| 20| 21| 22| 23| 24| 25| 26| 27| 28| 29| 30| 31|
	 *    +----------------+     \       +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
	 *    |       w2       |
	 *    +----------------+
	 *    |       w3       |
	 *    +----------------+
	 *    |       w0       |
	 *    +----------------+
	 *    |       w1       |
	 *           ...
     *
     */
    using TransWarpLayout = WarpLayout<2, 16, LayoutMajorType::RowMajor>;

    static_assert(kTransBlockSize % 8 == 0, "kTransBlockSize must be multiple of 8");

    const int& image_size_;
    const int& translation_num_;

    static const int kTMTransPerWarpTile = 2;
    static const int kTMNumTransWarpTile = kTransBlockSize / kTMTransPerWarpTile;
    static const int kTMNumTransWarpTilePerWarp = kTMNumTransWarpTile / kWarpNum;
	static const int kTMImgPerThread = 1;

    // __shared__ uint32_t s_bitmap_buffer_[2];

	uint32_t* s_bitmap_buffer_;

    // constructor
    __device__ __forceinline__
    TranslationMatrixHandler(const int& img_size, const int& trans_num, 
							 uint32_t* bitmap_buffer) : 
      image_size_(img_size), translation_num_(trans_num), s_bitmap_buffer_(bitmap_buffer) {
		s_bitmap_buffer_[0] = 0;
		s_bitmap_buffer_[1] = 0;
    }
	
	__device__ __forceinline__
    TranslationMatrixHandler(const int& img_size, const int& trans_num) : 
      image_size_(img_size), translation_num_(trans_num), s_bitmap_buffer_(nullptr) {
    }

	__device__ __forceinline__
	void store_translation_matrix_to_global(
		// out
		float* g_buffer, // size:: kTransBlockSize * kImgBlockSize + kTransBlockSize
		DeviceBitmap& bitmap,
		// in
		const int bitmap_set_pos,
		TransMatLayout& trans_mat,
		const int tid
	) {
		float4* s_trans_mat_float4 = reinterpret_cast<float4*>(trans_mat.data());
		float4* g_buffer_float4 = reinterpret_cast<float4*>(g_buffer);

		// store translation matrix
		static_assert(kTransBlockSize * kImgBlockSize % 4 == 0, "kTransBlockSize * kImgBlockSize must be multiple of 4");
		static_assert(kTransBlockSize % 4 == 0, "kTransBlockSize must be multiple of 4");
		#pragma unroll
		for (int i = tid; i < kTransBlockSize * kImgBlockSize * 2 / 4; i += kBlockSize) {
			// __stcg(&g_buffer_float4[i], s_trans_mat_float4[i]);
			g_buffer_float4[i] = s_trans_mat_float4[i];
			// __stcg(&g_buffer_float4[i], {(float)blockIdx.x, (float)blockIdx.x, (float)blockIdx.x, (float)blockIdx.x});
		}

		// fence
		__threadfence();
		// __threadfence_system();
		__syncthreads();

		// set bitmap
		// TODO: modify the bitmap to use voting, ensuring that all threads set the same bit
		if (tid == 0) {
			bitmap.set(bitmap_set_pos, true);
		}
	}

	__device__ __forceinline__
	void async_load_translation_matrix_from_global_to_shared(
		// out
		TransMatLayout& trans_mat,
		// in
		float* g_buffer, // size:: kTransBlockSize * kImgBlockSize + kTransBlockSize
		DeviceBitmap& bitmap,
		const int bitmap_set_pos,
		const int double_buffer_idx,
		const int tid

	) {
		assert(s_bitmap_buffer_ != nullptr);
		assert(double_buffer_idx >= 0 && double_buffer_idx < 2);

		float4* g_buffer_float4 = reinterpret_cast<float4*>(g_buffer);
		float4* s_trans_mat_float4 = reinterpret_cast<float4*>(trans_mat.data());
		// Order matters! 
		// first load bitmap
		__syncthreads();
		assert(bitmap_set_pos >= 0);
		// load bitmap
		if (tid == 0) {
			bitmap.get_async(bitmap_set_pos, s_bitmap_buffer_[double_buffer_idx]);
			// printf("bid %3d get_async %3d mod2 %3d\n", (int)blockIdx.x, (int)bitmap_set_pos, (int)bitmap_set_pos % 2);
		}
		// copy_traits::copy_async_commit();
		// __syncthreads();
		// __threadfence();

		// load translation matrix
		#pragma unroll
		for (int i = tid; i < kTransBlockSize * kImgBlockSize * 2 / 4; i += kBlockSize) {
			copy_traits::copy_async<float4, float4, 128, copy_traits::CacheOperator::kCacheAtGlobalLevel>(
				s_trans_mat_float4[i], g_buffer_float4[i]);
		}

		copy_traits::copy_async_commit();
	}
	
	__device__ __forceinline__
	bool sync_and_check_bitmap(
		// in
		TransMatLayout& trans_mat,
		DeviceBitmap& bitmap,
		const int bitmap_set_pos,
		const int double_buffer_idx,
		const int tid,
		const int wait_n
	) {
		// check bitmap
		assert(s_bitmap_buffer_ != nullptr);
		assert(double_buffer_idx >= 0 && double_buffer_idx < 2);
		assert(wait_n >= 0 && wait_n < 2);
		bool bitmap_set = false;
		__syncthreads();
		// s_bitmap_buffer_ is in shared memory, tid 0 stores it. So after a sync, all threads
		// can read the same value
		bitmap_set = bitmap.get_dump(bitmap_set_pos, s_bitmap_buffer_[double_buffer_idx]);
		// if (bitmap_set && tid == 0) {
		// 	printf("bid %3d get_dump %3d mod2 %3d true\n", (int)blockIdx.x, (int)bitmap_set_pos, (int)bitmap_set_pos % 2);
		// }
		// copy_traits::copy_wait<1>();
		if (wait_n == 0) {
			copy_traits::copy_wait<0>();
		} else if (wait_n == 1) {
			copy_traits::copy_wait<1>();
		} else if (wait_n == 2) {
			copy_traits::copy_wait<2>();
		} else {
			assert(false);
		}

		// if (tid == 0)
		// 	s_bitmap_buffer_[bitmap_set_pos % 2] = 0xFFFFFFFF;

		// float4* s_trans_mat_float4 = reinterpret_cast<float4*>(trans_mat.data());
		// bitmap_set = true;
		// float4 tmp;
		// #pragma unroll
		// for (int i = tid; i < kTransBlockSize * kImgBlockSize * 2 / 4; i += kBlockSize) {
		// 	tmp = s_trans_mat_float4[i];
		// 	if (__float_as_int(tmp.x) == 0xFFFFFFFF ||
		// 		__float_as_int(tmp.y) == 0xFFFFFFFF ||
		// 		__float_as_int(tmp.z) == 0xFFFFFFFF ||
		// 		__float_as_int(tmp.w) == 0xFFFFFFFF) {
		// 			bitmap_set = false;
		// 	}
		// }
		
		// __syncthreads();
		// atomicAnd(&s_bitmap_buffer_[bitmap_set_pos % 2], bitmap_set ? 0xFFFFFFFF : 0x00000000);
		// __syncthreads();

		// bitmap_set = s_bitmap_buffer_[bitmap_set_pos % 2] == 0xFFFFFFFF;
		
		// // if (tid == 0 && bitmap_set) {
		// // 	// clear the bitmap
		// // 	printf("!\n");
		// // }
		return bitmap_set;
	}

	__device__ __forceinline__
	void construct_translation_matrix(
		// out
		TransRealMatLayout s_trans_real_mat_block_swizzle,
		TransImagMatLayout s_trans_imag_mat_block_swizzle,
		float* s_trans_pow2_accumulator, // size: kTransBlockSize
		// in
		const float* s_trans_xy,      // size: kTransBlockSize * 2
		const float* s_img_real_imag, // size: kImgBlockSize * 2
		const float* s_fcoor_xy,      // size: kImgBlockSize * 2
		const float* s_corr_div_2,    // size: kImgBlockSize
        const int img_block_idx,
        const int trans_block_idx,
        const int warp_id,
        const int lane_id
    ) {
		XFLOAT reg_trans_xy[kTMNumTransWarpTilePerWarp][2]; // x x (x, y)

		XFLOAT reg_img_real_imag[2]; // (real, imag)
        XFLOAT reg_fcoor_xy[2]; // (x, y)
		XFLOAT reg_corr_div_2;

		XFLOAT reg_trans_real_imag[kTMNumTransWarpTilePerWarp][2]; // 8 x (real, imag)

		int s_img_idx = TransWarpLayout::get_col_idx(lane_id);
		
		// load img, fcoor, corr_div_2
		*reinterpret_cast<float2*>(&reg_img_real_imag[0])
			= reinterpret_cast<const float2*>(s_img_real_imag) [s_img_idx];
		*reinterpret_cast<float2*>(&reg_fcoor_xy[0])
			= reinterpret_cast<const float2*>(s_fcoor_xy) [s_img_idx];
		reg_corr_div_2 = s_corr_div_2[s_img_idx];

		#pragma unroll
		for (int i = 0; i < kTMNumTransWarpTilePerWarp; i ++) {
			int s_trans_idx = (i * kWarpNum + warp_id) * TransWarpLayout::rows 
							   + TransWarpLayout::get_row_idx(lane_id);
			
			*reinterpret_cast<float2*>(&reg_trans_xy[i][0]) 
				= reinterpret_cast<const float2*>(s_trans_xy)[s_trans_idx];
			
			XFLOAT& x = reg_fcoor_xy[0];
			XFLOAT& y = reg_fcoor_xy[1];
			XFLOAT& tx = reg_trans_xy[i][0];
			XFLOAT& ty = reg_trans_xy[i][1];
			XFLOAT& real = reg_img_real_imag[0];
			XFLOAT& imag = reg_img_real_imag[1];

			XFLOAT s, c;

			// __sincosf(x * tx + y * ty, &s, &c);
			sincosf(x * tx + y * ty, &s, &c);

			reg_trans_real_imag[i][0] = c * real - s * imag;
			reg_trans_real_imag[i][1] = c * imag + s * real;

			s_trans_real_mat_block_swizzle(s_trans_idx, s_img_idx) = -2 * reg_trans_real_imag[i][0] * reg_corr_div_2;
			s_trans_imag_mat_block_swizzle(s_trans_idx, s_img_idx) = -2 * reg_trans_real_imag[i][1] * reg_corr_div_2;

			XFLOAT magnitude_squared_sum = (reg_trans_real_imag[i][0] * reg_trans_real_imag[i][0] 
											+ reg_trans_real_imag[i][1] * reg_trans_real_imag[i][1])
											* reg_corr_div_2;
			magnitude_squared_sum = TransWarpLayout::reduce_by_rows(magnitude_squared_sum);
			if (TransWarpLayout::get_col_idx(lane_id) == 0) {
				s_trans_pow2_accumulator[s_trans_idx] += magnitude_squared_sum;
			}
		}
    }

	__device__ __forceinline__
	void construct_translation_matrix(
		// out
		TransRealMatLayout s_trans_real_mat_block_swizzle,
		TransImagMatLayout s_trans_imag_mat_block_swizzle,
		// in
		const float* s_trans_xy,      // size: kTransBlockSize * 2
		const float* s_img_real_imag, // size: kImgBlockSize * 2
		const float* s_fcoor_xy,      // size: kImgBlockSize * 2
		const float* s_corr_div_2,    // size: kImgBlockSize
        const int img_block_idx,
        const int trans_block_idx,
        const int warp_id,
        const int lane_id
    ) {
		XFLOAT reg_trans_xy[kTMNumTransWarpTilePerWarp][2]; // x x (x, y)

		XFLOAT reg_img_real_imag[2]; // (real, imag)
        XFLOAT reg_fcoor_xy[2]; // (x, y)
		XFLOAT reg_corr_div_2;

		XFLOAT reg_trans_real_imag[kTMNumTransWarpTilePerWarp][2]; // 8 x (real, imag)

		int s_img_idx = TransWarpLayout::get_col_idx(lane_id);
		
		// load img, fcoor, corr_div_2
		*reinterpret_cast<float2*>(&reg_img_real_imag[0])
			= reinterpret_cast<const float2*>(s_img_real_imag) [s_img_idx];
		*reinterpret_cast<float2*>(&reg_fcoor_xy[0])
			= reinterpret_cast<const float2*>(s_fcoor_xy) [s_img_idx];
		reg_corr_div_2 = s_corr_div_2[s_img_idx];

		#pragma unroll
		for (int i = 0; i < kTMNumTransWarpTilePerWarp; i ++) {
			int s_trans_idx = (i * kWarpNum + warp_id) * TransWarpLayout::rows 
							   + TransWarpLayout::get_row_idx(lane_id);
			
			*reinterpret_cast<float2*>(&reg_trans_xy[i][0]) 
				= reinterpret_cast<const float2*>(s_trans_xy)[s_trans_idx];
			
			XFLOAT& x = reg_fcoor_xy[0];
			XFLOAT& y = reg_fcoor_xy[1];
			XFLOAT& tx = reg_trans_xy[i][0];
			XFLOAT& ty = reg_trans_xy[i][1];
			XFLOAT& real = reg_img_real_imag[0];
			XFLOAT& imag = reg_img_real_imag[1];

			XFLOAT s, c;

			// __sincosf(x * tx + y * ty, &s, &c);
			sincosf(x * tx + y * ty, &s, &c);

			reg_trans_real_imag[i][0] = c * real - s * imag;
			reg_trans_real_imag[i][1] = c * imag + s * real;

			s_trans_real_mat_block_swizzle(s_trans_idx, s_img_idx) = -2 * reg_trans_real_imag[i][0] * reg_corr_div_2;
			s_trans_imag_mat_block_swizzle(s_trans_idx, s_img_idx) = -2 * reg_trans_real_imag[i][1] * reg_corr_div_2;
		}
    }
};



#endif // ORIENT_MATRIX_HANDLER_CUH