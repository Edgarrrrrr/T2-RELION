#ifndef TRANSLATION_MATRIX_HANDLER_CUH
#define TRANSLATION_MATRIX_HANDLER_CUH

#include "src/acc/acc_projectorkernel_impl.h"

#include "./warp_layout.cuh"
#include "./reg_bitmap.cuh"
#include "./mma_utils.cuh"

#include <assert.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <type_traits>

template <int kTransBlockSize, int kImgBlockSize, int kWarpNum, 
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

    
    // constructor
    __device__ __forceinline__
    TranslationMatrixHandler(const int& img_size, const int& trans_num) : 
      image_size_(img_size), translation_num_(trans_num) {
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
		XFLOAT reg_coor_idv_2;

		XFLOAT reg_trans_real_imag[kTMNumTransWarpTilePerWarp][2]; // 8 x (real, imag)

		int s_img_idx = TransWarpLayout::get_col_idx(lane_id);
		
		// load img, fcoor, corr_div_2
		*reinterpret_cast<float2*>(&reg_img_real_imag[0])
			= reinterpret_cast<const float2*>(s_img_real_imag) [s_img_idx];
		*reinterpret_cast<float2*>(&reg_fcoor_xy[0])
			= reinterpret_cast<const float2*>(s_fcoor_xy) [s_img_idx];
		reg_coor_idv_2 = s_corr_div_2[s_img_idx];

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

			__sincosf(x * tx + y * ty, &s, &c);

			reg_trans_real_imag[i][0] = c * real - s * imag;
			reg_trans_real_imag[i][1] = c * imag + s * real;

			s_trans_real_mat_block_swizzle(s_trans_idx, s_img_idx) = -2 * reg_trans_real_imag[i][0] * reg_coor_idv_2;
			s_trans_imag_mat_block_swizzle(s_trans_idx, s_img_idx) = -2 * reg_trans_real_imag[i][1] * reg_coor_idv_2;

			XFLOAT magnitude_squared_sum = (reg_trans_real_imag[i][0] * reg_trans_real_imag[i][0] 
											+ reg_trans_real_imag[i][1] * reg_trans_real_imag[i][1])
											* reg_coor_idv_2;
			magnitude_squared_sum = TransWarpLayout::reduce_by_rows(magnitude_squared_sum);
			if (TransWarpLayout::get_col_idx(lane_id) == 0) {
				s_trans_pow2_accumulator[s_trans_idx] += magnitude_squared_sum;
			}
		}
    }

};



#endif // ORIENT_MATRIX_HANDLER_CUH