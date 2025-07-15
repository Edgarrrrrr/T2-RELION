#ifndef FINE_MATRIX_KERNEL_IM2COL_SPLITIMG_BCF_PMO_CUH
#define FINE_MATRIX_KERNEL_IM2COL_SPLITIMG_BCF_PMO_CUH

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <fstream>

#include "src/acc/acc_projector.h"
#include "src/acc/acc_projectorkernel_impl.h"
#include "src/acc/cuda/cuda_settings.h"
#include "src/acc/cuda/cuda_kernels/cuda_device_utils.cuh"

#include "./mma_utils.cuh"
#include "./fine_scheduler.cuh"
#include "./reg_bitmap.cuh"
#include "./warp_layout.cuh"
#include "./orientation_matrix_handler.cuh"
#include "./translation_matrix_handler.cuh"
#include "./kernel_block_params.cuh"

template<typename TParams>
__launch_bounds__(128, 2)
__global__ void FineMatrixKernelIm2colSplitImgBCFProjOverlapKernel(
	XFLOAT *g_eulers,  // orientation_num * 9 (OrientBlockSize aligned)
    XFLOAT *trans_xyz, // translation_num * 3 (TransBlockSize aligned)
    Block<TParams::kTransBlockSize/TParams::kNrOverTrans, TParams::kNrOverTrans, TParams::kNrOverOrient> *blocks, // block_num
	XFLOAT *g_real, // image_size
	XFLOAT *g_imag, // image_size
	AccProjectorKernel projector,
	XFLOAT *g_corr, // image_size
	XFLOAT *g_diff2s,
	XFLOAT sum_init,
	const int translation_num,
	const int orientation_num,
	const int image_size,
    const int block_num,
	XFLOAT *g_coor_xy = nullptr);

template <typename TParams = void>
struct FineMatrixKernelIm2colSplitImgBCFProjOverlap {
  const int translation_num_;
  const int orientation_num_;
  const int image_size_;
  const int block_num_;
  const int sm_num_;

  FineMatrixKernelIm2colSplitImgBCFProjOverlap(
      int translation_num, int orientation_num, int image_size, int block_num, int sm_num = 108)
      : translation_num_(translation_num),
        orientation_num_(orientation_num),
        image_size_(image_size),
        block_num_(block_num),
        sm_num_(sm_num) {}

//   // 辅助调度函数，根据 translation_num_ 选择合适的 TParams 类型
//   template <typename Func>
//   auto dispatch_TParams(Func&& func) const {
//     if (translation_num_ <= 32)
//       return func(TypeHolder<CoarseTParam32x128_32x32>{});
//     else if (translation_num_ <= 64)
//       return func(TypeHolder<CoarseTParam64x128_32x64>{});
//     else if (translation_num_ <= 128)
//       return func(TypeHolder<CoarseTParam128x64_64x32>{});
//     else
//       return func(TypeHolder<CoarseTParam64x128_32x64>{});
//   }

  // 计算 workspace 大小
  size_t get_workspace_size_bytes() const {
    return 0;
  }

  // 运行核函数
  void run(XFLOAT *g_eulers,  // orientation_num * 9 (OrientBlockSize aligned)
		   XFLOAT *trans_xyz, // translation_num * 3 (TransBlockSize aligned)
		   Block<16, 4, 8> *blocks, // block_num
           XFLOAT *g_real,
           XFLOAT *g_imag, 
           AccProjectorKernel projector, 
           XFLOAT *g_corr,
           XFLOAT *g_diff2s, 
		   XFLOAT sum_init,
           cudaStream_t stream,
		   XFLOAT* g_coor_xy = nullptr,
           uint32_t *work_space = nullptr) {
    
    (void)work_space;

	FineScheduler<TParams::kImgBlockSize,
	FineSchedulerStrategy::SplitK,
                      2> scheduler(block_num_, image_size_, 1, 0);

      dim3 grid(sm_num_ * 2, 1, 1);
      dim3 block(TParams::kBlockSize, 1, 1);
      FineMatrixKernelIm2colSplitImgBCFProjOverlapKernel<TParams>
	  <<<grid, block, 0, stream>>>(
          g_eulers, trans_xyz, blocks, g_real, g_imag, projector, g_corr,
          g_diff2s, sum_init, translation_num_, orientation_num_, image_size_, block_num_, g_coor_xy);


    // if constexpr (!std::is_same_v<TParams, void>) {
    //   FineScheduler<TParams::kTransBlockSize,
    //                   TParams::kOrientBlockSize,
    //                   TParams::kImgBlockSize,
    //                   FineSchedulerStrategy::InterleavedSplitK,
    //                   2> scheduler(translation_num_, orientation_num_, image_size_, 1, 0);
    //   dim3 grid(sm_num_ * 2, 1, 1);
    //   dim3 block(TParams::kBlockSize, 1, 1);
    //   FineMatrixKernelIm2colSplitImgBCFProjOverlapKernel<TParams>
	//   <<<grid, block, 0, stream>>>(
    //       g_eulers, trans_x, trans_y, g_real, g_imag, projector, g_corr,
    //       g_diff2s, g_diff2s_dest, translation_num_, orientation_num_, image_size_, g_coor_xy);
    // } else {
    //   dispatch_TParams([=](auto dummy) {
    //     using SelectedTParams = typename decltype(dummy)::type;
    //     FineScheduler<SelectedTParams::kTransBlockSize,
    //                     SelectedTParams::kOrientBlockSize,
    //                     SelectedTParams::kImgBlockSize,
    //                     FineSchedulerStrategy::InterleavedSplitK,
    //                     2> scheduler(translation_num_, orientation_num_, image_size_, 1, 0);
        
	// 	dim3 grid(sm_num_ * 2, 1, 1);
    //     dim3 block(SelectedTParams::kBlockSize, 1, 1);
        
	// 	FineMatrixKernelIm2colSplitImgBCFProjOverlapKernel<SelectedTParams><<<grid, block, 0, stream>>>(
    //         g_eulers, trans_x, trans_y, g_real, g_imag, projector, g_corr,
    //         g_diff2s, g_diff2s_dest, translation_num_, orientation_num_, image_size_, g_coor_xy);
    //   });
    // }
  }

  void print_workspace_buffer(uint32_t* workspace) {
	printf("workspace buffer is empty\n");
  }
};


template<typename TParams>
__launch_bounds__(128, 2)
__global__ void FineMatrixKernelIm2colSplitImgBCFProjOverlapKernel(
	XFLOAT *g_eulers,  // orientation_num * 9 (OrientBlockSize aligned)
    XFLOAT *trans_xyz, // translation_num * 3 (TransBlockSize aligned)
    Block<TParams::kTransBlockSize/TParams::kNrOverTrans, TParams::kNrOverTrans, TParams::kNrOverOrient> *blocks, // block_num
	XFLOAT *g_real, // image_size
	XFLOAT *g_imag, // image_size
	AccProjectorKernel projector,
	XFLOAT *g_corr, // image_size
	XFLOAT *g_diff2s,
	XFLOAT sum_init,
	const int translation_num,
	const int orientation_num,
	const int image_size,
    const int block_num,
	XFLOAT *g_coor_xy) {
    using BlockType = Block<TParams::kTransBlockSize/TParams::kNrOverTrans, TParams::kNrOverTrans, TParams::kNrOverOrient>;
	
    static_assert(TParams::kBlockSize % 32 == 0, "kBlockSize must be a multiple of 32");
	static_assert(TParams::kBlockSize >= TParams::kTransBlockSize, "kBlockSize must be greater than or equal to kTransBlockSize");

    static_assert(TParams::kTransBlockSize / 4 == TParams::kOrientBlockSize / 8, "kTransBlockSize / 4 must be equal to kOrientBlockSize / 8");
	static_assert(TParams::kTransBlockSize % TParams::kWarpTransTileSize == 0, "kTransBlockSize must be a multiple of kWarpTransTileSize");
	static_assert(TParams::kOrientBlockSize % TParams::kWarpOrientTileSize == 0, "kOrientBlockSize must be a multiple of kWarpOrientTileSize");
	static_assert(TParams::kTransBlockSize % TParams::kMmaTransTileSize == 0, "kTransBlockSize must be a multiple of kMmaTransTileSize");
	static_assert(TParams::kOrientBlockSize % TParams::kMmaOrientTileSize == 0, "kOrientBlockSize must be a multiple of kMmaOrientTileSize");
	static_assert(TParams::kBlockSize / 32 == (TParams::kTransBlockSize / TParams::kWarpTransTileSize) * (TParams::kOrientBlockSize / TParams::kWarpOrientTileSize), "kBlockSize must be equal to the product of the number of warps in translation, orientation and image dimension");

	static_assert(TParams::kImgBlockSize == 16, "kImgBlockSize must be 16");
	static_assert(TParams::kImgBlockSize == TParams::kWarpImgTileSize, "kImgBlockSize must be equal to kWarpImgTileSize");

	const int tid = threadIdx.x;          // thread id in a block
	// const int bid = blockIdx.x;           // block id in a grid
	const int warp_id  = tid / 32;        // warp id in a block
	constexpr int kWarpNum = TParams::kBlockSize / 32; // number of warps in a block
	const int lane_id  = tid % 32;        // thread id in a warp

	// const int trans_block_num = (translation_num + TParams::kTransBlockSize - 1) / TParams::kTransBlockSize;
	// const int orient_block_num = (orientation_num + TParams::kOrientBlockSize - 1) / TParams::kOrientBlockSize;

	// int trans_block_idx = (bid % trans_block_num) * TParams::kTransBlockSize;
	// int orient_block_idx = (bid / trans_block_num) * TParams::kOrientBlockSize;

    int trans_block_idx = -1;
    int orient_block_idx = -1;
    int block_idx = -1;

	assert(blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0);

	FineScheduler<TParams::kImgBlockSize, 
					FineSchedulerStrategy::SplitK,
					2>
		scheduler(block_num, image_size);

	__align__(16) __shared__ XFLOAT s_trans_mat_block[2 * TParams::kTransBlockSize * TParams::kImgBlockSize];
	__align__(16) __shared__ XFLOAT s_trans_mat_block_bak[2 * TParams::kTransBlockSize * TParams::kImgBlockSize];
	
	// __shared__ XFLOAT s_orient_mat_block[2 * 2 * TParams::kOrientBlockSize * TParams::kImgBlockSize];
	__align__(16) __shared__ XFLOAT s_orient_mat_block[2 * TParams::kOrientBlockSize * TParams::kImgBlockSize];

	using TransMatLayout = SharedMemorySwizzle<float, TParams::kTransBlockSize, 2 * TParams::kImgBlockSize, 0>;
	using TransRealMatLayout = SharedMemorySwizzle<float, TParams::kTransBlockSize, TParams::kImgBlockSize, 0>;
	using TransImagMatLayout = SharedMemorySwizzle<float, TParams::kTransBlockSize, TParams::kImgBlockSize, TParams::kImgBlockSize>;
	
	using OrientMatLayout = SharedMemorySwizzle<float, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize, 0>;
	using OrientRealMatLayout = SharedMemorySwizzle<float, TParams::kOrientBlockSize, TParams::kImgBlockSize, 0>;
	using OrientImagMatLayout = SharedMemorySwizzle<float, TParams::kOrientBlockSize, TParams::kImgBlockSize, TParams::kImgBlockSize>;

	TransMatLayout s_trans_mat_block_swizzle(s_trans_mat_block);
	TransRealMatLayout s_trans_real_mat_block_swizzle(s_trans_mat_block);
	TransImagMatLayout s_trans_imag_mat_block_swizzle(s_trans_mat_block);

	TransMatLayout s_trans_mat_block_swizzle_bak(s_trans_mat_block_bak);
	TransRealMatLayout s_trans_real_mat_block_swizzle_bak(s_trans_mat_block_bak);
	TransImagMatLayout s_trans_imag_mat_block_swizzle_bak(s_trans_mat_block_bak);

	// OrientMatLayout s_orient_mat_block_swizzle(s_orient_mat_block);
	// OrientRealMatLayout s_orient_real_mat_block_swizzle(s_orient_mat_block);
	// OrientImagMatLayout s_orient_imag_mat_block_swizzle(s_orient_mat_block);

	OrientMatLayout s_orient_mat_block_swizzle[2] = {
		OrientMatLayout(s_orient_mat_block),
		OrientMatLayout(s_orient_mat_block)
		// OrientMatLayout(s_orient_mat_block + 2 * TParams::kOrientBlockSize * TParams::kImgBlockSize)
	};

	OrientRealMatLayout s_orient_real_mat_block_swizzle[2] = {
		OrientRealMatLayout(s_orient_mat_block),
		OrientRealMatLayout(s_orient_mat_block)
		// OrientRealMatLayout(s_orient_mat_block + 2 * TParams::kOrientBlockSize * TParams::kImgBlockSize)
	};
	OrientImagMatLayout s_orient_imag_mat_block_swizzle[2] = {
		OrientImagMatLayout(s_orient_mat_block),
		OrientImagMatLayout(s_orient_mat_block)
		// OrientImagMatLayout(s_orient_mat_block + 2 * TParams::kOrientBlockSize * TParams::kImgBlockSize)
	};


	OrientationMatrixHandler<TParams::kOrientBlockSize,
							 TParams::kImgBlockSize,
							 kWarpNum,
							 OrientRealMatLayout,
							 OrientImagMatLayout,
							 TransMatLayout, OrientMatLayout, 
							 TParams::kTransBlockSize, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize,
							 TParams::kWarpTransTileSize, TParams::kWarpOrientTileSize, 2 * TParams::kWarpImgTileSize,
							 TParams::kMmaTransTileSize, TParams::kMmaOrientTileSize, TParams::kMmaImgTileSize>
		orientation_matrix_handler(image_size, orientation_num);

	TranslationMatrixHandler<TParams::kTransBlockSize,
							 TParams::kImgBlockSize,
							 kWarpNum,
							 TParams::kBlockSize,
							 TransMatLayout,
							 TransRealMatLayout,
							 TransImagMatLayout>
		translation_matrix_handler(image_size, translation_num);

	// double buffer for s_corr_div_2, s_coor_x, s_coor_y
	__align__(16) __shared__ XFLOAT s_corr_div_2[2][TParams::kImgBlockSize];
	__align__(16) __shared__ XFLOAT s_coor_x[2][TParams::kImgBlockSize];
	__align__(16) __shared__ XFLOAT s_coor_y[2][TParams::kImgBlockSize];

	// ============================  new  ============================
	// double buffer
	__align__(16) __shared__ XFLOAT s_fcoor_xy[2][TParams::kImgBlockSize * 2]; // img -> x,y
	__align__(16) __shared__ XFLOAT s_img_real_imag[2][TParams::kImgBlockSize * 2]; // img -> real,imag

	// For a 2D scenario, e8 is not used, so it’s not stored in shared memory.
	// e2 and e5 are also unused, but they remain in shared memory for alignment.
	__align__(16) __shared__ XFLOAT s_eulers_scaled_head[TParams::kOrientBlockSize * 4]; // (e0 e1 e2 e3)  * projector.padding_factor
	__align__(16) __shared__ XFLOAT s_eulers_scaled_tail[TParams::kOrientBlockSize * 4]; // (e4 e5 e6 e7)  * projector.padding_factor

	__align__(16) __shared__ XFLOAT s_trans_xy[TParams::kTransBlockSize * 2]; // trans_num -> x,y 

	// reduce buffer
	__align__(16) __shared__ XFLOAT s_trans_pow2_accumulator[(TParams::kBlockSize / TParams::kTransBlockSize) * TParams::kTransBlockSize];
	// __shared__ XFLOAT s_orient_pow2_accumulator[(TParams::kBlockSize / TParams::kOrientBlockSize) * TParams::kOrientBlockSize];

	__align__(16) __shared__ XFLOAT s_trans_pow2_accumulator_bak[TParams::kTransBlockSize];
	__align__(16) __shared__ XFLOAT s_orient_pow2_accumulator[TParams::kOrientBlockSize];

	// used for dummy store shared
	__align__(16) __shared__ XFLOAT s_test_buffer[4 *  4 * 4];
	// register
	constexpr int kNumMmaTransInWarpTile = TParams::kWarpTransTileSize / TParams::kMmaTransTileSize;
	constexpr int kNumMmaOrientInWarpTile = TParams::kWarpOrientTileSize / TParams::kMmaOrientTileSize;
	constexpr int kNumMmaImgInWarpTile = TParams::kWarpImgTileSize / TParams::kMmaImgTileSize;

	constexpr int kFragmentASize = TParams::kMmaTransTileSize * TParams::kMmaImgTileSize / kWarpSize;
	constexpr int kFragmentBSize = TParams::kMmaOrientTileSize * TParams::kMmaImgTileSize / kWarpSize;
	constexpr int kFragmentCSize = TParams::kMmaTransTileSize * TParams::kMmaOrientTileSize / kWarpSize;

	XFLOAT fragment_c[kNumMmaTransInWarpTile][kNumMmaOrientInWarpTile][kFragmentCSize];

	// constexpr int kRegistersMmaPerThread = kNumMmaTransInWarpTile * kFragmentASize +
	// 									   kNumMmaOrientInWarpTile * kFragmentBSize +
	// 									   kNumMmaTransInWarpTile * kNumMmaOrientInWarpTile * kFragmentCSize;

	// #if kRegistersMmaPerThread >= 256
	// #warning "kRegistersMmaPerThread must be less than or equal to 256, otherwise register spilling will occur"
	// #endif


	// ============================= lambda function =============================
	//given current img_block_idx, load global array into corr_div_2, coord_x, coord_y
	auto load_coord_xy = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y, XFLOAT* fcoor_xy, XFLOAT* img_real_imag) {
		#pragma unroll 
		for (int i = tid; i < TParams::kImgBlockSize; i += TParams::kBlockSize) {
			if (img_block_idx + i < image_size) {
				corr_div_2[i] = g_corr[img_block_idx + i] / 2;
				img_real_imag[i * 2 + 0] = g_real[img_block_idx + i];
				img_real_imag[i * 2 + 1] = g_imag[img_block_idx + i];
				int x, y;
				pixel_index2coor_fine(img_block_idx + i, projector.imgX, projector.imgY, projector.maxR, x, y);
				coord_x[i] = x;
				coord_y[i] = y;
				if (g_coor_xy != nullptr) {
					fcoor_xy[i * 2 + 0] = g_coor_xy[(img_block_idx + i) * 2 + 0];
					fcoor_xy[i * 2 + 1] = g_coor_xy[(img_block_idx + i) * 2 + 1];
				} else {
					int x, y;
					pixel_index2coor_fine(img_block_idx + i, projector.imgX, projector.imgY, projector.maxR, x, y);
					fcoor_xy[2 * i + 0] = x;
					fcoor_xy[2 * i + 1] = y;
				}
			} else {
				corr_div_2[i] = 0.;
				img_real_imag[i * 2 + 0] = 0.;
				img_real_imag[i * 2 + 1] = 0.;
				fcoor_xy[2 * i + 0] = 0.;
				fcoor_xy[2 * i + 1] = 0.;
				coord_x[i] = 0;
				coord_y[i] = 0;
			}
		}
	};

    auto init_fragment_c = [&] () {
		if (scheduler.get_current_work_k_split_block() == 0) {
			#pragma unroll
			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
				#pragma unroll
				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
					#pragma unroll
					for (int k = 0; k < kFragmentCSize; ++k) {
						fragment_c[i][j][k] = sum_init;
						// printf("fragment_c[%d][%d][%d] = %f\n", i, j, k, fragment_c[i][j][k]);
					}
				}
			}
		} else {
			#pragma unroll
			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
				#pragma unroll
				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
					#pragma unroll
					for (int k = 0; k < kFragmentCSize; ++k) {
						fragment_c[i][j][k] = 0.0;
					}
				}
			}
		}
    };

	auto epilogue = [&] () {
		// use atomic add
        #pragma unroll
        for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
            #pragma unroll
            for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
                #pragma unroll
                for (int k = 0; k < kFragmentCSize; ++k) {
                    int m = fragment_c_m_idx_in_block<TParams>(warp_id, lane_id, i, k);
                    int n = fragment_c_n_idx_in_block<TParams>(warp_id, lane_id, j, k);
                    // modified:
					// if (block_idx >= block_num) {
					// 	printf("!!! block_idx > block_num\n");
					// 	assert(false);
					// }
					// if (m >= 64 || n >= 128) {
					// 	printf("### mn wrong\n");
					// }
                    auto weight_offset = blocks[block_idx].result_idx[m * TParams::kOrientBlockSize + n];

                    if (weight_offset != -1) {
                        // assert(m < translation_num && n < orientation_num);
						// if (isnan(fragment_c[i][j][k])){
						// 	printf("block_idx : %3d fragment_c nan!\n", block_idx);
						// }
						// if (fragment_c[i][j][k] > 1e3) {
						// 	scheduler.print_debug_info();
						// }
                        atomicAdd(&g_diff2s[weight_offset], fragment_c[i][j][k]);
						// if (g_diff2s[weight_offset] > 1e5) {
						// 	scheduler.print_debug_info2();
						// }

						// float test_e = 0;
						// for (int iii = 0; iii < 4; iii++) {
						// 	test_e += s_eulers_scaled_head[n * 4 + iii];
						// }
						// for (int iii = 0; iii < 4; iii++) {
						// 	test_e += s_eulers_scaled_tail[n * 4 + iii];
						// }
						// test_e /= projector.padding_factor;
						// // auto test_e = s_eulers_scaled_head[n * 4 + 0] / projector.padding_factor;
						// float test_t = 0;
						// for (int iii = 0; iii < 2; iii++) {
						// 	test_t += s_trans_xy[m * 2 + iii];
						// }

						// // auto test_t = s_trans_xy[m * 2 + 0];

						// g_diff2s[weight_offset] = test_e * test_t;
						// // g_diff2s[weight_offset] = 1.;
                    }
                }
            }
        }
	};

    // =====================================================================
	// ============================= main loop =============================
    // =====================================================================
    while (scheduler.has_work()) {
		__syncthreads();

        block_idx = scheduler.get_current_work_block_index();
        trans_block_idx = blocks[block_idx].startRow * TParams::kNrOverTrans;
        orient_block_idx = blocks[block_idx].startCol * TParams::kNrOverOrient;


		// read fragment_c from g_diff2s
        init_fragment_c();

		// load eulers to smem
		#pragma unroll
		for (int i = tid; i < TParams::kOrientBlockSize; i += TParams::kBlockSize) {
			if (orient_block_idx + i < orientation_num) {
				// TODO: check whether compiler uses load float4
				#pragma unroll
				for (int j = 0; j < 4; j ++) {
					s_eulers_scaled_head[i * 4 + j] = g_eulers[(orient_block_idx + i) * 9 + j] * projector.padding_factor;
					s_eulers_scaled_tail[i * 4 + j] = g_eulers[(orient_block_idx + i) * 9 + 4 + j] * projector.padding_factor;
				}
			} else {
				#pragma unroll
				for (int j = 0; j < 4; j ++) {
					s_eulers_scaled_head[i * 4 + j] = 0;
					s_eulers_scaled_tail[i * 4 + j] = 0;
				}
			}
		}

		// load trans to smem
		#pragma unroll
		for (int i = tid; i < TParams::kTransBlockSize; i += TParams::kBlockSize) {
			if (trans_block_idx + i < translation_num) {
				s_trans_xy[i * 2 + 0] = trans_xyz[(trans_block_idx + i) * 3 + 0];
				s_trans_xy[i * 2 + 1] = trans_xyz[(trans_block_idx + i) * 3 + 1];
			} else {
				s_trans_xy[i * 2 + 0] = 0.;
				s_trans_xy[i * 2 + 1] = 0.;
			}
		}

        // initialize shared memory to zero
        for (int i = tid; i < 2 * TParams::kTransBlockSize * TParams::kImgBlockSize; i += TParams::kBlockSize) {
            s_trans_mat_block[i] = 0.0;
        }
        // for (int i = tid; i < 2 * 2 * TParams::kOrientBlockSize * TParams::kImgBlockSize; i += TParams::kBlockSize) {
        for (int i = tid; i < 2 * TParams::kOrientBlockSize * TParams::kImgBlockSize; i += TParams::kBlockSize) {
            s_orient_mat_block[i] = 0.0;
        }

        s_trans_pow2_accumulator[tid] = 0.0;
        // s_orient_pow2_accumulator[tid] = 0.0;

		if (tid < TParams::kTransBlockSize) {
			s_trans_pow2_accumulator_bak[tid] = 0.;
		}
		if (tid < TParams::kOrientBlockSize) {
			s_orient_pow2_accumulator[tid] = 0.;
		}

        for (int i = tid; i < TParams::kImgBlockSize; i += TParams::kBlockSize) {
            s_corr_div_2[0][i] = 0.0;
            s_corr_div_2[1][i] = 0.0;
        }

        __syncthreads();

/*=============================== FOR IMAGE BLOCK ==============================*/
		int k_cycle;
        while (scheduler.get_current_work_next_k_cycle(k_cycle)) {
			__syncthreads();
			// __threadfence_block();
			if (k_cycle > scheduler.get_current_work_k_cycle_start()) {
				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle);
				int k_cycle_mod2 = scheduler.k_cycle_mod<2>(k_cycle);
				translation_matrix_handler.construct_translation_matrix(
					s_trans_real_mat_block_swizzle,
					s_trans_imag_mat_block_swizzle,
					s_trans_pow2_accumulator,
					s_trans_xy,
					s_img_real_imag[k_cycle_mod2],
					s_fcoor_xy[k_cycle_mod2],
					s_corr_div_2[k_cycle_mod2],
					img_block_idx,
					trans_block_idx,
					warp_id,
					lane_id
				);
			}

			if (k_cycle < scheduler.get_current_work_k_cycle_end() - 1) {
				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle + 1);
				int k_cycle_next_mod2 = scheduler.k_cycle_mod<2>(k_cycle + 1);
				load_coord_xy(img_block_idx, s_corr_div_2[k_cycle_next_mod2], 
							  s_coor_x[k_cycle_next_mod2], s_coor_y[k_cycle_next_mod2], 
							  s_fcoor_xy[k_cycle_next_mod2], s_img_real_imag[k_cycle_next_mod2]);
			}

			if (k_cycle > scheduler.get_current_work_k_cycle_start()) {
				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle);
				int k_cycle_mod2 = scheduler.k_cycle_mod<2>(k_cycle);
				orientation_matrix_handler.sync_and_store_orientation_matrix_with_reduce(
					s_orient_mat_block_swizzle[k_cycle_mod2],
					s_orient_real_mat_block_swizzle[k_cycle_mod2],
					s_orient_imag_mat_block_swizzle[k_cycle_mod2],
					s_orient_pow2_accumulator,
					s_corr_div_2[k_cycle_mod2],
					warp_id,
					lane_id
				);
			}

			__syncthreads();
			// __threadfence_block();
			if (k_cycle > scheduler.get_current_work_k_cycle_start() && 
				k_cycle < scheduler.get_current_work_k_cycle_end() - 1) {

				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle);
				int k_cycle_mod2 = scheduler.k_cycle_mod<2>(k_cycle);
				int k_cycle_next_mod2 = scheduler.k_cycle_mod<2>(k_cycle + 1);
				// fused
				orientation_matrix_handler.process_and_prefetch_orientation_matrix_fused_mma_tf32_sim_fp32(
					fragment_c, 
					s_trans_mat_block_swizzle, 
					s_orient_mat_block_swizzle[k_cycle_mod2],
					projector,
					s_eulers_scaled_head,
					s_eulers_scaled_tail,
					s_fcoor_xy[k_cycle_next_mod2],
					img_block_idx,
					orient_block_idx,
					warp_id,
					lane_id,
					s_test_buffer
				);
				
				// not fused
				// orientation_matrix_handler.process_and_prefetch_orientation_matrix(
				// 	projector,
				// 	s_eulers_scaled_head,
				// 	s_eulers_scaled_tail,
				// 	s_fcoor_xy[k_cycle_next_mod2],
				// 	img_block_idx,
				// 	orient_block_idx,
				// 	warp_id,
				// 	lane_id
				// );
				// // __syncthreads();
				// // __syncthreads();
				// /*=============================== COMPUTE CROSS TERM ==============================*/
				// block_mma_tf32_sim_fp32<TransMatLayout, OrientMatLayout, 
				// TParams::kTransBlockSize, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize,
				// TParams::kWarpTransTileSize, TParams::kWarpOrientTileSize, 2 * TParams::kWarpImgTileSize,
				// TParams::kMmaTransTileSize, TParams::kMmaOrientTileSize, TParams::kMmaImgTileSize>(
				// 	fragment_c, s_trans_mat_block_swizzle, s_orient_mat_block_swizzle[k_cycle_mod2], warp_id, lane_id);
			}

			if (k_cycle == scheduler.get_current_work_k_cycle_start()) {
				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle + 1);
				int k_cycle_next_mod2 = scheduler.k_cycle_mod<2>(k_cycle + 1);
				orientation_matrix_handler.process_and_prefetch_orientation_matrix(
					projector,
					s_eulers_scaled_head,
					s_eulers_scaled_tail,
					s_fcoor_xy[k_cycle_next_mod2],
					img_block_idx,
					orient_block_idx,
					warp_id,
					lane_id
				);
			}

			if (k_cycle == scheduler.get_current_work_k_cycle_end() - 1) {
				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle);
				int k_cycle_mod2 = scheduler.k_cycle_mod<2>(k_cycle);
				/*=============================== COMPUTE CROSS TERM ==============================*/
				block_mma_tf32_sim_fp32<TransMatLayout, OrientMatLayout, 
				TParams::kTransBlockSize, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize,
				TParams::kWarpTransTileSize, TParams::kWarpOrientTileSize, 2 * TParams::kWarpImgTileSize,
				TParams::kMmaTransTileSize, TParams::kMmaOrientTileSize, TParams::kMmaImgTileSize>(
					fragment_c, s_trans_mat_block_swizzle, s_orient_mat_block_swizzle[k_cycle_mod2], warp_id, lane_id);
			}

			// __syncthreads();

        } // end of image block
		__syncthreads();

        // reduce s_trans_pow2_accumulator
        // for (int i = 1; i < TParams::kBlockSize / TParams::kTransBlockSize; ++i) {
        //     if (tid < TParams::kTransBlockSize) {
        //         s_trans_pow2_accumulator[tid] += s_trans_pow2_accumulator[i * TParams::kTransBlockSize + tid];
        //     }
        // }
        // reduce s_orient_pow2_accumulator
        // for (int i = 1; i < TParams::kBlockSize / TParams::kOrientBlockSize; ++i) {
        //     if (tid < TParams::kOrientBlockSize) {
        //         s_orient_pow2_accumulator[tid] += s_orient_pow2_accumulator[i * TParams::kOrientBlockSize + tid];
        //     }
        // }

    /*=============================== REDUCE IN FRAGMENT_C ==============================*/
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
            #pragma unroll
            for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
                #pragma unroll
                for (int k = 0; k < kFragmentCSize; ++k) {
                    int m = fragment_c_m_idx_in_block<TParams>(warp_id, lane_id, i, k);
                    int n = fragment_c_n_idx_in_block<TParams>(warp_id, lane_id, j, k);
                    fragment_c[i][j][k] += s_trans_pow2_accumulator[m] + s_orient_pow2_accumulator[n];
                }
            }
        }
        __syncthreads();

    /*=============================== WRITE BACK ==============================*/
		epilogue();

        scheduler.advance_to_next_work();
    } // end of while has_work
}

#endif // FINE_MATRIX_KERNEL_IM2COL_SPLITIMG_BCF_PMO_CUH