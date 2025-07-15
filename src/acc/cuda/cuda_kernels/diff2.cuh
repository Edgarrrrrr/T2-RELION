#ifndef CUDA_DIFF2_KERNELS_CUH_
#define CUDA_DIFF2_KERNELS_CUH_

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <fstream>
#include "src/acc/acc_projector.h"
#include "src/acc/acc_projectorkernel_impl.h"
#include "src/acc/cuda/cuda_settings.h"
#include "src/acc/cuda/cuda_kernels/cuda_device_utils.cuh"
#include "./mma_utils.cuh"
#include "./coarse_scheduler.cuh"
#include "./reg_bitmap.cuh"
#include "./warp_layout.cuh"
#include "./orientation_matrix_handler.cuh"
#include "./translation_matrix_handler.cuh"

/*
 *   	DIFFERNECE-BASED KERNELS
 */

/*
 * Assuming block_sz % prefetch_fraction == 0 and prefetch_fraction < block_sz
 * Assuming block_sz % eulers_per_block == 0
 * Assuming eulers_per_block * 3 < block_sz
 */

// template<int kBLOCK_SZ>
// __global__ void check_array_equal(XFLOAT *a, XFLOAT *b, int size) {
// 	int tid = threadIdx.x;
// 	assert(blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0);
// 	assert(blockIdx.x == 0);

// 	bool equal = true;

// 	for (int i = tid; i < size; i += kBLOCK_SZ) {
// 		// if (a[i] != b[i]) {
// 		if (a[i] == 0 && b[i] == 0) continue;
// 		if (fabs((double)a[i] - (double)b[i]) >= 9e-6 * fabs((double)b[i])) {
// 			// if (tid == 0)
// 				printf("!!!ERR!!! a[%d] = %f, b[%d] = %f, epsilon = %e\n", i, a[i], i, b[i], fabs((double)a[i] - (double)b[i]) / fabs((double)b[i]));
// 			// assert(false);
// 			equal = false;
// 		}
// 	}

// 	__syncthreads();
// 	// assert(equal);
// }

// template<int kBLOCK_SZ>
// __global__ void cuda_kernel_coarse_setup_trans_matrix(
// 		XFLOAT *trans_x,
// 		XFLOAT *trans_y,
// 		XFLOAT *trans_z,
// 		XFLOAT *g_real,
// 		XFLOAT *g_imag,
// 		AccProjectorKernel projector,
// 		XFLOAT *g_trans_real_m,       // translation_num * image_size matrix, column-major
// 		XFLOAT *g_trans_imag_m,  
// 		XFLOAT *g_corr,     
// 		int translation_num,
// 		int image_size) {
// 	int tid = threadIdx.x;     // tid is the index of pixel
// 	int block_id = blockIdx.x; // block_id is the index of translation
// 	int trans_idx = block_id;

// 	assert(blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0);

// 	for (int i = tid; i < image_size; i += kBLOCK_SZ) {
// 		int x = i % projector.imgX;
// 		int y = floorfracf(i, projector.imgX);
// 		if (y > projector.maxR) {
// 			y -= projector.imgY;
// 		}
// 		XFLOAT real = g_real[i] * sqrt(g_corr[i]) / sqrt(2.);
// 		XFLOAT imag = g_imag[i] * sqrt(g_corr[i]) / sqrt(2.);

// 		XFLOAT trans_real, trans_imag;

// 		XFLOAT tx = trans_x[trans_idx];
// 		XFLOAT ty = trans_y[trans_idx];
// 		translatePixel(x, y, tx, ty, real, imag, trans_real, trans_imag);
// 		int matrix_index = trans_idx + i * translation_num;
// 		// g_trans_real_m[matrix_index] = trans_real;
// 		// g_trans_imag_m[matrix_index] = trans_imag;

// 		g_trans_real_m[matrix_index] = trans_real;
// 		g_trans_imag_m[matrix_index] = trans_imag;

// 		// g_trans_real_m[matrix_index] = i / (float)(trans_idx + 1);
// 		// g_trans_imag_m[matrix_index] = i / (float)(trans_idx + 1);

// 		// g_trans_real_m[matrix_index] = (float)(1);
// 		// g_trans_imag_m[matrix_index] = (float)(1);

// 	}
// }

// template<int kBLOCK_SZ>
// __global__ void cuda_kernel_coarse_setup_orient_matrix(
// 	XFLOAT *g_eulers,
// 	AccProjectorKernel projector,
// 	XFLOAT *g_orient_real_m,       // image_size * orientation_num matrix, column-major
// 	XFLOAT *g_orient_imag_m,
// 	XFLOAT *g_corr,
// 	int orientation_num,
// 	int image_size) {
	
// 	int tid = threadIdx.x;     // tid is the index of pixel
// 	int block_id = blockIdx.x; // block_id is the index of orientation

// 	assert(blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0);
	
// 	XFLOAT e0 = g_eulers[block_id * 9];
// 	XFLOAT e1 = g_eulers[block_id * 9 + 1];
// 	XFLOAT e2 = g_eulers[block_id * 9 + 2];
// 	XFLOAT e3 = g_eulers[block_id * 9 + 3];
// 	XFLOAT e4 = g_eulers[block_id * 9 + 4];
// 	XFLOAT e5 = g_eulers[block_id * 9 + 5];
// 	XFLOAT e6 = g_eulers[block_id * 9 + 6];
// 	XFLOAT e7 = g_eulers[block_id * 9 + 7];
// 	XFLOAT e8 = g_eulers[block_id * 9 + 8];
	
// 	for (int i = tid; i < image_size; i += kBLOCK_SZ) {
// 		int x = i % projector.imgX;
// 		// int y = floorfracf(i, projector.imgX);
// 		int y = i / projector.imgX;
// 		if (y > projector.maxR) {
// 			y -= projector.imgY;
// 		}

// 		XFLOAT orient_real, orient_imag;

// 		projector.project3Dmodel(x, y, e0, e1, e3, e4, e6, e7, orient_real, orient_imag);

// 		int matrix_index = i + block_id * image_size;
// 		g_orient_real_m[matrix_index] = orient_real * sqrt(g_corr[i]) / sqrt(2.);
// 		g_orient_imag_m[matrix_index] = orient_imag * sqrt(g_corr[i]) / sqrt(2.);

// 		// g_orient_real_m[matrix_index] = 1.;
// 		// g_orient_imag_m[matrix_index] = 1.;
		
// 	}
// }		

// template<int kBLOCK_SZ>
// __global__ void cuda_kernel_coarse_naive(
// 		XFLOAT *g_trans_real_m,
// 		XFLOAT *g_trans_imag_m,
// 		XFLOAT *g_orient_real_m,
// 		XFLOAT *g_orient_imag_m,
// 		XFLOAT *g_corr,
// 		XFLOAT *g_diff2s,
// 		int translation_num,
// 		int orientation_num,
// 		int image_size) {
	
// 	int tid = threadIdx.x;
// 	int block_id = blockIdx.x;

// 	int trans_idx = block_id % translation_num;
// 	int orient_idx = block_id / translation_num;

// 	assert(blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0);

// 	XFLOAT diff2 = 0.0;
// 	// for (int i = tid; i < image_size; i += kBLOCK_SZ) {
// 	// 	XFLOAT trans_real = g_trans_real_m[i * translation_num + trans_idx];
// 	// 	XFLOAT trans_imag = g_trans_imag_m[i * translation_num + trans_idx];

// 	// 	XFLOAT orient_real = g_orient_real_m[i + orient_idx * image_size];
// 	// 	XFLOAT orient_imag = g_orient_imag_m[i + orient_idx * image_size];
// 	// 	XFLOAT diff_real = trans_real - orient_real;
// 	// 	XFLOAT diff_imag = trans_imag - orient_imag;

// 	// 	diff2 += (diff_real * diff_real + diff_imag * diff_imag) * g_corr[i] / 2.;
// 	// }

// 	XFLOAT diff2_positive = 0.0;
// 	XFLOAT diff2_negative = 0.0;
// 	for (int i = tid; i < image_size; i += kBLOCK_SZ) {
// 		XFLOAT trans_real = g_trans_real_m[i * translation_num + trans_idx];
// 		XFLOAT trans_imag = g_trans_imag_m[i * translation_num + trans_idx];

// 		XFLOAT orient_real = g_orient_real_m[i + orient_idx * image_size];
// 		XFLOAT orient_imag = g_orient_imag_m[i + orient_idx * image_size];

// 		// diff2_positive += (trans_real * trans_real + orient_real * orient_real + trans_imag * trans_imag + orient_imag * orient_imag) * g_corr[i] / 2.;
// 		// diff2_negative += (2 * trans_real * orient_real + 2 * trans_imag * orient_imag) * g_corr[i] / 2.;

// 		diff2_positive += (trans_real * trans_real + orient_real * orient_real + trans_imag * trans_imag + orient_imag * orient_imag);
// 		// diff2_positive += (1 + 4 + 1 + 4);

// 		diff2_negative += (2 * trans_real * orient_real + 2 * trans_imag * orient_imag);
// 		// diff2_negative += (trans_real * orient_real + trans_imag * orient_imag);
// 		// diff2_negative += (1 * 2 + 1 * 2);
// 	}
// 	diff2 = diff2_positive - diff2_negative;
// 	// diff2 = diff2_positive; // debug check positive component

// 	// reduce diff2 in a block
// 	__shared__ XFLOAT s_diff2[kBLOCK_SZ];
// 	s_diff2[tid] = diff2;
// 	__syncthreads();
// 	for (int j = kBLOCK_SZ / 2; j > 0; j /= 2) {
// 		if (tid < j) {
// 			s_diff2[tid] += s_diff2[tid + j];
// 		}
// 		__syncthreads();
// 	}
// 	if (tid == 0) {
// 		// g_diff2s[block_id] += s_diff2[0];
// 		g_diff2s[block_id] += s_diff2[0];
// 	}
// }

// __device__ inline void pixel_index2coor(int idx, int imgX, int imgY, int maxR, int &x, int &y) {
// 	x = idx % imgX;
// 	y = idx / imgX;
// 	if (y > maxR) {
// 		y -= imgY;
// 	}
// }

// __device__ __forceinline__ XFLOAT warpReduceSum(XFLOAT sum){
//     sum += __shfl_down_sync(0xffffffff, sum, 16);
//     sum += __shfl_down_sync(0xffffffff, sum, 8);
//     sum += __shfl_down_sync(0xffffffff, sum, 4);
//     sum += __shfl_down_sync(0xffffffff, sum, 2);
//     sum += __shfl_down_sync(0xffffffff, sum, 1);
// 	sum = __shfl_sync(0xffffffff, sum, 0);
//     return sum;
// }

// // __inline__ __device__ XFLOAT warpReduceSum(XFLOAT val) {
// //     // Reduce within a warp
// //     for (int offset = warpSize / 2; offset > 0; offset /= 2) {
// //         val += __shfl_down_sync(0xffffffff, val, offset);
// //     }
// //     return val;
// // };

// template<int kWarpTileM, int kWarpTileN>
// __inline__ __device__ int get_warp_tile_offset(int warp_id, int warp_num, int tile_size) {
// 	return warp_id * tile_size;
// }

// // tiling in the translation dimension and the orientation dimension
// // tile size is kTransBlockSize * kOrientBlockSize * image_size
// template<int kBlockSize, int kTransBlockSize, int kOrientBlockSize>
// __global__ void cuda_kernel_coarse_matrixV1(
// 	XFLOAT *g_eulers,
// 	XFLOAT *trans_x,
// 	XFLOAT *trans_y,
// 	XFLOAT *g_real,
// 	XFLOAT *g_imag,
// 	AccProjectorKernel projector,
// 	XFLOAT *g_corr,
// 	XFLOAT *g_diff2s,
// 	XFLOAT *g_diff2s_opt,
// 	int translation_num,
// 	int orientation_num,
// 	int image_size,
// 	XFLOAT *g_trans_real_m,
// 	XFLOAT *g_trans_imag_m,
// 	XFLOAT *g_orient_real_m,
// 	XFLOAT *g_orient_imag_m) {

// 	constexpr int kImgBlockSize = 128;

// 	constexpr int kWarpTransTileSize = 16; // M
// 	constexpr int kWarpOrientTileSize = 8; // N
// 	constexpr int kWarpImgTileSize = 8;    // K

// 	constexpr int kMmaTransBlockSize = 16;  // M
// 	constexpr int kMmaOrientBlockSize = 8; // N
// 	constexpr int kMmaImgBlockSize = 4;    // K
	
// 	const int tid = threadIdx.x;          // thread id in a block
// 	const int bid = blockIdx.x;           // block id in a grid
// 	const int warp_id  = tid / 32;        // warp id in a block
// 	const int warp_num = kBlockSize / 32; // number of warps in a block
// 	const int lane_id  = tid % 32;        // thread id in a warp
	
// 	const int trans_block_num = (translation_num + kTransBlockSize - 1) / kTransBlockSize;
// 	const int orient_block_num = (orientation_num + kOrientBlockSize - 1) / kOrientBlockSize;

// 	const int trans_block_idx = bid % trans_block_num * kTransBlockSize;
// 	const int orient_block_idx = bid / trans_block_num * kOrientBlockSize;

// 	assert(blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0);
	
// 	// 'img' data is stored contiguously.
// 	__shared__ XFLOAT s_trans_real_mat_block[kTransBlockSize * kImgBlockSize];
// 	__shared__ XFLOAT s_trans_imag_mat_block[kTransBlockSize * kImgBlockSize];
// 	__shared__ XFLOAT s_orient_real_mat_block[kOrientBlockSize * kImgBlockSize];
// 	__shared__ XFLOAT s_orient_imag_mat_block[kOrientBlockSize * kImgBlockSize];
	

// 	// 'translation' data is stored contiguously.
// 	__shared__ XFLOAT s_diff_mat_block[kOrientBlockSize * kTransBlockSize];

// 	__shared__ XFLOAT s_corr_div_sqrt2[kImgBlockSize];
// 	__shared__ XFLOAT s_coor_x[kImgBlockSize];
// 	__shared__ XFLOAT s_coor_y[kImgBlockSize];

// 	// initialize shared memory to zero
// 	for (int i = tid; i < kTransBlockSize * kImgBlockSize; i += kBlockSize) {
// 		s_trans_real_mat_block[i] = 0.0;
// 		s_trans_imag_mat_block[i] = 0.0;
// 	}
// 	for (int i = tid; i < kOrientBlockSize * kImgBlockSize; i += kBlockSize) {
// 		s_orient_real_mat_block[i] = 0.0;
// 		s_orient_imag_mat_block[i] = 0.0;
// 	}
// 	for (int i = tid; i < kTransBlockSize * kOrientBlockSize; i += kBlockSize) {
// 		s_diff_mat_block[i] = 0.0;
// 	}

	
// 	// // construct diff_mat
// 	// for (int i = tid; i < kTransBlockSize * kOrientBlockSize; i += kBlockSize) {
// 	// 	int s_trans_idx = i % kTransBlockSize;
// 	// 	int s_orient_idx = i / kTransBlockSize;
// 	// 	int g_trans_idx = s_trans_idx + trans_block_idx;
// 	// 	int g_orient_idx = s_orient_idx + orient_block_idx;
// 	// 	if (g_trans_idx >= translation_num || g_orient_idx >= orientation_num) {
// 	// 		continue;
// 	// 	}
// 	// 	s_diff_mat_block[i] = g_diff2s[g_orient_idx * translation_num + g_trans_idx];
// 	// }
// 	__syncthreads();

// /*=============================== FOR IMAGE BLOCK ==============================*/
// 	for (int img_block_idx = 0; img_block_idx < image_size; img_block_idx += kImgBlockSize) {
// 		// TODO: offload this compute to CPU
// 		#pragma unroll 
// 		for (int i = tid; i < kImgBlockSize; i += kBlockSize) {
// 			if (img_block_idx + i < image_size) {
// 				s_corr_div_sqrt2[i] = (XFLOAT)sqrt((double) (g_corr[img_block_idx + i] / 2));
// 			} else {
// 				s_corr_div_sqrt2[i] = 0.;
// 			}
// 		}
// 		for (int i = tid; i < kImgBlockSize; i += kBlockSize) {
// 			int x, y;
// 			pixel_index2coor(img_block_idx + i, projector.imgX, projector.imgY, projector.maxR, x, y);
// 			s_coor_x[i] = x;
// 			s_coor_y[i] = y;
// 		}
// 		__syncthreads();

// /*=============================== CONSTRUCT TRANS_MAT & ORIENT_MAT ==============================*/
// 		// construct trans_mat
// 		// one warp computes a translation row
// 		for (int i = warp_id; i < kTransBlockSize; i += warp_num) {
// 			XFLOAT real = 0.;
// 			XFLOAT imag = 0.;
// 			int g_trans_idx = trans_block_idx + i;
// 			if (g_trans_idx >= translation_num) continue;

// 			XFLOAT tx = trans_x[g_trans_idx];
// 			XFLOAT ty = trans_y[g_trans_idx];

// 			#pragma unroll
// 			for (int j = lane_id; j < kImgBlockSize; j += warpSize) {
// 				int g_img_idx = img_block_idx + j;
// 				if (g_img_idx >= image_size) {
// 					s_trans_real_mat_block[i * kImgBlockSize + j] = 0.;
// 					s_trans_imag_mat_block[i * kImgBlockSize + j] = 0.;
// 					continue;
// 				}
// 				real = g_real[g_img_idx] * s_corr_div_sqrt2[j];// * sqrt(g_corr[g_img_idx]) / sqrt(2.);
// 				imag = g_imag[g_img_idx] * s_corr_div_sqrt2[j];// * sqrt(g_corr[g_img_idx]) / sqrt(2.);
// 				int x, y;
// 				// pixel_index2coor(g_img_idx, projector.imgX, projector.imgY, projector.maxR, x, y);
// 				x = s_coor_x[j];
// 				y = s_coor_y[j];
// 				XFLOAT trans_real, trans_imag;
// 				translatePixel(x, y, tx, ty, real, imag, trans_real, trans_imag);

// 				s_trans_real_mat_block[i * kImgBlockSize + j] = trans_real;
// 				s_trans_imag_mat_block[i * kImgBlockSize + j] = trans_imag;
// 			}
// 		}

// 		// construct orient_mat
// 		for (int i = warp_id; i < kOrientBlockSize; i += warp_num) {
// 			int g_orient_idx = orient_block_idx + i;
// 			if (g_orient_idx >= orientation_num) continue;
// 			// first load eulers
// 			XFLOAT e0 = g_eulers[g_orient_idx * 9];
// 			XFLOAT e1 = g_eulers[g_orient_idx * 9 + 1];
// 			XFLOAT e3 = g_eulers[g_orient_idx * 9 + 3];
// 			XFLOAT e4 = g_eulers[g_orient_idx * 9 + 4];
// 			XFLOAT e6 = g_eulers[g_orient_idx * 9 + 6];
// 			XFLOAT e7 = g_eulers[g_orient_idx * 9 + 7];

// 			#pragma unroll
// 			for (int j = lane_id; j < kImgBlockSize; j += warpSize) {
// 				int g_img_idx = img_block_idx + j;
// 				if (g_img_idx >= image_size) {
// 					s_orient_real_mat_block[i * kImgBlockSize + j] = 0.;
// 					s_orient_imag_mat_block[i * kImgBlockSize + j] = 0.;
// 					continue;
// 				}
// 				int x, y;
// 				// pixel_index2coor(g_img_idx, projector.imgX, projector.imgY, projector.maxR, x, y);
// 				x = s_coor_x[j];
// 				y = s_coor_y[j];
// 				XFLOAT orient_real, orient_imag;
// 				projector.project3Dmodel(x, y, e0, e1, e3, e4, e6, e7, orient_real, orient_imag);
// 				s_orient_real_mat_block[i * kImgBlockSize + j] = orient_real * s_corr_div_sqrt2[j];// * sqrt(g_corr[g_img_idx]) / sqrt(2.);
// 				s_orient_imag_mat_block[i * kImgBlockSize + j] = orient_imag * s_corr_div_sqrt2[j];// * sqrt(g_corr[g_img_idx]) / sqrt(2.);
// 			}
// 		}


// // /*=============================== DEBUG ==============================*/
// // 		// check s_trans_mat and g_trans_mat

// // 		for (int i = tid; i < kImgBlockSize; i += kBlockSize) {
// // 			for (int j = 0; j < kTransBlockSize; j++) {
// // 				int g_trans_idx = trans_block_idx + j;
// // 				int g_img_idx = img_block_idx + i;
// // 				if (g_trans_idx >= translation_num || g_img_idx >= image_size) {
// // 					continue;
// // 				}
// // 				int g_trans_mat_idx = g_img_idx * translation_num + g_trans_idx;
// // 				int s_trans_mat_idx = j * kImgBlockSize + i;
// // 				if (g_trans_real_m[g_trans_mat_idx] != s_trans_real_mat_block[s_trans_mat_idx]) {
// // 					printf("!!! trans : g = %f, s = %f\n", g_trans_real_m[g_trans_mat_idx], s_trans_real_mat_block[s_trans_mat_idx]);
// // 					assert(false);
// // 				}
// // 				if (g_trans_imag_m[g_trans_mat_idx] != s_trans_imag_mat_block[s_trans_mat_idx]) {
// // 					printf("!!! trans : g = %f, s = %f\n", g_trans_imag_m[g_trans_mat_idx], s_trans_imag_mat_block[s_trans_mat_idx]);
// // 					assert(false);
// // 				}
// // 			}
// // 		}

// // 		// check s_orient_mat and g_orient_mat
// // 		for (int i = tid; i < kImgBlockSize; i += kBlockSize) {
// // 			for (int j = 0; j < kOrientBlockSize; j++) {
// // 				int g_orient_idx = orient_block_idx + j;
// // 				int g_img_idx = img_block_idx + i;
// // 				if (g_orient_idx >= orientation_num || g_img_idx >= image_size) {
// // 					continue;
// // 				}
// // 				int g_orient_mat_idx = g_img_idx + g_orient_idx * image_size;
// // 				int s_orient_mat_idx = j * kImgBlockSize + i;
// // 				if (g_orient_real_m[g_orient_mat_idx] != s_orient_real_mat_block[s_orient_mat_idx]) {
// // 					printf("!!! orient : g = %f, s = %f\n", g_orient_real_m[g_orient_mat_idx], s_orient_real_mat_block[s_orient_mat_idx]);
// // 					assert(false);
// // 				}
// // 				if (g_orient_imag_m[g_orient_mat_idx] != s_orient_imag_mat_block[s_orient_mat_idx]) {
// // 					printf("!!! orient : g = %f, s = %f\n", g_orient_imag_m[g_orient_mat_idx], s_orient_imag_mat_block[s_orient_mat_idx]);
// // 					assert(false);
// // 				}
// // 			}
// // 		}
// // /*=============================== END DEBUG ==============================*/

// /*=============================== COMPUTE QUADRATIC COMPONENT ==============================*/
// 		// trans_mat, each warp computes a translation row
// 		// assert(kTransBlockSize % 32 == 0);
// 		// assert(kOrientBlockSize % 32 == 0);
// 		// if (img_block_idx + kImgBlockSize >= image_size) {
// 		// 	// printf("img_block_idx = %d, image_size = %d\n", img_block_idx, image_size);
// 		// 	// print mat
// 		// 	if (tid == 0 && bid == 0) {
// 		// 		printf("trans real\n");
// 		// 		for (int i = 0; i < kTransBlockSize; i++) {
// 		// 			for (int j = 0; j < kImgBlockSize; j++) {
// 		// 				printf("%4.2f ", s_trans_real_mat_block[i * kImgBlockSize + j]);
// 		// 			}
// 		// 			printf("\n");
// 		// 		}
// 		// 		printf("trans imag\n");
// 		// 		for (int i = 0; i < kTransBlockSize; i++) {
// 		// 			for (int j = 0; j < kImgBlockSize; j++) {
// 		// 				printf("%4.2f ", s_trans_imag_mat_block[i * kImgBlockSize + j]);
// 		// 			}
// 		// 			printf("\n");
// 		// 		}
// 		// 		printf("orient real\n");
// 		// 		for (int i = 0; i < kOrientBlockSize; i++) {
// 		// 			for (int j = 0; j < kImgBlockSize; j++) {
// 		// 				printf("%4.2f ", s_orient_real_mat_block[i * kImgBlockSize + j]);
// 		// 			}
// 		// 			printf("\n");
// 		// 		}
// 		// 		printf("orient imag\n");
// 		// 		for (int i = 0; i < kOrientBlockSize; i++) {
// 		// 			for (int j = 0; j < kImgBlockSize; j++) {
// 		// 				printf("%4.2f ", s_orient_imag_mat_block[i * kImgBlockSize + j]);
// 		// 			}
// 		// 			printf("\n");
// 		// 		}

// 		// 	}
// 		// }
// 		// // fflush(stdout);
// 		#pragma unroll
// 		for (int i = warp_id; i < kTransBlockSize; i += warp_num) {
// 			XFLOAT magnitude_squared_sum = 0.;
// 			// if (i >= kTransBlockSize) continue;
// 			#pragma unroll
// 			for (int j = lane_id; j < kImgBlockSize; j += 32) {
// 				// if (j >= kImgBlockSize) continue;
// 				XFLOAT real_tmp = s_trans_real_mat_block[i * kImgBlockSize + j];
// 				XFLOAT imag_tmp = s_trans_imag_mat_block[i * kImgBlockSize + j];

// 				magnitude_squared_sum += real_tmp * real_tmp + imag_tmp * imag_tmp;
// 			}
// 			// reduce real and imag in a warp
// 			magnitude_squared_sum = warpReduceSum(magnitude_squared_sum);

// 			// write back to diff_mat, each warp writes a row
// 			#pragma unroll
// 			for (int j = lane_id; j < kOrientBlockSize; j += warpSize) {
// 				// if (j >= kOrientBlockSize) continue;
// 				s_diff_mat_block[j * kTransBlockSize + i] += magnitude_squared_sum;
// 			}
// 		}
// 		__syncthreads();

// 		// orient_mat, each warp computes an orientation column
// 		#pragma unroll
// 		for (int i = warp_id; i < kOrientBlockSize; i += warp_num) {
// 			XFLOAT magnitude_squared_sum = 0.;
// 			// if (i >= kOrientBlockSize) continue;
			
// 			#pragma unroll
// 			for (int j = lane_id; j < kImgBlockSize; j += warpSize) {
// 				// if (j >= kImgBlockSize) continue;
// 				XFLOAT real_tmp = s_orient_real_mat_block[i * kImgBlockSize + j];
// 				XFLOAT imag_tmp = s_orient_imag_mat_block[i * kImgBlockSize + j];

// 				magnitude_squared_sum += real_tmp * real_tmp + imag_tmp * imag_tmp;
// 			}
// 			// reduce real and imag in a warp
// 			magnitude_squared_sum = warpReduceSum(magnitude_squared_sum);

// 			// write back to diff_mat, each warp writes a column
// 			#pragma unroll
// 			for (int j = lane_id; j < kTransBlockSize; j += warpSize) {
// 				// if (j >= kTransBlockSize) continue;
// 				s_diff_mat_block[i * kTransBlockSize + j] += magnitude_squared_sum;
// 			}
// 		}


// 		// for (int img_tile_idx = warp_id * kWarpImgTileSize; 
// 		// 		 img_tile_idx < kImgBlockSize; 
// 		// 		 img_tile_idx += warp_num * kWarpImgTileSize) {
			
// 		// 	for (int img_mma_idx = 0; 
// 		// 		     img_mma_idx < kWarpImgTileSize; 
// 		// 			 img_mma_idx += kMmaImgBlockSize) {
// 		// 		int img_idx = img_block_idx + img_tile_idx + img_mma_idx;

// 		// 		int trans_idx = trans_block_idx


// 		// }

// /*=============================== COMPUTE CROSS TERM ==============================*/



// 	}
		

// /*=============================== WRITE BACK ==============================*/
// 	// write back diff_mat
// 	for (int i = tid; i < kTransBlockSize * kOrientBlockSize; i += kBlockSize) {
// 		int s_trans_idx = i % kTransBlockSize;
// 		int s_orient_idx = i / kTransBlockSize;
// 		int g_trans_idx = s_trans_idx + trans_block_idx;
// 		int g_orient_idx = s_orient_idx + orient_block_idx;
// 		if (g_trans_idx >= translation_num || g_orient_idx >= orientation_num) {
// 			continue;
// 		}
// 		g_diff2s_opt[g_orient_idx * translation_num + g_trans_idx] = s_diff_mat_block[i];
// 	}

// }

// // test function

// template<typename TParams>
// __global__ void cuda_kernel_test_C_idx_func() {
// 	assert(TParams::kBlockSize == blockDim.x);
// 	const int tid = threadIdx.x;          // thread id in a block
// 	const int bid = blockIdx.x;           // block id in a grid
// 	const int warp_id  = tid / 32;        // warp id in a block
// 	const int warp_num = TParams::kBlockSize / 32; // number of warps in a block
// 	const int lane_id  = tid % 32;        // thread id in a warp

// 	// __shared__ float s_A[kTransBlockSize * kImgBlockSize];
// 	// __shared__ float s_B[kOrientBlockSize * kImgBlockSize];
// 	__shared__ short s_C[TParams::kTransBlockSize * TParams::kOrientBlockSize];

// 	// initialize s_A, s_B, s_C with 0
// 	// for (int i = tid; i < kTransBlockSize * kImgBlockSize; i += blockDim.x) {
// 	// 	s_A[i] = 0;
// 	// }
// 	// for (int i = tid; i < kOrientBlockSize * kImgBlockSize; i += blockDim.x) {
// 	// 	s_B[i] = 0;
// 	// }
// 	for (int i = tid; i < TParams::kTransBlockSize * TParams::kOrientBlockSize; i += blockDim.x) {
// 		s_C[i] = -1;
// 	}
// 	__syncthreads();


// 	// test fragment_c_m_idx_in_block
// 	constexpr int kNumMmaTransInWarpTile = TParams::kWarpTransTileSize / TParams::kMmaTransTileSize;
// 	constexpr int kNumMmaOrientInWarpTile = TParams::kWarpOrientTileSize / TParams::kMmaOrientTileSize;
// 	constexpr int kNumMmaImgInWarpTile = TParams::kWarpImgTileSize / TParams::kMmaImgTileSize;

// 	constexpr int kFragmentASize = TParams::kMmaTransTileSize * TParams::kMmaImgTileSize / kWarpSize;
// 	constexpr int kFragmentBSize = TParams::kMmaOrientTileSize * TParams::kMmaImgTileSize / kWarpSize;
// 	constexpr int kFragmentCSize = TParams::kMmaTransTileSize * TParams::kMmaOrientTileSize / kWarpSize;

// 	XFLOAT fragment_a[kNumMmaTransInWarpTile][kFragmentASize];
// 	XFLOAT fragment_b[kNumMmaOrientInWarpTile][kFragmentBSize];
// 	XFLOAT fragment_c[kNumMmaTransInWarpTile][kNumMmaOrientInWarpTile][kFragmentCSize];

// 	// all the fragment_c in one block must cover all the elements in matrix C

// 	#pragma unroll
// 	for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 		#pragma unroll
// 		for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 			#pragma unroll
// 			for (int k = 0; k < kFragmentCSize; ++k) {
// 				fragment_c[i][j][k] = lane_id;
// 			}
// 		}
// 	}

// 	__syncthreads();
	
// 	#pragma unroll
// 	for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 		#pragma unroll
// 		for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 			#pragma unroll
// 			for (int k = 0; k < kFragmentCSize; ++k) {
// 				int m = fragment_c_m_idx_in_block<TParams>(warp_id, lane_id, i, k);
// 				int n = fragment_c_n_idx_in_block<TParams>(warp_id, lane_id, j, k);
// 				assert(m >= 0 && m < TParams::kTransBlockSize);
// 				assert(n >= 0 && n < TParams::kOrientBlockSize);
// 				s_C[m * TParams::kOrientBlockSize + n] = fragment_c[i][j][k];
// 			}
// 		}
// 	}

// 	__syncthreads();

// 	// // tid 0 print the matrix C
// 	// if (tid == 0) {
// 	// 	for (int i = 0; i < TParams::kTransBlockSize; ++i) {
// 	// 		for (int j = 0; j < TParams::kOrientBlockSize / 2; ++j) {
// 	// 			printf("%2d ", s_C[i * TParams::kOrientBlockSize + j]);
// 	// 		}
			
// 	// 		printf("\n");
// 	// 	}

// 	// 	printf("\n\n=============================================================================\n\n");
// 	// 	for (int i = 0; i < TParams::kTransBlockSize; ++i) {
// 	// 		for (int j = TParams::kOrientBlockSize / 2; j < TParams::kOrientBlockSize; ++j) {
// 	// 			printf("%2d ", s_C[i * TParams::kOrientBlockSize + j]);
// 	// 		}
			
// 	// 		printf("\n");
// 	// 	}
// 	// }
// 	// __syncthreads();

// 	// test there is no zero element in matrix C
// 	for (int i = tid; i < TParams::kTransBlockSize * TParams::kOrientBlockSize; i += blockDim.x) {
// 		assert(s_C[i] != -1);
// 	}
// 	__syncthreads();


// 	// test fragment layout in MMA operator

// 	for (int i = tid; i < TParams::kTransBlockSize * TParams::kOrientBlockSize; i += blockDim.x) {
// 		s_C[i] = -1;
// 	}
// 	__syncthreads();

// 	#pragma unroll
// 	for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 		#pragma unroll
// 		for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 			#pragma unroll
// 			for (int k = 0; k < kFragmentCSize; ++k) {
// 				fragment_c[i][j][k] = k;
// 			}
// 		}
// 	}

// 	__syncthreads();

// 	#pragma unroll
// 	for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 		#pragma unroll
// 		for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 			#pragma unroll
// 			for (int k = 0; k < kFragmentCSize; ++k) {
// 				int m = fragment_c_m_idx_in_block<TParams>(warp_id, lane_id, i, k);
// 				int n = fragment_c_n_idx_in_block<TParams>(warp_id, lane_id, j, k);
// 				assert(m >= 0 && m < TParams::kTransBlockSize);
// 				assert(n >= 0 && n < TParams::kOrientBlockSize);
// 				s_C[m * TParams::kOrientBlockSize + n] = fragment_c[i][j][k];
// 			}
// 		}
// 	}

// 	__syncthreads();

// 	// // tid 0 print the matrix C
// 	// if (tid == 0) {
// 	// 	for (int i = 0; i < TParams::kTransBlockSize; ++i) {
// 	// 		for (int j = 0; j < TParams::kOrientBlockSize / 2; ++j) {
// 	// 			printf("%2d ", s_C[i * TParams::kOrientBlockSize + j]);
// 	// 		}
			
// 	// 		printf("\n");
// 	// 	}

// 	// 	printf("\n\n=============================================================================\n\n");
// 	// 	for (int i = 0; i < TParams::kTransBlockSize; ++i) {
// 	// 		for (int j = TParams::kOrientBlockSize / 2; j < TParams::kOrientBlockSize; ++j) {
// 	// 			printf("%2d ", s_C[i * TParams::kOrientBlockSize + j]);
// 	// 		}
			
// 	// 		printf("\n");
// 	// 	}
// 	// }
// 	// __syncthreads();

// 	// test there is no zero element in matrix C
// 	for (int i = tid; i < TParams::kTransBlockSize * TParams::kOrientBlockSize; i += blockDim.x) {
// 		assert(s_C[i] != -1);
// 	}
// 	__syncthreads();
// }


// template<typename TParams>
// __global__ void cuda_kernel_test_AB_idx_func() {
// 	assert(TParams::kBlockSize == blockDim.x);
// 	const int tid = threadIdx.x;          // thread id in a block
// 	const int bid = blockIdx.x;           // block id in a grid
// 	const int warp_id  = tid / 32;        // warp id in a block
// 	const int warp_num = TParams::kBlockSize / 32; // number of warps in a block
// 	const int lane_id  = tid % 32;        // thread id in a warp

// 	__shared__ XFLOAT s_A[TParams::kTransBlockSize * TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_B[TParams::kOrientBlockSize * TParams::kImgBlockSize];
// 	// __shared__ short s_C[TParams::kTransBlockSize * TParams::kOrientBlockSize];

// 	// initialize s_A, s_B, s_C with 0
// 	for (int i = tid; i < TParams::kTransBlockSize * TParams::kImgBlockSize; i += blockDim.x) {
// 		s_A[i] = -1;
// 	}
// 	for (int i = tid; i < TParams::kOrientBlockSize * TParams::kImgBlockSize; i += blockDim.x) {
// 		s_B[i] = -1;
// 	}
// 	__syncthreads();


// 	// test fragment_c_m_idx_in_block
// 	constexpr int kNumMmaTransInWarpTile = TParams::kWarpTransTileSize / TParams::kMmaTransTileSize;
// 	constexpr int kNumMmaOrientInWarpTile = TParams::kWarpOrientTileSize / TParams::kMmaOrientTileSize;
// 	constexpr int kNumMmaImgInWarpTile = TParams::kWarpImgTileSize / TParams::kMmaImgTileSize;

// 	constexpr int kFragmentASize = TParams::kMmaTransTileSize * TParams::kMmaImgTileSize / kWarpSize;
// 	constexpr int kFragmentBSize = TParams::kMmaOrientTileSize * TParams::kMmaImgTileSize / kWarpSize;
// 	constexpr int kFragmentCSize = TParams::kMmaTransTileSize * TParams::kMmaOrientTileSize / kWarpSize;

// 	XFLOAT fragment_a[kNumMmaTransInWarpTile][kFragmentASize];
// 	XFLOAT fragment_b[kNumMmaOrientInWarpTile][kFragmentBSize];
// 	// XFLOAT fragment_c[kNumMmaTransInWarpTile][kNumMmaOrientInWarpTile][kFragmentCSize];

// 	// all the fragment_c in one block must cover all the elements in matrix C

// 	#pragma unroll
// 	for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 		#pragma unroll
// 		for (int j = 0; j < kFragmentASize; ++j) {
// 			fragment_a[i][j] = lane_id;
// 		}
// 	}
// 	#pragma unroll
// 	for (int i = 0; i < kNumMmaOrientInWarpTile; ++i) {
// 		#pragma unroll
// 		for (int j = 0; j < kFragmentBSize; ++j) {
// 			fragment_b[i][j] = lane_id;
// 		}
// 	}
// 	__syncthreads();

// 	for (int kk = 0; kk < TParams::kImgBlockSize; kk += TParams::kMmaImgTileSize) {

// 		#pragma unroll
// 		for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 			#pragma unroll
// 			for (int k = 0; k < kFragmentASize; ++k) {
// 				int s_m = fragment_a_m_idx_in_block<TParams>(warp_id, lane_id, i, k);
// 				int s_k = fragment_a_k_idx_in_block<TParams>(warp_id, lane_id, kk / TParams::kMmaImgTileSize, k);
// 				assert(s_m >= 0 && s_m < TParams::kTransBlockSize);
// 				assert(s_k >= 0 && s_k < TParams::kImgBlockSize);
// 				s_A[s_m * TParams::kImgBlockSize + s_k] = fragment_a[i][k];
// 			}
// 		}

// 		#pragma unroll
// 		for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 			#pragma unroll
// 			for (int k = 0; k < kFragmentBSize; ++k) {
// 				int s_n = fragment_b_n_idx_in_block<TParams>(warp_id, lane_id, j, k);
// 				int s_k = fragment_b_k_idx_in_block<TParams>(warp_id, lane_id, kk / TParams::kMmaImgTileSize, k);
// 				assert(s_n >= 0 && s_n < TParams::kOrientBlockSize);
// 				assert(s_k >= 0 && s_k < TParams::kImgBlockSize);
// 				s_B[s_n * TParams::kImgBlockSize + s_k] = fragment_b[j][k];
// 			}
// 		}
// 	}

// 	__syncthreads();

// 	// print matrix A, B
// 	if (tid == 0) {
// 		for (int i = 0; i < TParams::kTransBlockSize; ++i) {
// 			for (int j = 0; j < TParams::kImgBlockSize; ++j) {
// 				printf("%2d ", (int)s_A[i * TParams::kImgBlockSize + j]);
// 			}
// 			printf("\n");
// 		}

// 		printf("\n\n=============================================================================\n\n");
// 		// for (int i = 0; i < TParams::kOrientBlockSize; ++i) {
// 		// 	for (int j = 0; j < TParams::kImgBlockSize / 2; ++j) {
// 		// 		printf("%2d ", (int)s_B[i * TParams::kImgBlockSize + j]);
// 		// 	}

// 		for (int j = 0; j < TParams::kImgBlockSize; ++j) {
// 			for (int i = 0; i < TParams::kOrientBlockSize; ++i) {
// 				printf("%2d ", (int)s_B[i * TParams::kImgBlockSize + j]);
// 			}
// 			printf("\n");
// 		}
// 		printf("\n");
// 	}
	

// 	__syncthreads();

// 	// test there is no -1 element in matrix A, B
// 	for (int i = tid; i < TParams::kTransBlockSize * TParams::kImgBlockSize; i += blockDim.x) {
// 		assert(s_A[i] != -1);
// 	}

// 	for (int i = tid; i < TParams::kOrientBlockSize * TParams::kImgBlockSize; i += blockDim.x) {
// 		assert(s_B[i] != -1);
// 	}

// 	__syncthreads();


// }


// // compute s_A * s_B = reg_C
// template<typename TParams>
// __launch_bounds__(128, 2)
// __global__ void cuda_kernel_test_s_gemm_func(XFLOAT* g_A, XFLOAT* g_B, XFLOAT* g_C, int max_iter=1) {
// 	assert(TParams::kBlockSize == blockDim.x);
// 	const int tid = threadIdx.x;          // thread id in a block
// 	const int bid = blockIdx.x;           // block id in a grid
// 	const int warp_id  = tid / 32;        // warp id in a block
// 	const int warp_num = TParams::kBlockSize / 32; // number of warps in a block
// 	const int lane_id  = tid % 32;        // thread id in a warp

// 	__shared__ XFLOAT s_A[TParams::kTransBlockSize * TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_B[TParams::kOrientBlockSize * TParams::kImgBlockSize];
// 	// __shared__ short s_C[TParams::kTransBlockSize * TParams::kOrientBlockSize];

// 	// initialize s_A, s_B, s_C with 0
// 	// for (int i = tid; i < TParams::kTransBlockSize * TParams::kImgBlockSize; i += blockDim.x) {
// 	// 	s_A[i] = 1.0;
// 	// }
// 	// for (int i = tid; i < TParams::kOrientBlockSize * TParams::kImgBlockSize; i += blockDim.x) {
// 	// 	s_B[i] = 1.0;
// 	// }

// 	// copy g_A, g_B to s_A, s_B
// 	for (int i = tid; i < TParams::kTransBlockSize * TParams::kImgBlockSize; i += blockDim.x) {
// 		s_A[i] = g_A[i];
// 	}
// 	for (int i = tid; i < TParams::kOrientBlockSize * TParams::kImgBlockSize; i += blockDim.x) {
// 		s_B[i] = g_B[i];
// 	}
// 	__syncthreads();


// 	// test fragment_c_m_idx_in_block
// 	constexpr int kNumMmaTransInWarpTile = TParams::kWarpTransTileSize / TParams::kMmaTransTileSize;
// 	constexpr int kNumMmaOrientInWarpTile = TParams::kWarpOrientTileSize / TParams::kMmaOrientTileSize;
// 	constexpr int kNumMmaImgInWarpTile = TParams::kWarpImgTileSize / TParams::kMmaImgTileSize;

// 	constexpr int kFragmentASize = TParams::kMmaTransTileSize * TParams::kMmaImgTileSize / kWarpSize;
// 	constexpr int kFragmentBSize = TParams::kMmaOrientTileSize * TParams::kMmaImgTileSize / kWarpSize;
// 	constexpr int kFragmentCSize = TParams::kMmaTransTileSize * TParams::kMmaOrientTileSize / kWarpSize;

// 	XFLOAT fragment_a[kNumMmaTransInWarpTile][kFragmentASize];
// 	XFLOAT fragment_b[kNumMmaOrientInWarpTile][kFragmentBSize];
// 	XFLOAT fragment_c[kNumMmaTransInWarpTile][kNumMmaOrientInWarpTile][kFragmentCSize];

// 	for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 		for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 			for (int k = 0; k < kFragmentCSize; ++k) {
// 				fragment_c[i][j][k] = 0;
// 			}
// 		}
// 	}
// 	__syncthreads();

// 	// all the fragment_c in one block must cover all the elements in matrix C

// 	// #pragma unroll
// 	// for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 	// 	#pragma unroll
// 	// 	for (int j = 0; j < kFragmentASize; ++j) {
// 	// 		fragment_a[i][j] = lane_id;
// 	// 	}
// 	// }
// 	// #pragma unroll
// 	// for (int i = 0; i < kNumMmaOrientInWarpTile; ++i) {
// 	// 	#pragma unroll
// 	// 	for (int j = 0; j < kFragmentBSize; ++j) {
// 	// 		fragment_b[i][j] = lane_id;
// 	// 	}
// 	// }
// 	// __syncthreads();

// 	// constexpr int max_iter = 10000;

// 	for (int iter = 0; iter < max_iter; iter ++) {

// 	for (int kk = 0; kk < TParams::kImgBlockSize; kk += TParams::kMmaImgTileSize) {
		
// 		// load slice A
// 		#pragma unroll
// 		for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 			#pragma unroll
// 			for (int k = 0; k < kFragmentASize; ++k) {
// 				int s_m = fragment_a_m_idx_in_block<TParams>(warp_id, lane_id, i, k);
// 				int s_k = fragment_a_k_idx_in_block<TParams>(warp_id, lane_id, kk / TParams::kMmaImgTileSize, k);
// 				assert(s_m >= 0 && s_m < TParams::kTransBlockSize);
// 				assert(s_k >= 0 && s_k < TParams::kImgBlockSize);
// 				// s_A[s_m * TParams::kImgBlockSize + s_k] = fragment_a[i][k];
// 				fragment_a[i][k] = s_A[s_m * TParams::kImgBlockSize + s_k];
// 			}
// 		}

// 		// load slice B
// 		#pragma unroll
// 		for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 			#pragma unroll
// 			for (int k = 0; k < kFragmentBSize; ++k) {
// 				int s_n = fragment_b_n_idx_in_block<TParams>(warp_id, lane_id, j, k);
// 				int s_k = fragment_b_k_idx_in_block<TParams>(warp_id, lane_id, kk / TParams::kMmaImgTileSize, k);
// 				assert(s_n >= 0 && s_n < TParams::kOrientBlockSize);
// 				assert(s_k >= 0 && s_k < TParams::kImgBlockSize);
// 				// s_B[s_n * TParams::kImgBlockSize + s_k] = fragment_b[j][k];
// 				fragment_b[j][k] = s_B[s_n * TParams::kImgBlockSize + s_k];

// 			}
// 		}

// 		// compute an outproduct
// 		#pragma unroll
// 		for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 			#pragma unroll
// 			for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 				mma_sync_aligned_m16n8k8_row_col_tf32(fragment_c[i][j], fragment_a[i], fragment_b[j]);
// 			}
// 		}
// 	}
	
// 	}

// 	__syncthreads();

// 	// store fragment_c to g_C
// 	for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 		#pragma unroll
// 		for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 			#pragma unroll
// 			for (int k = 0; k < kFragmentCSize; ++k) {
// 				int m = fragment_c_m_idx_in_global<TParams>(0, warp_id, lane_id, i, k);
// 				int n = fragment_c_n_idx_in_global<TParams>(0, warp_id, lane_id, j, k);
// 				assert(m >= 0 && m < TParams::kTransBlockSize);
// 				assert(n >= 0 && n < TParams::kOrientBlockSize);
// 				g_C[m * TParams::kOrientBlockSize + n] = fragment_c[i][j][k];
// 			}
// 		}
// 	}

// 	__syncthreads();
// }


// // 使用了tensor core，并使用tf32去模拟fp32的精度，但是1. shared atomic 耗时较高，存在性能问题。 2. 未使用pipeline将texture fetch延迟掩盖。 3. sqrt(corr / 2) 的方法存在精度问题。
// template<typename TParams>
// __launch_bounds__(128, 2)
// __global__ void cuda_kernel_coarse_matrixV2(
// 	XFLOAT *g_eulers,
// 	XFLOAT *trans_x,
// 	XFLOAT *trans_y,
// 	XFLOAT *g_real,
// 	XFLOAT *g_imag,
// 	AccProjectorKernel projector,
// 	XFLOAT *g_corr,
// 	XFLOAT *g_diff2s,
// 	XFLOAT *g_diff2s_opt,
// 	int translation_num,
// 	int orientation_num,
// 	int image_size,
// 	XFLOAT *g_trans_real_m,
// 	XFLOAT *g_trans_imag_m,
// 	XFLOAT *g_orient_real_m,
// 	XFLOAT *g_orient_imag_m) {
// 	// assert(false);

// 	static_assert(TParams::kBlockSize % 32 == 0, "kBlockSize must be a multiple of 32");
// 	static_assert(TParams::kImgBlockSize == TParams::kWarpImgTileSize, "kImgBlockSize must be equal to kWarpImgTileSize");
// 	static_assert(TParams::kTransBlockSize % TParams::kWarpTransTileSize == 0, "kTransBlockSize must be a multiple of kWarpTransTileSize");
// 	static_assert(TParams::kOrientBlockSize % TParams::kWarpOrientTileSize == 0, "kOrientBlockSize must be a multiple of kWarpOrientTileSize");
// 	static_assert(TParams::kTransBlockSize % TParams::kMmaTransTileSize == 0, "kTransBlockSize must be a multiple of kMmaTransTileSize");
// 	static_assert(TParams::kOrientBlockSize % TParams::kMmaOrientTileSize == 0, "kOrientBlockSize must be a multiple of kMmaOrientTileSize");
// 	static_assert(TParams::kBlockSize / 32 == (TParams::kTransBlockSize / TParams::kWarpTransTileSize) * (TParams::kOrientBlockSize / TParams::kWarpOrientTileSize), "kBlockSize must be equal to the product of the number of warps in translation, orientation and image dimension");

// 	const int tid = threadIdx.x;          // thread id in a block
// 	const int bid = blockIdx.x;           // block id in a grid
// 	const int warp_id  = tid / 32;        // warp id in a block
// 	const int warp_num = TParams::kBlockSize / 32; // number of warps in a block
// 	const int lane_id  = tid % 32;        // thread id in a warp
	
// 	const int trans_block_num = (translation_num + TParams::kTransBlockSize - 1) / TParams::kTransBlockSize;
// 	const int orient_block_num = (orientation_num + TParams::kOrientBlockSize - 1) / TParams::kOrientBlockSize;

// 	const int trans_block_idx = bid % trans_block_num * TParams::kTransBlockSize;
// 	const int orient_block_idx = bid / trans_block_num * TParams::kOrientBlockSize;

// 	assert(blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0);
	
// 	// 'img' data is stored contiguously.
// 	__shared__ XFLOAT s_trans_real_mat_block[TParams::kTransBlockSize * TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_trans_imag_mat_block[TParams::kTransBlockSize * TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_orient_real_mat_block[TParams::kOrientBlockSize * TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_orient_imag_mat_block[TParams::kOrientBlockSize * TParams::kImgBlockSize];

// 	// 'translation' data is stored contiguously.
// 	// __shared__ XFLOAT s_diff_mat_block[TParams::kOrientBlockSize * TParams::kTransBlockSize];

// 	__shared__ XFLOAT s_corr_div_sqrt2[TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_coor_x[TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_coor_y[TParams::kImgBlockSize];

// 	__shared__ XFLOAT s_trans_pow2_accumulator[TParams::kTransBlockSize];
// 	__shared__ XFLOAT s_orient_pow2_accumulator[TParams::kOrientBlockSize];

// 	// register
// 	constexpr int kNumMmaTransInWarpTile = TParams::kWarpTransTileSize / TParams::kMmaTransTileSize;
// 	constexpr int kNumMmaOrientInWarpTile = TParams::kWarpOrientTileSize / TParams::kMmaOrientTileSize;
// 	constexpr int kNumMmaImgInWarpTile = TParams::kWarpImgTileSize / TParams::kMmaImgTileSize;

// 	constexpr int kFragmentASize = TParams::kMmaTransTileSize * TParams::kMmaImgTileSize / kWarpSize;
// 	constexpr int kFragmentBSize = TParams::kMmaOrientTileSize * TParams::kMmaImgTileSize / kWarpSize;
// 	constexpr int kFragmentCSize = TParams::kMmaTransTileSize * TParams::kMmaOrientTileSize / kWarpSize;

// 	XFLOAT fragment_a[kNumMmaTransInWarpTile][kFragmentASize];
// 	XFLOAT fragment_b[kNumMmaOrientInWarpTile][kFragmentBSize];
// 	XFLOAT fragment_c[kNumMmaTransInWarpTile][kNumMmaOrientInWarpTile][kFragmentCSize];

// 	constexpr int kRegistersMmaPerThread = kNumMmaTransInWarpTile * kFragmentASize +
// 										   kNumMmaOrientInWarpTile * kFragmentBSize +
// 										   kNumMmaTransInWarpTile * kNumMmaOrientInWarpTile * kFragmentCSize;
	
	

// 	// initialize shared memory to zero
// 	for (int i = tid; i < TParams::kTransBlockSize * TParams::kImgBlockSize; i += TParams::kBlockSize) {
// 		s_trans_real_mat_block[i] = 0.0;
// 		s_trans_imag_mat_block[i] = 0.0;
// 	}
// 	for (int i = tid; i < TParams::kOrientBlockSize * TParams::kImgBlockSize; i += TParams::kBlockSize) {
// 		s_orient_real_mat_block[i] = 0.0;
// 		s_orient_imag_mat_block[i] = 0.0;
// 	}
// 	for (int i = tid; i < TParams::kTransBlockSize; i += TParams::kBlockSize) {
// 		s_trans_pow2_accumulator[i] = 0.0;
// 		s_orient_pow2_accumulator[i] = 0.0;
// 	}
	
// 	// read fragment_c from g_diff2s
// 	#pragma unroll
// 	for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 		#pragma unroll
// 		for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 			#pragma unroll
// 			for (int k = 0; k < kFragmentCSize; ++k) {
// 				int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
// 				int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);
// 				if (m < translation_num && n < orientation_num) {
// 					fragment_c[i][j][k] = g_diff2s[n * translation_num + m];
// 				} else {
// 					fragment_c[i][j][k] = 0.0;
// 				}
// 			}
// 		}
// 	}

// 	__syncthreads();

// /*=============================== FOR IMAGE BLOCK ==============================*/
// 	for (int img_block_idx = 0; img_block_idx < image_size; img_block_idx += TParams::kImgBlockSize) {
// 		// TODO: offload this compute to CPU
// 		#pragma unroll 
// 		for (int i = tid; i < TParams::kImgBlockSize; i += TParams::kBlockSize) {
// 			if (img_block_idx + i < image_size) {
// 				s_corr_div_sqrt2[i] = sqrt(g_corr[img_block_idx + i] / 2);
// 			} else {
// 				s_corr_div_sqrt2[i] = 0.;
// 			}
// 		}
// 		for (int i = tid; i < TParams::kImgBlockSize; i += TParams::kBlockSize) {
// 			int x, y;
// 			pixel_index2coor(img_block_idx + i, projector.imgX, projector.imgY, projector.maxR, x, y);
// 			s_coor_x[i] = x;
// 			s_coor_y[i] = y;
// 		}
// 		__syncthreads();

// /*=============================== CONSTRUCT TRANS_MAT & ORIENT_MAT ==============================*/
// 		// construct trans_mat
// 		// one warp computes a translation row
// 		for (int i = warp_id; i < TParams::kTransBlockSize; i += warp_num) {
// 			XFLOAT real = 0.;
// 			XFLOAT imag = 0.;
// 			int g_trans_idx = trans_block_idx + i;
// 			if (g_trans_idx >= translation_num) continue;

// 			XFLOAT tx = trans_x[g_trans_idx];
// 			XFLOAT ty = trans_y[g_trans_idx];

// 			#pragma unroll
// 			for (int j = lane_id; j < TParams::kImgBlockSize; j += warpSize) {
// 				int g_img_idx = img_block_idx + j;
// 				if (g_img_idx >= image_size) {
// 					s_trans_real_mat_block[i * TParams::kImgBlockSize + j] = 0.;
// 					s_trans_imag_mat_block[i * TParams::kImgBlockSize + j] = 0.;
// 					continue;
// 				}
// 				real = g_real[g_img_idx] * s_corr_div_sqrt2[j];// * sqrt(g_corr[g_img_idx]) / sqrt(2.);
// 				imag = g_imag[g_img_idx] * s_corr_div_sqrt2[j];// * sqrt(g_corr[g_img_idx]) / sqrt(2.);
// 				int x, y;
// 				// pixel_index2coor(g_img_idx, projector.imgX, projector.imgY, projector.maxR, x, y);
// 				x = s_coor_x[j];
// 				y = s_coor_y[j];
// 				XFLOAT trans_real, trans_imag;
// 				translatePixel(x, y, tx, ty, real, imag, trans_real, trans_imag);

// 				// // TODO: DEBUG
// 				trans_real = 1.0 * s_corr_div_sqrt2[j];
// 				trans_imag = 2.0 * s_corr_div_sqrt2[j];

// 				// s_trans_real_mat_block[i * TParams::kImgBlockSize + j] = trans_real;
// 				// s_trans_imag_mat_block[i * TParams::kImgBlockSize + j] = trans_imag;
				
// 				// scale -2 there, because we need to compute - 2 * A * B = (-2 * A) * B
// 				s_trans_real_mat_block[i * TParams::kImgBlockSize + j] = -2 * trans_real;
// 				s_trans_imag_mat_block[i * TParams::kImgBlockSize + j] = -2 * trans_imag;

// 				// s_trans_pow2_accumulator[i] += trans_real * trans_real + trans_imag * trans_imag;
// 				XFLOAT magnitude_squared_sum = trans_real * trans_real + trans_imag * trans_imag;
// 				atomicAdd(&s_trans_pow2_accumulator[i], magnitude_squared_sum);
// 			}
// 		}

// 		// construct orient_mat
// 		for (int i = warp_id; i < TParams::kOrientBlockSize; i += warp_num) {
// 			int g_orient_idx = orient_block_idx + i;
// 			if (g_orient_idx >= orientation_num) continue;
// 			// first load eulers
// 			XFLOAT e0 = g_eulers[g_orient_idx * 9];
// 			XFLOAT e1 = g_eulers[g_orient_idx * 9 + 1];
// 			XFLOAT e3 = g_eulers[g_orient_idx * 9 + 3];
// 			XFLOAT e4 = g_eulers[g_orient_idx * 9 + 4];
// 			XFLOAT e6 = g_eulers[g_orient_idx * 9 + 6];
// 			XFLOAT e7 = g_eulers[g_orient_idx * 9 + 7];

// 			#pragma unroll
// 			for (int j = lane_id; j < TParams::kImgBlockSize; j += warpSize) {
// 				int g_img_idx = img_block_idx + j;
// 				if (g_img_idx >= image_size) {
// 					s_orient_real_mat_block[i * TParams::kImgBlockSize + j] = 0.;
// 					s_orient_imag_mat_block[i * TParams::kImgBlockSize + j] = 0.;
// 					continue;
// 				}
// 				int x, y;
// 				// pixel_index2coor(g_img_idx, projector.imgX, projector.imgY, projector.maxR, x, y);
// 				x = s_coor_x[j];
// 				y = s_coor_y[j];
// 				XFLOAT orient_real, orient_imag;
// 				projector.project3Dmodel(x, y, e0, e1, e3, e4, e6, e7, orient_real, orient_imag);
// 				XFLOAT orint_real_scale = orient_real * s_corr_div_sqrt2[j];
// 				XFLOAT orint_imag_scale = orient_imag * s_corr_div_sqrt2[j];
				
// 				// // TODO: DEBUG
// 				orint_real_scale = 3.0 * s_corr_div_sqrt2[j];
// 				orint_imag_scale = 4.0 * s_corr_div_sqrt2[j];

// 				s_orient_real_mat_block[i * TParams::kImgBlockSize + j] = orint_real_scale;// * sqrt(g_corr[g_img_idx]) / sqrt(2.);
// 				s_orient_imag_mat_block[i * TParams::kImgBlockSize + j] = orint_imag_scale;// * sqrt(g_corr[g_img_idx]) / sqrt(2.);
				
// 				XFLOAT magnitude_squared_sum = orint_real_scale * orint_real_scale + orint_imag_scale * orint_imag_scale;
// 				atomicAdd(&s_orient_pow2_accumulator[i], magnitude_squared_sum);
// 				// s_orient_pow2_accumulator[i] += orient_real * orient_real + orient_imag * orient_imag;
// 			}
// 		}

// /*=============================== COMPUTE CROSS TERM ==============================*/

// 	for (int kk = 0; kk < TParams::kImgBlockSize; kk += TParams::kMmaImgTileSize) {
		
// 		// load slice A
// 		#pragma unroll
// 		for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 			#pragma unroll
// 			for (int k = 0; k < kFragmentASize; ++k) {
// 				int s_m = fragment_a_m_idx_in_block<TParams>(warp_id, lane_id, i, k);
// 				int s_k = fragment_a_k_idx_in_block<TParams>(warp_id, lane_id, kk / TParams::kMmaImgTileSize, k);
// 				assert(s_m >= 0 && s_m < TParams::kTransBlockSize);
// 				assert(s_k >= 0 && s_k < TParams::kImgBlockSize);
// 				// s_A[s_m * TParams::kImgBlockSize + s_k] = fragment_a[i][k];
// 				fragment_a[i][k] = s_trans_real_mat_block[s_m * TParams::kImgBlockSize + s_k];
// 			}
// 		}

// 		// load slice B
// 		#pragma unroll
// 		for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 			#pragma unroll
// 			for (int k = 0; k < kFragmentBSize; ++k) {
// 				int s_n = fragment_b_n_idx_in_block<TParams>(warp_id, lane_id, j, k);
// 				int s_k = fragment_b_k_idx_in_block<TParams>(warp_id, lane_id, kk / TParams::kMmaImgTileSize, k);
// 				assert(s_n >= 0 && s_n < TParams::kOrientBlockSize);
// 				assert(s_k >= 0 && s_k < TParams::kImgBlockSize);
// 				// s_B[s_n * TParams::kImgBlockSize + s_k] = fragment_b[j][k];
// 				fragment_b[j][k] = s_orient_real_mat_block[s_n * TParams::kImgBlockSize + s_k];

// 			}
// 		}

// 		// compute an outproduct
// 		#pragma unroll
// 		for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 			#pragma unroll
// 			for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 				// mma_sync_aligned_m16n8k8_row_col_tf32(fragment_c[i][j], fragment_a[i], fragment_b[j]);
// 				mma_sync_aligned_m16n8k8_row_col_tf32_simulated_fp32(fragment_c[i][j], fragment_a[i], fragment_b[j]);
// 			}
// 		}

// 		#pragma unroll
// 		for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 			#pragma unroll
// 			for (int k = 0; k < kFragmentASize; ++k) {
// 				int s_m = fragment_a_m_idx_in_block<TParams>(warp_id, lane_id, i, k);
// 				int s_k = fragment_a_k_idx_in_block<TParams>(warp_id, lane_id, kk / TParams::kMmaImgTileSize, k);
// 				assert(s_m >= 0 && s_m < TParams::kTransBlockSize);
// 				assert(s_k >= 0 && s_k < TParams::kImgBlockSize);
// 				// s_A[s_m * TParams::kImgBlockSize + s_k] = fragment_a[i][k];
// 				fragment_a[i][k] = s_trans_imag_mat_block[s_m * TParams::kImgBlockSize + s_k];
// 			}
// 		}

// 		// load slice B
// 		#pragma unroll
// 		for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 			#pragma unroll
// 			for (int k = 0; k < kFragmentBSize; ++k) {
// 				int s_n = fragment_b_n_idx_in_block<TParams>(warp_id, lane_id, j, k);
// 				int s_k = fragment_b_k_idx_in_block<TParams>(warp_id, lane_id, kk / TParams::kMmaImgTileSize, k);
// 				assert(s_n >= 0 && s_n < TParams::kOrientBlockSize);
// 				assert(s_k >= 0 && s_k < TParams::kImgBlockSize);
// 				// s_B[s_n * TParams::kImgBlockSize + s_k] = fragment_b[j][k];
// 				fragment_b[j][k] = s_orient_imag_mat_block[s_n * TParams::kImgBlockSize + s_k];

// 			}
// 		}

// 		// compute an outproduct
// 		#pragma unroll
// 		for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 			#pragma unroll
// 			for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 				// mma_sync_aligned_m16n8k8_row_col_tf32(fragment_c[i][j], fragment_a[i], fragment_b[j]);
// 				mma_sync_aligned_m16n8k8_row_col_tf32_simulated_fp32(fragment_c[i][j], fragment_a[i], fragment_b[j]);
// 			}
// 		}
// 	}

// 	}

// /*=============================== REDUCE IN FRAGMENT_C ==============================*/
// 	__syncthreads();
	
// 	#pragma unroll
// 	for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 		#pragma unroll
// 		for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 			#pragma unroll
// 			for (int k = 0; k < kFragmentCSize; ++k) {
// 				int m = fragment_c_m_idx_in_block<TParams>(warp_id, lane_id, i, k);
// 				int n = fragment_c_n_idx_in_block<TParams>(warp_id, lane_id, j, k);
// 				// fragment_c[i][j][k] += s_trans_pow2_accumulator[m] + s_orient_pow2_accumulator[n];
// 			}
// 		}
// 	}

// 	__syncthreads();
		

// /*=============================== WRITE BACK ==============================*/
// 	// write fragment_c back to g_diff2s_opt
// 	#pragma unroll
// 	for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 		#pragma unroll
// 		for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 			#pragma unroll
// 			for (int k = 0; k < kFragmentCSize; ++k) {
// 				int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
// 				int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);

// 				assert(m >= 0 && m / TParams::kTransBlockSize <= translation_num / TParams::kTransBlockSize);
// 				assert(n >= 0 && n / TParams::kOrientBlockSize<= orientation_num / TParams::kOrientBlockSize);
// 				if (m < translation_num && n < orientation_num) {
// 					g_diff2s_opt[n * translation_num + m] = fragment_c[i][j][k];
// 				}
// 			}
// 		}
// 	}

// }
		

// // 使用了tensor core，并使用tf32去模拟fp32的精度，放弃了sqrt(corr / 2)的方法，并解决了同步bug，但是1. shared atomic 耗时较高，存在性能问题。 2. 未使用pipeline将texture fetch延迟掩盖。
// template<typename TParams>
// __launch_bounds__(128, 2)
// __global__ void cuda_kernel_coarse_matrixV3(
// 	XFLOAT *g_eulers,
// 	XFLOAT *trans_x,
// 	XFLOAT *trans_y,
// 	XFLOAT *g_real,
// 	XFLOAT *g_imag,
// 	AccProjectorKernel projector,
// 	XFLOAT *g_corr,
// 	XFLOAT *g_diff2s,
// 	XFLOAT *g_diff2s_opt,
// 	int translation_num,
// 	int orientation_num,
// 	int image_size,
// 	XFLOAT *g_trans_real_m,
// 	XFLOAT *g_trans_imag_m,
// 	XFLOAT *g_orient_real_m,
// 	XFLOAT *g_orient_imag_m) {
// 	// assert(false);

// 	static_assert(TParams::kBlockSize % 32 == 0, "kBlockSize must be a multiple of 32");
// 	static_assert(TParams::kImgBlockSize == TParams::kWarpImgTileSize, "kImgBlockSize must be equal to kWarpImgTileSize");
// 	static_assert(TParams::kTransBlockSize % TParams::kWarpTransTileSize == 0, "kTransBlockSize must be a multiple of kWarpTransTileSize");
// 	static_assert(TParams::kOrientBlockSize % TParams::kWarpOrientTileSize == 0, "kOrientBlockSize must be a multiple of kWarpOrientTileSize");
// 	static_assert(TParams::kTransBlockSize % TParams::kMmaTransTileSize == 0, "kTransBlockSize must be a multiple of kMmaTransTileSize");
// 	static_assert(TParams::kOrientBlockSize % TParams::kMmaOrientTileSize == 0, "kOrientBlockSize must be a multiple of kMmaOrientTileSize");
// 	static_assert(TParams::kBlockSize / 32 == (TParams::kTransBlockSize / TParams::kWarpTransTileSize) * (TParams::kOrientBlockSize / TParams::kWarpOrientTileSize), "kBlockSize must be equal to the product of the number of warps in translation, orientation and image dimension");

// 	const int tid = threadIdx.x;          // thread id in a block
// 	const int bid = blockIdx.x;           // block id in a grid
// 	const int warp_id  = tid / 32;        // warp id in a block
// 	const int warp_num = TParams::kBlockSize / 32; // number of warps in a block
// 	const int lane_id  = tid % 32;        // thread id in a warp
	
// 	const int trans_block_num = (translation_num + TParams::kTransBlockSize - 1) / TParams::kTransBlockSize;
// 	const int orient_block_num = (orientation_num + TParams::kOrientBlockSize - 1) / TParams::kOrientBlockSize;

// 	const int trans_block_idx = bid % trans_block_num * TParams::kTransBlockSize;
// 	const int orient_block_idx = bid / trans_block_num * TParams::kOrientBlockSize;

// 	assert(blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0);
	
// 	// 'img' data is stored contiguously.
// 	__shared__ XFLOAT s_trans_real_mat_block[TParams::kTransBlockSize * TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_trans_imag_mat_block[TParams::kTransBlockSize * TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_orient_real_mat_block[TParams::kOrientBlockSize * TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_orient_imag_mat_block[TParams::kOrientBlockSize * TParams::kImgBlockSize];

// 	// 'translation' data is stored contiguously.
// 	// __shared__ XFLOAT s_diff_mat_block[TParams::kOrientBlockSize * TParams::kTransBlockSize];

// 	__shared__ XFLOAT s_corr_div_2[TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_coor_x[TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_coor_y[TParams::kImgBlockSize];

// 	__shared__ XFLOAT s_trans_pow2_accumulator[TParams::kTransBlockSize];
// 	__shared__ XFLOAT s_orient_pow2_accumulator[TParams::kOrientBlockSize];

// 	// register
// 	constexpr int kNumMmaTransInWarpTile = TParams::kWarpTransTileSize / TParams::kMmaTransTileSize;
// 	constexpr int kNumMmaOrientInWarpTile = TParams::kWarpOrientTileSize / TParams::kMmaOrientTileSize;
// 	constexpr int kNumMmaImgInWarpTile = TParams::kWarpImgTileSize / TParams::kMmaImgTileSize;

// 	constexpr int kFragmentASize = TParams::kMmaTransTileSize * TParams::kMmaImgTileSize / kWarpSize;
// 	constexpr int kFragmentBSize = TParams::kMmaOrientTileSize * TParams::kMmaImgTileSize / kWarpSize;
// 	constexpr int kFragmentCSize = TParams::kMmaTransTileSize * TParams::kMmaOrientTileSize / kWarpSize;

// 	XFLOAT fragment_a[kNumMmaTransInWarpTile][kFragmentASize];
// 	XFLOAT fragment_b[kNumMmaOrientInWarpTile][kFragmentBSize];
// 	XFLOAT fragment_c[kNumMmaTransInWarpTile][kNumMmaOrientInWarpTile][kFragmentCSize];

// 	constexpr int kRegistersMmaPerThread = kNumMmaTransInWarpTile * kFragmentASize +
// 										   kNumMmaOrientInWarpTile * kFragmentBSize +
// 										   kNumMmaTransInWarpTile * kNumMmaOrientInWarpTile * kFragmentCSize;
	
	

// 	// initialize shared memory to zero
// 	for (int i = tid; i < TParams::kTransBlockSize * TParams::kImgBlockSize; i += TParams::kBlockSize) {
// 		s_trans_real_mat_block[i] = 0.0;
// 		s_trans_imag_mat_block[i] = 0.0;
// 	}
// 	for (int i = tid; i < TParams::kOrientBlockSize * TParams::kImgBlockSize; i += TParams::kBlockSize) {
// 		s_orient_real_mat_block[i] = 0.0;
// 		s_orient_imag_mat_block[i] = 0.0;
// 	}
// 	for (int i = tid; i < TParams::kTransBlockSize; i += TParams::kBlockSize) {
// 		s_trans_pow2_accumulator[i] = 0.0;
// 		s_orient_pow2_accumulator[i] = 0.0;
// 	}
	
// 	// read fragment_c from g_diff2s
// 	#pragma unroll
// 	for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 		#pragma unroll
// 		for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 			#pragma unroll
// 			for (int k = 0; k < kFragmentCSize; ++k) {
// 				int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
// 				int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);
// 				if (m < translation_num && n < orientation_num) {
// 					fragment_c[i][j][k] = g_diff2s[n * translation_num + m];
// 					// fragment_c[i][j][k] = 0.0;
// 				} else {
// 					fragment_c[i][j][k] = 0.0;
// 				}
// 			}
// 		}
// 	}

// 	__syncthreads();

// /*=============================== FOR IMAGE BLOCK ==============================*/
// 	for (int img_block_idx = 0; img_block_idx < image_size; img_block_idx += TParams::kImgBlockSize) {
// 		// TODO: offload this compute to CPU
// 		#pragma unroll 
// 		for (int i = tid; i < TParams::kImgBlockSize; i += TParams::kBlockSize) {
// 			if (img_block_idx + i < image_size) {
// 				s_corr_div_2[i] = g_corr[img_block_idx + i] / 2;
// 			} else {
// 				s_corr_div_2[i] = 0.;
// 			}
// 		}
// 		for (int i = tid; i < TParams::kImgBlockSize; i += TParams::kBlockSize) {
// 			int x, y;
// 			pixel_index2coor(img_block_idx + i, projector.imgX, projector.imgY, projector.maxR, x, y);
// 			s_coor_x[i] = x;
// 			s_coor_y[i] = y;
// 		}
// 		__syncthreads();

// /*=============================== CONSTRUCT TRANS_MAT & ORIENT_MAT ==============================*/
// 		// construct trans_mat
// 		// one warp computes a translation row
// 		for (int i = warp_id; i < TParams::kTransBlockSize; i += warp_num) {
// 			XFLOAT real = 0.;
// 			XFLOAT imag = 0.;
// 			int g_trans_idx = trans_block_idx + i;
// 			if (g_trans_idx >= translation_num) continue;

// 			XFLOAT tx = trans_x[g_trans_idx];
// 			XFLOAT ty = trans_y[g_trans_idx];

// 			#pragma unroll
// 			for (int j = lane_id; j < TParams::kImgBlockSize; j += warpSize) {
// 				int g_img_idx = img_block_idx + j;
// 				if (g_img_idx >= image_size) {
// 					s_trans_real_mat_block[i * TParams::kImgBlockSize + j] = 0.;
// 					s_trans_imag_mat_block[i * TParams::kImgBlockSize + j] = 0.;
// 					continue;
// 				}
// 				// real = g_real[g_img_idx] * s_corr_div_2[j];
// 				// imag = g_imag[g_img_idx] * s_corr_div_2[j];
// 				real = g_real[g_img_idx];
// 				imag = g_imag[g_img_idx];
// 				int x, y;
// 				// pixel_index2coor(g_img_idx, projector.imgX, projector.imgY, projector.maxR, x, y);
// 				x = s_coor_x[j];
// 				y = s_coor_y[j];
// 				XFLOAT trans_real, trans_imag;
// 				translatePixel(x, y, tx, ty, real, imag, trans_real, trans_imag);

// 				// // TODO: DEBUG
// 				// trans_real = 1.0 * (g_img_idx% 2);
// 				// trans_imag = 2.0 * (g_img_idx% 2);

// 				// s_trans_real_mat_block[i * TParams::kImgBlockSize + j] = trans_real;
// 				// s_trans_imag_mat_block[i * TParams::kImgBlockSize + j] = trans_imag;
				
// 				// scale -2 there, because we need to compute - 2 * A * B = (-2 * A) * B
// 				s_trans_real_mat_block[i * TParams::kImgBlockSize + j] = -2 * trans_real * s_corr_div_2[j];
// 				s_trans_imag_mat_block[i * TParams::kImgBlockSize + j] = -2 * trans_imag * s_corr_div_2[j];

// 				// s_trans_real_mat_block[i * TParams::kImgBlockSize + j] = -2 * (g_img_idx % 11);
// 				// s_trans_imag_mat_block[i * TParams::kImgBlockSize + j] = -2 * (g_img_idx % 17);

// 				// s_trans_pow2_accumulator[i] += trans_real * trans_real + trans_imag * trans_imag;
// 				XFLOAT magnitude_squared_sum = trans_real * trans_real * s_corr_div_2[j] + trans_imag * trans_imag * s_corr_div_2[j];
// 				// XFLOAT magnitude_squared_sum = 1.0 * 1.0 * (g_img_idx% 2) + 2.0 * 2.0 * (g_img_idx% 2);
// 				atomicAdd(&s_trans_pow2_accumulator[i], magnitude_squared_sum);
// 			}
// 		}

// 		// construct orient_mat
// 		for (int i = warp_id; i < TParams::kOrientBlockSize; i += warp_num) {
// 			int g_orient_idx = orient_block_idx + i;
// 			if (g_orient_idx >= orientation_num) continue;
// 			// first load eulers
// 			XFLOAT e0 = g_eulers[g_orient_idx * 9];
// 			XFLOAT e1 = g_eulers[g_orient_idx * 9 + 1];
// 			XFLOAT e3 = g_eulers[g_orient_idx * 9 + 3];
// 			XFLOAT e4 = g_eulers[g_orient_idx * 9 + 4];
// 			XFLOAT e6 = g_eulers[g_orient_idx * 9 + 6];
// 			XFLOAT e7 = g_eulers[g_orient_idx * 9 + 7];

// 			#pragma unroll
// 			for (int j = lane_id; j < TParams::kImgBlockSize; j += warpSize) {
// 				int g_img_idx = img_block_idx + j;
// 				if (g_img_idx >= image_size) {
// 					s_orient_real_mat_block[i * TParams::kImgBlockSize + j] = 0.;
// 					s_orient_imag_mat_block[i * TParams::kImgBlockSize + j] = 0.;
// 					continue;
// 				}
// 				int x, y;
// 				// pixel_index2coor(g_img_idx, projector.imgX, projector.imgY, projector.maxR, x, y);
// 				x = s_coor_x[j];
// 				y = s_coor_y[j];
// 				XFLOAT orient_real, orient_imag;
// 				projector.project3Dmodel(x, y, e0, e1, e3, e4, e6, e7, orient_real, orient_imag);
// 				// TODO: do not scale here
// 				// XFLOAT orint_real_scale = orient_real * s_corr_div_2[j];
// 				// XFLOAT orint_imag_scale = orient_imag * s_corr_div_2[j];
				
// 				XFLOAT orint_real_scale = orient_real;
// 				XFLOAT orint_imag_scale = orient_imag;
// 				// // // TODO: DEBUG
// 				// orint_real_scale = 2.0;
// 				// orint_imag_scale = 3.0;

// 				s_orient_real_mat_block[i * TParams::kImgBlockSize + j] = orint_real_scale;
// 				s_orient_imag_mat_block[i * TParams::kImgBlockSize + j] = orint_imag_scale;

// 				// s_orient_real_mat_block[i * TParams::kImgBlockSize + j] = (g_img_idx % 7);
// 				// s_orient_imag_mat_block[i * TParams::kImgBlockSize + j] = (g_img_idx % 11);
				
// 				XFLOAT magnitude_squared_sum = orint_real_scale * orint_real_scale * s_corr_div_2[j] + orint_imag_scale * orint_imag_scale * s_corr_div_2[j];
// 				// XFLOAT magnitude_squared_sum = 2.0 * 2.0 * s_corr_div_2[j] + 3.0 * 3.0 * s_corr_div_2[j];
// 				atomicAdd(&s_orient_pow2_accumulator[i], magnitude_squared_sum);
// 				// s_orient_pow2_accumulator[i] += orient_real * orient_real + orient_imag * orient_imag;
// 			}
// 		}
	
// 	__syncthreads();

// /*=============================== COMPUTE CROSS TERM ==============================*/

// 	for (int kk = 0; kk < TParams::kImgBlockSize; kk += TParams::kMmaImgTileSize) {
		
// 		// load slice A
// 		#pragma unroll
// 		for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 			#pragma unroll
// 			for (int k = 0; k < kFragmentASize; ++k) {
// 				int s_m = fragment_a_m_idx_in_block<TParams>(warp_id, lane_id, i, k);
// 				int s_k = fragment_a_k_idx_in_block<TParams>(warp_id, lane_id, kk / TParams::kMmaImgTileSize, k);
// 				assert(s_m >= 0 && s_m < TParams::kTransBlockSize);
// 				assert(s_k >= 0 && s_k < TParams::kImgBlockSize);
// 				// s_A[s_m * TParams::kImgBlockSize + s_k] = fragment_a[i][k];
// 				fragment_a[i][k] = s_trans_real_mat_block[s_m * TParams::kImgBlockSize + s_k];
// 			}
// 		}

// 		// load slice B
// 		#pragma unroll
// 		for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 			#pragma unroll
// 			for (int k = 0; k < kFragmentBSize; ++k) {
// 				int s_n = fragment_b_n_idx_in_block<TParams>(warp_id, lane_id, j, k);
// 				int s_k = fragment_b_k_idx_in_block<TParams>(warp_id, lane_id, kk / TParams::kMmaImgTileSize, k);
// 				assert(s_n >= 0 && s_n < TParams::kOrientBlockSize);
// 				assert(s_k >= 0 && s_k < TParams::kImgBlockSize);
// 				// s_B[s_n * TParams::kImgBlockSize + s_k] = fragment_b[j][k];
// 				fragment_b[j][k] = s_orient_real_mat_block[s_n * TParams::kImgBlockSize + s_k];

// 			}
// 		}

// 		// compute an outproduct
// 		#pragma unroll
// 		for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 			#pragma unroll
// 			for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 				// mma_sync_aligned_m16n8k8_row_col_tf32(fragment_c[i][j], fragment_a[i], fragment_b[j]);
// 				mma_sync_aligned_m16n8k8_row_col_tf32_simulated_fp32(fragment_c[i][j], fragment_a[i], fragment_b[j]);
// 			}
// 		}

// 		#pragma unroll
// 		for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 			#pragma unroll
// 			for (int k = 0; k < kFragmentASize; ++k) {
// 				int s_m = fragment_a_m_idx_in_block<TParams>(warp_id, lane_id, i, k);
// 				int s_k = fragment_a_k_idx_in_block<TParams>(warp_id, lane_id, kk / TParams::kMmaImgTileSize, k);
// 				assert(s_m >= 0 && s_m < TParams::kTransBlockSize);
// 				assert(s_k >= 0 && s_k < TParams::kImgBlockSize);
// 				// s_A[s_m * TParams::kImgBlockSize + s_k] = fragment_a[i][k];
// 				fragment_a[i][k] = s_trans_imag_mat_block[s_m * TParams::kImgBlockSize + s_k];
// 			}
// 		}

// 		// load slice B
// 		#pragma unroll
// 		for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 			#pragma unroll
// 			for (int k = 0; k < kFragmentBSize; ++k) {
// 				int s_n = fragment_b_n_idx_in_block<TParams>(warp_id, lane_id, j, k);
// 				int s_k = fragment_b_k_idx_in_block<TParams>(warp_id, lane_id, kk / TParams::kMmaImgTileSize, k);
// 				assert(s_n >= 0 && s_n < TParams::kOrientBlockSize);
// 				assert(s_k >= 0 && s_k < TParams::kImgBlockSize);
// 				// s_B[s_n * TParams::kImgBlockSize + s_k] = fragment_b[j][k];
// 				fragment_b[j][k] = s_orient_imag_mat_block[s_n * TParams::kImgBlockSize + s_k];

// 			}
// 		}

// 		// compute an outproduct
// 		#pragma unroll
// 		for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 			#pragma unroll
// 			for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 				// mma_sync_aligned_m16n8k8_row_col_tf32(fragment_c[i][j], fragment_a[i], fragment_b[j]);
// 				mma_sync_aligned_m16n8k8_row_col_tf32_simulated_fp32(fragment_c[i][j], fragment_a[i], fragment_b[j]);
// 			}
// 		}
// 	}

// 	}

// /*=============================== REDUCE IN FRAGMENT_C ==============================*/
// 	__syncthreads();
	
// 	#pragma unroll
// 	for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 		#pragma unroll
// 		for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 			#pragma unroll
// 			for (int k = 0; k < kFragmentCSize; ++k) {
// 				int m = fragment_c_m_idx_in_block<TParams>(warp_id, lane_id, i, k);
// 				int n = fragment_c_n_idx_in_block<TParams>(warp_id, lane_id, j, k);
// 				fragment_c[i][j][k] += s_trans_pow2_accumulator[m] + s_orient_pow2_accumulator[n];
// 			}
// 		}
// 	}

// 	__syncthreads();
		

// /*=============================== WRITE BACK ==============================*/
// 	// write fragment_c back to g_diff2s_opt
// 	#pragma unroll
// 	for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 		#pragma unroll
// 		for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 			#pragma unroll
// 			for (int k = 0; k < kFragmentCSize; ++k) {
// 				int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
// 				int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);

// 				assert(m >= 0 && m / TParams::kTransBlockSize <= translation_num / TParams::kTransBlockSize);
// 				assert(n >= 0 && n / TParams::kOrientBlockSize<= orientation_num / TParams::kOrientBlockSize);
// 				if (m < translation_num && n < orientation_num) {
// 					g_diff2s_opt[n * translation_num + m] = fragment_c[i][j][k];
// 				}
// 			}
// 		}
// 	}

// }



// template<typename TParams>
// __launch_bounds__(128, 2)
// __global__ void cuda_kernel_coarse_matrixV4(
// 	XFLOAT *g_eulers,
// 	XFLOAT *trans_x,
// 	XFLOAT *trans_y,
// 	XFLOAT *g_real,
// 	XFLOAT *g_imag,
// 	AccProjectorKernel projector,
// 	XFLOAT *g_corr,
// 	XFLOAT *g_diff2s,
// 	XFLOAT *g_diff2s_opt,
// 	int translation_num,
// 	int orientation_num,
// 	int image_size,
// 	XFLOAT *g_trans_real_m,
// 	XFLOAT *g_trans_imag_m,
// 	XFLOAT *g_orient_real_m,
// 	XFLOAT *g_orient_imag_m) {
// 	static_assert(TParams::kBlockSize % 32 == 0, "kBlockSize must be a multiple of 32");
// 	static_assert(TParams::kImgBlockSize == TParams::kWarpImgTileSize, "kImgBlockSize must be equal to kWarpImgTileSize");
// 	static_assert(TParams::kBlockSize >= TParams::kTransBlockSize, "kBlockSize must be greater than or equal to kTransBlockSize");
// 	static_assert(TParams::kBlockSize >= TParams::kOrientBlockSize, "kBlockSize must be greater than or equal to kOrientBlockSize");

// 	static_assert(TParams::kTransBlockSize % TParams::kWarpTransTileSize == 0, "kTransBlockSize must be a multiple of kWarpTransTileSize");
// 	static_assert(TParams::kOrientBlockSize % TParams::kWarpOrientTileSize == 0, "kOrientBlockSize must be a multiple of kWarpOrientTileSize");
// 	static_assert(TParams::kTransBlockSize % TParams::kMmaTransTileSize == 0, "kTransBlockSize must be a multiple of kMmaTransTileSize");
// 	static_assert(TParams::kOrientBlockSize % TParams::kMmaOrientTileSize == 0, "kOrientBlockSize must be a multiple of kMmaOrientTileSize");
// 	static_assert(TParams::kBlockSize / 32 == (TParams::kTransBlockSize / TParams::kWarpTransTileSize) * (TParams::kOrientBlockSize / TParams::kWarpOrientTileSize), "kBlockSize must be equal to the product of the number of warps in translation, orientation and image dimension");

// 	const int tid = threadIdx.x;          // thread id in a block
// 	const int bid = blockIdx.x;           // block id in a grid
// 	const int warp_id  = tid / 32;        // warp id in a block
// 	const int warp_num = TParams::kBlockSize / 32; // number of warps in a block
// 	const int lane_id  = tid % 32;        // thread id in a warp
	
// 	const int trans_block_num = (translation_num + TParams::kTransBlockSize - 1) / TParams::kTransBlockSize;
// 	const int orient_block_num = (orientation_num + TParams::kOrientBlockSize - 1) / TParams::kOrientBlockSize;

// 	// bug:这里一开始没有括号
// 	const int trans_block_idx = (bid % trans_block_num) * TParams::kTransBlockSize;
// 	const int orient_block_idx = (bid / trans_block_num) * TParams::kOrientBlockSize;

// 	assert(blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0);
	
// 	// 'img' data is stored contiguously.
// 	__shared__ XFLOAT s_trans_real_mat_block[TParams::kTransBlockSize * TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_trans_imag_mat_block[TParams::kTransBlockSize * TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_orient_real_mat_block[TParams::kOrientBlockSize * TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_orient_imag_mat_block[TParams::kOrientBlockSize * TParams::kImgBlockSize];

// 	__shared__ XFLOAT s_corr_div_2[2][TParams::kImgBlockSize];
// 	//double buffer for s_coor_x, s_coor_y
// 	__shared__ XFLOAT s_coor_x[2][TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_coor_y[2][TParams::kImgBlockSize];

// 	// reduce buffer
// 	__shared__ XFLOAT s_trans_pow2_accumulator[(TParams::kBlockSize / TParams::kTransBlockSize) * TParams::kTransBlockSize];
// 	__shared__ XFLOAT s_orient_pow2_accumulator[(TParams::kBlockSize / TParams::kOrientBlockSize) * TParams::kOrientBlockSize];

// 	// register
// 	constexpr int kNumMmaTransInWarpTile = TParams::kWarpTransTileSize / TParams::kMmaTransTileSize;
// 	constexpr int kNumMmaOrientInWarpTile = TParams::kWarpOrientTileSize / TParams::kMmaOrientTileSize;
// 	constexpr int kNumMmaImgInWarpTile = TParams::kWarpImgTileSize / TParams::kMmaImgTileSize;

// 	constexpr int kFragmentASize = TParams::kMmaTransTileSize * TParams::kMmaImgTileSize / kWarpSize;
// 	constexpr int kFragmentBSize = TParams::kMmaOrientTileSize * TParams::kMmaImgTileSize / kWarpSize;
// 	constexpr int kFragmentCSize = TParams::kMmaTransTileSize * TParams::kMmaOrientTileSize / kWarpSize;

// 	XFLOAT fragment_a[kNumMmaTransInWarpTile][kFragmentASize];
// 	XFLOAT fragment_b[kNumMmaOrientInWarpTile][kFragmentBSize];
// 	XFLOAT fragment_c[kNumMmaTransInWarpTile][kNumMmaOrientInWarpTile][kFragmentCSize];

// 	constexpr int kRegistersMmaPerThread = kNumMmaTransInWarpTile * kFragmentASize +
// 										   kNumMmaOrientInWarpTile * kFragmentBSize +
// 										   kNumMmaTransInWarpTile * kNumMmaOrientInWarpTile * kFragmentCSize;
	
	

// 	// initialize shared memory to zero
// 	for (int i = tid; i < TParams::kTransBlockSize * TParams::kImgBlockSize; i += TParams::kBlockSize) {
// 		s_trans_real_mat_block[i] = 0.0;
// 		s_trans_imag_mat_block[i] = 0.0;
// 	}
// 	for (int i = tid; i < TParams::kOrientBlockSize * TParams::kImgBlockSize; i += TParams::kBlockSize) {
// 		s_orient_real_mat_block[i] = 0.0;
// 		s_orient_imag_mat_block[i] = 0.0;
// 	}
// 	// bug: 这里初始化有问题
// 	// for (int i = tid; i < TParams::kTransBlockSize; i += TParams::kBlockSize) {
// 	// 	s_trans_pow2_accumulator[i] = 0.0;
// 	// 	s_orient_pow2_accumulator[i] = 0.0;
// 	// }
// 	s_trans_pow2_accumulator[tid] = 0.0;
// 	s_orient_pow2_accumulator[tid] = 0.0;

// 	for (int i = tid; i < TParams::kImgBlockSize; i += TParams::kBlockSize) {
// 		s_corr_div_2[0][i] = 0.0;
// 		s_corr_div_2[1][i] = 0.0;
// 		s_coor_x[0][i] = 0.0;
// 		s_coor_y[0][i] = 0.0;
// 		s_coor_x[1][i] = 0.0;
// 		s_coor_y[1][i] = 0.0;
// 	}
	
// 	// read fragment_c from g_diff2s
// 	#pragma unroll
// 	for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 		#pragma unroll
// 		for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 			#pragma unroll
// 			for (int k = 0; k < kFragmentCSize; ++k) {
// 				int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
// 				int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);
// 				if (m < translation_num && n < orientation_num) {
// 					fragment_c[i][j][k] = g_diff2s[n * translation_num + m];
// 				} else {
// 					fragment_c[i][j][k] = 0.0;
// 				}
// 			}
// 		}
// 	}

// 	__syncthreads();
// 	//given current img_block_idx, load global array into corr_div_2, coord_x, coord_y
// 	auto load_coord_xy = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y) {
// 		#pragma unroll 
// 		for (int i = tid; i < TParams::kImgBlockSize; i += TParams::kBlockSize) {
// 			if (img_block_idx + i < image_size) {
// 				corr_div_2[i] = g_corr[img_block_idx + i] / 2;
// 			} else {
// 				corr_div_2[i] = 0.;
// 			}

// 			int x, y;
// 			pixel_index2coor(img_block_idx + i, projector.imgX, projector.imgY, projector.maxR, x, y);
// 			coord_x[i] = x;
// 			coord_y[i] = y;
// 		}
// 	};

// 	//given current img_block_idx, corr_div_2, coord_x, coord_y, load trans mat into s_trans_real_mat_block and s_trans_imag_mat_block
// 	auto load_trans_mat = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y) {
// 		assert(TParams::kBlockSize % TParams::kTransBlockSize == 0);
// 		#pragma unroll
// 		for (int i = tid / TParams::kTransBlockSize; i < TParams::kImgBlockSize; i += TParams::kBlockSize / TParams::kTransBlockSize) {
// 			int g_img_idx = img_block_idx + i;
// 			int trans_idx = tid % TParams::kTransBlockSize;
// 			if (g_img_idx >= image_size) {
// 				assert(trans_idx < TParams::kTransBlockSize);
// 				assert(i < TParams::kImgBlockSize);
// 				s_trans_real_mat_block[trans_idx * TParams::kImgBlockSize + i] = 0.;
// 				s_trans_imag_mat_block[trans_idx * TParams::kImgBlockSize + i] = 0.;
// 				continue;
// 			}
// 			int g_trans_idx = trans_block_idx + trans_idx;
// 			if (g_trans_idx >= translation_num) {
// 				continue;
// 			}
// 			XFLOAT tx = trans_x[g_trans_idx];
// 			XFLOAT ty = trans_y[g_trans_idx];
// 			XFLOAT real = g_real[g_img_idx];
// 			XFLOAT imag = g_imag[g_img_idx];

// 			int x = coord_x[i];
// 			int y = coord_y[i];
// 			XFLOAT trans_real, trans_imag;
// 			translatePixel(x, y, tx, ty, real, imag, trans_real, trans_imag);

// 			s_trans_real_mat_block[trans_idx * TParams::kImgBlockSize + i] = -2 * trans_real * corr_div_2[i];
// 			s_trans_imag_mat_block[trans_idx * TParams::kImgBlockSize + i] = -2 * trans_imag * corr_div_2[i];

// 			XFLOAT magnitude_squared_sum = trans_real * trans_real * corr_div_2[i] + trans_imag * trans_imag * corr_div_2[i];
// 			s_trans_pow2_accumulator[tid] += magnitude_squared_sum;
// 		}
// 	};

// 	auto project3Dmodel_sp = [&](
// 			XFLOAT x,
// 			XFLOAT y,
// 			XFLOAT e0,
// 			XFLOAT e1,
// 			XFLOAT e3,
// 			XFLOAT e4,
// 			XFLOAT e6,
// 			XFLOAT e7,
// 			XFLOAT &real,
// 			XFLOAT &imag,
// 			uint32_t& flag_minus,
// 			uint32_t mask) {
// 		XFLOAT xp = (e0 * x + e1 * y ) * projector.padding_factor;
// 		XFLOAT yp = (e3 * x + e4 * y ) * projector.padding_factor;
// 		XFLOAT zp = (e6 * x + e7 * y ) * projector.padding_factor;
// 		int r2 = xp*xp + yp*yp + zp*zp;
// 		if (r2 <= projector.maxR2_padded)
// 		{
// 			bool xp_neg = xp < 0;
// 			flag_minus += xp_neg ? mask : 0;
// 			// NOTICE: if xp_neg, imag = -imag
// 			if (xp_neg) {
// 				// Get complex conjugated hermitian symmetry pair
// 				xp = -xp;
// 				yp = -yp;
// 				zp = -zp;
// 				yp -= projector.mdlInitY;
// 				zp -= projector.mdlInitZ;
// 			}
// 			else {
// 				yp -= projector.mdlInitY;
// 				zp -= projector.mdlInitZ;
// 			}
// 			real =    tex3D<XFLOAT>(projector.mdlReal, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
// 			imag =    tex3D<XFLOAT>(projector.mdlImag, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
			
// 			// if(xp_neg) {
// 			// 	imag = -imag;
// 			// }
// 		}
// 		else {
// 			real = (XFLOAT)0;
// 			imag = (XFLOAT)0;
// 		}
// 	};

// 	constexpr int kDimOrientSlice = (TParams::kBlockSize / TParams::kOrientBlockSize);
// 	constexpr int kNumOrientSlice = (TParams::kImgBlockSize + kDimOrientSlice - 1) / kDimOrientSlice;
// 	assert(kNumOrientSlice <=32);
// 	XFLOAT orient_real_buf[kNumOrientSlice], orient_imag_buf[kNumOrientSlice];
// 	uint32_t flag_minus_buf[2] = {0, 0};

// 	//given current img_block_idx, corr_div_2, coord_x, coord_y, load orient mat into orient_real_buf and orient_imag_buf
// 	auto load_orient_mat_buf = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y, uint32_t& flag_minus)  {
// 		uint32_t flag_minus_loc = 0;

// 		#pragma unroll
// 		for (int cur_slice = 0; cur_slice < kNumOrientSlice; cur_slice++) {
// 			int i = tid / TParams::kOrientBlockSize + cur_slice * kDimOrientSlice;
// 			XFLOAT& orient_real = orient_real_buf[cur_slice];
// 			XFLOAT& orient_imag = orient_imag_buf[cur_slice];
// 			int g_img_idx = img_block_idx + i;
// 			int orient_idx = tid % TParams::kOrientBlockSize;
// 			int g_orient_idx = orient_block_idx + orient_idx;
// 			if (g_img_idx >= image_size || g_orient_idx >= orientation_num) {
// 				assert(orient_idx < TParams::kOrientBlockSize);
// 				assert(i < TParams::kImgBlockSize);
// 				orient_real = 0.0;
// 				orient_imag = 0.0;
// 			} else {
// 				XFLOAT e0 = g_eulers[g_orient_idx * 9];
// 				XFLOAT e1 = g_eulers[g_orient_idx * 9 + 1];
// 				XFLOAT e3 = g_eulers[g_orient_idx * 9 + 3];
// 				XFLOAT e4 = g_eulers[g_orient_idx * 9 + 4];
// 				XFLOAT e6 = g_eulers[g_orient_idx * 9 + 6];
// 				XFLOAT e7 = g_eulers[g_orient_idx * 9 + 7];


// 				project3Dmodel_sp(coord_x[i], coord_y[i], e0, e1, e3, e4, e6, e7, orient_real, orient_imag, flag_minus_loc, 1U << (cur_slice % 32));

// 			}
// 		}
// 		flag_minus += flag_minus_loc;
// 	};

// 	//given current img_block_idx, corr_div_2, coord_x, coord_y, dump orient_real_buf and orient_imag_buf into s_orient_real_mat_block and s_orient_imag_mat_block
// 	auto dump_orient_mat_shm = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y, uint32_t& flag_minus)  {
// 		#pragma unroll
// 		for (int cur_slice = 0; cur_slice < kNumOrientSlice; cur_slice++) {
// 			int i = tid / TParams::kOrientBlockSize + cur_slice * kDimOrientSlice;
// 			XFLOAT& orient_real = orient_real_buf[cur_slice];
// 			XFLOAT& orient_imag = orient_imag_buf[cur_slice];
			
// 			bool flag_cur_minus = (flag_minus & (1U << (cur_slice % 32))) >> (cur_slice % 32);
// 			orient_imag = flag_cur_minus ? -orient_imag : orient_imag;

// 			int orient_idx = tid % TParams::kOrientBlockSize;
// 			s_orient_real_mat_block[orient_idx * TParams::kImgBlockSize + i] = orient_real;
// 			s_orient_imag_mat_block[orient_idx * TParams::kImgBlockSize + i] = orient_imag;

// 			XFLOAT magnitude_squared_sum = orient_real * orient_real * corr_div_2[i] + orient_imag * orient_imag * corr_div_2[i];
// 			s_orient_pow2_accumulator[tid] += magnitude_squared_sum;
// 		}
// 		flag_minus = 0;
// 	};
	
// /*=============================== FOR IMAGE BLOCK ==============================*/
// 	for (int img_iter = 0; img_iter < (image_size + TParams::kImgBlockSize - 1) / TParams::kImgBlockSize; img_iter++) {
// 	// for (int img_block_idx = 0; img_block_idx < image_size; img_block_idx += TParams::kImgBlockSize) {
// 		int img_block_idx = img_iter * TParams::kImgBlockSize;
// 		if (img_iter == 0) {
// 			load_coord_xy(img_block_idx, s_corr_div_2[img_iter % 2], s_coor_x[img_iter % 2], s_coor_y[img_iter % 2]);
// 			__syncthreads();
// 			// construct orient_mat
// 			load_orient_mat_buf(img_block_idx, s_corr_div_2[img_iter % 2], s_coor_x[img_iter % 2], s_coor_y[img_iter % 2], flag_minus_buf[img_iter % 2]);
// 		}
// 		if (img_iter + 1 < (image_size + TParams::kImgBlockSize - 1) / TParams::kImgBlockSize) {
// 			__syncthreads();
// 			// construct trans_mat
// 			load_trans_mat(img_block_idx, s_corr_div_2[img_iter % 2], s_coor_x[img_iter % 2], s_coor_y[img_iter % 2]);
// 			// construct orient_mat on this iteratiobn
// 			dump_orient_mat_shm(img_block_idx, s_corr_div_2[img_iter % 2], s_coor_x[img_iter % 2], s_coor_y[img_iter % 2], flag_minus_buf[img_iter % 2]);
// 			__syncthreads();
// 			//load coord_xy on next iteration
// 			load_coord_xy(img_block_idx + TParams::kImgBlockSize, s_corr_div_2[(img_iter + 1) % 2], s_coor_x[(img_iter + 1) % 2], s_coor_y[(img_iter + 1) % 2]);
// 			__syncthreads();
// 			// construct orient_mat for next iteration
// 			load_orient_mat_buf(img_block_idx + TParams::kImgBlockSize, s_corr_div_2[(img_iter + 1) % 2], s_coor_x[(img_iter + 1) % 2], s_coor_y[(img_iter + 1) % 2], flag_minus_buf[(img_iter + 1) % 2]);
// 		} else {
// 			__syncthreads();
// 			// construct trans_mat
// 			load_trans_mat(img_block_idx, s_corr_div_2[img_iter % 2], s_coor_x[img_iter % 2], s_coor_y[img_iter % 2]);
// 			// construct orient_mat on this iteratiobn
// 			dump_orient_mat_shm(img_block_idx, s_corr_div_2[img_iter % 2], s_coor_x[img_iter % 2], s_coor_y[img_iter % 2], flag_minus_buf[img_iter % 2]);
// 			__syncthreads();
// 		}
// 		// load_coord_xy(img_block_idx, s_corr_div_2[img_iter % 2], s_coor_x[img_iter % 2], s_coor_y[img_iter % 2]);
// 		// __syncthreads();

// /*=============================== CONSTRUCT TRANS_MAT & ORIENT_MAT ==============================*/
// 		// // construct trans_mat
// 		// load_trans_mat(img_block_idx, s_corr_div_2[img_iter % 2], s_coor_x[img_iter % 2], s_coor_y[img_iter % 2]);

// 		// // construct orient_mat
// 		// load_orient_mat_buf(img_block_idx, s_corr_div_2[img_iter % 2], s_coor_x[img_iter % 2], s_coor_y[img_iter % 2]);
// 		// dump_orient_mat_shm(img_block_idx, s_corr_div_2[img_iter % 2], s_coor_x[img_iter % 2], s_coor_y[img_iter % 2]);
// 		// __syncthreads();

// /*=============================== COMPUTE CROSS TERM ==============================*/
// 		for (int kk = 0; kk < TParams::kImgBlockSize; kk += TParams::kMmaImgTileSize) {
			
// 			// load slice A
// 			#pragma unroll
// 			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 				#pragma unroll
// 				for (int k = 0; k < kFragmentASize; ++k) {
// 					int s_m = fragment_a_m_idx_in_block<TParams>(warp_id, lane_id, i, k);
// 					int s_k = fragment_a_k_idx_in_block<TParams>(warp_id, lane_id, kk / TParams::kMmaImgTileSize, k);
// 					assert(s_m >= 0 && s_m < TParams::kTransBlockSize);
// 					assert(s_k >= 0 && s_k < TParams::kImgBlockSize);
// 					// s_A[s_m * TParams::kImgBlockSize + s_k] = fragment_a[i][k];
// 					fragment_a[i][k] = s_trans_real_mat_block[s_m * TParams::kImgBlockSize + s_k];
// 				}
// 			}

// 			// load slice B
// 			#pragma unroll
// 			for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 				#pragma unroll
// 				for (int k = 0; k < kFragmentBSize; ++k) {
// 					int s_n = fragment_b_n_idx_in_block<TParams>(warp_id, lane_id, j, k);
// 					int s_k = fragment_b_k_idx_in_block<TParams>(warp_id, lane_id, kk / TParams::kMmaImgTileSize, k);
// 					assert(s_n >= 0 && s_n < TParams::kOrientBlockSize);
// 					assert(s_k >= 0 && s_k < TParams::kImgBlockSize);
// 					// s_B[s_n * TParams::kImgBlockSize + s_k] = fragment_b[j][k];
// 					fragment_b[j][k] = s_orient_real_mat_block[s_n * TParams::kImgBlockSize + s_k];

// 				}
// 			}

// 			// compute an outproduct
// 			#pragma unroll
// 			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 				#pragma unroll
// 				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 					// mma_sync_aligned_m16n8k8_row_col_tf32(fragment_c[i][j], fragment_a[i], fragment_b[j]);
// 					mma_sync_aligned_m16n8k8_row_col_tf32_simulated_fp32(fragment_c[i][j], fragment_a[i], fragment_b[j]);
// 				}
// 			}
			
// 			// load slice A
// 			#pragma unroll
// 			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 				#pragma unroll
// 				for (int k = 0; k < kFragmentASize; ++k) {
// 					int s_m = fragment_a_m_idx_in_block<TParams>(warp_id, lane_id, i, k);
// 					int s_k = fragment_a_k_idx_in_block<TParams>(warp_id, lane_id, kk / TParams::kMmaImgTileSize, k);
// 					assert(s_m >= 0 && s_m < TParams::kTransBlockSize);
// 					assert(s_k >= 0 && s_k < TParams::kImgBlockSize);
// 					// s_A[s_m * TParams::kImgBlockSize + s_k] = fragment_a[i][k];
// 					fragment_a[i][k] = s_trans_imag_mat_block[s_m * TParams::kImgBlockSize + s_k];
// 				}
// 			}

// 			// load slice B
// 			#pragma unroll
// 			for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 				#pragma unroll
// 				for (int k = 0; k < kFragmentBSize; ++k) {
// 					int s_n = fragment_b_n_idx_in_block<TParams>(warp_id, lane_id, j, k);
// 					int s_k = fragment_b_k_idx_in_block<TParams>(warp_id, lane_id, kk / TParams::kMmaImgTileSize, k);
// 					assert(s_n >= 0 && s_n < TParams::kOrientBlockSize);
// 					assert(s_k >= 0 && s_k < TParams::kImgBlockSize);
// 					// s_B[s_n * TParams::kImgBlockSize + s_k] = fragment_b[j][k];
// 					fragment_b[j][k] = s_orient_imag_mat_block[s_n * TParams::kImgBlockSize + s_k];

// 				}
// 			}

// 			// compute an outproduct
// 			#pragma unroll
// 			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 				#pragma unroll
// 				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 					// mma_sync_aligned_m16n8k8_row_col_tf32(fragment_c[i][j], fragment_a[i], fragment_b[j]);
// 					mma_sync_aligned_m16n8k8_row_col_tf32_simulated_fp32(fragment_c[i][j], fragment_a[i], fragment_b[j]);
// 				}
// 			}

// 		} // end of cross term
// 	} // end of image block

// 	// reduce s_trans_pow2_accumulator
// 	for (int i = 1; i < TParams::kBlockSize / TParams::kTransBlockSize; ++i) {
// 		if (tid < TParams::kTransBlockSize) {
// 			s_trans_pow2_accumulator[tid] += s_trans_pow2_accumulator[i * TParams::kTransBlockSize + tid];
// 		}
// 	}

// 	// reduce s_orient_pow2_accumulator
// 	for (int i = 1; i < TParams::kBlockSize / TParams::kOrientBlockSize; ++i) {
// 		if (tid < TParams::kOrientBlockSize) {
// 			s_orient_pow2_accumulator[tid] += s_orient_pow2_accumulator[i * TParams::kOrientBlockSize + tid];
// 		}
// 	}

// /*=============================== REDUCE IN FRAGMENT_C ==============================*/
// 	__syncthreads();
	
// 	#pragma unroll
// 	for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 		#pragma unroll
// 		for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 			#pragma unroll
// 			for (int k = 0; k < kFragmentCSize; ++k) {
// 				int m = fragment_c_m_idx_in_block<TParams>(warp_id, lane_id, i, k);
// 				int n = fragment_c_n_idx_in_block<TParams>(warp_id, lane_id, j, k);
// 				fragment_c[i][j][k] += s_trans_pow2_accumulator[m] + s_orient_pow2_accumulator[n];
// 			}
// 		}
// 	}

// 	__syncthreads();

// /*=============================== WRITE BACK ==============================*/
// 	// write fragment_c back to g_diff2s_opt
// 	#pragma unroll
// 	for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 		#pragma unroll
// 		for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 			#pragma unroll
// 			for (int k = 0; k < kFragmentCSize; ++k) {
// 				int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
// 				int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);

// 				if (m < translation_num && n < orientation_num) {
// 					g_diff2s_opt[n * translation_num + m] = fragment_c[i][j][k];
// 				}
// 			}
// 		}
// 	}

// }




// // 使用ldsm指令完成矩阵运算，但是trans matrix与orientation matrix还有bug
// template<typename TParams>
// __launch_bounds__(128, 2)
// __global__ void cuda_kernel_coarse_matrixV5(
// 	XFLOAT *g_eulers,
// 	XFLOAT *trans_x,
// 	XFLOAT *trans_y,
// 	XFLOAT *g_real,
// 	XFLOAT *g_imag,
// 	AccProjectorKernel projector,
// 	XFLOAT *g_corr,
// 	XFLOAT *g_diff2s,
// 	XFLOAT *g_diff2s_opt,
// 	int translation_num,
// 	int orientation_num,
// 	int image_size,
// 	XFLOAT *g_trans_real_m,
// 	XFLOAT *g_trans_imag_m,
// 	XFLOAT *g_orient_real_m,
// 	XFLOAT *g_orient_imag_m) {
// 	static_assert(TParams::kBlockSize % 32 == 0, "kBlockSize must be a multiple of 32");
// 	static_assert(TParams::kImgBlockSize == TParams::kWarpImgTileSize, "kImgBlockSize must be equal to kWarpImgTileSize");
// 	static_assert(TParams::kBlockSize >= TParams::kTransBlockSize, "kBlockSize must be greater than or equal to kTransBlockSize");
// 	static_assert(TParams::kBlockSize >= TParams::kOrientBlockSize, "kBlockSize must be greater than or equal to kOrientBlockSize");

// 	static_assert(TParams::kTransBlockSize % TParams::kWarpTransTileSize == 0, "kTransBlockSize must be a multiple of kWarpTransTileSize");
// 	static_assert(TParams::kOrientBlockSize % TParams::kWarpOrientTileSize == 0, "kOrientBlockSize must be a multiple of kWarpOrientTileSize");
// 	static_assert(TParams::kTransBlockSize % TParams::kMmaTransTileSize == 0, "kTransBlockSize must be a multiple of kMmaTransTileSize");
// 	static_assert(TParams::kOrientBlockSize % TParams::kMmaOrientTileSize == 0, "kOrientBlockSize must be a multiple of kMmaOrientTileSize");
// 	static_assert(TParams::kBlockSize / 32 == (TParams::kTransBlockSize / TParams::kWarpTransTileSize) * (TParams::kOrientBlockSize / TParams::kWarpOrientTileSize), "kBlockSize must be equal to the product of the number of warps in translation, orientation and image dimension");

// 	static_assert(TParams::kImgBlockSize == 16, "kImgBlockSize must be 16");

// 	const int tid = threadIdx.x;          // thread id in a block
// 	const int bid = blockIdx.x;           // block id in a grid
// 	const int warp_id  = tid / 32;        // warp id in a block
// 	const int warp_num = TParams::kBlockSize / 32; // number of warps in a block
// 	const int lane_id  = tid % 32;        // thread id in a warp

// 	const int trans_block_num = (translation_num + TParams::kTransBlockSize - 1) / TParams::kTransBlockSize;
// 	const int orient_block_num = (orientation_num + TParams::kOrientBlockSize - 1) / TParams::kOrientBlockSize;

// 	// bug:这里一开始没有括号
// 	const int trans_block_idx = (bid % trans_block_num) * TParams::kTransBlockSize;
// 	const int orient_block_idx = (bid / trans_block_num) * TParams::kOrientBlockSize;

// 	assert(blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0);
	
// 	// 'img' data is stored contiguously.
// 	// __shared__ XFLOAT s_trans_real_mat_block[TParams::kTransBlockSize * TParams::kImgBlockSize];
// 	// __shared__ XFLOAT s_trans_imag_mat_block[TParams::kTransBlockSize * TParams::kImgBlockSize];
// 	// __shared__ XFLOAT s_orient_real_mat_block[TParams::kOrientBlockSize * TParams::kImgBlockSize];
// 	// __shared__ XFLOAT s_orient_imag_mat_block[TParams::kOrientBlockSize * TParams::kImgBlockSize];

// 	__shared__ XFLOAT s_trans_mat_block[2 * TParams::kTransBlockSize * TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_orient_mat_block[2 * TParams::kOrientBlockSize * TParams::kImgBlockSize];
	
// 	SharedMemorySwizzle<float, TParams::kTransBlockSize, 2 * TParams::kImgBlockSize, 0> s_trans_mat_block_swizzle(s_trans_mat_block);
// 	SharedMemorySwizzle<float, TParams::kTransBlockSize, TParams::kImgBlockSize, 0> s_trans_real_mat_block_swizzle(s_trans_mat_block);
// 	SharedMemorySwizzle<float, TParams::kTransBlockSize, TParams::kImgBlockSize, TParams::kImgBlockSize> s_trans_imag_mat_block_swizzle(s_trans_mat_block);

// 	SharedMemorySwizzle<float, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize, 0> s_orient_mat_block_swizzle(s_orient_mat_block);
// 	SharedMemorySwizzle<float, TParams::kOrientBlockSize, TParams::kImgBlockSize, 0> s_orient_real_mat_block_swizzle(s_orient_mat_block);
// 	SharedMemorySwizzle<float, TParams::kOrientBlockSize, TParams::kImgBlockSize, TParams::kImgBlockSize> s_orient_imag_mat_block_swizzle(s_orient_mat_block);

// 	__shared__ XFLOAT s_corr_div_2[2][TParams::kImgBlockSize];
// 	//double buffer for s_coor_x, s_coor_y
// 	__shared__ XFLOAT s_coor_x[2][TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_coor_y[2][TParams::kImgBlockSize];

// 	// reduce buffer
// 	__shared__ XFLOAT s_trans_pow2_accumulator[(TParams::kBlockSize / TParams::kTransBlockSize) * TParams::kTransBlockSize];
// 	__shared__ XFLOAT s_orient_pow2_accumulator[(TParams::kBlockSize / TParams::kOrientBlockSize) * TParams::kOrientBlockSize];

// 	// register
// 	constexpr int kNumMmaTransInWarpTile = TParams::kWarpTransTileSize / TParams::kMmaTransTileSize;
// 	constexpr int kNumMmaOrientInWarpTile = TParams::kWarpOrientTileSize / TParams::kMmaOrientTileSize;
// 	constexpr int kNumMmaImgInWarpTile = TParams::kWarpImgTileSize / TParams::kMmaImgTileSize;

// 	constexpr int kFragmentASize = TParams::kMmaTransTileSize * TParams::kMmaImgTileSize / kWarpSize;
// 	constexpr int kFragmentBSize = TParams::kMmaOrientTileSize * TParams::kMmaImgTileSize / kWarpSize;
// 	constexpr int kFragmentCSize = TParams::kMmaTransTileSize * TParams::kMmaOrientTileSize / kWarpSize;

// 	// XFLOAT fragment_a[kNumMmaTransInWarpTile][kFragmentASize];
// 	// XFLOAT fragment_b[kNumMmaOrientInWarpTile][kFragmentBSize];
// 	XFLOAT fragment_c[kNumMmaTransInWarpTile][kNumMmaOrientInWarpTile][kFragmentCSize];

// 	constexpr int kRegistersMmaPerThread = kNumMmaTransInWarpTile * kFragmentASize +
// 										   kNumMmaOrientInWarpTile * kFragmentBSize +
// 										   kNumMmaTransInWarpTile * kNumMmaOrientInWarpTile * kFragmentCSize;
	
// 	#if kRegistersMmaPerThread >= 256
// 	#warning "kRegistersMmaPerThread must be less than or equal to 256, otherwise register spilling will occur"
// 	#endif

// 	// initialize shared memory to zero
// 	for (int i = tid; i < 2 * TParams::kTransBlockSize * TParams::kImgBlockSize; i += TParams::kBlockSize) {
// 		s_trans_mat_block[i] = 0.0;
// 	}
// 	for (int i = tid; i < 2 * TParams::kOrientBlockSize * TParams::kImgBlockSize; i += TParams::kBlockSize) {
// 		s_orient_mat_block[i] = 0.0;
// 	}
// 	// bug: 这里初始化有问题
// 	// for (int i = tid; i < TParams::kTransBlockSize; i += TParams::kBlockSize) {
// 	// 	s_trans_pow2_accumulator[i] = 0.0;
// 	// 	s_orient_pow2_accumulator[i] = 0.0;
// 	// }
// 	s_trans_pow2_accumulator[tid] = 0.0;
// 	s_orient_pow2_accumulator[tid] = 0.0;

// 	for (int i = tid; i < TParams::kImgBlockSize; i += TParams::kBlockSize) {
// 		s_corr_div_2[0][i] = 0.0;
// 		s_corr_div_2[1][i] = 0.0;
// 		s_coor_x[0][i] = 0.0;
// 		s_coor_y[0][i] = 0.0;
// 		s_coor_x[1][i] = 0.0;
// 		s_coor_y[1][i] = 0.0;
// 	}
	
// 	// read fragment_c from g_diff2s
// 	#pragma unroll
// 	for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 		#pragma unroll
// 		for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 			#pragma unroll
// 			for (int k = 0; k < kFragmentCSize; ++k) {
// 				int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
// 				int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);
// 				if (m < translation_num && n < orientation_num) {
// 					fragment_c[i][j][k] = g_diff2s[n * translation_num + m];
// 				} else {
// 					fragment_c[i][j][k] = 0.0;
// 				}
// 			}
// 		}
// 	}

// 	__syncthreads();
// 	//given current img_block_idx, load global array into corr_div_2, coord_x, coord_y
// 	auto load_coord_xy = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y) {
// 		#pragma unroll 
// 		for (int i = tid; i < TParams::kImgBlockSize; i += TParams::kBlockSize) {
// 			if (img_block_idx + i < image_size) {
// 				corr_div_2[i] = g_corr[img_block_idx + i] / 2;
// 			} else {
// 				corr_div_2[i] = 0.;
// 			}

// 			int x, y;
// 			pixel_index2coor(img_block_idx + i, projector.imgX, projector.imgY, projector.maxR, x, y);
// 			coord_x[i] = x;
// 			coord_y[i] = y;
// 		}
// 	};

// 	//given current img_block_idx, corr_div_2, coord_x, coord_y, load trans mat into s_trans_real_mat_block and s_trans_imag_mat_block
// 	auto load_trans_mat = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y) {
// 		assert(TParams::kBlockSize % TParams::kTransBlockSize == 0);
// 		#pragma unroll
// 		for (int i = tid / TParams::kTransBlockSize; i < TParams::kImgBlockSize; i += TParams::kBlockSize / TParams::kTransBlockSize) {
// 			int g_img_idx = img_block_idx + i;
// 			int trans_idx = tid % TParams::kTransBlockSize;
// 			if (g_img_idx >= image_size) {
// 				assert(trans_idx < TParams::kTransBlockSize);
// 				assert(i < TParams::kImgBlockSize);
// 				// s_trans_real_mat_block[trans_idx * TParams::kImgBlockSize + i] = 0.;
// 				// s_trans_imag_mat_block[trans_idx * TParams::kImgBlockSize + i] = 0.;
// 				s_trans_real_mat_block_swizzle(trans_idx, i) = 0.;
// 				s_trans_imag_mat_block_swizzle(trans_idx, i) = 0.;
// 				continue;
// 			}
// 			int g_trans_idx = trans_block_idx + trans_idx;
// 			if (g_trans_idx >= translation_num) {
// 				continue;
// 			}
// 			XFLOAT tx = trans_x[g_trans_idx];
// 			XFLOAT ty = trans_y[g_trans_idx];
// 			XFLOAT real = g_real[g_img_idx];
// 			XFLOAT imag = g_imag[g_img_idx];

// 			int x = coord_x[i];
// 			int y = coord_y[i];
// 			XFLOAT trans_real, trans_imag;
// 			translatePixel(x, y, tx, ty, real, imag, trans_real, trans_imag);

// 			// s_trans_real_mat_block[trans_idx * TParams::kImgBlockSize + i] = -2 * trans_real * corr_div_2[i];
// 			// s_trans_imag_mat_block[trans_idx * TParams::kImgBlockSize + i] = -2 * trans_imag * corr_div_2[i];
// 			s_trans_real_mat_block_swizzle(trans_idx, i) = -2 * trans_real * corr_div_2[i];
// 			s_trans_imag_mat_block_swizzle(trans_idx, i) = -2 * trans_imag * corr_div_2[i];

// 			XFLOAT magnitude_squared_sum = trans_real * trans_real * corr_div_2[i] + trans_imag * trans_imag * corr_div_2[i];
// 			s_trans_pow2_accumulator[tid] += magnitude_squared_sum;
// 		}
// 	};

// 	auto project3Dmodel_sp = [&](
// 			XFLOAT x,
// 			XFLOAT y,
// 			XFLOAT e0,
// 			XFLOAT e1,
// 			XFLOAT e3,
// 			XFLOAT e4,
// 			XFLOAT e6,
// 			XFLOAT e7,
// 			XFLOAT &real,
// 			XFLOAT &imag,
// 			uint32_t& flag_minus,
// 			uint32_t mask) {
// 		XFLOAT xp = (e0 * x + e1 * y ) * projector.padding_factor;
// 		XFLOAT yp = (e3 * x + e4 * y ) * projector.padding_factor;
// 		XFLOAT zp = (e6 * x + e7 * y ) * projector.padding_factor;
// 		int r2 = xp*xp + yp*yp + zp*zp;
// 		if (r2 <= projector.maxR2_padded)
// 		{
// 			bool xp_neg = xp < 0;
// 			flag_minus += xp_neg ? mask : 0;
// 			// NOTICE: if xp_neg, imag = -imag
// 			if (xp_neg) {
// 				// Get complex conjugated hermitian symmetry pair
// 				xp = -xp;
// 				yp = -yp;
// 				zp = -zp;
// 				yp -= projector.mdlInitY;
// 				zp -= projector.mdlInitZ;
// 			}
// 			else {
// 				yp -= projector.mdlInitY;
// 				zp -= projector.mdlInitZ;
// 			}
// 			real =    tex3D<XFLOAT>(projector.mdlReal, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
// 			imag =    tex3D<XFLOAT>(projector.mdlImag, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
			
// 			// if(xp_neg) {
// 			// 	imag = -imag;
// 			// }
// 		}
// 		else {
// 			real = (XFLOAT)0;
// 			imag = (XFLOAT)0;
// 		}
// 	};

// 	constexpr int kDimOrientSlice = (TParams::kBlockSize / TParams::kOrientBlockSize);
// 	constexpr int kNumOrientSlice = (TParams::kImgBlockSize + kDimOrientSlice - 1) / kDimOrientSlice;
// 	assert(kNumOrientSlice <=32);
// 	XFLOAT orient_real_buf[kNumOrientSlice], orient_imag_buf[kNumOrientSlice];
// 	uint32_t flag_minus_buf[2] = {0, 0};

// 	//given current img_block_idx, corr_div_2, coord_x, coord_y, load orient mat into orient_real_buf and orient_imag_buf
// 	auto load_orient_mat_buf = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y, uint32_t& flag_minus)  {
// 		uint32_t flag_minus_loc = 0;

// 		#pragma unroll
// 		for (int cur_slice = 0; cur_slice < kNumOrientSlice; cur_slice++) {
// 			int i = tid / TParams::kOrientBlockSize + cur_slice * kDimOrientSlice;
// 			XFLOAT& orient_real = orient_real_buf[cur_slice];
// 			XFLOAT& orient_imag = orient_imag_buf[cur_slice];
// 			int g_img_idx = img_block_idx + i;
// 			int orient_idx = tid % TParams::kOrientBlockSize;
// 			int g_orient_idx = orient_block_idx + orient_idx;
// 			if (g_img_idx >= image_size || g_orient_idx >= orientation_num) {
// 				assert(orient_idx < TParams::kOrientBlockSize);
// 				assert(i < TParams::kImgBlockSize);
// 				orient_real = 0.0;
// 				orient_imag = 0.0;
// 			} else {
// 				XFLOAT e0 = g_eulers[g_orient_idx * 9];
// 				XFLOAT e1 = g_eulers[g_orient_idx * 9 + 1];
// 				XFLOAT e3 = g_eulers[g_orient_idx * 9 + 3];
// 				XFLOAT e4 = g_eulers[g_orient_idx * 9 + 4];
// 				XFLOAT e6 = g_eulers[g_orient_idx * 9 + 6];
// 				XFLOAT e7 = g_eulers[g_orient_idx * 9 + 7];


// 				project3Dmodel_sp(coord_x[i], coord_y[i], e0, e1, e3, e4, e6, e7, orient_real, orient_imag, flag_minus_loc, 1U << (cur_slice % 32));

// 			}
// 		}
// 		flag_minus += flag_minus_loc;
// 	};

// 	//given current img_block_idx, corr_div_2, coord_x, coord_y, dump orient_real_buf and orient_imag_buf into s_orient_real_mat_block and s_orient_imag_mat_block
// 	auto dump_orient_mat_shm = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y, uint32_t& flag_minus)  {
// 		#pragma unroll
// 		for (int cur_slice = 0; cur_slice < kNumOrientSlice; cur_slice++) {
// 			int i = tid / TParams::kOrientBlockSize + cur_slice * kDimOrientSlice;
// 			XFLOAT& orient_real = orient_real_buf[cur_slice];
// 			XFLOAT& orient_imag = orient_imag_buf[cur_slice];
			
// 			bool flag_cur_minus = (flag_minus & (1U << (cur_slice % 32))) >> (cur_slice % 32);
// 			orient_imag = flag_cur_minus ? -orient_imag : orient_imag;

// 			int orient_idx = tid % TParams::kOrientBlockSize;
// 			// s_orient_real_mat_block[orient_idx * TParams::kImgBlockSize + i] = orient_real;
// 			// s_orient_imag_mat_block[orient_idx * TParams::kImgBlockSize + i] = orient_imag;

// 			s_orient_real_mat_block_swizzle(orient_idx, i) = orient_real;
// 			s_orient_imag_mat_block_swizzle(orient_idx, i) = orient_imag;

// 			XFLOAT magnitude_squared_sum = orient_real * orient_real * corr_div_2[i] + orient_imag * orient_imag * corr_div_2[i];
// 			s_orient_pow2_accumulator[tid] += magnitude_squared_sum;
// 		}
// 		flag_minus = 0;
// 	};
	
// /*=============================== FOR IMAGE BLOCK ==============================*/
// 	for (int img_iter = 0; img_iter < (image_size + TParams::kImgBlockSize - 1) / TParams::kImgBlockSize; img_iter++) {
// 	// for (int img_block_idx = 0; img_block_idx < image_size; img_block_idx += TParams::kImgBlockSize) {
// 		int img_block_idx = img_iter * TParams::kImgBlockSize;
// 		if (img_iter == 0) {
// 			load_coord_xy(img_block_idx, s_corr_div_2[img_iter % 2], s_coor_x[img_iter % 2], s_coor_y[img_iter % 2]);
// 			__syncthreads();
// 			// construct orient_mat
// 			load_orient_mat_buf(img_block_idx, s_corr_div_2[img_iter % 2], s_coor_x[img_iter % 2], s_coor_y[img_iter % 2], flag_minus_buf[img_iter % 2]);
// 		}
// 		if (img_iter + 1 < (image_size + TParams::kImgBlockSize - 1) / TParams::kImgBlockSize) {
// 			__syncthreads();
// 			// construct trans_mat
// 			load_trans_mat(img_block_idx, s_corr_div_2[img_iter % 2], s_coor_x[img_iter % 2], s_coor_y[img_iter % 2]);
// 			// construct orient_mat on this iteratiobn
// 			dump_orient_mat_shm(img_block_idx, s_corr_div_2[img_iter % 2], s_coor_x[img_iter % 2], s_coor_y[img_iter % 2], flag_minus_buf[img_iter % 2]);
// 			__syncthreads();
// 			//load coord_xy on next iteration
// 			load_coord_xy(img_block_idx + TParams::kImgBlockSize, s_corr_div_2[(img_iter + 1) % 2], s_coor_x[(img_iter + 1) % 2], s_coor_y[(img_iter + 1) % 2]);
// 			__syncthreads();
// 			// construct orient_mat for next iteration
// 			load_orient_mat_buf(img_block_idx + TParams::kImgBlockSize, s_corr_div_2[(img_iter + 1) % 2], s_coor_x[(img_iter + 1) % 2], s_coor_y[(img_iter + 1) % 2], flag_minus_buf[(img_iter + 1) % 2]);
// 		} else {
// 			__syncthreads();
// 			// construct trans_mat
// 			load_trans_mat(img_block_idx, s_corr_div_2[img_iter % 2], s_coor_x[img_iter % 2], s_coor_y[img_iter % 2]);
// 			// construct orient_mat on this iteratiobn
// 			dump_orient_mat_shm(img_block_idx, s_corr_div_2[img_iter % 2], s_coor_x[img_iter % 2], s_coor_y[img_iter % 2], flag_minus_buf[img_iter % 2]);
// 			__syncthreads();
// 		}
// 		// load_coord_xy(img_block_idx, s_corr_div_2[img_iter % 2], s_coor_x[img_iter % 2], s_coor_y[img_iter % 2]);
// 		// __syncthreads();

// /*=============================== CONSTRUCT TRANS_MAT & ORIENT_MAT ==============================*/
// 		// // construct trans_mat
// 		// load_trans_mat(img_block_idx, s_corr_div_2[img_iter % 2], s_coor_x[img_iter % 2], s_coor_y[img_iter % 2]);

// 		// // construct orient_mat
// 		// load_orient_mat_buf(img_block_idx, s_corr_div_2[img_iter % 2], s_coor_x[img_iter % 2], s_coor_y[img_iter % 2]);
// 		// dump_orient_mat_shm(img_block_idx, s_corr_div_2[img_iter % 2], s_coor_x[img_iter % 2], s_coor_y[img_iter % 2]);
// 		// __syncthreads();

// /*=============================== COMPUTE CROSS TERM ==============================*/

// 		block_mma_tf32_sim_fp32<decltype(s_trans_mat_block_swizzle), decltype(s_orient_mat_block_swizzle), 
// 		  TParams::kTransBlockSize, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize,
// 		  TParams::kWarpTransTileSize, TParams::kWarpOrientTileSize, 2 * TParams::kWarpImgTileSize,
// 		  TParams::kMmaTransTileSize, TParams::kMmaOrientTileSize, TParams::kMmaImgTileSize>(
// 			fragment_c, s_trans_mat_block_swizzle, s_orient_mat_block_swizzle, warp_id, lane_id);

// 		// } // end of cross term
// 	} // end of image block

// 	// reduce s_trans_pow2_accumulator
// 	for (int i = 1; i < TParams::kBlockSize / TParams::kTransBlockSize; ++i) {
// 		if (tid < TParams::kTransBlockSize) {
// 			s_trans_pow2_accumulator[tid] += s_trans_pow2_accumulator[i * TParams::kTransBlockSize + tid];
// 		}
// 	}

// 	// reduce s_orient_pow2_accumulator
// 	for (int i = 1; i < TParams::kBlockSize / TParams::kOrientBlockSize; ++i) {
// 		if (tid < TParams::kOrientBlockSize) {
// 			s_orient_pow2_accumulator[tid] += s_orient_pow2_accumulator[i * TParams::kOrientBlockSize + tid];
// 		}
// 	}

// /*=============================== REDUCE IN FRAGMENT_C ==============================*/
// 	__syncthreads();
	
// 	#pragma unroll
// 	for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 		#pragma unroll
// 		for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 			#pragma unroll
// 			for (int k = 0; k < kFragmentCSize; ++k) {
// 				int m = fragment_c_m_idx_in_block<TParams>(warp_id, lane_id, i, k);
// 				int n = fragment_c_n_idx_in_block<TParams>(warp_id, lane_id, j, k);
// 				fragment_c[i][j][k] += s_trans_pow2_accumulator[m] + s_orient_pow2_accumulator[n];
// 			}
// 		}
// 	}

// 	__syncthreads();

// /*=============================== WRITE BACK ==============================*/
// 	// write fragment_c back to g_diff2s_opt
// 	#pragma unroll
// 	for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 		#pragma unroll
// 		for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 			#pragma unroll
// 			for (int k = 0; k < kFragmentCSize; ++k) {
// 				int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
// 				int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);

// 				if (m < translation_num && n < orientation_num) {
// 					g_diff2s_opt[n * translation_num + m] = fragment_c[i][j][k];
// 				}
// 			}
// 		}
// 	}

// }



// // 完成split K
// template<typename TParams>
// __launch_bounds__(128, 2)
// __global__ void cuda_kernel_coarse_matrixV6(
// 	XFLOAT *g_eulers,
// 	XFLOAT *trans_x,
// 	XFLOAT *trans_y,
// 	XFLOAT *g_real,
// 	XFLOAT *g_imag,
// 	AccProjectorKernel projector,
// 	XFLOAT *g_corr,
// 	XFLOAT *g_diff2s,
// 	XFLOAT *g_diff2s_opt,
// 	int translation_num,
// 	int orientation_num,
// 	int image_size,
// 	XFLOAT *g_trans_real_m,
// 	XFLOAT *g_trans_imag_m,
// 	XFLOAT *g_orient_real_m,
// 	XFLOAT *g_orient_imag_m) {
// 	static_assert(TParams::kBlockSize % 32 == 0, "kBlockSize must be a multiple of 32");
// 	static_assert(TParams::kImgBlockSize == TParams::kWarpImgTileSize, "kImgBlockSize must be equal to kWarpImgTileSize");
// 	static_assert(TParams::kBlockSize >= TParams::kTransBlockSize, "kBlockSize must be greater than or equal to kTransBlockSize");
// 	static_assert(TParams::kBlockSize >= TParams::kOrientBlockSize, "kBlockSize must be greater than or equal to kOrientBlockSize");

// 	static_assert(TParams::kTransBlockSize % TParams::kWarpTransTileSize == 0, "kTransBlockSize must be a multiple of kWarpTransTileSize");
// 	static_assert(TParams::kOrientBlockSize % TParams::kWarpOrientTileSize == 0, "kOrientBlockSize must be a multiple of kWarpOrientTileSize");
// 	static_assert(TParams::kTransBlockSize % TParams::kMmaTransTileSize == 0, "kTransBlockSize must be a multiple of kMmaTransTileSize");
// 	static_assert(TParams::kOrientBlockSize % TParams::kMmaOrientTileSize == 0, "kOrientBlockSize must be a multiple of kMmaOrientTileSize");
// 	static_assert(TParams::kBlockSize / 32 == (TParams::kTransBlockSize / TParams::kWarpTransTileSize) * (TParams::kOrientBlockSize / TParams::kWarpOrientTileSize), "kBlockSize must be equal to the product of the number of warps in translation, orientation and image dimension");

// 	static_assert(TParams::kImgBlockSize == 16, "kImgBlockSize must be 16");

// 	const int tid = threadIdx.x;          // thread id in a block
// 	const int bid = blockIdx.x;           // block id in a grid
// 	const int warp_id  = tid / 32;        // warp id in a block
// 	const int warp_num = TParams::kBlockSize / 32; // number of warps in a block
// 	const int lane_id  = tid % 32;        // thread id in a warp

// 	const int trans_block_num = (translation_num + TParams::kTransBlockSize - 1) / TParams::kTransBlockSize;
// 	const int orient_block_num = (orientation_num + TParams::kOrientBlockSize - 1) / TParams::kOrientBlockSize;

// 	// bug:这里一开始没有括号
// 	int trans_block_idx = (bid % trans_block_num) * TParams::kTransBlockSize;
// 	int orient_block_idx = (bid / trans_block_num) * TParams::kOrientBlockSize;
//     // int trans_block_idx = 0; // forware declaration
//     // int orient_block_idx = 0;

// 	assert(blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0);
	
// 	CoarseScheduler<TParams::kTransBlockSize, 
// 					TParams::kOrientBlockSize, 
// 					TParams::kImgBlockSize, 
// 					CoarseSchedulerStrategy::SplitK>
// 		scheduler(translation_num, orientation_num, image_size);
	

// 	// 'img' data is stored contiguously.
// 	// __shared__ XFLOAT s_trans_real_mat_block[TParams::kTransBlockSize * TParams::kImgBlockSize];
// 	// __shared__ XFLOAT s_trans_imag_mat_block[TParams::kTransBlockSize * TParams::kImgBlockSize];
// 	// __shared__ XFLOAT s_orient_real_mat_block[TParams::kOrientBlockSize * TParams::kImgBlockSize];
// 	// __shared__ XFLOAT s_orient_imag_mat_block[TParams::kOrientBlockSize * TParams::kImgBlockSize];

// 	__shared__ XFLOAT s_trans_mat_block[2 * TParams::kTransBlockSize * TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_orient_mat_block[2 * TParams::kOrientBlockSize * TParams::kImgBlockSize];
	
// 	SharedMemorySwizzle<float, TParams::kTransBlockSize, 2 * TParams::kImgBlockSize, 0> s_trans_mat_block_swizzle(s_trans_mat_block);
// 	SharedMemorySwizzle<float, TParams::kTransBlockSize, TParams::kImgBlockSize, 0> s_trans_real_mat_block_swizzle(s_trans_mat_block);
// 	SharedMemorySwizzle<float, TParams::kTransBlockSize, TParams::kImgBlockSize, TParams::kImgBlockSize> s_trans_imag_mat_block_swizzle(s_trans_mat_block);

// 	SharedMemorySwizzle<float, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize, 0> s_orient_mat_block_swizzle(s_orient_mat_block);
// 	SharedMemorySwizzle<float, TParams::kOrientBlockSize, TParams::kImgBlockSize, 0> s_orient_real_mat_block_swizzle(s_orient_mat_block);
// 	SharedMemorySwizzle<float, TParams::kOrientBlockSize, TParams::kImgBlockSize, TParams::kImgBlockSize> s_orient_imag_mat_block_swizzle(s_orient_mat_block);

// 	// double buffer for s_corr_div_2, s_coor_x, s_coor_y
// 	__shared__ XFLOAT s_corr_div_2[2][TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_coor_x[2][TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_coor_y[2][TParams::kImgBlockSize];

// 	// reduce buffer
// 	__shared__ XFLOAT s_trans_pow2_accumulator[(TParams::kBlockSize / TParams::kTransBlockSize) * TParams::kTransBlockSize];
// 	__shared__ XFLOAT s_orient_pow2_accumulator[(TParams::kBlockSize / TParams::kOrientBlockSize) * TParams::kOrientBlockSize];

// 	// register
// 	constexpr int kNumMmaTransInWarpTile = TParams::kWarpTransTileSize / TParams::kMmaTransTileSize;
// 	constexpr int kNumMmaOrientInWarpTile = TParams::kWarpOrientTileSize / TParams::kMmaOrientTileSize;
// 	constexpr int kNumMmaImgInWarpTile = TParams::kWarpImgTileSize / TParams::kMmaImgTileSize;

// 	constexpr int kFragmentASize = TParams::kMmaTransTileSize * TParams::kMmaImgTileSize / kWarpSize;
// 	constexpr int kFragmentBSize = TParams::kMmaOrientTileSize * TParams::kMmaImgTileSize / kWarpSize;
// 	constexpr int kFragmentCSize = TParams::kMmaTransTileSize * TParams::kMmaOrientTileSize / kWarpSize;

// 	// XFLOAT fragment_a[kNumMmaTransInWarpTile][kFragmentASize];
// 	// XFLOAT fragment_b[kNumMmaOrientInWarpTile][kFragmentBSize];
// 	XFLOAT fragment_c[kNumMmaTransInWarpTile][kNumMmaOrientInWarpTile][kFragmentCSize];

// 	constexpr int kRegistersMmaPerThread = kNumMmaTransInWarpTile * kFragmentASize +
// 										   kNumMmaOrientInWarpTile * kFragmentBSize +
// 										   kNumMmaTransInWarpTile * kNumMmaOrientInWarpTile * kFragmentCSize;
	
// 	#if kRegistersMmaPerThread >= 256
// 	#warning "kRegistersMmaPerThread must be less than or equal to 256, otherwise register spilling will occur"
// 	#endif


// 	// ============================= lambda function =============================
// 	//given current img_block_idx, load global array into corr_div_2, coord_x, coord_y
// 	auto load_coord_xy = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y) {
// 		#pragma unroll 
// 		for (int i = tid; i < TParams::kImgBlockSize; i += TParams::kBlockSize) {
// 			if (img_block_idx + i < image_size) {
// 				corr_div_2[i] = g_corr[img_block_idx + i] / 2;
// 			} else {
// 				corr_div_2[i] = 0.;
// 			}

// 			int x, y;
// 			pixel_index2coor(img_block_idx + i, projector.imgX, projector.imgY, projector.maxR, x, y);
// 			coord_x[i] = x;
// 			coord_y[i] = y;
// 		}
// 	};

// 	//given current img_block_idx, corr_div_2, coord_x, coord_y, load trans mat into s_trans_real_mat_block and s_trans_imag_mat_block
// 	auto load_trans_mat = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y) {
// 		assert(TParams::kBlockSize % TParams::kTransBlockSize == 0);
// 		#pragma unroll
// 		for (int i = tid / TParams::kTransBlockSize; i < TParams::kImgBlockSize; i += TParams::kBlockSize / TParams::kTransBlockSize) {
// 			int g_img_idx = img_block_idx + i;
// 			int trans_idx = tid % TParams::kTransBlockSize;
// 			if (g_img_idx >= image_size) {
// 				assert(trans_idx < TParams::kTransBlockSize);
// 				assert(i < TParams::kImgBlockSize);
// 				s_trans_real_mat_block_swizzle(trans_idx, i) = 0.;
// 				s_trans_imag_mat_block_swizzle(trans_idx, i) = 0.;
// 				continue;
// 			}
// 			int g_trans_idx = trans_block_idx + trans_idx;
// 			if (g_trans_idx >= translation_num) {
// 				continue;
// 			}
// 			XFLOAT tx = trans_x[g_trans_idx];
// 			XFLOAT ty = trans_y[g_trans_idx];
// 			XFLOAT real = g_real[g_img_idx];
// 			XFLOAT imag = g_imag[g_img_idx];

// 			int x = coord_x[i];
// 			int y = coord_y[i];
// 			XFLOAT trans_real, trans_imag;
// 			translatePixel(x, y, tx, ty, real, imag, trans_real, trans_imag);

// 			// s_trans_real_mat_block[trans_idx * TParams::kImgBlockSize + i] = -2 * trans_real * corr_div_2[i];
// 			// s_trans_imag_mat_block[trans_idx * TParams::kImgBlockSize + i] = -2 * trans_imag * corr_div_2[i];
// 			s_trans_real_mat_block_swizzle(trans_idx, i) = -2 * trans_real * corr_div_2[i];
// 			s_trans_imag_mat_block_swizzle(trans_idx, i) = -2 * trans_imag * corr_div_2[i];

// 			XFLOAT magnitude_squared_sum = trans_real * trans_real * corr_div_2[i] + trans_imag * trans_imag * corr_div_2[i];
// 			s_trans_pow2_accumulator[tid] += magnitude_squared_sum;
// 		}
// 	};

// 	auto project3Dmodel_sp = [&](
// 			XFLOAT x,
// 			XFLOAT y,
// 			XFLOAT e0,
// 			XFLOAT e1,
// 			XFLOAT e3,
// 			XFLOAT e4,
// 			XFLOAT e6,
// 			XFLOAT e7,
// 			XFLOAT &real,
// 			XFLOAT &imag,
// 			uint32_t& flag_minus,
// 			uint32_t mask) {
// 		XFLOAT xp = (e0 * x + e1 * y ) * projector.padding_factor;
// 		XFLOAT yp = (e3 * x + e4 * y ) * projector.padding_factor;
// 		XFLOAT zp = (e6 * x + e7 * y ) * projector.padding_factor;
// 		int r2 = xp*xp + yp*yp + zp*zp;
// 		if (r2 <= projector.maxR2_padded)
// 		{
// 			bool xp_neg = xp < 0;
// 			flag_minus += xp_neg ? mask : 0;
// 			// NOTICE: if xp_neg, imag = -imag
// 			if (xp_neg) {
// 				// Get complex conjugated hermitian symmetry pair
// 				xp = -xp;
// 				yp = -yp;
// 				zp = -zp;
// 				yp -= projector.mdlInitY;
// 				zp -= projector.mdlInitZ;
// 			}
// 			else {
// 				yp -= projector.mdlInitY;
// 				zp -= projector.mdlInitZ;
// 			}
// 			real =    tex3D<XFLOAT>(projector.mdlReal, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
// 			imag =    tex3D<XFLOAT>(projector.mdlImag, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
			
// 			// if(xp_neg) {
// 			// 	imag = -imag;
// 			// }
// 		}
// 		else {
// 			real = (XFLOAT)0;
// 			imag = (XFLOAT)0;
// 		}
// 	};

// 	constexpr int kDimOrientSlice = (TParams::kBlockSize / TParams::kOrientBlockSize);
// 	constexpr int kNumOrientSlice = (TParams::kImgBlockSize + kDimOrientSlice - 1) / kDimOrientSlice;
// 	assert(kNumOrientSlice <=32);


// 	XFLOAT orient_real_buf[kNumOrientSlice], orient_imag_buf[kNumOrientSlice];
// 	uint32_t flag_minus_buf[2] = {0, 0};

// 	//given current img_block_idx, corr_div_2, coord_x, coord_y, load orient mat into orient_real_buf and orient_imag_buf
// 	auto load_orient_mat_buf = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y, uint32_t& flag_minus)  {
// 		uint32_t flag_minus_loc = 0;

// 		#pragma unroll
// 		for (int cur_slice = 0; cur_slice < kNumOrientSlice; cur_slice++) {
// 			int i = tid / TParams::kOrientBlockSize + cur_slice * kDimOrientSlice;
// 			XFLOAT& orient_real = orient_real_buf[cur_slice];
// 			XFLOAT& orient_imag = orient_imag_buf[cur_slice];
// 			int g_img_idx = img_block_idx + i;
// 			int orient_idx = tid % TParams::kOrientBlockSize;
// 			int g_orient_idx = orient_block_idx + orient_idx;
// 			if (g_img_idx >= image_size || g_orient_idx >= orientation_num) {
// 				assert(orient_idx < TParams::kOrientBlockSize);
// 				assert(i < TParams::kImgBlockSize);
// 				orient_real = 0.0;
// 				orient_imag = 0.0;
// 			} else {
// 				XFLOAT e0 = g_eulers[g_orient_idx * 9];
// 				XFLOAT e1 = g_eulers[g_orient_idx * 9 + 1];
// 				XFLOAT e3 = g_eulers[g_orient_idx * 9 + 3];
// 				XFLOAT e4 = g_eulers[g_orient_idx * 9 + 4];
// 				XFLOAT e6 = g_eulers[g_orient_idx * 9 + 6];
// 				XFLOAT e7 = g_eulers[g_orient_idx * 9 + 7];


// 				project3Dmodel_sp(coord_x[i], coord_y[i], e0, e1, e3, e4, e6, e7, orient_real, orient_imag, flag_minus_loc, 1U << (cur_slice % 32));

// 			}
// 		}
// 		flag_minus += flag_minus_loc;
// 	};

// 	//given current img_block_idx, corr_div_2, coord_x, coord_y, dump orient_real_buf and orient_imag_buf into s_orient_real_mat_block and s_orient_imag_mat_block
// 	auto dump_orient_mat_shm = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y, uint32_t& flag_minus)  {
// 		#pragma unroll
// 		for (int cur_slice = 0; cur_slice < kNumOrientSlice; cur_slice++) {
// 			int i = tid / TParams::kOrientBlockSize + cur_slice * kDimOrientSlice;
// 			XFLOAT& orient_real = orient_real_buf[cur_slice];
// 			XFLOAT& orient_imag = orient_imag_buf[cur_slice];
			
// 			bool flag_cur_minus = (flag_minus & (1U << (cur_slice % 32))) >> (cur_slice % 32);
// 			orient_imag = flag_cur_minus ? -orient_imag : orient_imag;

// 			int orient_idx = tid % TParams::kOrientBlockSize;

// 			s_orient_real_mat_block_swizzle(orient_idx, i) = orient_real;
// 			s_orient_imag_mat_block_swizzle(orient_idx, i) = orient_imag;

// 			XFLOAT magnitude_squared_sum = orient_real * orient_real * corr_div_2[i] + orient_imag * orient_imag * corr_div_2[i];
// 			s_orient_pow2_accumulator[tid] += magnitude_squared_sum;
// 		}
// 		flag_minus = 0;
// 	};

//     auto init_fragment_c = [&] () {
// 		// Default: need read from g_diff2s
// 		if (scheduler.get_strategy() == CoarseSchedulerStrategy::Default || 
// 		   (scheduler.get_strategy() == CoarseSchedulerStrategy::SplitK && g_diff2s != g_diff2s_opt)) {		
// 			#pragma unroll
// 			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 				#pragma unroll
// 				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 					#pragma unroll
// 					for (int k = 0; k < kFragmentCSize; ++k) {
// 						int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
// 						int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);
// 						if (m < translation_num && n < orientation_num) {
// 							fragment_c[i][j][k] = g_diff2s[n * translation_num + m];
// 						} else {
// 							fragment_c[i][j][k] = 0.0;
// 						}
// 					}
// 				}
// 			}
// 		}
// 		// SplitK: use atomicAdd to accumulate, if diff2s source == diff2s dest, no need to read from g_diff2s
// 		// else, read from g_diff2s
// 		else if (scheduler.get_strategy() == CoarseSchedulerStrategy::SplitK && g_diff2s == g_diff2s_opt) {
// 			#pragma unroll
// 			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 				#pragma unroll
// 				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 					#pragma unroll
// 					for (int k = 0; k < kFragmentCSize; ++k) {
// 						fragment_c[i][j][k] = 0.0;
// 					}
// 				}
// 			}
			
// 		}
// 		else {
// 			assert(false);
// 		}
//     };

// 	auto epilogue = [&] () {
// 		// write fragment_c back to g_diff2s_opt
//         if (scheduler.get_strategy() == CoarseSchedulerStrategy::Default) {
// 			#pragma unroll
// 			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 				#pragma unroll
// 				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 					#pragma unroll
// 					for (int k = 0; k < kFragmentCSize; ++k) {
// 						int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
// 						int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);

// 						if (m < translation_num && n < orientation_num) {
// 							g_diff2s_opt[n * translation_num + m] = fragment_c[i][j][k];
// 						}
// 					}
// 				}
// 			}
// 		} else if (scheduler.get_strategy() == CoarseSchedulerStrategy::SplitK) {
// 			// use atomic add
// 			#pragma unroll
// 			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 				#pragma unroll
// 				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 					#pragma unroll
// 					for (int k = 0; k < kFragmentCSize; ++k) {
// 						int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
// 						int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);

// 						if (m < translation_num && n < orientation_num) {
// 							atomicAdd(&g_diff2s_opt[n * translation_num + m], fragment_c[i][j][k]);
// 						}
// 					}
// 				}
// 			}
// 		} else {
// 			assert(false);
// 		}
// 	};

//     // =====================================================================
// 	// ============================= main loop =============================
//     // =====================================================================
//     while (scheduler.has_work()) {
// 		__syncthreads();

//         trans_block_idx = scheduler.get_current_work_m_block_offset();
//         orient_block_idx = scheduler.get_current_work_n_block_offset();
// 		// if (tid == 0) {
// 		// 	printf("bid : %3d  tb_idx : %6d ob_idx : %6d\n", bid, trans_block_idx, orient_block_idx);
// 		// }
		
// 		// if (tid == 0) {
// 		// 	printf("bid : %3d  tb_idx : %6d ob_idx : %6d\n", bid, trans_block_idx, orient_block_idx);
// 		// }

//         // initialize shared memory to zero
//         for (int i = tid; i < 2 * TParams::kTransBlockSize * TParams::kImgBlockSize; i += TParams::kBlockSize) {
//             s_trans_mat_block[i] = 0.0;
//         }
//         for (int i = tid; i < 2 * TParams::kOrientBlockSize * TParams::kImgBlockSize; i += TParams::kBlockSize) {
//             s_orient_mat_block[i] = 0.0;
//         }

//         s_trans_pow2_accumulator[tid] = 0.0;
//         s_orient_pow2_accumulator[tid] = 0.0;

//         for (int i = tid; i < TParams::kImgBlockSize; i += TParams::kBlockSize) {
//             s_corr_div_2[0][i] = 0.0;
//             s_corr_div_2[1][i] = 0.0;
//             s_coor_x[0][i] = 0.0;
//             s_coor_y[0][i] = 0.0;
//             s_coor_x[1][i] = 0.0;
//             s_coor_y[1][i] = 0.0;
//         }

// 		flag_minus_buf[0] = 0;
// 		flag_minus_buf[1] = 0;

//         // read fragment_c from g_diff2s
//         init_fragment_c();

//         __syncthreads();

// /*=============================== FOR IMAGE BLOCK ==============================*/
// 		int img_block_idx = -1;
//         while (scheduler.get_current_work_next_k_block_offset(img_block_idx)) {
// 			assert(img_block_idx >= 0);
// 			// int img_iter = img_block_idx / TParams::kImgBlockSize;
// 			int k_cycle = scheduler.get_current_work_k_cycle();
// 			int k_cycle_mod2 = k_cycle % 2;
// 			int k_cycle_next_mod2 = (k_cycle + 1) % 2;
// 			if (scheduler.is_first_k_cycle()){
// 			// if (img_iter == 0) {
// 				load_coord_xy(img_block_idx, s_corr_div_2[k_cycle_mod2], s_coor_x[k_cycle_mod2], s_coor_y[k_cycle_mod2]);
// 				__syncthreads();
// 				// construct orient_mat
// 				load_orient_mat_buf(img_block_idx, s_corr_div_2[k_cycle_mod2], s_coor_x[k_cycle_mod2], s_coor_y[k_cycle_mod2], flag_minus_buf[k_cycle_mod2]);
// 			}
// 			if (! scheduler.is_last_k_cycle()) {
// 				// assert(img_block_idx + TParams::kImgBlockSize < image_size);
// 			// if (img_iter + 1 < (image_size + TParams::kImgBlockSize - 1) / TParams::kImgBlockSize) {
// 				__syncthreads();
// 				// construct trans_mat
// 				load_trans_mat(img_block_idx, s_corr_div_2[k_cycle_mod2], s_coor_x[k_cycle_mod2], s_coor_y[k_cycle_mod2]);
// 				// construct orient_mat on this iteration
// 				dump_orient_mat_shm(img_block_idx, s_corr_div_2[k_cycle_mod2], s_coor_x[k_cycle_mod2], s_coor_y[k_cycle_mod2], flag_minus_buf[k_cycle_mod2]);
// 				__syncthreads();
// 				//load coord_xy on next iteration
// 				load_coord_xy(img_block_idx + TParams::kImgBlockSize, s_corr_div_2[k_cycle_next_mod2], s_coor_x[k_cycle_next_mod2], s_coor_y[k_cycle_next_mod2]);
// 				__syncthreads();
// 				// construct orient_mat for next iteration
// 				load_orient_mat_buf(img_block_idx + TParams::kImgBlockSize, s_corr_div_2[k_cycle_next_mod2], s_coor_x[k_cycle_next_mod2], s_coor_y[k_cycle_next_mod2], flag_minus_buf[k_cycle_next_mod2]);
// 			} else {
// 				__syncthreads();
// 				// construct trans_mat
// 				load_trans_mat(img_block_idx, s_corr_div_2[k_cycle_mod2], s_coor_x[k_cycle_mod2], s_coor_y[k_cycle_mod2]);
// 				// construct orient_mat on this iteratiobn
// 				dump_orient_mat_shm(img_block_idx, s_corr_div_2[k_cycle_mod2], s_coor_x[k_cycle_mod2], s_coor_y[k_cycle_mod2], flag_minus_buf[k_cycle_mod2]);
// 				__syncthreads();
// 			}
//     /*=============================== COMPUTE CROSS TERM ==============================*/

//             block_mma_tf32_sim_fp32<decltype(s_trans_mat_block_swizzle), decltype(s_orient_mat_block_swizzle), 
//             TParams::kTransBlockSize, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize,
//             TParams::kWarpTransTileSize, TParams::kWarpOrientTileSize, 2 * TParams::kWarpImgTileSize,
//             TParams::kMmaTransTileSize, TParams::kMmaOrientTileSize, TParams::kMmaImgTileSize>(
//                 fragment_c, s_trans_mat_block_swizzle, s_orient_mat_block_swizzle, warp_id, lane_id);

//         } // end of image block

//         // reduce s_trans_pow2_accumulator
//         for (int i = 1; i < TParams::kBlockSize / TParams::kTransBlockSize; ++i) {
//             if (tid < TParams::kTransBlockSize) {
//                 s_trans_pow2_accumulator[tid] += s_trans_pow2_accumulator[i * TParams::kTransBlockSize + tid];
//             }
//         }
//         // reduce s_orient_pow2_accumulator
//         for (int i = 1; i < TParams::kBlockSize / TParams::kOrientBlockSize; ++i) {
//             if (tid < TParams::kOrientBlockSize) {
//                 s_orient_pow2_accumulator[tid] += s_orient_pow2_accumulator[i * TParams::kOrientBlockSize + tid];
//             }
//         }

//     /*=============================== REDUCE IN FRAGMENT_C ==============================*/
//         __syncthreads();
        
//         #pragma unroll
//         for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
//             #pragma unroll
//             for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
//                 #pragma unroll
//                 for (int k = 0; k < kFragmentCSize; ++k) {
//                     int m = fragment_c_m_idx_in_block<TParams>(warp_id, lane_id, i, k);
//                     int n = fragment_c_n_idx_in_block<TParams>(warp_id, lane_id, j, k);
//                     fragment_c[i][j][k] += s_trans_pow2_accumulator[m] + s_orient_pow2_accumulator[n];
//                 }
//             }
//         }
//         __syncthreads();

//     /*=============================== WRITE BACK ==============================*/
//         // write fragment_c back to g_diff2s_opt
//         // #pragma unroll
//         // for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
//         //     #pragma unroll
//         //     for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
//         //         #pragma unroll
//         //         for (int k = 0; k < kFragmentCSize; ++k) {
//         //             int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
//         //             int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);

//         //             if (m < translation_num && n < orientation_num) {
//         //                 g_diff2s_opt[n * translation_num + m] = fragment_c[i][j][k];
// 		// 			}
//         //         }
//         //     }
//         // }

// 		epilogue();

//         scheduler.advance_to_next_work();
//     } // end of while has_work
// }

// // A debug version
// // // construct orientation matrix (use z-order)
// // template<typename TParams>
// // __launch_bounds__(128, 2)
// // __global__ void cuda_kernel_coarse_matrix(
// // 	XFLOAT *g_eulers,
// // 	XFLOAT *trans_x,
// // 	XFLOAT *trans_y,
// // 	XFLOAT *g_real,
// // 	XFLOAT *g_imag,
// // 	AccProjectorKernel projector,
// // 	XFLOAT *g_corr,
// // 	XFLOAT *g_diff2s,
// // 	XFLOAT *g_diff2s_opt,
// // 	const int translation_num,
// // 	const int orientation_num,
// // 	const int image_size,
// // 	XFLOAT *g_trans_real_m,
// // 	XFLOAT *g_trans_imag_m,
// // 	XFLOAT *g_orient_real_m,
// // 	XFLOAT *g_orient_imag_m) {
// // 	static_assert(TParams::kBlockSize % 32 == 0, "kBlockSize must be a multiple of 32");
// // 	static_assert(TParams::kImgBlockSize == TParams::kWarpImgTileSize, "kImgBlockSize must be equal to kWarpImgTileSize");
// // 	static_assert(TParams::kBlockSize >= TParams::kTransBlockSize, "kBlockSize must be greater than or equal to kTransBlockSize");
// // 	static_assert(TParams::kBlockSize >= TParams::kOrientBlockSize, "kBlockSize must be greater than or equal to kOrientBlockSize");

// // 	static_assert(TParams::kTransBlockSize % TParams::kWarpTransTileSize == 0, "kTransBlockSize must be a multiple of kWarpTransTileSize");
// // 	static_assert(TParams::kOrientBlockSize % TParams::kWarpOrientTileSize == 0, "kOrientBlockSize must be a multiple of kWarpOrientTileSize");
// // 	static_assert(TParams::kTransBlockSize % TParams::kMmaTransTileSize == 0, "kTransBlockSize must be a multiple of kMmaTransTileSize");
// // 	static_assert(TParams::kOrientBlockSize % TParams::kMmaOrientTileSize == 0, "kOrientBlockSize must be a multiple of kMmaOrientTileSize");
// // 	static_assert(TParams::kBlockSize / 32 == (TParams::kTransBlockSize / TParams::kWarpTransTileSize) * (TParams::kOrientBlockSize / TParams::kWarpOrientTileSize), "kBlockSize must be equal to the product of the number of warps in translation, orientation and image dimension");

// // 	static_assert(TParams::kImgBlockSize == 16, "kImgBlockSize must be 16");

// // 	const int tid = threadIdx.x;          // thread id in a block
// // 	const int bid = blockIdx.x;           // block id in a grid
// // 	const int warp_id  = tid / 32;        // warp id in a block
// // 	constexpr int kWarpNum = TParams::kBlockSize / 32; // number of warps in a block
// // 	const int lane_id  = tid % 32;        // thread id in a warp

// // 	const int trans_block_num = (translation_num + TParams::kTransBlockSize - 1) / TParams::kTransBlockSize;
// // 	const int orient_block_num = (orientation_num + TParams::kOrientBlockSize - 1) / TParams::kOrientBlockSize;

// // 	// bug:这里一开始没有括号
// // 	int trans_block_idx = (bid % trans_block_num) * TParams::kTransBlockSize;
// // 	int orient_block_idx = (bid / trans_block_num) * TParams::kOrientBlockSize;
// //     // int trans_block_idx = 0; // forware declaration
// //     // int orient_block_idx = 0;

// // 	assert(blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0);
	
// // 	CoarseScheduler<TParams::kTransBlockSize, 
// // 					TParams::kOrientBlockSize, 
// // 					TParams::kImgBlockSize, 
// // 					CoarseSchedulerStrategy::SplitK,
// // 					2>
// // 		scheduler(translation_num, orientation_num, image_size);
	
// // 	OrientationMatrixHandler<TParams::kOrientBlockSize,
// // 							 TParams::kImgBlockSize,
// // 							 kWarpNum,
// // 							 SharedMemorySwizzle<float, TParams::kOrientBlockSize, TParams::kImgBlockSize, 0>,
// // 							 SharedMemorySwizzle<float, TParams::kOrientBlockSize, TParams::kImgBlockSize, TParams::kImgBlockSize>>
// // 		orientation_matrix_handler(image_size, orientation_num);
	

// // 	// 'img' data is stored contiguously.
// // 	// __shared__ XFLOAT s_trans_real_mat_block[TParams::kTransBlockSize * TParams::kImgBlockSize];
// // 	// __shared__ XFLOAT s_trans_imag_mat_block[TParams::kTransBlockSize * TParams::kImgBlockSize];
// // 	// __shared__ XFLOAT s_orient_real_mat_block[TParams::kOrientBlockSize * TParams::kImgBlockSize];
// // 	// __shared__ XFLOAT s_orient_imag_mat_block[TParams::kOrientBlockSize * TParams::kImgBlockSize];

// // 	__shared__ XFLOAT s_trans_mat_block[2 * TParams::kTransBlockSize * TParams::kImgBlockSize];
// // 	__shared__ XFLOAT s_orient_mat_block[2 * TParams::kOrientBlockSize * TParams::kImgBlockSize];
	
// // 	SharedMemorySwizzle<float, TParams::kTransBlockSize, 2 * TParams::kImgBlockSize, 0> s_trans_mat_block_swizzle(s_trans_mat_block);
// // 	SharedMemorySwizzle<float, TParams::kTransBlockSize, TParams::kImgBlockSize, 0> s_trans_real_mat_block_swizzle(s_trans_mat_block);
// // 	SharedMemorySwizzle<float, TParams::kTransBlockSize, TParams::kImgBlockSize, TParams::kImgBlockSize> s_trans_imag_mat_block_swizzle(s_trans_mat_block);

// // 	SharedMemorySwizzle<float, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize, 0> s_orient_mat_block_swizzle(s_orient_mat_block);
// // 	SharedMemorySwizzle<float, TParams::kOrientBlockSize, TParams::kImgBlockSize, 0> s_orient_real_mat_block_swizzle(s_orient_mat_block);
// // 	SharedMemorySwizzle<float, TParams::kOrientBlockSize, TParams::kImgBlockSize, TParams::kImgBlockSize> s_orient_imag_mat_block_swizzle(s_orient_mat_block);

// // 	__shared__ XFLOAT s_orient_mat_block_bak[2 * TParams::kOrientBlockSize * TParams::kImgBlockSize];
// // 	SharedMemorySwizzle<float, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize, 0> s_orient_mat_block_swizzle_bak(s_orient_mat_block_bak);
// // 	SharedMemorySwizzle<float, TParams::kOrientBlockSize, TParams::kImgBlockSize, 0> s_orient_real_mat_block_swizzle_bak(s_orient_mat_block_bak);
// // 	SharedMemorySwizzle<float, TParams::kOrientBlockSize, TParams::kImgBlockSize, TParams::kImgBlockSize> s_orient_imag_mat_block_swizzle_bak(s_orient_mat_block_bak);


// // 	// double buffer for s_corr_div_2, s_coor_x, s_coor_y
// // 	__shared__ XFLOAT s_corr_div_2[2][TParams::kImgBlockSize];
// // 	__shared__ XFLOAT s_coor_x[2][TParams::kImgBlockSize];
// // 	__shared__ XFLOAT s_coor_y[2][TParams::kImgBlockSize];

// // 	// ============================  new  ============================
// // 	__shared__ XFLOAT s_fcoor_xy[2][TParams::kImgBlockSize * 2]; // img -> x,y
// // 	// For a 2D scenario, e8 is not used, so it’s not stored in shared memory.
// // 	// e2 and e5 are also unused, but they remain in shared memory for alignment.
// // 	__shared__ XFLOAT s_eulers_head[TParams::kOrientBlockSize * 4]; // e0 e1 e2 e3
// // 	__shared__ XFLOAT s_eulers_tail[TParams::kOrientBlockSize * 4]; // e4 e5 e6 e7

// // 	// reduce buffer
// // 	__shared__ XFLOAT s_trans_pow2_accumulator[(TParams::kBlockSize / TParams::kTransBlockSize) * TParams::kTransBlockSize];
// // 	__shared__ XFLOAT s_orient_pow2_accumulator[(TParams::kBlockSize / TParams::kOrientBlockSize) * TParams::kOrientBlockSize];

// // 	__shared__ XFLOAT s_orient_pow2_accumulator_bak[TParams::kOrientBlockSize];
// // 	// register
// // 	constexpr int kNumMmaTransInWarpTile = TParams::kWarpTransTileSize / TParams::kMmaTransTileSize;
// // 	constexpr int kNumMmaOrientInWarpTile = TParams::kWarpOrientTileSize / TParams::kMmaOrientTileSize;
// // 	constexpr int kNumMmaImgInWarpTile = TParams::kWarpImgTileSize / TParams::kMmaImgTileSize;

// // 	constexpr int kFragmentASize = TParams::kMmaTransTileSize * TParams::kMmaImgTileSize / kWarpSize;
// // 	constexpr int kFragmentBSize = TParams::kMmaOrientTileSize * TParams::kMmaImgTileSize / kWarpSize;
// // 	constexpr int kFragmentCSize = TParams::kMmaTransTileSize * TParams::kMmaOrientTileSize / kWarpSize;

// // 	// XFLOAT fragment_a[kNumMmaTransInWarpTile][kFragmentASize];
// // 	// XFLOAT fragment_b[kNumMmaOrientInWarpTile][kFragmentBSize];
// // 	XFLOAT fragment_c[kNumMmaTransInWarpTile][kNumMmaOrientInWarpTile][kFragmentCSize];

// // 	constexpr int kRegistersMmaPerThread = kNumMmaTransInWarpTile * kFragmentASize +
// // 										   kNumMmaOrientInWarpTile * kFragmentBSize +
// // 										   kNumMmaTransInWarpTile * kNumMmaOrientInWarpTile * kFragmentCSize;
	
// // 	#if kRegistersMmaPerThread >= 256
// // 	#warning "kRegistersMmaPerThread must be less than or equal to 256, otherwise register spilling will occur"
// // 	#endif


// // 	// ============================= lambda function =============================
// // 	//given current img_block_idx, load global array into corr_div_2, coord_x, coord_y
// // 	auto load_coord_xy = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y, XFLOAT* fcoor_xy) {
// // 		#pragma unroll 
// // 		for (int i = tid; i < TParams::kImgBlockSize; i += TParams::kBlockSize) {
// // 			if (img_block_idx + i < image_size) {
// // 				corr_div_2[i] = g_corr[img_block_idx + i] / 2;
// // 			} else {
// // 				corr_div_2[i] = 0.;
// // 			}

// // 			int x, y;
// // 			pixel_index2coor(img_block_idx + i, projector.imgX, projector.imgY, projector.maxR, x, y);
// // 			coord_x[i] = x;
// // 			coord_y[i] = y;
			
// // 			// fcoor_xy[i][0] = x;
// // 			// fcoor_xy[i][1] = y;
// // 			fcoor_xy[2 * i + 0] = x;
// // 			fcoor_xy[2 * i + 1] = y;
// // 		}
// // 	};

// // 	//given current img_block_idx, corr_div_2, coord_x, coord_y, load trans mat into s_trans_real_mat_block and s_trans_imag_mat_block
// // 	auto load_trans_mat = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y) {
// // 		assert(TParams::kBlockSize % TParams::kTransBlockSize == 0);
// // 		#pragma unroll
// // 		for (int i = tid / TParams::kTransBlockSize; i < TParams::kImgBlockSize; i += TParams::kBlockSize / TParams::kTransBlockSize) {
// // 			int g_img_idx = img_block_idx + i;
// // 			int trans_idx = tid % TParams::kTransBlockSize;
// // 			if (g_img_idx >= image_size) {
// // 				assert(trans_idx < TParams::kTransBlockSize);
// // 				assert(i < TParams::kImgBlockSize);
// // 				s_trans_real_mat_block_swizzle(trans_idx, i) = 0.;
// // 				s_trans_imag_mat_block_swizzle(trans_idx, i) = 0.;
// // 				continue;
// // 			}
// // 			int g_trans_idx = trans_block_idx + trans_idx;
// // 			if (g_trans_idx >= translation_num) {
// // 				continue;
// // 			}
// // 			XFLOAT tx = trans_x[g_trans_idx];
// // 			XFLOAT ty = trans_y[g_trans_idx];
// // 			XFLOAT real = g_real[g_img_idx];
// // 			XFLOAT imag = g_imag[g_img_idx];

// // 			int x = coord_x[i];
// // 			int y = coord_y[i];
// // 			XFLOAT trans_real, trans_imag;
// // 			translatePixel(x, y, tx, ty, real, imag, trans_real, trans_imag);

// // 			// s_trans_real_mat_block[trans_idx * TParams::kImgBlockSize + i] = -2 * trans_real * corr_div_2[i];
// // 			// s_trans_imag_mat_block[trans_idx * TParams::kImgBlockSize + i] = -2 * trans_imag * corr_div_2[i];
// // 			s_trans_real_mat_block_swizzle(trans_idx, i) = -2 * trans_real * corr_div_2[i];
// // 			s_trans_imag_mat_block_swizzle(trans_idx, i) = -2 * trans_imag * corr_div_2[i];

// // 			XFLOAT magnitude_squared_sum = trans_real * trans_real * corr_div_2[i] + trans_imag * trans_imag * corr_div_2[i];
// // 			s_trans_pow2_accumulator[tid] += magnitude_squared_sum;
// // 		}
// // 	};

// // 	// // z-order
// // 	// // row: 8 orientation block
// // 	// // col: 4 img block
// // 	// /**
// // 	//  *    +----------------+
// // 	//  *    |       w0       |
// // 	//  *    +----------------+
// // 	//  *    |       w1       |
// // 	//  *    +----------------+
// // 	//  *    |       w2       |
// // 	//  *    +----------------+
// // 	//  *    |       w3       |
// // 	//  *    +----------------+
// // 	//  *    |       w0       |
// // 	//  *    +----------------+
// // 	//  *    |       w1       |
// // 	//  *           ...
// // 	//  */

// // 	// //   +---+---+---+---+
// // 	// //   |  0|  2|  4|  6|
// // 	// //   +---+---+---+---+
// // 	// //   |  1|  3|  5|  7|
// // 	// //   +---+---+---+---+
// // 	// //   |  8| 10| 12| 14|
// // 	// //   +---+---+---+---+
// // 	// //   |  9| 11| 13| 15|
// // 	// //   +---+---+---+---+
// // 	// //   | 16| 18| 20| 22|
// // 	// //   +---+---+---+---+
// // 	// //   | 17| 19| 21| 23|
// // 	// //   +---+---+---+---+
// // 	// //   | 24| 26| 28| 30|
// // 	// //   +---+---+---+---+
// // 	// //   | 25| 27| 29| 31|
// // 	// //   +---+---+---+---+

// // 	// WarpLayout<4, 2, LayoutMajorType::ColumnMajor> orient_warp_layout;
// // 	// constexpr int kOMImgPerThread = 4;
// // 	// constexpr int kOMOrientPerThread = 1;
// // 	// constexpr int kOMOrientPerWarp = 8;
// // 	// static_assert(kOMOrientPerWarp == orient_warp_layout.rows, "kOMOrientPerWarp must be equal to orient_warp_layout.rows");
	
// // 	// constexpr int kOMNumOrientWarpTile = TParams::kOrientBlockSize / kOMOrientPerWarp;
// // 	// constexpr int kOMNumOrientWarpTilePerWarp = kOMNumOrientWarpTile / kWarpNum;
	
// // 	// // [real/imag] [kOMNumOrientWarpTilePerWarp] [kOMImgPerThread]
// // 	// //  2           x                             4               
// // 	// XFLOAT reg_tex_buf[2][kOMNumOrientWarpTilePerWarp][kOMImgPerThread];
	
// // 	// auto construct_orientation_matrix = [&](
// // 	// 	int img_block_idx,
// // 	// 	int orient_block_idx,
// // 	// 	int image_size,
// // 	// 	int orientation_num,
// // 	// 	XFLOAT* s_eulers_head, // kOrientBlockSize * 4
// // 	// 	XFLOAT* s_eulers_tail, // kOrientBlockSize * 4
// // 	// 	XFLOAT* s_fcoor_xy    // kImgBlockSize * 2
// // 	// ) {
// // 	// 	XFLOAT reg_fcoor_xy[kOMImgPerThread][2]; // 4 img (x, y)
// // 	// 	XFLOAT reg_eulers[kOMNumOrientWarpTilePerWarp][8];

// // 	// 	// debug print
// // 	// 	if (tid == 0) {
// // 	// 		printf("s_eulers:\n");
// // 	// 		for (int i = 0; i < TParams::kOrientBlockSize; i ++) {
// // 	// 			printf("%3d (%7.2e %7.2e %7.2e %7.2e %7.2e %7.2e %7.2e %7.2e)\n", i,
// // 	// 				s_eulers_head[i * 4 + 0], s_eulers_head[i * 4 + 1], 
// // 	// 				s_eulers_head[i * 4 + 2], s_eulers_head[i * 4 + 3], 
// // 	// 				s_eulers_tail[i * 4 + 0], s_eulers_tail[i * 4 + 1], 
// // 	// 				s_eulers_tail[i * 4 + 2], s_eulers_tail[i * 4 + 3]);	
// // 	// 		}
// // 	// 		printf("fcoor_xy:\n");
// // 	// 		for (int i = 0; i < TParams::kImgBlockSize; i ++) {
// // 	// 			printf("%3d (%4e %4e)\n", i, 
// // 	// 				s_fcoor_xy[i * 2 + 0], s_fcoor_xy[i * 2 + 1]);
// // 	// 		}
// // 	// 	}
// // 	// 	__syncthreads();
		
// // 	// 	// each ld128 will only need 2 transaction without bankconflict
// // 	// 	// load eulers
// // 	// 	#pragma unroll
// // 	// 	for (int i = warp_id; i < kOMNumOrientWarpTile; i += kWarpNum) {
// // 	// 		int s_orient_idx = i * kOMOrientPerWarp + orient_warp_layout.get_row_idx(lane_id);
// // 	// 		printf("warp id : %2d lane id : %2d, s_orient_idx : %5d\n", warp_id, lane_id, s_orient_idx);
// // 	// 		assert(s_orient_idx < TParams::kOrientBlockSize);
// // 	// 		assert(i / kWarpNum < kOMNumOrientWarpTilePerWarp);
// // 	// 		// load float4
// // 	// 		*reinterpret_cast<float4*>(&reg_eulers[i / kWarpNum][0])
// // 	// 			= reinterpret_cast<float4*>(s_eulers_head)[s_orient_idx];
// // 	// 		// float4 tmp = reinterpret_cast<float4*>(s_eulers_head)[s_orient_idx];
// // 	// 		*reinterpret_cast<float4*>(&reg_eulers[i / kWarpNum][4])
// // 	// 			= reinterpret_cast<float4*>(s_eulers_tail)[s_orient_idx];
// // 	// 	}

// // 	// 	// load coor
// // 	// 	// step is 2, for 2(step) x 2(x,y) = 4 (float4)
// // 	// 	#pragma unroll
// // 	// 	for (int i = 0; i < kOMImgPerThread; i += 2) {
// // 	// 		// float1 idx : (get_col_idx * kOMImgPerThread + i) * 2 (x,y)
// // 	// 		int s_img_float4_idx = (orient_warp_layout.get_col_idx(lane_id) * kOMImgPerThread + i) / 2;
			
// // 	// 		assert(i < kOMImgPerThread);
// // 	// 		assert(s_img_float4_idx * 4 < TParams::kImgBlockSize);
// // 	// 		*reinterpret_cast<float4*>(&reg_fcoor_xy[i][0])
// // 	// 			= reinterpret_cast<float4*>(s_fcoor_xy)[s_img_float4_idx];
// // 	// 	}
		
// // 	// 	// debug print
// // 	// 	for (int i = 0; i < TParams::kBlockSize; i ++) {
// // 	// 		if (i == tid) {
// // 	// 			printf("tid : %3d\n", tid);
// // 	// 			for (int j = 0; j < kOMNumOrientWarpTilePerWarp; j ++) {
// // 	// 				printf("  reg_eulers[%2d] : ", j);
// // 	// 				for (int k = 0; k < 8; k ++) {
// // 	// 					printf("%7.2e ", reg_eulers[j][k]);
// // 	// 				}
// // 	// 				printf("\n");

// // 	// 				int g_orient_idx = orient_block_idx + (j * kWarpNum + warp_id) * orient_warp_layout.rows + orient_warp_layout.get_row_idx(lane_id);
// // 	// 				printf("  g_orient_idx: %5d\n", g_orient_idx);
// // 	// 			}
// // 	// 			for (int j = 0; j < kOMImgPerThread; j ++) {
// // 	// 				printf("  reg_fcoor_xy[%2d] : %7.2e %7.2e\n", j, reg_fcoor_xy[j][0], reg_fcoor_xy[j][1]);
// // 	// 				int g_img_idx = img_block_idx + orient_warp_layout.get_col_idx(lane_id) * kOMImgPerThread + j;
// // 	// 				printf("  g_img_idx: %5d\n", g_img_idx);
// // 	// 			}
// // 	// 		}
// // 	// 		__syncthreads();
// // 	// 	}
		

// // 	// 	for (int i = 0; i < kOMNumOrientWarpTilePerWarp; i ++) {
// // 	// 		for (int j = 0; j < kOMImgPerThread; j ++) {
// // 	// 			int g_img_idx = img_block_idx + orient_warp_layout.get_col_idx(lane_id) * kOMImgPerThread + j;
// // 	// 			int g_orient_idx = orient_block_idx + (i * kWarpNum + warp_id) * orient_warp_layout.rows + orient_warp_layout.get_row_idx(lane_id);
// // 	// 			bool within_bounds = g_img_idx < image_size && g_orient_idx < orientation_num;

// // 	// 			XFLOAT& x = reg_fcoor_xy[j][0];
// // 	// 			XFLOAT& y = reg_fcoor_xy[j][1];
// // 	// 			XFLOAT& e0 = reg_eulers[i][0];
// // 	// 			XFLOAT& e1 = reg_eulers[i][1];
// // 	// 			XFLOAT& e3 = reg_eulers[i][3];
// // 	// 			XFLOAT& e4 = reg_eulers[i][4];
// // 	// 			XFLOAT& e6 = reg_eulers[i][6];
// // 	// 			XFLOAT& e7 = reg_eulers[i][7];
// // 	// 			XFLOAT xp = (e0 * x + e1 * y ) * projector.padding_factor;
// // 	// 			XFLOAT yp = (e3 * x + e4 * y ) * projector.padding_factor;
// // 	// 			XFLOAT zp = (e6 * x + e7 * y ) * projector.padding_factor;
// // 	// 			int r2 = xp*xp + yp*yp + zp*zp;
// // 	// 			if (r2 <= projector.maxR2_padded && within_bounds) {
// // 	// 				bool xp_neg = xp < 0;
// // 	// 				// flag_minus += xp_neg ? mask : 0;
// // 	// 				// NOTICE: if xp_neg, imag = -imag
// // 	// 				if (xp_neg) {
// // 	// 					// Get complex conjugated hermitian symmetry pair
// // 	// 					xp = -xp;
// // 	// 					yp = -yp;
// // 	// 					zp = -zp;
// // 	// 					yp -= projector.mdlInitY;
// // 	// 					zp -= projector.mdlInitZ;
// // 	// 				}
// // 	// 				else {
// // 	// 					yp -= projector.mdlInitY;
// // 	// 					zp -= projector.mdlInitZ;
// // 	// 				}
// // 	// 				reg_tex_buf[0][i][j] = tex3D<XFLOAT>(projector.mdlReal, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
// // 	// 				reg_tex_buf[1][i][j] = tex3D<XFLOAT>(projector.mdlImag, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
// // 	// 				// real =    tex3D<XFLOAT>(projector.mdlReal, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
// // 	// 				// imag =    tex3D<XFLOAT>(projector.mdlImag, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
// // 	// 				if(xp_neg) {
// // 	// 					// imag = -imag;
// // 	// 					reg_tex_buf[1][i][j] = -reg_tex_buf[1][i][j];
// // 	// 				}
// // 	// 			}
// // 	// 			else {
// // 	// 				// real = (XFLOAT)0;
// // 	// 				// imag = (XFLOAT)0;
// // 	// 				reg_tex_buf[0][i][j] = 0.;
// // 	// 				reg_tex_buf[1][i][j] = 0.;
// // 	// 			}
// // 	// 		}
// // 	// 	}

// // 	// 	// Store to smem
// // 	// 	for (int i = 0; i < kOMNumOrientWarpTilePerWarp; i ++) {
// // 	// 		for (int j = 0; j < kOMImgPerThread; j ++) {
// // 	// 			int s_orient_idx = i * kOMOrientPerWarp + orient_warp_layout.get_row_idx(lane_id);
// // 	// 			int s_img_idx = orient_warp_layout.get_col_idx(lane_id) * kOMImgPerThread + j;
// // 	// 			s_orient_real_mat_block_swizzle_bak(s_orient_idx, s_img_idx) = reg_tex_buf[0][i][j];
// // 	// 			s_orient_imag_mat_block_swizzle_bak(s_orient_idx, s_img_idx) = reg_tex_buf[1][i][j];

// // 	// 			// s_orient_real_mat_block_swizzle_bak(s_orient_idx, s_img_idx) = reg_tex_buf[0][i][j];
// // 	// 			// s_orient_imag_mat_block_swizzle_bak(s_orient_idx, s_img_idx) = reg_tex_buf[1][i][j];

// // 	// 		}
// // 	// 	}

// // 	// };

// // 	auto project3Dmodel_sp = [&](
// // 			XFLOAT x,
// // 			XFLOAT y,
// // 			XFLOAT e0,
// // 			XFLOAT e1,
// // 			XFLOAT e3,
// // 			XFLOAT e4,
// // 			XFLOAT e6,
// // 			XFLOAT e7,
// // 			XFLOAT &real,
// // 			XFLOAT &imag,
// // 			uint32_t& flag_minus,
// // 			uint32_t mask) {
// // 		XFLOAT xp = (e0 * x + e1 * y ) * projector.padding_factor;
// // 		XFLOAT yp = (e3 * x + e4 * y ) * projector.padding_factor;
// // 		XFLOAT zp = (e6 * x + e7 * y ) * projector.padding_factor;
// // 		int r2 = xp*xp + yp*yp + zp*zp;
// // 		if (r2 <= projector.maxR2_padded)
// // 		{
// // 			bool xp_neg = xp < 0;
// // 			flag_minus += xp_neg ? mask : 0;
// // 			// NOTICE: if xp_neg, imag = -imag
// // 			if (xp_neg) {
// // 				// Get complex conjugated hermitian symmetry pair
// // 				xp = -xp;
// // 				yp = -yp;
// // 				zp = -zp;
// // 				yp -= projector.mdlInitY;
// // 				zp -= projector.mdlInitZ;
// // 			}
// // 			else {
// // 				yp -= projector.mdlInitY;
// // 				zp -= projector.mdlInitZ;
// // 			}
// // 			real =    tex3D<XFLOAT>(projector.mdlReal, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
// // 			imag =    tex3D<XFLOAT>(projector.mdlImag, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
			
// // 			// real =    e0;
// // 			// imag =    x;
			
// // 			// if(xp_neg) {
// // 			// 	imag = -imag;
// // 			// }
// // 		}
// // 		else {
// // 			real = (XFLOAT)0;
// // 			imag = (XFLOAT)0;
// // 		}
// // 	};

// // 	constexpr int kDimOrientSlice = (TParams::kBlockSize / TParams::kOrientBlockSize);
// // 	constexpr int kNumOrientSlice = (TParams::kImgBlockSize + kDimOrientSlice - 1) / kDimOrientSlice;
// // 	assert(kNumOrientSlice <=32);


// // 	XFLOAT orient_real_buf[kNumOrientSlice], orient_imag_buf[kNumOrientSlice];
// // 	uint32_t flag_minus_buf[2] = {0, 0};

// // 	//given current img_block_idx, corr_div_2, coord_x, coord_y, load orient mat into orient_real_buf and orient_imag_buf
// // 	auto load_orient_mat_buf = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y, uint32_t& flag_minus)  {
// // 		uint32_t flag_minus_loc = 0;

// // 		#pragma unroll
// // 		for (int cur_slice = 0; cur_slice < kNumOrientSlice; cur_slice++) {
// // 			int i = tid / TParams::kOrientBlockSize + cur_slice * kDimOrientSlice;
// // 			XFLOAT& orient_real = orient_real_buf[cur_slice];
// // 			XFLOAT& orient_imag = orient_imag_buf[cur_slice];
// // 			int g_img_idx = img_block_idx + i;
// // 			int orient_idx = tid % TParams::kOrientBlockSize;
// // 			int g_orient_idx = orient_block_idx + orient_idx;
// // 			if (g_img_idx >= image_size || g_orient_idx >= orientation_num) {
// // 				assert(orient_idx < TParams::kOrientBlockSize);
// // 				assert(i < TParams::kImgBlockSize);
// // 				orient_real = 0.0;
// // 				orient_imag = 0.0;
// // 			} else {
// // 				XFLOAT e0 = g_eulers[g_orient_idx * 9];
// // 				XFLOAT e1 = g_eulers[g_orient_idx * 9 + 1];
// // 				XFLOAT e3 = g_eulers[g_orient_idx * 9 + 3];
// // 				XFLOAT e4 = g_eulers[g_orient_idx * 9 + 4];
// // 				XFLOAT e6 = g_eulers[g_orient_idx * 9 + 6];
// // 				XFLOAT e7 = g_eulers[g_orient_idx * 9 + 7];


// // 				project3Dmodel_sp(coord_x[i], coord_y[i], e0, e1, e3, e4, e6, e7, orient_real, orient_imag, flag_minus_loc, 1U << (cur_slice % 32));

// // 			}
// // 		}
// // 		flag_minus += flag_minus_loc;
// // 	};

// // 	//given current img_block_idx, corr_div_2, coord_x, coord_y, dump orient_real_buf and orient_imag_buf into s_orient_real_mat_block and s_orient_imag_mat_block
// // 	auto dump_orient_mat_shm = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y, uint32_t& flag_minus)  {
// // 		#pragma unroll
// // 		for (int cur_slice = 0; cur_slice < kNumOrientSlice; cur_slice++) {
// // 			int i = tid / TParams::kOrientBlockSize + cur_slice * kDimOrientSlice;
// // 			XFLOAT& orient_real = orient_real_buf[cur_slice];
// // 			XFLOAT& orient_imag = orient_imag_buf[cur_slice];
			
// // 			bool flag_cur_minus = (flag_minus & (1U << (cur_slice % 32))) >> (cur_slice % 32);
// // 			orient_imag = flag_cur_minus ? -orient_imag : orient_imag;

// // 			int orient_idx = tid % TParams::kOrientBlockSize;

// // 			s_orient_real_mat_block_swizzle(orient_idx, i) = orient_real;
// // 			s_orient_imag_mat_block_swizzle(orient_idx, i) = orient_imag;

// // 			XFLOAT magnitude_squared_sum = orient_real * orient_real * corr_div_2[i] + orient_imag * orient_imag * corr_div_2[i];
// // 			// magnitude_squared_sum = 1;
// // 			s_orient_pow2_accumulator[tid] += magnitude_squared_sum;
// // 		}
// // 		flag_minus = 0;
// // 	};

// //     auto init_fragment_c = [&] () {
// // 		// Default: need read from g_diff2s
// // 		if (scheduler.get_strategy() == CoarseSchedulerStrategy::Default || 
// // 		   (scheduler.get_strategy() == CoarseSchedulerStrategy::SplitK && g_diff2s != g_diff2s_opt)) {		
// // 			#pragma unroll
// // 			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// // 				#pragma unroll
// // 				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// // 					#pragma unroll
// // 					for (int k = 0; k < kFragmentCSize; ++k) {
// // 						int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
// // 						int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);
// // 						if (m < translation_num && n < orientation_num) {
// // 							fragment_c[i][j][k] = g_diff2s[n * translation_num + m];
// // 						} else {
// // 							fragment_c[i][j][k] = 0.0;
// // 						}
// // 					}
// // 				}
// // 			}
// // 		}
// // 		// SplitK: use atomicAdd to accumulate, if diff2s source == diff2s dest, no need to read from g_diff2s
// // 		// else, read from g_diff2s
// // 		else if (scheduler.get_strategy() == CoarseSchedulerStrategy::SplitK && g_diff2s == g_diff2s_opt) {
// // 			#pragma unroll
// // 			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// // 				#pragma unroll
// // 				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// // 					#pragma unroll
// // 					for (int k = 0; k < kFragmentCSize; ++k) {
// // 						fragment_c[i][j][k] = 0.0;
// // 					}
// // 				}
// // 			}
			
// // 		}
// // 		else {
// // 			assert(false);
// // 		}
// //     };

// // 	auto epilogue = [&] () {
// // 		// write fragment_c back to g_diff2s_opt
// //         if (scheduler.get_strategy() == CoarseSchedulerStrategy::Default) {
// // 			#pragma unroll
// // 			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// // 				#pragma unroll
// // 				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// // 					#pragma unroll
// // 					for (int k = 0; k < kFragmentCSize; ++k) {
// // 						int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
// // 						int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);

// // 						if (m < translation_num && n < orientation_num) {
// // 							g_diff2s_opt[n * translation_num + m] = fragment_c[i][j][k];
// // 						}
// // 					}
// // 				}
// // 			}
// // 		} else if (scheduler.get_strategy() == CoarseSchedulerStrategy::SplitK) {
// // 			// use atomic add
// // 			#pragma unroll
// // 			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// // 				#pragma unroll
// // 				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// // 					#pragma unroll
// // 					for (int k = 0; k < kFragmentCSize; ++k) {
// // 						int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
// // 						int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);

// // 						if (m < translation_num && n < orientation_num) {
// // 							atomicAdd(&g_diff2s_opt[n * translation_num + m], fragment_c[i][j][k]);
// // 						}
// // 					}
// // 				}
// // 			}
// // 		} else {
// // 			assert(false);
// // 		}
// // 	};

// //     // =====================================================================
// // 	// ============================= main loop =============================
// //     // =====================================================================
// //     while (scheduler.has_work()) {
// // 		__syncthreads();

// //         trans_block_idx = scheduler.get_current_work_m_block_offset();
// //         orient_block_idx = scheduler.get_current_work_n_block_offset();
// // 		// if (tid == 0) {
// // 		// 	printf("bid : %3d  tb_idx : %6d ob_idx : %6d\n", bid, trans_block_idx, orient_block_idx);
// // 		// }
		
// // 		// if (tid == 0) {
// // 		// 	printf("bid : %3d  tb_idx : %6d ob_idx : %6d\n", bid, trans_block_idx, orient_block_idx);
// // 		// }
// // 		// load eulers to smem
// // 		#pragma unroll
// // 		for (int i = tid; i < TParams::kOrientBlockSize; i += TParams::kBlockSize) {

// // 			// TODO: 
// // 			// for (int j = 0; j < 4; j ++) {
// // 			// 	s_eulers_head[i * 4 + j] = (float)i;
// // 			// 	s_eulers_tail[i * 4 + j] = (float)i;
// // 			// }
			
// // 			if (orient_block_idx + i < orientation_num) {
// // 				// TODO: check whether compiler uses load float4
// // 				#pragma unroll
// // 				for (int j = 0; j < 4; j ++) {
// // 					s_eulers_head[i * 4 + j] = g_eulers[(orient_block_idx + i) * 9 + j];
// // 					s_eulers_tail[i * 4 + j] = g_eulers[(orient_block_idx + i) * 9 + 4 + j];
// // 				}
// // 			} else {
// // 				#pragma unroll
// // 				for (int j = 0; j < 4; j ++) {
// // 					s_eulers_head[i * 4 + j] = 0;
// // 					s_eulers_tail[i * 4 + j] = 0;
// // 				}
// // 			}
// // 		}

// //         // initialize shared memory to zero
// //         for (int i = tid; i < 2 * TParams::kTransBlockSize * TParams::kImgBlockSize; i += TParams::kBlockSize) {
// //             s_trans_mat_block[i] = 0.0;
// //         }
// //         for (int i = tid; i < 2 * TParams::kOrientBlockSize * TParams::kImgBlockSize; i += TParams::kBlockSize) {
// //             s_orient_mat_block[i] = 0.0;
// //         }

// //         s_trans_pow2_accumulator[tid] = 0.0;
// //         s_orient_pow2_accumulator[tid] = 0.0;
// // 		if (tid < TParams::kOrientBlockSize) {
// // 			s_orient_pow2_accumulator_bak[tid] = 0.;
// // 		}

// //         for (int i = tid; i < TParams::kImgBlockSize; i += TParams::kBlockSize) {
// //             s_corr_div_2[0][i] = 0.0;
// //             s_corr_div_2[1][i] = 0.0;
// //             s_coor_x[0][i] = 0.0;
// //             s_coor_y[0][i] = 0.0;
// //             s_coor_x[1][i] = 0.0;
// //             s_coor_y[1][i] = 0.0;
// //         }

// // 		// for (int i = tid; i < TParams::kImgBlockSize; i += TParams::kBlockSize) {
// // 		// 	// TODO: change i to 0
// // 		// 	s_fcoor_xy[0][2 * i] = (float)(i);
// // 		// 	s_fcoor_xy[0][2 * i + 1] = (float)(i);
// // 		// 	s_fcoor_xy[1][2 * i] = (float)(i);
// // 		// 	s_fcoor_xy[1][2 * i + 1] = (float)(i);
// // 		// }

// // 		flag_minus_buf[0] = 0;
// // 		flag_minus_buf[1] = 0;

// //         // read fragment_c from g_diff2s
// //         init_fragment_c();

// //         __syncthreads();

// // 		// construct_orientation_matrix(
// // 		// 	0,
// // 		// 	orient_block_idx,
// // 		// 	image_size,
// // 		// 	orientation_num,
// // 		// 	s_eulers_head,
// // 		// 	s_eulers_tail,
// // 		// 	s_fcoor_xy[0]
// // 		// );

// // 		// __syncthreads();
// // 		// return;

// // /*=============================== FOR IMAGE BLOCK ==============================*/
// // 	// 	int img_block_idx = -1;
// //     //     while (scheduler.get_current_work_next_k_block_offset(img_block_idx)) {
// // 	// 		assert(img_block_idx >= 0);
// // 	// 		// int img_iter = img_block_idx / TParams::kImgBlockSize;
// // 	// 		int k_cycle = scheduler.get_current_work_k_cycle();
// // 	// 		int k_cycle_mod2 = k_cycle % 2;
// // 	// 		int k_cycle_next_mod2 = (k_cycle + 1) % 2;
// // 	// 		if (scheduler.is_first_k_cycle()){
// // 	// 		// if (img_iter == 0) {
// // 	// 			load_coord_xy(img_block_idx, s_corr_div_2[k_cycle_mod2], s_coor_x[k_cycle_mod2], s_coor_y[k_cycle_mod2]);
// // 	// 			__syncthreads();
// // 	// 			// construct orient_mat
// // 	// 			load_orient_mat_buf(img_block_idx, s_corr_div_2[k_cycle_mod2], s_coor_x[k_cycle_mod2], s_coor_y[k_cycle_mod2], flag_minus_buf[k_cycle_mod2]);
// // 	// 		}
// // 	// 		if (! scheduler.is_last_k_cycle()) {
// // 	// 			// assert(img_block_idx + TParams::kImgBlockSize < image_size);
// // 	// 		// if (img_iter + 1 < (image_size + TParams::kImgBlockSize - 1) / TParams::kImgBlockSize) {
// // 	// 			__syncthreads();
// // 	// 			// construct trans_mat
// // 	// 			load_trans_mat(img_block_idx, s_corr_div_2[k_cycle_mod2], s_coor_x[k_cycle_mod2], s_coor_y[k_cycle_mod2]);
// // 	// 			// construct orient_mat on this iteration
// // 	// 			dump_orient_mat_shm(img_block_idx, s_corr_div_2[k_cycle_mod2], s_coor_x[k_cycle_mod2], s_coor_y[k_cycle_mod2], flag_minus_buf[k_cycle_mod2]);
// // 	// 			__syncthreads();
// // 	// 			//load coord_xy on next iteration
// // 	// 			load_coord_xy(img_block_idx + TParams::kImgBlockSize, s_corr_div_2[k_cycle_next_mod2], s_coor_x[k_cycle_next_mod2], s_coor_y[k_cycle_next_mod2]);
// // 	// 			__syncthreads();
// // 	// 			// construct orient_mat for next iteration
// // 	// 			load_orient_mat_buf(img_block_idx + TParams::kImgBlockSize, s_corr_div_2[k_cycle_next_mod2], s_coor_x[k_cycle_next_mod2], s_coor_y[k_cycle_next_mod2], flag_minus_buf[k_cycle_next_mod2]);
// // 	// 		} else {
// // 	// 			__syncthreads();
// // 	// 			// construct trans_mat
// // 	// 			load_trans_mat(img_block_idx, s_corr_div_2[k_cycle_mod2], s_coor_x[k_cycle_mod2], s_coor_y[k_cycle_mod2]);
// // 	// 			// construct orient_mat on this iteratiobn
// // 	// 			dump_orient_mat_shm(img_block_idx, s_corr_div_2[k_cycle_mod2], s_coor_x[k_cycle_mod2], s_coor_y[k_cycle_mod2], flag_minus_buf[k_cycle_mod2]);
// // 	// 			__syncthreads();
// // 	// 		}
// //     // /*=============================== COMPUTE CROSS TERM ==============================*/

// //     //         block_mma_tf32_sim_fp32<decltype(s_trans_mat_block_swizzle), decltype(s_orient_mat_block_swizzle), 
// //     //         TParams::kTransBlockSize, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize,
// //     //         TParams::kWarpTransTileSize, TParams::kWarpOrientTileSize, 2 * TParams::kWarpImgTileSize,
// //     //         TParams::kMmaTransTileSize, TParams::kMmaOrientTileSize, TParams::kMmaImgTileSize>(
// //     //             fragment_c, s_trans_mat_block_swizzle, s_orient_mat_block_swizzle, warp_id, lane_id);

// //     //     } // end of image block

// // 		int k_cycle;
// //         while (scheduler.get_current_work_next_k_cycle(k_cycle)) {
// // 			// assert(img_block_idx >= 0);
// // 			// int img_iter = img_block_idx / TParams::kImgBlockSize;
// // 			// int k_cycle = scheduler.get_current_work_k_cycle();

// // 			__syncthreads();
// // 			if (k_cycle > scheduler.get_current_work_k_cycle_start()) {
// // 				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle);
// // 				int k_cycle_mod2 = scheduler.k_cycle_mod<2>(k_cycle);
// // 				load_trans_mat(img_block_idx, s_corr_div_2[k_cycle_mod2], s_coor_x[k_cycle_mod2], s_coor_y[k_cycle_mod2]);
// // 				if (bid == 0 && tid == 0) {
// // 					printf("k_cycle : %3d  load_trans_mat       img_block_idx : %3d k_cycle_mod2 : %2d\n", k_cycle, img_block_idx, k_cycle_mod2);
// // 				}
// // 			}
// // 			__syncthreads();
// // 			if (k_cycle > scheduler.get_current_work_k_cycle_start()) {
// // 				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle);
// // 				int k_cycle_mod2 = scheduler.k_cycle_mod<2>(k_cycle);
// // 				dump_orient_mat_shm(img_block_idx, s_corr_div_2[k_cycle_mod2], s_coor_x[k_cycle_mod2], s_coor_y[k_cycle_mod2], flag_minus_buf[k_cycle_mod2]);
// // 				if (bid == 0 && tid == 0) {
// // 					printf("k_cycle : %3d  dump_orient_mat_shm  img_block_idx : %3d k_cycle_mod2 : %2d\n", k_cycle, img_block_idx, k_cycle_mod2);
// // 				}
// // 			}

// // 			__syncthreads();
// // 			if (k_cycle > scheduler.get_current_work_k_cycle_start()) {
// // 				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle);
// // 				int k_cycle_mod2 = scheduler.k_cycle_mod<2>(k_cycle);
// // 				orientation_matrix_handler.sync_and_store_orientation_matrix_with_reduce(
// // 					s_orient_real_mat_block_swizzle_bak,
// // 					s_orient_imag_mat_block_swizzle_bak,
// // 					s_orient_pow2_accumulator_bak,
// // 					s_corr_div_2[k_cycle_mod2],
// // 					warp_id,
// // 					lane_id
// // 				);

// // 				if (bid == 0 && tid == 0) {
// // 					printf("k_cycle : %3d  sync_and_store_orientation_matrix_with_reduce  img_block_idx : %3d k_cycle_mod2 : %2d\n", k_cycle, img_block_idx, k_cycle_mod2);
// // 				}
// // 				__syncthreads();
// // 				// if (bid == 0) {
// // 				// 	if (tid == 0) {
// // 				// 		printf("s_orient_mat_block_swizzle: \n");
// // 				// 		s_orient_mat_block_swizzle.print_logical_memory();
// // 				// 		printf("\ns_orient_mat_block_swizzle_bak: \n");
// // 				// 		s_orient_mat_block_swizzle_bak.print_logical_memory();
// // 				// 	}
// // 				// }
// // 				if (tid == 0) {
// // 					for (int ii = 0; ii < TParams::kOrientBlockSize; ii ++) {
// // 						for (int jj = 0; jj < TParams::kImgBlockSize * 2; jj ++) {
// // 							assert(s_orient_mat_block_swizzle(ii, jj) == s_orient_mat_block_swizzle_bak(ii, jj));
// // 						}
// // 					}
// // 				}
// // 				__syncthreads();
// // 			}

// // 			__syncthreads();
// // 			if (k_cycle < scheduler.get_current_work_k_cycle_end() - 1) {
// // 				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle + 1);
// // 				int k_cycle_next_mod2 = scheduler.k_cycle_mod<2>(k_cycle + 1);
// // 				load_coord_xy(img_block_idx, s_corr_div_2[k_cycle_next_mod2], s_coor_x[k_cycle_next_mod2], s_coor_y[k_cycle_next_mod2], s_fcoor_xy[k_cycle_next_mod2]);
// // 				if (bid == 0 && tid == 0) {
// // 					printf("k_cycle : %3d  load_coord_xy        img_block_idx : %3d k_cycle_next_mod2 : %2d\n", k_cycle, img_block_idx, k_cycle_next_mod2);
// // 				}
// // 			}
// // 			__syncthreads();
// // 			if (k_cycle < scheduler.get_current_work_k_cycle_end() - 1) {
// // 				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle + 1);
// // 				int k_cycle_next_mod2 = scheduler.k_cycle_mod<2>(k_cycle + 1);

// // 				load_orient_mat_buf(img_block_idx, s_corr_div_2[k_cycle_next_mod2], s_coor_x[k_cycle_next_mod2], s_coor_y[k_cycle_next_mod2], flag_minus_buf[k_cycle_next_mod2]);
// // 				if (bid == 0 && tid == 0) {
// // 					printf("k_cycle : %3d  load_orient_mat_buf  img_block_idx : %3d k_cycle_next_mod2 : %2d\n", k_cycle, img_block_idx, k_cycle_next_mod2);
// // 				}
// // 			}
// // 			__syncthreads();
// // 			if (k_cycle < scheduler.get_current_work_k_cycle_end() - 1) {
// // 				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle + 1);
// // 				int k_cycle_next_mod2 = scheduler.k_cycle_mod<2>(k_cycle + 1);
// // 				orientation_matrix_handler.process_and_prefetch_orientation_matrix(
// // 					projector,
// // 					s_eulers_head,
// // 					s_eulers_tail,
// // 					s_fcoor_xy[k_cycle_next_mod2],
// // 					img_block_idx,
// // 					orient_block_idx,
// // 					warp_id,
// // 					lane_id
// // 				);
// // 				if (bid == 0 && tid == 0) {
// // 					printf("k_cycle : %3d  process_and_prefetch_orientation_matrix  img_block_idx : %3d k_cycle_next_mod2 : %2d\n", k_cycle, img_block_idx, k_cycle_next_mod2);
// // 				}
// // 			}
			
			
// // 			__syncthreads();
// // 			if (k_cycle > scheduler.get_current_work_k_cycle_start()) {
// // 				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle);
// // 				int k_cycle_mod2 = scheduler.k_cycle_mod<2>(k_cycle);

// // 				if (bid == 0 && tid == 0) {
// // 					printf("k_cycle : %3d  matmul               img_block_idx : %3d k_cycle_mod2 : %2d\n", k_cycle, img_block_idx, k_cycle_mod2);
// // 				}
// //     /*=============================== COMPUTE CROSS TERM ==============================*/
// // 				block_mma_tf32_sim_fp32<decltype(s_trans_mat_block_swizzle), decltype(s_orient_mat_block_swizzle), 
// // 				TParams::kTransBlockSize, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize,
// // 				TParams::kWarpTransTileSize, TParams::kWarpOrientTileSize, 2 * TParams::kWarpImgTileSize,
// // 				TParams::kMmaTransTileSize, TParams::kMmaOrientTileSize, TParams::kMmaImgTileSize>(
// // 					fragment_c, s_trans_mat_block_swizzle, s_orient_mat_block_swizzle, warp_id, lane_id);
// // 			}

// //         } // end of image block

// //         // reduce s_trans_pow2_accumulator
// //         for (int i = 1; i < TParams::kBlockSize / TParams::kTransBlockSize; ++i) {
// //             if (tid < TParams::kTransBlockSize) {
// //                 s_trans_pow2_accumulator[tid] += s_trans_pow2_accumulator[i * TParams::kTransBlockSize + tid];
// //             }
// //         }
// //         // reduce s_orient_pow2_accumulator
// //         for (int i = 1; i < TParams::kBlockSize / TParams::kOrientBlockSize; ++i) {
// //             if (tid < TParams::kOrientBlockSize) {
// //                 s_orient_pow2_accumulator[tid] += s_orient_pow2_accumulator[i * TParams::kOrientBlockSize + tid];
// //             }
// //         }

// // 		// __syncthreads();
// // 		// if (tid == 0) {
// // 		// 	for (int i = 0; i < TParams::kOrientBlockSize; i ++) {
// // 		// 		if (s_orient_pow2_accumulator[i] != s_orient_pow2_accumulator_bak[i]) {
// // 		// 			printf("s_orient_pow2_accumulator[%d] : %f != s_orient_pow2_accumulator_bak[%d] : %f\n", i, s_orient_pow2_accumulator[i], i, s_orient_pow2_accumulator_bak[i]);
// // 		// 		}
// // 		// 		assert(s_orient_pow2_accumulator[i] == s_orient_pow2_accumulator_bak[i]);
// // 		// 	}
// // 		// }

// //     /*=============================== REDUCE IN FRAGMENT_C ==============================*/
// //         __syncthreads();
        
// //         #pragma unroll
// //         for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// //             #pragma unroll
// //             for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// //                 #pragma unroll
// //                 for (int k = 0; k < kFragmentCSize; ++k) {
// //                     int m = fragment_c_m_idx_in_block<TParams>(warp_id, lane_id, i, k);
// //                     int n = fragment_c_n_idx_in_block<TParams>(warp_id, lane_id, j, k);
// //                     fragment_c[i][j][k] += s_trans_pow2_accumulator[m] + s_orient_pow2_accumulator[n];
// //                 }
// //             }
// //         }
// //         __syncthreads();

// //     /*=============================== WRITE BACK ==============================*/
// // 		epilogue();

// //         scheduler.advance_to_next_work();
// //     } // end of while has_work
// // }




// // construct orientation matrix (use z-order)
// // tex overlap exp(with overlap)
// template<typename TParams>
// __launch_bounds__(128, 2)
// // __launch_bounds__(128, 4)
// __global__ void cuda_kernel_coarse_matrixV7(
// 	XFLOAT *g_eulers,
// 	XFLOAT *trans_x,
// 	XFLOAT *trans_y,
// 	XFLOAT *g_real,
// 	XFLOAT *g_imag,
// 	AccProjectorKernel projector,
// 	XFLOAT *g_corr,
// 	XFLOAT *g_diff2s,
// 	XFLOAT *g_diff2s_opt,
// 	const int translation_num,
// 	const int orientation_num,
// 	const int image_size,
// 	XFLOAT *g_trans_real_m,
// 	XFLOAT *g_trans_imag_m,
// 	XFLOAT *g_orient_real_m,
// 	XFLOAT *g_orient_imag_m) {
// 	static_assert(TParams::kBlockSize % 32 == 0, "kBlockSize must be a multiple of 32");
// 	static_assert(TParams::kImgBlockSize == TParams::kWarpImgTileSize, "kImgBlockSize must be equal to kWarpImgTileSize");
// 	static_assert(TParams::kBlockSize >= TParams::kTransBlockSize, "kBlockSize must be greater than or equal to kTransBlockSize");
// 	// static_assert(TParams::kBlockSize >= TParams::kOrientBlockSize, "kBlockSize must be greater than or equal to kOrientBlockSize");

// 	static_assert(TParams::kTransBlockSize % TParams::kWarpTransTileSize == 0, "kTransBlockSize must be a multiple of kWarpTransTileSize");
// 	static_assert(TParams::kOrientBlockSize % TParams::kWarpOrientTileSize == 0, "kOrientBlockSize must be a multiple of kWarpOrientTileSize");
// 	static_assert(TParams::kTransBlockSize % TParams::kMmaTransTileSize == 0, "kTransBlockSize must be a multiple of kMmaTransTileSize");
// 	static_assert(TParams::kOrientBlockSize % TParams::kMmaOrientTileSize == 0, "kOrientBlockSize must be a multiple of kMmaOrientTileSize");
// 	static_assert(TParams::kBlockSize / 32 == (TParams::kTransBlockSize / TParams::kWarpTransTileSize) * (TParams::kOrientBlockSize / TParams::kWarpOrientTileSize), "kBlockSize must be equal to the product of the number of warps in translation, orientation and image dimension");

// 	static_assert(TParams::kImgBlockSize == 16, "kImgBlockSize must be 16");

// 	const int tid = threadIdx.x;          // thread id in a block
// 	const int bid = blockIdx.x;           // block id in a grid
// 	const int warp_id  = tid / 32;        // warp id in a block
// 	constexpr int kWarpNum = TParams::kBlockSize / 32; // number of warps in a block
// 	const int lane_id  = tid % 32;        // thread id in a warp

// 	const int trans_block_num = (translation_num + TParams::kTransBlockSize - 1) / TParams::kTransBlockSize;
// 	const int orient_block_num = (orientation_num + TParams::kOrientBlockSize - 1) / TParams::kOrientBlockSize;

// 	int trans_block_idx = (bid % trans_block_num) * TParams::kTransBlockSize;
// 	int orient_block_idx = (bid / trans_block_num) * TParams::kOrientBlockSize;
//     // int trans_block_idx = 0; // forware declaration
//     // int orient_block_idx = 0;

// 	assert(blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0);
	
// 	CoarseScheduler<TParams::kTransBlockSize, 
// 					TParams::kOrientBlockSize, 
// 					TParams::kImgBlockSize, 
// 					CoarseSchedulerStrategy::SplitK,
// 					2>
// 		scheduler(translation_num, orientation_num, image_size);
	
// 	// OrientationMatrixHandler<TParams::kOrientBlockSize,
// 	// 						 TParams::kImgBlockSize,
// 	// 						 kWarpNum,
// 	// 						 SharedMemorySwizzle<float, TParams::kOrientBlockSize, TParams::kImgBlockSize, 0>,
// 	// 						 SharedMemorySwizzle<float, TParams::kOrientBlockSize, TParams::kImgBlockSize, TParams::kImgBlockSize>>
// 	// 	orientation_matrix_handler(image_size, orientation_num);


// 	__shared__ XFLOAT s_trans_mat_block[2 * TParams::kTransBlockSize * TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_orient_mat_block[2 * TParams::kOrientBlockSize * TParams::kImgBlockSize];
	
// 	using TransMatLayout = SharedMemorySwizzle<float, TParams::kTransBlockSize, 2 * TParams::kImgBlockSize, 0>;
// 	using TransRealMatLayout = SharedMemorySwizzle<float, TParams::kTransBlockSize, TParams::kImgBlockSize, 0>;
// 	using TransImagMatLayout = SharedMemorySwizzle<float, TParams::kTransBlockSize, TParams::kImgBlockSize, TParams::kImgBlockSize>;
	
// 	using OrientMatLayout = SharedMemorySwizzle<float, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize, 0>;
// 	using OrientRealMatLayout = SharedMemorySwizzle<float, TParams::kOrientBlockSize, TParams::kImgBlockSize, 0>;
// 	using OrientImagMatLayout = SharedMemorySwizzle<float, TParams::kOrientBlockSize, TParams::kImgBlockSize, TParams::kImgBlockSize>;

// 	TransMatLayout s_trans_mat_block_swizzle(s_trans_mat_block);
// 	TransRealMatLayout s_trans_real_mat_block_swizzle(s_trans_mat_block);
// 	TransImagMatLayout s_trans_imag_mat_block_swizzle(s_trans_mat_block);

// 	OrientMatLayout s_orient_mat_block_swizzle(s_orient_mat_block);
// 	OrientRealMatLayout s_orient_real_mat_block_swizzle(s_orient_mat_block);
// 	OrientImagMatLayout s_orient_imag_mat_block_swizzle(s_orient_mat_block);

	
// 	OrientationMatrixHandler<TParams::kOrientBlockSize,
// 							TParams::kImgBlockSize,
// 							kWarpNum,
// 							OrientRealMatLayout,
// 							OrientImagMatLayout,
// 							TransMatLayout, OrientMatLayout, 
// 							TParams::kTransBlockSize, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize,
// 							TParams::kWarpTransTileSize, TParams::kWarpOrientTileSize, 2 * TParams::kWarpImgTileSize,
// 							TParams::kMmaTransTileSize, TParams::kMmaOrientTileSize, TParams::kMmaImgTileSize>
// 		orientation_matrix_handler(image_size, orientation_num);

// 	// double buffer for s_corr_div_2, s_coor_x, s_coor_y
// 	__shared__ XFLOAT s_corr_div_2[2][TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_coor_x[2][TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_coor_y[2][TParams::kImgBlockSize];

// 	// ============================  new  ============================
// 	__shared__ XFLOAT s_fcoor_xy[2][TParams::kImgBlockSize * 2]; // img -> x,y
// 	// For a 2D scenario, e8 is not used, so it’s not stored in shared memory.
// 	// e2 and e5 are also unused, but they remain in shared memory for alignment.
// 	// __shared__ XFLOAT s_eulers_head[TParams::kOrientBlockSize * 4]; // e0 e1 e2 e3
// 	// __shared__ XFLOAT s_eulers_tail[TParams::kOrientBlockSize * 4]; // e4 e5 e6 e7
// 	__shared__ XFLOAT s_eulers_scaled_head[TParams::kOrientBlockSize * 4]; // (e0 e1 e2 e3)  * projector.padding_factor
// 	__shared__ XFLOAT s_eulers_scaled_tail[TParams::kOrientBlockSize * 4]; // (e4 e5 e6 e7)  * projector.padding_factor

// 	// reduce buffer
// 	__shared__ XFLOAT s_trans_pow2_accumulator[(TParams::kBlockSize / TParams::kTransBlockSize) * TParams::kTransBlockSize];
// 	// __shared__ XFLOAT s_orient_pow2_accumulator[(TParams::kBlockSize / TParams::kOrientBlockSize) * TParams::kOrientBlockSize];

// 	__shared__ XFLOAT s_orient_pow2_accumulator[TParams::kOrientBlockSize];
// 	// register
// 	constexpr int kNumMmaTransInWarpTile = TParams::kWarpTransTileSize / TParams::kMmaTransTileSize;
// 	constexpr int kNumMmaOrientInWarpTile = TParams::kWarpOrientTileSize / TParams::kMmaOrientTileSize;
// 	constexpr int kNumMmaImgInWarpTile = TParams::kWarpImgTileSize / TParams::kMmaImgTileSize;

// 	constexpr int kFragmentASize = TParams::kMmaTransTileSize * TParams::kMmaImgTileSize / kWarpSize;
// 	constexpr int kFragmentBSize = TParams::kMmaOrientTileSize * TParams::kMmaImgTileSize / kWarpSize;
// 	constexpr int kFragmentCSize = TParams::kMmaTransTileSize * TParams::kMmaOrientTileSize / kWarpSize;

// 	XFLOAT fragment_c[kNumMmaTransInWarpTile][kNumMmaOrientInWarpTile][kFragmentCSize];

// 	constexpr int kRegistersMmaPerThread = kNumMmaTransInWarpTile * kFragmentASize +
// 										   kNumMmaOrientInWarpTile * kFragmentBSize +
// 										   kNumMmaTransInWarpTile * kNumMmaOrientInWarpTile * kFragmentCSize;
	
// 	#if kRegistersMmaPerThread >= 256
// 	#warning "kRegistersMmaPerThread must be less than or equal to 256, otherwise register spilling will occur"
// 	#endif


// 	// ============================= lambda function =============================
// 	//given current img_block_idx, load global array into corr_div_2, coord_x, coord_y
// 	auto load_coord_xy = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y, XFLOAT* fcoor_xy) {
// 		#pragma unroll 
// 		for (int i = tid; i < TParams::kImgBlockSize; i += TParams::kBlockSize) {
// 			if (img_block_idx + i < image_size) {
// 				corr_div_2[i] = g_corr[img_block_idx + i] / 2;
// 			} else {
// 				corr_div_2[i] = 0.;
// 			}

// 			int x, y;
// 			pixel_index2coor(img_block_idx + i, projector.imgX, projector.imgY, projector.maxR, x, y);
// 			coord_x[i] = x;
// 			coord_y[i] = y;
			
// 			fcoor_xy[2 * i + 0] = x;
// 			fcoor_xy[2 * i + 1] = y;
// 		}
// 	};

// 	//given current img_block_idx, corr_div_2, coord_x, coord_y, load trans mat into s_trans_real_mat_block and s_trans_imag_mat_block
// 	auto load_trans_mat = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y) {
// 		assert(TParams::kBlockSize % TParams::kTransBlockSize == 0);
// 		#pragma unroll
// 		for (int i = tid / TParams::kTransBlockSize; i < TParams::kImgBlockSize; i += TParams::kBlockSize / TParams::kTransBlockSize) {
// 			int g_img_idx = img_block_idx + i;
// 			int trans_idx = tid % TParams::kTransBlockSize;
// 			if (g_img_idx >= image_size) {
// 				assert(trans_idx < TParams::kTransBlockSize);
// 				assert(i < TParams::kImgBlockSize);
// 				s_trans_real_mat_block_swizzle(trans_idx, i) = 0.;
// 				s_trans_imag_mat_block_swizzle(trans_idx, i) = 0.;
// 				continue;
// 			}
// 			int g_trans_idx = trans_block_idx + trans_idx;
// 			if (g_trans_idx >= translation_num) {
// 				continue;
// 			}
// 			XFLOAT tx = trans_x[g_trans_idx];
// 			XFLOAT ty = trans_y[g_trans_idx];
// 			XFLOAT real = g_real[g_img_idx];
// 			XFLOAT imag = g_imag[g_img_idx];

// 			int x = coord_x[i];
// 			int y = coord_y[i];
// 			XFLOAT trans_real, trans_imag;
// 			translatePixel(x, y, tx, ty, real, imag, trans_real, trans_imag);

// 			// s_trans_real_mat_block[trans_idx * TParams::kImgBlockSize + i] = -2 * trans_real * corr_div_2[i];
// 			// s_trans_imag_mat_block[trans_idx * TParams::kImgBlockSize + i] = -2 * trans_imag * corr_div_2[i];
// 			s_trans_real_mat_block_swizzle(trans_idx, i) = -2 * trans_real * corr_div_2[i];
// 			s_trans_imag_mat_block_swizzle(trans_idx, i) = -2 * trans_imag * corr_div_2[i];

// 			XFLOAT magnitude_squared_sum = trans_real * trans_real * corr_div_2[i] + trans_imag * trans_imag * corr_div_2[i];
// 			s_trans_pow2_accumulator[tid] += magnitude_squared_sum;
// 		}
// 	};

//     auto init_fragment_c = [&] () {
// 		// Default: need read from g_diff2s
// 		if (scheduler.get_strategy() == CoarseSchedulerStrategy::Default || 
// 		   (scheduler.get_strategy() == CoarseSchedulerStrategy::SplitK && g_diff2s != g_diff2s_opt)) {		
// 			#pragma unroll
// 			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 				#pragma unroll
// 				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 					#pragma unroll
// 					for (int k = 0; k < kFragmentCSize; ++k) {
// 						int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
// 						int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);
// 						if (m < translation_num && n < orientation_num) {
// 							fragment_c[i][j][k] = g_diff2s[n * translation_num + m];
// 						} else {
// 							fragment_c[i][j][k] = 0.0;
// 						}
// 					}
// 				}
// 			}
// 		}
// 		// SplitK: use atomicAdd to accumulate, if diff2s source == diff2s dest, no need to read from g_diff2s
// 		// else, read from g_diff2s
// 		else if (scheduler.get_strategy() == CoarseSchedulerStrategy::SplitK && g_diff2s == g_diff2s_opt) {
// 			#pragma unroll
// 			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 				#pragma unroll
// 				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 					#pragma unroll
// 					for (int k = 0; k < kFragmentCSize; ++k) {
// 						fragment_c[i][j][k] = 0.0;
// 					}
// 				}
// 			}
			
// 		}
// 		else {
// 			assert(false);
// 		}
//     };

// 	auto epilogue = [&] () {
// 		// write fragment_c back to g_diff2s_opt
//         if (scheduler.get_strategy() == CoarseSchedulerStrategy::Default) {
// 			#pragma unroll
// 			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 				#pragma unroll
// 				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 					#pragma unroll
// 					for (int k = 0; k < kFragmentCSize; ++k) {
// 						int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
// 						int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);

// 						if (m < translation_num && n < orientation_num) {
// 							g_diff2s_opt[n * translation_num + m] = fragment_c[i][j][k];
// 						}
// 					}
// 				}
// 			}
// 		} else if (scheduler.get_strategy() == CoarseSchedulerStrategy::SplitK) {
// 			// use atomic add
// 			#pragma unroll
// 			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 				#pragma unroll
// 				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 					#pragma unroll
// 					for (int k = 0; k < kFragmentCSize; ++k) {
// 						int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
// 						int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);

// 						if (m < translation_num && n < orientation_num) {
// 							atomicAdd(&g_diff2s_opt[n * translation_num + m], fragment_c[i][j][k]);
// 						}
// 					}
// 				}
// 			}
// 		} else {
// 			assert(false);
// 		}
// 	};

//     // =====================================================================
// 	// ============================= main loop =============================
//     // =====================================================================
//     while (scheduler.has_work()) {
// 		__syncthreads();

//         trans_block_idx = scheduler.get_current_work_m_block_offset();
//         orient_block_idx = scheduler.get_current_work_n_block_offset();
// 		// if (tid == 0) {
// 		// 	printf("bid : %3d  tb_idx : %6d ob_idx : %6d\n", bid, trans_block_idx, orient_block_idx);
// 		// }
		
// 		// if (tid == 0) {
// 		// 	printf("bid : %3d  tb_idx : %6d ob_idx : %6d\n", bid, trans_block_idx, orient_block_idx);
// 		// }
// 		// load eulers to smem
// 		#pragma unroll
// 		for (int i = tid; i < TParams::kOrientBlockSize; i += TParams::kBlockSize) {
// 			if (orient_block_idx + i < orientation_num) {
// 				// TODO: check whether compiler uses load float4
// 				#pragma unroll
// 				for (int j = 0; j < 4; j ++) {
// 					s_eulers_scaled_head[i * 4 + j] = g_eulers[(orient_block_idx + i) * 9 + j] * projector.padding_factor;
// 					s_eulers_scaled_tail[i * 4 + j] = g_eulers[(orient_block_idx + i) * 9 + 4 + j] * projector.padding_factor;
// 				}
// 			} else {
// 				#pragma unroll
// 				for (int j = 0; j < 4; j ++) {
// 					s_eulers_scaled_head[i * 4 + j] = 0;
// 					s_eulers_scaled_tail[i * 4 + j] = 0;
// 				}
// 			}
// 		}

//         // initialize shared memory to zero
//         for (int i = tid; i < 2 * TParams::kTransBlockSize * TParams::kImgBlockSize; i += TParams::kBlockSize) {
//             s_trans_mat_block[i] = 0.0;
//         }
//         for (int i = tid; i < 2 * TParams::kOrientBlockSize * TParams::kImgBlockSize; i += TParams::kBlockSize) {
//             s_orient_mat_block[i] = 0.0;
//         }

//         s_trans_pow2_accumulator[tid] = 0.0;
//         // s_orient_pow2_accumulator[tid] = 0.0;
// 		if (tid < TParams::kOrientBlockSize) {
// 			s_orient_pow2_accumulator[tid] = 0.;
// 		}

//         for (int i = tid; i < TParams::kImgBlockSize; i += TParams::kBlockSize) {
//             s_corr_div_2[0][i] = 0.0;
//             s_corr_div_2[1][i] = 0.0;
//         }

//         // read fragment_c from g_diff2s
//         init_fragment_c();

//         __syncthreads();

// /*=============================== FOR IMAGE BLOCK ==============================*/
// 		int k_cycle;
//         while (scheduler.get_current_work_next_k_cycle(k_cycle)) {
// 			__syncthreads();
// 			if (k_cycle > scheduler.get_current_work_k_cycle_start()) {
// 				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle);
// 				int k_cycle_mod2 = scheduler.k_cycle_mod<2>(k_cycle);
// 				load_trans_mat(img_block_idx, s_corr_div_2[k_cycle_mod2], s_coor_x[k_cycle_mod2], s_coor_y[k_cycle_mod2]);
// 				// if (bid == 0 && tid == 0) {
// 				// 	printf("k_cycle : %3d  load_trans_mat       img_block_idx : %3d k_cycle_mod2 : %2d\n", k_cycle, img_block_idx, k_cycle_mod2);
// 				// }
// 			}

// 			__syncthreads();
// 			if (k_cycle > scheduler.get_current_work_k_cycle_start()) {
// 				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle);
// 				int k_cycle_mod2 = scheduler.k_cycle_mod<2>(k_cycle);
// 				orientation_matrix_handler.sync_and_store_orientation_matrix_with_reduce(
// 					s_orient_real_mat_block_swizzle,
// 					s_orient_imag_mat_block_swizzle,
// 					s_orient_pow2_accumulator,
// 					s_corr_div_2[k_cycle_mod2],
// 					warp_id,
// 					lane_id
// 				);
// 				// if (bid == 0 && tid == 0) {
// 				// 	printf("k_cycle : %3d  sync_and_store_orientation_matrix_with_reduce  img_block_idx : %3d k_cycle_mod2 : %2d\n", k_cycle, img_block_idx, k_cycle_mod2);
// 				// }
// 			}

// 			__syncthreads();
// 			if (k_cycle < scheduler.get_current_work_k_cycle_end() - 1) {
// 				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle + 1);
// 				int k_cycle_next_mod2 = scheduler.k_cycle_mod<2>(k_cycle + 1);
// 				load_coord_xy(img_block_idx, s_corr_div_2[k_cycle_next_mod2], s_coor_x[k_cycle_next_mod2], s_coor_y[k_cycle_next_mod2], s_fcoor_xy[k_cycle_next_mod2]);
// 				// if (bid == 0 && tid == 0) {
// 				// 	printf("k_cycle : %3d  load_coord_xy        img_block_idx : %3d k_cycle_next_mod2 : %2d\n", k_cycle, img_block_idx, k_cycle_next_mod2);
// 				// }
// 			}

// 			__syncthreads();
// 			if (k_cycle < scheduler.get_current_work_k_cycle_end() - 1) {
// 				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle + 1);
// 				int k_cycle_next_mod2 = scheduler.k_cycle_mod<2>(k_cycle + 1);
// 				orientation_matrix_handler.process_and_prefetch_orientation_matrix(
// 					projector,
// 					s_eulers_scaled_head,
// 					s_eulers_scaled_tail,
// 					s_fcoor_xy[k_cycle_next_mod2],
// 					img_block_idx,
// 					orient_block_idx,
// 					warp_id,
// 					lane_id
// 				);
// 				// if (bid == 0 && tid == 0) {
// 				// 	printf("k_cycle : %3d  process_and_prefetch_orientation_matrix  img_block_idx : %3d k_cycle_next_mod2 : %2d\n", k_cycle, img_block_idx, k_cycle_next_mod2);
// 				// }
// 			}
			
			
// 			__syncthreads();
// 			if (k_cycle > scheduler.get_current_work_k_cycle_start()) {
// 				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle);
// 				int k_cycle_mod2 = scheduler.k_cycle_mod<2>(k_cycle);

// 				// if (bid == 0 && tid == 0) {
// 				// 	printf("k_cycle : %3d  matmul               img_block_idx : %3d k_cycle_mod2 : %2d\n", k_cycle, img_block_idx, k_cycle_mod2);
// 				// }
//     /*=============================== COMPUTE CROSS TERM ==============================*/
// 				block_mma_tf32_sim_fp32<decltype(s_trans_mat_block_swizzle), decltype(s_orient_mat_block_swizzle), 
// 				TParams::kTransBlockSize, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize,
// 				TParams::kWarpTransTileSize, TParams::kWarpOrientTileSize, 2 * TParams::kWarpImgTileSize,
// 				TParams::kMmaTransTileSize, TParams::kMmaOrientTileSize, TParams::kMmaImgTileSize>(
// 					fragment_c, s_trans_mat_block_swizzle, s_orient_mat_block_swizzle, warp_id, lane_id);
// 			}

//         } // end of image block

//         // reduce s_trans_pow2_accumulator
//         for (int i = 1; i < TParams::kBlockSize / TParams::kTransBlockSize; ++i) {
//             if (tid < TParams::kTransBlockSize) {
//                 s_trans_pow2_accumulator[tid] += s_trans_pow2_accumulator[i * TParams::kTransBlockSize + tid];
//             }
//         }
//         // reduce s_orient_pow2_accumulator
//         // for (int i = 1; i < TParams::kBlockSize / TParams::kOrientBlockSize; ++i) {
//         //     if (tid < TParams::kOrientBlockSize) {
//         //         s_orient_pow2_accumulator[tid] += s_orient_pow2_accumulator[i * TParams::kOrientBlockSize + tid];
//         //     }
//         // }

//     /*=============================== REDUCE IN FRAGMENT_C ==============================*/
//         __syncthreads();
        
//         #pragma unroll
//         for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
//             #pragma unroll
//             for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
//                 #pragma unroll
//                 for (int k = 0; k < kFragmentCSize; ++k) {
//                     int m = fragment_c_m_idx_in_block<TParams>(warp_id, lane_id, i, k);
//                     int n = fragment_c_n_idx_in_block<TParams>(warp_id, lane_id, j, k);
//                     fragment_c[i][j][k] += s_trans_pow2_accumulator[m] + s_orient_pow2_accumulator[n];
//                 }
//             }
//         }
//         __syncthreads();

//     /*=============================== WRITE BACK ==============================*/
// 		epilogue();

//         scheduler.advance_to_next_work();
//     } // end of while has_work
// }



// // construct orientation matrix (use z-order)
// // tex overlap exp(without overlap)
// template<typename TParams>
// __launch_bounds__(128, 2)
// // __launch_bounds__(128, 4)
// __global__ void cuda_kernel_coarse_matrixV8(
// 	XFLOAT *g_eulers,
// 	XFLOAT *trans_x,
// 	XFLOAT *trans_y,
// 	XFLOAT *g_real,
// 	XFLOAT *g_imag,
// 	AccProjectorKernel projector,
// 	XFLOAT *g_corr,
// 	XFLOAT *g_diff2s,
// 	XFLOAT *g_diff2s_opt,
// 	const int translation_num,
// 	const int orientation_num,
// 	const int image_size,
// 	XFLOAT *g_trans_real_m,
// 	XFLOAT *g_trans_imag_m,
// 	XFLOAT *g_orient_real_m,
// 	XFLOAT *g_orient_imag_m) {
// 	static_assert(TParams::kBlockSize % 32 == 0, "kBlockSize must be a multiple of 32");
// 	static_assert(TParams::kImgBlockSize == TParams::kWarpImgTileSize, "kImgBlockSize must be equal to kWarpImgTileSize");
// 	static_assert(TParams::kBlockSize >= TParams::kTransBlockSize, "kBlockSize must be greater than or equal to kTransBlockSize");
// 	// static_assert(TParams::kBlockSize >= TParams::kOrientBlockSize, "kBlockSize must be greater than or equal to kOrientBlockSize");

// 	static_assert(TParams::kTransBlockSize % TParams::kWarpTransTileSize == 0, "kTransBlockSize must be a multiple of kWarpTransTileSize");
// 	static_assert(TParams::kOrientBlockSize % TParams::kWarpOrientTileSize == 0, "kOrientBlockSize must be a multiple of kWarpOrientTileSize");
// 	static_assert(TParams::kTransBlockSize % TParams::kMmaTransTileSize == 0, "kTransBlockSize must be a multiple of kMmaTransTileSize");
// 	static_assert(TParams::kOrientBlockSize % TParams::kMmaOrientTileSize == 0, "kOrientBlockSize must be a multiple of kMmaOrientTileSize");
// 	static_assert(TParams::kBlockSize / 32 == (TParams::kTransBlockSize / TParams::kWarpTransTileSize) * (TParams::kOrientBlockSize / TParams::kWarpOrientTileSize), "kBlockSize must be equal to the product of the number of warps in translation, orientation and image dimension");

// 	static_assert(TParams::kImgBlockSize == 16, "kImgBlockSize must be 16");

// 	const int tid = threadIdx.x;          // thread id in a block
// 	const int bid = blockIdx.x;           // block id in a grid
// 	const int warp_id  = tid / 32;        // warp id in a block
// 	constexpr int kWarpNum = TParams::kBlockSize / 32; // number of warps in a block
// 	const int lane_id  = tid % 32;        // thread id in a warp

// 	const int trans_block_num = (translation_num + TParams::kTransBlockSize - 1) / TParams::kTransBlockSize;
// 	const int orient_block_num = (orientation_num + TParams::kOrientBlockSize - 1) / TParams::kOrientBlockSize;

// 	int trans_block_idx = (bid % trans_block_num) * TParams::kTransBlockSize;
// 	int orient_block_idx = (bid / trans_block_num) * TParams::kOrientBlockSize;
//     // int trans_block_idx = 0; // forware declaration
//     // int orient_block_idx = 0;

// 	assert(blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0);
	
// 	CoarseScheduler<TParams::kTransBlockSize, 
// 					TParams::kOrientBlockSize, 
// 					TParams::kImgBlockSize, 
// 					CoarseSchedulerStrategy::SplitK,
// 					2>
// 		scheduler(translation_num, orientation_num, image_size);
	
// 	// OrientationMatrixHandler<TParams::kOrientBlockSize,
// 	// 						 TParams::kImgBlockSize,
// 	// 						 kWarpNum,
// 	// 						 SharedMemorySwizzle<float, TParams::kOrientBlockSize, TParams::kImgBlockSize, 0>,
// 	// 						 SharedMemorySwizzle<float, TParams::kOrientBlockSize, TParams::kImgBlockSize, TParams::kImgBlockSize>>
// 	// 	orientation_matrix_handler(image_size, orientation_num);


// 	__shared__ XFLOAT s_trans_mat_block[2 * TParams::kTransBlockSize * TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_orient_mat_block[2 * TParams::kOrientBlockSize * TParams::kImgBlockSize];
	
// 	using TransMatLayout = SharedMemorySwizzle<float, TParams::kTransBlockSize, 2 * TParams::kImgBlockSize, 0>;
// 	using TransRealMatLayout = SharedMemorySwizzle<float, TParams::kTransBlockSize, TParams::kImgBlockSize, 0>;
// 	using TransImagMatLayout = SharedMemorySwizzle<float, TParams::kTransBlockSize, TParams::kImgBlockSize, TParams::kImgBlockSize>;
	
// 	using OrientMatLayout = SharedMemorySwizzle<float, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize, 0>;
// 	using OrientRealMatLayout = SharedMemorySwizzle<float, TParams::kOrientBlockSize, TParams::kImgBlockSize, 0>;
// 	using OrientImagMatLayout = SharedMemorySwizzle<float, TParams::kOrientBlockSize, TParams::kImgBlockSize, TParams::kImgBlockSize>;

// 	TransMatLayout s_trans_mat_block_swizzle(s_trans_mat_block);
// 	TransRealMatLayout s_trans_real_mat_block_swizzle(s_trans_mat_block);
// 	TransImagMatLayout s_trans_imag_mat_block_swizzle(s_trans_mat_block);

// 	OrientMatLayout s_orient_mat_block_swizzle(s_orient_mat_block);
// 	OrientRealMatLayout s_orient_real_mat_block_swizzle(s_orient_mat_block);
// 	OrientImagMatLayout s_orient_imag_mat_block_swizzle(s_orient_mat_block);
	
// 	OrientationMatrixHandler<TParams::kOrientBlockSize,
// 							TParams::kImgBlockSize,
// 							kWarpNum,
// 							OrientRealMatLayout,
// 							OrientImagMatLayout,
// 							TransMatLayout, OrientMatLayout, 
// 							TParams::kTransBlockSize, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize,
// 							TParams::kWarpTransTileSize, TParams::kWarpOrientTileSize, 2 * TParams::kWarpImgTileSize,
// 							TParams::kMmaTransTileSize, TParams::kMmaOrientTileSize, TParams::kMmaImgTileSize>
// 		orientation_matrix_handler(image_size, orientation_num);

// 	// double buffer for s_corr_div_2, s_coor_x, s_coor_y
// 	__shared__ XFLOAT s_corr_div_2[2][TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_coor_x[2][TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_coor_y[2][TParams::kImgBlockSize];

// 	// ============================  new  ============================
// 	__shared__ XFLOAT s_fcoor_xy[2][TParams::kImgBlockSize * 2]; // img -> x,y
// 	// For a 2D scenario, e8 is not used, so it’s not stored in shared memory.
// 	// e2 and e5 are also unused, but they remain in shared memory for alignment.
// 	// __shared__ XFLOAT s_eulers_head[TParams::kOrientBlockSize * 4]; // e0 e1 e2 e3
// 	// __shared__ XFLOAT s_eulers_tail[TParams::kOrientBlockSize * 4]; // e4 e5 e6 e7
// 	__shared__ XFLOAT s_eulers_scaled_head[TParams::kOrientBlockSize * 4]; // (e0 e1 e2 e3)  * projector.padding_factor
// 	__shared__ XFLOAT s_eulers_scaled_tail[TParams::kOrientBlockSize * 4]; // (e4 e5 e6 e7)  * projector.padding_factor

// 	// reduce buffer
// 	__shared__ XFLOAT s_trans_pow2_accumulator[(TParams::kBlockSize / TParams::kTransBlockSize) * TParams::kTransBlockSize];
// 	// __shared__ XFLOAT s_orient_pow2_accumulator[(TParams::kBlockSize / TParams::kOrientBlockSize) * TParams::kOrientBlockSize];

// 	__shared__ XFLOAT s_orient_pow2_accumulator[TParams::kOrientBlockSize];
// 	// register
// 	constexpr int kNumMmaTransInWarpTile = TParams::kWarpTransTileSize / TParams::kMmaTransTileSize;
// 	constexpr int kNumMmaOrientInWarpTile = TParams::kWarpOrientTileSize / TParams::kMmaOrientTileSize;
// 	constexpr int kNumMmaImgInWarpTile = TParams::kWarpImgTileSize / TParams::kMmaImgTileSize;

// 	constexpr int kFragmentASize = TParams::kMmaTransTileSize * TParams::kMmaImgTileSize / kWarpSize;
// 	constexpr int kFragmentBSize = TParams::kMmaOrientTileSize * TParams::kMmaImgTileSize / kWarpSize;
// 	constexpr int kFragmentCSize = TParams::kMmaTransTileSize * TParams::kMmaOrientTileSize / kWarpSize;

// 	XFLOAT fragment_c[kNumMmaTransInWarpTile][kNumMmaOrientInWarpTile][kFragmentCSize];

// 	constexpr int kRegistersMmaPerThread = kNumMmaTransInWarpTile * kFragmentASize +
// 										   kNumMmaOrientInWarpTile * kFragmentBSize +
// 										   kNumMmaTransInWarpTile * kNumMmaOrientInWarpTile * kFragmentCSize;
	
// 	#if kRegistersMmaPerThread >= 256
// 	#warning "kRegistersMmaPerThread must be less than or equal to 256, otherwise register spilling will occur"
// 	#endif


// 	// ============================= lambda function =============================
// 	//given current img_block_idx, load global array into corr_div_2, coord_x, coord_y
// 	auto load_coord_xy = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y, XFLOAT* fcoor_xy) {
// 		#pragma unroll 
// 		for (int i = tid; i < TParams::kImgBlockSize; i += TParams::kBlockSize) {
// 			if (img_block_idx + i < image_size) {
// 				corr_div_2[i] = g_corr[img_block_idx + i] / 2;
// 			} else {
// 				corr_div_2[i] = 0.;
// 			}

// 			int x, y;
// 			pixel_index2coor(img_block_idx + i, projector.imgX, projector.imgY, projector.maxR, x, y);
// 			coord_x[i] = x;
// 			coord_y[i] = y;
			
// 			fcoor_xy[2 * i + 0] = x;
// 			fcoor_xy[2 * i + 1] = y;
// 		}
// 	};

// 	//given current img_block_idx, corr_div_2, coord_x, coord_y, load trans mat into s_trans_real_mat_block and s_trans_imag_mat_block
// 	auto load_trans_mat = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y) {
// 		assert(TParams::kBlockSize % TParams::kTransBlockSize == 0);
// 		#pragma unroll
// 		for (int i = tid / TParams::kTransBlockSize; i < TParams::kImgBlockSize; i += TParams::kBlockSize / TParams::kTransBlockSize) {
// 			int g_img_idx = img_block_idx + i;
// 			int trans_idx = tid % TParams::kTransBlockSize;
// 			if (g_img_idx >= image_size) {
// 				assert(trans_idx < TParams::kTransBlockSize);
// 				assert(i < TParams::kImgBlockSize);
// 				s_trans_real_mat_block_swizzle(trans_idx, i) = 0.;
// 				s_trans_imag_mat_block_swizzle(trans_idx, i) = 0.;
// 				continue;
// 			}
// 			int g_trans_idx = trans_block_idx + trans_idx;
// 			if (g_trans_idx >= translation_num) {
// 				continue;
// 			}
// 			XFLOAT tx = trans_x[g_trans_idx];
// 			XFLOAT ty = trans_y[g_trans_idx];
// 			XFLOAT real = g_real[g_img_idx];
// 			XFLOAT imag = g_imag[g_img_idx];

// 			int x = coord_x[i];
// 			int y = coord_y[i];
// 			XFLOAT trans_real, trans_imag;
// 			translatePixel(x, y, tx, ty, real, imag, trans_real, trans_imag);

// 			// s_trans_real_mat_block[trans_idx * TParams::kImgBlockSize + i] = -2 * trans_real * corr_div_2[i];
// 			// s_trans_imag_mat_block[trans_idx * TParams::kImgBlockSize + i] = -2 * trans_imag * corr_div_2[i];
// 			s_trans_real_mat_block_swizzle(trans_idx, i) = -2 * trans_real * corr_div_2[i];
// 			s_trans_imag_mat_block_swizzle(trans_idx, i) = -2 * trans_imag * corr_div_2[i];

// 			XFLOAT magnitude_squared_sum = trans_real * trans_real * corr_div_2[i] + trans_imag * trans_imag * corr_div_2[i];
// 			s_trans_pow2_accumulator[tid] += magnitude_squared_sum;
// 		}
// 	};

//     auto init_fragment_c = [&] () {
// 		// Default: need read from g_diff2s
// 		if (scheduler.get_strategy() == CoarseSchedulerStrategy::Default || 
// 		   (scheduler.get_strategy() == CoarseSchedulerStrategy::SplitK && g_diff2s != g_diff2s_opt)) {		
// 			#pragma unroll
// 			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 				#pragma unroll
// 				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 					#pragma unroll
// 					for (int k = 0; k < kFragmentCSize; ++k) {
// 						int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
// 						int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);
// 						if (m < translation_num && n < orientation_num) {
// 							fragment_c[i][j][k] = g_diff2s[n * translation_num + m];
// 						} else {
// 							fragment_c[i][j][k] = 0.0;
// 						}
// 					}
// 				}
// 			}
// 		}
// 		// SplitK: use atomicAdd to accumulate, if diff2s source == diff2s dest, no need to read from g_diff2s
// 		// else, read from g_diff2s
// 		else if (scheduler.get_strategy() == CoarseSchedulerStrategy::SplitK && g_diff2s == g_diff2s_opt) {
// 			#pragma unroll
// 			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 				#pragma unroll
// 				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 					#pragma unroll
// 					for (int k = 0; k < kFragmentCSize; ++k) {
// 						fragment_c[i][j][k] = 0.0;
// 					}
// 				}
// 			}
			
// 		}
// 		else {
// 			assert(false);
// 		}
//     };

// 	auto epilogue = [&] () {
// 		// write fragment_c back to g_diff2s_opt
//         if (scheduler.get_strategy() == CoarseSchedulerStrategy::Default) {
// 			#pragma unroll
// 			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 				#pragma unroll
// 				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 					#pragma unroll
// 					for (int k = 0; k < kFragmentCSize; ++k) {
// 						int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
// 						int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);

// 						if (m < translation_num && n < orientation_num) {
// 							g_diff2s_opt[n * translation_num + m] = fragment_c[i][j][k];
// 						}
// 					}
// 				}
// 			}
// 		} else if (scheduler.get_strategy() == CoarseSchedulerStrategy::SplitK) {
// 			// use atomic add
// 			#pragma unroll
// 			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 				#pragma unroll
// 				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 					#pragma unroll
// 					for (int k = 0; k < kFragmentCSize; ++k) {
// 						int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
// 						int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);

// 						if (m < translation_num && n < orientation_num) {
// 							atomicAdd(&g_diff2s_opt[n * translation_num + m], fragment_c[i][j][k]);
// 						}
// 					}
// 				}
// 			}
// 		} else {
// 			assert(false);
// 		}
// 	};

//     // =====================================================================
// 	// ============================= main loop =============================
//     // =====================================================================
//     while (scheduler.has_work()) {
// 		__syncthreads();

//         trans_block_idx = scheduler.get_current_work_m_block_offset();
//         orient_block_idx = scheduler.get_current_work_n_block_offset();
// 		// if (tid == 0) {
// 		// 	printf("bid : %3d  tb_idx : %6d ob_idx : %6d\n", bid, trans_block_idx, orient_block_idx);
// 		// }
		
// 		// if (tid == 0) {
// 		// 	printf("bid : %3d  tb_idx : %6d ob_idx : %6d\n", bid, trans_block_idx, orient_block_idx);
// 		// }
// 		// load eulers to smem
// 		#pragma unroll
// 		for (int i = tid; i < TParams::kOrientBlockSize; i += TParams::kBlockSize) {
// 			if (orient_block_idx + i < orientation_num) {
// 				// TODO: check whether compiler uses load float4
// 				#pragma unroll
// 				for (int j = 0; j < 4; j ++) {
// 					s_eulers_scaled_head[i * 4 + j] = g_eulers[(orient_block_idx + i) * 9 + j] * projector.padding_factor;
// 					s_eulers_scaled_tail[i * 4 + j] = g_eulers[(orient_block_idx + i) * 9 + 4 + j] * projector.padding_factor;
// 				}
// 			} else {
// 				#pragma unroll
// 				for (int j = 0; j < 4; j ++) {
// 					s_eulers_scaled_head[i * 4 + j] = 0;
// 					s_eulers_scaled_tail[i * 4 + j] = 0;
// 				}
// 			}
// 		}

//         // initialize shared memory to zero
//         for (int i = tid; i < 2 * TParams::kTransBlockSize * TParams::kImgBlockSize; i += TParams::kBlockSize) {
//             s_trans_mat_block[i] = 0.0;
//         }
//         for (int i = tid; i < 2 * TParams::kOrientBlockSize * TParams::kImgBlockSize; i += TParams::kBlockSize) {
//             s_orient_mat_block[i] = 0.0;
//         }

//         s_trans_pow2_accumulator[tid] = 0.0;
//         // s_orient_pow2_accumulator[tid] = 0.0;
// 		if (tid < TParams::kOrientBlockSize) {
// 			s_orient_pow2_accumulator[tid] = 0.;
// 		}

//         for (int i = tid; i < TParams::kImgBlockSize; i += TParams::kBlockSize) {
//             s_corr_div_2[0][i] = 0.0;
//             s_corr_div_2[1][i] = 0.0;
//         }

//         // read fragment_c from g_diff2s
//         init_fragment_c();

//         __syncthreads();

// /*=============================== FOR IMAGE BLOCK ==============================*/
// 		int k_cycle;
//         while (scheduler.get_current_work_next_k_cycle(k_cycle)) {
// 			__syncthreads();
// 			if (k_cycle > scheduler.get_current_work_k_cycle_start()) {
// 				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle);
// 				int k_cycle_mod2 = scheduler.k_cycle_mod<2>(k_cycle);
// 				load_trans_mat(img_block_idx, s_corr_div_2[k_cycle_mod2], s_coor_x[k_cycle_mod2], s_coor_y[k_cycle_mod2]);
// 				// if (bid == 0 && tid == 0) {
// 				// 	printf("k_cycle : %3d  load_trans_mat       img_block_idx : %3d k_cycle_mod2 : %2d\n", k_cycle, img_block_idx, k_cycle_mod2);
// 				// }
// 			}

// 			__syncthreads();
// 			if (k_cycle > scheduler.get_current_work_k_cycle_start()) {
// 				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle);
// 				int k_cycle_mod2 = scheduler.k_cycle_mod<2>(k_cycle);
// 				orientation_matrix_handler.construct_orientation_matrix(
// 					s_orient_real_mat_block_swizzle,
// 					s_orient_imag_mat_block_swizzle,
// 					s_orient_pow2_accumulator,
// 					projector,
// 					s_eulers_scaled_head,
// 					s_eulers_scaled_tail,
// 					s_fcoor_xy[k_cycle_mod2],
// 					s_corr_div_2[k_cycle_mod2],
// 					img_block_idx,
// 					orient_block_idx,
// 					warp_id,
// 					lane_id
// 				);

// 				__syncthreads();
// 				/*=============================== COMPUTE CROSS TERM ==============================*/
// 				block_mma_tf32_sim_fp32<decltype(s_trans_mat_block_swizzle), decltype(s_orient_mat_block_swizzle), 
// 				TParams::kTransBlockSize, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize,
// 				TParams::kWarpTransTileSize, TParams::kWarpOrientTileSize, 2 * TParams::kWarpImgTileSize,
// 				TParams::kMmaTransTileSize, TParams::kMmaOrientTileSize, TParams::kMmaImgTileSize>(
// 					fragment_c, s_trans_mat_block_swizzle, s_orient_mat_block_swizzle, warp_id, lane_id);
// 			}

// 			__syncthreads();
// 			if (k_cycle < scheduler.get_current_work_k_cycle_end() - 1) {
// 				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle + 1);
// 				int k_cycle_next_mod2 = scheduler.k_cycle_mod<2>(k_cycle + 1);
// 				load_coord_xy(img_block_idx, s_corr_div_2[k_cycle_next_mod2], s_coor_x[k_cycle_next_mod2], s_coor_y[k_cycle_next_mod2], s_fcoor_xy[k_cycle_next_mod2]);
// 			}

//         } // end of image block

//         // reduce s_trans_pow2_accumulator
//         for (int i = 1; i < TParams::kBlockSize / TParams::kTransBlockSize; ++i) {
//             if (tid < TParams::kTransBlockSize) {
//                 s_trans_pow2_accumulator[tid] += s_trans_pow2_accumulator[i * TParams::kTransBlockSize + tid];
//             }
//         }
//         // reduce s_orient_pow2_accumulator
//         // for (int i = 1; i < TParams::kBlockSize / TParams::kOrientBlockSize; ++i) {
//         //     if (tid < TParams::kOrientBlockSize) {
//         //         s_orient_pow2_accumulator[tid] += s_orient_pow2_accumulator[i * TParams::kOrientBlockSize + tid];
//         //     }
//         // }

//     /*=============================== REDUCE IN FRAGMENT_C ==============================*/
//         __syncthreads();
        
//         #pragma unroll
//         for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
//             #pragma unroll
//             for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
//                 #pragma unroll
//                 for (int k = 0; k < kFragmentCSize; ++k) {
//                     int m = fragment_c_m_idx_in_block<TParams>(warp_id, lane_id, i, k);
//                     int n = fragment_c_n_idx_in_block<TParams>(warp_id, lane_id, j, k);
//                     fragment_c[i][j][k] += s_trans_pow2_accumulator[m] + s_orient_pow2_accumulator[n];
//                 }
//             }
//         }
//         __syncthreads();

//     /*=============================== WRITE BACK ==============================*/
// 		epilogue();

//         scheduler.advance_to_next_work();
//     } // end of while has_work
// }



// // construct orientation matrix (use z-order)
// // fused fetch tex and mma
// template<typename TParams>
// __launch_bounds__(128, 2)
// // __launch_bounds__(128, 4)
// __global__ void cuda_kernel_coarse_matrixV9(
// 	XFLOAT *g_eulers,
// 	XFLOAT *trans_x,
// 	XFLOAT *trans_y,
// 	XFLOAT *g_real,
// 	XFLOAT *g_imag,
// 	AccProjectorKernel projector,
// 	XFLOAT *g_corr,
// 	XFLOAT *g_diff2s,
// 	XFLOAT *g_diff2s_opt,
// 	const int translation_num,
// 	const int orientation_num,
// 	const int image_size,
// 	XFLOAT *g_trans_real_m,
// 	XFLOAT *g_trans_imag_m,
// 	XFLOAT *g_orient_real_m,
// 	XFLOAT *g_orient_imag_m) {
// 	static_assert(TParams::kBlockSize % 32 == 0, "kBlockSize must be a multiple of 32");
// 	static_assert(TParams::kBlockSize >= TParams::kTransBlockSize, "kBlockSize must be greater than or equal to kTransBlockSize");
// 	// static_assert(TParams::kBlockSize >= TParams::kOrientBlockSize, "kBlockSize must be greater than or equal to kOrientBlockSize");

// 	static_assert(TParams::kTransBlockSize % TParams::kWarpTransTileSize == 0, "kTransBlockSize must be a multiple of kWarpTransTileSize");
// 	static_assert(TParams::kOrientBlockSize % TParams::kWarpOrientTileSize == 0, "kOrientBlockSize must be a multiple of kWarpOrientTileSize");
// 	static_assert(TParams::kTransBlockSize % TParams::kMmaTransTileSize == 0, "kTransBlockSize must be a multiple of kMmaTransTileSize");
// 	static_assert(TParams::kOrientBlockSize % TParams::kMmaOrientTileSize == 0, "kOrientBlockSize must be a multiple of kMmaOrientTileSize");
// 	static_assert(TParams::kBlockSize / 32 == (TParams::kTransBlockSize / TParams::kWarpTransTileSize) * (TParams::kOrientBlockSize / TParams::kWarpOrientTileSize), "kBlockSize must be equal to the product of the number of warps in translation, orientation and image dimension");

// 	static_assert(TParams::kImgBlockSize == 16, "kImgBlockSize must be 16");
// 	static_assert(TParams::kImgBlockSize == TParams::kWarpImgTileSize, "kImgBlockSize must be equal to kWarpImgTileSize");

// 	const int tid = threadIdx.x;          // thread id in a block
// 	const int bid = blockIdx.x;           // block id in a grid
// 	const int warp_id  = tid / 32;        // warp id in a block
// 	constexpr int kWarpNum = TParams::kBlockSize / 32; // number of warps in a block
// 	const int lane_id  = tid % 32;        // thread id in a warp

// 	const int trans_block_num = (translation_num + TParams::kTransBlockSize - 1) / TParams::kTransBlockSize;
// 	const int orient_block_num = (orientation_num + TParams::kOrientBlockSize - 1) / TParams::kOrientBlockSize;

// 	int trans_block_idx = (bid % trans_block_num) * TParams::kTransBlockSize;
// 	int orient_block_idx = (bid / trans_block_num) * TParams::kOrientBlockSize;

// 	assert(blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0);
	
// 	CoarseScheduler<TParams::kTransBlockSize, 
// 					TParams::kOrientBlockSize, 
// 					TParams::kImgBlockSize, 
// 					CoarseSchedulerStrategy::SplitK,
// 					2>
// 		scheduler(translation_num, orientation_num, image_size);

// 	__shared__ XFLOAT s_trans_mat_block[2 * TParams::kTransBlockSize * TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_orient_mat_block[2 * 2 * TParams::kOrientBlockSize * TParams::kImgBlockSize];

// 	using TransMatLayout = SharedMemorySwizzle<float, TParams::kTransBlockSize, 2 * TParams::kImgBlockSize, 0>;
// 	using TransRealMatLayout = SharedMemorySwizzle<float, TParams::kTransBlockSize, TParams::kImgBlockSize, 0>;
// 	using TransImagMatLayout = SharedMemorySwizzle<float, TParams::kTransBlockSize, TParams::kImgBlockSize, TParams::kImgBlockSize>;
	
// 	using OrientMatLayout = SharedMemorySwizzle<float, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize, 0>;
// 	using OrientRealMatLayout = SharedMemorySwizzle<float, TParams::kOrientBlockSize, TParams::kImgBlockSize, 0>;
// 	using OrientImagMatLayout = SharedMemorySwizzle<float, TParams::kOrientBlockSize, TParams::kImgBlockSize, TParams::kImgBlockSize>;

// 	TransMatLayout s_trans_mat_block_swizzle(s_trans_mat_block);
// 	TransRealMatLayout s_trans_real_mat_block_swizzle(s_trans_mat_block);
// 	TransImagMatLayout s_trans_imag_mat_block_swizzle(s_trans_mat_block);

// 	// OrientMatLayout s_orient_mat_block_swizzle(s_orient_mat_block);
// 	// OrientRealMatLayout s_orient_real_mat_block_swizzle(s_orient_mat_block);
// 	// OrientImagMatLayout s_orient_imag_mat_block_swizzle(s_orient_mat_block);

// 	OrientMatLayout s_orient_mat_block_swizzle[2] = {
// 		OrientMatLayout(s_orient_mat_block),
// 		OrientMatLayout(s_orient_mat_block + 2 * TParams::kOrientBlockSize * TParams::kImgBlockSize)
// 	};

// 	OrientRealMatLayout s_orient_real_mat_block_swizzle[2] = {
// 		OrientRealMatLayout(s_orient_mat_block),
// 		OrientRealMatLayout(s_orient_mat_block + 2 * TParams::kOrientBlockSize * TParams::kImgBlockSize)
// 	};
// 	OrientImagMatLayout s_orient_imag_mat_block_swizzle[2] = {
// 		OrientImagMatLayout(s_orient_mat_block),
// 		OrientImagMatLayout(s_orient_mat_block + 2 * TParams::kOrientBlockSize * TParams::kImgBlockSize)
// 	};


// 	OrientationMatrixHandler<TParams::kOrientBlockSize,
// 							 TParams::kImgBlockSize,
// 							 kWarpNum,
// 							 OrientRealMatLayout,
// 							 OrientImagMatLayout,
// 							 TransMatLayout, OrientMatLayout, 
// 							 TParams::kTransBlockSize, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize,
// 							 TParams::kWarpTransTileSize, TParams::kWarpOrientTileSize, 2 * TParams::kWarpImgTileSize,
// 							 TParams::kMmaTransTileSize, TParams::kMmaOrientTileSize, TParams::kMmaImgTileSize>
// 		orientation_matrix_handler(image_size, orientation_num);


// 	// double buffer for s_corr_div_2, s_coor_x, s_coor_y
// 	__shared__ XFLOAT s_corr_div_2[2][TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_coor_x[2][TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_coor_y[2][TParams::kImgBlockSize];

// 	// ============================  new  ============================
// 	__shared__ XFLOAT s_fcoor_xy[2][TParams::kImgBlockSize * 2]; // img -> x,y
// 	// For a 2D scenario, e8 is not used, so it’s not stored in shared memory.
// 	// e2 and e5 are also unused, but they remain in shared memory for alignment.
// 	__shared__ XFLOAT s_eulers_scaled_head[TParams::kOrientBlockSize * 4]; // (e0 e1 e2 e3)  * projector.padding_factor
// 	__shared__ XFLOAT s_eulers_scaled_tail[TParams::kOrientBlockSize * 4]; // (e4 e5 e6 e7)  * projector.padding_factor

// 	// reduce buffer
// 	__shared__ XFLOAT s_trans_pow2_accumulator[(TParams::kBlockSize / TParams::kTransBlockSize) * TParams::kTransBlockSize];
// 	// __shared__ XFLOAT s_orient_pow2_accumulator[(TParams::kBlockSize / TParams::kOrientBlockSize) * TParams::kOrientBlockSize];

// 	__shared__ XFLOAT s_orient_pow2_accumulator[TParams::kOrientBlockSize];

// 	// used for dummy store shared
// 	__shared__ XFLOAT s_test_buffer[3 *  4 * 4];
// 	// register
// 	constexpr int kNumMmaTransInWarpTile = TParams::kWarpTransTileSize / TParams::kMmaTransTileSize;
// 	constexpr int kNumMmaOrientInWarpTile = TParams::kWarpOrientTileSize / TParams::kMmaOrientTileSize;
// 	constexpr int kNumMmaImgInWarpTile = TParams::kWarpImgTileSize / TParams::kMmaImgTileSize;

// 	constexpr int kFragmentASize = TParams::kMmaTransTileSize * TParams::kMmaImgTileSize / kWarpSize;
// 	constexpr int kFragmentBSize = TParams::kMmaOrientTileSize * TParams::kMmaImgTileSize / kWarpSize;
// 	constexpr int kFragmentCSize = TParams::kMmaTransTileSize * TParams::kMmaOrientTileSize / kWarpSize;

// 	XFLOAT fragment_c[kNumMmaTransInWarpTile][kNumMmaOrientInWarpTile][kFragmentCSize];

// 	constexpr int kRegistersMmaPerThread = kNumMmaTransInWarpTile * kFragmentASize +
// 										   kNumMmaOrientInWarpTile * kFragmentBSize +
// 										   kNumMmaTransInWarpTile * kNumMmaOrientInWarpTile * kFragmentCSize;

// 	#if kRegistersMmaPerThread >= 256
// 	#warning "kRegistersMmaPerThread must be less than or equal to 256, otherwise register spilling will occur"
// 	#endif


// 	// ============================= lambda function =============================
// 	//given current img_block_idx, load global array into corr_div_2, coord_x, coord_y
// 	auto load_coord_xy = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y, XFLOAT* fcoor_xy) {
// 		#pragma unroll 
// 		for (int i = tid; i < TParams::kImgBlockSize; i += TParams::kBlockSize) {
// 			if (img_block_idx + i < image_size) {
// 				corr_div_2[i] = g_corr[img_block_idx + i] / 2;
// 			} else {
// 				corr_div_2[i] = 0.;
// 			}

// 			int x, y;
// 			pixel_index2coor(img_block_idx + i, projector.imgX, projector.imgY, projector.maxR, x, y);
// 			coord_x[i] = x;
// 			coord_y[i] = y;
			
// 			fcoor_xy[2 * i + 0] = x;
// 			fcoor_xy[2 * i + 1] = y;
// 		}
// 	};

// 	//given current img_block_idx, corr_div_2, coord_x, coord_y, load trans mat into s_trans_real_mat_block and s_trans_imag_mat_block
// 	auto load_trans_mat = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y) {
// 		assert(TParams::kBlockSize % TParams::kTransBlockSize == 0);
// 		#pragma unroll
// 		for (int i = tid / TParams::kTransBlockSize; i < TParams::kImgBlockSize; i += TParams::kBlockSize / TParams::kTransBlockSize) {
// 			int g_img_idx = img_block_idx + i;
// 			int trans_idx = tid % TParams::kTransBlockSize;
// 			if (g_img_idx >= image_size) {
// 				assert(trans_idx < TParams::kTransBlockSize);
// 				assert(i < TParams::kImgBlockSize);
// 				s_trans_real_mat_block_swizzle(trans_idx, i) = 0.;
// 				s_trans_imag_mat_block_swizzle(trans_idx, i) = 0.;
// 				continue;
// 			}
// 			int g_trans_idx = trans_block_idx + trans_idx;
// 			if (g_trans_idx >= translation_num) {
// 				continue;
// 			}
// 			XFLOAT tx = trans_x[g_trans_idx];
// 			XFLOAT ty = trans_y[g_trans_idx];
// 			XFLOAT real = g_real[g_img_idx];
// 			XFLOAT imag = g_imag[g_img_idx];

// 			int x = coord_x[i];
// 			int y = coord_y[i];
// 			XFLOAT trans_real, trans_imag;
// 			translatePixel(x, y, tx, ty, real, imag, trans_real, trans_imag);

// 			// s_trans_real_mat_block[trans_idx * TParams::kImgBlockSize + i] = -2 * trans_real * corr_div_2[i];
// 			// s_trans_imag_mat_block[trans_idx * TParams::kImgBlockSize + i] = -2 * trans_imag * corr_div_2[i];
// 			s_trans_real_mat_block_swizzle(trans_idx, i) = -2 * trans_real * corr_div_2[i];
// 			s_trans_imag_mat_block_swizzle(trans_idx, i) = -2 * trans_imag * corr_div_2[i];

// 			XFLOAT magnitude_squared_sum = trans_real * trans_real * corr_div_2[i] + trans_imag * trans_imag * corr_div_2[i];
// 			s_trans_pow2_accumulator[tid] += magnitude_squared_sum;
// 		}
// 	};

//     auto init_fragment_c = [&] () {
// 		// Default: need read from g_diff2s
// 		if (scheduler.get_strategy() == CoarseSchedulerStrategy::Default || 
// 		   (scheduler.get_strategy() == CoarseSchedulerStrategy::SplitK && g_diff2s != g_diff2s_opt)) {		
// 			#pragma unroll
// 			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 				#pragma unroll
// 				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 					#pragma unroll
// 					for (int k = 0; k < kFragmentCSize; ++k) {
// 						int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
// 						int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);
// 						if (m < translation_num && n < orientation_num) {
// 							fragment_c[i][j][k] = g_diff2s[n * translation_num + m];
// 						} else {
// 							fragment_c[i][j][k] = 0.0;
// 						}
// 					}
// 				}
// 			}
// 		}
// 		// SplitK: use atomicAdd to accumulate, if diff2s source == diff2s dest, no need to read from g_diff2s
// 		// else, read from g_diff2s
// 		else if (scheduler.get_strategy() == CoarseSchedulerStrategy::SplitK && g_diff2s == g_diff2s_opt) {
// 			#pragma unroll
// 			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 				#pragma unroll
// 				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 					#pragma unroll
// 					for (int k = 0; k < kFragmentCSize; ++k) {
// 						fragment_c[i][j][k] = 0.0;
// 					}
// 				}
// 			}
			
// 		}
// 		else {
// 			assert(false);
// 		}
//     };

// 	auto epilogue = [&] () {
// 		// write fragment_c back to g_diff2s_opt
//         if (scheduler.get_strategy() == CoarseSchedulerStrategy::Default) {
// 			#pragma unroll
// 			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 				#pragma unroll
// 				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 					#pragma unroll
// 					for (int k = 0; k < kFragmentCSize; ++k) {
// 						int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
// 						int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);

// 						if (m < translation_num && n < orientation_num) {
// 							g_diff2s_opt[n * translation_num + m] = fragment_c[i][j][k];
// 						}
// 					}
// 				}
// 			}
// 		} else if (scheduler.get_strategy() == CoarseSchedulerStrategy::SplitK) {
// 			// use atomic add
// 			#pragma unroll
// 			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 				#pragma unroll
// 				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 					#pragma unroll
// 					for (int k = 0; k < kFragmentCSize; ++k) {
// 						int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
// 						int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);

// 						if (m < translation_num && n < orientation_num) {
// 							atomicAdd(&g_diff2s_opt[n * translation_num + m], fragment_c[i][j][k]);
// 						}
// 					}
// 				}
// 			}
// 		} else {
// 			assert(false);
// 		}
// 	};

//     // =====================================================================
// 	// ============================= main loop =============================
//     // =====================================================================
//     while (scheduler.has_work()) {
// 		__syncthreads();

//         trans_block_idx = scheduler.get_current_work_m_block_offset();
//         orient_block_idx = scheduler.get_current_work_n_block_offset();

// 		// load eulers to smem
// 		#pragma unroll
// 		for (int i = tid; i < TParams::kOrientBlockSize; i += TParams::kBlockSize) {
// 			if (orient_block_idx + i < orientation_num) {
// 				// TODO: check whether compiler uses load float4
// 				#pragma unroll
// 				for (int j = 0; j < 4; j ++) {
// 					s_eulers_scaled_head[i * 4 + j] = g_eulers[(orient_block_idx + i) * 9 + j] * projector.padding_factor;
// 					s_eulers_scaled_tail[i * 4 + j] = g_eulers[(orient_block_idx + i) * 9 + 4 + j] * projector.padding_factor;
// 				}
// 			} else {
// 				#pragma unroll
// 				for (int j = 0; j < 4; j ++) {
// 					s_eulers_scaled_head[i * 4 + j] = 0;
// 					s_eulers_scaled_tail[i * 4 + j] = 0;
// 				}
// 			}
// 		}

//         // initialize shared memory to zero
//         for (int i = tid; i < 2 * TParams::kTransBlockSize * TParams::kImgBlockSize; i += TParams::kBlockSize) {
//             s_trans_mat_block[i] = 0.0;
//         }
//         for (int i = tid; i < 2 * 2 * TParams::kOrientBlockSize * TParams::kImgBlockSize; i += TParams::kBlockSize) {
//             s_orient_mat_block[i] = 0.0;
//         }

//         s_trans_pow2_accumulator[tid] = 0.0;
//         // s_orient_pow2_accumulator[tid] = 0.0;
// 		if (tid < TParams::kOrientBlockSize) {
// 			s_orient_pow2_accumulator[tid] = 0.;
// 		}

//         for (int i = tid; i < TParams::kImgBlockSize; i += TParams::kBlockSize) {
//             s_corr_div_2[0][i] = 0.0;
//             s_corr_div_2[1][i] = 0.0;
//         }

//         // read fragment_c from g_diff2s
//         init_fragment_c();

//         __syncthreads();

// /*=============================== FOR IMAGE BLOCK ==============================*/
// 		int k_cycle;
//         while (scheduler.get_current_work_next_k_cycle(k_cycle)) {
// 			__syncthreads();
// 			if (k_cycle > scheduler.get_current_work_k_cycle_start()) {
// 				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle);
// 				int k_cycle_mod2 = scheduler.k_cycle_mod<2>(k_cycle);
// 				load_trans_mat(img_block_idx, s_corr_div_2[k_cycle_mod2], s_coor_x[k_cycle_mod2], s_coor_y[k_cycle_mod2]);
// 			}

// 			if (k_cycle < scheduler.get_current_work_k_cycle_end() - 1) {
// 				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle + 1);
// 				int k_cycle_next_mod2 = scheduler.k_cycle_mod<2>(k_cycle + 1);
// 				load_coord_xy(img_block_idx, s_corr_div_2[k_cycle_next_mod2], s_coor_x[k_cycle_next_mod2], s_coor_y[k_cycle_next_mod2], s_fcoor_xy[k_cycle_next_mod2]);
// 			}

// 			if (k_cycle > scheduler.get_current_work_k_cycle_start()) {
// 				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle);
// 				int k_cycle_mod2 = scheduler.k_cycle_mod<2>(k_cycle);
// 				orientation_matrix_handler.sync_and_store_orientation_matrix_with_reduce(
// 					s_orient_mat_block_swizzle[k_cycle_mod2],
// 					s_orient_real_mat_block_swizzle[k_cycle_mod2],
// 					s_orient_imag_mat_block_swizzle[k_cycle_mod2],
// 					s_orient_pow2_accumulator,
// 					s_corr_div_2[k_cycle_mod2],
// 					warp_id,
// 					lane_id
// 				);
// 			}

// 			__syncthreads();
// 			if (k_cycle > scheduler.get_current_work_k_cycle_start() && 
// 				k_cycle < scheduler.get_current_work_k_cycle_end() - 1) {

// 				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle);
// 				int k_cycle_mod2 = scheduler.k_cycle_mod<2>(k_cycle);
// 				int k_cycle_next_mod2 = scheduler.k_cycle_mod<2>(k_cycle + 1);
// 				// fused
// 				orientation_matrix_handler.process_and_prefetch_orientation_matrix_fused_mma_tf32_sim_fp32(
// 					fragment_c, 
// 					s_trans_mat_block_swizzle, 
// 					s_orient_mat_block_swizzle[k_cycle_mod2],
// 					projector,
// 					s_eulers_scaled_head,
// 					s_eulers_scaled_tail,
// 					s_fcoor_xy[k_cycle_next_mod2],
// 					img_block_idx,
// 					orient_block_idx,
// 					warp_id,
// 					lane_id,
// 					s_test_buffer
// 				);
				
// 				// not fused
// 				// orientation_matrix_handler.process_and_prefetch_orientation_matrix(
// 				// 	projector,
// 				// 	s_eulers_scaled_head,
// 				// 	s_eulers_scaled_tail,
// 				// 	s_fcoor_xy[k_cycle_next_mod2],
// 				// 	img_block_idx,
// 				// 	orient_block_idx,
// 				// 	warp_id,
// 				// 	lane_id
// 				// );
// 				// // __syncthreads();
// 				// // __syncthreads();
// 				// /*=============================== COMPUTE CROSS TERM ==============================*/
// 				// block_mma_tf32_sim_fp32<TransMatLayout, OrientMatLayout, 
// 				// TParams::kTransBlockSize, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize,
// 				// TParams::kWarpTransTileSize, TParams::kWarpOrientTileSize, 2 * TParams::kWarpImgTileSize,
// 				// TParams::kMmaTransTileSize, TParams::kMmaOrientTileSize, TParams::kMmaImgTileSize>(
// 				// 	fragment_c, s_trans_mat_block_swizzle, s_orient_mat_block_swizzle[k_cycle_mod2], warp_id, lane_id);
// 			}

// 			if (k_cycle == scheduler.get_current_work_k_cycle_start()) {
// 				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle + 1);
// 				int k_cycle_next_mod2 = scheduler.k_cycle_mod<2>(k_cycle + 1);
// 				orientation_matrix_handler.process_and_prefetch_orientation_matrix(
// 					projector,
// 					s_eulers_scaled_head,
// 					s_eulers_scaled_tail,
// 					s_fcoor_xy[k_cycle_next_mod2],
// 					img_block_idx,
// 					orient_block_idx,
// 					warp_id,
// 					lane_id
// 				);
// 			}

// 			if (k_cycle == scheduler.get_current_work_k_cycle_end() - 1) {
// 				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle);
// 				int k_cycle_mod2 = scheduler.k_cycle_mod<2>(k_cycle);
// 				/*=============================== COMPUTE CROSS TERM ==============================*/
// 				block_mma_tf32_sim_fp32<TransMatLayout, OrientMatLayout, 
// 				TParams::kTransBlockSize, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize,
// 				TParams::kWarpTransTileSize, TParams::kWarpOrientTileSize, 2 * TParams::kWarpImgTileSize,
// 				TParams::kMmaTransTileSize, TParams::kMmaOrientTileSize, TParams::kMmaImgTileSize>(
// 					fragment_c, s_trans_mat_block_swizzle, s_orient_mat_block_swizzle[k_cycle_mod2], warp_id, lane_id);
// 			}

// 			__syncthreads();

//         } // end of image block

//         // reduce s_trans_pow2_accumulator
//         for (int i = 1; i < TParams::kBlockSize / TParams::kTransBlockSize; ++i) {
//             if (tid < TParams::kTransBlockSize) {
//                 s_trans_pow2_accumulator[tid] += s_trans_pow2_accumulator[i * TParams::kTransBlockSize + tid];
//             }
//         }
//         // reduce s_orient_pow2_accumulator
//         // for (int i = 1; i < TParams::kBlockSize / TParams::kOrientBlockSize; ++i) {
//         //     if (tid < TParams::kOrientBlockSize) {
//         //         s_orient_pow2_accumulator[tid] += s_orient_pow2_accumulator[i * TParams::kOrientBlockSize + tid];
//         //     }
//         // }

//     /*=============================== REDUCE IN FRAGMENT_C ==============================*/
//         __syncthreads();
        
//         #pragma unroll
//         for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
//             #pragma unroll
//             for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
//                 #pragma unroll
//                 for (int k = 0; k < kFragmentCSize; ++k) {
//                     int m = fragment_c_m_idx_in_block<TParams>(warp_id, lane_id, i, k);
//                     int n = fragment_c_n_idx_in_block<TParams>(warp_id, lane_id, j, k);
//                     fragment_c[i][j][k] += s_trans_pow2_accumulator[m] + s_orient_pow2_accumulator[n];
//                 }
//             }
//         }
//         __syncthreads();

//     /*=============================== WRITE BACK ==============================*/
// 		epilogue();

//         scheduler.advance_to_next_work();
//     } // end of while has_work
// }





// // construct translation matrix
// // add translation_matrix_handler
// template<typename TParams>
// __launch_bounds__(128, 2)
// // __launch_bounds__(128, 4)
// __global__ void cuda_kernel_coarse_matrix(
// 	XFLOAT *g_eulers,
// 	XFLOAT *trans_x,
// 	XFLOAT *trans_y,
// 	XFLOAT *g_real,
// 	XFLOAT *g_imag,
// 	AccProjectorKernel projector,
// 	XFLOAT *g_corr,
// 	XFLOAT *g_diff2s,
// 	XFLOAT *g_diff2s_opt,
// 	const int translation_num,
// 	const int orientation_num,
// 	const int image_size,
// 	XFLOAT *g_trans_real_m,
// 	XFLOAT *g_trans_imag_m,
// 	XFLOAT *g_orient_real_m,
// 	XFLOAT *g_orient_imag_m) {
// 	static_assert(TParams::kBlockSize % 32 == 0, "kBlockSize must be a multiple of 32");
// 	static_assert(TParams::kBlockSize >= TParams::kTransBlockSize, "kBlockSize must be greater than or equal to kTransBlockSize");
// 	// static_assert(TParams::kBlockSize >= TParams::kOrientBlockSize, "kBlockSize must be greater than or equal to kOrientBlockSize");

// 	static_assert(TParams::kTransBlockSize % TParams::kWarpTransTileSize == 0, "kTransBlockSize must be a multiple of kWarpTransTileSize");
// 	static_assert(TParams::kOrientBlockSize % TParams::kWarpOrientTileSize == 0, "kOrientBlockSize must be a multiple of kWarpOrientTileSize");
// 	static_assert(TParams::kTransBlockSize % TParams::kMmaTransTileSize == 0, "kTransBlockSize must be a multiple of kMmaTransTileSize");
// 	static_assert(TParams::kOrientBlockSize % TParams::kMmaOrientTileSize == 0, "kOrientBlockSize must be a multiple of kMmaOrientTileSize");
// 	static_assert(TParams::kBlockSize / 32 == (TParams::kTransBlockSize / TParams::kWarpTransTileSize) * (TParams::kOrientBlockSize / TParams::kWarpOrientTileSize), "kBlockSize must be equal to the product of the number of warps in translation, orientation and image dimension");

// 	static_assert(TParams::kImgBlockSize == 16, "kImgBlockSize must be 16");
// 	static_assert(TParams::kImgBlockSize == TParams::kWarpImgTileSize, "kImgBlockSize must be equal to kWarpImgTileSize");

// 	const int tid = threadIdx.x;          // thread id in a block
// 	const int bid = blockIdx.x;           // block id in a grid
// 	const int warp_id  = tid / 32;        // warp id in a block
// 	constexpr int kWarpNum = TParams::kBlockSize / 32; // number of warps in a block
// 	const int lane_id  = tid % 32;        // thread id in a warp

// 	const int trans_block_num = (translation_num + TParams::kTransBlockSize - 1) / TParams::kTransBlockSize;
// 	const int orient_block_num = (orientation_num + TParams::kOrientBlockSize - 1) / TParams::kOrientBlockSize;

// 	int trans_block_idx = (bid % trans_block_num) * TParams::kTransBlockSize;
// 	int orient_block_idx = (bid / trans_block_num) * TParams::kOrientBlockSize;

// 	assert(blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0);
	
// 	CoarseScheduler<TParams::kTransBlockSize, 
// 					TParams::kOrientBlockSize, 
// 					TParams::kImgBlockSize, 
// 					CoarseSchedulerStrategy::SplitK,
// 					2>
// 		scheduler(translation_num, orientation_num, image_size);

// 	__shared__ XFLOAT s_trans_mat_block[2 * TParams::kTransBlockSize * TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_trans_mat_block_bak[2 * TParams::kTransBlockSize * TParams::kImgBlockSize];
	
// 	// __shared__ XFLOAT s_orient_mat_block[2 * 2 * TParams::kOrientBlockSize * TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_orient_mat_block[2 * TParams::kOrientBlockSize * TParams::kImgBlockSize];

// 	using TransMatLayout = SharedMemorySwizzle<float, TParams::kTransBlockSize, 2 * TParams::kImgBlockSize, 0>;
// 	using TransRealMatLayout = SharedMemorySwizzle<float, TParams::kTransBlockSize, TParams::kImgBlockSize, 0>;
// 	using TransImagMatLayout = SharedMemorySwizzle<float, TParams::kTransBlockSize, TParams::kImgBlockSize, TParams::kImgBlockSize>;
	
// 	using OrientMatLayout = SharedMemorySwizzle<float, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize, 0>;
// 	using OrientRealMatLayout = SharedMemorySwizzle<float, TParams::kOrientBlockSize, TParams::kImgBlockSize, 0>;
// 	using OrientImagMatLayout = SharedMemorySwizzle<float, TParams::kOrientBlockSize, TParams::kImgBlockSize, TParams::kImgBlockSize>;

// 	TransMatLayout s_trans_mat_block_swizzle(s_trans_mat_block);
// 	TransRealMatLayout s_trans_real_mat_block_swizzle(s_trans_mat_block);
// 	TransImagMatLayout s_trans_imag_mat_block_swizzle(s_trans_mat_block);

// 	TransMatLayout s_trans_mat_block_swizzle_bak(s_trans_mat_block_bak);
// 	TransRealMatLayout s_trans_real_mat_block_swizzle_bak(s_trans_mat_block_bak);
// 	TransImagMatLayout s_trans_imag_mat_block_swizzle_bak(s_trans_mat_block_bak);

// 	// OrientMatLayout s_orient_mat_block_swizzle(s_orient_mat_block);
// 	// OrientRealMatLayout s_orient_real_mat_block_swizzle(s_orient_mat_block);
// 	// OrientImagMatLayout s_orient_imag_mat_block_swizzle(s_orient_mat_block);

// 	OrientMatLayout s_orient_mat_block_swizzle[2] = {
// 		OrientMatLayout(s_orient_mat_block),
// 		OrientMatLayout(s_orient_mat_block)
// 		// OrientMatLayout(s_orient_mat_block + 2 * TParams::kOrientBlockSize * TParams::kImgBlockSize)
// 	};

// 	OrientRealMatLayout s_orient_real_mat_block_swizzle[2] = {
// 		OrientRealMatLayout(s_orient_mat_block),
// 		OrientRealMatLayout(s_orient_mat_block)
// 		// OrientRealMatLayout(s_orient_mat_block + 2 * TParams::kOrientBlockSize * TParams::kImgBlockSize)
// 	};
// 	OrientImagMatLayout s_orient_imag_mat_block_swizzle[2] = {
// 		OrientImagMatLayout(s_orient_mat_block),
// 		OrientImagMatLayout(s_orient_mat_block)
// 		// OrientImagMatLayout(s_orient_mat_block + 2 * TParams::kOrientBlockSize * TParams::kImgBlockSize)
// 	};


// 	OrientationMatrixHandler<TParams::kOrientBlockSize,
// 							 TParams::kImgBlockSize,
// 							 kWarpNum,
// 							 OrientRealMatLayout,
// 							 OrientImagMatLayout,
// 							 TransMatLayout, OrientMatLayout, 
// 							 TParams::kTransBlockSize, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize,
// 							 TParams::kWarpTransTileSize, TParams::kWarpOrientTileSize, 2 * TParams::kWarpImgTileSize,
// 							 TParams::kMmaTransTileSize, TParams::kMmaOrientTileSize, TParams::kMmaImgTileSize>
// 		orientation_matrix_handler(image_size, orientation_num);

// 	TranslationMatrixHandler<TParams::kTransBlockSize,
// 							 TParams::kImgBlockSize,
// 							 kWarpNum,
// 							 TransRealMatLayout,
// 							 TransImagMatLayout>
// 		translation_matrix_handler(image_size, trans_block_num);

// 	// double buffer for s_corr_div_2, s_coor_x, s_coor_y
// 	__shared__ XFLOAT s_corr_div_2[2][TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_coor_x[2][TParams::kImgBlockSize];
// 	__shared__ XFLOAT s_coor_y[2][TParams::kImgBlockSize];

// 	// ============================  new  ============================
// 	// double buffer
// 	__shared__ XFLOAT s_fcoor_xy[2][TParams::kImgBlockSize * 2]; // img -> x,y
// 	__shared__ XFLOAT s_img_real_imag[2][TParams::kImgBlockSize * 2]; // img -> real,imag

// 	// For a 2D scenario, e8 is not used, so it’s not stored in shared memory.
// 	// e2 and e5 are also unused, but they remain in shared memory for alignment.
// 	__shared__ XFLOAT s_eulers_scaled_head[TParams::kOrientBlockSize * 4]; // (e0 e1 e2 e3)  * projector.padding_factor
// 	__shared__ XFLOAT s_eulers_scaled_tail[TParams::kOrientBlockSize * 4]; // (e4 e5 e6 e7)  * projector.padding_factor

// 	__shared__ XFLOAT s_trans_xy[TParams::kTransBlockSize * 2]; // trans_num -> x,y 

// 	// reduce buffer
// 	__shared__ XFLOAT s_trans_pow2_accumulator[(TParams::kBlockSize / TParams::kTransBlockSize) * TParams::kTransBlockSize];
// 	// __shared__ XFLOAT s_orient_pow2_accumulator[(TParams::kBlockSize / TParams::kOrientBlockSize) * TParams::kOrientBlockSize];

// 	__shared__ XFLOAT s_trans_pow2_accumulator_bak[TParams::kTransBlockSize];
// 	__shared__ XFLOAT s_orient_pow2_accumulator[TParams::kOrientBlockSize];

// 	// used for dummy store shared
// 	__shared__ XFLOAT s_test_buffer[3 *  4 * 4];
// 	// register
// 	constexpr int kNumMmaTransInWarpTile = TParams::kWarpTransTileSize / TParams::kMmaTransTileSize;
// 	constexpr int kNumMmaOrientInWarpTile = TParams::kWarpOrientTileSize / TParams::kMmaOrientTileSize;
// 	constexpr int kNumMmaImgInWarpTile = TParams::kWarpImgTileSize / TParams::kMmaImgTileSize;

// 	constexpr int kFragmentASize = TParams::kMmaTransTileSize * TParams::kMmaImgTileSize / kWarpSize;
// 	constexpr int kFragmentBSize = TParams::kMmaOrientTileSize * TParams::kMmaImgTileSize / kWarpSize;
// 	constexpr int kFragmentCSize = TParams::kMmaTransTileSize * TParams::kMmaOrientTileSize / kWarpSize;

// 	XFLOAT fragment_c[kNumMmaTransInWarpTile][kNumMmaOrientInWarpTile][kFragmentCSize];

// 	constexpr int kRegistersMmaPerThread = kNumMmaTransInWarpTile * kFragmentASize +
// 										   kNumMmaOrientInWarpTile * kFragmentBSize +
// 										   kNumMmaTransInWarpTile * kNumMmaOrientInWarpTile * kFragmentCSize;

// 	#if kRegistersMmaPerThread >= 256
// 	#warning "kRegistersMmaPerThread must be less than or equal to 256, otherwise register spilling will occur"
// 	#endif


// 	// ============================= lambda function =============================
// 	//given current img_block_idx, load global array into corr_div_2, coord_x, coord_y
// 	auto load_coord_xy = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y, XFLOAT* fcoor_xy, XFLOAT* img_real_imag) {
// 		#pragma unroll 
// 		for (int i = tid; i < TParams::kImgBlockSize; i += TParams::kBlockSize) {
// 			if (img_block_idx + i < image_size) {
// 				corr_div_2[i] = g_corr[img_block_idx + i] / 2;
// 				img_real_imag[i * 2 + 0] = g_real[img_block_idx + i];
// 				img_real_imag[i * 2 + 1] = g_imag[img_block_idx + i];
// 			} else {
// 				corr_div_2[i] = 0.;
// 				img_real_imag[i * 2 + 0] = 0.;
// 				img_real_imag[i * 2 + 1] = 0.;
// 			}

// 			int x, y;
// 			pixel_index2coor(img_block_idx + i, projector.imgX, projector.imgY, projector.maxR, x, y);
// 			coord_x[i] = x;
// 			coord_y[i] = y;
			
// 			fcoor_xy[2 * i + 0] = x;
// 			fcoor_xy[2 * i + 1] = y;
// 		}
// 	};

// 	//given current img_block_idx, corr_div_2, coord_x, coord_y, load trans mat into s_trans_real_mat_block and s_trans_imag_mat_block
// 	auto load_trans_mat = [&](int img_block_idx, XFLOAT* corr_div_2, XFLOAT* coord_x, XFLOAT* coord_y) {
// 		assert(TParams::kBlockSize % TParams::kTransBlockSize == 0);
// 		#pragma unroll
// 		for (int i = tid / TParams::kTransBlockSize; i < TParams::kImgBlockSize; i += TParams::kBlockSize / TParams::kTransBlockSize) {
// 			int g_img_idx = img_block_idx + i;
// 			int trans_idx = tid % TParams::kTransBlockSize;
// 			if (g_img_idx >= image_size) {
// 				assert(trans_idx < TParams::kTransBlockSize);
// 				assert(i < TParams::kImgBlockSize);
// 				s_trans_real_mat_block_swizzle(trans_idx, i) = 0.;
// 				s_trans_imag_mat_block_swizzle(trans_idx, i) = 0.;
// 				continue;
// 			}
// 			int g_trans_idx = trans_block_idx + trans_idx;
// 			if (g_trans_idx >= translation_num) {
// 				continue;
// 			}
// 			XFLOAT tx = trans_x[g_trans_idx];
// 			XFLOAT ty = trans_y[g_trans_idx];
// 			XFLOAT real = g_real[g_img_idx];
// 			XFLOAT imag = g_imag[g_img_idx];

// 			int x = coord_x[i];
// 			int y = coord_y[i];
// 			XFLOAT trans_real, trans_imag;
// 			translatePixel(x, y, tx, ty, real, imag, trans_real, trans_imag);

// 			// s_trans_real_mat_block[trans_idx * TParams::kImgBlockSize + i] = -2 * trans_real * corr_div_2[i];
// 			// s_trans_imag_mat_block[trans_idx * TParams::kImgBlockSize + i] = -2 * trans_imag * corr_div_2[i];
// 			s_trans_real_mat_block_swizzle(trans_idx, i) = -2 * trans_real * corr_div_2[i];
// 			s_trans_imag_mat_block_swizzle(trans_idx, i) = -2 * trans_imag * corr_div_2[i];

// 			// s_trans_real_mat_block_swizzle(trans_idx, i) = tx;
// 			// s_trans_imag_mat_block_swizzle(trans_idx, i) = ty;

// 			XFLOAT magnitude_squared_sum = trans_real * trans_real * corr_div_2[i] + trans_imag * trans_imag * corr_div_2[i];
// 			s_trans_pow2_accumulator[tid] += magnitude_squared_sum;
// 		}
// 	};

//     auto init_fragment_c = [&] () {
// 		// Default: need read from g_diff2s
// 		if (scheduler.get_strategy() == CoarseSchedulerStrategy::Default || 
// 		   (scheduler.get_strategy() == CoarseSchedulerStrategy::SplitK && g_diff2s != g_diff2s_opt)) {		
// 			#pragma unroll
// 			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 				#pragma unroll
// 				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 					#pragma unroll
// 					for (int k = 0; k < kFragmentCSize; ++k) {
// 						int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
// 						int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);
// 						if (m < translation_num && n < orientation_num) {
// 							fragment_c[i][j][k] = g_diff2s[n * translation_num + m];
// 						} else {
// 							fragment_c[i][j][k] = 0.0;
// 						}
// 					}
// 				}
// 			}
// 		}
// 		// SplitK: use atomicAdd to accumulate, if diff2s source == diff2s dest, no need to read from g_diff2s
// 		// else, read from g_diff2s
// 		else if (scheduler.get_strategy() == CoarseSchedulerStrategy::SplitK && g_diff2s == g_diff2s_opt) {
// 			#pragma unroll
// 			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 				#pragma unroll
// 				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 					#pragma unroll
// 					for (int k = 0; k < kFragmentCSize; ++k) {
// 						fragment_c[i][j][k] = 0.0;
// 					}
// 				}
// 			}
			
// 		}
// 		else {
// 			assert(false);
// 		}
//     };

// 	auto epilogue = [&] () {
// 		// write fragment_c back to g_diff2s_opt
//         if (scheduler.get_strategy() == CoarseSchedulerStrategy::Default) {
// 			#pragma unroll
// 			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 				#pragma unroll
// 				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 					#pragma unroll
// 					for (int k = 0; k < kFragmentCSize; ++k) {
// 						int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
// 						int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);

// 						if (m < translation_num && n < orientation_num) {
// 							g_diff2s_opt[n * translation_num + m] = fragment_c[i][j][k];
// 						}
// 					}
// 				}
// 			}
// 		} else if (scheduler.get_strategy() == CoarseSchedulerStrategy::SplitK) {
// 			// use atomic add
// 			#pragma unroll
// 			for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
// 				#pragma unroll
// 				for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
// 					#pragma unroll
// 					for (int k = 0; k < kFragmentCSize; ++k) {
// 						int m = fragment_c_m_idx_in_global<TParams>(trans_block_idx, warp_id, lane_id, i, k);
// 						int n = fragment_c_n_idx_in_global<TParams>(orient_block_idx, warp_id, lane_id, j, k);

// 						if (m < translation_num && n < orientation_num) {
// 							atomicAdd(&g_diff2s_opt[n * translation_num + m], fragment_c[i][j][k]);
// 						}
// 					}
// 				}
// 			}
// 		} else {
// 			assert(false);
// 		}
// 	};

//     // =====================================================================
// 	// ============================= main loop =============================
//     // =====================================================================
//     while (scheduler.has_work()) {
// 		__syncthreads();

//         trans_block_idx = scheduler.get_current_work_m_block_offset();
//         orient_block_idx = scheduler.get_current_work_n_block_offset();

// 		// read fragment_c from g_diff2s
//         init_fragment_c();

// 		// load eulers to smem
// 		#pragma unroll
// 		for (int i = tid; i < TParams::kOrientBlockSize; i += TParams::kBlockSize) {
// 			if (orient_block_idx + i < orientation_num) {
// 				// TODO: check whether compiler uses load float4
// 				#pragma unroll
// 				for (int j = 0; j < 4; j ++) {
// 					s_eulers_scaled_head[i * 4 + j] = g_eulers[(orient_block_idx + i) * 9 + j] * projector.padding_factor;
// 					s_eulers_scaled_tail[i * 4 + j] = g_eulers[(orient_block_idx + i) * 9 + 4 + j] * projector.padding_factor;
// 				}
// 			} else {
// 				#pragma unroll
// 				for (int j = 0; j < 4; j ++) {
// 					s_eulers_scaled_head[i * 4 + j] = 0;
// 					s_eulers_scaled_tail[i * 4 + j] = 0;
// 				}
// 			}
// 		}

// 		// load trans to smem
// 		#pragma unroll
// 		for (int i = tid; i < TParams::kTransBlockSize; i += TParams::kBlockSize) {
// 			if (trans_block_idx + i < translation_num) {
// 				s_trans_xy[i * 2 + 0] = trans_x[trans_block_idx + i];
// 				s_trans_xy[i * 2 + 1] = trans_y[trans_block_idx + i];
// 			} else {
// 				s_trans_xy[i * 2 + 0] = 0.;
// 				s_trans_xy[i * 2 + 1] = 0.;
// 			}
// 		}

//         // initialize shared memory to zero
//         for (int i = tid; i < 2 * TParams::kTransBlockSize * TParams::kImgBlockSize; i += TParams::kBlockSize) {
//             s_trans_mat_block[i] = 0.0;
//         }
//         // for (int i = tid; i < 2 * 2 * TParams::kOrientBlockSize * TParams::kImgBlockSize; i += TParams::kBlockSize) {
//         for (int i = tid; i < 2 * TParams::kOrientBlockSize * TParams::kImgBlockSize; i += TParams::kBlockSize) {
//             s_orient_mat_block[i] = 0.0;
//         }

//         s_trans_pow2_accumulator[tid] = 0.0;
//         // s_orient_pow2_accumulator[tid] = 0.0;

// 		if (tid < TParams::kTransBlockSize) {
// 			s_trans_pow2_accumulator_bak[tid] = 0.;
// 		}
// 		if (tid < TParams::kOrientBlockSize) {
// 			s_orient_pow2_accumulator[tid] = 0.;
// 		}

//         for (int i = tid; i < TParams::kImgBlockSize; i += TParams::kBlockSize) {
//             s_corr_div_2[0][i] = 0.0;
//             s_corr_div_2[1][i] = 0.0;
//         }

//         __syncthreads();

// /*=============================== FOR IMAGE BLOCK ==============================*/
// 		int k_cycle;
//         while (scheduler.get_current_work_next_k_cycle(k_cycle)) {
// 			__syncthreads();
// 			// __threadfence_block();
// 			if (k_cycle > scheduler.get_current_work_k_cycle_start()) {
// 				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle);
// 				int k_cycle_mod2 = scheduler.k_cycle_mod<2>(k_cycle);
// 				translation_matrix_handler.construct_translation_matrix(
// 					s_trans_real_mat_block_swizzle,
// 					s_trans_imag_mat_block_swizzle,
// 					s_trans_pow2_accumulator,
// 					s_trans_xy,
// 					s_img_real_imag[k_cycle_mod2],
// 					s_fcoor_xy[k_cycle_mod2],
// 					s_corr_div_2[k_cycle_mod2],
// 					img_block_idx,
// 					trans_block_idx,
// 					warp_id,
// 					lane_id
// 				);

// 				// load_trans_mat(img_block_idx, s_corr_div_2[k_cycle_mod2], s_coor_x[k_cycle_mod2], s_coor_y[k_cycle_mod2]);

// 				// __syncthreads();
// 				// if (bid == 0 && tid == 0) {
// 				// 	printf("handler:\n");
// 				// 	s_trans_mat_block_swizzle_bak.print_logical_memory();
// 				// 	printf("origin:\n");
// 				// 	s_trans_mat_block_swizzle.print_logical_memory();
// 				// }

// 				// __syncthreads();
// 			}

// 			if (k_cycle < scheduler.get_current_work_k_cycle_end() - 1) {
// 				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle + 1);
// 				int k_cycle_next_mod2 = scheduler.k_cycle_mod<2>(k_cycle + 1);
// 				load_coord_xy(img_block_idx, s_corr_div_2[k_cycle_next_mod2], 
// 							  s_coor_x[k_cycle_next_mod2], s_coor_y[k_cycle_next_mod2], 
// 							  s_fcoor_xy[k_cycle_next_mod2], s_img_real_imag[k_cycle_next_mod2]);
// 			}

// 			if (k_cycle > scheduler.get_current_work_k_cycle_start()) {
// 				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle);
// 				int k_cycle_mod2 = scheduler.k_cycle_mod<2>(k_cycle);
// 				orientation_matrix_handler.sync_and_store_orientation_matrix_with_reduce(
// 					s_orient_mat_block_swizzle[k_cycle_mod2],
// 					s_orient_real_mat_block_swizzle[k_cycle_mod2],
// 					s_orient_imag_mat_block_swizzle[k_cycle_mod2],
// 					s_orient_pow2_accumulator,
// 					s_corr_div_2[k_cycle_mod2],
// 					warp_id,
// 					lane_id
// 				);
// 			}

// 			__syncthreads();
// 			// __threadfence_block();
// 			if (k_cycle > scheduler.get_current_work_k_cycle_start() && 
// 				k_cycle < scheduler.get_current_work_k_cycle_end() - 1) {

// 				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle);
// 				int k_cycle_mod2 = scheduler.k_cycle_mod<2>(k_cycle);
// 				int k_cycle_next_mod2 = scheduler.k_cycle_mod<2>(k_cycle + 1);
// 				// fused
// 				orientation_matrix_handler.process_and_prefetch_orientation_matrix_fused_mma_tf32_sim_fp32(
// 					fragment_c, 
// 					s_trans_mat_block_swizzle, 
// 					s_orient_mat_block_swizzle[k_cycle_mod2],
// 					projector,
// 					s_eulers_scaled_head,
// 					s_eulers_scaled_tail,
// 					s_fcoor_xy[k_cycle_next_mod2],
// 					img_block_idx,
// 					orient_block_idx,
// 					warp_id,
// 					lane_id,
// 					s_test_buffer
// 				);
				
// 				// not fused
// 				// orientation_matrix_handler.process_and_prefetch_orientation_matrix(
// 				// 	projector,
// 				// 	s_eulers_scaled_head,
// 				// 	s_eulers_scaled_tail,
// 				// 	s_fcoor_xy[k_cycle_next_mod2],
// 				// 	img_block_idx,
// 				// 	orient_block_idx,
// 				// 	warp_id,
// 				// 	lane_id
// 				// );
// 				// // __syncthreads();
// 				// // __syncthreads();
// 				// /*=============================== COMPUTE CROSS TERM ==============================*/
// 				// block_mma_tf32_sim_fp32<TransMatLayout, OrientMatLayout, 
// 				// TParams::kTransBlockSize, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize,
// 				// TParams::kWarpTransTileSize, TParams::kWarpOrientTileSize, 2 * TParams::kWarpImgTileSize,
// 				// TParams::kMmaTransTileSize, TParams::kMmaOrientTileSize, TParams::kMmaImgTileSize>(
// 				// 	fragment_c, s_trans_mat_block_swizzle, s_orient_mat_block_swizzle[k_cycle_mod2], warp_id, lane_id);
// 			}

// 			if (k_cycle == scheduler.get_current_work_k_cycle_start()) {
// 				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle + 1);
// 				int k_cycle_next_mod2 = scheduler.k_cycle_mod<2>(k_cycle + 1);
// 				orientation_matrix_handler.process_and_prefetch_orientation_matrix(
// 					projector,
// 					s_eulers_scaled_head,
// 					s_eulers_scaled_tail,
// 					s_fcoor_xy[k_cycle_next_mod2],
// 					img_block_idx,
// 					orient_block_idx,
// 					warp_id,
// 					lane_id
// 				);
// 			}

// 			if (k_cycle == scheduler.get_current_work_k_cycle_end() - 1) {
// 				int img_block_idx = scheduler.get_k_block_offset_from_k_cycle(k_cycle);
// 				int k_cycle_mod2 = scheduler.k_cycle_mod<2>(k_cycle);
// 				/*=============================== COMPUTE CROSS TERM ==============================*/
// 				block_mma_tf32_sim_fp32<TransMatLayout, OrientMatLayout, 
// 				TParams::kTransBlockSize, TParams::kOrientBlockSize, 2 * TParams::kImgBlockSize,
// 				TParams::kWarpTransTileSize, TParams::kWarpOrientTileSize, 2 * TParams::kWarpImgTileSize,
// 				TParams::kMmaTransTileSize, TParams::kMmaOrientTileSize, TParams::kMmaImgTileSize>(
// 					fragment_c, s_trans_mat_block_swizzle, s_orient_mat_block_swizzle[k_cycle_mod2], warp_id, lane_id);
// 			}

// 			// __syncthreads();

//         } // end of image block
// 		__syncthreads();

//         // reduce s_trans_pow2_accumulator
//         // for (int i = 1; i < TParams::kBlockSize / TParams::kTransBlockSize; ++i) {
//         //     if (tid < TParams::kTransBlockSize) {
//         //         s_trans_pow2_accumulator[tid] += s_trans_pow2_accumulator[i * TParams::kTransBlockSize + tid];
//         //     }
//         // }
//         // reduce s_orient_pow2_accumulator
//         // for (int i = 1; i < TParams::kBlockSize / TParams::kOrientBlockSize; ++i) {
//         //     if (tid < TParams::kOrientBlockSize) {
//         //         s_orient_pow2_accumulator[tid] += s_orient_pow2_accumulator[i * TParams::kOrientBlockSize + tid];
//         //     }
//         // }

//     /*=============================== REDUCE IN FRAGMENT_C ==============================*/
//         __syncthreads();
        
//         #pragma unroll
//         for (int i = 0; i < kNumMmaTransInWarpTile; ++i) {
//             #pragma unroll
//             for (int j = 0; j < kNumMmaOrientInWarpTile; ++j) {
//                 #pragma unroll
//                 for (int k = 0; k < kFragmentCSize; ++k) {
//                     int m = fragment_c_m_idx_in_block<TParams>(warp_id, lane_id, i, k);
//                     int n = fragment_c_n_idx_in_block<TParams>(warp_id, lane_id, j, k);
//                     fragment_c[i][j][k] += s_trans_pow2_accumulator[m] + s_orient_pow2_accumulator[n];
//                 }
//             }
//         }
//         __syncthreads();

//     /*=============================== WRITE BACK ==============================*/
// 		epilogue();

//         scheduler.advance_to_next_work();
//     } // end of while has_work
// }




template<bool REF3D, bool DATA3D, int block_sz, int eulers_per_block, int prefetch_fraction>
__global__ void cuda_kernel_diff2_coarse(
		XFLOAT *g_eulers,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *trans_z,
		XFLOAT *g_real,
		XFLOAT *g_imag,
		AccProjectorKernel projector,
		XFLOAT *g_corr,
		XFLOAT *g_diff2s,
		int translation_num,
		int image_size
		)
{
	int tid = threadIdx.x;

	//Prefetch euler matrices
	__shared__ XFLOAT s_eulers[eulers_per_block * 9];

	int max_block_pass_euler( ceilfracf(eulers_per_block*9, block_sz) * block_sz);

	for (int i = tid; i < max_block_pass_euler; i += block_sz)
		if (i < eulers_per_block * 9)
			s_eulers[i] = g_eulers[blockIdx.x * eulers_per_block * 9 + i];


	//Setup variables
	__shared__ XFLOAT s_ref_real[block_sz/prefetch_fraction * eulers_per_block];
	__shared__ XFLOAT s_ref_imag[block_sz/prefetch_fraction * eulers_per_block];

	__shared__ XFLOAT s_real[block_sz];
	__shared__ XFLOAT s_imag[block_sz];
	__shared__ XFLOAT s_corr[block_sz];

	XFLOAT diff2s[eulers_per_block] = {0.f};

	XFLOAT tx = trans_x[tid%translation_num];
	XFLOAT ty = trans_y[tid%translation_num];
	XFLOAT tz = trans_z[tid%translation_num];

	//Step through data
	int max_block_pass_pixel( ceilfracf(image_size,block_sz) * block_sz );

	for (int init_pixel = 0; init_pixel < max_block_pass_pixel; init_pixel += block_sz/prefetch_fraction)
	{
		__syncthreads();

		//Prefetch block-fraction-wise
		if(init_pixel + tid/prefetch_fraction < image_size)
		{
			int x,y,z,xy;
			if(DATA3D)
			{
				assert(false);
				z =  floorfracf(init_pixel + tid/prefetch_fraction, projector.imgX*projector.imgY);
				xy = (init_pixel + tid/prefetch_fraction) % (projector.imgX*projector.imgY);
				x =             xy  % projector.imgX;
				y = floorfracf( xy,   projector.imgX);
				if (z > projector.maxR)
					z -= projector.imgZ;
			}
			else
			{
				x =           ( init_pixel + tid/prefetch_fraction) % projector.imgX;
				y = floorfracf( init_pixel + tid/prefetch_fraction  , projector.imgX);
			}
			if (y > projector.maxR)
				y -= projector.imgY;

//			#pragma unroll
			for (int i = tid%prefetch_fraction; i < eulers_per_block; i += prefetch_fraction)
			{
				if(DATA3D) // if DATA3D, then REF3D as well.
					projector.project3Dmodel(
						x,y,z,
						s_eulers[i*9  ],
						s_eulers[i*9+1],
						s_eulers[i*9+2],
						s_eulers[i*9+3],
						s_eulers[i*9+4],
						s_eulers[i*9+5],
						s_eulers[i*9+6],
						s_eulers[i*9+7],
						s_eulers[i*9+8],
						s_ref_real[eulers_per_block * (tid/prefetch_fraction) + i],
						s_ref_imag[eulers_per_block * (tid/prefetch_fraction) + i]);
				else if(REF3D)
					projector.project3Dmodel(
						x,y,
						s_eulers[i*9  ],
						s_eulers[i*9+1],
						s_eulers[i*9+3],
						s_eulers[i*9+4],
						s_eulers[i*9+6],
						s_eulers[i*9+7],
						s_ref_real[eulers_per_block * (tid/prefetch_fraction) + i],
						s_ref_imag[eulers_per_block * (tid/prefetch_fraction) + i]);
				else
					projector.project2Dmodel(
						x,y,
						s_eulers[i*9  ],
						s_eulers[i*9+1],
						s_eulers[i*9+3],
						s_eulers[i*9+4],
						s_ref_real[eulers_per_block * (tid/prefetch_fraction) + i],
						s_ref_imag[eulers_per_block * (tid/prefetch_fraction) + i]);
			}
		}

		//Prefetch block-wise
		if (init_pixel % block_sz == 0 && init_pixel + tid < image_size)
		{
			s_real[tid] = g_real[init_pixel + tid];
			s_imag[tid] = g_imag[init_pixel + tid];
			s_corr[tid] = g_corr[init_pixel + tid] / 2;
		}

		__syncthreads();

		if (tid/translation_num < block_sz/translation_num) // NOTE int division A/B==C/B !=> A==C
		for (int i = tid / translation_num;
				i < block_sz/prefetch_fraction;
				i += block_sz/translation_num)
		{
			if((init_pixel + i) >= image_size) break;

			int x,y,z,xy;
			if(DATA3D)
			{
				z =  floorfracf( init_pixel + i   ,  projector.imgX*projector.imgY); //TODO optimize index extraction.
				xy =           ( init_pixel + i ) % (projector.imgX*projector.imgY);
				x =             xy  % projector.imgX;
				y = floorfracf( xy,   projector.imgX);
				if (z > projector.maxR)
					z -= projector.imgZ;
			}
			else
			{
				x =           ( init_pixel + i ) % projector.imgX;
				y = floorfracf( init_pixel + i   , projector.imgX);
			}
			if (y > projector.maxR)
				y -= projector.imgY;

			XFLOAT real, imag;

			if(DATA3D)
				translatePixel(x, y, z, tx, ty, tz, s_real[i + init_pixel % block_sz], s_imag[i + init_pixel % block_sz], real, imag);
			else
				translatePixel(x, y,    tx, ty,     s_real[i + init_pixel % block_sz], s_imag[i + init_pixel % block_sz], real, imag);
			
			// double r2 = (x *x+ y* y)* projector.padding_factor * projector.padding_factor;
			// if(r2>projector.maxR2_padded)
			// {
			// 	real=0.;
			// 	imag=0.;
			// }

			#pragma unroll
			for (int j = 0; j < eulers_per_block; j ++)
			{
				XFLOAT diff_real =  s_ref_real[eulers_per_block * i + j] - real;
				XFLOAT diff_imag =  s_ref_imag[eulers_per_block * i + j] - imag;

				XFLOAT real1 = s_ref_real[eulers_per_block * i + j];
				XFLOAT imag1 = s_ref_imag[eulers_per_block * i + j];
				
				// int img_idx = init_pixel + i;

				// XFLOAT diff_real =  s_ref_real[eulers_per_block * i + j] - (float)(i + init_pixel);
				// XFLOAT diff_imag =  s_ref_imag[eulers_per_block * i + j] - (float)(i + init_pixel);

				// XFLOAT diff_real =  2. - (float)(i + init_pixel / (float)(tid % translation_num + 1));
				// XFLOAT diff_imag =  2. - (float)(i + init_pixel / (float)(tid % translation_num + 1));

				
				diff2s[j] += (diff_real * diff_real + diff_imag * diff_imag) * s_corr[i + init_pixel % block_sz];

				// diff2s[j] += (-2 * (s_ref_real[eulers_per_block * i + j] * real + s_ref_imag[eulers_per_block * i + j] * imag)) * s_corr[i + init_pixel % block_sz];
				// diff2s[j] += (real1 * real1 + imag1 * imag1 + real * real + imag * imag) * s_corr[i + init_pixel % block_sz];
				// diff2s[j] += -2 * s_corr[i + init_pixel % block_sz] * (real1 * real + imag1 * imag);
				// diff2s[j] += -2 *  ((img_idx % 11) * (img_idx % 7) + (img_idx % 17) * (img_idx % 11));
				// diff2s[j] += ((1 - 2) * (1 - 2) + (2 - 3) * (2 - 3)) * s_corr[i + init_pixel % block_sz];
				// diff2s[j] += ((1 - 2) * (1 - 2) + (2 - 3) * (2 - 3)) * ((i + init_pixel)% 2);
				// diff2s[j] += 0.0;
			}
		}
	}

	//Set global
	#pragma unroll
	for (int i = 0; i < eulers_per_block; i ++)
		cuda_atomic_add(&g_diff2s[(blockIdx.x * eulers_per_block + i) * translation_num + tid % translation_num], diff2s[i]);
}


template<bool REF3D, bool DATA3D, int block_sz, int chunk_sz>
__global__ void cuda_kernel_diff2_fine(
		XFLOAT *g_eulers,
		XFLOAT *g_imgs_real,
		XFLOAT *g_imgs_imag,
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *trans_z,
		AccProjectorKernel projector,
		XFLOAT *g_corr_img,
		XFLOAT *g_diff2s,
		unsigned image_size,
		XFLOAT sum_init,
		unsigned long orientation_num,
		unsigned long translation_num,
		unsigned long todo_blocks,
		unsigned long *d_rot_idx,
		unsigned long *d_trans_idx,
		unsigned long *d_job_idx,
		unsigned long *d_job_num
		)
{
	unsigned long bid = blockIdx.x;
	unsigned long tid = threadIdx.x;

//    // Specialize BlockReduce for a 1D block of 128 threads on type XFLOAT
//    typedef cub::BlockReduce<XFLOAT, 128> BlockReduce;
//    // Allocate shared memory for BlockReduce
//    __shared__ typename BlockReduce::TempStorage temp_storage;

	unsigned long pixel;
	XFLOAT ref_real, ref_imag,
		shifted_real, shifted_imag,
		diff_real, diff_imag;

	__shared__ XFLOAT s[block_sz*chunk_sz]; //We MAY have to do up to chunk_sz translations in each block
	__shared__ XFLOAT s_outs[chunk_sz];
	// inside the padded 2D orientation gri
//	if( bid < todo_blocks ) // we only need to make
	{
		unsigned trans_num  = (unsigned)d_job_num[bid]; //how many transes we have for this rot
		for (int itrans=0; itrans<trans_num; itrans++)
		{
			s[itrans*block_sz+tid] = (XFLOAT)0.0;
		}
		// index of comparison
		unsigned long int ix = d_rot_idx[d_job_idx[bid]];
		unsigned long int iy;
		unsigned pass_num(ceilfracf(image_size,block_sz));

		for (unsigned pass = 0; pass < pass_num; pass++) // finish an entire ref image each block
		{
			pixel = (pass * block_sz) + tid;

			if(pixel < image_size)
			{
				int x,y,z,xy;
				if(DATA3D)
				{
					z =  floorfracf(pixel, projector.imgX*projector.imgY);
					xy = pixel % (projector.imgX*projector.imgY);
					x =             xy  % projector.imgX;
					y = floorfracf( xy,   projector.imgX);
					if (z > projector.maxR)
					{
						if (z >= projector.imgZ - projector.maxR)
							z = z - projector.imgZ;
						else
							x = projector.maxR;
					}
				}
				else
				{
					x =             pixel % projector.imgX;
					y = floorfracf( pixel , projector.imgX);
				}
				if (y > projector.maxR)
				{
					if (y >= projector.imgY - projector.maxR)
						y = y - projector.imgY;
					else
						x = projector.maxR;
				}

				if(DATA3D)
					projector.project3Dmodel(
						x,y,z,
						__ldg(&g_eulers[ix*9  ]), __ldg(&g_eulers[ix*9+1]), __ldg(&g_eulers[ix*9+2]),
						__ldg(&g_eulers[ix*9+3]), __ldg(&g_eulers[ix*9+4]), __ldg(&g_eulers[ix*9+5]),
						__ldg(&g_eulers[ix*9+6]), __ldg(&g_eulers[ix*9+7]), __ldg(&g_eulers[ix*9+8]),
						ref_real, ref_imag);
				else if(REF3D)
					projector.project3Dmodel(
						x,y,
						__ldg(&g_eulers[ix*9  ]), __ldg(&g_eulers[ix*9+1]),
						__ldg(&g_eulers[ix*9+3]), __ldg(&g_eulers[ix*9+4]),
						__ldg(&g_eulers[ix*9+6]), __ldg(&g_eulers[ix*9+7]),
						ref_real, ref_imag);
				else
					projector.project2Dmodel(
						x,y,
						__ldg(&g_eulers[ix*9  ]), __ldg(&g_eulers[ix*9+1]),
						__ldg(&g_eulers[ix*9+3]), __ldg(&g_eulers[ix*9+4]),
						ref_real, ref_imag);

				for (int itrans=0; itrans<trans_num; itrans++) // finish all translations in each partial pass
				{
					iy = d_trans_idx[d_job_idx[bid]] + itrans;

					if(DATA3D)
						translatePixel(x, y, z, trans_x[iy], trans_y[iy], trans_z[iy], g_imgs_real[pixel], g_imgs_imag[pixel], shifted_real, shifted_imag);
					else
						translatePixel(x, y, trans_x[iy], trans_y[iy], g_imgs_real[pixel], g_imgs_imag[pixel], shifted_real, shifted_imag);
					// double r2 = (x *x+ y* y)* projector.padding_factor * projector.padding_factor;
					// if(r2>projector.maxR2_padded)
					// {
					// 	shifted_real=0.;
					// 	shifted_imag=0.;
					// }

					diff_real =  ref_real - shifted_real;
					diff_imag =  ref_imag - shifted_imag;
					s[itrans*block_sz + tid] += (diff_real * diff_real + diff_imag * diff_imag) * (XFLOAT)0.5 * __ldg(&g_corr_img[pixel]);
				}
			}
			__syncthreads();
		}
		for(int j=(block_sz/2); j>0; j/=2)
		{
			if(tid<j)
			{
				for (int itrans=0; itrans<trans_num; itrans++) // finish all translations in each partial pass
				{
					s[itrans*block_sz+tid] += s[itrans*block_sz+tid+j];
				}
			}
			__syncthreads();
		}
		if (tid < trans_num)
		{
			s_outs[tid]=s[tid*block_sz]+sum_init;
		}
		if (tid < trans_num)
		{
			iy=d_job_idx[bid]+tid;
			g_diff2s[iy] = s_outs[tid];
		}
	}
}




/*
 *   	CROSS-CORRELATION-BASED KERNELS
 */

template<bool REF3D, bool DATA3D, int block_sz>
__global__ void cuda_kernel_diff2_CC_coarse(
		XFLOAT *g_eulers,
		XFLOAT *g_imgs_real,
		XFLOAT *g_imgs_imag,
		XFLOAT *g_trans_x,
		XFLOAT *g_trans_y,
		XFLOAT *g_trans_z,
		AccProjectorKernel projector,
		XFLOAT *g_corr_img,
		XFLOAT *g_diff2s,
		unsigned translation_num,
		int image_size,
		XFLOAT exp_local_sqrtXi2
		)
{

	int iorient = blockIdx.x;
	int itrans =  blockIdx.y;
	int tid = threadIdx.x;

    __shared__ XFLOAT s_weight[block_sz];
    s_weight[tid] = (XFLOAT)0.0;
	__shared__ XFLOAT s_norm[block_sz];
	s_norm[tid] = (XFLOAT)0.0;

	XFLOAT real, imag, ref_real, ref_imag;

	XFLOAT e0,e1,e2,e3,e4,e5,e6,e7,e8;
	e0 = __ldg(&g_eulers[iorient*9  ]);
	e1 = __ldg(&g_eulers[iorient*9+1]);
	e2 = __ldg(&g_eulers[iorient*9+2]);
	e3 = __ldg(&g_eulers[iorient*9+3]);
	e4 = __ldg(&g_eulers[iorient*9+4]);
	e5 = __ldg(&g_eulers[iorient*9+5]);
	e6 = __ldg(&g_eulers[iorient*9+6]);
	e7 = __ldg(&g_eulers[iorient*9+7]);
	e8 = __ldg(&g_eulers[iorient*9+8]);

	__syncthreads();

	unsigned pixel_pass_num( ceilfracf(image_size,block_sz) );
	for (unsigned pass = 0; pass < pixel_pass_num; pass++)
	{
		unsigned pixel = (pass * block_sz) + tid;

		if(pixel < image_size)
		{
			int x,y,z,xy;
			if(DATA3D)
			{
				z =  floorfracf(pixel, projector.imgX*projector.imgY);
				xy = pixel % (projector.imgX*projector.imgY);
				x =             xy  % projector.imgX;
				y = floorfracf( xy,   projector.imgX);
				if (z > projector.maxR)
				{
					if (z >= projector.imgZ - projector.maxR)
						z = z - projector.imgZ;
					else
						x = projector.maxR;
				}
			}
			else
			{
				x =             pixel % projector.imgX;
				y = floorfracf( pixel , projector.imgX);
			}
			if (y > projector.maxR)
			{
				if (y >= projector.imgY - projector.maxR)
					y = y - projector.imgY;
				else
					x = projector.maxR;
			}

			if(DATA3D)
				projector.project3Dmodel(
					x,y,z,
					e0,e1,e2,e3,e4,e5,e6,e7,e8,
					ref_real, ref_imag);
			else if(REF3D)
				projector.project3Dmodel(
					x,y,
					e0,e1,e3,e4,e6,e7,
					ref_real, ref_imag);
			else
				projector.project2Dmodel(
					x,y,
					e0,e1,e3,e4,
					ref_real, ref_imag);

			if(DATA3D)
				translatePixel(x, y, z, g_trans_x[itrans], g_trans_y[itrans], g_trans_z[itrans], g_imgs_real[pixel], g_imgs_imag[pixel], real, imag);
			else
				translatePixel(x, y,    g_trans_x[itrans], g_trans_y[itrans],                    g_imgs_real[pixel], g_imgs_imag[pixel], real, imag);

			s_weight[tid] += (ref_real * real     + ref_imag * imag)      * __ldg(&g_corr_img[pixel]);
			s_norm[tid]   += (ref_real * ref_real + ref_imag * ref_imag ) * __ldg(&g_corr_img[pixel]);
		}
		__syncthreads();
	}


	for(int j=(block_sz/2); j>0; j/=2)
	{
		if(tid<j)
		{
			s_weight[tid] += s_weight[tid+j];
			s_norm[tid]   += s_norm[tid+j];
		}
		__syncthreads();
	}
#ifdef ACC_DOUBLE_PRECISION
	g_diff2s[iorient * translation_num + itrans] = - ( s_weight[0] / sqrt(s_norm[0]));
#else
	g_diff2s[iorient * translation_num + itrans] = - ( s_weight[0] / sqrtf(s_norm[0]));
#endif
}

template<bool REF3D, bool DATA3D, int block_sz,int chunk_sz>
__global__ void cuda_kernel_diff2_CC_fine(
		XFLOAT *g_eulers,
		XFLOAT *g_imgs_real,
		XFLOAT *g_imgs_imag,
		XFLOAT *g_trans_x,
		XFLOAT *g_trans_y,
		XFLOAT *g_trans_z,
		AccProjectorKernel projector,
		XFLOAT *g_corr_img,
		XFLOAT *g_diff2s,
		unsigned image_size,
		XFLOAT sum_init,
		XFLOAT exp_local_sqrtXi2,
		unsigned long orientation_num,
		unsigned long translation_num,
		unsigned long todo_blocks,
		unsigned long *d_rot_idx,
		unsigned long *d_trans_idx,
		unsigned long *d_job_idx,
		unsigned long *d_job_num
		)
{
	int bid = blockIdx.y * gridDim.x + blockIdx.x;
	int tid = threadIdx.x;

//    // Specialize BlockReduce for a 1D block of 128 threads on type XFLOAT
//    typedef cub::BlockReduce<XFLOAT, 128> BlockReduce;
//    // Allocate shared memory for BlockReduce
//    __shared__ typename BlockReduce::TempStorage temp_storage;

	int pixel;
	XFLOAT ref_real, ref_imag, shifted_real, shifted_imag;

	__shared__ XFLOAT      s[block_sz*chunk_sz]; //We MAY have to do up to chunk_sz translations in each block
	__shared__ XFLOAT   s_cc[block_sz*chunk_sz];
	__shared__ XFLOAT s_outs[chunk_sz];

	if( bid < todo_blocks ) // we only need to make
	{
		unsigned trans_num   = d_job_num[bid]; //how many transes we have for this rot
		for (int itrans=0; itrans<trans_num; itrans++)
		{
			s[   itrans*block_sz+tid] = 0.0f;
			s_cc[itrans*block_sz+tid] = 0.0f;
		}
		__syncthreads();
		// index of comparison
		unsigned long int ix = d_rot_idx[d_job_idx[bid]];
		unsigned long int iy;
		unsigned pass_num( ceilfracf(image_size,block_sz) );

		for (unsigned pass = 0; pass < pass_num; pass++) // finish an entire ref image each block
		{
			pixel = (pass * block_sz) + tid;

			if(pixel < image_size)
			{
				int x,y,z,xy;
				if(DATA3D)
				{
					z =  floorfracf(pixel, projector.imgX*projector.imgY);
					xy = pixel % (projector.imgX*projector.imgY);
					x =             xy  % projector.imgX;
					y = floorfracf( xy,   projector.imgX);
					if (z > projector.maxR)
					{
						if (z >= projector.imgZ - projector.maxR)
							z = z - projector.imgZ;
						else
							x = projector.maxR;
					}
				}
				else
				{
					x =             pixel % projector.imgX;
					y = floorfracf( pixel , projector.imgX);
				}

				if (y > projector.maxR)
				{
					if (y >= projector.imgY - projector.maxR)
						y = y - projector.imgY;
					else
						x = projector.maxR;
				}

				if(DATA3D)
					projector.project3Dmodel(
						x,y,z,
						__ldg(&g_eulers[ix*9  ]), __ldg(&g_eulers[ix*9+1]), __ldg(&g_eulers[ix*9+2]),
						__ldg(&g_eulers[ix*9+3]), __ldg(&g_eulers[ix*9+4]), __ldg(&g_eulers[ix*9+5]),
						__ldg(&g_eulers[ix*9+6]), __ldg(&g_eulers[ix*9+7]), __ldg(&g_eulers[ix*9+8]),
						ref_real, ref_imag);
				else if(REF3D)
					projector.project3Dmodel(
						x,y,
						__ldg(&g_eulers[ix*9  ]), __ldg(&g_eulers[ix*9+1]),
						__ldg(&g_eulers[ix*9+3]), __ldg(&g_eulers[ix*9+4]),
						__ldg(&g_eulers[ix*9+6]), __ldg(&g_eulers[ix*9+7]),
						ref_real, ref_imag);
				else
					projector.project2Dmodel(
						x,y,
						__ldg(&g_eulers[ix*9  ]), __ldg(&g_eulers[ix*9+1]),
						__ldg(&g_eulers[ix*9+3]), __ldg(&g_eulers[ix*9+4]),
						ref_real, ref_imag);

				for (int itrans=0; itrans<trans_num; itrans++) // finish all translations in each partial pass
				{
					iy = d_trans_idx[d_job_idx[bid]] + itrans;

					if(DATA3D)
						translatePixel(x, y, z, g_trans_x[iy], g_trans_y[iy], g_trans_z[iy], g_imgs_real[pixel], g_imgs_imag[pixel], shifted_real, shifted_imag);
					else
						translatePixel(x, y,    g_trans_x[iy], g_trans_y[iy],                g_imgs_real[pixel], g_imgs_imag[pixel], shifted_real, shifted_imag);

					s[   itrans*block_sz + tid] += (ref_real * shifted_real + ref_imag * shifted_imag) * __ldg(&g_corr_img[pixel]);
					s_cc[itrans*block_sz + tid] += (ref_real*ref_real + ref_imag*ref_imag) * __ldg(&g_corr_img[pixel]);
				}
			}
			__syncthreads();
		}
		for(int j=(block_sz/2); j>0; j/=2)
		{
			if(tid<j)
			{
				for (int itrans=0; itrans<trans_num; itrans++) // finish all translations in each partial pass
				{
					s[   itrans*block_sz+tid] += s[   itrans*block_sz+tid+j];
					s_cc[itrans*block_sz+tid] += s_cc[itrans*block_sz+tid+j];
				}
			}
			__syncthreads();
		}
		if (tid < trans_num)
		{
#ifdef ACC_DOUBLE_PRECISION
			s_outs[tid]= - s[tid*block_sz] / (sqrt(s_cc[tid*block_sz]));
#else
			s_outs[tid]= - s[tid*block_sz] / (sqrtf(s_cc[tid*block_sz]));
#endif
		}
		if (tid < trans_num)
		{
			iy=d_job_idx[bid]+tid;
			g_diff2s[iy] = s_outs[tid];
		}
	}
}
#endif /* CUDA_DIFF2_KERNELS_CUH_ */
