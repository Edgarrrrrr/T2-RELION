#ifndef MMA_UTILS_CUH
#define MMA_UTILS_CUH

#include <assert.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <type_traits>




// tiling in the translation dimension and the orientation dimension
// tile size is kTransBlockSize * kOrientBlockSize * image_size

// The default value of kImgBlockSize is 16 for the reason that matrix trans and orient have the real and imag part.
// At the level of Block, we use shared memory to store these two matrices, which means the shared memory size is
// 2 * kTransBlockSize * kImgBlockSize + 2 * kOrientBlockSize * kImgBlockSize = 32KB.
// Default use 16x8x8 tf32 MMA instruction.
// 16x8x8 tf32 Layout:
/**

                                                                                
                                                                                       
                                                                                        
                                                                              ↑  +---+---+---+---+---+---+---+---+       
							                                    		    							b0   32b | T | T | T | T | T | T | T | T |       
							                                    		    										|  | 0 | 4 | 8 | 12| 16| 20| 24| 28|
							                                    		    										↓  +---+---+---+---+---+---+---+---+
							                                    		    										   | T | T | T | T | T | T | T | T |
							                                    		    										   | 1 | 5 | 9 | 13| 17| 21| 25| 29|
							                                    		    										   +---+---+---+---+---+---+---+---+
							                                    		    										   | T | T | T | T | T | T | T | T |
							                                    		    										   | 2 | 6 | 10| 14| 18| 22| 26| 30|
							                                    		    										   +---+---+---+---+---+---+---+---+
							                                    		    										   | T | T | T | T | T | T | T | T |
							                                    		    										   | 3 | 7 | 11| 15| 19| 23| 27| 31|
							                                    		    										   +---+---+---+---+---+---+---+---+
                                         
                                                                                 +---+---+---+---+---+---+---+---+       
							                                    		    							b1       | T | T | T | T | T | T | T | T |       
							                                    		    										   | 0 | 4 | 8 | 12| 16| 20| 24| 28|
							                                    		    										   +---+---+---+---+---+---+---+---+
							                                    		    										   | T | T | T | T | T | T | T | T |
							                                    		    										   | 1 | 5 | 9 | 13| 17| 21| 25| 29|
							                                    		    										   +---+---+---+---+---+---+---+---+
							                                    		    										   | T | T | T | T | T | T | T | T |
							                                    		    										   | 2 | 6 | 10| 14| 18| 22| 26| 30|
							                                    		    										   +---+---+---+---+---+---+---+---+
							                                    		    										   | T | T | T | T | T | T | T | T |
							                                    		    										   | 3 | 7 | 11| 15| 19| 23| 27| 31|
							                                    		    										   +---+---+---+---+---+---+---+---+
    
	<--32b-->						    										   <--64b-->	   
  +-------+-------+-------+-------+       +-------+-------+-------+-------+      +-------+-------+-------+-------+
a0|  T0   |  T1   |  T2   |  T3   |   a2  |  T0   |  T1   |  T2   |  T3   |      |  T0   |  T1   |  T2   |  T3   |  c0 c1
  +-------+-------+-------+-------+       +-------+-------+-------+-------+      +-------+-------+-------+-------+
	|  T4   |  T5   |  T6   |  T7   |       |  T4   |  T5   |  T6   |  T7   |      |  T4   |  T5   |  T6   |  T7   |
	+-------+-------+-------+-------+       +-------+-------+-------+-------+      +-------+-------+-------+-------+
	|  T8   |  T9   |  T10  |  T11  |       |  T8   |  T9   |  T10  |  T11  |      |  T8   |  T9   |  T10  |  T11  |
	+-------+-------+-------+-------+       +-------+-------+-------+-------+      +-------+-------+-------+-------+
	|  T12  |  T13  |  T14  |  T15  |       |  T12  |  T13  |  T14  |  T15  |      |  T12  |  T13  |  T14  |  T15  |
	+-------+-------+-------+-------+       +-------+-------+-------+-------+      +-------+-------+-------+-------+
	|  T16  |  T17  |  T18  |  T19  |       |  T16  |  T17  |  T18  |  T19  |      |  T16  |  T17  |  T18  |  T19  |
	+-------+-------+-------+-------+       +-------+-------+-------+-------+      +-------+-------+-------+-------+
	|  T20  |  T21  |  T22  |  T23  |       |  T20  |  T21  |  T22  |  T23  |      |  T20  |  T21  |  T22  |  T23  |
	+-------+-------+-------+-------+       +-------+-------+-------+-------+      +-------+-------+-------+-------+
	|  T24  |  T25  |  T26  |  T27  |       |  T24  |  T25  |  T26  |  T27  |      |  T24  |  T25  |  T26  |  T27  |
	+-------+-------+-------+-------+       +-------+-------+-------+-------+      +-------+-------+-------+-------+
	|  T28  |  T29  |  T30  |  T31  |       |  T28  |  T29  |  T30  |  T31  |      |  T28  |  T29  |  T30  |  T31  |
	+-------+-------+-------+-------+       +-------+-------+-------+-------+      +-------+-------+-------+-------+
                                                                                                                                 
  +-------+-------+-------+-------+       +-------+-------+-------+-------+      +-------+-------+-------+-------+
a1|  T0   |  T1   |  T2   |  T3   |   a3  |  T0   |  T1   |  T2   |  T3   |      |  T0   |  T1   |  T2   |  T3   |  c2 c3
  +-------+-------+-------+-------+       +-------+-------+-------+-------+      +-------+-------+-------+-------+
	|  T4   |  T5   |  T6   |  T7   |       |  T4   |  T5   |  T6   |  T7   |      |  T4   |  T5   |  T6   |  T7   |
	+-------+-------+-------+-------+       +-------+-------+-------+-------+      +-------+-------+-------+-------+
	|  T8   |  T9   |  T10  |  T11  |       |  T8   |  T9   |  T10  |  T11  |      |  T8   |  T9   |  T10  |  T11  |
	+-------+-------+-------+-------+       +-------+-------+-------+-------+      +-------+-------+-------+-------+
	|  T12  |  T13  |  T14  |  T15  |       |  T12  |  T13  |  T14  |  T15  |      |  T12  |  T13  |  T14  |  T15  |
	+-------+-------+-------+-------+       +-------+-------+-------+-------+      +-------+-------+-------+-------+
	|  T16  |  T17  |  T18  |  T19  |       |  T16  |  T17  |  T18  |  T19  |      |  T16  |  T17  |  T18  |  T19  |
	+-------+-------+-------+-------+       +-------+-------+-------+-------+      +-------+-------+-------+-------+
	|  T20  |  T21  |  T22  |  T23  |       |  T20  |  T21  |  T22  |  T23  |      |  T20  |  T21  |  T22  |  T23  |
	+-------+-------+-------+-------+       +-------+-------+-------+-------+      +-------+-------+-------+-------+
	|  T24  |  T25  |  T26  |  T27  |       |  T24  |  T25  |  T26  |  T27  |      |  T24  |  T25  |  T26  |  T27  |
	+-------+-------+-------+-------+       +-------+-------+-------+-------+      +-------+-------+-------+-------+
	|  T28  |  T29  |  T30  |  T31  |       |  T28  |  T29  |  T30  |  T31  |      |  T28  |  T29  |  T30  |  T31  |
	+-------+-------+-------+-------+       +-------+-------+-------+-------+      +-------+-------+-------+-------+

 *
 *
 *
 */
//  struct CoarseKernelBlockTParams {
//     static constexpr int kBlockSize = 128;
//     static constexpr int kTransBlockSize = 64;
//     static constexpr int kOrientBlockSize = 128;
//     static constexpr int kImgBlockSize = 16;
//     static constexpr int kWarpTransTileSize = 32;
//     static constexpr int kWarpOrientTileSize = 64;
//     static constexpr int kWarpImgTileSize = 16;
//     static constexpr int kMmaTransTileSize = 16;
//     static constexpr int kMmaOrientTileSize = 8;
//     static constexpr int kMmaImgTileSize = 8;
// };

template <int BlockSize, int TransBlockSize, int OrientBlockSize, int ImgBlockSize, 
          int WarpTransTileSize, int WarpOrientTileSize, int WarpImgTileSize, 
          int MmaTransTileSize, int MmaOrientTileSize, int MmaImgTileSize>
struct CoarseKernelBlockTParams {
    static constexpr int kBlockSize = BlockSize;
    static constexpr int kTransBlockSize = TransBlockSize;
    static constexpr int kOrientBlockSize = OrientBlockSize;
    static constexpr int kImgBlockSize = ImgBlockSize;
    static constexpr int kWarpTransTileSize = WarpTransTileSize;
    static constexpr int kWarpOrientTileSize = WarpOrientTileSize;
    static constexpr int kWarpImgTileSize = WarpImgTileSize;
    static constexpr int kMmaTransTileSize = MmaTransTileSize;
    static constexpr int kMmaOrientTileSize = MmaOrientTileSize;
    static constexpr int kMmaImgTileSize = MmaImgTileSize;
};

constexpr int kWarpSize = 32;

// five level : global -> block -> warp_tile -> mma_tile -> fragment

template<typename TParams>
__device__ __forceinline__ int warp_tile_m_idx_in_block(int warp_id) {
	constexpr int kNumWarpTransTileInBlock = TParams::kTransBlockSize / TParams::kWarpTransTileSize;
	return warp_id % kNumWarpTransTileInBlock;
}

template<typename TParams>
__device__ __forceinline__ int warp_tile_n_idx_in_block(int warp_id)
{
	constexpr int kNumWarpTransTileInBlock = TParams::kTransBlockSize / TParams::kWarpTransTileSize;
	return warp_id / kNumWarpTransTileInBlock;
}

template<typename TParams>
__device__ __forceinline__ int mma_tile_m_idx_in_block(int warp_id, int mma_m_idx_in_warp_tile) {
	constexpr int kNumMmaTransInWarpTile = TParams::kWarpTransTileSize / TParams::kMmaTransTileSize;
	assert(mma_m_idx_in_warp_tile < kNumMmaTransInWarpTile);
	return warp_tile_m_idx_in_block<TParams>(warp_id) * kNumMmaTransInWarpTile + mma_m_idx_in_warp_tile; 
}

template<typename TParams>
__device__ __forceinline__ int mma_tile_n_idx_in_block(int warp_id, int mma_n_idx_in_warp_tile) {
	constexpr int kNumMmaOrientInWarpTile = TParams::kWarpOrientTileSize / TParams::kMmaOrientTileSize;
	assert(mma_n_idx_in_warp_tile < kNumMmaOrientInWarpTile);
	return warp_tile_n_idx_in_block<TParams>(warp_id) * kNumMmaOrientInWarpTile + mma_n_idx_in_warp_tile;
}

template<typename TParams>
__device__ __forceinline__ int warp_tile_k_idx_in_block(int warp_id) {
	constexpr int kNumWarpImgTileInBlock = TParams::kImgBlockSize / TParams::kWarpImgTileSize;
	assert(kNumWarpImgTileInBlock == 1);
	return 0;
}

template<typename TParams>
__device__ __forceinline__ int mma_tile_k_idx_in_block(int warp_id, int mma_k_idx_in_warp_tile) {
	constexpr int kNumMmaImgInWarpTile = TParams::kWarpImgTileSize / TParams::kMmaImgTileSize;
	assert(mma_k_idx_in_warp_tile < kNumMmaImgInWarpTile);
	return warp_tile_k_idx_in_block<TParams>(warp_id) * kNumMmaImgInWarpTile + mma_k_idx_in_warp_tile;
}

// different MMA operator may have different fragment size and layout
// here we use 16x8x8 tf32 MMA operator
// The computing method of the row and column of a matrix fragment c showed in PTX document is as follows:
// groupID           = laneid >> 2
// threadID_in_group = laneid % 4 
// row =  groupID      for c0 and c1
//        groupID + 8  for c2 and c3
// col =  (threadID_in_group * 2) + (i & 0x1) for ci   where i = {0,...,3}
template<typename TParams>
__device__ __forceinline__ int fragment_c_m_idx_in_mma_tile(int lane_id, int fragment_idx) {
	constexpr int kFragmentCSize = TParams::kMmaTransTileSize * TParams::kMmaOrientTileSize / kWarpSize;
	assert(fragment_idx < kFragmentCSize);
	return (lane_id / 4) + 8 * (fragment_idx / 2);
}

template<typename TParams>
__device__ __forceinline__ int fragment_c_n_idx_in_mma_tile(int lane_id, int fragment_idx) {
	constexpr int kFragmentCSize = TParams::kMmaTransTileSize * TParams::kMmaOrientTileSize / kWarpSize;
	assert(fragment_idx < kFragmentCSize);
	return (lane_id % 4) * 2 + (fragment_idx & 0x1);
}

// The computing method of the row and column of a matrix fragment a showed in PTX document is as follows:
// groupID           = laneid >> 2
// threadID_in_group = laneid % 4 
// row =  groupID               for a0 and a2
//        groupID + 8           for a1 and a3
// col =  threadID_in_group     for a0 and a1
//        threadID_in_group + 4 for a2 and a3
template<typename TParams>
__device__ __forceinline__ int fragment_a_m_idx_in_mma_tile(int lane_id, int fragment_idx) {
	constexpr int kFragmentASize = TParams::kMmaTransTileSize * TParams::kMmaImgTileSize / kWarpSize;
	assert(fragment_idx < kFragmentASize);
	return (lane_id / 4) + 8 * (fragment_idx % 2);
}

template<typename TParams>
__device__ __forceinline__ int fragment_a_k_idx_in_mma_tile(int lane_id, int fragment_idx) {
	constexpr int kFragmentASize = TParams::kMmaTransTileSize * TParams::kMmaImgTileSize / kWarpSize;
	assert(fragment_idx < kFragmentASize);
	return (lane_id % 4) + 4 * (fragment_idx / 2);
}


// The computing method of the row and column of a matrix fragment b showed in PTX document is as follows:
// groupID           = laneid >> 2
// threadID_in_group = laneid % 4
// row =  threadID_in_group     for b0 
//        threadID_in_group + 4 for b1
// col =  groupID               
template<typename TParams>
__device__ __forceinline__ int fragment_b_k_idx_in_mma_tile(int lane_id, int fragment_idx) {
	constexpr int kFragmentBSize = TParams::kMmaOrientTileSize * TParams::kMmaImgTileSize / kWarpSize;
	assert(fragment_idx < kFragmentBSize);
	return (lane_id % 4) + 4 * (fragment_idx % 2);
}

template<typename TParams>
__device__ __forceinline__ int fragment_b_n_idx_in_mma_tile(int lane_id, int fragment_idx) {
	constexpr int kFragmentBSize = TParams::kMmaOrientTileSize * TParams::kMmaImgTileSize / kWarpSize;
	assert(fragment_idx < kFragmentBSize);
	return (lane_id / 4);
}

template<typename TParams>
__device__ __forceinline__ int fragment_c_m_idx_in_block(int warp_id, int lane_id, int mma_m_idx_in_warp_tile, int fragment_idx) {
	constexpr int kFragmentCSize = TParams::kMmaTransTileSize * TParams::kMmaOrientTileSize / kWarpSize;
	assert(fragment_idx < kFragmentCSize);

	return mma_tile_m_idx_in_block<TParams>(warp_id, mma_m_idx_in_warp_tile) * TParams::kMmaTransTileSize + fragment_c_m_idx_in_mma_tile<TParams>(lane_id, fragment_idx);
}

template<typename TParams>
__device__ __forceinline__ int fragment_c_n_idx_in_block(int warp_id, int lane_id, int mma_n_idx_in_warp_tile, int fragment_idx) {
	constexpr int kFragmentCSize = TParams::kMmaTransTileSize * TParams::kMmaOrientTileSize / kWarpSize;
	assert(fragment_idx < kFragmentCSize);

	return mma_tile_n_idx_in_block<TParams>(warp_id, mma_n_idx_in_warp_tile) * TParams::kMmaOrientTileSize + fragment_c_n_idx_in_mma_tile<TParams>(lane_id, fragment_idx);
}

template<typename TParams>
__device__ __forceinline__ int fragment_a_m_idx_in_block(int warp_id, int lane_id, int mma_m_idx_in_warp_tile, int fragment_idx) {
	constexpr int kFragmentASize = TParams::kMmaTransTileSize * TParams::kMmaImgTileSize / kWarpSize;
	assert(fragment_idx < kFragmentASize);

	return mma_tile_m_idx_in_block<TParams>(warp_id, mma_m_idx_in_warp_tile) * TParams::kMmaTransTileSize + fragment_a_m_idx_in_mma_tile<TParams>(lane_id, fragment_idx);
}

template<typename TParams>
__device__ __forceinline__ int fragment_a_k_idx_in_block(int warp_id, int lane_id, int mma_k_idx_in_warp_tile, int fragment_idx) {
	constexpr int kFragmentASize = TParams::kMmaTransTileSize * TParams::kMmaImgTileSize / kWarpSize;
	assert(fragment_idx < kFragmentASize);

	return mma_tile_k_idx_in_block<TParams>(warp_id, mma_k_idx_in_warp_tile) * TParams::kMmaImgTileSize + fragment_a_k_idx_in_mma_tile<TParams>(lane_id, fragment_idx);
}

template<typename TParams>
__device__ __forceinline__ int fragment_b_k_idx_in_block(int warp_id, int lane_id, int mma_k_idx_in_warp_tile, int fragment_idx) {
	constexpr int kFragmentBSize = TParams::kMmaOrientTileSize * TParams::kMmaImgTileSize / kWarpSize;
	assert(fragment_idx < kFragmentBSize);

	return mma_tile_k_idx_in_block<TParams>(warp_id, mma_k_idx_in_warp_tile) * TParams::kMmaImgTileSize + fragment_b_k_idx_in_mma_tile<TParams>(lane_id, fragment_idx);
}

template<typename TParams>
__device__ __forceinline__ int fragment_b_n_idx_in_block(int warp_id, int lane_id, int mma_n_idx_in_warp_tile, int fragment_idx) {
	constexpr int kFragmentBSize = TParams::kMmaOrientTileSize * TParams::kMmaImgTileSize / kWarpSize;
	assert(fragment_idx < kFragmentBSize);

	return mma_tile_n_idx_in_block<TParams>(warp_id, mma_n_idx_in_warp_tile) * TParams::kMmaOrientTileSize + fragment_b_n_idx_in_mma_tile<TParams>(lane_id, fragment_idx);
}

template<typename TParams>
__device__ __forceinline__ int fragment_c_m_idx_in_global(int g_offset_m, int warp_id, int lane_id, int mma_m_idx_in_warp_tile, int fragment_idx) {
	return g_offset_m + fragment_c_m_idx_in_block<TParams>(warp_id, lane_id, mma_m_idx_in_warp_tile, fragment_idx);
}

template<typename TParams>
__device__ __forceinline__ int fragment_c_n_idx_in_global(int g_offset_n, int warp_id, int lane_id, int mma_n_idx_in_warp_tile, int fragment_idx) {
	return g_offset_n + fragment_c_n_idx_in_block<TParams>(warp_id, lane_id, mma_n_idx_in_warp_tile, fragment_idx);
}


// ============================================================================
// tensor core function wrapper
// ============================================================================
__device__ __forceinline__ void mma_sync_aligned_m16n8k8_row_col_tf32x3(
  float C[4], uint32_t const A_big[4], uint32_t const A_small[4],
  uint32_t const B_big[2], uint32_t const B_small[2]) {
asm("{\n\t"
    // Compute C += A_hi * B_hi
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
    "{%0, %1, %2, %3},\n\t"
    "{%4, %5, %6, %7},\n\t"
    "{%12, %13},\n\t"
    "{%0, %1, %2, %3};\n\t"

    // Compute C += A_hi * B_lo
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
    "{%0, %1, %2, %3},\n\t"
    "{%4, %5, %6, %7},\n\t"
    "{%14, %15},\n\t"
    "{%0, %1, %2, %3};\n\t"

    // Compute C += A_lo * B_hi
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
    "{%0, %1, %2, %3},\n\t"
    "{%8, %9, %10, %11},\n\t"
    "{%12, %13},\n\t"
    "{%0, %1, %2, %3};\n\t"

    "}\n\t"

    : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
    : "r"(A_big[0]), "r"(A_big[1]), "r"(A_big[2]), "r"(A_big[3]),
      "r"(A_small[0]), "r"(A_small[1]), "r"(A_small[2]), "r"(A_small[3]),
      "r"(B_big[0]), "r"(B_big[1]), "r"(B_small[0]), "r"(B_small[1]));
}

__device__ __forceinline__ void mma_sync_aligned_m16n8k8_row_col_tf32x4(
  float C[4], uint32_t const A_big[4], uint32_t const A_small[4],
  uint32_t const B_big[2], uint32_t const B_small[2]) {
asm("{\n\t"
    // Compute C += A_hi * B_hi
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
    "{%0, %1, %2, %3},\n\t"
    "{%4, %5, %6, %7},\n\t"
    "{%12, %13},\n\t"
    "{%0, %1, %2, %3};\n\t"

    // Compute C += A_hi * B_lo
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
    "{%0, %1, %2, %3},\n\t"
    "{%4, %5, %6, %7},\n\t"
    "{%14, %15},\n\t"
    "{%0, %1, %2, %3};\n\t"

    // Compute C += A_lo * B_hi
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
    "{%0, %1, %2, %3},\n\t"
    "{%8, %9, %10, %11},\n\t"
    "{%12, %13},\n\t"
    "{%0, %1, %2, %3};\n\t"

  // Compute C += A_lo * B_lo
  "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
  "{%0, %1, %2, %3},\n\t"
  "{%8, %9, %10, %11},\n\t"
  "{%14, %15},\n\t"
  "{%0, %1, %2, %3};\n\t"

    "}\n\t"

    : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
    : "r"(A_big[0]), "r"(A_big[1]), "r"(A_big[2]), "r"(A_big[3]),
      "r"(A_small[0]), "r"(A_small[1]), "r"(A_small[2]), "r"(A_small[3]),
      "r"(B_big[0]), "r"(B_big[1]), "r"(B_small[0]), "r"(B_small[1]));
}


__device__ __forceinline__ void mma_sync_aligned_m16n8k8_row_col_tf32x1(
  float C[4], uint32_t const A_big[4], uint32_t const A_small[4],
  uint32_t const B_big[2], uint32_t const B_small[2]) {
// mark xx_small as unused, ignore compiler warning
(void)A_small;
(void)B_small;

asm("{\n\t"
    // Compute C += A_hi * B_hi
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
    "{%0, %1, %2, %3},\n\t"
    "{%4, %5, %6, %7},\n\t"
    "{%8, %9},\n\t"
    "{%0, %1, %2, %3};\n\t"
    "}\n\t"

    : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
    : "r"(A_big[0]), "r"(A_big[1]), "r"(A_big[2]), "r"(A_big[3]),
      "r"(B_big[0]), "r"(B_big[1]));

}


// ============================================================================
// tensor core function wrapper #old#
// ============================================================================
__device__ __forceinline__ void mma_sync_aligned_m16n8k8_row_col_tf32(
  float C[4], float const A[4], float const B[2]) {
asm("{\n\t"
    ".reg .b32 tf_a<4>, tf_b<2>; \n\t"
    "cvt.rna.tf32.f32 tf_a0, %4;\n\t"
    "cvt.rna.tf32.f32 tf_a1, %5;\n\t"
    "cvt.rna.tf32.f32 tf_a2, %6;\n\t"
    "cvt.rna.tf32.f32 tf_a3, %7;\n\t"
    "cvt.rna.tf32.f32 tf_b0, %8;\n\t"
    "cvt.rna.tf32.f32 tf_b1, %9;\n\t"
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
    "{%0, %1, %2, %3},\n\t"
    "{tf_a0, tf_a1, tf_a2, tf_a3},\n\t"
    "{tf_b0, tf_b1},\n\t"
    "{%0, %1, %2, %3};\n\t"
    "}\n\t"
    : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
    : "f"(A[0]), "f"(A[1]), "f"(A[2]), "f"(A[3]), "f"(B[0]), "f"(B[1]));
}

__device__ __forceinline__ void
mma_sync_aligned_m16n8k8_row_col_tf32_simulated_fp32(float C[4],
                                                   float const A[4],
                                                   float const B[2]) {
asm("{\n\t"
    ".reg .b32 tf_a_hi<4>, tf_b_hi<2>; \n\t"
    ".reg .b32 tf_a_lo<4>, tf_b_lo<2>; \n\t"
    ".reg .f32 a_hi<4>, a_lo<4>, b_hi<2>, b_lo<2>; \n\t"

    "cvt.rna.tf32.f32 tf_a_hi0, %4;\n\t"
    "cvt.rna.tf32.f32 tf_a_hi1, %5;\n\t"
    "cvt.rna.tf32.f32 tf_a_hi2, %6;\n\t"
    "cvt.rna.tf32.f32 tf_a_hi3, %7;\n\t"
    "cvt.rna.tf32.f32 tf_b_hi0, %8;\n\t"
    "cvt.rna.tf32.f32 tf_b_hi1, %9;\n\t"

    "mov.f32 a_hi0, tf_a_hi0;\n\t"
    "mov.f32 a_hi1, tf_a_hi1;\n\t"
    "mov.f32 a_hi2, tf_a_hi2;\n\t"
    "mov.f32 a_hi3, tf_a_hi3;\n\t"
    "mov.f32 b_hi0, tf_b_hi0;\n\t"
    "mov.f32 b_hi1, tf_b_hi1;\n\t"

    "sub.f32 a_lo0, %4, a_hi0;\n\t"
    "sub.f32 a_lo1, %5, a_hi1;\n\t"
    "sub.f32 a_lo2, %6, a_hi2;\n\t"
    "sub.f32 a_lo3, %7, a_hi3;\n\t"
    "sub.f32 b_lo0, %8, b_hi0;\n\t"
    "sub.f32 b_lo1, %9, b_hi1;\n\t"

    "cvt.rna.tf32.f32 tf_a_lo0, a_lo0;\n\t"
    "cvt.rna.tf32.f32 tf_a_lo1, a_lo1;\n\t"
    "cvt.rna.tf32.f32 tf_a_lo2, a_lo2;\n\t"
    "cvt.rna.tf32.f32 tf_a_lo3, a_lo3;\n\t"
    "cvt.rna.tf32.f32 tf_b_lo0, b_lo0;\n\t"
    "cvt.rna.tf32.f32 tf_b_lo1, b_lo1;\n\t"

    // Compute C += A_hi * B_hi
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
    "{%0, %1, %2, %3},\n\t"
    "{tf_a_hi0, tf_a_hi1, tf_a_hi2, tf_a_hi3},\n\t"
    "{tf_b_hi0, tf_b_hi1},\n\t"
    "{%0, %1, %2, %3};\n\t"

    // Compute C += A_hi * B_lo
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
    "{%0, %1, %2, %3},\n\t"
    "{tf_a_hi0, tf_a_hi1, tf_a_hi2, tf_a_hi3},\n\t"
    "{tf_b_lo0, tf_b_lo1},\n\t"
    "{%0, %1, %2, %3};\n\t"

    // Compute C += A_lo * B_hi
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
    "{%0, %1, %2, %3},\n\t"
    "{tf_a_lo0, tf_a_lo1, tf_a_lo2, tf_a_lo3},\n\t"
    "{tf_b_hi0, tf_b_hi1},\n\t"
    "{%0, %1, %2, %3};\n\t"

    "}\n\t"

    : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
    : "f"(A[0]), "f"(A[1]), "f"(A[2]), "f"(A[3]), "f"(B[0]), "f"(B[1]));
}


/// helper to cast SMEM pointer to unsigned
__forceinline__ __device__ uint32_t
cast_smem_ptr_to_uint(void const *const ptr) {
// We prefer to use the new CVTA intrinsics if they are available, otherwise we
// will fall back to the previous internal intrinsics if they are available.
#if __CUDACC_VER_MAJOR__ >= 11
  //
  // This NVVM intrinsic converts an address in shared memory to a plain
  // unsigned integer. This is necessary to pass to shared memory instructions
  // in inline PTX.
  //
  // In CUDA 11 and beyond, this replaces __nvvm_get_smem_pointer()  [only
  // available in 10.2].
  //
  //__device__ size_t __cvta_generic_to_shared(void* ptr);

  /// CUTE helper to get SMEM pointer
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

#elif __CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2

  return __nvvm_get_smem_pointer(ptr);

#elif defined(__CUDA_ARCH__)

  uint32_t smem_ptr;

  asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, "
      "smem_ptr; }\n"
      : "=r"(smem_ptr)
      : "l"(ptr));

  return smem_ptr;

#else

  (void)ptr;
  static_assert(false, "cast_smem_ptr_to_uint not supported but used.");
  printf("ERROR: cast_smem_ptr_to_uint not supported but used.\n");
  return 0;

#endif
}

/// CUTLASS helper to get SMEM pointer
__forceinline__ __device__ unsigned get_smem_pointer(void *ptr) {
  return cast_smem_ptr_to_uint(ptr);
}

/// CUTLASS helper to get SMEM pointer
__forceinline__ __device__ unsigned get_smem_pointer(void const *ptr) {
  return get_smem_pointer(const_cast<void *>(ptr));
}




// ============================================================================
// Conversion functions between fp32 and tf32
// ============================================================================
__device__ __forceinline__ float convert_tf32_to_fp32(uint32_t &tf32_reg) {
  // clean the lower 13 bits
  uint32_t bits = (tf32_reg & ~0x1fffu);
  return reinterpret_cast<float const &>(bits);
}

__device__ __forceinline__ uint32_t
convert_fp32_to_tf32_round_towards_zero(float fp32_reg) {
  uint32_t bits = reinterpret_cast<uint32_t const &>(fp32_reg);
  // clean the lower 13 bits
  return (bits & 0xffffe000);
}

__device__ __forceinline__ uint32_t
convert_fp32_to_tf32_round_to_nearest(float fp32_reg) {
  uint32_t bits = reinterpret_cast<uint32_t const &>(fp32_reg);

  asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(bits) : "r"(bits));

  return bits;
}

__device__ __forceinline__ uint32_t
convert_fp32_to_tf32_round_half_ulp_truncate(float fp32_reg) {
  uint32_t tf32_bits = reinterpret_cast<uint32_t const &>(fp32_reg);

  if (::isfinite(fp32_reg)) {
    tf32_bits += 0x1000u;
  }
  // ignore the lower 13 bits
  return tf32_bits;
}

/// This rounding operation is similar to half_ulp_truncate except it rounds
/// denorms toward zero. It avoids predicated code, though it requires a
/// temporary register. ref: cutlass/numerical_conversion.h
__device__ __forceinline__ uint32_t
convert_fp32_to_tf32_round_half_ulp_truncate_dntz(float fp32_reg) {
  uint32_t y = reinterpret_cast<uint32_t const &>(fp32_reg);
  y = y & 0xff800000;
  float d = reinterpret_cast<float const &>(y);
  float z = d / float(1 << 11) + fp32_reg;

  return reinterpret_cast<uint32_t const &>(z);
}

// big : round towards zero
// small : round_half_ulp_truncate
__device__ __forceinline__ void convert_fp32_to_tf32_big_small(
    uint32_t &A_big, uint32_t &A_small, const float A) {
  // first convert big
//   A_big = convert_fp32_to_tf32_round_to_nearest(A);
  A_big = convert_fp32_to_tf32_round_towards_zero(A);
  // these functions don't clear the lower 13 bits, so we need to do it manually
  // A_big = convert_fp32_to_tf32_round_half_ulp_truncate(A);
  // A_big = convert_fp32_to_tf32_round_half_ulp_truncate_dntz(A);

  float A_res = A - convert_tf32_to_fp32(A_big);

  A_small = convert_fp32_to_tf32_round_half_ulp_truncate(A_res);
  // A_small = convert_fp32_to_tf32_round_half_ulp_truncate_dntz(A_res);
//   A_small = convert_fp32_to_tf32_round_to_nearest(A_res);
  // A_small = convert_fp32_to_tf32_round_towards_zero(A_res);
}

// 模板声明
template <int N>
__forceinline__ __device__ void ldsm(float (&D)[N], void const *ptr);

template <>
__forceinline__ __device__ void ldsm<1>(float (&D)[1], void const *ptr) {
  // #if __CUDA_ARCH__ >= 750

  unsigned addr = get_smem_pointer(ptr);

  int x;
  asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16{%0}, [%1];"

               : "=r"(x)
               : "r"(addr));

  reinterpret_cast<int &>(D[0]) = x;  // 将寄存器内容转换为 int

  // #else

  //     static_assert(false, "ldsm<1> not supported on this architecture.");

  // #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <>
__forceinline__ __device__ void ldsm<2>(float (&D)[2], void const *ptr) {
  // #if __CUDA_ARCH__ >= 750
  unsigned addr = get_smem_pointer(ptr);

  int x, y;
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];"
               : "=r"(x), "=r"(y)
               : "r"(addr));
  // 将寄存器内容直接赋值给数组
  D[0] = *reinterpret_cast<float *>(&x);  // 直接将int转回float
  D[1] = *reinterpret_cast<float *>(&y);  // 直接将int转回float

  // #else

  //     static_assert(false, "ldsm<2> not supported on this architecture.");

  // #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <>
__forceinline__ __device__ void ldsm<4>(float (&D)[4], void const *ptr) {
  // #if __CUDA_ARCH__ >= 750

  unsigned addr = get_smem_pointer(ptr);

  int x, y, z, w;
  asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
      : "r"(addr));
  // 将寄存器内容直接赋值给数组
  D[0] = *reinterpret_cast<float *>(&x);  // 将int转回float
  D[1] = *reinterpret_cast<float *>(&y);  // 将int转回float
  D[2] = *reinterpret_cast<float *>(&z);  // 将int转回float
  D[3] = *reinterpret_cast<float *>(&w);  // 将int转回float

  // #else

  //     static_assert(false, "ldsm<2> not supported on this architecture.");

  // #endif
}


// ============================================================================
// Shared memory swizzle
// ============================================================================
template <typename T, int kRows, int kColumns, int kOffset>
struct SharedMemorySwizzle {
  // Shared memory start address (assumed to be a pointer)
  T* shared_memory;
  constexpr static int kBanks = 32;  // Number of banks
  static_assert(
      kColumns <= kBanks,
      "Number of columns must be less than or equal to the number of banks");
  static_assert(kBanks % kColumns == 0,
                "Number of banks must be divisible by the number of columns");
  static_assert(
      kOffset + kColumns <= kBanks,
      "Offset + number of columns must be less than the number of banks");
  static_assert(kOffset % 4 == 0, "Offset must be divisible by 4");

//   static constexpr int kRows = kRows;
//   static constexpr int kColumns = kColumns;

  // 构造函数，初始化共享内存地址、行数和bank数量
  __device__ SharedMemorySwizzle(T* shared_memory)
      : shared_memory(shared_memory) {}

  __device__ __forceinline__
  T* data() const {
    return shared_memory;
  }

  // __host__ __device__ T* operator()(int i, int j) const {
  //     assert(i < kRows && j < kColumns);  // Check bounds

  //     const int col_in_physical_mod_4 = (j + kOffset) % 4;
  //     const int col_in_physical_div_4 = (j + kOffset) / 4;

  //     int physical_j_div_4 = col_in_physical_div_4 ^ (i % 8);

  //     int physical_j = physical_j_div_4 * 4 + col_in_physical_mod_4;

  //     int physical_address = (i * kBanks + physical_j);
  //     // printf("(%3d, %3d) -> (%3d, %3d)  val : %9f\n", i, j, i, physical_j,
  //     shared_memory[physical_address]); return
  //     &shared_memory[physical_address];
  // }

  // __forceinline__ __host__ __device__ T* operator()(int i, int j) const {
  //   assert(i < kRows && j < kColumns);  // Check bounds

  //   const int physical_address =
  //       (i * kBanks + (j + kOffset) % 4) + (((j + kOffset) / 4) ^ (i % 8)) * 4;
  //   return &shared_memory[physical_address];
  // }


  __forceinline__ __device__ T& operator()(int i, int j) const {
    assert(i < kRows && j < kColumns);  // Check bounds

    const int physical_address =
        (i * kBanks + (j + kOffset) % 4) + (((j + kOffset) / 4) ^ (i % 8)) * 4;
    return shared_memory[physical_address];
  }

  // // column major:
  // //               reg0 reg2
  // //               reg1 reg3

  // template <int kFragmentSize>
  // __forceinline__ __device__ void ld_smatrix_to_reg_col_major(float
  // (&reg)[kFragmentSize], const int row, const int col_div4) const {
  //     static_assert("Not implemented");
  // }

  // // row major:
  // //               reg0 reg1
  // //               reg2 reg3
  // template <int kFragmentSize>
  // __forceinline__ __device__ void ld_smatrix_to_reg_row_major(float
  // (&reg)[kFragmentSize], const int row, const int col_div4) const {
  //     static_assert("Not implemented");
  // }

  __forceinline__ __device__ void ld_smatrix_to_reg_col_major_4(
      float (&reg)[4], const int row_start, const int col_div4_start,
      const int lane_id) const {
    assert(row_start + 16 <= kRows &&
           col_div4_start + 1 <= (kColumns + kOffset) / 4);  // Check bounds

    const int thread_row = row_start + lane_id % 16;
    const int thread_col_div4 = col_div4_start + lane_id / 16;

    const int physical_address =
        thread_row * kBanks +
        ((thread_col_div4 + (kOffset / 4)) ^ (thread_row % 8)) * 4;

    ldsm<4>(reg, &shared_memory[physical_address]);
  }

  // template <>

  __forceinline__ __device__ void ld_smatrix_to_reg_row_major_4(
      float (&reg)[4], const int row_start, const int col_div4_start,
      const int lane_id) const {
    assert(row_start + 16 <= kRows &&
           col_div4_start + 1 <= (kColumns + kOffset) / 4);  // Check bounds

    const int thread_row = row_start + lane_id % 8 + 8 * (lane_id / 16);

    const int thread_col_div4 = col_div4_start + (lane_id / 8) % 2;

    const int physical_address =
        thread_row * kBanks +
        ((thread_col_div4 + (kOffset / 4)) ^ (thread_row % 8)) * 4;

    ldsm<4>(reg, &shared_memory[physical_address]);
  }

  __forceinline__ __device__ void ld_smatrix_to_reg_row_major_2(
      float (&reg)[2], const int row_start, const int col_div4_start,
      const int lane_id) const {
    assert(row_start + 8 <= kRows &&
           col_div4_start + 1 <= (kColumns + kOffset) / 4);  // Check bounds

    const int thread_row = row_start + lane_id % 8;
    // thread 16-31 may access out of bounds, but we use ldsm<2>, which will
    // ignore the
    const int thread_col_div4 = col_div4_start + lane_id / 8;

    const int physical_address =
        thread_row * kBanks +
        ((thread_col_div4 + (kOffset / 4)) ^ (thread_row % 8)) * 4;

    ldsm<2>(reg, &shared_memory[physical_address]);
  }

  __device__ void print_logical_memory() const {
    if (threadIdx.x != 0) {
      return;
    }
    printf("Logical memory layout (%d x %d):\n", kRows, kColumns);
    for (int i = 0; i < kRows; ++i) {
      for (int j = 0; j < kColumns; ++j) {
        printf("%10.3e ", (*this)(i, j));
      }
      printf("\n");
    }
  }

  __device__ void print_physical_memory() const {
    if (threadIdx.x != 0) {
      return;
    }
    printf("Physical memory layout (kBanks columns):\n");
    printf("Bank  : ");
    for (int j = 0; j < kBanks; ++j) {
      // Print bank number
      printf("%10d ", j);
    }
    printf("\n");
    for (int i = 0; i < kRows; ++i) {
      printf("Row%3d: ", i);
      for (int j = 0; j < kBanks; ++j) {
        printf("%10.3e ", shared_memory[i * kBanks + j]);
      }
      printf("\n");
    }
  }

  // debug
//   __host__ __device__ void fill(T val) {
//     for (int i = 0; i < kRows; i++) {
//       for (int j = 0; j < kColumns; j++) {
//         (*this)(i, j) = val;
//       }
//     }
//   }

  __device__ void fill(T val) {
    #pragma unroll
    for(int ele_idx = threadIdx.x; ele_idx < kRows * kColumns; ele_idx += blockDim.x) {
      (*this)(ele_idx / kColumns, ele_idx % kColumns) = val;
    }
  }
};




template <typename SmemLayoutA, typename SmemLayoutB, int kMblock, int kNblock,
          int kKblock, int kMwarp, int kNwarp, int kKwarp, int kMmma, int kNmma,
          int kKmma>
__device__ __forceinline__ void block_mma_tf32_sim_fp32(
    float C[kMwarp / kMmma][kNwarp / kNmma][kMmma * kNmma / 32], SmemLayoutA s_A, SmemLayoutB s_B,
    int warp_id, int lane_id) {

  static_assert(kKblock == kKwarp, "kKblock != kKwarp");


  constexpr int kFragmentASize = kMmma * kKmma / 32; // 4
  constexpr int kFragmentBSize = kNmma * kKmma / 32; // 2
  // constexpr int kFragmentCSize = kMmma * kNmma / 32; // 4

  float A[kMwarp / kMmma][kFragmentASize] = {};
  float B[kNwarp / kNmma][kFragmentBSize] = {};

  uint32_t A_tf32_big[kMwarp / kMmma][kFragmentASize] = {};
  uint32_t A_tf32_small[kMwarp / kMmma][kFragmentASize] = {};

  uint32_t B_tf32_big[kNwarp / kNmma][kFragmentBSize] = {};
  uint32_t B_tf32_small[kNwarp / kNmma][kFragmentBSize] = {};

  // column major:
  const int warp_tile_m_offset = (warp_id % (kMblock / kMwarp)) * kMwarp;
  // bug free, use kMxxx to calculate n is correct
  const int warp_tile_n_offset = (warp_id / (kMblock / kMwarp)) * kNwarp; 
#pragma unroll
  for (int k = 0; k < kKwarp; k += kKmma) {
#pragma unroll
    for (int i = 0; i < kMwarp / kMmma; i++) {
      s_A.ld_smatrix_to_reg_col_major_4(A[i], i * kMmma + warp_tile_m_offset,
                                        k / 4, lane_id);

// convert to tf32
#pragma unroll
      for (int j = 0; j < kFragmentASize; j++) {
        convert_fp32_to_tf32_big_small(A_tf32_big[i][j], A_tf32_small[i][j],
                                       A[i][j]);
      }
    }

// a little bit ugly, but it works. Use ldsm<4>to load 2 B fragments
#pragma unroll
    for (int j = 0; j < kNwarp / 8; j += 2) {
      float B_frag[4] = {B[j][0], B[j][1], B[j + 1][0], B[j + 1][1]};
      s_B.ld_smatrix_to_reg_row_major_4(B_frag, j * 8 + warp_tile_n_offset,
                                        k / 4, lane_id);

      B[j][0] = B_frag[0];
      B[j][1] = B_frag[1];
      B[j + 1][0] = B_frag[2];
      B[j + 1][1] = B_frag[3];
    }

    for (int j = 0; j < kNwarp / 8; j++) {
// convert to tf32
#pragma unroll
      for (int i = 0; i < kFragmentBSize; i++) {
        convert_fp32_to_tf32_big_small(B_tf32_big[j][i], B_tf32_small[j][i],
                                       B[j][i]);
      }
    }

#pragma unroll
    for (int i = 0; i < kMwarp / 16; i++) {
#pragma unroll
      for (int j = 0; j < kNwarp / 8; j++) {
        mma_sync_aligned_m16n8k8_row_col_tf32x3(C[i][j], A_tf32_big[i],
                                                A_tf32_small[i], B_tf32_big[j],
                                                B_tf32_small[j]);
      }
    }
  }
}



// block level. Convert fp32 to tf32, using mma.
template <typename SmemLayoutA, typename SmemLayoutB, int kMblock, int kNblock,
          int kKblock, int kMwarp, int kNwarp, int kKwarp, int kMmma, int kNmma,
          int kKmma>
__device__ __forceinline__ void block_mma_tf32(
    float C[kMwarp / kMmma][kNwarp / kNmma][4], SmemLayoutA s_A, SmemLayoutB s_B,
    int warp_id, int lane_id) {

  static_assert(kKblock == kKwarp, "kKblock != kKwarp");


  constexpr int kFragmentASize = kMmma * kKmma / 32; // 4
  constexpr int kFragmentBSize = kNmma * kKmma / 32; // 2
  constexpr int kFragmentCSize = kMmma * kNmma / 32; // 4

  float A[kMwarp / kMmma][kFragmentASize] = {};
  float B[kNwarp / kNmma][kFragmentBSize] = {};

  uint32_t A_tf32[kMwarp / kMmma][kFragmentASize] = {};
  uint32_t B_tf32[kNwarp / kNmma][kFragmentBSize] = {};

  // column major:
  const int warp_tile_m_offset = (warp_id % (kMblock / kMwarp)) * kMwarp;
  // bug free, use kMxxx to calculate n is correct
  const int warp_tile_n_offset = (warp_id / (kMblock / kMwarp)) * kNwarp; 
#pragma unroll
  for (int k = 0; k < kKwarp; k += kKmma) {
#pragma unroll
    for (int i = 0; i < kMwarp / kMmma; i++) {
      s_A.ld_smatrix_to_reg_col_major_4(A[i], i * kMmma + warp_tile_m_offset,
                                        k / 4, lane_id);

// convert to tf32

#pragma unroll
      for (int j = 0; j < kFragmentASize; j++) {
        A_tf32[i][j] = convert_fp32_to_tf32_round_towards_zero(A[i][j]); 
      }
    }

// a little bit ugly, but it works. Use ldsm<4>to load 2 B fragments
#pragma unroll
    for (int j = 0; j < kNwarp / 8; j += 2) {
      float B_frag[4] = {B[j][0], B[j][1], B[j + 1][0], B[j + 1][1]};
      s_B.ld_smatrix_to_reg_row_major_4(B_frag, j * 8 + warp_tile_n_offset,
                                        k / 4, lane_id);

      B[j][0] = B_frag[0];
      B[j][1] = B_frag[1];
      B[j + 1][0] = B_frag[2];
      B[j + 1][1] = B_frag[3];
    }

    for (int j = 0; j < kNwarp / 8; j++) {
// convert to tf32
#pragma unroll
      for (int i = 0; i < kFragmentBSize; i++) {
        B_tf32[j][i] = convert_fp32_to_tf32_round_towards_zero(B[j][i]);
      }
    }

#pragma unroll
    for (int i = 0; i < kMwarp / 16; i++) {
#pragma unroll
      for (int j = 0; j < kNwarp / 8; j++) {
        mma_sync_aligned_m16n8k8_row_col_tf32x1(C[i][j], A_tf32[i], nullptr, B_tf32[j], nullptr);
      }
    }
  }
}

#endif  // MMA_UTILS_CUH