#ifndef COARSE_KERNEL_BLOCK_PARAMS_CUH
#define COARSE_KERNEL_BLOCK_PARAMS_CUH

template<typename T>
struct TypeHolder { using type = T; };

template <int BlockSize, int TransBlockSize, int OrientBlockSize, int ImgBlockSize, 
          int WarpTransTileSize, int WarpOrientTileSize, int WarpImgTileSize, 
          int MmaTransTileSize, int MmaOrientTileSize, int MmaImgTileSize>
struct CoarseKernelBlockCoarseTParams {
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


template <int BlockSize, int TransBlockSize, int OrientBlockSize, int ImgBlockSize, 
          int WarpTransTileSize, int WarpOrientTileSize, int WarpImgTileSize, 
          int MmaTransTileSize, int MmaOrientTileSize, int MmaImgTileSize, 
          int NrOverTrans, int NrOverOrient>
struct FineKernelBlockCoarseTParams {
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
    static constexpr int kNrOverTrans = NrOverTrans;
    static constexpr int kNrOverOrient = NrOverOrient;
};


using CoarseTParam128x64_64x32 = CoarseKernelBlockCoarseTParams<128, 128, 64, 16, 64, 32, 16, 16, 8, 8>;
using CoarseTParam96x64_48x32  = CoarseKernelBlockCoarseTParams<128, 96,  64, 16, 48, 32, 16, 16, 8, 8>;
using CoarseTParam64x128_32x64 = CoarseKernelBlockCoarseTParams<128, 64, 128, 16, 32, 64, 16, 16, 8, 8>;
using CoarseTParam64x64_32x32 = CoarseKernelBlockCoarseTParams<128, 64, 64, 16, 32, 32, 16, 16, 8, 8>;
using CoarseTParam48x128_48x32 = CoarseKernelBlockCoarseTParams<128, 48, 128, 16, 48, 32, 16, 16, 8, 8>;
using CoarseTParam32x128_32x32 = CoarseKernelBlockCoarseTParams<128, 32, 128, 16, 32, 32, 16, 16, 8, 8>;
using CoarseTParam32x128_16x64 = CoarseKernelBlockCoarseTParams<128, 32, 128, 16, 16, 64, 16, 16, 8, 8>;

using CoarseTParam32x64_16x32 = CoarseKernelBlockCoarseTParams<128, 32, 64, 16, 16, 32, 16, 16, 8, 8>;


using FineTParam64x128_32x64_4_8 = FineKernelBlockCoarseTParams<128, 64, 128, 16, 32, 64, 16, 16, 8, 8, 4, 8>;

#endif // COARSE_KERNEL_BLOCK_PARAMS_CUH