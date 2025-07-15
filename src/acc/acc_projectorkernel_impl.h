#ifndef ACC_PROJECTORKERNELIMPL_H_
#define ACC_PROJECTORKERNELIMPL_H_


#ifndef PROJECTOR_NO_TEXTURES
#define PROJECTOR_PTR_TYPE cudaTextureObject_t
#else
#define PROJECTOR_PTR_TYPE XFLOAT *
#endif

#ifdef _CUDA_HALF
#include "src/acc/cuda/cuda_settings.h"
#endif
#include <assert.h>

class AccProjectorKernel
{

public:
	int mdlX, mdlXY, mdlZ,
		imgX, imgY, imgZ,
		mdlInitY, mdlInitZ,
		maxR, maxR2, maxR2_padded;
	XFLOAT 	padding_factor;
#ifdef _CUDA_HALF
	CUDA_HALF padding_factor_h;
#endif

	PROJECTOR_PTR_TYPE mdlReal;
	PROJECTOR_PTR_TYPE mdlImag;
#ifdef _CUDA_ENABLED
	PROJECTOR_PTR_TYPE mdlComplex;
#else
	std::complex<XFLOAT> *mdlComplex;
#endif

	AccProjectorKernel(
			int mdlX, int mdlY, int mdlZ,
			int imgX, int imgY, int imgZ,
			int mdlInitY, int mdlInitZ,
			XFLOAT padding_factor,
			int maxR,
#ifdef _CUDA_ENABLED
			PROJECTOR_PTR_TYPE mdlComplex
#else
			std::complex<XFLOAT> *mdlComplex
#endif
			):
			mdlX(mdlX), mdlXY(mdlX*mdlY), mdlZ(mdlZ),
			imgX(imgX), imgY(imgY), imgZ(imgZ),
			mdlInitY(mdlInitY), mdlInitZ(mdlInitZ),
			padding_factor(padding_factor),
#ifdef _CUDA_HALF
	        padding_factor_h((CUDA_HALF)padding_factor),
#endif
			maxR(maxR), maxR2(maxR*maxR), maxR2_padded(maxR*maxR*padding_factor*padding_factor),
			mdlComplex(mdlComplex)
		{};

	AccProjectorKernel(
			int mdlX, int mdlY, int mdlZ,
			int imgX, int imgY, int imgZ,
			int mdlInitY, int mdlInitZ,
			XFLOAT padding_factor,
			int maxR,
			PROJECTOR_PTR_TYPE mdlReal, PROJECTOR_PTR_TYPE mdlImag
			):
				mdlX(mdlX), mdlXY(mdlX*mdlY), mdlZ(mdlZ),
				imgX(imgX), imgY(imgY), imgZ(imgZ),
				mdlInitY(mdlInitY), mdlInitZ(mdlInitZ),
				padding_factor(padding_factor),
#ifdef _CUDA_HALF
	        	padding_factor_h((CUDA_HALF)padding_factor),
#endif
				maxR(maxR), maxR2(maxR*maxR), maxR2_padded(maxR*maxR*padding_factor*padding_factor),
				mdlReal(mdlReal), mdlImag(mdlImag)
			{
#ifndef _CUDA_ENABLED
std::complex<XFLOAT> *pData = mdlComplex;
				for(size_t i=0; i<(size_t)mdlX * (size_t)mdlY * (size_t)mdlZ; i++) {
					std::complex<XFLOAT> arrayval(*mdlReal ++, *mdlImag ++);
					pData[i] = arrayval;
				}
#endif
			};

#ifdef _CUDA_ENABLED
	__device__ __forceinline__
#else
	#ifndef __INTEL_COMPILER
	__attribute__((always_inline))
	#endif
	inline
#endif
	void project3Dmodel(
			int x,
			int y,
			int z,
			XFLOAT e0,
			XFLOAT e1,
			XFLOAT e2,
			XFLOAT e3,
			XFLOAT e4,
			XFLOAT e5,
			XFLOAT e6,
			XFLOAT e7,
			XFLOAT e8,
			XFLOAT &real,
			XFLOAT &imag)
	{
		XFLOAT xp = (e0 * x + e1 * y + e2 * z) * padding_factor;
		XFLOAT yp = (e3 * x + e4 * y + e5 * z) * padding_factor;
		XFLOAT zp = (e6 * x + e7 * y + e8 * z) * padding_factor;

		int r2 = xp*xp + yp*yp + zp*zp;

		if (r2 <= maxR2_padded)
		{

#ifdef PROJECTOR_NO_TEXTURES
			bool invers(xp < 0);
			if (invers)
			{
				xp = -xp;
				yp = -yp;
				zp = -zp;
			}

#ifdef _CUDA_ENABLED
real =   no_tex3D(mdlReal, xp, yp, zp, mdlX, mdlXY, mdlInitY, mdlInitZ);
			imag = - no_tex3D(mdlImag, xp, yp, zp, mdlX, mdlXY, mdlInitY, mdlInitZ);
#else
			CpuKernels::complex3D(mdlComplex, real, imag, xp, yp, zp, mdlX, mdlXY, mdlInitY, mdlInitZ);
#endif

			if(invers)
			    imag = -imag;


#else
			if (xp < 0)
			{
				// Get complex conjugated hermitian symmetry pair
				xp = -xp;
				yp = -yp;
				zp = -zp;

				yp -= mdlInitY;
				zp -= mdlInitZ;

				real =    tex3D<XFLOAT>(mdlReal, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
				imag =  - tex3D<XFLOAT>(mdlImag, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
			}
			else
			{
				yp -= mdlInitY;
				zp -= mdlInitZ;

				real =   tex3D<XFLOAT>(mdlReal, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
				imag =   tex3D<XFLOAT>(mdlImag, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
			}
#endif
		}
		else
		{
			real = (XFLOAT)0;
			imag = (XFLOAT)0;
		}
	}

#ifdef _CUDA_ENABLED
	__device__ __forceinline__
#else
	#ifndef __INTEL_COMPILER
	__attribute__((always_inline))
	#endif
	inline
#endif
	void project3Dmodel(
			int x,
			int y,
			XFLOAT e0,
			XFLOAT e1,
			XFLOAT e3,
			XFLOAT e4,
			XFLOAT e6,
			XFLOAT e7,
			XFLOAT &real,
			XFLOAT &imag)
	{
		XFLOAT xp = (e0 * x + e1 * y ) * padding_factor;
		XFLOAT yp = (e3 * x + e4 * y ) * padding_factor;
		XFLOAT zp = (e6 * x + e7 * y ) * padding_factor;
		
		// XFLOAT left = xp*xp + yp*yp + zp*zp;
		// XFLOAT right = padding_factor * padding_factor * ((XFLOAT)x*(XFLOAT)x + (XFLOAT)y*(XFLOAT)y);
		// if (left != right) {
		// 	printf("%10e  %10e", left, right);
		// 	assert(false);
		// }
		int r2 = xp*xp + yp*yp + zp*zp;

		if (r2 <= maxR2_padded)
		{

#ifdef PROJECTOR_NO_TEXTURES
			bool invers(xp < 0);
			if (invers)
			{
				xp = -xp;
				yp = -yp;
				zp = -zp;
			}

	#ifdef _CUDA_ENABLED
real = no_tex3D(mdlReal, xp, yp, zp, mdlX, mdlXY, mdlInitY, mdlInitZ);
			imag = no_tex3D(mdlImag, xp, yp, zp, mdlX, mdlXY, mdlInitY, mdlInitZ);
	#else
			CpuKernels::complex3D(mdlComplex, real, imag, xp, yp, zp, mdlX, mdlXY, mdlInitY, mdlInitZ);
	#endif

			if(invers)
			    imag = -imag;
#else
			if (xp < 0)
			{
				// Get complex conjugated hermitian symmetry pair
				xp = -xp;
				yp = -yp;
				zp = -zp;

				yp -= mdlInitY;
				zp -= mdlInitZ;

				real =    tex3D<XFLOAT>(mdlReal, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
				imag =  - tex3D<XFLOAT>(mdlImag, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
			}
			else
			{
				yp -= mdlInitY;
				zp -= mdlInitZ;

				real =   tex3D<XFLOAT>(mdlReal, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
				imag =   tex3D<XFLOAT>(mdlImag, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5, zp + (XFLOAT)0.5);
			}
#endif
		}
		else
		{
			real = (XFLOAT)0;
			imag = (XFLOAT)0;
		}
	}

#ifdef _CUDA_HALF
#ifdef _CUDA_ENABLED
	__device__ __forceinline__
#else
	#ifndef __INTEL_COMPILER
	__attribute__((always_inline))
	#endif
	inline
#endif
	void project3Dmodel(
			int x,
			int y,
			CUDA_HALF e0,
			CUDA_HALF e1,
			CUDA_HALF e3,
			CUDA_HALF e4,
			CUDA_HALF e6,
			CUDA_HALF e7,
			CUDA_HALF &real,
			CUDA_HALF &imag)
	{
		CUDA_HALF xp = __hmul((__hmul(e0, __int2half_rn(x)) + __hmul(e1, __int2half_rn(y)) ) ,  padding_factor_h);
		CUDA_HALF yp = __hmul((__hmul(e3, __int2half_rn(x)) + __hmul(e4, __int2half_rn(y)) ) ,  padding_factor_h);
		CUDA_HALF zp = __hmul((__hmul(e6, __int2half_rn(x)) + __hmul(e7, __int2half_rn(y)) ) ,  padding_factor_h);

		int r2 = __half2int_rz(__hmul(xp, xp) + __hmul(yp, yp) + __hmul(zp, zp));

		if (r2 <= maxR2_padded)
		{

#ifdef PROJECTOR_NO_TEXTURES
			bool invers(xp < 0);
			if (invers)
			{
				xp = -xp;
				yp = -yp;
				zp = -zp;
			}

	#ifdef _CUDA_ENABLED
real = (CUDA_HALF)no_tex3D(mdlReal, xp, yp, zp, mdlX, mdlXY, mdlInitY, mdlInitZ);
			imag = (CUDA_HALF)no_tex3D(mdlImag, xp, yp, zp, mdlX, mdlXY, mdlInitY, mdlInitZ);
	#else
			CpuKernels::complex3D(mdlComplex, real, imag, xp, yp, zp, mdlX, mdlXY, mdlInitY, mdlInitZ);
	#endif

			if(invers)
			    imag = -imag;
#else
			if (xp < (CUDA_HALF)0)
			{
				// Get complex conjugated hermitian symmetry pair
				xp = -xp;
				yp = -yp;
				zp = -zp;

				yp -= mdlInitY;
				zp -= mdlInitZ;

				// real =    (CUDA_HALF)tex3D<XFLOAT>(mdlReal, xp + (CUDA_HALF)0.5, yp + (CUDA_HALF)0.5, zp + (CUDA_HALF)0.5);
				real =    (CUDA_HALF)tex3D<XFLOAT>(mdlReal, __half2float(xp) + 0.5, __half2float(yp) + 0.5, __half2float(zp) + 0.5);
				// imag = (CUDA_HALF) - tex3D<XFLOAT>(mdlImag, xp + (CUDA_HALF)0.5, yp + (CUDA_HALF)0.5, zp + (CUDA_HALF)0.5);
				imag = (CUDA_HALF) - tex3D<XFLOAT>(mdlImag, __half2float(xp) + 0.5, __half2float(yp) + 0.5, __half2float(zp) + 0.5);
			}
			else
			{
				yp -= mdlInitY;
				zp -= mdlInitZ;

				real =   (CUDA_HALF)tex3D<XFLOAT>(mdlReal, xp + (CUDA_HALF)0.5, yp + (CUDA_HALF)0.5, zp + (CUDA_HALF)0.5);
				imag =   (CUDA_HALF)tex3D<XFLOAT>(mdlImag, xp + (CUDA_HALF)0.5, yp + (CUDA_HALF)0.5, zp + (CUDA_HALF)0.5);
			}
#endif
		}
		else
		{
			real = (CUDA_HALF)0;
			imag = (CUDA_HALF)0;
		}
	}


#ifdef _CUDA_ENABLED
	__device__ __forceinline__
#else
	#ifndef __INTEL_COMPILER
	__attribute__((always_inline))
	#endif
	inline
#endif
	void project3Dmodel(
			int x,
			int y,
			int z,
			CUDA_HALF e0,
			CUDA_HALF e1,
			CUDA_HALF e2,
			CUDA_HALF e3,
			CUDA_HALF e4,
			CUDA_HALF e5,
			CUDA_HALF e6,
			CUDA_HALF e7,
			CUDA_HALF e8,
			CUDA_HALF &real,
			CUDA_HALF &imag)
	{
		CUDA_HALF xp = __hmul((__hmul(e0, __int2half_rn(x)) + __hmul(e1, __int2half_rn(y)) + __hmul(e2, __int2half_rn(z)) ) ,  padding_factor_h);
		CUDA_HALF yp = __hmul((__hmul(e3, __int2half_rn(x)) + __hmul(e4, __int2half_rn(y)) + __hmul(e5, __int2half_rn(z)) ) ,  padding_factor_h);
		CUDA_HALF zp = __hmul((__hmul(e6, __int2half_rn(x)) + __hmul(e7, __int2half_rn(y)) + __hmul(e8, __int2half_rn(z)) ) ,  padding_factor_h);

		int r2 = __half2int_rz(__hmul(xp, xp) + __hmul(yp, yp) + __hmul(zp, zp));

		if (r2 <= maxR2_padded)
		{

#ifdef PROJECTOR_NO_TEXTURES
			bool invers(xp < 0);
			if (invers)
			{
				xp = -xp;
				yp = -yp;
				zp = -zp;
			}

#ifdef _CUDA_ENABLED
real =   no_tex3D(mdlReal, xp, yp, zp, mdlX, mdlXY, mdlInitY, mdlInitZ);
			imag = - no_tex3D(mdlImag, xp, yp, zp, mdlX, mdlXY, mdlInitY, mdlInitZ);
#else
			CpuKernels::complex3D(mdlComplex, real, imag, xp, yp, zp, mdlX, mdlXY, mdlInitY, mdlInitZ);
#endif

			if(invers)
			    imag = -imag;


#else
			if (xp < (CUDA_HALF)0)
			{
				// Get complex conjugated hermitian symmetry pair
				xp = -xp;
				yp = -yp;
				zp = -zp;

				yp -= mdlInitY;
				zp -= mdlInitZ;

				real = (CUDA_HALF)   tex3D<XFLOAT>(mdlReal, xp + (CUDA_HALF)0.5, yp + (CUDA_HALF)0.5, zp + (CUDA_HALF)0.5);
				imag = (CUDA_HALF) - tex3D<XFLOAT>(mdlImag, xp + (CUDA_HALF)0.5, yp + (CUDA_HALF)0.5, zp + (CUDA_HALF)0.5);
			}
			else
			{
				yp -= mdlInitY;
				zp -= mdlInitZ;

				real =   (CUDA_HALF)tex3D<XFLOAT>(mdlReal, xp + (CUDA_HALF)0.5, yp + (CUDA_HALF)0.5, zp + (CUDA_HALF)0.5);
				imag =   (CUDA_HALF)tex3D<XFLOAT>(mdlImag, xp + (CUDA_HALF)0.5, yp + (CUDA_HALF)0.5, zp + (CUDA_HALF)0.5);
			}
#endif
		}
		else
		{
			real = (CUDA_HALF)0;
			imag = (CUDA_HALF)0;
		}
	}


#ifdef _CUDA_ENABLED
__device__ __forceinline__
#else
	#ifndef __INTEL_COMPILER
	__attribute__((always_inline))
	#endif
	inline
#endif
	void project2Dmodel(
				int x,
				int y,
				CUDA_HALF e0,
				CUDA_HALF e1,
				CUDA_HALF e3,
				CUDA_HALF e4,
				CUDA_HALF &real,
				CUDA_HALF &imag)
	{
		CUDA_HALF xp = __hmul((__hmul(e0, __int2half_rn(x)) + __hmul(e1, __int2half_rn(y)) ) ,  padding_factor_h);
		CUDA_HALF yp = __hmul((__hmul(e3, __int2half_rn(x)) + __hmul(e4, __int2half_rn(y)) ) ,  padding_factor_h);

		int r2 = __half2int_rz(__hmul(xp, xp) + __hmul(yp, yp));

		if (r2 <= maxR2_padded)
		{
#ifdef PROJECTOR_NO_TEXTURES
			bool invers(xp < 0);
			if (invers)
			{
				xp = -xp;
				yp = -yp;
			}

	#ifdef _CUDA_ENABLED
real = no_tex2D(mdlReal, xp, yp, mdlX, mdlInitY);
			imag = no_tex2D(mdlImag, xp, yp, mdlX, mdlInitY);
	#else
			CpuKernels::complex2D(mdlComplex, real, imag, xp, yp, mdlX, mdlInitY);
	#endif

			if(invers)
			    imag = -imag;

#else
			if (xp < (CUDA_HALF)0)
			{
				// Get complex conjugated hermitian symmetry pair
				xp = -xp;
				yp = -yp;
				yp -= mdlInitY;

				real = (CUDA_HALF)  tex2D<XFLOAT>(mdlReal, xp + (CUDA_HALF)0.5, yp + (CUDA_HALF)0.5);
				imag = (CUDA_HALF)- tex2D<XFLOAT>(mdlImag, xp + (CUDA_HALF)0.5, yp + (CUDA_HALF)0.5);
			}
			else
			{
				yp -= mdlInitY;
				real = (CUDA_HALF)  tex2D<XFLOAT>(mdlReal, xp + (CUDA_HALF)0.5, yp + (CUDA_HALF)0.5);
				imag = (CUDA_HALF)  tex2D<XFLOAT>(mdlImag, xp + (CUDA_HALF)0.5, yp + (CUDA_HALF)0.5);
			}
#endif
		}
		else
		{
			real=(CUDA_HALF)0;
			imag=(CUDA_HALF)0;
		}
	}

#endif


#ifdef _CUDA_ENABLED
__device__ __forceinline__
#else
	#ifndef __INTEL_COMPILER
	__attribute__((always_inline))
	#endif
	inline
#endif
	void project2Dmodel(
				int x,
				int y,
				XFLOAT e0,
				XFLOAT e1,
				XFLOAT e3,
				XFLOAT e4,
				XFLOAT &real,
				XFLOAT &imag)
	{
		XFLOAT xp = (e0 * x + e1 * y ) * padding_factor;
		XFLOAT yp = (e3 * x + e4 * y ) * padding_factor;

		int r2 = xp*xp + yp*yp;

		if (r2 <= maxR2_padded)
		{
#ifdef PROJECTOR_NO_TEXTURES
			bool invers(xp < 0);
			if (invers)
			{
				xp = -xp;
				yp = -yp;
			}

	#ifdef _CUDA_ENABLED
real = no_tex2D(mdlReal, xp, yp, mdlX, mdlInitY);
			imag = no_tex2D(mdlImag, xp, yp, mdlX, mdlInitY);
	#else
			CpuKernels::complex2D(mdlComplex, real, imag, xp, yp, mdlX, mdlInitY);
	#endif

			if(invers)
			    imag = -imag;

#else
			if (xp < 0)
			{
				// Get complex conjugated hermitian symmetry pair
				xp = -xp;
				yp = -yp;
				yp -= mdlInitY;

				real =   tex2D<XFLOAT>(mdlReal, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5);
				imag = - tex2D<XFLOAT>(mdlImag, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5);
			}
			else
			{
				yp -= mdlInitY;
				real =   tex2D<XFLOAT>(mdlReal, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5);
				imag =   tex2D<XFLOAT>(mdlImag, xp + (XFLOAT)0.5, yp + (XFLOAT)0.5);
			}
#endif
		}
		else
		{
			real=(XFLOAT)0;
			imag=(XFLOAT)0;
		}
	}

	static AccProjectorKernel makeKernel(AccProjector &p, int imgX, int imgY, int imgZ, int imgMaxR)
	{
		int maxR = p.mdlMaxR >= imgMaxR ? imgMaxR : p.mdlMaxR;

		AccProjectorKernel k(
					p.mdlX, p.mdlY, p.mdlZ,
					imgX, imgY, imgZ,
					p.mdlInitY, p.mdlInitZ,
					p.padding_factor,
					maxR,
#ifndef PROJECTOR_NO_TEXTURES
					*p.mdlReal,
					*p.mdlImag
#else
#ifdef _CUDA_ENABLED
p.mdlReal,
					p.mdlImag
#else
					p.mdlComplex
#endif
#endif
				);
		return k;
	}
};  // class AccProjectorKernel


#endif
