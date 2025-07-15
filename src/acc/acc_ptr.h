#ifndef ACC_PTR_H_
#define ACC_PTR_H_

#include "src/acc/settings.h"
#ifdef _CUDA_ENABLED
#include "src/acc/cuda/cuda_settings.h"
#include <cuda_runtime.h>
#include "src/acc/cuda/custom_allocator.cuh"
#include "src/acc/cuda/cuda_mem_utils.h"
#include "src/acc/cuda/shortcuts.cuh"
#include "src/acc/cuda/pinned_allocator.cuh"
#else
#include "src/acc/cpu/cpu_settings.h"
#endif

#include <signal.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cstring>

#include "src/macros.h"
#include "src/error.h"
#include "src/parallel.h"

#ifndef MEM_ALIGN
	#define MEM_ALIGN 64
#endif

// #define NO_PINNED_ALLOCATOR
// #define ALLOCATOR_LOG

#ifdef TIMING_FILES
#define	CTIC(timer,timing) (timer.cuda_cpu_tic(timing))
#define	CTOC(timer,timing) (timer.cuda_cpu_toc(timing))
#define	GTIC(timer,timing) (timer.cuda_gpu_tic(timing))
#define	GTOC(timer,timing) (timer.cuda_gpu_toc(timing))
#define	GATHERGPUTIMINGS(timer) (timer.cuda_gpu_printtictoc())
#elif defined CUDA_PROFILING
	#include <nvToolsExt.h>
	#define	CTIC(timer,timing) (nvtxRangePush(timing))
	#define	CTOC(timer,timing) (nvtxRangePop())
	#define	GTIC(timer,timing)
	#define	GTOC(timer,timing)
	#define	GATHERGPUTIMINGS(timer)
#else
	#define	CTIC(timer,timing)
	#define	CTOC(timer,timing)
	#define	GTIC(timer,timing)
	#define	GTOC(timer,timing)
	#define	GATHERGPUTIMINGS(timer)
#endif

#define ACC_PTR_DEBUG_FATAL( err ) (HandleAccPtrDebugFatal( err, __FILE__, __LINE__ ))
static void HandleAccPtrDebugFatal( const char *err, const char *file, int line )
{
    	fprintf(stderr, "DEBUG ERROR: %s in %s:%d\n", err, file, line );
		fflush(stdout);
#ifdef DEBUG_CUDA
		raise(SIGSEGV);
#else
		CRITICAL(ERRGPUKERN);
#endif
}

#define ACC_PTR_DEBUG_INFO( err ) (HandleAccPtrDebugInformational( err, __FILE__, __LINE__ ))
static void HandleAccPtrDebugInformational( const char *err, const char *file, int line )
{
    	fprintf(stderr, "POSSIBLE ISSUE: %s in %s:%d\n", err, file, line );
		fflush(stdout);
}

enum AccType {accUNSET, accCUDA, accCPU};


#ifdef _CUDA_ENABLED
typedef cudaStream_t StreamType;
typedef CudaCustomAllocator AllocatorType;
typedef CudaCustomAllocator::Alloc AllocationType;
#else
typedef float StreamType; //Dummy type
typedef double AllocatorType;  //Dummy type
typedef double AllocationType;  //Dummy type
#endif

template <typename T>
class AccPtr
{
protected:
	AllocatorType *allocator;
	AllocationType *alloc;
	StreamType stream;
	
	AccType accType;

	size_t size; //Size used when copying data from and to device
	T *hPtr, *dPtr; //Host and device pointers
	bool doFreeDevice; //True if host or device needs to be freed

public:
	bool doFreeHost; //TODO make this private

	/*======================================================
				CONSTRUCTORS WITH ALLOCATORS
	======================================================*/

	AccPtr(AllocatorType *allocator):
		size(0), hPtr(NULL), dPtr(NULL), doFreeHost(false),
		doFreeDevice(false), allocator(allocator), alloc(NULL), stream(cudaStreamPerThread),
#ifdef _CUDA_ENABLED
		accType(accCUDA)
#else
		accType(accCPU)
#endif
	{}

	AccPtr(StreamType stream, AllocatorType *allocator):
		size(0), hPtr(NULL), dPtr(NULL), doFreeHost(false),
		doFreeDevice(false), allocator(allocator), alloc(NULL), stream(stream),
#ifdef _CUDA_ENABLED
		accType(accCUDA)
#else
		accType(accCPU)
#endif
	{}

	AccPtr(size_t size, AllocatorType *allocator):
		size(size), dPtr(NULL), doFreeHost(true),
		doFreeDevice(false), allocator(allocator), alloc(NULL), stream(cudaStreamPerThread),
#ifdef _CUDA_ENABLED
		accType(accCUDA)
#else
		accType(accCPU)
#endif
	{
        #ifdef NO_PINNED_ALLOCATOR
        {
            if(posix_memalign((void **)&hPtr, MEM_ALIGN, sizeof(T) * size))
			    CRITICAL(RAMERR);
            
        }
        #else
        {
            if(pin_alloc((void **)&hPtr, sizeof(T) * size))
                CRITICAL(RAMERR);
        }
        #endif
        
        #ifdef ALLOCATOR_LOG
        {
            printf("start host alloc %lu %p noend\n",sizeof(T) * size,hPtr);
            fflush(stdout);
        }
        #endif
	}

	AccPtr(size_t size, StreamType stream, AllocatorType *allocator):
		size(size), dPtr(NULL), doFreeHost(true),
		doFreeDevice(false), allocator(allocator), alloc(NULL), stream(stream),
#ifdef _CUDA_ENABLED
		accType(accCUDA)
#else
		accType(accCPU)
#endif
	{
        #ifdef NO_PINNED_ALLOCATOR
        {
            if(posix_memalign((void **)&hPtr, MEM_ALIGN, sizeof(T) * size))
			    CRITICAL(RAMERR);
        }
        #else
        {
            if(pin_alloc((void **)&hPtr, sizeof(T) * size))
                CRITICAL(RAMERR);
        }
        #endif
        
        #ifdef ALLOCATOR_LOG
        {
            printf("start host alloc %lu %p noend\n",sizeof(T) * size,hPtr);
            fflush(stdout);
        }
        #endif
    }

	AccPtr(T * h_start, size_t size, AllocatorType *allocator):
		size(size), hPtr(h_start), dPtr(NULL), doFreeHost(false),
		doFreeDevice(false), allocator(allocator), alloc(NULL), stream(cudaStreamPerThread),
#ifdef _CUDA_ENABLED
		accType(accCUDA)
#else
		accType(accCPU)
#endif
	{}

	AccPtr(T * h_start, size_t size, StreamType stream, AllocatorType *allocator):
		size(size), hPtr(h_start), dPtr(NULL), doFreeHost(false),
		doFreeDevice(false), allocator(allocator), alloc(NULL), stream(cudaStreamPerThread),
#ifdef _CUDA_ENABLED
		accType(accCUDA)
#else
		accType(accCPU)
#endif
	{}

	AccPtr(T * h_start, T * d_start, size_t size, AllocatorType *allocator):
		size(size), hPtr(h_start), dPtr(d_start), doFreeHost(false),
		doFreeDevice(false), allocator(allocator), alloc(NULL), stream(cudaStreamPerThread),
#ifdef _CUDA_ENABLED
		accType(accCUDA)
#else
		accType(accCPU)
#endif
	{}

	AccPtr(T * h_start, T * d_start, size_t size, StreamType stream, AllocatorType *allocator):
		size(size), hPtr(h_start), dPtr(d_start), doFreeHost(false),
		doFreeDevice(false), allocator(allocator), alloc(NULL), stream(stream),
#ifdef _CUDA_ENABLED
		accType(accCUDA)
#else
		accType(accCPU)
#endif
	{}

	/*======================================================
	                      CONSTRUCTORS
	======================================================*/

	AccPtr():
		size(0), hPtr(NULL), dPtr(NULL), doFreeHost(false),
		doFreeDevice(false), allocator(NULL), alloc(NULL), stream(cudaStreamPerThread),
#ifdef _CUDA_ENABLED
		accType(accCUDA)
#else
		accType(accCPU)
#endif
	{}

	AccPtr(StreamType stream):
		size(0), hPtr(NULL), dPtr(NULL), doFreeHost(false),
		doFreeDevice(false), allocator(NULL), alloc(NULL), stream(stream),
#ifdef _CUDA_ENABLED
		accType(accCUDA)
#else
		accType(accCPU)
#endif
	{}

	AccPtr(size_t size):
		size(size), dPtr(NULL), doFreeHost(true),
		doFreeDevice(false), allocator(NULL), alloc(NULL), stream(cudaStreamPerThread),
#ifdef _CUDA_ENABLED
		accType(accCUDA)
#else
		accType(accCPU)
#endif
	{
        #ifdef NO_PINNED_ALLOCATOR
        {
            if(posix_memalign((void **)&hPtr, MEM_ALIGN, sizeof(T) * size))
			    CRITICAL(RAMERR);
        }
        #else
        {
            if(pin_alloc((void **)&hPtr, sizeof(T) * size))
                CRITICAL(RAMERR);
        }
        #endif
	
        #ifdef ALLOCATOR_LOG
        {
            printf("start host alloc %lu %p noend\n",sizeof(T) * size,hPtr);
            fflush(stdout);
        }
        #endif
    }

	AccPtr(size_t size, StreamType stream):
		size(size), dPtr(NULL), doFreeHost(true),
		doFreeDevice(false), allocator(NULL), alloc(NULL), stream(stream),
#ifdef _CUDA_ENABLED
		accType(accCUDA)
#else
		accType(accCPU)
#endif
	{
        #ifdef NO_PINNED_ALLOCATOR
        {
            if(posix_memalign((void **)&hPtr, MEM_ALIGN, sizeof(T) * size))
			    CRITICAL(RAMERR);
        }
        #else
        {
            if(pin_alloc((void **)&hPtr, sizeof(T) * size))
                CRITICAL(RAMERR);
        }
        #endif

        #ifdef ALLOCATOR_LOG
        {
            printf("start host alloc %lu %p noend\n",sizeof(T) * size,hPtr);
            fflush(stdout);
        }
        #endif
	}

	AccPtr(T * h_start, size_t size):
		size(size), hPtr(h_start), dPtr(NULL), doFreeHost(false),
		doFreeDevice(false), allocator(NULL), alloc(NULL), stream(cudaStreamPerThread),
#ifdef _CUDA_ENABLED
		accType(accCUDA)
#else
		accType(accCPU)
#endif
	{}

	AccPtr(T * h_start, size_t size, StreamType stream):
		size(size), hPtr(h_start), dPtr(NULL), doFreeHost(false),
		doFreeDevice(false), allocator(NULL), alloc(NULL), stream(cudaStreamPerThread),
#ifdef _CUDA_ENABLED
		accType(accCUDA)
#else
		accType(accCPU)
#endif
	{}

	AccPtr(T * h_start, T * d_start, size_t size):
		size(size), hPtr(h_start), dPtr(d_start), doFreeHost(false),
		doFreeDevice(false), allocator(NULL), alloc(NULL), stream(cudaStreamPerThread),
#ifdef _CUDA_ENABLED
		accType(accCUDA)
#else
		accType(accCPU)
#endif
	{}

	AccPtr(T * h_start, T * d_start, size_t size, StreamType stream):
		size(size), hPtr(h_start), dPtr(d_start), doFreeHost(false),
		doFreeDevice(false), allocator(NULL), alloc(NULL), stream(stream),
#ifdef _CUDA_ENABLED
		accType(accCUDA)
#else
		accType(accCPU)
#endif
	{}

	/*======================================================
	       CONSTRUCTORS WITH OTHER POINTERS
	======================================================*/

	AccPtr(const AccPtr &ptr):
		size(ptr.size), hPtr(ptr.hPtr), dPtr(ptr.dPtr), doFreeHost(false),
		doFreeDevice(false), allocator(ptr.allocator), alloc(NULL), stream(ptr.stream),
		accType(ptr.accType)
	{}

	AccPtr(const AccPtr<T> &ptr, size_t start_idx, size_t size):
		size(size), hPtr(&ptr.hPtr[start_idx]), dPtr(&ptr.dPtr[start_idx]), doFreeHost(false),
		doFreeDevice(false), allocator(ptr.allocator), alloc(NULL), stream(ptr.stream),
		accType(ptr.accType)
	{}


	/*======================================================
	                     METHOD BODY
	======================================================*/

	void setAccType(AccType accT)
	{
		accType = accT;
	}

	void markReadyEvent()
	{
#ifdef _CUDA_ENABLED
		if (accType == accCUDA)
		{
#ifdef DEBUG_CUDA
			if (alloc == NULL)
				ACC_PTR_DEBUG_FATAL("markReadyEvent called on null allocation.\n");
#endif
			alloc->markReadyEvent(stream);
		}
#endif
	}

	/**
	 * Allocate memory on device
	 */
	void deviceAlloc()
	{
		CTIC("","deviceAlloc");
#ifdef _CUDA_ENABLED
		if (accType == accCUDA)
		{
#ifdef DEBUG_CUDA
			if(size==0)
				ACC_PTR_DEBUG_FATAL("deviceAlloc called with size == 0");
			if (doFreeDevice)
				ACC_PTR_DEBUG_FATAL("Device double allocation.\n");
#endif

				doFreeDevice = true;
				// printf("Allocating %lu bytes %luon device\n", size * sizeof(T),size);

				alloc = allocator->alloc(size * sizeof(T));

				dPtr = (T*) alloc->getPtr();

		}
#endif
		CTOC("","deviceAlloc");
	}

	/**
	 * Allocate memory on device with given size
	 */
	void deviceAlloc(size_t newSize)
	{
		size = newSize;
		deviceAlloc();
	}

	/**
	 * Allocate memory on host
	 */
	void hostAlloc()
	{
		CTIC("","hostAlloc");
#ifdef DEBUG_CUDA
		if(size==0)
			ACC_PTR_DEBUG_FATAL("deviceAlloc called with size == 0");
		if (doFreeHost)
			ACC_PTR_DEBUG_FATAL("Host double allocation.\n");
#endif
		doFreeHost = true;
		// printf("start host alloc %lu \n",sizeof(T) * size);fflush(stdout);

		// TODO - alternatively, this could be aligned std::vector
		// int error=posix_memalign((void **)&hPtr, MEM_ALIGN, sizeof(T) * size);

        #ifdef NO_PINNED_ALLOCATOR
        {
            hPtr=(T*)aligned_alloc(MEM_ALIGN, sizeof(T) * size);
        }
        #else
        {
            if(pin_alloc((void **)&hPtr, sizeof(T) * size))
                CRITICAL(RAMERR);
        }
        #endif
        
        #ifdef ALLOCATOR_LOG
        {
            printf("start host alloc %lu %p noend\n",sizeof(T) * size,hPtr);
            fflush(stdout);
        }
        #endif

		// printf("end host alloc %lu %p\n",sizeof(T) * size, hPtr);fflush(stdout);
		if(hPtr==NULL)
		{
			CRITICAL(RAMERR);
		}
		CTOC("","hostAlloc");


	}

	/**
	 * Allocate memory on host with given size
	 */
	void hostAlloc(size_t newSize)
	{
		size = newSize;
		hostAlloc();
	}

	void allAlloc()
	{
		CTIC("","allAlloc");
		deviceAlloc();
		hostAlloc();
		CTOC("","allAlloc");

	}

	void allAlloc(size_t newSize)
	{
		CTIC("","allAlloc");

		size = newSize;
		deviceAlloc();
		hostAlloc();
		CTOC("","allAlloc");

	}

	void accAlloc()
	{
		if (accType == accCUDA)
			deviceAlloc();
		else
			hostAlloc();
	}

	void accAlloc(size_t newSize)
	{
		if (accType == accCUDA)
			deviceAlloc(newSize);
		else
			hostAlloc(newSize);
	}

	// Allocate storage of a new size for the array
	void resizeHost(size_t newSize)
	{
#ifdef DEBUG_CUDA
		if (size==0)
			ACC_PTR_DEBUG_INFO("Resizing from size zero (permitted).\n");
#endif
		// TODO - alternatively, this could be aligned std::vector
		T* newArr;
        #ifdef NO_PINNED_ALLOCATOR
        {
            if(posix_memalign((void **)&newArr, MEM_ALIGN, sizeof(T) * newSize))
			    CRITICAL(RAMERR);
        }
        #else
        {
            if(pin_alloc((void **)&newArr, sizeof(T) * newSize))
                CRITICAL(RAMERR);
        }
        #endif
		
        #ifdef ALLOCATOR_LOG
        {
            printf("start host alloc %lu %p noend\n",sizeof(T) * newSize,newArr);
            fflush(stdout);
        }
        #endif

		memset( newArr, 0x0, sizeof(T) * newSize);

#ifdef DEBUG_CUDA
		if (dPtr!=NULL)
			ACC_PTR_DEBUG_FATAL("resizeHost: Resizing host with present device allocation.\n");
		if (newSize==0)
			ACC_PTR_DEBUG_INFO("resizeHost: Array resized to size zero (permitted with fear).  Something may break downstream\n");
#endif
		freeHostIfSet();	
	    setSize(newSize);
	    setHostPtr(newArr);
	    doFreeHost=true;
	}
	
	// Resize retaining as much of the original contents as possible
	void resizeHostCopy(size_t newSize)
	{
#ifdef DEBUG_CUDA
//		if (size==0)
//			ACC_PTR_DEBUG_INFO("Resizing from size zero (permitted).\n");
#endif
		// TODO - alternatively, this could be aligned std::vector
		T* newArr;
		#ifdef NO_PINNED_ALLOCATOR
        {
            if(posix_memalign((void **)&newArr, MEM_ALIGN, sizeof(T) * newSize))
			    CRITICAL(RAMERR);
        }
        #else
        {
            if(pin_alloc((void **)&newArr, sizeof(T) * newSize))
                CRITICAL(RAMERR);
        }
        #endif
        
        #ifdef ALLOCATOR_LOG
        {
            printf("start host alloc %lu %p noend\n",sizeof(T) * newSize,newArr);
            fflush(stdout);
        }
        #endif

		// Copy in what we can from the original matrix
		if ((size > 0) && (hPtr != NULL))
		{
			if (newSize < size)
				memcpy( newArr, hPtr, newSize * sizeof(T) );
			else
				memcpy( newArr, hPtr, size * sizeof(T) );  
			
			// Initialize remaining memory if any
			if (newSize > size)
			{
				size_t theRest = sizeof(T) * (newSize - size);
				memset( newArr, 0x0, theRest);
			}
		}
		
		// There was nothing from before to copy - clear new memory
		if (hPtr == NULL)
		{
			memset( newArr, 0x0, sizeof(T) * newSize);
		}

#ifdef DEBUG_CUDA
		if (dPtr!=NULL)
			ACC_PTR_DEBUG_FATAL("resizeHostCopy: Resizing host with present device allocation.\n");
		if (newSize==0)
			ACC_PTR_DEBUG_INFO("resizeHostCopy: Array resized to size zero (permitted with fear).  Something may break downstream\n");
#endif
		freeHostIfSet();
	    setSize(newSize);
	    setHostPtr(newArr);
	    doFreeHost=true;
	}
	
	/**
	 * Initiate device memory with provided value
	 */
	void deviceInit(int value)
	{
#ifdef _CUDA_ENABLED
		if (accType == accCUDA)
		{
#ifdef DEBUG_CUDA
			if (dPtr == NULL)
				ACC_PTR_DEBUG_FATAL("Memset requested before allocation in deviceInit().\n");
#endif
			cudaMemInit<T>( dPtr, value, size, stream);
		}
#endif
	}

	/**
	 * Initiate host memory with provided value
	 */
	void hostInit(int value)
	{
#ifdef DEBUG_CUDA
		if (hPtr == NULL)
			ACC_PTR_DEBUG_FATAL("Memset requested before allocation in hostInit().\n");
#endif
		memset(hPtr, value, size * sizeof(T));
	}

	/**
	 * Initiate memory with provided value
	 */
	void accInit(int value)
	{
		if (accType == accCUDA)
			deviceInit(value);
		else
			hostInit(value);
	}

	/**
	 * Initiate all used memory with provided value
	 */
	void allInit(int value)
	{
		hostInit(value);
		if (accType == accCUDA)
			deviceInit(value);
	}

	/**
	 * Copy a number (size) of bytes to device stored in the host pointer
	 */
	void cpToDevice()
	{
#ifdef _CUDA_ENABLED
		if (accType == accCUDA)
		{
#ifdef DEBUG_CUDA
			if (dPtr == NULL)
				ACC_PTR_DEBUG_FATAL("cpToDevice() called before allocation.\n");
			if (hPtr == NULL)
				ACC_PTR_DEBUG_FATAL("NULL host pointer in cpToDevice().\n");
#endif
			CudaShortcuts::cpyHostToDevice<T>(hPtr, dPtr, size, stream);
            // streamSync();
		}
#endif
	}

	/**
	 * Copy a number (size) of bytes to device stored in the provided host pointer
	 */
	void cpToDevice(T * hostPtr)
	{
		if (accType == accCUDA)
		{
#ifdef DEBUG_CUDA
			if (hostPtr == NULL)
				ACC_PTR_DEBUG_FATAL("Null-pointer given in cpToDevice(hostPtr).\n");
#endif
			hPtr = hostPtr;
			cpToDevice();
		}
	}

	/**
	 * alloc and copy
	 */
	void putOnDevice()
	{
		deviceAlloc();
		cpToDevice();
	}

	/**
	 * alloc size and copy
	 */
	void putOnDevice(size_t newSize)
	{
		size=newSize;
		deviceAlloc();
		cpToDevice();
	}


	/**
	 * Copy a number (size) of bytes from device to the host pointer
	 */
	void cpToHost()
	{
#ifdef _CUDA_ENABLED
		if (accType == accCUDA)
		{
#ifdef DEBUG_CUDA
			if (dPtr == NULL)
				ACC_PTR_DEBUG_FATAL("cp_to_host() called before device allocation.\n");
			if (hPtr == NULL)
				ACC_PTR_DEBUG_FATAL("NULL host pointer in cp_to_host().\n");
#endif
			cudaCpyDeviceToHost<T>(dPtr, hPtr, size, stream);
        }
#endif
	}

	/**
	 * Copy a number (thisSize) of bytes from device to the host pointer
	 */
	void cpToHost(size_t thisSize)
	{
#ifdef _CUDA_ENABLED
		if (accType == accCUDA)
		{
#ifdef DEBUG_CUDA
			if (dPtr == NULL)
				ACC_PTR_DEBUG_FATAL("cp_to_host(thisSize) called before device allocation.\n");
			if (hPtr == NULL)
				ACC_PTR_DEBUG_FATAL("NULL host pointer in cp_to_host(thisSize).\n");
#endif
			cudaCpyDeviceToHost<T>(dPtr, hPtr, thisSize, stream);
		}
#endif
	}

	/**
	 * Copy a number (thisSize) of bytes from device to a specific host pointer
	 */
	void cpToHost(T* hstPtr, size_t thisSize)
	{
#ifdef _CUDA_ENABLED
		if (accType == accCUDA)
		{
#ifdef DEBUG_CUDA
			if (dPtr == NULL)
				ACC_PTR_DEBUG_FATAL("cp_to_host(hstPtr, thisSize) called before device allocation.\n");
			if (hstPtr == NULL)
				ACC_PTR_DEBUG_FATAL("NULL host pointer in cp_to_host(hstPtr, thisSize).\n");
#endif
			cudaCpyDeviceToHost<T>(dPtr, hstPtr, thisSize, stream);
		}
#endif
	}

	/**
	 * Copy a number (size) of bytes from device to the host pointer
	 */
	void cpToHostOnStream(StreamType s)
	{
#ifdef _CUDA_ENABLED
		if (accType == accCUDA)
		{
#ifdef DEBUG_CUDA
			if (dPtr == NULL)
				ACC_PTR_DEBUG_FATAL("cp_to_host_on_stream(s) called before device allocation.\n");
			if (hPtr == NULL)
				ACC_PTR_DEBUG_FATAL("NULL host pointer in cp_to_host_on_stream(s).\n");
#endif
			cudaCpyDeviceToHost<T>(dPtr, hPtr, size, s);
		}
#endif
	}

	/**
	 * Copy a number (size) of bytes from device pointer to the provided new device pointer
	 */
	void cpOnDevice(T * dstDevPtr)
	{
#ifdef _CUDA_ENABLED
		if (accType == accCUDA)
		{
#ifdef DEBUG_CUDA
			if (dstDevPtr == NULL)
				ACC_PTR_DEBUG_FATAL("NULL-pointer given in cpOnDevice(dstDevPtr).\n");
#endif
			CudaShortcuts::cpyDeviceToDevice(dPtr, dstDevPtr, size, stream);
		}
#endif
	}

	/**
	 * Copy a number (size) of bytes from host pointer to the provided new host pointer
	 */
	void cpOnHost(T * dstDevPtr)
	{
#ifdef DEBUG_CUDA
		if (dstDevPtr == NULL)
			ACC_PTR_DEBUG_FATAL("NULL-pointer given in cp_on_host(dstDevPtr).\n");
		if (hPtr == NULL)
			ACC_PTR_DEBUG_FATAL("NULL input pointer given in cp_on_host(hPtr).\n");
#endif
		memcpy ( dstDevPtr, hPtr, size * sizeof(T));
	}

	void cpOnAcc(T * dstDevPtr)
	{
		if (accType == accCUDA)
			cpOnDevice(dstDevPtr);
		else
			cpOnHost(dstDevPtr);
	}

	void cpOnAcc(AccPtr<T> &devPtr)
	{
		if (accType == accCUDA)
			cpOnDevice(devPtr.dPtr);
		else
			cpOnHost(devPtr.hPtr);
	}

	/**
	 * Host data quick access
	 */
	const T& operator[](size_t idx) const
	{
#ifdef DEBUG_CUDA
		if (hPtr == NULL)
			ACC_PTR_DEBUG_FATAL("const operator[] called with NULL host pointer.\n");
#endif
		return hPtr[idx];
	};

	/**
	 * Host data quick access
	 */
	T& operator[](size_t idx)
	{
#ifdef DEBUG_CUDA
		if (hPtr == NULL)
			ACC_PTR_DEBUG_FATAL("operator[] called with NULL host pointer.\n");
#endif
		return hPtr[idx];
	};
	
	/**
	 * Device data quick access
	 */
	T& operator()(size_t idx) 
	{ 
#ifdef DEBUG_CUDA
			if (dPtr == NULL)
				ACC_PTR_DEBUG_FATAL("operator(idx) called with NULL acc pointer.\n");
#endif
		return dPtr[idx]; 
	};


	/**
	 * Device data quick access
	 */
	const T& operator()(size_t idx) const 
	{ 
#ifdef DEBUG_CUDA
			if (dPtr == NULL)
				ACC_PTR_DEBUG_FATAL("operator(idx) called with NULL acc pointer.\n");
#endif
		return dPtr[idx]; 
	};
	
	/**
	 * Raw data pointer quick access
	 */
	T* operator()()
	{
		// TODO - this could cause considerable confusion given the above operators.  But it
		// also simplifies code that uses it.   What to do...
		if (accType == accCUDA)
		{
#ifdef DEBUG_CUDA
			if (dPtr == NULL)
				ACC_PTR_DEBUG_FATAL("operator() called with NULL device pointer.\n");
#endif
			return dPtr;
		}
		else
		{
#ifdef DEBUG_CUDA
			if (hPtr == NULL)
				ACC_PTR_DEBUG_FATAL("operator() called with NULL host pointer.\n");
#endif
			return hPtr;
		}
	};

	T* operator~() //按位取反
	{
		// TODO - this could cause considerable confusion given the above operators.  But it
		// also simplifies code that uses it.   What to do...
		if (accType == accCUDA)
		{
#ifdef DEBUG_CUDA
			if ( dPtr == 0)
				ACC_PTR_DEBUG_FATAL("DEBUG_WARNING: \"kernel cast\" on null device pointer.\n");
#endif
			return dPtr;
		}
		else
		{
#ifdef DEBUG_CUDA
		if ( hPtr == 0)
			ACC_PTR_DEBUG_FATAL("DEBUG_WARNING: \"kernel cast\" on null host pointer.\n");
#endif
		return hPtr;
		}
	}
	
	void streamSync()
	{
#ifdef _CUDA_ENABLED
		if (accType == accCUDA)
			DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));
#endif
	}

	T getAccValueAt(size_t idx)
	{
#ifdef _CUDA_ENABLED
		if (accType == accCUDA)
		{
			T* value;
            pin_alloc((void **)&value, sizeof(T));
			cudaCpyDeviceToHost<T>(&dPtr[idx], value, 1, stream);
			streamSync();
            T temp_value = (*value);
            pin_free(value);
			return temp_value;
		}
		else
#endif
			return hPtr[idx];
	}

	T getDeviceAt(size_t idx)
	{
#ifdef _CUDA_ENABLED
		if (accType == accCUDA)
		{
			T* value;
            pin_alloc((void **)&value, sizeof(T));
			cudaCpyDeviceToHost<T>(&dPtr[idx], value, 1, stream);
			streamSync();
            T temp_value = (*value);
            pin_free(value);
			return temp_value;
		}
#else
		return NULL;
#endif
	}

	void dumpDeviceToFile(std::string fileName)
	{

#ifdef _CUDA_ENABLED
		if (accType == accCUDA)
		{
			T *tmp;
            pin_alloc((void **)&tmp, sizeof(T) * size);
			cudaCpyDeviceToHost<T>(dPtr, tmp, size, stream);

			std::ofstream f;
			f.open(fileName.c_str());
			streamSync();
			for (unsigned i = 0; i < size; i ++)
				f << tmp[i] << std::endl;
			f.close();
            pin_free(tmp);
		}
		else
#endif
		{
			std::ofstream f;
			f.open(fileName.c_str());
			f << "Pointer has no device support." << std::endl;
			f.close();
		}
	}

	void dumpHostToFile(std::string fileName)
	{
		std::ofstream f;
		f.open(fileName.c_str());
		for (unsigned i = 0; i < size; i ++)
			f << hPtr[i] << std::endl;
		f.close();
	}

	void dumpAccToFile(std::string fileName)
	{
		if (accType == accCUDA)
			dumpDeviceToFile(fileName);
		else
			dumpHostToFile(fileName);
	}

	/**
	 * Delete device data
	 */
	void freeDevice()
	{
#ifdef _CUDA_ENABLED
		if (accType == accCUDA)
		{
#ifdef DEBUG_CUDA
			if (dPtr == NULL)
				ACC_PTR_DEBUG_FATAL("Free device memory was called on NULL pointer in free_device().\n");
#endif
			doFreeDevice = false;

			if (alloc->getReadyEvent() == 0)
				alloc->markReadyEvent(stream);
			alloc->doFreeWhenReady();
			alloc = NULL;

//			DEBUG_HANDLE_ERROR(cudaFree(dPtr));

			dPtr = NULL;
		}
#endif
	}

	/**
	 * Delete host data
	 */
	void freeHost()
	{
// #ifdef DEBUG_CUDA
// 		if (hPtr == NULL)
// 			ACC_PTR_DEBUG_FATAL("free_host() called on NULL pointer.\n");
// #endif
		if(doFreeHost==false || hPtr==NULL)
		{
			printf("!!!!!!!!!!!!!!!!!free error host in ptr %p\n",hPtr);fflush(stdout);
		}
		doFreeHost = false;
		// printf("free host in ptr %p\n",hPtr);fflush(stdout);
		if (NULL != hPtr)
        {
            #ifdef NO_PINNED_ALLOCATOR
            {
                free(hPtr); 
            }
            #else
            {
                pin_free(hPtr);
            }
            #endif
        
            #ifdef ALLOCATOR_LOG
            {
                printf("free host in ptr %p\n",hPtr);
                fflush(stdout);
            }
            #endif
        }
        hPtr = NULL;
		// printf("free host in ptr end\n");fflush(stdout);

	}

	void freeHostIfSet()
	{
		if (doFreeHost)
			freeHost();
	}

	void freeDeviceIfSet()
	{
		if (doFreeDevice)
			freeDevice();
	}

	/**
	 * Delete both device and host data
	 */
	void freeBoth()
	{
		freeDevice();
		freeHost();
	}

	void freeIfSet()
	{
		freeHostIfSet();
		freeDeviceIfSet();
	}

	~AccPtr()
	{
		// printf("AccPtr destructor called.\n");
		freeIfSet();
		// printf("AccPtr destructor finished.\n");
	}


	/*======================================================
	                   GETTERS AND SETTERS
	======================================================*/


	bool willFreeHost()
	{
		return doFreeHost;
	}

	bool willFreeDevice()
	{
		return doFreeDevice;
	}

	void setStream(StreamType s)
	{
		stream = s;
	}

	StreamType getStream()
	{
		return stream;
	}

	void setSize(size_t s)
	{
		size = s;
	}
	
	size_t getSize()
	{
		return size;
	}

	T *getDevicePtr()
	{
		return dPtr;
	}

	T *getHostPtr()
	{
		return hPtr;
	}

	T *getAccPtr()
	{
		if (accType == accCUDA)
			return dPtr;
		else
			return hPtr;
	}

	void setAllocator(AllocatorType *a)
	{
		freeDeviceIfSet();
		allocator = a;
	};

	AllocatorType *getAllocator()
	{
		return allocator;
	}

	void setDevicePtr(T *ptr)
	{
#ifdef DEBUG_CUDA
			if (doFreeDevice)
				ACC_PTR_DEBUG_FATAL("Device pointer set without freeing the old one.\n");
#endif
		dPtr = ptr;
	}

	void setDevicePtr(const AccPtr<T> &ptr)
	{
#ifdef DEBUG_CUDA
		if (ptr.dPtr == NULL)
			ACC_PTR_DEBUG_FATAL("Device pointer is not set.\n");
#endif
		setDevicePtr(ptr.dPtr);
	}

	void setHostPtr(T *ptr)
	{
#ifdef DEBUG_CUDA
		if (doFreeHost)
			ACC_PTR_DEBUG_FATAL("Host pointer set without freeing the old one.\n");
#endif
		hPtr = ptr;
	}

	void setHostPtr(const AccPtr<T> &ptr)
	{
#ifdef DEBUG_CUDA
		if (ptr.hPtr == NULL)
			ACC_PTR_DEBUG_FATAL("Host pointer is not set.\n");
#endif
		setHostPtr(ptr.hPtr);
	}

	void setAccPtr(const AccPtr<T> &ptr)
	{
		if (accType == accCUDA)
			setDevicePtr(ptr.hPtr);
		else
			setHostPtr(ptr.hPtr);
	}

	void setAccPtr(T *ptr)
	{
		if (accType == accCUDA)
			setDevicePtr(ptr);
		else
			setHostPtr(ptr);
	}

	AccType getAccType()
	{
		return accType;
	}

	template <typename Tn>
	AccPtr<Tn> make()
	{
		CTIC("","make");
		AccPtr<Tn> ptr(stream, allocator);
		ptr.setAccType(accType);
		CTOC("","make");

		return ptr;
	}

	template <typename Tn>
	AccPtr<Tn> make(size_t s)
	{
		CTIC("","make");
		AccPtr<Tn> ptr(stream, allocator);
		ptr.setAccType(accType);
		ptr.setSize(s);
		CTOC("","make");

		return ptr;
	}
};

typedef unsigned char AccPtrBundleByte;

class AccPtrBundle: public AccPtr<AccPtrBundleByte>
{
private:
	size_t current_packed_pos;

public:
	AccPtrBundle():
		AccPtr<AccPtrBundleByte>(),
		current_packed_pos(0)
	{}

	AccPtrBundle(const AccPtrBundle& other) :
		AccPtr<AccPtrBundleByte>(other),
		current_packed_pos(other.current_packed_pos)
	{}

	AccPtrBundle(StreamType stream, AllocatorType *allocator):
		AccPtr<AccPtrBundleByte>(stream, allocator),
		current_packed_pos(0)  
	{}

	AccPtrBundle(size_t size, StreamType stream, AllocatorType *allocator):
		AccPtr<AccPtrBundleByte>(stream, allocator),
		current_packed_pos(0)
	{
		setSize(size);
	}

	template <typename T>
	void pack(AccPtr<T> &ptr)
	{
#ifdef _CUDA_ENABLED
	#ifdef DEBUG_CUDA
		if (current_packed_pos + ptr.getSize() > size)
			ACC_PTR_DEBUG_FATAL("Packing exceeds bundle total size.\n");
		if (hPtr == NULL)
			ACC_PTR_DEBUG_FATAL("Pack called on null host pointer.\n");
	#endif
		if (ptr.getHostPtr() != NULL)
			memcpy ( &hPtr[current_packed_pos], ptr.getHostPtr(), ptr.getSize() * sizeof(T));
		ptr.freeHostIfSet();
		ptr.setHostPtr((T*) &hPtr[current_packed_pos]);
		ptr.setDevicePtr((T*) &dPtr[current_packed_pos]);

		current_packed_pos += ptr.getSize() * sizeof(T);
#else
		if (ptr.getHostPtr() == NULL)
			ptr.hostAlloc();
#endif
	}
	
	//Overwrite allocation methods and block for no device
	
	void allAlloc()
	{
#ifdef _CUDA_ENABLED
		AccPtr<AccPtrBundleByte>::allAlloc();
#endif
	}
	
	void allAlloc(size_t size)
	{
#ifdef _CUDA_ENABLED
		AccPtr<AccPtrBundleByte>::allAlloc(size);
#endif
	}
	
	void hostAlloc()
	{
#ifdef _CUDA_ENABLED
AccPtr<AccPtrBundleByte>::hostAlloc();
#endif
	}
	
	void hostAlloc(size_t size)
	{
#ifdef _CUDA_ENABLED
AccPtr<AccPtrBundleByte>::hostAlloc(size);
#endif
	}
	
};

class AccPtrFactory
{
protected:
	AllocatorType *allocator;
	StreamType stream;

	AccType accType;

public:
	AccPtrFactory():
		allocator(NULL), stream(0), accType(accUNSET)
	{}

	AccPtrFactory(AccType accT):
		allocator(NULL), stream(0), accType(accT)
	{}

	AccPtrFactory(AllocatorType *alloc):
		allocator(alloc), stream(0), accType(accCUDA)
	{}

	AccPtrFactory(AllocatorType *alloc, StreamType s):
		allocator(alloc), stream(s), accType(accCUDA)
	{
		// printf("AccPtrFactory constructor called.\n");
	}

	~AccPtrFactory()
	{
		// printf("AccPtrFactory destructor called.\n");
	}

	template <typename T>
	AccPtr<T> make()
	{
		AccPtr<T> ptr(stream, allocator);
		ptr.setAccType(accType);

		return ptr;
	}

	template <typename T>
	AccPtr<T> make(size_t size)
	{
		AccPtr<T> ptr(stream, allocator);
		ptr.setAccType(accType);
		ptr.setSize(size);

		return ptr;
	}


	// template <typename T>
	// AccPtr<T> make(size_t size, StreamType s)
	// {

	// 	AccPtr<T> ptr(s, allocator);
	// 	ptr.setAccType(accType);
	// 	ptr.setSize(size);

	// 	return ptr;
	// }

	AccPtrBundle makeBundle()
	{
		AccPtrBundle bundle(stream, allocator);
		bundle.setAccType(accType);

		return bundle;
	}

	AccPtrBundle makeBundle(size_t size)
	{
		AccPtrBundle bundle(size, stream, allocator);
		bundle.setAccType(accType);

		return bundle;
	}


};


template <typename T> 
class AccPtrNew: public AccPtr<T>
{
protected:
	CudaAllocatorTask *allocator_task;
	size_t size_gpualloc=-1;
public:
	AccPtrNew():
		AccPtr<T>(),allocator_task(NULL)
	{}

	// AccPtrNew(CudaAllocatorTask *allocator):
	// 	AccPtr<T>(),allocator_task(allocator)
	// {}
	// AccPtrNew(CudaAllocatorTask *allocator,AllocatorType *allocator_old):
	// 	AccPtr<T>(allocator_old),allocator_task(allocator)
	// {}

	AccPtrNew(StreamType stream, CudaAllocatorTask *allocator):
		AccPtr<T>(stream),allocator_task(allocator)
	{}
	AccPtrNew(StreamType stream, CudaAllocatorTask *allocator,AllocatorType *allocator_old):
		AccPtr<T>(stream,allocator_old),allocator_task(allocator)
	{}
	AccPtrNew(size_t size, StreamType stream, CudaAllocatorTask *allocator):
		AccPtr<T>(size,stream),allocator_task(allocator)
	{}
	AccPtrNew(size_t size, StreamType stream, CudaAllocatorTask *allocator,AllocatorType *allocator_old):
		AccPtr<T>(size,stream,allocator_old),allocator_task(allocator)
	{}
	AccPtrNew(const AccPtrNew<T> &ptr, size_t start_idx, size_t size):
		AccPtr<T>(ptr,start_idx,size),allocator_task(ptr.allocator_task)
	{}
	AccPtrNew(const AccPtrNew<T> &ptr):
		AccPtr<T>(ptr),allocator_task(ptr.allocator_task)
	{}
	




	void deviceAlloc()
	{
		CTIC("","deviceAlloc");
#ifdef _CUDA_ENABLED
		if (this->accType == accCUDA)
		{
#ifdef DEBUG_CUDA
			if(size==0)
				ACC_PTR_DEBUG_FATAL("deviceAlloc called with size == 0");
			if (doFreeDevice)
				ACC_PTR_DEBUG_FATAL("Device double allocation.\n");
#endif
				this->doFreeDevice = true;
				size_gpualloc= this->size;
				// printf("Allocating  %lu bytes %luon device\n", this->size,sizeof(T));
				this->dPtr = (T*)allocator_task->alloc(this->size * sizeof(T));
		}
#endif
		CTOC("","deviceAlloc");
	}

	void freeDevice()
	{
#ifdef _CUDA_ENABLED
		if (this->accType == accCUDA)
		{
#ifdef DEBUG_CUDA
			if (dPtr == NULL || this->doFreeDevice==true)
			{
				printf("error free device\n");
				ACC_PTR_DEBUG_FATAL("Free device memory was called on NULL pointer in free_device().\n");
			}
#endif
			this->doFreeDevice = false;
			// printf("Freeing %lu  %lu bytes on device\n", size_gpualloc,sizeof(T));
			allocator_task->free((BYTE*)this->dPtr,size_gpualloc * sizeof(T));
			allocator_task = NULL;
			this->dPtr = NULL;
		}
#endif
	}

	// Following functions copy directly from AccPtr
	// Nothing to do here
	void deviceAlloc(size_t newSize)
	{
		this->size = newSize;
		size_gpualloc=newSize;
		deviceAlloc();
	}
	void allAlloc()
	{
		CTIC("","allAlloc");
		// printf("allAlloc\n");
		// fflush(stdout);
		deviceAlloc();
		this->hostAlloc();
		// printf("endallalloc\n");
		// fflush(stdout);
		CTOC("","allAlloc");

	}
	void allAlloc(size_t newSize)
	{
		CTIC("","allAlloc");

		this->size = newSize;
		size_gpualloc=newSize;
		deviceAlloc();
		this->hostAlloc();
		CTOC("","allAlloc");

	}

	void accAlloc()
	{
		if (this->accType == accCUDA)
			deviceAlloc();
		else
			this->hostAlloc();
	}

	void accAlloc(size_t newSize)
	{
		this->size = newSize;
		size_gpualloc=newSize;
		if (this->accType == accCUDA)
			deviceAlloc(newSize);
		else
			this->hostAlloc(newSize);
	}

	void putOnDevice()
	{
		deviceAlloc();
		this->cpToDevice();
	}

	void putOnDevice(size_t newSize)
	{
		this->size=newSize;
		deviceAlloc();
		this->cpToDevice();
	}


	void freeDeviceIfSet()
	{
		if (this->doFreeDevice)
			freeDevice();
	}

	void freeBoth()
	{
		freeDevice();
		this->freeHost();
	}

	void freeIfSet()
	{
		this->freeHostIfSet();
		freeDeviceIfSet();
	}

	void setAllocatorTask(CudaAllocatorTask *a)
	{
		this->freeDeviceIfSet();
		allocator_task = a;
	};

	CudaAllocatorTask* getAllocator_task()
	{
		return allocator_task;
	}

#ifdef _CUDA_ENABLED
	void putToPin()
	{
		cudaHostRegister(this->hPtr,sizeof(T) * this->size,cudaHostRegisterPortable);
	}
#endif

	template <typename Tn>
	AccPtrNew<Tn> make()
	{
		CTIC("","make");
		AccPtrNew<Tn> ptr(this->stream, allocator_task, this->allocator);
		ptr.setAccType(this->accType);
		CTOC("","make");

		return ptr;
	}

	template <typename Tn>
	AccPtrNew<Tn> make(size_t s)
	{
		CTIC("","make");
		AccPtrNew<Tn> ptr(this->stream, allocator_task, this->allocator);
		ptr.setAccType(this->accType);
		ptr.setSize(s);
		CTOC("","make");

		return ptr;
	}

	void copyFrom(AccPtr<T> &ptr)
	{
		if(ptr.getSize()!=this->size||this->hPtr==NULL)
		{
			printf("copyFrom: input size %lu not equal to this size %lu\n",ptr.getSize(),this->size);
			fflush(stdout);
			ACC_PTR_DEBUG_FATAL("copyFrom: input size not equal to this size");
		}
		memcpy(this->hPtr,ptr.getHostPtr(),this->size*sizeof(T));
	}



	void resizeHostCopy(size_t newSize)
	{
#ifdef DEBUG_CUDA
//		if (size==0)
//			ACC_PTR_DEBUG_INFO("Resizing from size zero (permitted).\n");
#endif
		// TODO - alternatively, this could be aligned std::vector
		T* newArr;
        #ifdef NO_PINNED_ALLOCATOR
        {
            if(posix_memalign((void **)&newArr, MEM_ALIGN, sizeof(T) * newSize))
			    CRITICAL(RAMERR);
        }
        #else
        {
            if(pin_alloc((void **)&newArr, sizeof(T) * newSize))
                CRITICAL(RAMERR);
        }
        #endif

        #ifdef ALLOCATOR_LOG
        {
            printf("start host alloc %lu %p noend\n",sizeof(T) * newSize,newArr);
            fflush(stdout);
        }
        #endif
		
		// Copy in what we can from the original matrix
		if ((this->size > 0) && (this->hPtr != NULL))
		{
			if (newSize < this->size)
				memcpy( newArr, this->hPtr, newSize * sizeof(T) );
			else
				memcpy( newArr, this->hPtr, this->size * sizeof(T) );  
			
			// Initialize remaining memory if any
			if (newSize > this->size)
			{
				size_t theRest = sizeof(T) * (newSize - this->size);
				memset( newArr, 0x0, theRest);
			}
		}
		
		// There was nothing from before to copy - clear new memory
		if (this->hPtr == NULL)
		{
			memset( newArr, 0x0, sizeof(T) * newSize);
		}

#ifdef DEBUG_CUDA
		if (dPtr!=NULL)
			ACC_PTR_DEBUG_FATAL("resizeHostCopy: Resizing host with present device allocation.\n");
		if (newSize==0)
			ACC_PTR_DEBUG_INFO("resizeHostCopy: Array resized to size zero (permitted with fear).  Something may break downstream\n");
#endif
		this->freeHostIfSet();
	    setSize(newSize);
	    this->setHostPtr(newArr);
	    this->doFreeHost=true;
	}
	
	void setSize(size_t s)
	{
		if(size_gpualloc<s)
		{
			printf("change size from %lu to %lu error\n",size_gpualloc,s);
			ACC_PTR_DEBUG_FATAL("setSize: input size larger than origin size");
		}
		this->size = s;
	}

	~AccPtrNew()
	{
		freeIfSet();
	}

	void print_detail()
	{
		printf("AccPtrNew: size %lu, size_gpualloc %lu, doFreeDevice %d, doFreeHost %d, hPtr %p, dPtr %p\n",
					this->size,size_gpualloc,this->doFreeDevice,this->doFreeHost,this->hPtr,this->dPtr);
	
	}
	//setXXXPtr/get/cp可能不需要，因为可以用AccPtr的类来完成指针赋值而，无需调用task相关内容
};



class AccPtrBundleNew: public AccPtrNew<AccPtrBundleByte>
{
private:
	size_t current_packed_pos;

public:
	// AccPtrBundleNew(StreamType stream, CudaAllocatorTask *allocator):
	// 	AccPtrNew<AccPtrBundleByte>(stream, allocator),
	// 	current_packed_pos(0)
	// {}

	// AccPtrBundleNew(size_t size, StreamType stream, CudaAllocatorTask *allocator):
	// 	AccPtrNew<AccPtrBundleByte>(stream, allocator),
	// 	current_packed_pos(0)
	// {
	// 	setSize(size);
	// }
	AccPtrBundleNew():
		AccPtrNew<AccPtrBundleByte>(),current_packed_pos(0)
	{}
	AccPtrBundleNew(const AccPtrBundleNew& other):
		AccPtrNew<AccPtrBundleByte>(other),current_packed_pos(0)
	{}
	
	AccPtrBundleNew(StreamType stream, CudaAllocatorTask *allocator,AllocatorType *allocator_old):
		AccPtrNew<AccPtrBundleByte>(stream, allocator,allocator_old),
		current_packed_pos(0)
	{}

	AccPtrBundleNew(size_t size, StreamType stream, CudaAllocatorTask *allocator,AllocatorType *allocator_old):
		AccPtrNew<AccPtrBundleByte>(stream, allocator,allocator_old),
		current_packed_pos(0)
	{
		setSize(size);
	}

	template <typename T>
	void pack(AccPtrNew<T> &ptr)
	{
#ifdef _CUDA_ENABLED
	#ifdef DEBUG_CUDA
		if (current_packed_pos + ptr.getSize() > size)
			ACC_PTR_DEBUG_FATAL("Packing exceeds bundle total size.\n");
		if (hPtr == NULL)
			ACC_PTR_DEBUG_FATAL("Pack called on null host pointer.\n");
	#endif
		if (ptr.getHostPtr() != NULL)
			memcpy ( &hPtr[current_packed_pos], ptr.getHostPtr(), ptr.getSize() * sizeof(T));
		ptr.freeHostIfSet();
		ptr.setHostPtr((T*) &hPtr[current_packed_pos]);
		ptr.setDevicePtr((T*) &dPtr[current_packed_pos]);

		current_packed_pos += ptr.getSize() * sizeof(T);
#else
		if (ptr.getHostPtr() == NULL)
			ptr.hostAlloc();
#endif
	}

	void change_task_allocator(CudaAllocatorTask *allocator)
	{
		this->allocator_task=allocator;
	}
	template <typename T>
	void pack(AccPtr<T> &ptr)
	{
#ifdef _CUDA_ENABLED
	#ifdef DEBUG_CUDA
		if (current_packed_pos + ptr.getSize() > size)
			ACC_PTR_DEBUG_FATAL("Packing exceeds bundle total size.\n");
		if (hPtr == NULL)
			ACC_PTR_DEBUG_FATAL("Pack called on null host pointer.\n");
	#endif
		if (ptr.getHostPtr() != NULL)
			memcpy ( &hPtr[current_packed_pos], ptr.getHostPtr(), ptr.getSize() * sizeof(T));
		ptr.freeHostIfSet();
		ptr.setHostPtr((T*) &hPtr[current_packed_pos]);
		ptr.setDevicePtr((T*) &dPtr[current_packed_pos]);

		current_packed_pos += ptr.getSize() * sizeof(T);
#else
		if (ptr.getHostPtr() == NULL)
			ptr.hostAlloc();
#endif
	}
	
	//Overwrite allocation methods and block for no device
	
	void allAlloc()
	{
#ifdef _CUDA_ENABLED
		AccPtrNew<AccPtrBundleByte>::allAlloc();
#endif
	}
	
	void allAlloc(size_t size)
	{
#ifdef _CUDA_ENABLED
		AccPtrNew<AccPtrBundleByte>::allAlloc(size);
#endif
	}
	
	void hostAlloc()
	{
#ifdef _CUDA_ENABLED
		AccPtrNew<AccPtrBundleByte>::hostAlloc();
#endif
	}
	
	void hostAlloc(size_t size)
	{
#ifdef _CUDA_ENABLED
		AccPtrNew<AccPtrBundleByte>::hostAlloc(size);
#endif
	}

	void free()
	{
		AccPtrNew<AccPtrBundleByte>::freeIfSet();
	}
	
};



class AccPtrFactoryNew: public AccPtrFactory
//用来在各个函数之间，传递allocator和stream，方便新建AccPtr的
{
protected:
	CudaAllocatorThread *allocator_thread;
	CudaAllocatorTask *allocator_task;
public:

	AccPtrFactoryNew(StreamType s,CudaAllocatorThread *allocator_thread,AllocatorType *allocator):
		AccPtrFactory(allocator,s),allocator_thread(allocator_thread),allocator_task(NULL)
	{
		// printf("AccPtrFactoryNew constructor called. %p %p\n",this,allocator_thread);fflush(stdout);
	}
	AccPtrFactoryNew():
		AccPtrFactory(),allocator_thread(NULL),allocator_task(NULL)
	{}

	bool ifTaskCanUse(int partid,size_t size)
	{
		size_t double_size;//=size*3;
		if(partid>1 && size*1.2<100*1024*1024)
		{
			double_size=100*1024*1024;
		}
		else
		{
			double_size=size*1.2;
		}
		return allocator_thread->canAdd(double_size, partid);
	}
	void getOneTaskAllocator(int partid,size_t size)
	{
		size_t double_size;//=size*3;
		if(partid>1 && size*1.2<100*1024*1024)
		{
			double_size=100*1024*1024;
		}
		else
		{
			double_size=size*1.2;
		}
		freeTaskAllocator();
		// printf("getOneTaskAllocator %d %lu %p %p\n",partid,double_size,allocator_thread,allocator_task);fflush(stdout);
		// double_size=1000;
		// if(allocator_task!=NULL)
		// 	allocator_thread->freeOneTask(allocator_task);
			// delete allocator_task;
		// printf("getOneTaskAllocator %d %lu %p %p\n",partid,double_size,allocator_thread,this);fflush(stdout);
		// exit(-1);
		allocator_task=allocator_thread->addOneTask(partid,double_size);
	}
	void freeTaskAllocator()
	{
		if(allocator_task!=NULL)
		{
			// printf("task not null\n");
			// fflush(stdout);
			// delete allocator_task;
			// printf("free task %p %p\n",allocator_task->getStart(),allocator_thread);fflush(stdout);
			allocator_thread->freeOneTask(allocator_task);
			allocator_task=NULL;
		}
		else
		{
			// printf("task null\n");
			// fflush(stdout);
		}
	}

	CudaAllocatorTask* getAllocator_task()
	{
		return allocator_task;
	}

	template <typename T>
	AccPtrNew<T> make()
	{
		AccPtrNew<T> ptr(this->stream, allocator_task,this->allocator);
		// AccPtrNew<T> ptr(this->stream, this->allocator);
		ptr.setAccType(accType);

		return ptr;
	}

	template <typename T>
	AccPtrNew<T> make(size_t size)
	{
		AccPtrNew<T> ptr(this->stream, allocator_task,this->allocator);
		// AccPtrNew<T> ptr(this->stream, this->allocator);
		ptr.setAccType(accType);
		ptr.setSize(size);

		return ptr;
	}


	// template <typename T>
	// AccPtrNew<T> make(size_t size, StreamType s)
	// {
	// 	// AccPtrNew<T> ptr(s,this->allocator);
	// 	AccPtrNew<T> ptr(s, allocator_task,this->allocator);
	// 	ptr.setAccType(accType);
	// 	ptr.setSize(size);

	// 	return ptr;
	// }


	AccPtrBundleNew makeBundle()
	{
		AccPtrBundleNew bundle(this->stream, allocator_task,this->allocator);
		bundle.setAccType(accType);

		return bundle;
	}

	AccPtrBundleNew makeBundle(size_t size)
	{
		AccPtrBundleNew bundle(size, this->stream, allocator_task,this->allocator);
		bundle.setAccType(accType);

		return bundle;
	}
	
	StreamType getStream()
	{
		return stream;
	}

};
#endif
