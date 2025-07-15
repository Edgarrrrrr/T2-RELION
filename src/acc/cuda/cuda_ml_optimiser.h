#ifndef CUDA_ML_OPTIMISER_H_
#define CUDA_ML_OPTIMISER_H_

#include "src/mpi.h"
#include "src/ml_optimiser.h"
#include "src/acc/cuda/cuda_mem_utils.h"
#include "src/acc/acc_projector_plan.h"
#include "src/acc/acc_projector.h"
#include "src/acc/acc_backprojector.h"
#include "src/acc/cuda/cuda_fft.h"
#include "src/acc/cuda/cuda_benchmark_utils.h"
#include <stack>
//#include <cufft.h>

#include "src/acc/acc_ml_optimiser.h"
#include "src/acc/acc_ptr.h"

class MlDeviceBundle
{
public:

	//The CUDA accelerated projector set
	std::vector< AccProjector > projectors;

	//The CUDA accelerated back-projector set
	std::vector< AccBackprojector > backprojectors;

	//Used for precalculations of projection setup
	CudaCustomAllocator *allocator;
	CudaAllocatorProcess *allocator_p;

	//Used for precalculations of projection setup
	bool generateProjectionPlanOnTheFly;
	std::vector< AccProjectorPlan > coarseProjectionPlans;

	MlOptimiser *baseMLO;

	int device_id;

	int rank_shared_count;

	bool haveWarnedRefinementMem;

	MlDeviceBundle(MlOptimiser *baseMLOptimiser):
			baseMLO(baseMLOptimiser),
			generateProjectionPlanOnTheFly(false),
			rank_shared_count(1),
			device_id(-1),
			haveWarnedRefinementMem(false),
			allocator(NULL),
			allocator_p(NULL)
	{};

	void setDevice(int did)
	{
		device_id = did;
	}

	size_t checkFixedSizedObjects(int shares);
	void setupFixedSizedObjects();
	void setupTunableSizedObjects(size_t allocationSize,int threadnum=1);

	void syncAllBackprojects()
	{
		DEBUG_HANDLE_ERROR(cudaDeviceSynchronize());
	}


	~MlDeviceBundle()
	{
		projectors.clear();
		backprojectors.clear();
		coarseProjectionPlans.clear();
		//Delete this lastly
		delete allocator;
		// printf("tmpdebugxjl\n");
		if(allocator_p != NULL)
			delete allocator_p;
	}

};
class MlOptimiserCuda
{
public:
	// transformer as holder for reuse of fftw_plans
	FourierTransformer transformer;

   //Class streams ( for concurrent scheduling of class-specific kernels)
	std::vector< cudaStream_t > classStreams;
	cudaError_t errorStatus;

	CudaFFT transformer1;
	CudaFFT transformer2;
	MlOptimiser *baseMLO;

	bool refIs3D;
	bool dataIs3D;

	int device_id;

	MlDeviceBundle *bundle;

	//Used for precalculations of projection setup
	CudaCustomAllocator *allocator;
	CudaAllocatorProcess *allocator_p;

	//Used for precalculations of projection setup
	bool generateProjectionPlanOnTheFly;

	//Used for task parallelism
	int nr_task_limit = 1;
	std::stack<cudaStream_t> streamPool;


#ifdef TIMING_FILES
	relion_timer timer;
#endif

	MlOptimiserCuda(MlOptimiser *baseMLOptimiser, MlDeviceBundle* bundle, const char * timing_fnm) :
			baseMLO(baseMLOptimiser),
			transformer1(cudaStreamPerThread, bundle->allocator, baseMLOptimiser->mymodel.data_dim),
			transformer2(cudaStreamPerThread, bundle->allocator, baseMLOptimiser->mymodel.data_dim),
			refIs3D(baseMLO->mymodel.ref_dim == 3),
			dataIs3D(baseMLO->mymodel.data_dim == 3),
			bundle(bundle),
			device_id(bundle->device_id),
#ifdef TIMING_FILES
			timer(timing_fnm),
#endif
			errorStatus((cudaError_t)0),
			allocator(bundle->allocator),
			allocator_p(bundle->allocator_p),
			generateProjectionPlanOnTheFly(bundle->generateProjectionPlanOnTheFly)
	{};

	void resetData();

	void doThreadExpectationSomeParticles(int thread_id,int tasknum);

	~MlOptimiserCuda()
	{
		for (int i = 0; i < classStreams.size(); i++)
			if (classStreams[i] != NULL)
				HANDLE_ERROR(cudaStreamDestroy(classStreams[i]));
		while (!streamPool.empty()) {
            cudaStream_t stream = streamPool.top();
            streamPool.pop();
            HANDLE_ERROR(cudaStreamDestroy(stream));
        }
	}

	CudaCustomAllocator *getAllocator()	
	{
		return (bundle->allocator);
	};

	// This funciton is not thread safe
	cudaStream_t acquireStream()
    {
        if (streamPool.empty())
        {
            // If the pool is empty, create a new stream
            cudaStream_t newStream;
            errorStatus = cudaStreamCreate(&newStream);
            HANDLE_ERROR(errorStatus);
            return newStream;
        }
        else
        {
            // Reuse an existing stream from the pool
            cudaStream_t reusedStream = streamPool.top();
            streamPool.pop();
            return reusedStream;
        }
    }

	// This funciton is not thread safe
	void releaseStream(cudaStream_t stream)
    {
        // Optionally, you can synchronize the stream before reusing
        errorStatus = cudaStreamSynchronize(stream);
        HANDLE_ERROR(errorStatus);

        // Push the stream back into the pool for reuse
        streamPool.push(stream);
    }
	
};

#endif
