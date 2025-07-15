#include "src/acc/cuda/custom_allocator.cuh"
CudaAllocatorTask::~CudaAllocatorTask()
{
	if(task_p!=task_start)
	{
		printf("\033[0;31m  ERROR: Still have array in the task???\n \033[0m");
	}
	// printf("~~\033[0;32m  Freeing task %p\n \033[0m", this);
	// threadAllocator->freeOneTask(this);
}
