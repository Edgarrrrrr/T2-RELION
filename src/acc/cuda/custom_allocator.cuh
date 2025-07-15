#ifndef CUDA_CUSTOM_ALLOCATOR_CUH_
#define CUDA_CUSTOM_ALLOCATOR_CUH_
// This is where custom allocator should be. Commented out for now, to avoid double declaration.

#ifdef _CUDA_ENABLED
#include "src/acc/cuda/cuda_settings.h"
#include "src/acc/cuda/cuda_benchmark_utils.h"

#include <cuda_runtime.h>
#endif

#include <signal.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <list>
#include <stdexcept>

#include "src/macros.h"
#include "src/error.h"
#include "src/parallel.h"

#ifdef CUSTOM_ALLOCATOR_MEMGUARD
#include <execinfo.h>
#include <cxxabi.h>
#endif

#ifdef DUMP_CUSTOM_ALLOCATOR_ACTIVITY
#define CUSTOM_ALLOCATOR_REGION_NAME( name ) (fprintf(stderr, "\n%s", name))
#else
#define CUSTOM_ALLOCATOR_REGION_NAME( name ) //Do nothing
#endif


#include <execinfo.h>
#include <cxxabi.h>

inline void printStackTrace() {
    const int maxFrames = 64;
    void* buffer[maxFrames];
    int numFrames = backtrace(buffer, maxFrames);
    char** symbols = backtrace_symbols(buffer, numFrames);

    if (symbols == nullptr) {
        printf("Failed to retrieve stack trace.\n");
        return;
    }

    printf("Stack trace:\n");
    for (int i = 0; i < numFrames; ++i) {
        char* demangledName = nullptr;
        char* leftParen = strchr(symbols[i], '(');
        char* plusSign = strchr(symbols[i], '+');
        if (leftParen && plusSign && leftParen < plusSign) {
            *plusSign = '\0';
            int status;
            demangledName = abi::__cxa_demangle(leftParen + 1, nullptr, nullptr, &status);
            *plusSign = '+';
        }

        if (demangledName) {
            printf("  [%d] %s\n", i, demangledName);
            free(demangledName);
        } else {
            printf("  [%d] %s\n", i, symbols[i]);
        }
    }

    free(symbols);
}

typedef unsigned char BYTE;
class CudaAllocatorThread;
class CudaAllocatorTask
{
private:
	const BYTE *task_start;
	const size_t alignmentSize;
	size_t task_size;
	BYTE *task_p;//空间内部当前指向的位置
	int partid;
	CudaAllocatorThread* threadAllocator; 
	int died;

public:
	CudaAllocatorTask(BYTE *task_start,size_t alignmentSize,size_t task_size,int partid,CudaAllocatorThread* threadAllocator):
		task_start(task_start),alignmentSize(alignmentSize),task_size(task_size),partid(partid),threadAllocator(threadAllocator), died(0)
	{
		task_p = task_start;
	}
	~CudaAllocatorTask();//因为用到之后才定义的thread，所以要在cu文件里面定义


	BYTE *alloc(size_t requestedSize)
	{
		// FILE *file = fopen("real_mem.txt", "a+"); 
		size_t size = alignmentSize*ceilf( (float)requestedSize / (float)alignmentSize);
		// size=size;
		// fprintf(file,"%lld\n",task_p+size-task_start);
		// fclose(file);
		// printf("task request:%lu  alloc:%lu  place: %lu %p %p %p\n",requestedSize,size,
		// 		task_p-task_start,task_p,task_p+size,this);
		if(task_p+size<=task_start+task_size)
		{
			BYTE *tmp=task_p;
			task_p+=size;
			// printf("task request:%lu  alloc:%lu  place: %lu %p %p %p\n",requestedSize,size,
			// 		task_p-task_start,tmp,task_p,this);
			// fflush(stdout);
			return tmp;
		}
		else
		{
			printf("\033[0;31m ERROR: CudaAllocatorTask out of memory when add a variable\n [requestedSpace:             %lu B]\n [FreeSpacetotail:              %lu B]\n [totalSpace:             %lu B]\n",
					(unsigned long) size, (unsigned long) (task_start+task_size-task_p), (unsigned long) (task_size));
			printf(" place over:%p %p %p %lu\n",task_start,task_p,task_start+task_size,task_start+task_size-task_p);
			fflush(stdout);
			// CRITICAL(ERRCUDACAOOM);
			throw std::bad_alloc();
			return NULL;
		}
	}
	void free(BYTE *ptr)
	{
		if(ptr<task_start||ptr>=task_start+task_size)
		{
			printf("ERROR: CudaAllocatorTask free out of range! please check your free order\n");
			// CRITICAL(ERRCUDACAOOM);
			throw std::runtime_error("CudaAllocatorTask: Bad free detected. Pointer is out of range.");

		}
		else
		{
			task_p=ptr;
		}
	}
	void free(BYTE *ptr,size_t size)
	{
		size = alignmentSize*ceilf( (float)size / (float)alignmentSize);
		// size=size*2;
		// FILE *file = fopen("real_mem.txt", "a+"); 
		// fprintf(file,"%lld\n",task_p-size-task_start);fclose(file);
		if(task_p-size==ptr && ptr>=task_start && ptr<task_start+task_size)
		{
			task_p=ptr;
			// printf("task free %lu %lu after:%p\n",size,task_p-task_start,ptr);
			// fflush(stdout);
		}
		else
		{
			// printf("task free %lu %lu %p\n",size,task_p-task_start,this);
			printf("task error free %p %p  ptr %p %lu     %p\n",task_p,task_p-size,ptr,size,this);

			printf("ERROR: CudaAllocatorTask free out of range!please check your free order\n");
			fflush(stdout);
			printStackTrace();  // 打印调用栈


			// CRITICAL(ERRCUDACAOOM);
			throw std::runtime_error("CudaAllocatorTask: Bad free detected. Pointer is out of range.");
		}
	}
	void addTaskSize(size_t size)
	{
		task_size+=size;
	}
	int getPartid()
	{
		return partid;
	}
	size_t getSize()
	{
		return task_size;
	}
	void print_p()
	{
		printf("task_p:%p task_start:%p task_size:%lld taskalign:%lu\n",
				task_p,task_start,task_size,alignmentSize);
	}
	const BYTE *getStart()
	{
		return task_start;
	}
	void setDied()
	{
		died=1;
	}
	int getDied()
	{
		return died;
	}
};

class CudaAllocatorThread//不需要保证先进先出，但是后进先出的部分不会释放空间哦
{
private:
	BYTE *thread_start;
	size_t alignmentSize;
	size_t thread_size;

	// BYTE *part1_start,*part2_start,*part3_start;//整体空间的开始和结束
	// size_t part1_size,part2_size,part3_size;

	// BYTE *part1_s,*part1_e,*part2_s,*part2_e,*part3_s,*part3_e;//当前使用了区间的开始和结束
	// size_t part1_remained,part2_remained,part3_remained;
	// std::list<CudaAllocatorTask*> task_list1,task_list2,task_list3;

	BYTE *part_start,*part_s,*part_e;
	size_t part_size,part_remained;
	std::list<CudaAllocatorTask*> task_list;


	void _add_task(BYTE* &part_s,BYTE* &part_e,size_t &part_remained,
					BYTE *part_start,size_t part_size,size_t size,
					std::list<CudaAllocatorTask*> &task_list,int partid)
	{
		if(part_remained<size)
		{
			printf("ERROR: CudaAllocatorThread out of memory\n [requestedSpace:             %lu B]\n [largestContinuousFreeSpace: %lu B]\n [totalSpace:             %lu B]\n",(unsigned long) size, (unsigned long) part_remained, (unsigned long) (part_size));
			printf("ERROR: CudaAllocatorThread out of memory:%lf M\n",(float)size/1024/1024);
			// CRITICAL(ERRCUDACAOOM);
			throw std::bad_alloc();
		}
		BYTE* part_end=part_start+part_size;
		// printf("debug allocator thread:%p %p   %p %p  %lld %lld\n",part_start,part_s,part_e,part_e+size,size,part_remained);

		// std::list<CudaAllocatorTask*>::iterator it;
    	// for (it = task_list.begin(); it != task_list.end(); ++it) {
        // 	(*it)->print_p();
    	// }
		// printf("debug allocator thread alloc0:%p %p   %p %p\t %p %d\n",part_start,part_end,part_s,part_e,this,size);

		if(part_s>part_e)
		{
			if(part_e+size<part_s)
			{
				BYTE* task_start=part_e;
				part_e+=size;
				part_remained-=size;
				CudaAllocatorTask *task = new CudaAllocatorTask(task_start,alignmentSize,size,partid,this);//记得要析构！！！！！
				task_list.push_back(task);
			}
			else{
				printf("something wrong with the allocator thread\n This should not happen\n");
				throw std::bad_alloc();
			}
		}
		else{
			if(part_e+size<part_end)
			{
				BYTE* task_start=part_e;
				part_e+=size;
				part_remained-=size;
				CudaAllocatorTask *task = new CudaAllocatorTask(task_start,alignmentSize,size,partid,this);//记得要析构！！！！！

				task_list.push_back(task);
			}
			else
			{
				if(task_list.size()==0)
				{
					part_e=part_s=part_start;
					part_remained=part_size;
				}
				else
				{
					CudaAllocatorTask *last_task=task_list.back();
					last_task->addTaskSize(part_end-part_e);
					part_remained-=part_end-part_e;
					// part_e=part_s;//???

					// CudaAllocatorTask *first_task=task_list.front();
					// printf("\t\t\t\tdebug allocator thread\t\t\t\t:%p %p   %p %p \n",part_start,part_end,part_s,part_e);
					part_e=part_start;
					// part_s=part_start;//???
					// part_e=first_task->getStart();
					// part_e = const_cast<BYTE*>(first_task->getStart());

				}
				_add_task(part_s,part_e,part_remained,part_start,part_size,size,task_list,partid);

			}
		}
		// printf("debug allocator thread alloc:%p %p   %p %p\t %p %d\n",part_start,part_end,part_s,part_e,this,size);
		// if(part_s+size>part_e)
		// {
		// 	printf("ERROR: alloc error!!!: %p %p   %p %p \t %p %d\n",part_start,part_end,part_s,part_e,this,task_list.size());
		// 	// CRITICAL(ERRCUDACAOOM);
		// 	fflush(stdout);
		// 	throw std::runtime_error("CudaAllocatorThread: Bad allocation detected. Pointer is out of range.");
		// }

	}
	void _free_task(BYTE* &part_s,BYTE* part_e,size_t &part_remained,
					BYTE *part_start,size_t part_size,size_t size,
					std::list<CudaAllocatorTask*> &task_list,CudaAllocatorTask* ptr)
	{
		
		CudaAllocatorTask* tmp=task_list.front();
		BYTE* part_end=part_start+part_size;
		ptr->setDied();

		// printf("debug allocator thread free:%p %p   %p %p \t %p  %d\n",part_start,part_end,part_s,part_e,this,size);
		// printf("debug allocator thread free0:%p %d %d\n",tmp->getStart(),tmp->getSize(),tmp->getPartid());
		// fflush(stdout);
		if(tmp==ptr)
		{
			for (std::list<CudaAllocatorTask*>::iterator it = task_list.begin(); it != task_list.end(); )
			{
				CudaAllocatorTask* task = *it;

				// 判断条件是否满足
				if (task->getDied()==1)
				{
					size_t size = task->getSize();	
					part_remained+=size;
					part_s+=size;
					if(part_s==part_end)
					{
						part_s=part_start;
					}
					else if(part_s>part_end)
					{
						printf("ERROR: CudaAllocatorThread pointer error: %p %p   %p %p \t %p %d\n",part_start,part_end,part_s,part_e,this,size);
						// CRITICAL(ERRCUDACAOOM);
						fflush(stdout);
						throw std::runtime_error("CudaAllocatorThread: Bad free detected. Pointer is out of range.");
					}
					it = task_list.erase(it); // erase 会返回下一个有效的迭代器
					delete task; // 确保释放内存
				}
				else
				{
					// 不满足条件，停止遍历
					break;
				}
				// if(part_s!=task_list.front()->getStart())
				// {
				// 	printf("something wrong with the allocator thread\n This should not happen\n");
				// 	throw std::bad_alloc();
				// }
			}
		}
		else
		{
			for(std::list<CudaAllocatorTask*>::iterator it=task_list.begin();it!=task_list.end();it++)
			{
				if(*it==ptr)
				{
					// task_list.erase(it);
					// CudaAllocatorTask* tmp=*(it--);
					// tmp->addTaskSize(size);
					// printf("%p\t",*it);

					// printf("\t\t debug allocator thread:%p %p   %p %p \n",part_start,part_end,part_s,part_e);
					return;
				}
			}
			printf("Cannot find task in task_list%d!!!!!!!!!!!!!!!!%p %p\n",ptr->getPartid(), ptr->getStart(), tmp->getStart());
			throw std::runtime_error("CudaAllocatorThread: Bad free detected.");

		}
		// printf("free task %p %p end\n",tmp,ptr);
		// printf("\t\tdebug free thread:%p    %p %p %lld %lld\n",part_start,part_e,part_e+size,size,part_remained);
		// printf("\t\t debug allocator thread:%p %p   %p %p \n",part_start,part_end,part_s,part_e);

	}

public:
	CudaAllocatorThread(){}
	CudaAllocatorThread(BYTE *thread_start,size_t alignmentSize,size_t thread_size):
		thread_start(thread_start),alignmentSize(alignmentSize),thread_size(thread_size){}
	~CudaAllocatorThread()
	{
		if(task_list.size()==0)
		{
			return ;
		}
		for(std::list<CudaAllocatorTask*>::iterator it=task_list.begin();it!=task_list.end();it++)
		{
			delete *it;
		}
		task_list.clear();
		/*
		if(task_list1.size()==0 && task_list2.size()==0 && task_list3.size()==0)
		{
			return ;
		}
		
		for(std::list<CudaAllocatorTask*>::iterator it=task_list1.begin();it!=task_list1.end();it++)
		{
			delete *it;
		}
		for(std::list<CudaAllocatorTask*>::iterator it=task_list2.begin();it!=task_list2.end();it++)
		{
			delete *it;
		}
		for(std::list<CudaAllocatorTask*>::iterator it=task_list3.begin();it!=task_list3.end();it++)
		{
			delete *it;
		}
		*/
		printf("\033[0;31m ERROR: There are still task when kill thread allocator????\n\n\n\n \033[0m\n");
	}
	void init(BYTE *thread_start_input,size_t alignmentSize_input,
			size_t thread_size_input,float division_ratio1=0.01,float division_ratio2=0.3)
	{
		thread_start = thread_start_input;
		alignmentSize = alignmentSize_input;
		thread_size = thread_size_input;

		part_start = thread_start;
		part_size = thread_size;
		part_remained = part_size;
		part_s =part_e= part_start;
		/*
		size_t tmp = thread_size*division_ratio1;
		size_t tmp2= thread_size*division_ratio2;
		// std::cout<<tmp<<" "<<tmp2<<" "<<division_ratio1<<" "<<
			// division_ratio2<<" "<<thread_size-tmp-tmp2<<" "<<alignmentSize<<std::endl;
		part1_size = alignmentSize*floorf( (float)tmp / (float)alignmentSize);
		part2_size	= alignmentSize*floorf( (float)tmp2 / (float)alignmentSize);
		// std::cout<<part1_size<<" "<<part2_size<<" "<<thread_size-part2_size-part1_size<<std::endl;
		part3_size = thread_size-part1_size-part2_size;
		part1_start = thread_start;
		part2_start = part1_start + part1_size;
		part3_start = part2_start + part2_size;
		
		
		resetThreadInfo();
		// printf("part1_size,part2_size:%lld %lld %lld %lld\n",part1_size,part2_size,part3_size,thread_size_input);

		// printf("part1_start,part2_start:%p %p %p %p %p\n",part1_start,part2_start,part1_s,part1_e,this);
		*/

	}
	/*
	void resetThreadInfo(size_t part1_size_input,size_t part2_size_input,size_t part3_size_input)
	{
		part1_size = part1_size_input;
		part2_size = part2_size_input;
		part3_size = part3_size_input;
		part1_start = thread_start;
		part2_start = part1_start + part1_size;
		part3_start = part2_start + part2_size;

		resetThreadInfo();
	}
	void resetThreadInfo()
	{
		part1_s =part1_e= part1_start;
		part2_s =part2_e= part2_start;
		part3_s =part3_e= part3_start;

		part1_remained = part1_size;
		part2_remained = part2_size;
		part3_remained = part3_size;

	}*/
	CudaAllocatorTask* addOneTask(int partid,size_t required_size)
	{
		// printf("addOneTask %d %p %lld\n",partid,this,required_size);
		size_t size = alignmentSize*ceilf( (float)required_size / (float)alignmentSize);
		_add_task(part_s,part_e,part_remained,part_start,part_size,size,task_list,partid);
		return task_list.back();
		

		/*
		if(partid==1)
		{
			// printf("xjldebug task1 %p %p %llu %llu  %p\n",part1_s,part1_e,part1_remained,part1_size,part1_start);
			// fflush(stdout);
			_add_task(part1_s,part1_e,part1_remained,part1_start,part1_size,size,task_list1,partid);
			return task_list1.back();
		}
		else if(partid==2)
		{
			// printf("xjldebug task2 %p %p %llu %llu  %p %p\n",part2_s,part2_e,part2_remained,size,part2_start);
			_add_task(part2_s,part2_e,part2_remained,part2_start,part2_size,size,task_list2,partid);
			return task_list2.back();
		}
		else{
			_add_task(part3_s,part3_e,part3_remained,part3_start,part3_size,size,task_list3,partid);
			return task_list3.back();
		}
		*/
	}
	void freeOneTask(CudaAllocatorTask *ptr)
	{
		int partid=ptr->getPartid();
		size_t size = ptr->getSize();
		// printf("freeOneTask called by%d %p %lld\n",partid,ptr,size);
		_free_task(part_s,part_e,part_remained,part_start,part_size,size,task_list,ptr);
		/*
		int partid=ptr->getPartid();
		size_t size = ptr->getSize();
		// printf("free one task  %d %lld\n",partid,alignmentSize);
		if(partid==1)
		{
			_free_task(part1_s,part1_e,part1_remained,part1_start,part1_size,size,task_list1,ptr);
		}
		else if(partid==2)
		{
			_free_task(part2_s,part2_e,part2_remained,part2_start,part2_size,size,task_list2,ptr);
		}
		else
		{
			_free_task(part3_s,part3_e,part3_remained,part3_start,part3_size,size,task_list3,ptr);
		}
		// printf("free one task end%d\n",partid);
		*/
	}


	bool canAdd(size_t size,int partid)
	{
		size_t real_size = alignmentSize*ceilf( (float)size / (float)alignmentSize);
		if(real_size>part_remained)
		{
			return false;
		}
		// printf("debug allocator thread canAdd %p %p   %p %p \t %p %d\n",part_start,part_start+part_size,part_s,part_e,this,real_size);
		BYTE* part_end=part_start+part_size;
		// if(part_e+size<part_end) return true;
		// return false;

		if(part_s>part_e)
		{
			if(part_e+real_size<part_s)
			{
				return true;
			}
			else{
				return false;
			}
		}
		else{
			if(part_e+real_size<part_end)
			{
				return true;
			}
			else if(part_start+real_size<part_s)
			{
				return true;
			}
			else
			{
				return false;
			}
		}
	}
};


// CudaAllocatorTask::~CudaAllocatorTask()
// {
// 	if(task_p!=task_start)
// 	{
// 		printf("\033[0;31m  ERROR: Still have array in the task???\n \033[0m");
// 	}
// 	threadAllocator->freeOneTask(this);
// }

class CudaAllocatorProcess
{
private:
	// Alloc *first;
	BYTE *start;
	int threadCount;
	size_t totalSize;
	size_t alignmentSize;
	size_t threadSize;
	std::vector<CudaAllocatorThread> threads;

public:
	CudaAllocatorProcess(size_t size,int threadCount, size_t alignmentSize=64):
		totalSize(size), alignmentSize(alignmentSize),threadCount(threadCount)
	{
		HANDLE_ERROR(cudaMalloc( (void**) &(start), totalSize));
		size_t threadSize_avg = totalSize / threadCount;
		// threadCount=10;//xjldebug
		
		threadSize = alignmentSize*floorf( (float)threadSize_avg / (float)alignmentSize);
		threads.resize(threadCount);
		for(int i=0;i<threadCount;i++)
		{
			BYTE * thread_start_i = start + i*threadSize;
			threads[i].init(thread_start_i,alignmentSize,threadSize,0.01,0.2);
		}
		printf("threadSize:%lld\n",threadSize);
		printf("threadCount:%lld\n",threadCount);
		printf("totalSize:%lld\n",totalSize);
	}

	~CudaAllocatorProcess()
	{		
		DEBUG_HANDLE_ERROR(cudaFree(start));
		// printf("CudaAllocatorProcess::~CudaAllocatorProcess()\n");
		// fflush(stdout);
	}


	CudaAllocatorThread* get_thread(int threadid)
	{
		return &(threads[threadid]);
	}

};

class CudaCustomAllocator
{

	typedef unsigned char BYTE;

	const static unsigned GUARD_SIZE = 4;
	const static BYTE GUARD_VALUE = 145;
	const static int ALLOC_RETRY = 500;

public:

	class Alloc
	{
		friend class CudaCustomAllocator;

	private:
		Alloc *prev, *next;//链表的前后邻居
		BYTE *ptr;//地址
		size_t size;
		bool free;//块是否空闲
		cudaEvent_t readyEvent; //Event record used for auto free
		bool freeWhenReady;//看到了就释放


#ifdef CUSTOM_ALLOCATOR_MEMGUARD
		BYTE *guardPtr;
		void *backtrace[20];
		size_t backtraceSize;
#endif

		Alloc():
			prev(NULL), next(NULL),
			ptr(NULL),
			size(0),
			free(0),
			readyEvent(0),
			freeWhenReady(false)
		{}

		~Alloc()
		{
			prev = NULL;
			next = NULL;
			ptr = NULL;

			if (readyEvent != 0)
				DEBUG_HANDLE_ERROR(cudaEventDestroy(readyEvent));
		}

	public:
		inline
		BYTE *getPtr() { return ptr; }

		inline
		size_t getSize() { return size; }

		inline
		bool isFree() { return free; }

		inline
		cudaEvent_t getReadyEvent() { return readyEvent; }

		inline
		void markReadyEvent(cudaStream_t stream = 0)
		{
			//TODO add a debug warning if event already set
			DEBUG_HANDLE_ERROR(cudaEventCreate(&readyEvent));
			DEBUG_HANDLE_ERROR(cudaEventRecord(readyEvent, stream));
		}

		inline
		void doFreeWhenReady() { freeWhenReady = true; }
	};

private:

	Alloc *first;
	size_t totalSize;
	size_t alignmentSize;

	bool cache;//块空的时候是否要和前后合并

	omp_lock_t mutex;


	//Look for the first suited space
	Alloc *_getFirstSuitedFree(size_t size)
	{
		Alloc *a = first;
		//If not the last and too small or not free go to next allocation region
		while (a != NULL && ( a->size <= size || ! a->free ) )
			a = a->next;

		return a;
	}

	//Free allocs with recorded ready events
	bool _syncReadyEvents()
	{
		bool somethingReady(false);
		Alloc *a = first;

		while (a != NULL)
		{
			if (! a->free && a->freeWhenReady && a->readyEvent != 0)
			{
				DEBUG_HANDLE_ERROR(cudaEventSynchronize(a->readyEvent));
				somethingReady = true;
			}

			a = a->next;
		}

		return somethingReady;
	}

	//Free allocs with recorded ready events
	bool _freeReadyAllocs()
	{
		bool somethingFreed(false);
		Alloc *next = first;
		Alloc *curr;

		while (next != NULL)
		{
			curr = next;
			next = curr->next;

			if (! curr->free && curr->freeWhenReady && curr->readyEvent != 0)
			{
				cudaError_t e = cudaEventQuery(curr->readyEvent);

				if (e == cudaSuccess)
				{
					_free(curr);
					next = first; //List modified, restart
					somethingFreed = true;
				}
				else if (e != cudaErrorNotReady)
				{
					_printState();
					HandleError( e, __FILE__, __LINE__ );
				}
			}
		}
		return somethingFreed;
	}

	size_t _getTotalFreeSpace()
	{
		if (cache)
		{
			size_t total = 0;
			Alloc *a = first;

			while (a != NULL)
			{
				if (a->free)
					total += a->size;
				a = a->next;
			}

			return total;
		}
		else
		{
			size_t free, total;
			DEBUG_HANDLE_ERROR(cudaMemGetInfo( &free, &total ));
			return free;
		}
	}

	size_t _getTotalUsedSpace()
	{
		size_t total = 0;
		Alloc *a = first;

		while (a != NULL)
		{
			if (!a->free)
				total += a->size;
			a = a->next;
		}

		return total;
	}

	size_t _getNumberOfAllocs()
	{

		size_t total = 0;
		Alloc *a = first;

		while (a != NULL)
		{
			if (!a->free)
				total ++;
			a = a->next;
		}

		return total;
	}

	size_t _getLargestContinuousFreeSpace()
	{
		if (cache)
		{
			size_t largest = 0;
			Alloc *a = first;

			while (a != NULL)
			{
				if (a->free && a->size > largest)
					largest = a->size;
				a = a->next;
			}

			return largest;
		}
		else
			return _getTotalFreeSpace();
	}

	void _printState()
	{
		size_t total = 0;
		Alloc *a = first;

		while (a != NULL)
		{
			total += a->size;
			if (a->free)
				printf("[%luB] ", (unsigned long) a->size);
			else if (a->freeWhenReady)
				printf("<%luB> ", (unsigned long) a->size);
			else
				printf("(%luB) ", (unsigned long) a->size);

			a = a->next;
		}

		printf("= %luB\n", (unsigned long) total);
		fflush(stdout);
	}

	void _free(Alloc* a)
	{
//		printf("free: %u ", a->size);
//		_printState();


#ifdef CUSTOM_ALLOCATOR_MEMGUARD
		size_t guardCount = a->size - (a->guardPtr - a->ptr);
		BYTE *guards = new BYTE[guardCount];
		cudaStream_t stream = 0;
		CudaShortcuts::cpyDeviceToHost<BYTE>( a->guardPtr, guards, guardCount, stream);
		DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));
		for (int i = 0; i < guardCount; i ++)
			if (guards[i] != GUARD_VALUE)
			{
				fprintf (stderr, "ERROR: CORRUPTED BYTE GUARDS DETECTED\n");

				char ** messages = backtrace_symbols(a->backtrace, a->backtraceSize);

				// skip first stack frame (points here)
				for (int i = 1; i < a->backtraceSize && messages != NULL; ++i)
				{
					char *mangled_name = 0, *offset_begin = 0, *offset_end = 0;

					// find parantheses and +address offset surrounding mangled name
					for (char *p = messages[i]; *p; ++p)
					{
						if (*p == '(')
						{
							mangled_name = p;
						}
						else if (*p == '+')
						{
							offset_begin = p;
						}
						else if (*p == ')')
						{
							offset_end = p;
							break;
						}
					}

					// if the line could be processed, attempt to demangle the symbol
					if (mangled_name && offset_begin && offset_end &&
						mangled_name < offset_begin)
					{
						*mangled_name++ = '\0';
						*offset_begin++ = '\0';
						*offset_end++ = '\0';

						int status;
						char * real_name = abi::__cxa_demangle(mangled_name, 0, 0, &status);

						// if demangling is successful, output the demangled function name
						if (status == 0)
						{
							std::cerr << "[bt]: (" << i << ") " << messages[i] << " : "
									  << real_name << "+" << offset_begin << offset_end
									  << std::endl;

						}
						// otherwise, output the mangled function name
						else
						{
							std::cerr << "[bt]: (" << i << ") " << messages[i] << " : "
									  << mangled_name << "+" << offset_begin << offset_end
									  << std::endl;
						}
//						free(real_name);
					}
					// otherwise, print the whole line
					else
					{
						std::cerr << "[bt]: (" << i << ") " << messages[i] << std::endl;
					}
				}
				std::cerr << std::endl;

//				free(messages);

				exit(EXIT_FAILURE);
			}
		delete[] guards;
#endif

		a->free = true;

		if (cache)
		{
			//Previous neighbor is free, concatenate
			if ( a->prev != NULL && a->prev->free)
			{
				//Resize and set pointer
				a->size += a->prev->size;
				a->ptr = a->prev->ptr;

				//Fetch secondary neighbor
				Alloc *ppL = a->prev->prev;

				//Remove primary neighbor
				if (ppL == NULL) //If the previous is first in chain
					first = a;
				else
					ppL->next = a;

				delete a->prev;

				//Attach secondary neighbor
				a->prev = ppL;
			}

			//Next neighbor is free, concatenate
			if ( a->next != NULL && a->next->free)
			{
				//Resize and set pointer
				a->size += a->next->size;

				//Fetch secondary neighbor
				Alloc *nnL = a->next->next;

				//Remove primary neighbor
				if (nnL != NULL)
					nnL->prev = a;
				delete a->next;

				//Attach secondary neighbor
				a->next = nnL;
			}
		}
		else
		{
			DEBUG_HANDLE_ERROR(cudaFree( a->ptr ));
			a->ptr = NULL;

			if ( a->prev != NULL)
				a->prev->next = a->next;
			else
				first = a->next; //This is the first link

			if ( a->next != NULL)
				a->next->prev = a->prev;

			delete a;
		}
	};

	void _setup()
	{
		first = new Alloc();
		first->prev = NULL;
		first->next = NULL;
		first->size = totalSize;
		first->free = true;

		if (totalSize > 0)
		{
			HANDLE_ERROR(cudaMalloc( (void**) &(first->ptr), totalSize));
			cache = true;
		}
		else
			cache = false;
	}

	void _clear()
	{
		if (first->ptr != NULL)
			DEBUG_HANDLE_ERROR(cudaFree( first->ptr ));

		first->ptr = NULL;

		Alloc *a = first, *nL;

		while (a != NULL)
		{
			nL = a->next;
			delete a;
			a = nL;
		}
	}

public:

	CudaCustomAllocator(size_t size, size_t alignmentSize):
		totalSize(size), alignmentSize(alignmentSize), first(0), cache(true)
	{
		_setup();

		omp_init_lock(&mutex);
	}

	void resize(size_t size)
	{
		Lock ml(&mutex);
		_clear();
		totalSize = size;
		_setup();
	}


	Alloc* alloc(size_t requestedSize)
	{
		CTIC("","deviceAlloc0");

		Lock ml(&mutex);

		_freeReadyAllocs();

		// printf("alloc: %u ", requestedSize);
		// _printState();

		size_t size = requestedSize;

#ifdef CUSTOM_ALLOCATOR_MEMGUARD
		//Ad byte-guards
		size += alignmentSize * GUARD_SIZE; //Ad an integer multiple of alignment size as byte guard size
#endif

#ifdef DUMP_CUSTOM_ALLOCATOR_ACTIVITY
		fprintf(stderr, " %.4f", 100.*(float)size/(float)totalSize);
#endif

		Alloc *newAlloc(NULL);

		if (cache)
		{
			size = alignmentSize*ceilf( (float)size / (float)alignmentSize) ; //To prevent miss-aligned memory

			Alloc *curAlloc = _getFirstSuitedFree(size);
			// printf("%lf KB  cudaallocator_debug  %lfMB %lf KB \n",(double)requestedSize/1024,(double)_getTotalUsedSpace()/1024/1024,(float)size/1024);

			//If out of memory
			if (curAlloc == NULL)
			{
	#ifdef DEBUG_CUDA
				size_t spaceDiff = _getTotalFreeSpace();
	#endif
				//Try to recover before throwing error
				for (int i = 0; i <= ALLOC_RETRY; i ++)
				{
					if (_syncReadyEvents() && _freeReadyAllocs())
					{
						curAlloc = _getFirstSuitedFree(size); //Is there space now?
						if (curAlloc != NULL)
							break; //Success
					}
					else
						usleep(10000); // 10 ms, Order of magnitude of largest kernels
				}
	#ifdef DEBUG_CUDA
				spaceDiff =  _getTotalFreeSpace() - spaceDiff;
				printf("DEBUG_INFO: Out of memory handled by waiting for unfinished tasks, which freed %lu B.\n", spaceDiff);
	#endif

				//Did we manage to recover?
				if (curAlloc == NULL)
				{
					printf("ERROR: CudaCustomAllocator out of memory\n [requestedSpace:             %lu B]\n [largestContinuousFreeSpace: %lu B]\n [totalFreeSpace:             %lu B]\n",
							(unsigned long) size, (unsigned long) _getLargestContinuousFreeSpace(), (unsigned long) _getTotalFreeSpace());

					_printState();

					fflush(stdout);
					CRITICAL(ERRCUDACAOOM);
				}
			}

			if (curAlloc->size == size)
			{
				curAlloc->free = false;
				newAlloc = curAlloc;
			}
			else //Or curAlloc->size is smaller than size
			{
				//Setup new pointer
				newAlloc = new Alloc();
				newAlloc->next = curAlloc;
				newAlloc->ptr = curAlloc->ptr;
				newAlloc->size = size;
				newAlloc->free = false;

				//Modify old pointer
				curAlloc->ptr = &(curAlloc->ptr[size]);
				curAlloc->size -= size;

				//Insert new allocation region into chain
				if(curAlloc->prev == NULL) //If the first allocation region
					first = newAlloc;
				else
					curAlloc->prev->next = newAlloc;
				newAlloc->prev = curAlloc->prev;
				newAlloc->next = curAlloc;
				curAlloc->prev = newAlloc;
			}
		}
		else
		{
			newAlloc = new Alloc();
			newAlloc->size = size;
			newAlloc->free = false;
			DEBUG_HANDLE_ERROR(cudaMalloc( (void**) &(newAlloc->ptr), size));

			//Just add to start by replacing first
			newAlloc->next = first;
			first->prev = newAlloc;
			first = newAlloc;
		}

#ifdef CUSTOM_ALLOCATOR_MEMGUARD
		newAlloc->backtraceSize = backtrace(newAlloc->backtrace, 20);
		newAlloc->guardPtr = newAlloc->ptr + requestedSize;
		cudaStream_t stream = 0;
		CudaShortcuts::memInit<BYTE>( newAlloc->guardPtr, GUARD_VALUE, size - requestedSize, stream); //TODO switch to specialized stream
		DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream));
#endif
CTOC("","deviceAlloc0");

		return newAlloc;
	};

	~CudaCustomAllocator()
	{
		{
			Lock ml(&mutex);
			_clear();
		}
		omp_destroy_lock(&mutex);
	}

	//Thread-safe wrapper functions

	void free(Alloc* a)
	{
		Lock ml(&mutex);
		_free(a);
	}

	void syncReadyEvents()
	{
		Lock ml(&mutex);
		_syncReadyEvents();
	}

	void freeReadyAllocs()
	{
		Lock ml(&mutex);
		_freeReadyAllocs();
	}

	size_t getTotalFreeSpace()
	{
		Lock ml(&mutex);
		size_t size = _getTotalFreeSpace();
		return size;
	}

	size_t getTotalUsedSpace()
	{
		Lock ml(&mutex);
		size_t size = _getTotalUsedSpace();
		return size;
	}

	size_t getNumberOfAllocs()
	{
		Lock ml(&mutex);
		size_t size = _getNumberOfAllocs();
		return size;
	}

	size_t getLargestContinuousFreeSpace()
	{
		Lock ml(&mutex);
		size_t size = _getLargestContinuousFreeSpace();
		return size;
	}

	void printState()
	{
		Lock ml(&mutex);
		_printState();
	}
};
//

#endif
