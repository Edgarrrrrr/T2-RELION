#undef ALTCPU
#include <sys/time.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <ctime>
#include <vector>
#include <map>
#include <queue>
#include <list>
#include <functional>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <random>

#include "src/ml_optimiser.h"
#include "src/acc/acc_ptr.h"
#include "src/acc/acc_projector.h"
#include "src/acc/acc_backprojector.h"
#include "src/acc/acc_projector_plan.h"
#include "src/acc/cuda/cuda_benchmark_utils.h"
#include "src/acc/cuda/cuda_kernels/helper.cuh"
#include "src/acc/cuda/cuda_kernels/diff2.cuh"
#include "src/acc/cuda/cuda_kernels/wavg.cuh"
#include "src/acc/cuda/cuda_mem_utils.h"
#include "src/acc/cuda/cuda_fft.h"
#include "src/acc/data_types.h"
#include "src/complex.h"
#include "src/helix.h"
#include "src/error.h"
#include <fstream>
#include "src/parallel.h"
#include <signal.h>

#ifdef CUDA_FORCESTL
#include "src/acc/cuda/cuda_utils_stl.cuh"
#else
#include "src/acc/cuda/cuda_utils_cub.cuh"
#endif

#include "src/acc/utilities.h"
#include "src/acc/utilities_impl.h"

#include "src/acc/acc_ml_optimiser.h"
#include "src/acc/cuda/cuda_ml_optimiser.h"
#include "src/acc/acc_helper_functions.h"
#include "src/acc/acc_ml_optimiser_impl.h"

// #define LOGGING
class Log {
public:
    enum LogLevel {
        INFO,
        WARNING,
        ERROR
    };

    static void log(LogLevel level, const std::string& message) {
#ifdef LOGGING
        std::string levelStr;
        std::string colorCode;

        // Get thread ID
        std::string threadId = std::to_string(omp_get_thread_num());

        // Set color and log level
        switch (level) {
            case INFO:
                levelStr = "INFO";
                colorCode = "\033[1;32m"; // Green color for INFO
                break;
            case WARNING:
                levelStr = "WARNING";
                colorCode = "\033[1;33m"; // Yellow color for WARNING
                break;
            case ERROR:
                levelStr = "ERROR";
                colorCode = "\033[1;31m"; // Red color for ERROR
                break;
            default:
                levelStr = "UNKNOWN";
                colorCode = "\033[1;37m"; // White color for UNKNOWN
                break;
        }

        // Reset color code
        std::string resetColor = "\033[0m";

        // Output log with color and thread ID
        std::cout << "[" << colorCode << levelStr << resetColor << "] [" << threadId << "] " << message << std::endl;
#endif
    }
};


enum NodeType {CPU, GPU, Memcpy, Sync, Start, Finish, Undefined};
const static int NodeCount = 39;
enum NodeName : int {
    StartNode = 0,
    AccDoExpectationOneParticlePre = 1,
	AccDoExpectationOneParticlePreMemAlloc = 2,
    ForIbodyInit = 3,
    ForIbodyCond = 4,
	AccDoExpectationOneParticlePostPerBodyGetFTAndCtfs = 5,
	AccDoExpectationOneParticlePostPerBodyGetFTAndCtfsMemAlloc = 6,
	GetAllSquaredDifferencesCoarsePre = 7,
	ForCoarseImgIdInit = 8,
	ForCoarseImgIdCond = 9,
	GetAllSquaredDifferencesCoarsePostPerImg = 10,
	GetAllSquaredDifferencesCoarsePostPerImgLaunchKernel = 11,
	GetAllSquaredDifferencesCoarsePostPerImgSync = 12,
	GetAllSquaredDifferencesCoarsePostPerImgGetMin = 13,
	ForCoarseImgIdUpdate = 14,
	AccDoExpectationOneParticlePostPerBodyCvtDToWCoarse = 15,
	AccDoExpectationOneParticlePostPerBodyCvtDToWCoarseMemAlloc = 16,
	GetAllSquaredDifferencesFinePre = 17,
	ForFineImgIdInit = 18,
	ForFineImgIdCond = 19,
	GetAllSquaredDifferencesFinePostPerImg = 20,
	GetAllSquaredDifferencesFinePostPerImgMemcpyHtoD = 21,
	GetAllSquaredDifferencesFinePostPerImgMemcpyHtoDSync = 22,	
	GetAllSquaredDifferencesFinePostPerImgLaunchKernel = 23,
	GetAllSquaredDifferencesFinePostPerImgSync = 24,
	GetAllSquaredDifferencesFinePostPerImgGetMin = 25,
	ForFineImgIdUpdate = 26,
	AccDoExpectationOneParticlePostPerBodyCvtDToWFine = 27,
	StoreWeightedSumsCollectData = 28,
	ForStoreWeightedSumsImgIdInit = 29,
	ForStoreWeightedSumsImgIdCond = 30,
	StoreWeightedSumsMaximizationPerImgSetup = 31,
	StoreWeightedSumsMaximizationPerImgLaunchKernel = 32,
	StoreWeightedSumsMaximizationPerImgSync = 33,
	ForStoreWeightedSumsUpdate = 34,
	StoreWeightedSumsStorePost = 35,
	AccDoExpectationOneParticleCleanup = 36,
    ForIbodyUpdate = 37,
    FinishNode = 38,
};
template <class Mlclass>
class Node {
public:
    NodeType type;
	NodeName name;
    std::function<void(Context<Mlclass> &)> func;

    Node(NodeName name, NodeType type, std::function<void(Context<Mlclass> &)> func) : name(name), type(type), func(func) {}
    Node(NodeName name, NodeType type) : name(name), type(type), func(nullptr) {}
	Node() : name(NodeName::FinishNode), type(NodeType::Undefined) {}
};


enum class EdgeType {Normal, Condition, Undefined};
template <class Mlclass>
class Edge {
public:
	Node<Mlclass> *src;
	Node<Mlclass> *dst;
	EdgeType type;
	std::function<bool(Context<Mlclass>&)> condition;

	Edge(Node<Mlclass> *src, Node<Mlclass> *dst, EdgeType type, std::function<bool(Context<Mlclass>&)> condition) : src(src), dst(dst), type(type), condition(condition) {}
	Edge(Node<Mlclass> *src, Node<Mlclass> *dst) : src(src), dst(dst), type(EdgeType::Normal) {}
	Edge() : src(nullptr), dst(nullptr), type(EdgeType::Undefined) {
		// Log::log(Log::ERROR, "Undefined edge created");
	}
};


template <class Mlclass>
class Graph {
public:
	Node<Mlclass> nodes[NodeCount];
	Edge<Mlclass> edges[NodeCount][NodeCount];

	bool hasPrintedGraph = false;

	void InitStaticGraph() {
	//  startNode
	//  accDoExpectationOneParticlePre
	//  accDoExpectationOneParticlePreMemAlloc
	//  for ibody {
	// 		accDoExpectationOneParticlePostPerBodyGetFTAndCtfs
	// 		getAllSquaredDifferencesCoarsePre
	// 		for imgid {
	// 			getAllSquaredDifferencesCoarsePostPerImg
	// 			getAllSquaredDifferencesCoarsePostPerImgLaunchKernel
	// 			getAllSquaredDifferencesCoarsePostPerImgSync
	// 			getAllSquaredDifferencesCoarsePostPerImgGetMin
	//  	}
	// 		accDoExpectationOneParticlePostPerBodyCvtDToWCoarse
	//  	getAllSquaredDifferencesFinePre
	//      for imgid {
	// 			getAllSquaredDifferencesFinePostPerImg
	// 			getAllSquaredDifferencesFinePostPerImgMemcpyHtoD
	// 			getAllSquaredDifferencesFinePostPerImgMemcpyHtoDSync
	// 			getAllSquaredDifferencesFinePostPerImgLaunchKernel
	// 			getAllSquaredDifferencesFinePostPerImgSync
	// 			getAllSquaredDifferencesFinePostPerImgGetMin
	//      }
	// 		accDoExpectationOneParticlePostPerBodyCvtDToWFine
	//      storeWeightedSumsCollectData
	//      if (storeWeightedSumsDoMaximization) {
	//          for imgid {
	//              storeWeightedSumsMaximizationPerImgSetup
	//              storeWeightedSumsMaximizationPerImgLaunchKernel
	//              storeWeightedSumsMaximizationPerImgSync
	//          }
	//          storeWeightedSumsStorePost
	//      }
	//      accDoExpectationOneParticleCleanup
	//  }
	//  finishNode
		nodes[(int)NodeName::StartNode] = Node<Mlclass>(NodeName::StartNode, NodeType::Start);
		nodes[(int)NodeName::AccDoExpectationOneParticlePre] = Node<Mlclass>(NodeName::AccDoExpectationOneParticlePre, NodeType::CPU, &accDoExpectationOneParticlePre<Mlclass>);
		nodes[(int)NodeName::AccDoExpectationOneParticlePreMemAlloc] = Node<Mlclass>(NodeName::AccDoExpectationOneParticlePreMemAlloc, NodeType::CPU, &accDoExpectationOneParticlePreMemAlloc<Mlclass>);
		nodes[(int)NodeName::ForIbodyInit] = Node<Mlclass>(NodeName::ForIbodyInit, NodeType::CPU, &forIbodyInit<Mlclass>);
		nodes[(int)NodeName::ForIbodyCond] = Node<Mlclass>(NodeName::ForIbodyCond, NodeType::CPU, &forIbodyCond<Mlclass>);
		nodes[(int)NodeName::AccDoExpectationOneParticlePostPerBodyGetFTAndCtfs] = Node<Mlclass>(NodeName::AccDoExpectationOneParticlePostPerBodyGetFTAndCtfs, NodeType::CPU, &accDoExpectationOneParticlePostPerBodyGetFTAndCtfs<Mlclass>);
		nodes[(int)NodeName::AccDoExpectationOneParticlePostPerBodyGetFTAndCtfsMemAlloc] = Node<Mlclass>(NodeName::AccDoExpectationOneParticlePostPerBodyGetFTAndCtfsMemAlloc, NodeType::CPU, &accDoExpectationOneParticlePostPerBodyGetFTAndCtfsMemAlloc<Mlclass>);
		nodes[(int)NodeName::GetAllSquaredDifferencesCoarsePre] = Node<Mlclass>(NodeName::GetAllSquaredDifferencesCoarsePre, NodeType::CPU, &getAllSquaredDifferencesCoarsePre<Mlclass>);
		nodes[(int)NodeName::ForCoarseImgIdInit] = Node<Mlclass>(NodeName::ForCoarseImgIdInit, NodeType::CPU, &forCoarseImgIdInit<Mlclass>);
		nodes[(int)NodeName::ForCoarseImgIdCond] = Node<Mlclass>(NodeName::ForCoarseImgIdCond, NodeType::CPU, &forCoarseImgIdCond<Mlclass>);
		nodes[(int)NodeName::GetAllSquaredDifferencesCoarsePostPerImg] = Node<Mlclass>(NodeName::GetAllSquaredDifferencesCoarsePostPerImg, NodeType::CPU, &getAllSquaredDifferencesCoarsePostPerImg<Mlclass>);
		nodes[(int)NodeName::GetAllSquaredDifferencesCoarsePostPerImgLaunchKernel] = Node<Mlclass>(NodeName::GetAllSquaredDifferencesCoarsePostPerImgLaunchKernel, NodeType::GPU, &getAllSquaredDifferencesCoarsePostPerImgLaunchKernel<Mlclass>);
		nodes[(int)NodeName::GetAllSquaredDifferencesCoarsePostPerImgSync] = Node<Mlclass>(NodeName::GetAllSquaredDifferencesCoarsePostPerImgSync, NodeType::Sync, &getAllSquaredDifferencesCoarsePostPerImgSync<Mlclass>);
		nodes[(int)NodeName::GetAllSquaredDifferencesCoarsePostPerImgGetMin] = Node<Mlclass>(NodeName::GetAllSquaredDifferencesCoarsePostPerImgGetMin, NodeType::CPU, &getAllSquaredDifferencesCoarsePostPerImgGetMin<Mlclass>);
		nodes[(int)NodeName::ForCoarseImgIdUpdate] = Node<Mlclass>(NodeName::ForCoarseImgIdUpdate, NodeType::CPU, &forCoarseImgIdUpdate<Mlclass>);
		nodes[(int)NodeName::AccDoExpectationOneParticlePostPerBodyCvtDToWCoarse] = Node<Mlclass>(NodeName::AccDoExpectationOneParticlePostPerBodyCvtDToWCoarse, NodeType::CPU, &accDoExpectationOneParticlePostPerBodyCvtDToWCoarse<Mlclass>);
		nodes[(int)NodeName::AccDoExpectationOneParticlePostPerBodyCvtDToWCoarseMemAlloc] = Node<Mlclass>(NodeName::AccDoExpectationOneParticlePostPerBodyCvtDToWCoarseMemAlloc, NodeType::CPU, &accDoExpectationOneParticlePostPerBodyCvtDToWCoarseMemAlloc<Mlclass>);
		nodes[(int)NodeName::GetAllSquaredDifferencesFinePre] = Node<Mlclass>(NodeName::GetAllSquaredDifferencesFinePre, NodeType::CPU, &getAllSquaredDifferencesFinePre<Mlclass>);
		nodes[(int)NodeName::ForFineImgIdInit] = Node<Mlclass>(NodeName::ForFineImgIdInit, NodeType::CPU, &forFineImgIdInit<Mlclass>);
		nodes[(int)NodeName::ForFineImgIdCond] = Node<Mlclass>(NodeName::ForFineImgIdCond, NodeType::CPU, &forFineImgIdCond<Mlclass>);
		nodes[(int)NodeName::GetAllSquaredDifferencesFinePostPerImg] = Node<Mlclass>(NodeName::GetAllSquaredDifferencesFinePostPerImg, NodeType::CPU, &getAllSquaredDifferencesFinePostPerImg<Mlclass>);
		nodes[(int)NodeName::GetAllSquaredDifferencesFinePostPerImgMemcpyHtoD] = Node<Mlclass>(NodeName::GetAllSquaredDifferencesFinePostPerImgMemcpyHtoD, NodeType::Memcpy, &getAllSquaredDifferencesFinePostPerImgMemcpyHtoD<Mlclass>);
		nodes[(int)NodeName::GetAllSquaredDifferencesFinePostPerImgMemcpyHtoDSync] = Node<Mlclass>(NodeName::GetAllSquaredDifferencesFinePostPerImgMemcpyHtoDSync, NodeType::Sync, &getAllSquaredDifferencesFinePostPerImgMemcpyHtoDSync<Mlclass>);
		nodes[(int)NodeName::GetAllSquaredDifferencesFinePostPerImgLaunchKernel] = Node<Mlclass>(NodeName::GetAllSquaredDifferencesFinePostPerImgLaunchKernel, NodeType::GPU, &getAllSquaredDifferencesFinePostPerImgLaunchKernel<Mlclass>);
		nodes[(int)NodeName::GetAllSquaredDifferencesFinePostPerImgSync] = Node<Mlclass>(NodeName::GetAllSquaredDifferencesFinePostPerImgSync, NodeType::Sync, &getAllSquaredDifferencesFinePostPerImgSync<Mlclass>);
		nodes[(int)NodeName::GetAllSquaredDifferencesFinePostPerImgGetMin] = Node<Mlclass>(NodeName::GetAllSquaredDifferencesFinePostPerImgGetMin, NodeType::CPU, &getAllSquaredDifferencesFinePostPerImgGetMin<Mlclass>);
		nodes[(int)NodeName::ForFineImgIdUpdate] = Node<Mlclass>(NodeName::ForFineImgIdUpdate, NodeType::CPU, &forFineImgIdUpdate<Mlclass>);
		nodes[(int)NodeName::AccDoExpectationOneParticlePostPerBodyCvtDToWFine] = Node<Mlclass>(NodeName::AccDoExpectationOneParticlePostPerBodyCvtDToWFine, NodeType::CPU, &accDoExpectationOneParticlePostPerBodyCvtDToWFine<Mlclass>);
		nodes[(int)NodeName::StoreWeightedSumsCollectData] = Node<Mlclass>(NodeName::StoreWeightedSumsCollectData, NodeType::CPU, &storeWeightedSumsCollectData<Mlclass>);
		nodes[(int)NodeName::ForStoreWeightedSumsImgIdInit] = Node<Mlclass>(NodeName::ForStoreWeightedSumsImgIdInit, NodeType::CPU, &forStoreWeightedSumsImgIdInit<Mlclass>);
		nodes[(int)NodeName::ForStoreWeightedSumsImgIdCond] = Node<Mlclass>(NodeName::ForStoreWeightedSumsImgIdCond, NodeType::CPU, &forStoreWeightedSumsImgIdCond<Mlclass>);
		nodes[(int)NodeName::StoreWeightedSumsMaximizationPerImgSetup] = Node<Mlclass>(NodeName::StoreWeightedSumsMaximizationPerImgSetup, NodeType::CPU, &storeWeightedSumsMaximizationPerImgSetup<Mlclass>);
		nodes[(int)NodeName::StoreWeightedSumsMaximizationPerImgLaunchKernel] = Node<Mlclass>(NodeName::StoreWeightedSumsMaximizationPerImgLaunchKernel, NodeType::GPU, &storeWeightedSumsMaximizationPerImgLaunchKernel<Mlclass>);
		nodes[(int)NodeName::StoreWeightedSumsMaximizationPerImgSync] = Node<Mlclass>(NodeName::StoreWeightedSumsMaximizationPerImgSync, NodeType::Sync, &storeWeightedSumsMaximizationPerImgSync<Mlclass>);
		nodes[(int)NodeName::ForStoreWeightedSumsUpdate] = Node<Mlclass>(NodeName::ForStoreWeightedSumsUpdate, NodeType::CPU, &forStoreWeightedSumsUpdate<Mlclass>);
		nodes[(int)NodeName::StoreWeightedSumsStorePost] = Node<Mlclass>(NodeName::StoreWeightedSumsStorePost, NodeType::CPU, &storeWeightedSumsStorePost<Mlclass>);
		nodes[(int)NodeName::AccDoExpectationOneParticleCleanup] = Node<Mlclass>(NodeName::AccDoExpectationOneParticleCleanup, NodeType::CPU, &accDoExpectationOneParticleCleanup<Mlclass>);
		nodes[(int)NodeName::ForIbodyUpdate] = Node<Mlclass>(NodeName::ForIbodyUpdate, NodeType::CPU, &forIbodyUpdate<Mlclass>);
		nodes[(int)NodeName::FinishNode] = Node<Mlclass>(NodeName::FinishNode, NodeType::Finish);

		edges[(int)NodeName::StartNode][(int)NodeName::AccDoExpectationOneParticlePre] = Edge<Mlclass>(&nodes[(int)NodeName::StartNode], &nodes[(int)NodeName::AccDoExpectationOneParticlePre]);
		edges[(int)NodeName::AccDoExpectationOneParticlePre][(int)NodeName::AccDoExpectationOneParticlePreMemAlloc] = Edge<Mlclass>(&nodes[(int)NodeName::AccDoExpectationOneParticlePre], &nodes[(int)NodeName::AccDoExpectationOneParticlePreMemAlloc]);
		edges[(int)NodeName::AccDoExpectationOneParticlePreMemAlloc][(int)NodeName::ForIbodyInit] = Edge<Mlclass>(&nodes[(int)NodeName::AccDoExpectationOneParticlePreMemAlloc], &nodes[(int)NodeName::ForIbodyInit]);
		edges[(int)NodeName::ForIbodyInit][(int)NodeName::ForIbodyCond] = Edge<Mlclass>(&nodes[(int)NodeName::ForIbodyInit], &nodes[(int)NodeName::ForIbodyCond]);
		edges[(int)NodeName::ForIbodyCond][(int)NodeName::AccDoExpectationOneParticlePostPerBodyGetFTAndCtfs] = Edge<Mlclass>(&nodes[(int)NodeName::ForIbodyCond], &nodes[(int)NodeName::AccDoExpectationOneParticlePostPerBodyGetFTAndCtfs], EdgeType::Condition, &forIbodyCond<Mlclass>);
		edges[(int)NodeName::AccDoExpectationOneParticlePostPerBodyGetFTAndCtfs][(int)NodeName::AccDoExpectationOneParticlePostPerBodyGetFTAndCtfsMemAlloc] = Edge<Mlclass>(&nodes[(int)NodeName::AccDoExpectationOneParticlePostPerBodyGetFTAndCtfs], &nodes[(int)NodeName::AccDoExpectationOneParticlePostPerBodyGetFTAndCtfsMemAlloc]);
		edges[(int)NodeName::AccDoExpectationOneParticlePostPerBodyGetFTAndCtfsMemAlloc][(int)NodeName::GetAllSquaredDifferencesCoarsePre] = Edge<Mlclass>(&nodes[(int)NodeName::AccDoExpectationOneParticlePostPerBodyGetFTAndCtfsMemAlloc], &nodes[(int)NodeName::GetAllSquaredDifferencesCoarsePre]);
		edges[(int)NodeName::GetAllSquaredDifferencesCoarsePre][(int)NodeName::ForCoarseImgIdInit] = Edge<Mlclass>(&nodes[(int)NodeName::GetAllSquaredDifferencesCoarsePre], &nodes[(int)NodeName::ForCoarseImgIdInit]);
		edges[(int)NodeName::ForCoarseImgIdInit][(int)NodeName::ForCoarseImgIdCond] = Edge<Mlclass>(&nodes[(int)NodeName::ForCoarseImgIdInit], &nodes[(int)NodeName::ForCoarseImgIdCond]);
		edges[(int)NodeName::ForCoarseImgIdCond][(int)NodeName::GetAllSquaredDifferencesCoarsePostPerImg] = Edge<Mlclass>(&nodes[(int)NodeName::ForCoarseImgIdCond], &nodes[(int)NodeName::GetAllSquaredDifferencesCoarsePostPerImg], EdgeType::Condition, &forCoarseImgIdCond<Mlclass>);
		edges[(int)NodeName::GetAllSquaredDifferencesCoarsePostPerImg][(int)NodeName::GetAllSquaredDifferencesCoarsePostPerImgLaunchKernel] = Edge<Mlclass>(&nodes[(int)NodeName::GetAllSquaredDifferencesCoarsePostPerImg], &nodes[(int)NodeName::GetAllSquaredDifferencesCoarsePostPerImgLaunchKernel]);
		edges[(int)NodeName::GetAllSquaredDifferencesCoarsePostPerImgLaunchKernel][(int)NodeName::GetAllSquaredDifferencesCoarsePostPerImgSync] = Edge<Mlclass>(&nodes[(int)NodeName::GetAllSquaredDifferencesCoarsePostPerImgLaunchKernel], &nodes[(int)NodeName::GetAllSquaredDifferencesCoarsePostPerImgSync]);
		edges[(int)NodeName::GetAllSquaredDifferencesCoarsePostPerImgSync][(int)NodeName::GetAllSquaredDifferencesCoarsePostPerImgGetMin] = Edge<Mlclass>(&nodes[(int)NodeName::GetAllSquaredDifferencesCoarsePostPerImgSync], &nodes[(int)NodeName::GetAllSquaredDifferencesCoarsePostPerImgGetMin]);
		edges[(int)NodeName::GetAllSquaredDifferencesCoarsePostPerImgGetMin][(int)NodeName::ForCoarseImgIdUpdate] = Edge<Mlclass>(&nodes[(int)NodeName::GetAllSquaredDifferencesCoarsePostPerImgGetMin], &nodes[(int)NodeName::ForCoarseImgIdUpdate]);
		edges[(int)NodeName::ForCoarseImgIdUpdate][(int)NodeName::ForCoarseImgIdCond] = Edge<Mlclass>(&nodes[(int)NodeName::ForCoarseImgIdUpdate], &nodes[(int)NodeName::ForCoarseImgIdCond]);
		edges[(int)NodeName::ForCoarseImgIdCond][(int)NodeName::AccDoExpectationOneParticlePostPerBodyCvtDToWCoarse] = Edge<Mlclass>(&nodes[(int)NodeName::ForCoarseImgIdCond], &nodes[(int)NodeName::AccDoExpectationOneParticlePostPerBodyCvtDToWCoarse], EdgeType::Condition, &forCoarseImgIdCondNot<Mlclass>);
		edges[(int)NodeName::AccDoExpectationOneParticlePostPerBodyCvtDToWCoarse][(int)NodeName::AccDoExpectationOneParticlePostPerBodyCvtDToWCoarseMemAlloc] = Edge<Mlclass>(&nodes[(int)NodeName::AccDoExpectationOneParticlePostPerBodyCvtDToWCoarse], &nodes[(int)NodeName::AccDoExpectationOneParticlePostPerBodyCvtDToWCoarseMemAlloc]);
		edges[(int)NodeName::AccDoExpectationOneParticlePostPerBodyCvtDToWCoarseMemAlloc][(int)NodeName::GetAllSquaredDifferencesFinePre] = Edge<Mlclass>(&nodes[(int)NodeName::AccDoExpectationOneParticlePostPerBodyCvtDToWCoarseMemAlloc], &nodes[(int)NodeName::GetAllSquaredDifferencesFinePre]);
		edges[(int)NodeName::GetAllSquaredDifferencesFinePre][(int)NodeName::ForFineImgIdInit] = Edge<Mlclass>(&nodes[(int)NodeName::GetAllSquaredDifferencesFinePre], &nodes[(int)NodeName::ForFineImgIdInit]);
		edges[(int)NodeName::ForFineImgIdInit][(int)NodeName::ForFineImgIdCond] = Edge<Mlclass>(&nodes[(int)NodeName::ForFineImgIdInit], &nodes[(int)NodeName::ForFineImgIdCond]);
		edges[(int)NodeName::ForFineImgIdCond][(int)NodeName::GetAllSquaredDifferencesFinePostPerImg] = Edge<Mlclass>(&nodes[(int)NodeName::ForFineImgIdCond], &nodes[(int)NodeName::GetAllSquaredDifferencesFinePostPerImg], EdgeType::Condition, &forFineImgIdCond<Mlclass>);
		edges[(int)NodeName::GetAllSquaredDifferencesFinePostPerImg][(int)NodeName::GetAllSquaredDifferencesFinePostPerImgMemcpyHtoD] = Edge<Mlclass>(&nodes[(int)NodeName::GetAllSquaredDifferencesFinePostPerImg], &nodes[(int)NodeName::GetAllSquaredDifferencesFinePostPerImgMemcpyHtoD]);
		edges[(int)NodeName::GetAllSquaredDifferencesFinePostPerImgMemcpyHtoD][(int)NodeName::GetAllSquaredDifferencesFinePostPerImgMemcpyHtoDSync] = Edge<Mlclass>(&nodes[(int)NodeName::GetAllSquaredDifferencesFinePostPerImgMemcpyHtoD], &nodes[(int)NodeName::GetAllSquaredDifferencesFinePostPerImgMemcpyHtoDSync]);
		edges[(int)NodeName::GetAllSquaredDifferencesFinePostPerImgMemcpyHtoDSync][(int)NodeName::GetAllSquaredDifferencesFinePostPerImgLaunchKernel] = Edge<Mlclass>(&nodes[(int)NodeName::GetAllSquaredDifferencesFinePostPerImgMemcpyHtoDSync], &nodes[(int)NodeName::GetAllSquaredDifferencesFinePostPerImgLaunchKernel]);
		edges[(int)NodeName::GetAllSquaredDifferencesFinePostPerImgLaunchKernel][(int)NodeName::GetAllSquaredDifferencesFinePostPerImgSync] = Edge<Mlclass>(&nodes[(int)NodeName::GetAllSquaredDifferencesFinePostPerImgLaunchKernel], &nodes[(int)NodeName::GetAllSquaredDifferencesFinePostPerImgSync]);
		edges[(int)NodeName::GetAllSquaredDifferencesFinePostPerImgSync][(int)NodeName::GetAllSquaredDifferencesFinePostPerImgGetMin] = Edge<Mlclass>(&nodes[(int)NodeName::GetAllSquaredDifferencesFinePostPerImgSync], &nodes[(int)NodeName::GetAllSquaredDifferencesFinePostPerImgGetMin]);
		edges[(int)NodeName::GetAllSquaredDifferencesFinePostPerImgGetMin][(int)NodeName::ForFineImgIdUpdate] = Edge<Mlclass>(&nodes[(int)NodeName::GetAllSquaredDifferencesFinePostPerImgGetMin], &nodes[(int)NodeName::ForFineImgIdUpdate]);
		edges[(int)NodeName::ForFineImgIdUpdate][(int)NodeName::ForFineImgIdCond] = Edge<Mlclass>(&nodes[(int)NodeName::ForFineImgIdUpdate], &nodes[(int)NodeName::ForFineImgIdCond]);
		edges[(int)NodeName::ForFineImgIdCond][(int)NodeName::AccDoExpectationOneParticlePostPerBodyCvtDToWFine] = Edge<Mlclass>(&nodes[(int)NodeName::ForFineImgIdCond], &nodes[(int)NodeName::AccDoExpectationOneParticlePostPerBodyCvtDToWFine], EdgeType::Condition, &forFineImgIdCondNot<Mlclass>);
		edges[(int)NodeName::AccDoExpectationOneParticlePostPerBodyCvtDToWFine][(int)NodeName::StoreWeightedSumsCollectData] = Edge<Mlclass>(&nodes[(int)NodeName::AccDoExpectationOneParticlePostPerBodyCvtDToWFine], &nodes[(int)NodeName::StoreWeightedSumsCollectData]);
		edges[(int)NodeName::StoreWeightedSumsCollectData][(int)NodeName::ForStoreWeightedSumsImgIdInit] = Edge<Mlclass>(&nodes[(int)NodeName::StoreWeightedSumsCollectData], &nodes[(int)NodeName::ForStoreWeightedSumsImgIdInit], EdgeType::Condition, &ifStoreWeightedSumsDoMaximizationCond<Mlclass>);
		edges[(int)NodeName::ForStoreWeightedSumsImgIdInit][(int)NodeName::ForStoreWeightedSumsImgIdCond] = Edge<Mlclass>(&nodes[(int)NodeName::ForStoreWeightedSumsImgIdInit], &nodes[(int)NodeName::ForStoreWeightedSumsImgIdCond]);
		edges[(int)NodeName::ForStoreWeightedSumsImgIdCond][(int)NodeName::StoreWeightedSumsMaximizationPerImgSetup] = Edge<Mlclass>(&nodes[(int)NodeName::ForStoreWeightedSumsImgIdCond], &nodes[(int)NodeName::StoreWeightedSumsMaximizationPerImgSetup], EdgeType::Condition, &forStoreWeightedSumsImgIdCond<Mlclass>);
		edges[(int)NodeName::StoreWeightedSumsMaximizationPerImgSetup][(int)NodeName::StoreWeightedSumsMaximizationPerImgLaunchKernel] = Edge<Mlclass>(&nodes[(int)NodeName::StoreWeightedSumsMaximizationPerImgSetup], &nodes[(int)NodeName::StoreWeightedSumsMaximizationPerImgLaunchKernel]);
		edges[(int)NodeName::StoreWeightedSumsMaximizationPerImgLaunchKernel][(int)NodeName::StoreWeightedSumsMaximizationPerImgSync] = Edge<Mlclass>(&nodes[(int)NodeName::StoreWeightedSumsMaximizationPerImgLaunchKernel], &nodes[(int)NodeName::StoreWeightedSumsMaximizationPerImgSync]);
		edges[(int)NodeName::StoreWeightedSumsMaximizationPerImgSync][(int)NodeName::ForStoreWeightedSumsUpdate] = Edge<Mlclass>(&nodes[(int)NodeName::StoreWeightedSumsMaximizationPerImgSync], &nodes[(int)NodeName::ForStoreWeightedSumsUpdate]);
		edges[(int)NodeName::ForStoreWeightedSumsUpdate][(int)NodeName::ForStoreWeightedSumsImgIdCond] = Edge<Mlclass>(&nodes[(int)NodeName::ForStoreWeightedSumsUpdate], &nodes[(int)NodeName::ForStoreWeightedSumsImgIdCond]);
		edges[(int)NodeName::ForStoreWeightedSumsImgIdCond][(int)NodeName::StoreWeightedSumsStorePost] = Edge<Mlclass>(&nodes[(int)NodeName::ForStoreWeightedSumsImgIdCond], &nodes[(int)NodeName::StoreWeightedSumsStorePost], EdgeType::Condition, &forStoreWeightedSumsImgIdCondNot<Mlclass>);
		edges[(int)NodeName::StoreWeightedSumsStorePost][(int)NodeName::AccDoExpectationOneParticleCleanup] = Edge<Mlclass>(&nodes[(int)NodeName::StoreWeightedSumsStorePost], &nodes[(int)NodeName::AccDoExpectationOneParticleCleanup]);
		edges[(int)NodeName::AccDoExpectationOneParticleCleanup][(int)NodeName::ForIbodyUpdate] = Edge<Mlclass>(&nodes[(int)NodeName::AccDoExpectationOneParticleCleanup], &nodes[(int)NodeName::ForIbodyUpdate]);
		edges[(int)NodeName::StoreWeightedSumsCollectData][(int)NodeName::AccDoExpectationOneParticleCleanup] = Edge<Mlclass>(&nodes[(int)NodeName::StoreWeightedSumsCollectData], &nodes[(int)NodeName::AccDoExpectationOneParticleCleanup], EdgeType::Condition, &ifStoreWeightedSumsDoMaximizationCondNot<Mlclass>);
		edges[(int)NodeName::ForIbodyUpdate][(int)NodeName::ForIbodyCond] = Edge<Mlclass>(&nodes[(int)NodeName::ForIbodyUpdate], &nodes[(int)NodeName::ForIbodyCond]);
		edges[(int)NodeName::ForIbodyCond][(int)NodeName::FinishNode] = Edge<Mlclass>(&nodes[(int)NodeName::ForIbodyCond], &nodes[(int)NodeName::FinishNode], EdgeType::Condition, &forIbodyCondNot<Mlclass>);
	}

	Graph()
	{
		InitStaticGraph();
	}

	static Graph& getInstance() {
        static Graph instance;
        return instance;
    }
	
    void printGraph() {
		if (hasPrintedGraph) {
			return;
		}
		hasPrintedGraph = true;
		Log::log(Log::INFO, "================ Printing graph ================");
		Log::log(Log::INFO, "Nodes:");
        for (int i = 0; i < NodeCount; ++i) {
            if (nodes[i].type != NodeType::Undefined) {
				Log::log(Log::INFO, "Node Name: " + getNodeName(nodes[i].name) + ", Node Type: " + std::to_string(nodes[i].type));
            }
        }
		Log::log(Log::INFO, "Edges:");
        for (int i = 0; i < NodeCount; ++i) {
            for (int j = 0; j < NodeCount; ++j) {
                if (edges[i][j].type != EdgeType::Undefined) {
					Log::log(Log::INFO, "Edge from " + getNodeName(static_cast<NodeName>(i)) + " to " + getNodeName(static_cast<NodeName>(j)));
					if (edges[i][j].type == EdgeType::Condition) {
						Log::log(Log::INFO, "Condition edge");
					}

                }
            }
        }
		printGraphToDotFile();
		Log::log(Log::INFO, "================ End printing graph ================\n");
    }

	void printGraphToDotFile() {
		// Open a DOT file for writing
		std::ofstream dotFile("graph.dot");
		if (!dotFile.is_open()) {
			Log::log(Log::ERROR, "Failed to open DOT file for writing");
			return;
		}
		// Write the DOT file header
		dotFile << "digraph G {" << std::endl;
		// Print nodes
		for (int i = 0; i < NodeCount; ++i) {
			if (nodes[i].type != NodeType::Undefined) {
				dotFile << "    " << getNodeName(nodes[i].name)
						<< " [label=\"" << getNodeName(nodes[i].name)
						<< "\\nType: " << getNodeTypeStr(nodes[i].type) << "\"];" << std::endl;
			}
		}
		// Print edges
		for (int i = 0; i < NodeCount; ++i) {
			for (int j = 0; j < NodeCount; ++j) {
				if (edges[i][j].type != EdgeType::Undefined) {
					dotFile << "    " << getNodeName(nodes[i].name)
							<< " -> " << getNodeName(nodes[j].name);
					if (edges[i][j].type == EdgeType::Condition) {
						dotFile << " [label=\"Condition\"]";
					}
					dotFile << ";" << std::endl;
				}
			}
		}
		// Write the DOT file footer
		dotFile << "}" << std::endl;
		dotFile.close();
		// Convert the DOT file to a PNG image
		std::system("dot -Tpng graph.dot -o graph.png");
	}

	Node<Mlclass>& getStartNode() {
		return nodes[NodeName::StartNode];
	}

	static std::string getNodeTypeStr(NodeType nodeType) {
		switch (nodeType) {
			case CPU:    return "CPU";
			case GPU:    return "GPU";
			case Memcpy: return "Memcpy";
			case Sync:   return "Sync";
			case Start:  return "Start";
			case Finish: return "Finish";
			default:     return "Unknown";
		}
	}

	static std::string getNodeName(NodeName nodeName) {
        switch (nodeName) {
            case StartNode:
                return "StartNode";
            case AccDoExpectationOneParticlePre:
                return "AccDoExpectationOneParticlePre";
			case AccDoExpectationOneParticlePreMemAlloc:
				return "AccDoExpectationOneParticlePreMemAlloc";
            case ForIbodyInit:
                return "ForIbodyInit";
			case ForCoarseImgIdInit:
				return "ForCoarseImgIdInit";
			case ForCoarseImgIdCond:
				return "ForCoarseImgIdCond";
            case ForIbodyCond:
                return "ForIbodyCond";
			case AccDoExpectationOneParticlePostPerBodyGetFTAndCtfs:
				return "AccDoExpectationOneParticlePostPerBodyGetFTAndCtfs";
			case AccDoExpectationOneParticlePostPerBodyGetFTAndCtfsMemAlloc:
				return "AccDoExpectationOneParticlePostPerBodyGetFTAndCtfsMemAlloc";
			case GetAllSquaredDifferencesCoarsePre:
				return "GetAllSquaredDifferencesCoarsePre";
			case GetAllSquaredDifferencesCoarsePostPerImg:
				return "GetAllSquaredDifferencesCoarsePostPerImg";
			case GetAllSquaredDifferencesCoarsePostPerImgLaunchKernel:
				return "GetAllSquaredDifferencesCoarsePostPerImgLaunchKernel";
			case GetAllSquaredDifferencesCoarsePostPerImgSync:
				return "GetAllSquaredDifferencesCoarsePostPerImgSync";
			case GetAllSquaredDifferencesCoarsePostPerImgGetMin:
				return "GetAllSquaredDifferencesCoarsePostPerImgGetMin";
            case ForIbodyUpdate:
                return "ForIbodyUpdate";			
			case AccDoExpectationOneParticlePostPerBodyCvtDToWCoarse:
				return "AccDoExpectationOneParticlePostPerBodyCvtDToWCoarse";
			case AccDoExpectationOneParticlePostPerBodyCvtDToWCoarseMemAlloc:
				return "AccDoExpectationOneParticlePostPerBodyCvtDToWCoarseMemAlloc";
			case GetAllSquaredDifferencesFinePre:
				return "GetAllSquaredDifferencesFinePre";
			case ForFineImgIdInit:
				return "ForFineImgIdInit";
			case ForFineImgIdCond:
				return "ForFineImgIdCond";
			case GetAllSquaredDifferencesFinePostPerImg:
				return "GetAllSquaredDifferencesFinePostPerImg";
			case GetAllSquaredDifferencesFinePostPerImgMemcpyHtoD:
				return "GetAllSquaredDifferencesFinePostPerImgMemcpyHtoD";
			case GetAllSquaredDifferencesFinePostPerImgMemcpyHtoDSync:
				return "GetAllSquaredDifferencesFinePostPerImgMemcpyHtoDSync";
			case GetAllSquaredDifferencesFinePostPerImgLaunchKernel:
				return "GetAllSquaredDifferencesFinePostPerImgLaunchKernel";
			case GetAllSquaredDifferencesFinePostPerImgSync:
				return "GetAllSquaredDifferencesFinePostPerImgSync";
			case GetAllSquaredDifferencesFinePostPerImgGetMin:
				return "GetAllSquaredDifferencesFinePostPerImgGetMin";
			case ForFineImgIdUpdate:
				return "ForFineImgIdUpdate";
			case AccDoExpectationOneParticlePostPerBodyCvtDToWFine:
				return "AccDoExpectationOneParticlePostPerBodyCvtDToWFine";
			case StoreWeightedSumsCollectData:
				return "StoreWeightedSumsCollectData";
			case ForStoreWeightedSumsImgIdInit:
				return "ForStoreWeightedSumsImgIdInit";
			case ForStoreWeightedSumsImgIdCond:
				return "ForStoreWeightedSumsImgIdCond";
			case StoreWeightedSumsMaximizationPerImgSetup:
				return "StoreWeightedSumsMaximizationPerImgSetup";
			case StoreWeightedSumsMaximizationPerImgLaunchKernel:
				return "StoreWeightedSumsMaximizationPerImgLaunchKernel";
			case StoreWeightedSumsMaximizationPerImgSync:
				return "StoreWeightedSumsMaximizationPerImgSync";
			case ForStoreWeightedSumsUpdate:
				return "ForStoreWeightedSumsUpdate";
			case StoreWeightedSumsStorePost:
				return "StoreWeightedSumsStorePost";
			case AccDoExpectationOneParticleCleanup:
				return "AccDoExpectationOneParticleCleanup";
			case ForCoarseImgIdUpdate:
				return "ForCoarseImgIdUpdate";
            case FinishNode:
                return "FinishNode";
            default:
                return "UnknownNode";
        }
    }

	std::vector<Node<Mlclass>*> getSuccessors(Node<Mlclass>& node, Context<Mlclass>& context) {
		std::vector<Node<Mlclass>*> successors;
		for (int i = 0; i < NodeCount; i++) {
			if (edges[node.name][i].type == EdgeType::Normal) {
				successors.push_back(edges[node.name][i].dst);
			}
			else if (edges[node.name][i].type == EdgeType::Condition) {
				if (edges[node.name][i].condition(context)) {
					successors.push_back(edges[node.name][i].dst);
				}
			}
		}
		Log::log(Log::WARNING, "Successors of node: " + getNodeName(node.name));
		for (Node<Mlclass>* successor : successors) {
			Log::log(Log::WARNING, "Successor: " + getNodeName(successor->name));
		}
		return successors;
	}

private:
	Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
};


template <class Mlclass>
class Task {
public:
	Context<Mlclass>* context;

	int part_id;
	int thread_id;
	std::vector<cudaStream_t> classStreams;
	
	std::vector<std::chrono::nanoseconds> executionTimes;
	cudaStream_t cudaStreamPerTask;
	CudaCustomAllocator* allocator;
	CudaAllocatorThread *allocator_thread;

	Task(Mlclass *myInstance, CudaAllocatorThread *allocator_thread, 
		CudaCustomAllocator* allocator, int thread_id, int part_id) : 
		thread_id(thread_id),
		part_id(part_id), 
		allocator_thread(allocator_thread),
		allocator(allocator),
		// context(cudaStreamPerTask, classStreams),
		executionTimes(NodeCount)
	{
		// Log::log(Log::INFO, "Creating task. Task id: " + std::to_string(part_id));

		// cudaStreamCreate(&cudaStreamPerTask);
		// Log::log(Log::INFO, "cudaStreamPerTask address: 0x" + std::to_string(reinterpret_cast<uintptr_t>(&cudaStreamPerTask)));
		// int nr_classes = myInstance->baseMLO->mymodel.nr_classes;
		// for (int i = 0; i < nr_classes; i++) {
		// 	cudaStream_t stream;
		// 	cudaStreamCreate(&stream);
		// 	classStreams.push_back(stream);
		// }

		// context = new Context<Mlclass>();
		cudaStreamPerTask = myInstance->acquireStream();
		Log::log(Log::INFO, "cudaStreamPerTask address: 0x" + std::to_string(reinterpret_cast<uintptr_t>(&cudaStreamPerTask)));
		int nr_classes = myInstance->baseMLO->mymodel.nr_classes;
		for (int i = 0; i < nr_classes; i++) {
			classStreams.push_back(myInstance->acquireStream());
		}

		context = new Context<Mlclass>();



		context->baseMLO = myInstance->baseMLO;
		context->accMLO = myInstance;
#ifdef NEWMEM
		context->ptrFactory = AccPtrFactoryNew(cudaStreamPerTask,allocator_thread,allocator);
#else
		context->ptrFactory = AccPtrFactory(allocator, cudaStreamPerTask);
#endif
		context->thread_id = thread_id;
		context->part_id_sorted = part_id; 

		context->cudaStreamPerTask = cudaStreamPerTask;
		context->classStreams = classStreams;
	}

	~Task() {
		// Log::log(Log::WARNING, "Destroying task. Task id: " + std::to_string(part_id));
		// for (cudaStream_t& stream : classStreams) {
		// 	cudaStreamDestroy(stream);
		// }
		// Log::log(Log::INFO, "cudaStreamPerTask address: 0x" + std::to_string(reinterpret_cast<uintptr_t>(&cudaStreamPerTask)));
		// cudaStreamDestroy(cudaStreamPerTask);

		context->accMLO->releaseStream(cudaStreamPerTask);
		for (cudaStream_t& stream : classStreams) {
			context->accMLO->releaseStream(stream);
		}

		delete context;
		Log::log(Log::WARNING, "Task destroyed. Task id: " + std::to_string(part_id));
	}

    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;
    Task(Task&&) = delete;
    Task& operator=(Task&&) = delete;

	void execNode(Node<Mlclass>& node) {
		auto start = std::chrono::high_resolution_clock::now();
		
		if (node.type != NodeType::Start && node.type != NodeType::Finish) {
			node.func(*context);
		}

		auto end = std::chrono::high_resolution_clock::now();
		executionTimes[(int)node.name] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	}

	void printExecutionTimes() {
        Log::log(Log::INFO, "==================== Execution Times ====================");
		Log::log(Log::INFO, "Task id: " + std::to_string(part_id));
		for (int i = 0; i < NodeCount; i++) {
			std::string nodeName = Graph<Mlclass>::getNodeName(static_cast<NodeName>(i));
			std::string paddedNodeName = nodeName + std::string(100 - nodeName.size(), ' ');
			Log::log(Log::INFO, paddedNodeName + ": " + std::to_string(executionTimes[i].count() * 1e-6) + " ms");
		}
		Log::log(Log::INFO, "=========================================================");
    }
};

void preciseSleep(int milliseconds) {
    auto start = std::chrono::high_resolution_clock::now();
    auto end = start + std::chrono::milliseconds(milliseconds);
    while (std::chrono::high_resolution_clock::now() < end) {
        // 可以选择让线程稍微休眠一下以减少CPU占用
        // std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

#if 0
template <class Mlclass>
class Scheduler {
public:
	Mlclass *myInstance;
	CudaCustomAllocator* allocator;
	CudaAllocatorProcess *allocator_p;
	int thread_id;

	std::list<Task<Mlclass>> tasks;
	// max number of tasks that can be scheduled at once
	int nr_task_limit;

	// execution queue
	std::queue<std::pair<Task<Mlclass>&, Node<Mlclass>&>> queue;

	size_t ipart_first, ipart_last;

	std::list<std::pair<size_t, size_t>> ipart_list;

	Scheduler(Mlclass *myInstance,CudaAllocatorProcess *allocator_p, 
			CudaCustomAllocator* allocator, int thread_id,int task_num=1) : 
		myInstance(myInstance), allocator_p(allocator_p),allocator(allocator), thread_id(thread_id),nr_task_limit(task_num)
	{
		nr_task_limit = myInstance->nr_task_limit;

		ipart_first = 0;
		ipart_last = 0;
		// printf("xjldebug:Scheduler%d\n",nr_task_limit);
	}
	
	void run() {
		Graph<Mlclass>::getInstance().printGraph();
		// preciseSleep(thread_id * 10);

		while (tasks.size() < nr_task_limit) {
			bool hasNewTask = fetchNewTaskAndAddStartNodeToQueue();
			if (!hasNewTask) {
				Log::log(Log::INFO, "No new task");
				break;
			}
		}
		while (!queue.empty()) {
			/*  1. Get task and node from queue
			*	2. Execute node
			*   3. Push successors in queue
			*	4. Delete task if the node is a finish node
			*   5. Fetch new task and add start node to queue if no task is left
			*/
			std::pair<Task<Mlclass>&, Node<Mlclass>&> task_node_pair = queue.front();
			printQueue();
			queue.pop();
			Task<Mlclass>& task = task_node_pair.first;
			Node<Mlclass>& node = task_node_pair.second;

			execTaskNode(task, node);
			pushSuccessorsInqueue(task, node);
			
			if (node.type == NodeType::Finish) {
				task.printExecutionTimes();
				Log::log(Log::INFO, "Task finished Task id: " + std::to_string(task.part_id));
				// tasks.erase(std::remove(tasks.begin(), tasks.end(), task), tasks.end());
				// TODO
				auto it = std::find_if(tasks.begin(), tasks.end(),
					[&task](const Task<Mlclass>& t) { return t.part_id == task.part_id; });

				if (it != tasks.end()) {
					Log::log(Log::INFO, "Delete task. Task id: " + std::to_string(task.part_id));
					Log::log(Log::INFO, "before erase Task id: " + std::to_string(task.part_id));
					tasks.erase(it);
					Log::log(Log::INFO, "after erase Task id: " + std::to_string(task.part_id));
				}

				Log::log(Log::INFO, "Check if there is new task");
				bool hasNewTask = fetchNewTaskAndAddStartNodeToQueue();
				if (!hasNewTask) {
					Log::log(Log::INFO, "No new task");
				}
			}
		}
	}

	void printQueue() {
		Log::log(Log::INFO, "================ Printing queue ================");
		std::queue<std::pair<Task<Mlclass>&, Node<Mlclass>&>> queue_copy = queue;
		while (!queue_copy.empty()) {
			std::pair<Task<Mlclass>&, Node<Mlclass>&> task_node_pair = queue_copy.front();
			queue_copy.pop();
			Task<Mlclass>& task = task_node_pair.first;
			Node<Mlclass>& node = task_node_pair.second;
			Log::log(Log::INFO, "Task id: " + std::to_string(task.part_id) + ", Node name: " + Graph<Mlclass>::getNodeName(node.name));
		}
		Log::log(Log::INFO, "================ End printing queue ================\n");
	}

	bool addNewIpartToIpartList() {
		bool hasNewIpart = myInstance->baseMLO->exp_ipart_ThreadTaskDistributor->getTasks(ipart_first, ipart_last);
		// bool hasNewIpart = getTasks(ipart_first, ipart_last);
		if (hasNewIpart) {
			ipart_list.push_back(std::make_pair(ipart_first, ipart_last));
			Log::log(Log::INFO, "ipart_list push back. ipart_first: " + std::to_string(ipart_first) + ", ipart_last: " + std::to_string(ipart_last));
		}
		return hasNewIpart;
	}

	bool fetchNewTaskAndAddStartNodeToQueue() {
		Log::log(Log::INFO, "fetchNewTaskAndAddStartNodeToQueue : Fetching new task");
		if (ipart_list.empty()) {
			bool hasNewIpart = addNewIpartToIpartList();
			if (!hasNewIpart) {
				Log::log(Log::INFO, "fetchNewTaskAndAddStartNodeToQueue : No new ipart");
				return false;
			}
		}
		Log::log(Log::INFO, "fetchNewTaskAndAddStartNodeToQueue : before push back");
		tasks.emplace_back(myInstance, allocator_p->get_thread(thread_id), allocator, thread_id, 
						ipart_list.front().first + myInstance->baseMLO->exp_my_first_part_id);
		Log::log(Log::INFO, "fetchNewTaskAndAddStartNodeToQueue : after push back");
		Log::log(Log::INFO, "fetchNewTaskAndAddStartNodeToQueue : push back new task node pair");
		queue.push(std::make_pair(std::ref(tasks.back()), std::ref(Graph<Mlclass>::getInstance().getStartNode())));
		ipart_list.front().first++;
		if (ipart_list.front().first > ipart_list.front().second) {
			ipart_list.pop_front();
		}
		Log::log(Log::INFO, "fetchNewTaskAndAddStartNodeToQueue : New task created");
		return true;
	}

	void execTaskNode (Task<Mlclass>& task, Node<Mlclass>& node) {
		Log::log(Log::INFO, "Executing node: " + Graph<Mlclass>::getNodeName(node.name) + " Task id: " + std::to_string(task.part_id));
		// std::cout<<"\t\t"<<"Exegicuting node: " + Graph<Mlclass>::getNodeName(node.name) + " Task id: " + std::to_string(task.part_id)<<std::endl;
		if (node.name == NodeName::FinishNode) {
			Log::log(Log::INFO, "Task finished. Task id: " + std::to_string(task.part_id));
		}
		task.execNode(node);
	}
	
	void pushSuccessorsInqueue (Task<Mlclass>& task, Node<Mlclass>& node) {
		std::vector<Node<Mlclass>*> successors = Graph<Mlclass>::getInstance().getSuccessors(node, *task.context);
		for (Node<Mlclass>* successor :  successors) {
			Log::log(Log::INFO, "Push node : " + Graph<Mlclass>::getNodeName(successor->name) + " in queue");
			queue.push(std::make_pair(std::ref(task), std::ref(*successor)));
		}
	}
};


#else


template <class Mlclass>
class Scheduler {
public:
	Mlclass *myInstance;
	CudaCustomAllocator* allocator;
	CudaAllocatorProcess *allocator_p;
	int thread_id;

	std::list<Task<Mlclass>> tasks;
	// max number of tasks that can be scheduled at once
	int nr_task_limit;

	// // execution queue
	// std::queue<std::pair<Task<Mlclass>&, Node<Mlclass>&>> queue;

	// Separate queues for each NodeType
	std::queue<std::pair<Task<Mlclass>&, Node<Mlclass>&>> cpuQueue;
    std::queue<std::pair<Task<Mlclass>&, Node<Mlclass>&>> gpuQueue;
    std::queue<std::pair<Task<Mlclass>&, Node<Mlclass>&>> memcpyQueue;
    std::queue<std::pair<Task<Mlclass>&, Node<Mlclass>&>> syncQueue;

	size_t ipart_first, ipart_last;
	std::list<std::pair<size_t, size_t>> ipart_list;

	Scheduler(Mlclass *myInstance,CudaAllocatorProcess *allocator_p, 
			CudaCustomAllocator* allocator, int thread_id,int task_num=1) : 
		myInstance(myInstance), allocator_p(allocator_p),allocator(allocator), thread_id(thread_id),nr_task_limit(task_num)
	{
		nr_task_limit = myInstance->nr_task_limit;
		ipart_first = 0;
		ipart_last = 0;
	}

	void run() {
		Graph<Mlclass>::getInstance().printGraph();

		// unsigned int seed = static_cast<unsigned int>(
		// 	std::chrono::system_clock::now().time_since_epoch().count()
		// ) + omp_get_thread_num();
		// thread_local std::mt19937 generator(seed);

		// 定义队列选择的分布（0: gpu, 1: memcpy, 2: cpu, 3: sync）
		// std::uniform_int_distribution<int> queueDist(0, 3);
    
		// 定义 issue_task_limit 的分布（1 到 nr_task_limit / 3）
		// std::uniform_int_distribution<int> limitDist(2, nr_task_limit / 2);


		// fill the queues with initial tasks
		int issue_task_limit = max(1, (thread_id % nr_task_limit));
		while (canAddNewTask() && issue_task_limit > 0) {
			fetchNewTaskAndAddStartNodeToQueue();
			issue_task_limit--;
		}

		// main loop
		int queueChoice=0;
		while (!cpuQueue.empty() || !gpuQueue.empty() || !memcpyQueue.empty() || !syncQueue.empty()) {
			printAllQueues();
			bool exec_all = true;

			// 随机选择一个队列
			// int queueChoice = queueDist(generator);
			std::string selectedQueue = "cpu";
			// switch(queueChoice) {
			// 	case 0: 
			// 		selectedQueue = "gpu"; 
			// 		if (gpuQueue.empty()) exec_all = true;
			// 		break;
			// 	case 1: 
			// 		selectedQueue = "memcpy"; 
			// 		if (memcpyQueue.empty()) exec_all = true;
			// 		break;
			// 	case 2: 
			// 		selectedQueue = "cpu"; 
			// 		if (cpuQueue.empty()) exec_all = true;
			// 		break;
			// 	case 3: 
			// 		selectedQueue = "sync"; 
			// 		if (selectedQueue.empty()) exec_all = true;
			// 		break;
			// 	default: 
			// 		exec_all = true;
			// 		break;
			// }
			// queueChoice = (queueChoice + 1) % 4;
	
			// Log::log(Log::INFO, "Selected queue: " + selectedQueue);
	
			// // 随机决定 issue_task_limit
			// int current_issue_limit = limitDist(generator);
			int current_issue_limit = max(rand()%nr_task_limit, 1);
			// Log::log(Log::INFO, "issue_task_limit: " + std::to_string(current_issue_limit));

			// // Inorder to pipeline the execution, we issue a limited number of tasks to each queue
			// issue_task_limit = std::min(current_issue_limit, (int)tasks.size());
			// Log::log(Log::INFO, "issue_task_limit: " + std::to_string(issue_task_limit));			

			// Execute GPU tasks
			while (!gpuQueue.empty() && issue_task_limit > 0 && (selectedQueue == "gpu" || exec_all)) {
				Log::log(Log::INFO, "Executing GPU tasks");
				printAllQueues();
				std::pair<Task<Mlclass>&, Node<Mlclass>&> task_node_pair = gpuQueue.front();
				gpuQueue.pop();
				Task<Mlclass>& task = task_node_pair.first;
				Node<Mlclass>& node = task_node_pair.second;
				execTaskNode(task, node);
				pushSuccessorsInqueue(task, node);

				issue_task_limit--;
			}

			issue_task_limit = std::min(current_issue_limit, (int)tasks.size());

			// Execute Memcpy tasks
			while (!memcpyQueue.empty() && issue_task_limit > 0 && (selectedQueue == "memcpy" || exec_all)) {
				Log::log(Log::INFO, "Executing Memcpy tasks");
				printAllQueues();
				std::pair<Task<Mlclass>&, Node<Mlclass>&> task_node_pair = memcpyQueue.front();
				memcpyQueue.pop();
				Task<Mlclass>& task = task_node_pair.first;
				Node<Mlclass>& node = task_node_pair.second;
				execTaskNode(task, node);
				pushSuccessorsInqueue(task, node);

				issue_task_limit--;
			}

			std::queue<std::pair<Task<Mlclass>&, Node<Mlclass>&>> buffer;
			issue_task_limit = std::min(current_issue_limit, (int)tasks.size());
			// Execute CPU tasks
			while (!cpuQueue.empty() && issue_task_limit > 0 && (selectedQueue == "cpu" || exec_all)) {
				Log::log(Log::INFO, "Executing CPU tasks");
				printAllQueues();
				std::pair<Task<Mlclass>&, Node<Mlclass>&> task_node_pair = cpuQueue.front();
				cpuQueue.pop();
				Task<Mlclass>& task = task_node_pair.first;
				Node<Mlclass>& node = task_node_pair.second;
				if (taskNodeCanExec(task, node)) {
					execTaskNode(task, node);
					pushSuccessorsInqueue(task, node);
					checkFinishNodeAndCleanUpTask(task, node);
					issue_task_limit--;
				} else {
					// 当前任务不能执行，放入缓冲区
					buffer.push(task_node_pair);
				}
			}
			// 将缓冲区中的任务重新推回 cpuQueue
			while (!buffer.empty()) {
				cpuQueue.push(buffer.front());
				buffer.pop();
			}

			issue_task_limit = std::min(current_issue_limit, (int)tasks.size());

			// Execute Sync tasks
			while (!syncQueue.empty() && issue_task_limit > 0 && (selectedQueue == "sync" || exec_all)) {
				Log::log(Log::INFO, "Executing Sync tasks");
				printAllQueues();
				std::pair<Task<Mlclass>&, Node<Mlclass>&> task_node_pair = syncQueue.front();
				syncQueue.pop();
				Task<Mlclass>& task = task_node_pair.first;
				Node<Mlclass>& node = task_node_pair.second;
				execTaskNode(task, node);
				pushSuccessorsInqueue(task, node);

				issue_task_limit--;
			}

			// check if we can add new task to the queue
			issue_task_limit = current_issue_limit;
			while (canAddNewTask() && issue_task_limit > 0) {
				fetchNewTaskAndAddStartNodeToQueue();
				issue_task_limit--;
			}

			printAllQueues();
		}
	}

	// void printQueue() {
	// 	Log::log(Log::INFO, "================ Printing queue ================");
	// 	std::queue<std::pair<Task<Mlclass>&, Node<Mlclass>&>> queue_copy = queue;
	// 	while (!queue_copy.empty()) {
	// 		std::pair<Task<Mlclass>&, Node<Mlclass>&> task_node_pair = queue_copy.front();
	// 		queue_copy.pop();
	// 		Task<Mlclass>& task = task_node_pair.first;
	// 		Node<Mlclass>& node = task_node_pair.second;
	// 		Log::log(Log::INFO, "Task id: " + std::to_string(task.part_id) + ", Node name: " + Graph<Mlclass>::getNodeName(node.name));
	// 	}
	// 	Log::log(Log::INFO, "================ End printing queue ================\n");
	// }


    // Utility method to print the current state of all queues (optional for debugging)
    void printAllQueues() {
        Log::log(Log::INFO, "================ Current Queue States ===============");
        printQueueState("CPU Queue", cpuQueue);
        printQueueState("GPU Queue", gpuQueue);
        printQueueState("Memcpy Queue", memcpyQueue);
        printQueueState("Sync Queue", syncQueue);
        Log::log(Log::INFO, "================ End Queue States ===============\n");
    }

    void printQueueState(const std::string& queueName, std::queue<std::pair<Task<Mlclass>&, Node<Mlclass>&>> queue_copy) {
        Log::log(Log::INFO, queueName + " Size: " + std::to_string(queue_copy.size()));
        while (!queue_copy.empty()) {
            auto task_node_pair = queue_copy.front();
            queue_copy.pop();
            Task<Mlclass>& task = task_node_pair.first;
            Node<Mlclass>& node = task_node_pair.second;
            Log::log(Log::INFO, "    Task id: " + std::to_string(task.part_id) + ", Node: " + Graph<Mlclass>::getNodeName(node.name));
        }
    }

	// if we can add new task to the queue, return true
	// otherwise, return false
	// TODO: add memory bound check
	bool canAddNewTask() {
		if (tasks.size() >= nr_task_limit) {
			Log::log(Log::INFO, "can not add new task, because the number of tasks is already at the limit");
			return false;
		}

		if (ipart_list.empty()) {
			bool hasNewIpart = addNewIpartToIpartList();
			if (!hasNewIpart) {
				Log::log(Log::INFO, "can not add new task, because no new ipart");
				return false;
			}
		}
		return true;
	}

	bool addNewIpartToIpartList() {
		bool hasNewIpart = myInstance->baseMLO->exp_ipart_ThreadTaskDistributor->getTasks(ipart_first, ipart_last);
		// bool hasNewIpart = getTasks(ipart_first, ipart_last);
		if (hasNewIpart) {
			ipart_list.push_back(std::make_pair(ipart_first, ipart_last));
			Log::log(Log::INFO, "ipart_list push back. ipart_first: " + std::to_string(ipart_first) + ", ipart_last: " + std::to_string(ipart_last));
		}
		return hasNewIpart;
	}

	void fetchNewTaskAndAddStartNodeToQueue() {
		// Log::log(Log::INFO, "fetchNewTaskAndAddStartNodeToQueue : Fetching new task");
		// if (ipart_list.empty()) {
		// 	bool hasNewIpart = addNewIpartToIpartList();
		// 	if (!hasNewIpart) {
		// 		Log::log(Log::INFO, "fetchNewTaskAndAddStartNodeToQueue : No new ipart");
		// 		return false;
		// 	}
		// }
		// Log::log(Log::INFO, "fetchNewTaskAndAddStartNodeToQueue : before push back");
		assert(!ipart_list.empty());

		tasks.emplace_back(myInstance, allocator_p->get_thread(thread_id), allocator, thread_id, 
						ipart_list.front().first + myInstance->baseMLO->exp_my_first_part_id);
		Log::log(Log::INFO, "fetchNewTaskAndAddStartNodeToQueue : after push back");
		Log::log(Log::INFO, "fetchNewTaskAndAddStartNodeToQueue : push back new task node pair");
		
		enqueueTaskNode(std::ref(tasks.back()), std::ref(Graph<Mlclass>::getInstance().getStartNode()));
		// queue.push(std::make_pair(std::ref(tasks.back()), std::ref(Graph<Mlclass>::getInstance().getStartNode())));
		ipart_list.front().first++;
		if (ipart_list.front().first > ipart_list.front().second) {
			ipart_list.pop_front();
		}
		Log::log(Log::INFO, "fetchNewTaskAndAddStartNodeToQueue : New task created");
	}

	void execTaskNode (Task<Mlclass>& task, Node<Mlclass>& node) {
		Log::log(Log::INFO, "Executing node: " + Graph<Mlclass>::getNodeName(node.name) + " Task id: " + std::to_string(task.part_id));
		// std::cout<<"\t\t"<<"Exegicuting node: " + Graph<Mlclass>::getNodeName(node.name) + " Task id: " + std::to_string(task.part_id)<<std::endl;
		if (node.name == NodeName::FinishNode) {
			Log::log(Log::INFO, "Task finished. Task id: " + std::to_string(task.part_id));
		}
		task.execNode(node);
	}

	bool taskNodeCanExec (Task<Mlclass>& task, Node<Mlclass>& node) {

		bool can_exec = true;
#ifdef NEWMEM
		size_t max_mem = 0.;
		auto& ctx = *task.context;
		if (node.name == NodeName::AccDoExpectationOneParticlePreMemAlloc) {
			max_mem = calculate_mem_xjl1<Mlclass>(ctx.baseMLO, ctx.accMLO, ctx.part_id, ctx.sp);
			ctx.ptrFactory.freeTaskAllocator();
			can_exec = ctx.ptrFactory.ifTaskCanUse(1, max_mem);
		} else if (node.name == NodeName::AccDoExpectationOneParticlePostPerBodyGetFTAndCtfsMemAlloc) {
			max_mem = calculate_mem_xjl2<Mlclass>(ctx.baseMLO, ctx.accMLO, ctx.part_id, ctx.sp);
			ctx.ptrFactory.freeTaskAllocator();
			can_exec = ctx.ptrFactory.ifTaskCanUse(2, max_mem);
		} else if (node.name == NodeName::AccDoExpectationOneParticlePostPerBodyCvtDToWCoarseMemAlloc) {
			max_mem = calculate_mem_xjl3<Mlclass>(ctx.baseMLO, ctx.accMLO, ctx.part_id, ctx.sp, *ctx.FineProjectionData);
			ctx.ptrFactory.freeTaskAllocator();
			can_exec = ctx.ptrFactory.ifTaskCanUse(3, max_mem);
		}
#endif
		Log::log(Log::INFO, "taskNodeCanExec check node: " + Graph<Mlclass>::getNodeName(node.name) + 
		" Task id: " + std::to_string(task.part_id) + 
		" can_exec: " + std::to_string(can_exec) + 
		" request memory size: " + std::to_string(max_mem));
		return can_exec;
	}

    // Enqueue a node into the appropriate queue based on its NodeType
	void enqueueTaskNode(Task<Mlclass>& task, Node<Mlclass>& node) {
		Log::log(Log::INFO, "Enqueue node: " + Graph<Mlclass>::getNodeName(node.name) + " Task id: " + std::to_string(task.part_id));
        switch (node.type) {
            case NodeType::CPU:
                cpuQueue.emplace(task, node);
                break;
            case NodeType::GPU:
                gpuQueue.emplace(task, node);
                break;
            case NodeType::Memcpy:
                memcpyQueue.emplace(task, node);
                break;
            case NodeType::Sync:
                syncQueue.emplace(task, node);
                break;
            case NodeType::Start:
                cpuQueue.emplace(task, node);
                break;
            case NodeType::Finish:
                cpuQueue.emplace(task, node);
                break;
            default:
                Log::log(Log::ERROR, "Unknown NodeType encountered during enqueuing.");
                break;
        }
    }
	
	void pushSuccessorsInqueue (Task<Mlclass>& task, Node<Mlclass>& node) {
		std::vector<Node<Mlclass>*> successors = Graph<Mlclass>::getInstance().getSuccessors(node, *task.context);
		for (Node<Mlclass>* successor :  successors) {
			Log::log(Log::INFO, "Push node: " + Graph<Mlclass>::getNodeName(successor->name) + " in queue");
			// queue.push(std::make_pair(std::ref(task), std::ref(*successor)));
			enqueueTaskNode(std::ref(task), std::ref(*successor));
		}
	}

	 // If node is FinishNode, clean up the task
	 void checkFinishNodeAndCleanUpTask(Task<Mlclass>& task, Node<Mlclass>& node) {
		Log::log(Log::INFO, "checkFinishNodeAndCleanUpTask");
        if (node.type == NodeType::Finish) {
            task.printExecutionTimes();
            Log::log(Log::INFO, "Task finished Task id: " + std::to_string(task.part_id));

            // Remove task from the list of tasks
            auto it = std::find_if(tasks.begin(), tasks.end(),
                [&task](const Task<Mlclass>& t) { return t.part_id == task.part_id; });

            if (it != tasks.end()) {
                Log::log(Log::INFO, "Delete task. Task id: " + std::to_string(task.part_id));
                tasks.erase(it);
            }
        }
    }

};

#endif
// -------------------------------  Some explicit template instantiations
template __global__ void CudaKernels::cuda_kernel_translate2D<XFLOAT>(XFLOAT *,
    XFLOAT*, int, int, int, int, int);

template __global__ void CudaKernels::cuda_kernel_translate3D<XFLOAT>(XFLOAT *,
    XFLOAT *, int, int, int, int, int, int, int);

template __global__ void cuda_kernel_multi<XFLOAT>( XFLOAT *,
	XFLOAT *, XFLOAT, int);

template __global__ void CudaKernels::cuda_kernel_multi<XFLOAT>( XFLOAT *,
	XFLOAT, int);

template __global__ void cuda_kernel_multi<XFLOAT>( XFLOAT *, XFLOAT *,
	XFLOAT *, XFLOAT, int);

// ----------------------------------------------------------------------

// High-level CUDA objects

size_t MlDeviceBundle::checkFixedSizedObjects(int shares)
{
	int devCount;
	size_t BoxLimit;
	HANDLE_ERROR(cudaGetDeviceCount(&devCount));
	if(device_id >= devCount)
		CRITICAL(ERR_GPUID);

	HANDLE_ERROR(cudaSetDevice(device_id));

	size_t free(0), total(0);
	DEBUG_HANDLE_ERROR(cudaMemGetInfo( &free, &total ));
	float margin(1.05);
	BoxLimit = pow(free/(margin*2.5*sizeof(XFLOAT)*((float)shares)),(1/3.0)) / ((float) baseMLO->mymodel.padding_factor);
	//size_t BytesNeeded = ((float)shares)*margin*2.5*sizeof(XFLOAT)*pow((baseMLO->mymodel.ori_size*baseMLO->mymodel.padding_factor),3);

	return(BoxLimit);
}
void MlDeviceBundle::setupFixedSizedObjects()
{
	int devCount;
	HANDLE_ERROR(cudaGetDeviceCount(&devCount));
	if(device_id >= devCount)
	{
		//std::cerr << " using device_id=" << device_id << " (device no. " << device_id+1 << ") which is higher than the available number of devices=" << devCount << std::endl;
		CRITICAL(ERR_GPUID);
	}
	else
		HANDLE_ERROR(cudaSetDevice(device_id));

	//Can we pre-generate projector plan and corresponding euler matrices for all particles
	if (baseMLO->do_skip_align || baseMLO->do_skip_rotate || baseMLO->do_auto_refine || baseMLO->mymodel.orientational_prior_mode != NOPRIOR)
		generateProjectionPlanOnTheFly = true;
	else
		generateProjectionPlanOnTheFly = false;

	unsigned nr_proj = baseMLO->mymodel.PPref.size();
	unsigned nr_bproj = baseMLO->wsum_model.BPref.size();

	projectors.resize(nr_proj);
	backprojectors.resize(nr_bproj);

	/*======================================================
	              PROJECTOR AND BACKPROJECTOR
	======================================================*/

	for (int imodel = 0; imodel < nr_proj; imodel++)
	{
		projectors[imodel].setMdlDim(
				baseMLO->mymodel.PPref[imodel].data.xdim,
				baseMLO->mymodel.PPref[imodel].data.ydim,
				baseMLO->mymodel.PPref[imodel].data.zdim,
				baseMLO->mymodel.PPref[imodel].data.yinit,
				baseMLO->mymodel.PPref[imodel].data.zinit,
				baseMLO->mymodel.PPref[imodel].r_max,
				baseMLO->mymodel.PPref[imodel].padding_factor);

		projectors[imodel].initMdl(baseMLO->mymodel.PPref[imodel].data.data);

	}

	for (int imodel = 0; imodel < nr_bproj; imodel++)
	{
		backprojectors[imodel].setMdlDim(
				baseMLO->wsum_model.BPref[imodel].data.xdim,
				baseMLO->wsum_model.BPref[imodel].data.ydim,
				baseMLO->wsum_model.BPref[imodel].data.zdim,
				baseMLO->wsum_model.BPref[imodel].data.yinit,
				baseMLO->wsum_model.BPref[imodel].data.zinit,
				baseMLO->wsum_model.BPref[imodel].r_max,
				baseMLO->wsum_model.BPref[imodel].padding_factor);

		backprojectors[imodel].initMdl();
	}

	/*======================================================
	                    CUSTOM ALLOCATOR
	======================================================*/

	int memAlignmentSize;
	cudaDeviceGetAttribute ( &memAlignmentSize, cudaDevAttrTextureAlignment, device_id );
	allocator = new CudaCustomAllocator(0, memAlignmentSize);
}

void MlDeviceBundle::setupTunableSizedObjects(size_t allocationSize,int threadnum)
{
	unsigned nr_models = baseMLO->mymodel.nr_classes;
	int devCount;
	HANDLE_ERROR(cudaGetDeviceCount(&devCount));
	if(device_id >= devCount)
	{
		//std::cerr << " using device_id=" << device_id << " (device no. " << device_id+1 << ") which is higher than the available number of devices=" << devCount << std::endl;
		CRITICAL(ERR_GPUID);
	}
	else
		HANDLE_ERROR(cudaSetDevice(device_id));

	/*======================================================
	                    CUSTOM ALLOCATOR
	======================================================*/
#ifdef DEBUG_CUDA
	printf("DEBUG: Total GPU allocation size set to %zu MB on device id %d.\n", allocationSize / (1000*1000), device_id);
#endif
#ifdef NEWMEM
	allocator->resize(0.1*allocationSize);
	allocator_p=new CudaAllocatorProcess(0.9*allocationSize,threadnum);
#else
// #ifndef CUDA_NO_CUSTOM_ALLOCATION
	allocator->resize(allocationSize);
#endif


	/*======================================================
	                    PROJECTION PLAN
	======================================================*/

	coarseProjectionPlans.resize(nr_models, allocator);

	for (int iclass = 0; iclass < nr_models; iclass++)
	{
		//If doing predefined projector plan at all and is this class significant
		if (!generateProjectionPlanOnTheFly && baseMLO->mymodel.pdf_class[iclass] > 0.)
		{
			std::vector<int> exp_pointer_dir_nonzeroprior;
			std::vector<int> exp_pointer_psi_nonzeroprior;
			std::vector<RFLOAT> exp_directions_prior;
			std::vector<RFLOAT> exp_psi_prior;

			long unsigned itrans_max = baseMLO->sampling.NrTranslationalSamplings() - 1;
			long unsigned nr_idir = baseMLO->sampling.NrDirections(0, &exp_pointer_dir_nonzeroprior);
			long unsigned nr_ipsi = baseMLO->sampling.NrPsiSamplings(0, &exp_pointer_psi_nonzeroprior );

			coarseProjectionPlans[iclass].setup(
					baseMLO->sampling,
					exp_directions_prior,
					exp_psi_prior,
					exp_pointer_dir_nonzeroprior,
					exp_pointer_psi_nonzeroprior,
					NULL, //Mcoarse_significant
					baseMLO->mymodel.pdf_class,
					baseMLO->mymodel.pdf_direction,
					nr_idir,
					nr_ipsi,
					0, //idir_min
					nr_idir - 1, //idir_max
					0, //ipsi_min
					nr_ipsi - 1, //ipsi_max
					0, //itrans_min
					itrans_max,
					0, //current_oversampling
					1, //nr_oversampled_rot
					iclass,
					true, //coarse
					!IS_NOT_INV,
					baseMLO->do_skip_align,
					baseMLO->do_skip_rotate,
					baseMLO->mymodel.orientational_prior_mode
					);
		}
	}
};

void MlOptimiserCuda::resetData()
{
	int devCount;
	HANDLE_ERROR(cudaGetDeviceCount(&devCount));
	if(device_id >= devCount)
	{
		//std::cerr << " using device_id=" << device_id << " (device no. " << device_id+1 << ") which is higher than the available number of devices=" << devCount << std::endl;
		CRITICAL(ERR_GPUID);
	}
	else
		HANDLE_ERROR(cudaSetDevice(device_id));

	unsigned nr_classes = baseMLO->mymodel.nr_classes;

	classStreams.resize(nr_classes, 0);
	for (int i = 0; i < nr_classes; i++)
		HANDLE_ERROR(cudaStreamCreate(&classStreams[i])); //HANDLE_ERROR(cudaStreamCreateWithFlags(&classStreams[i],cudaStreamNonBlocking));

	const int initialStreamPoolSize = nr_task_limit * (nr_classes + 1);
	 for (int i = 0; i < initialStreamPoolSize; ++i) {
		cudaStream_t stream;
		HANDLE_ERROR(cudaStreamCreate(&stream));
		streamPool.push(stream);
	}


	transformer1.clear();
	transformer2.clear();
};

void MlOptimiserCuda::doThreadExpectationSomeParticles(int thread_id,int task_num=1)
{
#ifdef TIMING
	// Only time one thread
	if (thread_id == 0)
		baseMLO->timer.tic(baseMLO->TIMING_ESP_THR);
#endif
//	CTOC(cudaMLO->timer,"interParticle");
	// LAUNCH_PRIVATE_ERROR(cudaGetLastError(),this->errorStatus);

	int devCount;
	HANDLE_ERROR(cudaGetDeviceCount(&devCount));
	if(device_id >= devCount)
	{
		//std::cerr << " using device_id=" << device_id << " (device no. " << device_id+1 << ") which is higher than the available number of devices=" << devCount << std::endl;
		CRITICAL(ERR_GPUID);
	}
	else
		DEBUG_HANDLE_ERROR(cudaSetDevice(device_id));

	LAUNCH_PRIVATE_ERROR(cudaGetLastError(),this->errorStatus);

	//put mweight allocation here
	size_t first_ipart = 0, last_ipart = 0;

// 	while (baseMLO->exp_ipart_ThreadTaskDistributor->getTasks(first_ipart, last_ipart))
// 	{
// 		CTIC(timer,"oneTask");
// 		for (long unsigned ipart = first_ipart; ipart <= last_ipart; ipart++)
// 		{
// #ifdef TIMING
// 	// Only time one thread
// 	if (thread_id == 0)
// 		baseMLO->timer.tic(baseMLO->TIMING_ESP_DIFF2_A);
// #endif

// #ifdef NEWMEM
// 			AccPtrFactoryNew ptrFactory(cudaStreamPerThread,allocator_p->get_thread(thread_id),allocator);
// 			// AccPtrFactoryNew ptrFactory(cudaStreamPerThread,NULL,allocator);
// 			// AccPtrFactoryNew ptrFactory(cudaStreamPerThread,allocator);
// 			// AccPtrFactory ptrFactory(allocator, cudaStreamPerThread);

// 			accDoExpectationOneParticle<MlOptimiserCuda>(this, baseMLO->exp_my_first_part_id + ipart, thread_id, ptrFactory);
// #else
// 			AccPtrFactory ptrFactory(allocator, cudaStreamPerThread);
//             accDoExpectationOneParticle<MlOptimiserCuda>(this, baseMLO->exp_my_first_part_id + ipart, thread_id, ptrFactory);
// #endif
// 	LAUNCH_PRIVATE_ERROR(cudaGetLastError(),this->errorStatus);

// 		}
// 		CTOC(timer,"oneTask");

// 	}
	Log::log(Log::ERROR, "doThreadExpectationSomeParticles : Fetching new task");
	Scheduler<MlOptimiserCuda> scheduler(this, allocator_p, allocator, thread_id,task_num);
	scheduler.run();


#ifdef TIMING
	// Only time one thread
	if (thread_id == 0)
		baseMLO->timer.toc(baseMLO->TIMING_ESP_THR);
#endif
}

