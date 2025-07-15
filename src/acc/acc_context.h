#ifndef ACC_CONTEXT_H_
#define ACC_CONTEXT_H_

#include "src/ml_optimiser_mpi.h"

template<class MlClass>
class Context {
public:
    cudaStream_t cudaStreamPerTask;
    std::vector<cudaStream_t> classStreams;

// accDoExpectationOneParticlePre
    unsigned long part_id_sorted;
    long int part_id;
    

// ibody loop
    int ibody;

// accDoExpectationOneParticlePerBody

#ifdef NEWMEM
    std::vector <IndexedDataArrayNew >* CoarsePassWeights;
    std::vector <IndexedDataArrayNew >* FinePassWeights;
    // std::vector <IndexedDataArray >* CoarsePassWeights;
    // std::vector <IndexedDataArray >* FinePassWeights;
#else
    std::vector <IndexedDataArray >* CoarsePassWeights;
    std::vector <IndexedDataArray >* FinePassWeights;
#endif

    std::vector < std::vector <IndexedDataArrayMask > >* FinePassClassMasks;
    std::vector < ProjectionParams >* FineProjectionData;

#ifdef NEWMEM
    std::vector < AccPtrBundleNew >* bundleD2;
    std::vector < AccPtrBundleNew >* bundleSWS;
#else
    std::vector < AccPtrBundle >* bundleD2;
    std::vector < AccPtrBundle >* bundleSWS;
#endif

// getAllSquaredDifferencesFine  
#ifdef NEWMEM
    AccPtrBundleNew* AllEulers;
    unsigned long newDataSize;
    std::vector < AccPtrNew<XFLOAT> >* eulers;
    // AccPtrBundle* AllEulers;

#else
    AccPtrBundle* AllEulers;
    unsigned long newDataSize;
    std::vector < AccPtr<XFLOAT> >* eulers;
#endif

// getAllSquaredDifferencesCoarse
    unsigned exp_ipass;
    OptimisationParamters op;
    SamplingParameters sp;
    MlOptimiser *baseMLO;
    MlClass *accMLO;

#ifdef NEWMEM
    AccPtrFactoryNew ptrFactory;
    AccPtrNew<XFLOAT> Mweight;
    // AccPtr<XFLOAT> Mweight;
#else
    AccPtrFactory ptrFactory;
    AccPtr<XFLOAT> Mweight;
#endif
    int thread_id;

// getAllSquaredDifferencesCoarsePost
	unsigned long weightsPerPart;
#ifdef NEWMEM
	std::vector< AccProjectorPlanNew >projectorPlans;
    AccPtrNew<XFLOAT> allWeights;
#else
	std::vector< AccProjectorPlan >projectorPlans;
    AccPtr<XFLOAT> allWeights;
#endif
	long int allWeights_pos;
	bool do_CC;
    int img_id;

// Fine & coarse
#ifdef NEWMEM
    AccPtrNew<XFLOAT> Fimg_, trans_xyz, corr_img;  // StoreWeightedSums
#else
    AccPtr<XFLOAT> Fimg_, trans_xyz, corr_img;  // StoreWeightedSums
#endif
    long unsigned translation_num, image_size; // StoreWeightedSums
    size_t trans_x_offset, trans_y_offset, trans_z_offset; // StoreWeightedSums
    size_t img_re_offset, img_im_offset;

// StoreWeightedSums
    std::vector<RFLOAT>* oversampled_translations_x;
    std::vector<RFLOAT>* oversampled_translations_y;
    std::vector<RFLOAT>* oversampled_translations_z;
    std::vector<MultidimArray<RFLOAT> >* thr_wsum_pdf_direction;
	std::vector<RFLOAT>* thr_wsum_norm_correction, *thr_sumw_group, *thr_wsum_pdf_class, *thr_wsum_prior_offsetx_class, *thr_wsum_prior_offsety_class;
	RFLOAT thr_wsum_sigma2_offset;
	MultidimArray<RFLOAT> *thr_metadata, *zeroArray;

    std::vector<RFLOAT>* exp_wsum_norm_correction;
	std::vector<RFLOAT>* exp_wsum_scale_correction_XA, *exp_wsum_scale_correction_AA;
	std::vector<RFLOAT>* thr_wsum_signal_product_spectra, *thr_wsum_reference_power_spectra;
	std::vector<MultidimArray<RFLOAT> > *thr_wsum_sigma2_noise, *thr_wsum_ctf2, *thr_wsum_stMulti;

    std::vector<MultidimArray<RFLOAT> >* exp_local_STMulti;
    bool do_subtomo_correction;
#ifdef NEWMEM
    AccPtrNew<XFLOAT> Fimgs;
    AccPtrNew<XFLOAT> sorted_weights;
    AccPtrNew<XFLOAT> ctfs;
    AccPtrNew<XFLOAT> Minvsigma2s;
    AccPtrNew<XFLOAT> wdiff2s;
#else
    AccPtr<XFLOAT> Fimgs;
    AccPtr<XFLOAT> sorted_weights;
    AccPtr<XFLOAT> ctfs;
    AccPtr<XFLOAT> Minvsigma2s;
    AccPtr<XFLOAT> wdiff2s;
#endif
    int my_metadata_offset, group_id, optics_group;
    RFLOAT my_pixel_size;
    bool ctf_premultiplied;
    XFLOAT part_scale;



    Context(cudaStream_t& cudaStream, std::vector<cudaStream_t>& streams)
        : cudaStreamPerTask(cudaStream), classStreams(streams) {

    }

    Context() {}

    void setStream(cudaStream_t& cudaStream, std::vector<cudaStream_t>& streams) {
        cudaStreamPerTask = cudaStream;
        classStreams = streams;
    }
};

#endif  // ACC_CONTEXT_H_