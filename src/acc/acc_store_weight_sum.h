#ifndef ACC_STORE_WEIGHT_SUM_H_
#define ACC_STORE_WEIGHT_SUM_H_

#include "src/ml_optimiser_mpi.h"



template<int kTransBlockSize, int kOrientBlockSize,
		 int kNrOverTrans, int kNrOverOrient, bool DATA3D>
__global__ void cuda_kernel_collect2jobs_block(
	Block<kTransBlockSize/kNrOverTrans, kNrOverTrans, kNrOverOrient> *blocks,
	size_t *trans_rearranged_index,
	size_t *orient_rearranged_index,
	XFLOAT *g_oo_otrans_x,          // otrans-size -> make const
    XFLOAT *g_oo_otrans_y,          // otrans-size -> make const
    XFLOAT *g_oo_otrans_z,          // otrans-size -> make const
    XFLOAT *g_myp_oo_otrans_x2y2z2, // otrans-size -> make const
    XFLOAT *g_i_weights,
    XFLOAT op_significant_weight,    // TODO Put in const
    XFLOAT op_sum_weight,            // TODO Put in const
    XFLOAT *g_o_weights,
    XFLOAT *g_thr_wsum_prior_offsetx_class,
    XFLOAT *g_thr_wsum_prior_offsety_class,
    XFLOAT *g_thr_wsum_prior_offsetz_class,
    XFLOAT *g_thr_wsum_sigma2_offset) {
	int block_id = blockIdx.x;
	int row_id = blockIdx.y;

	int thread_id = threadIdx.x;
	int thread_num = blockDim.x;

	auto& block = blocks[block_id];

	int trans_id = trans_rearranged_index[block.startRow + row_id / kNrOverTrans] * kNrOverTrans + row_id % kNrOverTrans;
	int c_trans_id = trans_id / kNrOverTrans;

	#pragma unroll
	for (int col_id = thread_id; col_id < kOrientBlockSize; col_id += thread_num) {
		auto rot_idx = orient_rearranged_index[block.startCol + col_id / kNrOverOrient] * kNrOverOrient + col_id % kNrOverOrient; 
		// auto rot_id = coarse_index2rot_id[block.startCol + col_id / kNrOverOrient];
        XFLOAT weight = 0.;
        XFLOAT thr_wsum_sigma2_offset = weight * 0.;
        XFLOAT thr_wsum_prior_offsetx_class = 0.;
        XFLOAT thr_wsum_prior_offsety_class = 0.;
        XFLOAT thr_wsum_prior_offsetz_class = 0.;

		int weight_idx = block.result_idx[row_id * kOrientBlockSize + col_id];
		if (weight_idx != -1) {
            weight = g_i_weights[weight_idx];
			if (weight >= op_significant_weight)
                weight /= op_sum_weight;
			else
                weight = 0.;
            
            XFLOAT thr_wsum_sigma2_offset = weight * g_myp_oo_otrans_x2y2z2[trans_id];
            XFLOAT thr_wsum_prior_offsetx_class = weight * g_oo_otrans_x[trans_id];
            XFLOAT thr_wsum_prior_offsety_class = weight * g_oo_otrans_y[trans_id];

            if (DATA3D)
                thr_wsum_prior_offsetz_class = weight * g_oo_otrans_z[trans_id];

            atomicAdd(&g_o_weights[rot_idx], weight);
            atomicAdd(&g_thr_wsum_sigma2_offset[rot_idx], thr_wsum_sigma2_offset);
            atomicAdd(&g_thr_wsum_prior_offsetx_class[rot_idx], thr_wsum_prior_offsetx_class);
            atomicAdd(&g_thr_wsum_prior_offsety_class[rot_idx], thr_wsum_prior_offsety_class);
            if (DATA3D)
                atomicAdd(&g_thr_wsum_prior_offsetz_class[rot_idx], thr_wsum_prior_offsetz_class);
        }  
	}
}


#ifdef _CUDA_ENABLED
template<int kTransBlockSize, int kOrientBlockSize,
		 int kNrOverTrans, int kNrOverOrient>
void collect2jobs_block(
	int block_num,
	Block<kTransBlockSize/kNrOverTrans, kNrOverTrans, kNrOverOrient> *blocks,
	size_t *trans_rearranged_index,
	size_t *orient_rearranged_index,
	XFLOAT *g_oo_otrans_x,          // otrans-size -> make const
    XFLOAT *g_oo_otrans_y,          // otrans-size -> make const
    XFLOAT *g_oo_otrans_z,          // otrans-size -> make const
    XFLOAT *g_myp_oo_otrans_x2y2z2, // otrans-size -> make const
    XFLOAT *g_i_weights,
    XFLOAT op_significant_weight,    // TODO Put in const
    XFLOAT op_sum_weight,            // TODO Put in const
    XFLOAT *g_o_weights,
    XFLOAT *g_thr_wsum_prior_offsetx_class,
    XFLOAT *g_thr_wsum_prior_offsety_class,
    XFLOAT *g_thr_wsum_prior_offsetz_class,
    XFLOAT *g_thr_wsum_sigma2_offset,
    bool isData3D,
	cudaStream_t stream) {
		// block : y : block num x : kTransBlockSize
		dim3 grid(block_num, kTransBlockSize);
		dim3 block(128);

    if (isData3D)
		cuda_kernel_collect2jobs_block<
		kTransBlockSize, kOrientBlockSize, kNrOverTrans, kNrOverOrient, 1
		><<<grid, block, 0, stream>>>(
				blocks,
				trans_rearranged_index,
				orient_rearranged_index,
				g_oo_otrans_x,          // otrans-size -> make const
                g_oo_otrans_y,          // otrans-size -> make const
                g_oo_otrans_z,          // otrans-size -> make const
                g_myp_oo_otrans_x2y2z2, // otrans-size -> make const
                g_i_weights,
                op_significant_weight,    // TODO Put in const
                op_sum_weight,            // TODO Put in const
                g_o_weights,
                g_thr_wsum_prior_offsetx_class,
                g_thr_wsum_prior_offsety_class,
                g_thr_wsum_prior_offsetz_class,
                g_thr_wsum_sigma2_offset);
    else
        cuda_kernel_collect2jobs_block<
		kTransBlockSize, kOrientBlockSize, kNrOverTrans, kNrOverOrient, 0
		><<<grid, block, 0, stream>>>(
				blocks,
				trans_rearranged_index,
				orient_rearranged_index,
				g_oo_otrans_x,          // otrans-size -> make const
                g_oo_otrans_y,          // otrans-size -> make const
                g_oo_otrans_z,          // otrans-size -> make const
                g_myp_oo_otrans_x2y2z2, // otrans-size -> make const
                g_i_weights,
                op_significant_weight,    // TODO Put in const
                op_sum_weight,            // TODO Put in const
                g_o_weights,
                g_thr_wsum_prior_offsetx_class,
                g_thr_wsum_prior_offsety_class,
                g_thr_wsum_prior_offsetz_class,
                g_thr_wsum_sigma2_offset);
}
#endif

// ----------------------------------------------------------------------------
// -------------------------- storeWeightedSums -------------------------------
// ----------------------------------------------------------------------------
template<class MlClass>
void storeWeightedSumsCollectData(Context<MlClass> &ctx)
{
    LAUNCH_HANDLE_ERROR(cudaGetLastError());

    OptimisationParamters& op = ctx.op;
	SamplingParameters& sp    = ctx.sp;
    MlOptimiser* baseMLO      = ctx.baseMLO;
	MlClass *accMLO           = ctx.accMLO;
#ifdef NEWMEM
    std::vector<IndexedDataArrayNew>& FinePassWeights = *ctx.FinePassWeights;
    // std::vector<IndexedDataArray>& FinePassWeights = *ctx.FinePassWeights;
    std::vector<ProjectionParams>& ProjectionData = *ctx.FineProjectionData;
    std::vector<std::vector<IndexedDataArrayMask > >& FPCMasks = *ctx.FinePassClassMasks;
    AccPtrFactoryNew& ptrFactory = ctx.ptrFactory;
    int ibody = ctx.ibody;
    std::vector< AccPtrBundleNew >& bundleSWS = *ctx.bundleSWS;
    // std::vector< AccPtrBundle >& bundleSWS = *ctx.bundleSWS;

    std::vector < AccPtrNew<Block<16, 4, 8>> >& blocks64x128 = *ctx.blocks64x128;
	std::vector < AccPtrNew<size_t> >& OrientRearrangedIndex = *ctx.OrientRearrangedIndex;
	AccPtrNew<size_t>& TransRearrangedIndex = ctx.TransRearrangedIndex;
#else
    std::vector<IndexedDataArray>& FinePassWeights = *ctx.FinePassWeights;
    std::vector<ProjectionParams>& ProjectionData = *ctx.FineProjectionData;
    std::vector<std::vector<IndexedDataArrayMask > >& FPCMasks = *ctx.FinePassClassMasks;
    AccPtrFactory& ptrFactory = ctx.ptrFactory;
    int ibody = ctx.ibody;
    std::vector< AccPtrBundle >& bundleSWS = *ctx.bundleSWS;
    
    std::vector < AccPtr<Block<16, 4, 8>> >& blocks64x128 = *ctx.blocks64x128;
	std::vector < AccPtr<size_t> >& OrientRearrangedIndex = *ctx.OrientRearrangedIndex;
	AccPtr<size_t>& TransRearrangedIndex = ctx.TransRearrangedIndex;

#endif
    int thread_id = ctx.thread_id;
#ifdef TIMING
	// if (op.part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
		baseMLO->timer.tic(baseMLO->TIMING_ESP_WSUM);
#endif
    CTIC(accMLO->timer,"storeWeightedSumsCollectData");

	CTIC(accMLO->timer,"store_init");

	// Re-do below because now also want unmasked images AND if (stricht_highres_exp >0.) then may need to resize
	std::vector<MultidimArray<Complex > > dummy;
	std::vector<std::vector<MultidimArray<Complex > > > dummy2;
	// std::vector<MultidimArray<RFLOAT> > exp_local_STMulti;

    ctx.exp_local_STMulti = new std::vector<MultidimArray<RFLOAT> >;
    std::vector<MultidimArray<RFLOAT> >& exp_local_STMulti = *ctx.exp_local_STMulti;
    bool& do_subtomo_correction = ctx.do_subtomo_correction;
	do_subtomo_correction = op.FstMulti.size() > 0 && NZYXSIZE(op.FstMulti[0]) > 0;
	if (do_subtomo_correction)
		exp_local_STMulti.resize(sp.nr_images);

	baseMLO->precalculateShiftedImagesCtfsAndInvSigma2s(false, true, op.part_id, sp.current_oversampling, op.metadata_offset, // inserted SHWS 12112015
			sp.itrans_min, sp.itrans_max, op.Fimg, op.Fimg_nomask, op.Fctf, dummy2, dummy2,
			op.local_Fctf, op.local_sqrtXi2, op.local_Minvsigma2, op.FstMulti, exp_local_STMulti);

	// In doThreadPrecalculateShiftedImagesCtfsAndInvSigma2s() the origin of the op.local_Minvsigma2s was omitted.
	// Set those back here
	for (int img_id = 0; img_id < sp.nr_images; img_id++)
	{
		int optics_group = baseMLO->mydata.getOpticsGroup(op.part_id, img_id);
		DIRECT_MULTIDIM_ELEM(op.local_Minvsigma2[img_id], 0) = 1. / (baseMLO->sigma2_fudge * DIRECT_A1D_ELEM(baseMLO->mymodel.sigma2_noise[optics_group], 0));
	}

	// For norm_correction and scale_correction of all images of this particle
	// std::vector<RFLOAT> exp_wsum_norm_correction;
	// std::vector<RFLOAT> exp_wsum_scale_correction_XA, exp_wsum_scale_correction_AA;
	// std::vector<RFLOAT> thr_wsum_signal_product_spectra, thr_wsum_reference_power_spectra;
	// exp_wsum_norm_correction.resize(sp.nr_images, 0.);
	// std::vector<MultidimArray<RFLOAT> > thr_wsum_sigma2_noise, thr_wsum_ctf2, thr_wsum_stMulti;

    ctx.exp_wsum_norm_correction = new std::vector<RFLOAT>;
    ctx.exp_wsum_scale_correction_XA = new std::vector<RFLOAT>;
    ctx.exp_wsum_scale_correction_AA = new std::vector<RFLOAT>;
    ctx.thr_wsum_signal_product_spectra = new std::vector<RFLOAT>;
    ctx.thr_wsum_reference_power_spectra = new std::vector<RFLOAT>;
    ctx.thr_wsum_sigma2_noise = new std::vector<MultidimArray<RFLOAT> >;
    ctx.thr_wsum_ctf2 = new std::vector<MultidimArray<RFLOAT> >;
    ctx.thr_wsum_stMulti = new std::vector<MultidimArray<RFLOAT> >;

    std::vector<RFLOAT>& exp_wsum_norm_correction = *ctx.exp_wsum_norm_correction;
    std::vector<RFLOAT>& exp_wsum_scale_correction_XA = *ctx.exp_wsum_scale_correction_XA;
    std::vector<RFLOAT>& exp_wsum_scale_correction_AA = *ctx.exp_wsum_scale_correction_AA;
    std::vector<RFLOAT>& thr_wsum_signal_product_spectra = *ctx.thr_wsum_signal_product_spectra;
    std::vector<RFLOAT>& thr_wsum_reference_power_spectra = *ctx.thr_wsum_reference_power_spectra;
    std::vector<MultidimArray<RFLOAT> >& thr_wsum_sigma2_noise = *ctx.thr_wsum_sigma2_noise;
    std::vector<MultidimArray<RFLOAT> >& thr_wsum_ctf2 = *ctx.thr_wsum_ctf2;
    std::vector<MultidimArray<RFLOAT> >& thr_wsum_stMulti = *ctx.thr_wsum_stMulti;

    exp_wsum_norm_correction.resize(sp.nr_images, 0.);

	// for noise estimation (per image)
	thr_wsum_sigma2_noise.resize(sp.nr_images);
    thr_wsum_ctf2.resize(sp.nr_images);
    thr_wsum_stMulti.resize(sp.nr_images);

	// For scale_correction
	if (baseMLO->do_scale_correction)
	{
		exp_wsum_scale_correction_XA.resize(sp.nr_images);
		exp_wsum_scale_correction_AA.resize(sp.nr_images);
		thr_wsum_signal_product_spectra.resize(sp.nr_images);
		thr_wsum_reference_power_spectra.resize(sp.nr_images);
	}

	// Possibly different array sizes in different optics groups!
	for (int img_id = 0; img_id < sp.nr_images; img_id++)
	{
		int optics_group = baseMLO->mydata.getOpticsGroup(op.part_id, img_id);
		thr_wsum_sigma2_noise[img_id].initZeros(baseMLO->image_full_size[optics_group]/2 + 1);
        thr_wsum_stMulti[img_id].initZeros(baseMLO->image_full_size[optics_group]/2 + 1);
        thr_wsum_ctf2[img_id].initZeros(baseMLO->image_full_size[optics_group]/2 + 1);
		if (baseMLO->do_scale_correction)
		{
			exp_wsum_scale_correction_AA[img_id] = 0.;
			exp_wsum_scale_correction_XA[img_id] = 0.;
			thr_wsum_signal_product_spectra[img_id] = 0.;
			thr_wsum_reference_power_spectra[img_id] = 0.;
		}
	}

	// std::vector<RFLOAT> oversampled_translations_x, oversampled_translations_y, oversampled_translations_z;
    ctx.oversampled_translations_x = new std::vector<RFLOAT>;
    ctx.oversampled_translations_y = new std::vector<RFLOAT>;
    ctx.oversampled_translations_z = new std::vector<RFLOAT>;
    std::vector<RFLOAT>& oversampled_translations_x = *ctx.oversampled_translations_x;
    std::vector<RFLOAT>& oversampled_translations_y = *ctx.oversampled_translations_y;
    std::vector<RFLOAT>& oversampled_translations_z = *ctx.oversampled_translations_z;
	// bool have_warned_small_scale = false;

	// Make local copies of weighted sums (except BPrefs, which are too big)
	// so that there are not too many mutex locks below
	// std::vector<MultidimArray<RFLOAT> > thr_wsum_pdf_direction;
	// std::vector<RFLOAT> thr_wsum_norm_correction, thr_sumw_group, thr_wsum_pdf_class, thr_wsum_prior_offsetx_class, thr_wsum_prior_offsety_class;
	// RFLOAT thr_wsum_sigma2_offset;
	// MultidimArray<RFLOAT> thr_metadata, zeroArray;
    ctx.thr_wsum_pdf_direction = new std::vector<MultidimArray<RFLOAT> >;
    ctx.thr_wsum_norm_correction = new std::vector<RFLOAT>;
    ctx.thr_sumw_group = new std::vector<RFLOAT>;
    ctx.thr_wsum_pdf_class = new std::vector<RFLOAT>;
    ctx.thr_wsum_prior_offsetx_class = new std::vector<RFLOAT>;
    ctx.thr_wsum_prior_offsety_class = new std::vector<RFLOAT>;
    ctx.thr_metadata = new MultidimArray<RFLOAT>;
    ctx.zeroArray = new MultidimArray<RFLOAT>;

    std::vector<MultidimArray<RFLOAT> >& thr_wsum_pdf_direction = *ctx.thr_wsum_pdf_direction;
    std::vector<RFLOAT>& thr_wsum_norm_correction = *ctx.thr_wsum_norm_correction;
    std::vector<RFLOAT>& thr_sumw_group = *ctx.thr_sumw_group;
    std::vector<RFLOAT>& thr_wsum_pdf_class = *ctx.thr_wsum_pdf_class;
    std::vector<RFLOAT>& thr_wsum_prior_offsetx_class = *ctx.thr_wsum_prior_offsetx_class;
    std::vector<RFLOAT>& thr_wsum_prior_offsety_class = *ctx.thr_wsum_prior_offsety_class;
    RFLOAT& thr_wsum_sigma2_offset = ctx.thr_wsum_sigma2_offset;
    MultidimArray<RFLOAT>& thr_metadata = *ctx.thr_metadata;
    MultidimArray<RFLOAT>& zeroArray = *ctx.zeroArray;


	// wsum_pdf_direction is a 1D-array (of length sampling.NrDirections()) for each class
	zeroArray.initZeros(baseMLO->sampling.NrDirections());
	thr_wsum_pdf_direction.resize(baseMLO->mymodel.nr_classes * baseMLO->mymodel.nr_bodies, zeroArray);
	// sumw_group is a RFLOAT for each group
	thr_sumw_group.resize(sp.nr_images, 0.);
	// wsum_pdf_class is a RFLOAT for each class
	thr_wsum_pdf_class.resize(baseMLO->mymodel.nr_classes, 0.);
	if (baseMLO->mymodel.ref_dim == 2)
	{
		thr_wsum_prior_offsetx_class.resize(baseMLO->mymodel.nr_classes, 0.);
		thr_wsum_prior_offsety_class.resize(baseMLO->mymodel.nr_classes, 0.);
	}
	// wsum_sigma2_offset is just a RFLOAT
	thr_wsum_sigma2_offset = 0.;
	CTOC(accMLO->timer,"store_init");

	/*=======================================================================================
	                           COLLECT 2 AND SET METADATA
	=======================================================================================*/

	CTIC(accMLO->timer,"collect_data_2");
	unsigned long nr_transes = sp.nr_trans*sp.nr_oversampled_trans;
	unsigned long nr_fake_classes = (sp.iclass_max-sp.iclass_min+1);
	unsigned long oversamples = sp.nr_oversampled_trans * sp.nr_oversampled_rot;
	std::vector<long int> block_nums(sp.nr_images*nr_fake_classes);

	for (int img_id = 0; img_id < sp.nr_images; img_id++)
	{
		// here we introduce offsets for the oo_transes in an array as it is more efficient to
		// copy one big array to/from GPU rather than four small arrays
		size_t otrans_x      = 0*(size_t)nr_fake_classes*nr_transes;
		size_t otrans_y      = 1*(size_t)nr_fake_classes*nr_transes;
		size_t otrans_z      = 2*(size_t)nr_fake_classes*nr_transes;
		size_t otrans_x2y2z2 = 3*(size_t)nr_fake_classes*nr_transes;

		// Allocate space for all classes, so that we can pre-calculate data for all classes, copy in one operation, call kenrels on all classes, and copy back in one operation
#ifdef NEWMEM
		AccPtrNew<XFLOAT>          oo_otrans = ptrFactory.make<XFLOAT>((size_t)nr_fake_classes*nr_transes*4);
#else
		AccPtr<XFLOAT>          oo_otrans = ptrFactory.make<XFLOAT>((size_t)nr_fake_classes*nr_transes*4);
#endif
		oo_otrans.allAlloc();

		int sumBlockNum =0;

		int my_metadata_offset = op.metadata_offset + img_id;
		int group_id = baseMLO->mydata.getGroupId(op.part_id, img_id);
		const int optics_group = baseMLO->mydata.getOpticsGroup(op.part_id, img_id);
		RFLOAT my_pixel_size = baseMLO->mydata.getImagePixelSize(op.part_id, img_id);

		CTIC(accMLO->timer,"collect_data_2_pre_kernel");
		for (unsigned long exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
		{
			unsigned long fake_class = exp_iclass-sp.iclass_min; // if we only have the third class to do, the third class will be the "first" we do, i.e. the "fake" first.
			if ((baseMLO->mymodel.pdf_class[exp_iclass] == 0.) || (ProjectionData[img_id].class_entries[exp_iclass] == 0) )
				continue;

			// Use the constructed mask to construct a partial class-specific input


#ifdef NEWMEM
			IndexedDataArrayNew thisClassFinePassWeights(FinePassWeights[img_id],FPCMasks[img_id][exp_iclass]);
			block_nums[nr_fake_classes*img_id + fake_class] = makeJobsForCollect1(thisClassFinePassWeights, FPCMasks[img_id][exp_iclass], ProjectionData[img_id].orientation_num[exp_iclass]);
#else
			IndexedDataArray thisClassFinePassWeights(FinePassWeights[img_id],FPCMasks[img_id][exp_iclass]);
						// Re-define the job-partition of the indexedArray of weights so that the collect-kernel can work with it.//合并rot相同的mask类（之前是会把rot相同的分成若干个小块）
			block_nums[nr_fake_classes*img_id + fake_class] = makeJobsForCollect(thisClassFinePassWeights, FPCMasks[img_id][exp_iclass], ProjectionData[img_id].orientation_num[exp_iclass]);
#endif
			// Re-define the job-partition of the indexedArray of weights so that the collect-kernel can work with it.
			// block_nums[nr_fake_classes*img_id + fake_class] = makeJobsForCollect(thisClassFinePassWeights, FPCMasks[img_id][exp_iclass], ProjectionData[img_id].orientation_num[exp_iclass]);

			bundleSWS[img_id].pack(FPCMasks[img_id][exp_iclass].jobOrigin);
			bundleSWS[img_id].pack(FPCMasks[img_id][exp_iclass].jobExtent);

			sumBlockNum+=block_nums[nr_fake_classes*img_id + fake_class];

			RFLOAT myprior_x, myprior_y, myprior_z, old_offset_z;
			RFLOAT old_offset_x = XX(op.old_offset[img_id]);
			RFLOAT old_offset_y = YY(op.old_offset[img_id]);

			if (baseMLO->mymodel.ref_dim == 2 && baseMLO->mymodel.nr_bodies == 1)
			{
				myprior_x = XX(baseMLO->mymodel.prior_offset_class[exp_iclass]);
				myprior_y = YY(baseMLO->mymodel.prior_offset_class[exp_iclass]);
			}
			else
			{
				myprior_x = XX(op.prior[img_id]);
				myprior_y = YY(op.prior[img_id]);
				if (baseMLO->mymodel.data_dim == 3)
				{
					myprior_z = ZZ(op.prior[img_id]);
					old_offset_z = ZZ(op.old_offset[img_id]);
				}
			}

			/*======================================================
								COLLECT 2
			======================================================*/

			//Pregenerate oversampled translation objects for kernel-call
			for (long int itrans = 0, iitrans = 0; itrans < sp.nr_trans; itrans++)
			{
				baseMLO->sampling.getTranslationsInPixel(itrans, baseMLO->adaptive_oversampling, my_pixel_size,
						oversampled_translations_x, oversampled_translations_y, oversampled_translations_z,
						(baseMLO->do_helical_refine) && (! baseMLO->ignore_helical_symmetry));
				for (long int iover_trans = 0; iover_trans < sp.nr_oversampled_trans; iover_trans++, iitrans++)
				{
					oo_otrans[otrans_x+fake_class*nr_transes+iitrans] = old_offset_x + oversampled_translations_x[iover_trans];
					oo_otrans[otrans_y+fake_class*nr_transes+iitrans] = old_offset_y + oversampled_translations_y[iover_trans];
					if (accMLO->dataIs3D)
						oo_otrans[otrans_z+fake_class*nr_transes+iitrans] = old_offset_z + oversampled_translations_z[iover_trans];

					// Calculate the vector length of myprior
					RFLOAT mypriors_len2 = myprior_x * myprior_x + myprior_y * myprior_y;
					if (accMLO->dataIs3D)
						mypriors_len2 += myprior_z * myprior_z;

					// If it is doing helical refinement AND Cartesian vector myprior has a length > 0, transform the vector to its helical coordinates
					if ( (baseMLO->do_helical_refine) && (! baseMLO->ignore_helical_symmetry) && (mypriors_len2 > 0.00001) )
					{

						RFLOAT rot_deg = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_ROT);

						RFLOAT tilt_deg = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_TILT);

						RFLOAT psi_deg = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_PSI);
						transformCartesianAndHelicalCoords(myprior_x, myprior_y, myprior_z, myprior_x, myprior_y, myprior_z, rot_deg, tilt_deg, psi_deg, (accMLO->dataIs3D) ? (3) : (2), CART_TO_HELICAL_COORDS);
					}

					if ( (! baseMLO->do_helical_refine) || (baseMLO->ignore_helical_symmetry) )
						RFLOAT diffx = myprior_x - oo_otrans[otrans_x+fake_class*nr_transes+iitrans];
					RFLOAT diffx = myprior_x - oo_otrans[otrans_x+fake_class*nr_transes+iitrans];
					RFLOAT diffy = myprior_y - oo_otrans[otrans_y+fake_class*nr_transes+iitrans];
					RFLOAT diffz = 0;
					if (accMLO->dataIs3D)
						diffz = myprior_z - (old_offset_z + oversampled_translations_z[iover_trans]);

					oo_otrans[otrans_x2y2z2+fake_class*nr_transes+iitrans] = diffx*diffx + diffy*diffy + diffz*diffz;
				}
			}
		}

		bundleSWS[img_id].cpToDevice();
		oo_otrans.cpToDevice();

		DEBUG_HANDLE_ERROR(cudaStreamSynchronize(cudaStreamPerThread));
		DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.cudaStreamPerTask));

		// here we introduce offsets for the clases in an array as it is more efficient to
		// copy one big array to/from GPU rather than four small arrays
		size_t offsetx_class = 0*(size_t)sumBlockNum;
		size_t offsety_class = 1*(size_t)sumBlockNum;
		size_t offsetz_class = 2*(size_t)sumBlockNum;
		size_t sigma2_offset = 3*(size_t)sumBlockNum;

#ifdef NEWMEM
		AccPtrNew<XFLOAT>                      p_weights = ptrFactory.make<XFLOAT>((size_t)sumBlockNum);
		AccPtrNew<XFLOAT> p_thr_wsum_prior_offsetxyz_class = ptrFactory.make<XFLOAT>((size_t)sumBlockNum*4);
#else
		AccPtr<XFLOAT>                      p_weights = ptrFactory.make<XFLOAT>((size_t)sumBlockNum);
		AccPtr<XFLOAT> p_thr_wsum_prior_offsetxyz_class = ptrFactory.make<XFLOAT>((size_t)sumBlockNum*4);
#endif

		p_weights.allAlloc();
		p_thr_wsum_prior_offsetxyz_class.allAlloc();
        
        // TEST
        // auto p_weights_test = ptrFactory.make<XFLOAT>((size_t)sumBlockNum);
        // auto p_thr_wsum_prior_offsetxyz_class_test = ptrFactory.make<XFLOAT>((size_t)sumBlockNum*4);
        // p_weights_test.allAlloc();
        // p_thr_wsum_prior_offsetxyz_class_test.allAlloc();
        // deviceInitValue<XFLOAT>(p_weights_test, 0, p_weights_test.getSize(), ctx.cudaStreamPerTask);
        deviceInitValue<XFLOAT>(p_weights, 0, p_weights.getSize(), ctx.cudaStreamPerTask);
        // deviceInitValue<XFLOAT>(p_thr_wsum_prior_offsetxyz_class_test, 0, p_thr_wsum_prior_offsetxyz_class_test.getSize(), ctx.cudaStreamPerTask);
        deviceInitValue<XFLOAT>(p_thr_wsum_prior_offsetxyz_class, 0, p_thr_wsum_prior_offsetxyz_class.getSize(), ctx.cudaStreamPerTask);
        // DEBUG_HANDLE_ERROR(cudaStreamSynchronize(cudaStreamPerThread));
		// DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.cudaStreamPerTask));
        // END TEST

		CTOC(accMLO->timer,"collect_data_2_pre_kernel");
		int partial_pos=0;

		for (long int exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
		{
			long int fake_class = exp_iclass-sp.iclass_min; // if we only have the third class to do, the third class will be the "first" we do, i.e. the "fake" first.
			if ((baseMLO->mymodel.pdf_class[exp_iclass] == 0.) || (ProjectionData[img_id].class_entries[exp_iclass] == 0) )
				continue;

			// Use the constructed mask to construct a partial class-specific input
#ifdef NEWMEM
			IndexedDataArrayNew thisClassFinePassWeights(FinePassWeights[img_id],FPCMasks[img_id][exp_iclass]);
#else
			IndexedDataArray thisClassFinePassWeights(FinePassWeights[img_id],FPCMasks[img_id][exp_iclass]);
#endif
			long int cpos=fake_class*nr_transes;
			int block_num = block_nums[nr_fake_classes*img_id + fake_class];

			// runCollect2jobs(block_num,
			// 			&(~oo_otrans)[otrans_x+cpos],          // otrans-size -> make const
			// 			&(~oo_otrans)[otrans_y+cpos],          // otrans-size -> make const
			// 			&(~oo_otrans)[otrans_z+cpos],          // otrans-size -> make const
			// 			&(~oo_otrans)[otrans_x2y2z2+cpos], // otrans-size -> make const
			// 			~thisClassFinePassWeights.weights,
			// 			(XFLOAT)op.significant_weight[img_id],
			// 			(XFLOAT)op.sum_weight[img_id],
			// 			sp.nr_trans,
			// 			sp.nr_oversampled_trans,
			// 			sp.nr_oversampled_rot,
			// 			oversamples,
			// 			(baseMLO->do_skip_align || baseMLO->do_skip_rotate ),
			// 			&(~p_weights)[partial_pos],
			// 			&(~p_thr_wsum_prior_offsetxyz_class)[offsetx_class+partial_pos],
			// 			&(~p_thr_wsum_prior_offsetxyz_class)[offsety_class+partial_pos],
			// 			&(~p_thr_wsum_prior_offsetxyz_class)[offsetz_class+partial_pos],
			// 			&(~p_thr_wsum_prior_offsetxyz_class)[sigma2_offset+partial_pos],
			// 			~thisClassFinePassWeights.rot_idx,
			// 			~thisClassFinePassWeights.trans_idx,
			// 			~FPCMasks[img_id][exp_iclass].jobOrigin,
			// 			~FPCMasks[img_id][exp_iclass].jobExtent,
			// 			accMLO->dataIs3D);
			// LAUNCH_PRIVATE_ERROR(cudaGetLastError(),accMLO->errorStatus);


            collect2jobs_block<64, 128, 4, 8>(
                blocks64x128[exp_iclass].getSize(),
                ~blocks64x128[exp_iclass],
                ~(TransRearrangedIndex),
                ~(OrientRearrangedIndex[exp_iclass]),
                &(~oo_otrans)[otrans_x+cpos],          // otrans-size -> make const
                &(~oo_otrans)[otrans_y+cpos],          // otrans-size -> make const
                &(~oo_otrans)[otrans_z+cpos],          // otrans-size -> make const
                &(~oo_otrans)[otrans_x2y2z2+cpos],     // otrans-size -> make const
                ~thisClassFinePassWeights.weights,
                (XFLOAT)op.significant_weight[img_id],
				(XFLOAT)op.sum_weight[img_id],
                &(~p_weights)[partial_pos],
                &(~p_thr_wsum_prior_offsetxyz_class)[offsetx_class+partial_pos],
                &(~p_thr_wsum_prior_offsetxyz_class)[offsety_class+partial_pos],
                &(~p_thr_wsum_prior_offsetxyz_class)[offsetz_class+partial_pos],
                &(~p_thr_wsum_prior_offsetxyz_class)[sigma2_offset+partial_pos],
                accMLO->dataIs3D,
                ctx.cudaStreamPerTask
            );
        
			partial_pos+=block_num;
		}
		// XXX: runCollect2jobs() is a kernel call, but without a stream argument. nsys reports that it is called from the default stream.
		//      and it is simutaneously run with the following kernel call.(fixed)
		// DEBUG_HANDLE_ERROR(cudaStreamSynchronize(cudaStreamPerThread));

		CTIC(accMLO->timer,"collect_data_2_post_kernel");
		p_weights.cpToHost();
		p_thr_wsum_prior_offsetxyz_class.cpToHost();

		DEBUG_HANDLE_ERROR(cudaStreamSynchronize(cudaStreamPerThread));
		DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.cudaStreamPerTask));
		int iorient = 0;
		partial_pos=0;
		for (long int iclass = sp.iclass_min; iclass <= sp.iclass_max; iclass++)
		{
			long int fake_class = iclass-sp.iclass_min; // if we only have the third class to do, the third class will be the "first" we do, i.e. the "fake" first.
			if ((baseMLO->mymodel.pdf_class[iclass] == 0.) || (ProjectionData[img_id].class_entries[iclass] == 0) )
				continue;
			int block_num = block_nums[nr_fake_classes*img_id + fake_class];

			for (long int n = partial_pos; n < partial_pos+block_num; n++)
			{
				iorient= FinePassWeights[img_id].rot_id[FPCMasks[img_id][iclass].jobOrigin[n-partial_pos]+FPCMasks[img_id][iclass].firstPos];

				long int mydir, idir=floor(iorient/sp.nr_psi);
				if (baseMLO->mymodel.orientational_prior_mode == NOPRIOR)
					mydir = idir;
				else
					mydir = op.pointer_dir_nonzeroprior[idir];

				// store partials according to indices of the relevant dimension
				unsigned ithr_wsum_pdf_direction = baseMLO->mymodel.nr_bodies > 1 ? ibody : iclass;
				DIRECT_MULTIDIM_ELEM(thr_wsum_pdf_direction[ithr_wsum_pdf_direction], mydir) += p_weights[n];
				thr_sumw_group[img_id]                                                       += p_weights[n];
				thr_wsum_pdf_class[iclass]                                                   += p_weights[n];

				thr_wsum_sigma2_offset                                                       += my_pixel_size * my_pixel_size * p_thr_wsum_prior_offsetxyz_class[sigma2_offset+n];

				if (baseMLO->mymodel.ref_dim == 2)
				{
					thr_wsum_prior_offsetx_class[iclass] += my_pixel_size * p_thr_wsum_prior_offsetxyz_class[offsetx_class+n];
					thr_wsum_prior_offsety_class[iclass] += my_pixel_size * p_thr_wsum_prior_offsetxyz_class[offsety_class+n];
				}
			}
			partial_pos+=block_num;
		} // end loop iclass
		CTOC(accMLO->timer,"collect_data_2_post_kernel");
	} // end loop img_id

	/*======================================================
	                     SET METADATA
	======================================================*/

	std::vector< RFLOAT> oversampled_rot, oversampled_tilt, oversampled_psi;
	for (long int img_id = 0; img_id < sp.nr_images; img_id++)
	{

		int my_metadata_offset = op.metadata_offset + img_id;
		RFLOAT my_pixel_size = baseMLO->mydata.getImagePixelSize(op.part_id, img_id);

		CTIC(accMLO->timer,"setMetadata");

		if(baseMLO->adaptive_oversampling!=0)
			op.max_index[img_id].fineIndexToFineIndices(sp); // set partial indices corresponding to the found max_index, to be used below
		else
			op.max_index[img_id].coarseIndexToCoarseIndices(sp);

		baseMLO->sampling.getTranslationsInPixel(op.max_index[img_id].itrans, baseMLO->adaptive_oversampling, my_pixel_size,
				oversampled_translations_x, oversampled_translations_y, oversampled_translations_z,
				(baseMLO->do_helical_refine) && (! baseMLO->ignore_helical_symmetry));

		//TODO We already have rot, tilt and psi don't calculated them again
		if(baseMLO->do_skip_align || baseMLO->do_skip_rotate)
			   baseMLO->sampling.getOrientations(sp.idir_min, sp.ipsi_min, baseMLO->adaptive_oversampling, oversampled_rot, oversampled_tilt, oversampled_psi,
					   op.pointer_dir_nonzeroprior, op.directions_prior, op.pointer_psi_nonzeroprior, op.psi_prior);
		else
			   baseMLO->sampling.getOrientations(op.max_index[img_id].idir, op.max_index[img_id].ipsi, baseMLO->adaptive_oversampling, oversampled_rot, oversampled_tilt, oversampled_psi,
					op.pointer_dir_nonzeroprior, op.directions_prior, op.pointer_psi_nonzeroprior, op.psi_prior);

		baseMLO->sampling.getOrientations(op.max_index[img_id].idir, op.max_index[img_id].ipsi, baseMLO->adaptive_oversampling, oversampled_rot, oversampled_tilt, oversampled_psi,
				op.pointer_dir_nonzeroprior, op.directions_prior, op.pointer_psi_nonzeroprior, op.psi_prior);

		RFLOAT rot = oversampled_rot[op.max_index[img_id].ioverrot];
		RFLOAT tilt = oversampled_tilt[op.max_index[img_id].ioverrot];
		RFLOAT psi = oversampled_psi[op.max_index[img_id].ioverrot];

		int icol_rot  = (baseMLO->mymodel.nr_bodies == 1) ? METADATA_ROT  : 0 + METADATA_LINE_LENGTH_BEFORE_BODIES + (ibody) * METADATA_NR_BODY_PARAMS;
		int icol_tilt = (baseMLO->mymodel.nr_bodies == 1) ? METADATA_TILT : 1 + METADATA_LINE_LENGTH_BEFORE_BODIES + (ibody) * METADATA_NR_BODY_PARAMS;
		int icol_psi  = (baseMLO->mymodel.nr_bodies == 1) ? METADATA_PSI  : 2 + METADATA_LINE_LENGTH_BEFORE_BODIES + (ibody) * METADATA_NR_BODY_PARAMS;
		int icol_xoff = (baseMLO->mymodel.nr_bodies == 1) ? METADATA_XOFF : 3 + METADATA_LINE_LENGTH_BEFORE_BODIES + (ibody) * METADATA_NR_BODY_PARAMS;
		int icol_yoff = (baseMLO->mymodel.nr_bodies == 1) ? METADATA_YOFF : 4 + METADATA_LINE_LENGTH_BEFORE_BODIES + (ibody) * METADATA_NR_BODY_PARAMS;
		int icol_zoff = (baseMLO->mymodel.nr_bodies == 1) ? METADATA_ZOFF : 5 + METADATA_LINE_LENGTH_BEFORE_BODIES + (ibody) * METADATA_NR_BODY_PARAMS;


		RFLOAT old_rot = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, icol_rot);

		DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, icol_rot) = rot;

		RFLOAT old_tilt = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, icol_tilt);

		DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, icol_tilt) = tilt;

		RFLOAT old_psi = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, icol_psi);

		DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, icol_psi) = psi;

		Matrix1D<RFLOAT> shifts(baseMLO->mymodel.data_dim);

		XX(shifts) = XX(op.old_offset[img_id]) + oversampled_translations_x[op.max_index[img_id].iovertrans];
		YY(shifts) = YY(op.old_offset[img_id]) + oversampled_translations_y[op.max_index[img_id].iovertrans];
		if (accMLO->dataIs3D)
		{
			ZZ(shifts) = ZZ(op.old_offset[img_id]) + oversampled_translations_z[op.max_index[img_id].iovertrans];
		}

		// Use oldpsi-angle to rotate back the XX(exp_old_offset[img_id]) + oversampled_translations_x[iover_trans] and
		if ( (baseMLO->do_helical_refine) && (! baseMLO->ignore_helical_symmetry) )
			transformCartesianAndHelicalCoords(shifts, shifts, old_rot, old_tilt, old_psi, HELICAL_TO_CART_COORDS);


		DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, icol_xoff) = XX(shifts);

		DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, icol_yoff) = YY(shifts);
		if (accMLO->dataIs3D)

			DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, icol_zoff) = ZZ(shifts);

		if (ibody == 0)
		{

			DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_CLASS) = (RFLOAT)op.max_index[img_id].iclass + 1;
			RFLOAT pmax = op.max_weight[img_id]/op.sum_weight[img_id];
			if(pmax>1) //maximum normalised probability weight is (unreasonably) larger than unity
				CRITICAL("Relion is finding a normalised probability greater than 1");

			DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_PMAX) = pmax;
		}
		CTOC(accMLO->timer,"setMetadata");
	}
	CTOC(accMLO->timer,"collect_data_2");
    CTOC(accMLO->timer,"storeWeightedSumsCollectData");

}

template<class MlClass>
void storeWeightedSumsMaximizationPerImgSetup(Context<MlClass> &ctx)
{
    OptimisationParamters& op = ctx.op;
	SamplingParameters& sp    = ctx.sp;
    MlOptimiser* baseMLO      = ctx.baseMLO;
	MlClass *accMLO           = ctx.accMLO;

#ifdef NEWMEM
    std::vector<IndexedDataArrayNew>& FinePassWeights = *ctx.FinePassWeights;
    // std::vector<IndexedDataArray>& FinePassWeights = *ctx.FinePassWeights;

    std::vector<ProjectionParams>& ProjectionData = *ctx.FineProjectionData;
    std::vector<std::vector<IndexedDataArrayMask > >& FPCMasks = *ctx.FinePassClassMasks;
    AccPtrFactoryNew& ptrFactory = ctx.ptrFactory;
    int ibody = ctx.ibody;
#else
    std::vector<IndexedDataArray>& FinePassWeights = *ctx.FinePassWeights;
    std::vector<ProjectionParams>& ProjectionData = *ctx.FineProjectionData;
    std::vector<std::vector<IndexedDataArrayMask > >& FPCMasks = *ctx.FinePassClassMasks;
    AccPtrFactory& ptrFactory = ctx.ptrFactory;
    int ibody = ctx.ibody;
#endif
    std::vector<RFLOAT>& oversampled_translations_x = *ctx.oversampled_translations_x;
    std::vector<RFLOAT>& oversampled_translations_y = *ctx.oversampled_translations_y;
    std::vector<RFLOAT>& oversampled_translations_z = *ctx.oversampled_translations_z;
    bool have_warned_small_scale = false;

    CTIC(accMLO->timer,"storeWeightedSumsMaximizationPerImgSetup");

	/*=======================================================================================
	                                   MAXIMIZATION
	=======================================================================================*/
    int& img_id = ctx.img_id;
    // int my_metadata_offset = op.metadata_offset + img_id;
    // int group_id = baseMLO->mydata.getGroupId(op.part_id, img_id);
    // const int optics_group = baseMLO->mydata.getOpticsGroup(op.part_id, img_id);
    // RFLOAT my_pixel_size = baseMLO->mydata.getImagePixelSize(op.part_id, img_id);
    // bool ctf_premultiplied = baseMLO->mydata.obsModel.getCtfPremultiplied(optics_group);

    ctx.my_metadata_offset = op.metadata_offset + img_id;
    int& my_metadata_offset = ctx.my_metadata_offset;
    ctx.group_id = baseMLO->mydata.getGroupId(op.part_id, img_id);
    int& group_id = ctx.group_id;
    ctx.optics_group = baseMLO->mydata.getOpticsGroup(op.part_id, img_id);
    const int& optics_group = ctx.optics_group;
    ctx.my_pixel_size = baseMLO->mydata.getImagePixelSize(op.part_id, img_id);
    RFLOAT& my_pixel_size = ctx.my_pixel_size;
    ctx.ctf_premultiplied = baseMLO->mydata.obsModel.getCtfPremultiplied(optics_group);    
    bool& ctf_premultiplied = ctx.ctf_premultiplied;

    /*======================================================
                            TRANSLATIONS
    ======================================================*/

    // long unsigned translation_num((sp.itrans_max - sp.itrans_min + 1) * sp.nr_oversampled_trans);
    ctx.translation_num = (sp.itrans_max - sp.itrans_min + 1) * sp.nr_oversampled_trans;
    long unsigned& translation_num = ctx.translation_num;

    // size_t trans_x_offset = 0*(size_t)translation_num;
    // size_t trans_y_offset = 1*(size_t)translation_num;
    // size_t trans_z_offset = 2*(size_t)translation_num;

    ctx.trans_x_offset = 0*(size_t)translation_num;
    ctx.trans_y_offset = 1*(size_t)translation_num;
    ctx.trans_z_offset = 2*(size_t)translation_num;
    size_t& trans_x_offset = ctx.trans_x_offset;
    size_t& trans_y_offset = ctx.trans_y_offset;
    size_t& trans_z_offset = ctx.trans_z_offset;
    // AccPtr<XFLOAT> trans_xyz = ptrFactory.make<XFLOAT>((size_t)translation_num*3);
    
    // assert(ctx.trans_xyz.doFreeHost == true); // NO USE
    ctx.trans_xyz.freeIfSet();
    ctx.trans_xyz = ptrFactory.make<XFLOAT>((size_t)translation_num*3);
#ifdef NEWMEM
    AccPtrNew<XFLOAT>& trans_xyz = ctx.trans_xyz;
#else
    AccPtr<XFLOAT>& trans_xyz = ctx.trans_xyz;
#endif
    trans_xyz.allAlloc();


    int j = 0;
    for (long int itrans = 0; itrans < (sp.itrans_max - sp.itrans_min + 1); itrans++)
    {
        //TODO Called multiple time to generate same list, reuse the same list
        baseMLO->sampling.getTranslationsInPixel(itrans, baseMLO->adaptive_oversampling, my_pixel_size,
                oversampled_translations_x, oversampled_translations_y, oversampled_translations_z,
                (baseMLO->do_helical_refine) && (! baseMLO->ignore_helical_symmetry));

        for (long int iover_trans = 0; iover_trans < oversampled_translations_x.size(); iover_trans++)
        {
            RFLOAT xshift = 0., yshift = 0., zshift = 0.;

            xshift = oversampled_translations_x[iover_trans];
            yshift = oversampled_translations_y[iover_trans];
            if (accMLO->dataIs3D)
                zshift = oversampled_translations_z[iover_trans];

            if ( (baseMLO->do_helical_refine) && (! baseMLO->ignore_helical_symmetry) )
            {

                RFLOAT rot_deg = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_ROT);

                RFLOAT tilt_deg = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_TILT);

                RFLOAT psi_deg = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_PSI);
                transformCartesianAndHelicalCoords(xshift, yshift, zshift, xshift, yshift, zshift, rot_deg, tilt_deg, psi_deg, (accMLO->dataIs3D) ? (3) : (2), HELICAL_TO_CART_COORDS);
            }

            trans_xyz[trans_x_offset+j] = -2 * PI * xshift / (double)baseMLO->image_full_size[optics_group];
            trans_xyz[trans_y_offset+j] = -2 * PI * yshift / (double)baseMLO->image_full_size[optics_group];
            trans_xyz[trans_z_offset+j] = -2 * PI * zshift / (double)baseMLO->image_full_size[optics_group];
            j ++;
        }
    }

    trans_xyz.cpToDevice();


    /*======================================================
                            IMAGES
    ======================================================*/

    CUSTOM_ALLOCATOR_REGION_NAME("TRANS_3");

    CTIC(accMLO->timer,"translation_3");


    MultidimArray<Complex > Fimg, Fimg_nonmask;
    windowFourierTransform(op.Fimg[img_id], Fimg, baseMLO->image_current_size[optics_group]); //TODO PO isen't this already done in getFourierTransformsAndCtfs?
    windowFourierTransform(op.Fimg_nomask[img_id], Fimg_nonmask, baseMLO->image_current_size[optics_group]);
    // unsigned long image_size = Fimg.nzyxdim;
    ctx.image_size = Fimg.nzyxdim;
    unsigned long& image_size = ctx.image_size;

    size_t re_offset = 0*(size_t)image_size;
    size_t im_offset = 1*(size_t)image_size;
    size_t re_nomask_offset = 2*(size_t)image_size;
    size_t im_nomask_offset = 3*(size_t)image_size;

    // AccPtr<XFLOAT> Fimgs = ptrFactory.make<XFLOAT>(4*(size_t)image_size);

    ctx.Fimgs = ptrFactory.make<XFLOAT>(4*(size_t)image_size);
#ifdef NEWMEM
    AccPtrNew<XFLOAT>& Fimgs = ctx.Fimgs;
#else
    AccPtr<XFLOAT>& Fimgs = ctx.Fimgs;
#endif
    Fimgs.allAlloc();

    for (unsigned long i = 0; i < image_size; i ++)
    {
        Fimgs[re_offset+i] = Fimg.data[i].real;
        Fimgs[im_offset+i] = Fimg.data[i].imag;
        Fimgs[re_nomask_offset+i] = Fimg_nonmask.data[i].real;
        Fimgs[im_nomask_offset+i] = Fimg_nonmask.data[i].imag;
    }

    Fimgs.cpToDevice();

    CTOC(accMLO->timer,"translation_3");


    /*======================================================
                            SCALE
    ======================================================*/

    // XFLOAT part_scale(1.);
    ctx.part_scale = 1.;
    XFLOAT& part_scale = ctx.part_scale;

    if (baseMLO->do_scale_correction)
    {
        part_scale = baseMLO->mymodel.scale_correction[group_id];
        if (part_scale > 10000.)
        {
            std::cerr << " rlnMicrographScaleCorrection= " << part_scale << " group= " << group_id + 1 << std::endl;
            CRITICAL(ERRHIGHSCALE);
        }
        else if (part_scale < 0.001)
        {
            if (!have_warned_small_scale)
            {
                std::cout << " WARNING: ignoring group " << group_id + 1 << " with very small or negative scale (" << part_scale <<
                        "); Use larger groups for more stable scale estimates." << std::endl;
                have_warned_small_scale = true;
            }
            part_scale = 0.001;
        }
    }

    // AccPtr<XFLOAT> ctfs = ptrFactory.make<XFLOAT>((size_t)image_size);
    ctx.ctfs = ptrFactory.make<XFLOAT>((size_t)image_size);
#ifdef NEWMEM
    AccPtrNew<XFLOAT>& ctfs = ctx.ctfs;
#else
    AccPtr<XFLOAT>& ctfs = ctx.ctfs;
#endif
    ctfs.allAlloc();

    if (baseMLO->do_ctf_correction)
    {
        for (unsigned long i = 0; i < image_size; i++)
            ctfs[i] = (XFLOAT) op.local_Fctf[img_id].data[i] * part_scale;
    }
    else //TODO should be handled by memset
    {
        for (unsigned long i = 0; i < image_size; i++)
            ctfs[i] = part_scale;
    }

    ctfs.cpToDevice();

    /*======================================================
                            MINVSIGMA
    ======================================================*/

    // AccPtr<XFLOAT> Minvsigma2s = ptrFactory.make<XFLOAT>((size_t)image_size);
    ctx.Minvsigma2s = ptrFactory.make<XFLOAT>((size_t)image_size);
#ifdef NEWMEM
    AccPtrNew<XFLOAT>& Minvsigma2s = ctx.Minvsigma2s;
#else
    AccPtr<XFLOAT>& Minvsigma2s = ctx.Minvsigma2s;
#endif
    Minvsigma2s.allAlloc();

    if (baseMLO->do_map)
        for (unsigned long i = 0; i < image_size; i++)
            Minvsigma2s[i] = op.local_Minvsigma2[img_id].data[i];
    else
        for (unsigned long i = 0; i < image_size; i++)
            Minvsigma2s[i] = 1;

    Minvsigma2s.cpToDevice();

    /*======================================================
                            CLASS LOOP
    ======================================================*/

    CUSTOM_ALLOCATOR_REGION_NAME("wdiff2s");

    size_t wdiff2s_buf = (size_t)(baseMLO->mymodel.nr_classes*image_size)*2+(size_t)image_size;
    // size_t AA_offset =  0*(size_t)(baseMLO->mymodel.nr_classes*image_size);
    // size_t XA_offset =  1*(size_t)(baseMLO->mymodel.nr_classes*image_size);
    // size_t sum_offset = 2*(size_t)(baseMLO->mymodel.nr_classes*image_size);

    // AccPtr<XFLOAT> wdiff2s    = ptrFactory.make<XFLOAT>(wdiff2s_buf);
    ctx.wdiff2s    = ptrFactory.make<XFLOAT>(wdiff2s_buf);
#ifdef NEWMEM
    AccPtrNew<XFLOAT>& wdiff2s    = ctx.wdiff2s;
#else
    AccPtr<XFLOAT>& wdiff2s    = ctx.wdiff2s;
#endif
    wdiff2s.allAlloc();
    wdiff2s.accInit(0);

    // unsigned long AAXA_pos=0;

    CUSTOM_ALLOCATOR_REGION_NAME("BP_data");

    // Loop from iclass_min to iclass_max to deal with seed generation in first iteration
    // AccPtr<XFLOAT> sorted_weights = ptrFactory.make<XFLOAT>((size_t)(ProjectionData[img_id].orientationNumAllClasses * translation_num));
    ctx.sorted_weights = ptrFactory.make<XFLOAT>((size_t)(ProjectionData[img_id].orientationNumAllClasses * translation_num));
#ifdef NEWMEM
    AccPtrNew<XFLOAT>& sorted_weights = ctx.sorted_weights;
    sorted_weights.allAlloc();
    ctx.eulers = new std::vector< AccPtrNew<XFLOAT> >(baseMLO->mymodel.nr_classes, ptrFactory.make<XFLOAT>());
    std::vector< AccPtrNew<XFLOAT> > &eulers = *ctx.eulers;
#else
    AccPtr<XFLOAT>& sorted_weights = ctx.sorted_weights;
    sorted_weights.allAlloc();
    // std::vector<AccPtr<XFLOAT> > eulers(baseMLO->mymodel.nr_classes, ptrFactory.make<XFLOAT>());
    ctx.eulers = new std::vector< AccPtr<XFLOAT> >(baseMLO->mymodel.nr_classes, ptrFactory.make<XFLOAT>());
    std::vector< AccPtr<XFLOAT> > &eulers = *ctx.eulers;
#endif

    unsigned long classPos = 0;

    for (unsigned long exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
        DEBUG_HANDLE_ERROR(cudaStreamSynchronize(accMLO->classStreams[exp_iclass]));
    DEBUG_HANDLE_ERROR(cudaStreamSynchronize(cudaStreamPerThread));

    for (unsigned long exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
        DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.classStreams[exp_iclass]));
    DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.cudaStreamPerTask));

    for (unsigned long iclass = sp.iclass_min; iclass <= sp.iclass_max; iclass++)
    {
        if((baseMLO->mymodel.pdf_class[iclass] == 0.) || (ProjectionData[img_id].class_entries[iclass] == 0))
            continue;

#ifdef NEWMEM
        // Use the constructed mask to construct a partial class-specific input
        IndexedDataArrayNew thisClassFinePassWeights(FinePassWeights[img_id],FPCMasks[img_id][iclass]);
#else
        IndexedDataArray thisClassFinePassWeights(FinePassWeights[img_id],FPCMasks[img_id][iclass]);
#endif


        CTIC(accMLO->timer,"thisClassProjectionSetupCoarse");
        // use "slice" constructor with class-specific parameters to retrieve a temporary ProjectionParams with data for this class
        ProjectionParams thisClassProjectionData(	ProjectionData[img_id],
                                                    ProjectionData[img_id].class_idx[iclass],
                                                    ProjectionData[img_id].class_idx[iclass]+ProjectionData[img_id].class_entries[iclass]);

        thisClassProjectionData.orientation_num[0] = ProjectionData[img_id].orientation_num[iclass];
        CTOC(accMLO->timer,"thisClassProjectionSetupCoarse");

        long unsigned orientation_num(thisClassProjectionData.orientation_num[0]);

        /*======================================================
                            PROJECTIONS
        ======================================================*/


        Matrix2D<RFLOAT> MBL, MBR;

        if (baseMLO->mymodel.nr_bodies > 1)
        {
            Matrix2D<RFLOAT> Aori;
            RFLOAT rot_ori = DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset, METADATA_ROT);
            RFLOAT tilt_ori = DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset, METADATA_TILT);
            RFLOAT psi_ori = DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset, METADATA_PSI);
            Euler_angles2matrix(rot_ori, tilt_ori, psi_ori, Aori, false);

            MBL = Aori * (baseMLO->mymodel.orient_bodies[ibody]).transpose() * baseMLO->A_rot90;
            MBR = baseMLO->mymodel.orient_bodies[ibody];
        }

        eulers[iclass].setSize(orientation_num * 9);
        // eulers[iclass].setStream(accMLO->classStreams[iclass]);
        eulers[iclass].setStream(ctx.classStreams[iclass]);
        eulers[iclass].hostAlloc();

        CTIC(accMLO->timer,"generateEulerMatricesProjector");

        Matrix2D<RFLOAT> mag;
        mag.initIdentity(3);
        mag = baseMLO->mydata.obsModel.applyAnisoMag(mag, optics_group);
        mag = baseMLO->mydata.obsModel.applyScaleDifference(mag, optics_group, baseMLO->mymodel.ori_size, baseMLO->mymodel.pixel_size);
        if (!mag.isIdentity())
        {
            if (MBL.mdimx == 3 && MBL.mdimx ==3) MBL = mag * MBL;
            else MBL = mag;
        }

        generateEulerMatrices(
                thisClassProjectionData,
                &eulers[iclass][0],
                true,
                MBL,
                MBR);

        eulers[iclass].deviceAlloc();
        eulers[iclass].cpToDevice();

        CTOC(accMLO->timer,"generateEulerMatricesProjector");

        /*======================================================
                                MAP WEIGHTS
        ======================================================*/

        CTIC(accMLO->timer,"pre_wavg_map");

        for (long unsigned i = 0; i < orientation_num*translation_num; i++)
            sorted_weights[classPos+i] = -std::numeric_limits<XFLOAT>::max();

        for (long unsigned i = 0; i < thisClassFinePassWeights.weights.getSize(); i++)
            sorted_weights[classPos+(thisClassFinePassWeights.rot_idx[i]) * translation_num + thisClassFinePassWeights.trans_idx[i] ]
                            = thisClassFinePassWeights.weights[i];

        classPos+=orientation_num*translation_num;
        CTOC(accMLO->timer,"pre_wavg_map");
    }
    sorted_weights.cpToDevice();

    // These syncs are necessary (for multiple ranks on the same GPU), and (assumed) low-cost.
    for (unsigned long iclass = sp.iclass_min; iclass <= sp.iclass_max; iclass++)
        DEBUG_HANDLE_ERROR(cudaStreamSynchronize(accMLO->classStreams[iclass]));

    DEBUG_HANDLE_ERROR(cudaStreamSynchronize(cudaStreamPerThread));

    for (unsigned long exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
        DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.classStreams[exp_iclass]));
    DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.cudaStreamPerTask));

    CTOC(accMLO->timer,"storeWeightedSumsMaximizationPerImgSetup");


}

template<class MlClass>
void storeWeightedSumsMaximizationPerImgLaunchKernel(Context<MlClass> &ctx) {
    OptimisationParamters& op = ctx.op;
	SamplingParameters& sp    = ctx.sp;
    MlOptimiser* baseMLO      = ctx.baseMLO;
	MlClass *accMLO           = ctx.accMLO;
    int ibody = ctx.ibody;
    int thread_id = ctx.thread_id;

#ifdef NEWMEM
    std::vector<IndexedDataArrayNew>& FinePassWeights = *ctx.FinePassWeights;
    // std::vector<IndexedDataArray>& FinePassWeights = *ctx.FinePassWeights;

    std::vector<ProjectionParams>& ProjectionData = *ctx.FineProjectionData;
    std::vector<std::vector<IndexedDataArrayMask > >& FPCMasks = *ctx.FinePassClassMasks;
    std::vector<RFLOAT>& thr_wsum_pdf_class = *ctx.thr_wsum_pdf_class;

    unsigned long& image_size = ctx.image_size;
    long unsigned& translation_num = ctx.translation_num;
    int& img_id = ctx.img_id;
    std::vector< AccPtrNew<XFLOAT> > &eulers = *ctx.eulers;
    AccPtrNew<XFLOAT>& Fimgs = ctx.Fimgs;
    AccPtrNew<XFLOAT>& trans_xyz = ctx.trans_xyz;
    AccPtrNew<XFLOAT>& sorted_weights = ctx.sorted_weights;
    AccPtrNew<XFLOAT>& ctfs = ctx.ctfs;
    AccPtrNew<XFLOAT>& Minvsigma2s = ctx.Minvsigma2s;
    AccPtrNew<XFLOAT>& wdiff2s    = ctx.wdiff2s;
#else
    std::vector<IndexedDataArray>& FinePassWeights = *ctx.FinePassWeights;
    std::vector<ProjectionParams>& ProjectionData = *ctx.FineProjectionData;
    std::vector<std::vector<IndexedDataArrayMask > >& FPCMasks = *ctx.FinePassClassMasks;
    std::vector<RFLOAT>& thr_wsum_pdf_class = *ctx.thr_wsum_pdf_class;

    unsigned long& image_size = ctx.image_size;
    long unsigned& translation_num = ctx.translation_num;
    int& img_id = ctx.img_id;
    std::vector< AccPtr<XFLOAT> > &eulers = *ctx.eulers;
    AccPtr<XFLOAT>& Fimgs = ctx.Fimgs;
    AccPtr<XFLOAT>& trans_xyz = ctx.trans_xyz;
    AccPtr<XFLOAT>& sorted_weights = ctx.sorted_weights;
    AccPtr<XFLOAT>& ctfs = ctx.ctfs;
    AccPtr<XFLOAT>& Minvsigma2s = ctx.Minvsigma2s;
    AccPtr<XFLOAT>& wdiff2s    = ctx.wdiff2s;
#endif

    size_t& trans_x_offset = ctx.trans_x_offset;
    size_t& trans_y_offset = ctx.trans_y_offset;
    size_t& trans_z_offset = ctx.trans_z_offset;

    CTIC(accMLO->timer,"storeWeightedSumsMaximizationPerImgLaunchKernel");

    size_t re_offset = 0*(size_t)image_size;
    size_t im_offset = 1*(size_t)image_size;
    size_t re_nomask_offset = 2*(size_t)image_size;
    size_t im_nomask_offset = 3*(size_t)image_size;

    size_t AA_offset =  0*(size_t)(baseMLO->mymodel.nr_classes*image_size);
    size_t XA_offset =  1*(size_t)(baseMLO->mymodel.nr_classes*image_size);
    size_t sum_offset = 2*(size_t)(baseMLO->mymodel.nr_classes*image_size);
    unsigned long AAXA_pos=0;

    int& my_metadata_offset = ctx.my_metadata_offset;
    int& group_id = ctx.group_id;
    const int& optics_group = ctx.optics_group;
    RFLOAT& my_pixel_size = ctx.my_pixel_size;
    bool& ctf_premultiplied = ctx.ctf_premultiplied;

    XFLOAT& part_scale = ctx.part_scale;
    
    int classPos = 0;
    for (unsigned long iclass = sp.iclass_min; iclass <= sp.iclass_max; iclass++)
    {
        int iproj;
        if (baseMLO->mymodel.nr_bodies > 1) iproj = ibody;
        else                                iproj = iclass;

        if((baseMLO->mymodel.pdf_class[iclass] == 0.) || (ProjectionData[img_id].class_entries[iclass] == 0))
            continue;
        /*======================================================
                                KERNEL CALL
        ======================================================*/

        long unsigned orientation_num(ProjectionData[img_id].orientation_num[iclass]);

        AccProjectorKernel projKernel = AccProjectorKernel::makeKernel(
                accMLO->bundle->projectors[iproj],
                op.local_Minvsigma2[img_id].xdim,
                op.local_Minvsigma2[img_id].ydim,
                op.local_Minvsigma2[img_id].zdim,
                op.local_Minvsigma2[img_id].xdim-1);
#ifdef TIMING
if(ctx.thread_id==0)
baseMLO->timer.tic(baseMLO->TIMING_EXTRA3);
// baseMLO->timer.tic(baseMLO->TIMING_EXTRA3_T[thread_id]);
#endif
        // printf("_xjldebug op.sumweiht,op.sigweith:%e %e\n",op.sum_weight[img_id],op.significant_weight[img_id]);
        runWavgKernel(
                projKernel,
                ~eulers[iclass],
                &(~Fimgs)[re_offset], //~Fimgs_real,
                &(~Fimgs)[im_offset], //~Fimgs_imag,
                &(~trans_xyz)[trans_x_offset], //~trans_x,
                &(~trans_xyz)[trans_y_offset], //~trans_y,
                &(~trans_xyz)[trans_z_offset], //~trans_z,
                &(~sorted_weights)[classPos],
                ~ctfs,
                &(~wdiff2s)[sum_offset],
                &(~wdiff2s)[AA_offset+AAXA_pos],
                &(~wdiff2s)[XA_offset+AAXA_pos],
                op,
                orientation_num,
                translation_num,
                image_size,
                img_id,
                group_id,
                iclass,
                part_scale,
                baseMLO->refs_are_ctf_corrected,
                ctf_premultiplied,
                accMLO->dataIs3D,
                // accMLO->classStreams[iclass]
                ctx.classStreams[iclass]
                );
#ifdef TIMING
if(ctx.thread_id==0)
baseMLO->timer.toc(baseMLO->TIMING_EXTRA3);
// baseMLO->timer.toc(baseMLO->TIMING_EXTRA3_T[thread_id]);
#endif
        AAXA_pos += image_size;
        classPos += orientation_num*translation_num;
    }

    /*======================================================
                                SOM
    ======================================================*/

    int nr_classes = baseMLO->mymodel.nr_classes;
    std::vector<RFLOAT> class_sum_weight(nr_classes, baseMLO->is_som_iter ? 0 : op.sum_weight[img_id]);

    if (baseMLO->is_som_iter)
    {
        std::vector<unsigned> s = SomGraph::arg_sort(op.sum_weight_class[img_id], false);
        unsigned bpu = s[0];
        unsigned sbpu = s[1];

        baseMLO->wsum_model.som.add_edge_activity(bpu, sbpu);

        class_sum_weight[bpu] = op.sum_weight_class[img_id][bpu];
        thr_wsum_pdf_class[bpu] += 1;
        baseMLO->wsum_model.som.add_node_activity(bpu);
        baseMLO->mymodel.som.add_node_age(bpu);

        std::vector<std::pair<unsigned, float> > weights = baseMLO->mymodel.som.get_neighbours(bpu);

        for (int i = 0; i < weights.size(); i++) {
            unsigned idx = weights[i].first;
            float w = weights[i].second * baseMLO->som_neighbour_pull;
            class_sum_weight[idx] = op.sum_weight_class[img_id][idx] / w;
            thr_wsum_pdf_class[idx] += w;
            baseMLO->wsum_model.som.add_node_activity(idx, w);
            baseMLO->mymodel.som.add_node_age(idx, w);
        }
    }

    /*======================================================
                        BACKPROJECTION
    ======================================================*/

    classPos = 0;
    for (unsigned long iclass = sp.iclass_min; iclass <= sp.iclass_max; iclass++)
    {

        int iproj;
        if (baseMLO->mymodel.nr_bodies > 1) iproj = ibody;
        else                                iproj = iclass;

        if((baseMLO->mymodel.pdf_class[iclass] == 0.) || (ProjectionData[img_id].class_entries[iclass] == 0))
            continue;

        if ( baseMLO->is_som_iter && class_sum_weight[iclass] == 0)
            continue;

        long unsigned orientation_num(ProjectionData[img_id].orientation_num[iclass]);

        AccProjectorKernel projKernel = AccProjectorKernel::makeKernel(
                accMLO->bundle->projectors[iproj],
                op.local_Minvsigma2[img_id].xdim,
                op.local_Minvsigma2[img_id].ydim,
                op.local_Minvsigma2[img_id].zdim,
                op.local_Minvsigma2[img_id].xdim - 1);

#ifdef TIMING
        if (op.part_id == baseMLO->exp_my_first_part_id)
            baseMLO->timer.tic(baseMLO->TIMING_WSUM_BACKPROJ);
#endif

        // If doing pseudo gold standard select random half-model
        int iproj_offset = 0;
        if (baseMLO->grad_pseudo_halfsets)
            // Backproject every other particle into separate volumes
            iproj_offset = (op.part_id % 2) * baseMLO->mymodel.nr_classes;
#ifdef TIMING
if(ctx.thread_id==0)
baseMLO->timer.tic(baseMLO->TIMING_EXTRA4);
// baseMLO->timer.tic(baseMLO->TIMING_EXTRA4_T[thread_id]);
#endif
        CTIC(accMLO->timer,"backproject");
        runBackProjectKernel(
            accMLO->bundle->backprojectors[iproj + iproj_offset],
            projKernel,
            &(~Fimgs)[re_nomask_offset], //~Fimgs_nomask_real,
            &(~Fimgs)[im_nomask_offset], //~Fimgs_nomask_imag,
            &(~trans_xyz)[trans_x_offset], //~trans_x,
            &(~trans_xyz)[trans_y_offset], //~trans_y,
            &(~trans_xyz)[trans_z_offset], //~trans_z,
            &(~sorted_weights)[classPos],
            ~Minvsigma2s,
            ~ctfs,
            translation_num,
            (XFLOAT) op.significant_weight[img_id],
            (XFLOAT) (baseMLO->is_som_iter ? class_sum_weight[iclass] : op.sum_weight[img_id]),
            ~eulers[iclass],
            op.local_Minvsigma2[img_id].xdim,
            op.local_Minvsigma2[img_id].ydim,
            op.local_Minvsigma2[img_id].zdim,
            orientation_num,
            accMLO->dataIs3D,
            (baseMLO->do_grad),
            ctf_premultiplied,
            // accMLO->classStreams[iclass]
            ctx.classStreams[iclass]);
#ifdef TIMING
if(ctx.thread_id==0)
baseMLO->timer.toc(baseMLO->TIMING_EXTRA4);
// baseMLO->timer.toc(baseMLO->TIMING_EXTRA4_T[thread_id]);
#endif
        CTOC(accMLO->timer,"backproject");

#ifdef TIMING
        if (op.part_id == baseMLO->exp_my_first_part_id)
            baseMLO->timer.toc(baseMLO->TIMING_WSUM_BACKPROJ);
#endif

        //Update indices
        AAXA_pos += image_size;
        classPos += orientation_num*translation_num;

    } // end loop iclass

    CTOC(accMLO->timer,"storeWeightedSumsMaximizationPerImgLaunchKernel");

}

template<class MlClass>
void storeWeightedSumsMaximizationPerImgSync(Context<MlClass> &ctx) {
    OptimisationParamters& op = ctx.op;
	SamplingParameters& sp    = ctx.sp;
    MlOptimiser* baseMLO      = ctx.baseMLO;
	MlClass *accMLO           = ctx.accMLO;

    unsigned long& image_size = ctx.image_size;
    int& img_id = ctx.img_id;
    const int& optics_group = ctx.optics_group;

    std::vector<ProjectionParams>& ProjectionData = *ctx.FineProjectionData;

    std::vector<RFLOAT>& exp_wsum_norm_correction = *ctx.exp_wsum_norm_correction;
    std::vector<RFLOAT>& exp_wsum_scale_correction_XA = *ctx.exp_wsum_scale_correction_XA;
    std::vector<RFLOAT>& exp_wsum_scale_correction_AA = *ctx.exp_wsum_scale_correction_AA;
    std::vector<MultidimArray<RFLOAT> >& thr_wsum_sigma2_noise = *ctx.thr_wsum_sigma2_noise;
#ifdef NEWMEM
    AccPtrNew<XFLOAT>& wdiff2s    = ctx.wdiff2s;
#else
    AccPtr<XFLOAT>& wdiff2s    = ctx.wdiff2s;
#endif
    CTIC(accMLO->timer,"storeWeightedSumsMaximizationPerImgSync");

    size_t AA_offset =  0*(size_t)(baseMLO->mymodel.nr_classes*image_size);
    size_t XA_offset =  1*(size_t)(baseMLO->mymodel.nr_classes*image_size);
    size_t sum_offset = 2*(size_t)(baseMLO->mymodel.nr_classes*image_size);

    CUSTOM_ALLOCATOR_REGION_NAME("UNSET");

    // NOTE: We've never seen that this sync is necessary, but it is needed in principle, and
    // its absence in other parts of the code has caused issues. It is also very low-cost.
    for (unsigned long exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
        DEBUG_HANDLE_ERROR(cudaStreamSynchronize(accMLO->classStreams[exp_iclass]));
    DEBUG_HANDLE_ERROR(cudaStreamSynchronize(cudaStreamPerThread));

    for (unsigned long exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
        DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.classStreams[exp_iclass]));
    DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.cudaStreamPerTask));
    wdiff2s.cpToHost();
    DEBUG_HANDLE_ERROR(cudaStreamSynchronize(cudaStreamPerThread)); // NO_USE
    DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.cudaStreamPerTask));

    unsigned long AAXA_pos=0;
    // float debug1=0;
    for (unsigned long exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
    {
        if((baseMLO->mymodel.pdf_class[exp_iclass] == 0.) || (ProjectionData[img_id].class_entries[exp_iclass] == 0))
            continue;
        for (long int j = 0; j < image_size; j++)
        {
            int ires = DIRECT_MULTIDIM_ELEM(baseMLO->Mresol_fine[optics_group], j);
            if (ires > -1 && baseMLO->do_scale_correction &&
                    DIRECT_A1D_ELEM(baseMLO->mymodel.data_vs_prior_class[exp_iclass], ires) > 3.)
            {
                exp_wsum_scale_correction_AA[img_id] += wdiff2s[AA_offset+AAXA_pos+j];
                exp_wsum_scale_correction_XA[img_id] += wdiff2s[XA_offset+AAXA_pos+j];
            }
            // debug1+=wdiff2s[AA_offset+AAXA_pos+j];
        }
        AAXA_pos += image_size;
    } // end loop iclass


    for (unsigned long j = 0; j < image_size; j++)
    {
        int ires = DIRECT_MULTIDIM_ELEM(baseMLO->Mresol_fine[optics_group], j);
        if (ires > -1)
        {
            thr_wsum_sigma2_noise[img_id].data[ires] += (RFLOAT) wdiff2s[sum_offset+j];
            exp_wsum_norm_correction[img_id] += (RFLOAT) wdiff2s[sum_offset+j]; //TODO could be gpu-reduced
        }
    }

    for(int i=baseMLO->mymodel.nr_classes-1;i>=0;i--)
    {
        (*ctx.eulers)[i].freeIfSet();
    }
    delete ctx.eulers;
    ctx.sorted_weights.freeIfSet();
    ctx.wdiff2s.freeIfSet();
    ctx.Minvsigma2s.freeIfSet();
    ctx.ctfs.freeIfSet();
    ctx.Fimgs.freeIfSet();
    ctx.trans_xyz.freeIfSet();
    CTOC(accMLO->timer,"storeWeightedSumsMaximizationPerImgSync");

}

template<class MlClass>
void storeWeightedSumsStorePost(Context<MlClass> &ctx) {
    OptimisationParamters& op = ctx.op;
	SamplingParameters& sp    = ctx.sp;
    MlOptimiser* baseMLO      = ctx.baseMLO;
	MlClass *accMLO           = ctx.accMLO;

    std::vector<MultidimArray<RFLOAT> >& thr_wsum_pdf_direction = *ctx.thr_wsum_pdf_direction;
    std::vector<RFLOAT>& thr_sumw_group = *ctx.thr_sumw_group;
    std::vector<RFLOAT>& thr_wsum_pdf_class = *ctx.thr_wsum_pdf_class;
    std::vector<RFLOAT>& thr_wsum_prior_offsetx_class = *ctx.thr_wsum_prior_offsetx_class;
    std::vector<RFLOAT>& thr_wsum_prior_offsety_class = *ctx.thr_wsum_prior_offsety_class;
    RFLOAT& thr_wsum_sigma2_offset = ctx.thr_wsum_sigma2_offset;

    std::vector<RFLOAT>& exp_wsum_norm_correction = *ctx.exp_wsum_norm_correction;
    std::vector<RFLOAT>& exp_wsum_scale_correction_XA = *ctx.exp_wsum_scale_correction_XA;
    std::vector<RFLOAT>& exp_wsum_scale_correction_AA = *ctx.exp_wsum_scale_correction_AA;
    std::vector<RFLOAT>& thr_wsum_signal_product_spectra = *ctx.thr_wsum_signal_product_spectra;
    std::vector<RFLOAT>& thr_wsum_reference_power_spectra = *ctx.thr_wsum_reference_power_spectra;
    std::vector<MultidimArray<RFLOAT> >& thr_wsum_sigma2_noise = *ctx.thr_wsum_sigma2_noise;
    std::vector<MultidimArray<RFLOAT> >& thr_wsum_ctf2 = *ctx.thr_wsum_ctf2;
    std::vector<MultidimArray<RFLOAT> >& thr_wsum_stMulti = *ctx.thr_wsum_stMulti;

    bool& do_subtomo_correction = ctx.do_subtomo_correction;
    std::vector<MultidimArray<RFLOAT> >& exp_local_STMulti = *ctx.exp_local_STMulti;

    unsigned long& image_size = ctx.image_size;

    int& my_metadata_offset = ctx.my_metadata_offset;
    int& group_id = ctx.group_id;
    const int& optics_group = ctx.optics_group;
    RFLOAT& my_pixel_size = ctx.my_pixel_size;

    CTIC(accMLO->timer,"storeWeightedSumsStorePost");

    // Extend norm_correction and sigma2_noise estimation to higher resolutions for all particles
    // Also calculate dLL for each particle and store in metadata
    // loop over all images inside this particle
    RFLOAT thr_avg_norm_correction = 0.;
    RFLOAT thr_sum_dLL = 0., thr_sum_Pmax = 0.;
    for (int img_id = 0; img_id < sp.nr_images; img_id++)
    {


        int my_metadata_offset = op.metadata_offset + img_id;
        int group_id = baseMLO->mydata.getGroupId(op.part_id, img_id);
        const int optics_group = baseMLO->mydata.getOpticsGroup(op.part_id, img_id);
        RFLOAT my_pixel_size = baseMLO->mydata.getOpticsPixelSize(optics_group);
        int my_image_size = baseMLO->mydata.getOpticsImageSize(optics_group);

        // If the current images were smaller than the original size, fill the rest of wsum_model.sigma2_noise with the power_class spectrum of the images
        for (unsigned long ires = baseMLO->image_current_size[optics_group]/2 + 1; ires < baseMLO->image_full_size[optics_group]/2 + 1; ires++)
        {
            DIRECT_A1D_ELEM(thr_wsum_sigma2_noise[img_id], ires) += DIRECT_A1D_ELEM(op.power_img[img_id], ires);
            // Also extend the weighted sum of the norm_correction
            exp_wsum_norm_correction[img_id] += DIRECT_A1D_ELEM(op.power_img[img_id], ires);
        }

        // Store norm_correction
        // Multiply by old value because the old norm_correction term was already applied to the image
        if (baseMLO->do_norm_correction && baseMLO->mymodel.nr_bodies == 1)
        {

            RFLOAT old_norm_correction = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_NORM);
            old_norm_correction /= baseMLO->mymodel.avg_norm_correction;
            // The factor two below is because exp_wsum_norm_correctiom is similar to sigma2_noise, which is the variance for the real/imag components
            // The variance of the total image (on which one normalizes) is twice this value!
            RFLOAT normcorr = old_norm_correction * sqrt(exp_wsum_norm_correction[img_id] * 2.);
            thr_avg_norm_correction += normcorr;

            // Now set the new norm_correction in the relevant position of exp_metadata

            DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_NORM) = normcorr;


            // Print warning for strange norm-correction values

            if (!((baseMLO->iter == 1 && baseMLO->do_firstiter_cc) || baseMLO->do_always_cc) && DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_NORM) > 10.)
            {

                std::cout << " WARNING: norm_correction= "<< DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_NORM)
                        << " for particle " << op.part_id << " in group " << group_id + 1
                        << "; Are your groups large enough? Or is the reference on the correct greyscale?" << std::endl;
            }

        }

        // Store weighted sums for scale_correction
        if (baseMLO->do_scale_correction)
        {
            // Divide XA by the old scale_correction and AA by the square of that, because was incorporated into Fctf
            exp_wsum_scale_correction_XA[img_id] /= baseMLO->mymodel.scale_correction[group_id];
            exp_wsum_scale_correction_AA[img_id] /= baseMLO->mymodel.scale_correction[group_id] * baseMLO->mymodel.scale_correction[group_id];

            thr_wsum_signal_product_spectra[img_id] += exp_wsum_scale_correction_XA[img_id];
            thr_wsum_reference_power_spectra[img_id] += exp_wsum_scale_correction_AA[img_id];
        }

        // Calculate DLL for each particle
        RFLOAT logsigma2 = 0.;
        RFLOAT remap_image_sizes = (baseMLO->mymodel.ori_size * baseMLO->mymodel.pixel_size) / (my_image_size * my_pixel_size);
        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(baseMLO->Mresol_fine[optics_group])
        {
            int ires = DIRECT_MULTIDIM_ELEM(baseMLO->Mresol_fine[optics_group], n);
            int ires_remapped = ROUND(remap_image_sizes * ires);
            // Note there is no sqrt in the normalisation term because of the 2-dimensionality of the complex-plane
            // Also exclude origin from logsigma2, as this will not be considered in the P-calculations
            if (ires > 0 && ires_remapped < XSIZE(baseMLO->mymodel.sigma2_noise[optics_group]))
                logsigma2 += log( 2. * PI * DIRECT_A1D_ELEM(baseMLO->mymodel.sigma2_noise[optics_group], ires_remapped));
        }
        RFLOAT dLL;

        if ((baseMLO->iter==1 && baseMLO->do_firstiter_cc) || baseMLO->do_always_cc)
            dLL = -op.min_diff2[img_id];
        else
            dLL = log(op.sum_weight[img_id]) - op.min_diff2[img_id] - logsigma2;

        // Store dLL of each image in the output array, and keep track of total sum

        DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_DLL) = dLL;
        thr_sum_dLL += dLL;

        // Also store sum of Pmax

        thr_sum_Pmax += DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_PMAX);

    }

    // for (int img_id = 0; img_id < sp.nr_images; img_id++)
    // {
    //     long int igroup = baseMLO->mydata.getGroupId(op.part_id, img_id);
    //     int optics_group = baseMLO->mydata.getOpticsGroup(op.part_id, img_id);

    //     if (baseMLO->mydata.obsModel.getCtfPremultiplied(optics_group))
    //     {
    //         RFLOAT myscale = XMIPP_MAX(0.001, baseMLO->mymodel.scale_correction[igroup]);
    //         FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(baseMLO->Mresol_fine[optics_group])
    //         {
    //             int ires = DIRECT_MULTIDIM_ELEM(baseMLO->Mresol_fine[optics_group], n);
    //             if (ires > -1)
    //                 #pragma omp atomic
    //                 DIRECT_MULTIDIM_ELEM(thr_wsum_ctf2[img_id], ires) += myscale * DIRECT_MULTIDIM_ELEM(op.local_Fctf[img_id], n);
    //         }
    //     }

    //     if (do_subtomo_correction)
    //     {
    //         FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(baseMLO->Mresol_fine[optics_group])
    //         {
    //             int ires = DIRECT_MULTIDIM_ELEM(baseMLO->Mresol_fine[optics_group], n);
    //             if (ires > -1)
    //                 #pragma omp atomic
    //                 DIRECT_MULTIDIM_ELEM(thr_wsum_stMulti[img_id], ires) += DIRECT_MULTIDIM_ELEM(exp_local_STMulti[img_id], n);
    //         }
    //     }

    //     int my_image_size = baseMLO->mydata.getOpticsImageSize(optics_group);
    //     RFLOAT my_pixel_size = baseMLO->mydata.getOpticsPixelSize(optics_group);
    //     RFLOAT remap_image_sizes = (baseMLO->mymodel.ori_size * baseMLO->mymodel.pixel_size) / (my_image_size * my_pixel_size);
    //     FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(thr_wsum_sigma2_noise[img_id])
    //     {
    //         int i_resam = ROUND(i * remap_image_sizes);
    //         if (i_resam < XSIZE(baseMLO->wsum_model.sigma2_noise[optics_group]))
    //         {
    //             #pragma omp atomic
    //             DIRECT_A1D_ELEM(baseMLO->wsum_model.sigma2_noise[optics_group], i_resam) += DIRECT_A1D_ELEM(thr_wsum_sigma2_noise[img_id], i);
    //             #pragma omp atomic
    //             DIRECT_A1D_ELEM(baseMLO->wsum_model.sumw_ctf2[optics_group], i_resam) += DIRECT_A1D_ELEM(thr_wsum_ctf2[img_id], i);

    //             if (do_subtomo_correction)
    //                 #pragma omp atomic
    //                 DIRECT_A1D_ELEM(baseMLO->wsum_model.sumw_stMulti[optics_group], i_resam) += DIRECT_A1D_ELEM(thr_wsum_stMulti[img_id], i);
    //         }
    //     }

    //     #pragma omp atomic
    //     baseMLO->wsum_model.sumw_group[optics_group] += thr_sumw_group[img_id];

    //     if (baseMLO->do_scale_correction)
    //     {
    //         #pragma omp atomic
    //         baseMLO->wsum_model.wsum_signal_product[igroup] += thr_wsum_signal_product_spectra[img_id];
    //         #pragma omp atomic
    //         baseMLO->wsum_model.wsum_reference_power[igroup] += thr_wsum_reference_power_spectra[img_id];
    //     }
    // }

    // for (int n = 0; n < baseMLO->mymodel.nr_classes; n++)
    // {
    //     #pragma omp atomic
    //     baseMLO->wsum_model.pdf_class[n] += thr_wsum_pdf_class[n];
    //     if (baseMLO->mymodel.ref_dim == 2)
    //     {
    //         #pragma omp atomic
    //         XX(baseMLO->wsum_model.prior_offset_class[n]) += thr_wsum_prior_offsetx_class[n];
    //         #pragma omp atomic
    //         YY(baseMLO->wsum_model.prior_offset_class[n]) += thr_wsum_prior_offsety_class[n];
    //     }
    // }

    // for (int n = 0; n < baseMLO->mymodel.nr_classes * baseMLO->mymodel.nr_bodies; n++)
    // {
    //     if (!(baseMLO->do_skip_align || baseMLO->do_skip_rotate) )
    //         #pragma omp atomic
    //         baseMLO->wsum_model.pdf_direction[n] += thr_wsum_pdf_direction[n];
    // }

    // #pragma omp atomic
    // baseMLO->wsum_model.sigma2_offset += thr_wsum_sigma2_offset;

    // if (baseMLO->do_norm_correction && baseMLO->mymodel.nr_bodies == 1)
    //     #pragma omp atomic
    //     baseMLO->wsum_model.avg_norm_correction += thr_avg_norm_correction;

    // #pragma omp atomic
    // baseMLO->wsum_model.LL += thr_sum_dLL;

    // #pragma omp atomic
    // baseMLO->wsum_model.ave_Pmax += thr_sum_Pmax;

    // // Now, inside a global_mutex, update the other weighted sums among all threads
    #pragma omp critical(AccMLO_global)
    {
        for (int img_id = 0; img_id < sp.nr_images; img_id++)
        {
            long int igroup = baseMLO->mydata.getGroupId(op.part_id, img_id);
            int optics_group = baseMLO->mydata.getOpticsGroup(op.part_id, img_id);


            if (baseMLO->mydata.obsModel.getCtfPremultiplied(optics_group))
            {
                RFLOAT myscale = XMIPP_MAX(0.001, baseMLO->mymodel.scale_correction[igroup]);
                FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(baseMLO->Mresol_fine[optics_group])
                {
                    int ires = DIRECT_MULTIDIM_ELEM(baseMLO->Mresol_fine[optics_group], n);
                    if (ires > -1)
                        DIRECT_MULTIDIM_ELEM(thr_wsum_ctf2[img_id], ires) += myscale * DIRECT_MULTIDIM_ELEM(op.local_Fctf[img_id], n);
                }
            }

            if (do_subtomo_correction)
            {
                FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(baseMLO->Mresol_fine[optics_group])
                {
                    int ires = DIRECT_MULTIDIM_ELEM(baseMLO->Mresol_fine[optics_group], n);
                    if (ires > -1)
                        DIRECT_MULTIDIM_ELEM(thr_wsum_stMulti[img_id], ires) += DIRECT_MULTIDIM_ELEM(exp_local_STMulti[img_id], n);
                }
            }

            int my_image_size = baseMLO->mydata.getOpticsImageSize(optics_group);
            RFLOAT my_pixel_size = baseMLO->mydata.getOpticsPixelSize(optics_group);
            RFLOAT remap_image_sizes = (baseMLO->mymodel.ori_size * baseMLO->mymodel.pixel_size) / (my_image_size * my_pixel_size);
            FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(thr_wsum_sigma2_noise[img_id])
            {
                int i_resam = ROUND(i * remap_image_sizes);
                if (i_resam < XSIZE(baseMLO->wsum_model.sigma2_noise[optics_group]))
                {
                    DIRECT_A1D_ELEM(baseMLO->wsum_model.sigma2_noise[optics_group], i_resam) += DIRECT_A1D_ELEM(thr_wsum_sigma2_noise[img_id], i);
                    DIRECT_A1D_ELEM(baseMLO->wsum_model.sumw_ctf2[optics_group], i_resam) += DIRECT_A1D_ELEM(thr_wsum_ctf2[img_id], i);

                    if (do_subtomo_correction)
                        DIRECT_A1D_ELEM(baseMLO->wsum_model.sumw_stMulti[optics_group], i_resam) += DIRECT_A1D_ELEM(thr_wsum_stMulti[img_id], i);
                }
            }
            baseMLO->wsum_model.sumw_group[optics_group] += thr_sumw_group[img_id];
            if (baseMLO->do_scale_correction)
            {
                baseMLO->wsum_model.wsum_signal_product[igroup] += thr_wsum_signal_product_spectra[img_id];
                baseMLO->wsum_model.wsum_reference_power[igroup] += thr_wsum_reference_power_spectra[img_id];
            }
        }
        for (int n = 0; n < baseMLO->mymodel.nr_classes; n++)
        {
            baseMLO->wsum_model.pdf_class[n] += thr_wsum_pdf_class[n];
            if (baseMLO->mymodel.ref_dim == 2)
            {
                XX(baseMLO->wsum_model.prior_offset_class[n]) += thr_wsum_prior_offsetx_class[n];
                YY(baseMLO->wsum_model.prior_offset_class[n]) += thr_wsum_prior_offsety_class[n];
            }
        }

        for (int n = 0; n < baseMLO->mymodel.nr_classes * baseMLO->mymodel.nr_bodies; n++)
        {
            if (!(baseMLO->do_skip_align || baseMLO->do_skip_rotate) )
                baseMLO->wsum_model.pdf_direction[n] += thr_wsum_pdf_direction[n];
        }


        baseMLO->wsum_model.sigma2_offset += thr_wsum_sigma2_offset;

        if (baseMLO->do_norm_correction && baseMLO->mymodel.nr_bodies == 1)
            baseMLO->wsum_model.avg_norm_correction += thr_avg_norm_correction;

        baseMLO->wsum_model.LL += thr_sum_dLL;
        baseMLO->wsum_model.ave_Pmax += thr_sum_Pmax;

    }
    CTOC(accMLO->timer,"storeWeightedSumsStorePost");
#ifdef TIMING
	// if (op.part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
		baseMLO->timer.toc(baseMLO->TIMING_ESP_WSUM);
#endif
	LAUNCH_HANDLE_ERROR(cudaGetLastError());

}


#endif // ACC_STORE_WEIGHT_SUM_H_