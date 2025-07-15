#ifndef ACC_COARSE_H_
#define ACC_COARSE_H_
#include "src/ml_optimiser_mpi.h"
#include "src/acc/acc_context.h"

// ----------------------------------------------------------------------------
// ------------------ getAllSquaredDifferencesCoarse --------------------------
// ----------------------------------------------------------------------------
template <class MlClass>
void getAllSquaredDifferencesCoarsePre(Context<MlClass>& ctx)
{
    CTIC(ctx.accMLO->timer,"getAllSquaredDifferencesCoarsePre");
	unsigned exp_ipass        = ctx.exp_ipass;
	OptimisationParamters& op = ctx.op;
	SamplingParameters& sp    = ctx.sp;
	MlOptimiser* baseMLO      = ctx.baseMLO;
	MlClass *accMLO           = ctx.accMLO;
#ifdef NEWMEM
	AccPtrNew<XFLOAT>& Mweight   = ctx.Mweight;
    AccPtrFactoryNew& ptrFactory = ctx.ptrFactory;
#else
	AccPtr<XFLOAT>& Mweight   = ctx.Mweight;
    AccPtrFactory& ptrFactory = ctx.ptrFactory;
#endif
	int ibody                 = ctx.ibody;
	int thread_id             = ctx.thread_id;
#ifdef TIMING
	// if (op.part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
		baseMLO->timer.tic(baseMLO->TIMING_ESP_DIFF1);
#endif

	CUSTOM_ALLOCATOR_REGION_NAME("DIFF_COARSE");

	CTIC(accMLO->timer,"diff_pre_gpu");
	// unsigned long weightsPerPart(baseMLO->mymodel.nr_classes * sp.nr_dir * sp.nr_psi * sp.nr_trans * sp.nr_oversampled_rot * sp.nr_oversampled_trans);
	ctx.weightsPerPart = baseMLO->mymodel.nr_classes * sp.nr_dir * sp.nr_psi * sp.nr_trans * sp.nr_oversampled_rot * sp.nr_oversampled_trans;

	std::vector<MultidimArray<Complex > > dummy;
	std::vector<std::vector<MultidimArray<Complex > > > dummy2;
	std::vector<MultidimArray<RFLOAT> > dummyRF;
	baseMLO->precalculateShiftedImagesCtfsAndInvSigma2s(false, false, op.part_id, sp.current_oversampling, op.metadata_offset, // inserted SHWS 12112015
			sp.itrans_min, sp.itrans_max, op.Fimg, dummy, op.Fctf, dummy2, dummy2,
			op.local_Fctf, op.local_sqrtXi2, op.local_Minvsigma2, op.FstMulti, dummyRF);

	CTOC(accMLO->timer,"diff_pre_gpu");

	// ctx.projectorPlans = new std::vector< AccProjectorPlan>(0, (CudaCustomAllocator *)accMLO->getAllocator());
	// std::vector< AccProjectorPlan >& projectorPlans = *ctx.projectorPlans;

#ifdef NEWMEM
	std::vector< AccProjectorPlanNew >& projectorPlans = ctx.projectorPlans;
#else
	ctx.projectorPlans = std::vector< AccProjectorPlan>(0, (CudaCustomAllocator *)accMLO->getAllocator());
	std::vector< AccProjectorPlan >& projectorPlans = ctx.projectorPlans;
#endif

	// std::vector< AccProjectorPlan > projectorPlans(0, (CudaCustomAllocator *)accMLO->getAllocator());

	//If particle specific sampling plan required
	if (accMLO->generateProjectionPlanOnTheFly)
	{
		CTIC(accMLO->timer,"generateProjectionSetupCoarse");
#ifdef NEWMEM
		for(int i=0;i<baseMLO->mymodel.nr_classes;i++)
			projectorPlans.emplace_back(
                                (StreamType)ptrFactory.getStream(),
                                (CudaAllocatorTask *)ptrFactory.getAllocator_task(),
                                (CudaCustomAllocator *)accMLO->getAllocator());
#else
		projectorPlans.resize(baseMLO->mymodel.nr_classes, (CudaCustomAllocator *)accMLO->getAllocator());
#endif

		for (unsigned long iclass = sp.iclass_min; iclass <= sp.iclass_max; iclass++)
		{
			if (baseMLO->mymodel.pdf_class[iclass] > 0.)
			{
				Matrix2D<RFLOAT> MBL, MBR, Aori;

				if (baseMLO->mymodel.nr_bodies > 1)
				{
					// img_id=0 because in multi-body refinement we do not do movie frames!
					RFLOAT rot_ori = DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset, METADATA_ROT);
					RFLOAT tilt_ori = DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset, METADATA_TILT);
					RFLOAT psi_ori = DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset, METADATA_PSI);
					Euler_angles2matrix(rot_ori, tilt_ori, psi_ori, Aori, false);

					MBL = Aori * (baseMLO->mymodel.orient_bodies[ibody]).transpose() * baseMLO->A_rot90;
					MBR = baseMLO->mymodel.orient_bodies[ibody];
				}

				int optics_group = baseMLO->mydata.getOpticsGroup(op.part_id, 0); // get optics group of first image for this particle...
				Matrix2D<RFLOAT> mag;
				mag.initIdentity(3);
				mag = baseMLO->mydata.obsModel.applyAnisoMag(mag, optics_group);
				mag = baseMLO->mydata.obsModel.applyScaleDifference(mag, optics_group, baseMLO->mymodel.ori_size, baseMLO->mymodel.pixel_size);
				if (!mag.isIdentity())
				{
					if (MBL.mdimx == 3 && MBL.mdimx ==3) MBL = mag * MBL;
					else MBL = mag;
				}

				projectorPlans[iclass].setup(
						baseMLO->sampling,
						op.directions_prior,
						op.psi_prior,
						op.pointer_dir_nonzeroprior,
						op.pointer_psi_nonzeroprior,
						NULL, //Mcoarse_significant
						baseMLO->mymodel.pdf_class,
						baseMLO->mymodel.pdf_direction,
						sp.nr_dir,
						sp.nr_psi,
						sp.idir_min,
						sp.idir_max,
						sp.ipsi_min,
						sp.ipsi_max,
						sp.itrans_min,
						sp.itrans_max,
						0, //current_oversampling
						1, //nr_oversampled_rot
						iclass,
						true, //coarse
						!IS_NOT_INV,
						baseMLO->do_skip_align,
						baseMLO->do_skip_rotate,
						baseMLO->mymodel.orientational_prior_mode,
						MBL,
						MBR
						);
			}
		}
		CTOC(accMLO->timer,"generateProjectionSetupCoarse");
	}
	else
#ifdef NEWMEM
    {
		int now_plan_num=accMLO->bundle->coarseProjectionPlans.size();
		projectorPlans.resize(now_plan_num, AccProjectorPlanNew((StreamType)ptrFactory.getStream(),
                                (CudaAllocatorTask *)ptrFactory.getAllocator_task(),
                                (CudaCustomAllocator *)accMLO->getAllocator()));
        // (CudaAllocatorTask *)ptrFactory.getAllocator_task());
		for (unsigned long iclass = 0; iclass < now_plan_num; iclass++)
		{
			projectorPlans[iclass].set_up_from_old_plan(accMLO->bundle->coarseProjectionPlans[iclass]);
		}
		// projectorPlans = accMLO->bundle->coarseProjectionPlans;
	}
#else
		projectorPlans = accMLO->bundle->coarseProjectionPlans;
#endif

#ifndef NEWMEM
    DEBUG_HANDLE_ERROR(cudaStreamSynchronize(cudaStreamPerThread));
#endif
	// Loop only from sp.iclass_min to sp.iclass_max to deal with seed generation in first iteration
	size_t allWeights_size(0);
	for (int exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
		allWeights_size += projectorPlans[exp_iclass].orientation_num * sp.nr_trans*sp.nr_oversampled_trans;

	// AccPtr<XFLOAT> allWeights = ptrFactory.make<XFLOAT>(allWeights_size);
	ctx.allWeights =  ptrFactory.make<XFLOAT>(allWeights_size);
#ifdef NEWMEM
	AccPtrNew<XFLOAT>& allWeights = ctx.allWeights;
#else
	AccPtr<XFLOAT>& allWeights = ctx.allWeights;
#endif
	allWeights.accAlloc();
	deviceInitValue<XFLOAT>(allWeights, 0);  // Make sure entire array initialized
    
    allWeights.streamSync();

    //这里没有对plan的stream（默认）进行同步，所以出问题了-》stream改成cudaStreamPerTask
    //但是这里这个cudaStreamPerThread，是哪来的？

	// long int allWeights_pos=0;	bool do_CC = (baseMLO->iter == 1 && baseMLO->do_firstiter_cc) || baseMLO->do_always_cc;
	ctx.allWeights_pos=0;	
	ctx.do_CC = (baseMLO->iter == 1 && baseMLO->do_firstiter_cc) || baseMLO->do_always_cc;

#ifdef TIMING
	// if (op.part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
		baseMLO->timer.toc(baseMLO->TIMING_ESP_DIFF1);
#endif
    CTOC(ctx.accMLO->timer,"getAllSquaredDifferencesCoarsePre");
}


template <class MlClass>
void getAllSquaredDifferencesCoarsePostPerImg(Context<MlClass>& ctx) {
    CTIC(ctx.accMLO->timer,"getAllSquaredDifferencesCoarsePostPerImg");
    unsigned exp_ipass        = ctx.exp_ipass;
	OptimisationParamters& op = ctx.op;
	SamplingParameters& sp    = ctx.sp;
	MlOptimiser* baseMLO      = ctx.baseMLO;
	MlClass *accMLO           = ctx.accMLO;
	int ibody                 = ctx.ibody;
	int thread_id             = ctx.thread_id;
#ifdef TIMING
	// if (op.part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
		baseMLO->timer.tic(baseMLO->TIMING_ESP_DIFF1);
#endif

    unsigned long weightsPerPart                    = ctx.weightsPerPart;
	long int allWeights_pos                         = ctx.allWeights_pos;
	bool do_CC                                      = ctx.do_CC;
	int img_id                 = ctx.img_id;

#ifdef NEWMEM
    std::vector< AccProjectorPlanNew >& projectorPlans = ctx.projectorPlans;
    AccPtrNew<XFLOAT>& allWeights = ctx.allWeights;
    AccPtrNew<XFLOAT>& Mweight   = ctx.Mweight;
    AccPtrFactoryNew& ptrFactory = ctx.ptrFactory;
#else
    std::vector< AccProjectorPlan >& projectorPlans = ctx.projectorPlans;
    AccPtr<XFLOAT>& allWeights = ctx.allWeights;
    AccPtr<XFLOAT>& Mweight   = ctx.Mweight;
    AccPtrFactory& ptrFactory = ctx.ptrFactory;
#endif 
	// printf("#### sizeof(context) = %d\n", sizeof(Context<MlClass>));
	// printf("#### sizeof(AccPtr)  = %d\n", sizeof(AccPtr<XFLOAT>));
	// printf("#### sizeof(Op)      = %d\n", sizeof(OptimisationParamters));
	// printf("#### sizeof(Sp)      = %d\n", sizeof(SamplingParameters));

    int my_metadata_offset = op.metadata_offset + img_id;
    long int group_id = baseMLO->mydata.getGroupId(op.part_id, img_id);
    RFLOAT my_pixel_size = baseMLO->mydata.getImagePixelSize(op.part_id, img_id);
    int optics_group = baseMLO->mydata.getOpticsGroup(op.part_id, img_id);

    ctx.image_size = op.local_Minvsigma2[img_id].nzyxdim;
    unsigned long& image_size = ctx.image_size;
    // unsigned long image_size = op.local_Minvsigma2[img_id].nzyxdim;

    /*====================================
            Generate Translations
    ======================================*/

    CTIC(accMLO->timer,"translation_1");

    ctx.translation_num = (sp.itrans_max - sp.itrans_min + 1) * sp.nr_oversampled_trans;
    long unsigned& translation_num = ctx.translation_num;
    // long unsigned translation_num((sp.itrans_max - sp.itrans_min + 1) * sp.nr_oversampled_trans);
    // here we introduce offsets for the trans_ and img_ in an array as it is more efficient to
    // copy one big array to/from GPU rather than four small arrays

    ctx.trans_x_offset = 0*(size_t)translation_num;
    ctx.trans_y_offset = 1*(size_t)translation_num;
    ctx.trans_z_offset = 2*(size_t)translation_num;
    ctx.img_re_offset = 0*(size_t)image_size;
    ctx.img_im_offset = 1*(size_t)image_size;
    size_t& trans_x_offset = ctx.trans_x_offset;
    size_t& trans_y_offset = ctx.trans_y_offset;
    size_t& trans_z_offset = ctx.trans_z_offset;
    size_t& img_re_offset = ctx.img_re_offset;
    size_t& img_im_offset = ctx.img_im_offset;

    ctx.Fimg_ = ptrFactory.make<XFLOAT>((size_t)image_size*2);
    ctx.trans_xyz = ptrFactory.make<XFLOAT>((size_t)translation_num*3);
#ifdef NEWMEM
    // AccPtrNew<XFLOAT>& Fimg_ = ptrFactory.make<XFLOAT>((size_t)image_size*2);
    // AccPtrNew<XFLOAT>& trans_xyz = ptrFactory.make<XFLOAT>((size_t)translation_num*3);
    AccPtrNew<XFLOAT>& Fimg_ = ctx.Fimg_;
    AccPtrNew<XFLOAT>& trans_xyz = ctx.trans_xyz;
#else
    AccPtr<XFLOAT>& Fimg_ = ctx.Fimg_;
    AccPtr<XFLOAT>& trans_xyz = ctx.trans_xyz;
#endif
    // AccPtr<int> pixel_index = ptrFactory.make<int>((size_t)image_size*2);

    Fimg_.allAlloc();
    trans_xyz.allAlloc();

    std::vector<RFLOAT> oversampled_translations_x, oversampled_translations_y, oversampled_translations_z;

    for (long int itrans = 0; itrans < translation_num; itrans++)
    {
        baseMLO->sampling.getTranslationsInPixel(itrans, 0, my_pixel_size, oversampled_translations_x,
                oversampled_translations_y, oversampled_translations_z,
                (baseMLO->do_helical_refine) && (! baseMLO->ignore_helical_symmetry));

        RFLOAT xshift = 0., yshift = 0., zshift = 0.;

        xshift = oversampled_translations_x[0];
        yshift = oversampled_translations_y[0];
        if (accMLO->dataIs3D)
            zshift = oversampled_translations_z[0];

        if ( (baseMLO->do_helical_refine) && (! baseMLO->ignore_helical_symmetry) )
        {
            RFLOAT rot_deg = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_ROT);
            RFLOAT tilt_deg = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_TILT);
            RFLOAT psi_deg = DIRECT_A2D_ELEM(baseMLO->exp_metadata,my_metadata_offset, METADATA_PSI);
            transformCartesianAndHelicalCoords(xshift, yshift, zshift, xshift, yshift, zshift, rot_deg, tilt_deg, psi_deg, (accMLO->dataIs3D) ? (3) : (2), HELICAL_TO_CART_COORDS);
        }

        trans_xyz[trans_x_offset+itrans] = -2 * PI * xshift / (double)baseMLO->image_full_size[optics_group];
        trans_xyz[trans_y_offset+itrans] = -2 * PI * yshift / (double)baseMLO->image_full_size[optics_group];
        trans_xyz[trans_z_offset+itrans] = -2 * PI * zshift / (double)baseMLO->image_full_size[optics_group];
    }

    XFLOAT scale_correction = baseMLO->do_scale_correction ? baseMLO->mymodel.scale_correction[group_id] : 1;
    int exp_current_image_size = (baseMLO->strict_highres_exp > 0.|| baseMLO->adaptive_oversampling > 0) ?
            baseMLO->image_coarse_size[optics_group] : baseMLO->image_current_size[optics_group];
    MultidimArray<Complex > Fimg;
    windowFourierTransform(op.Fimg[img_id], Fimg, exp_current_image_size);

    for (unsigned long i = 0; i < image_size; i ++)
    {
        XFLOAT pixel_correction = 1.0/scale_correction;
        if (baseMLO->do_ctf_correction && fabs(op.local_Fctf[img_id].data[i]) > 1e-8)
        {
            // if ctf[i]==0, pix_corr[i] becomes NaN.
            // However, corr_img[i]==0, so pix-diff in kernel==0.
            // This is ok since originally, pix-diff==Img.real^2 + Img.imag^2,
            // which is ori-indep, and we subtract min_diff form ALL orients.

            if (baseMLO->refs_are_ctf_corrected)
            {
                pixel_correction /= op.local_Fctf[img_id].data[i];
            }
        }
        Fimg_[img_re_offset+i] = Fimg.data[i].real * pixel_correction;
        Fimg_[img_im_offset+i] = Fimg.data[i].imag * pixel_correction;
    }

    trans_xyz.cpToDevice();
    Fimg_.cpToDevice();

    CTOC(accMLO->timer,"translation_1");

    // To speed up calculation, several image-corrections are grouped into a single pixel-wise "filter", or image-correciton

    ctx.corr_img = ptrFactory.make<XFLOAT>((size_t)image_size);
#ifdef NEWMEM
    AccPtrNew<XFLOAT>& corr_img = ctx.corr_img;
#else
    AccPtr<XFLOAT>& corr_img = ctx.corr_img;
#endif
    corr_img.allAlloc();

    buildCorrImage(baseMLO,op,corr_img,img_id,group_id);
    corr_img.cpToDevice();

    deviceInitValue<XFLOAT>(allWeights, (XFLOAT) (op.highres_Xi2_img[img_id] / 2.));
    allWeights_pos = 0;

    for (int exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
        DEBUG_HANDLE_ERROR(cudaStreamSynchronize(accMLO->classStreams[exp_iclass]));
    DEBUG_HANDLE_ERROR(cudaStreamSynchronize(cudaStreamPerThread));
    for (unsigned long exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
		DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.classStreams[exp_iclass]));
	DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.cudaStreamPerTask));

#ifdef TIMING
	// if (op.part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
		baseMLO->timer.toc(baseMLO->TIMING_ESP_DIFF1);
#endif
    CTOC(ctx.accMLO->timer,"getAllSquaredDifferencesCoarsePostPerImg");
}

template <class MlClass>
void getAllSquaredDifferencesCoarsePostPerImgLaunchKernel(Context<MlClass>& ctx) {
    CTIC(ctx.accMLO->timer,"getAllSquaredDifferencesCoarsePostPerImgLaunchKernel");

    long unsigned& translation_num = ctx.translation_num;
    long unsigned& image_size = ctx.image_size;
    size_t& trans_x_offset = ctx.trans_x_offset;
    size_t& trans_y_offset = ctx.trans_y_offset;
    size_t& trans_z_offset = ctx.trans_z_offset;
    size_t& img_re_offset = ctx.img_re_offset;
    size_t& img_im_offset = ctx.img_im_offset;

	OptimisationParamters& op = ctx.op;
	SamplingParameters& sp    = ctx.sp;
	MlOptimiser* baseMLO      = ctx.baseMLO;
	MlClass *accMLO           = ctx.accMLO;
	int ibody                 = ctx.ibody;
	int thread_id             = ctx.thread_id;
    unsigned long weightsPerPart                    = ctx.weightsPerPart;
	long int allWeights_pos                         = ctx.allWeights_pos;
	bool do_CC                                      = ctx.do_CC;
	int img_id                 = ctx.img_id;
#ifdef NEWMEM
    AccPtrNew<XFLOAT>& allWeights = ctx.allWeights;
    AccPtrNew<XFLOAT>& Fimg_ = ctx.Fimg_;
    AccPtrNew<XFLOAT>& trans_xyz = ctx.trans_xyz;
    AccPtrNew<XFLOAT>& corr_img = ctx.corr_img;
    AccPtrNew<XFLOAT>& Mweight   = ctx.Mweight;
    AccPtrFactoryNew& ptrFactory = ctx.ptrFactory;
	std::vector< AccProjectorPlanNew >& projectorPlans = ctx.projectorPlans;
#else
    AccPtr<XFLOAT>& allWeights = ctx.allWeights;
    AccPtr<XFLOAT>& Fimg_ = ctx.Fimg_;
    AccPtr<XFLOAT>& trans_xyz = ctx.trans_xyz;
    AccPtr<XFLOAT>& corr_img = ctx.corr_img;
    AccPtr<XFLOAT>& Mweight   = ctx.Mweight;
    AccPtrFactory& ptrFactory = ctx.ptrFactory;
	std::vector< AccProjectorPlan >& projectorPlans = ctx.projectorPlans;
#endif
#ifdef TIMING
	// if (op.part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
		baseMLO->timer.tic(baseMLO->TIMING_ESP_DIFF1);
#endif
// // ========================= save data to file =========================
// int iter = baseMLO->iter;
// // std::string path = "/home/fujy/relion/test/test_coarse/data/trpv1/iter_" + std::to_string(iter) + "/";
// std::string path = "/home/xujingle/tensor_core/data/trpv1/"+ std::to_string(iter) + "/";

// // mpi rank
// int rank = 0;
// int mpi_initialized = 0;
// MPI_Initialized(&mpi_initialized);
// if (mpi_initialized) MPI_Comm_rank(MPI_COMM_WORLD, &rank);
// std::string filename = "coarse_data" + std::to_string(ctx.part_id_sorted) + "_" + std::to_string(rank) + ".dat";
// std::string filepath = path + filename;

// if (!std::ifstream(filepath) && (ctx.part_id_sorted % 100 == 0)) {
//     std::ofstream ofs(filepath, std::ios::binary);
//     if (!ofs) {
//         std::cerr << "failed to open file : " << filepath << std::endl;
//         exit(1);
//     }

//     ofs.write((char*)&op.local_Minvsigma2[img_id].xdim, sizeof(int));
//     ofs.write((char*)&op.local_Minvsigma2[img_id].ydim, sizeof(int));
//     ofs.write((char*)&op.local_Minvsigma2[img_id].zdim, sizeof(int));
//     int _rmax = op.local_Minvsigma2[img_id].xdim - 1;
//     ofs.write((char*)&_rmax, sizeof(int));


//     ofs.write((char*)&translation_num, sizeof(int));
//     ofs.write((char*)&projectorPlans[sp.iclass_min].orientation_num, sizeof(int));
//     ofs.write((char*)&image_size, sizeof(int));

//     projectorPlans[sp.iclass_min].eulers.cpToHost();
//     trans_xyz.cpToHost();
//     Fimg_.cpToHost();
//     corr_img.cpToHost();
//     // allWeights.cpToHost();

//     projectorPlans[sp.iclass_min].eulers.streamSync();
//     trans_xyz.streamSync();
//     Fimg_.streamSync();
//     corr_img.streamSync();
//     // allWeights.streamSync();

//     DEBUG_HANDLE_ERROR(cudaStreamSynchronize(cudaStreamPerThread));

//     // eulers
//     ofs.write((char*)projectorPlans[sp.iclass_min].eulers.getHostPtr(), sizeof(XFLOAT) * 9 * projectorPlans[sp.iclass_min].orientation_num);
//     // // 打印出前10个euler角
//     // for (int i = 0; i < 10; i++) {
//     //     printf("euler %d : %f %f %f %f %f %f %f %f %f\n", i,
//     //             projectorPlans[sp.iclass_min].eulers[i*9+0],
//     //             projectorPlans[sp.iclass_min].eulers[i*9+1],
//     //             projectorPlans[sp.iclass_min].eulers[i*9+2],
//     //             projectorPlans[sp.iclass_min].eulers[i*9+3],
//     //             projectorPlans[sp.iclass_min].eulers[i*9+4],
//     //             projectorPlans[sp.iclass_min].eulers[i*9+5],
//     //             projectorPlans[sp.iclass_min].eulers[i*9+6],
//     //             projectorPlans[sp.iclass_min].eulers[i*9+7],
//     //             projectorPlans[sp.iclass_min].eulers[i*9+8]);
//     // }
//     ofs.write((char*)trans_xyz.getHostPtr(), sizeof(XFLOAT) * 2 * translation_num);
//     ofs.write((char*)Fimg_.getHostPtr(), sizeof(XFLOAT) * 2 * image_size);
//     ofs.write((char*)corr_img.getHostPtr(), sizeof(XFLOAT) * image_size);
//     ofs.write((char*)allWeights.getHostPtr(), sizeof(XFLOAT) * projectorPlans[sp.iclass_min].orientation_num * translation_num);
//     ofs.close();

//     printf("save data to %s\n", filepath.c_str());
// }


// =====================================================================

    for (unsigned long iclass = sp.iclass_min; iclass <= sp.iclass_max; iclass++)
    {
        int iproj;
        if (baseMLO->mymodel.nr_bodies > 1) iproj = ibody;
        else                                iproj = iclass;

        if ( projectorPlans[iclass].orientation_num > 0 )
        {
            AccProjectorKernel projKernel = AccProjectorKernel::makeKernel(
                    accMLO->bundle->projectors[iproj],
                    op.local_Minvsigma2[img_id].xdim,
                    op.local_Minvsigma2[img_id].ydim,
                    op.local_Minvsigma2[img_id].zdim,
                    op.local_Minvsigma2[img_id].xdim-1);
#ifdef TIMING
if(ctx.thread_id==0)
    baseMLO->timer.tic(baseMLO->TIMING_EXTRA1);
// baseMLO->timer.tic(baseMLO->TIMING_EXTRA1_T[thread_id]);
#endif
            // fprintf(stderr, "#### acc_ml_optimiser_impl.h runDiff2KernelCoarse\n");
            // fprintf(stderr, "     orien_num : %4d  trans_num : %4d  img_size : %4d\n", projectorPlans[iclass].orientation_num, translation_num, image_size);
 #if 1
            runDiff2KernelCoarse(
                    projKernel,
                    &(~trans_xyz)[trans_x_offset], //~trans_x,
                    &(~trans_xyz)[trans_y_offset], //~trans_y,
                    &(~trans_xyz)[trans_z_offset], //~trans_z,
                    ~corr_img,
                    &(~Fimg_)[img_re_offset], //~Fimg_real,
                    &(~Fimg_)[img_im_offset], //~Fimg_imag,
                    ~projectorPlans[iclass].eulers,
                    &(~allWeights)[allWeights_pos],
                    (XFLOAT) op.local_sqrtXi2[img_id],
                    projectorPlans[iclass].orientation_num,
                    translation_num,
                    image_size,
                    // accMLO->classStreams[iclass],
                    ctx.classStreams[iclass],
                    do_CC,
                    accMLO->dataIs3D);
#else
  using CT = CoarseKernelBlockTParams<
  128, // block_size
  64,  // trans_block_size
  128, // orient_block_size
  16,  // img_block_size
  32,  // warp_trans_tils_size
  64,  // warp_orient_tile_size
  16,  // warp_img_tile
  16,  // mma_trans_tile_size
  8,   // mma_orient_tile_size
  8>;   // mma_img_tile_size>


  int SM_num=baseMLO->multiProcessorCount;
  int block_num=SM_num*2;
// int block_num = ((translation_num + CT::kTransBlockSize - 1) / CT::kTransBlockSize) * ((projectorPlans[iclass].orientation_num + CT::kOrientBlockSize - 1) / CT::kOrientBlockSize);
// printf("#### image_size : %d\n", image_size);

cuda_kernel_coarse_matrix<CT>
<<<block_num, CT::kBlockSize, 0, ctx.classStreams[iclass]>>>(
    ~projectorPlans[iclass].eulers,
  &(~trans_xyz)[trans_x_offset], //~trans_x,
  &(~trans_xyz)[trans_y_offset], //~trans_y,
  &(~Fimg_)[img_re_offset], //~Fimg_real,
  &(~Fimg_)[img_im_offset], //~Fimg_imag,
  projKernel,
  ~corr_img,
  &(~allWeights)[allWeights_pos],
  &(~allWeights)[allWeights_pos],
  translation_num,
  projectorPlans[iclass].orientation_num,
  image_size,
  nullptr,
  nullptr,
  nullptr,
  nullptr
  );



#endif

// allWeights.hostAlloc();
// allWeights.cpToHost();
// allWeights.streamSync();
// cudaStreamSynchronize(ctx.classStreams[iclass]);
// for(int i=0;i<10;i++)
// {
//     printf("%lf ", allWeights[allWeights_pos+i]);
// }
// printf("\n");     


#ifdef TIMING
if(ctx.thread_id==0)
    baseMLO->timer.toc(baseMLO->TIMING_EXTRA1);
// baseMLO->timer.toc(baseMLO->TIMING_EXTRA1_T[thread_id]);
#endif
            mapAllWeightsToMweights(
                    ~projectorPlans[iclass].iorientclasses,
                    &(~allWeights)[allWeights_pos],
                    &(~Mweight)[img_id*weightsPerPart],
                    projectorPlans[iclass].orientation_num,
                    translation_num,
                    // accMLO->classStreams[iclass]
                    ctx.classStreams[iclass]
                    );

            /*====================================
                        Retrieve Results
            ======================================*/
            allWeights_pos += projectorPlans[iclass].orientation_num*translation_num;

        }
    }
#ifdef TIMING
	// if (op.part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
		baseMLO->timer.toc(baseMLO->TIMING_ESP_DIFF1);
#endif

    CTOC(ctx.accMLO->timer,"getAllSquaredDifferencesCoarsePostPerImgLaunchKernel");
}

template <class MlClass>
void getAllSquaredDifferencesCoarsePostPerImgSync(Context<MlClass>& ctx) {
    CTIC(ctx.accMLO->timer,"getAllSquaredDifferencesCoarsePostPerImgSync");
	OptimisationParamters& op = ctx.op;
	SamplingParameters& sp    = ctx.sp;
	MlClass *accMLO           = ctx.accMLO;
    int thread_id             = ctx.thread_id;
    MlOptimiser* baseMLO      = ctx.baseMLO;
#ifdef TIMING
	// if (op.part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
		baseMLO->timer.tic(baseMLO->TIMING_ESP_DIFF1);
#endif

    for (unsigned long exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++) {
        DEBUG_HANDLE_ERROR(cudaStreamSynchronize(accMLO->classStreams[exp_iclass]));
        DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.classStreams[exp_iclass]));
    }
    DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.cudaStreamPerTask));
    DEBUG_HANDLE_ERROR(cudaStreamSynchronize(cudaStreamPerThread)); // does not appear to be NEEDED FOR NON-BLOCKING CLASS STREAMS in tests, but should be to sync against classStreams
#ifdef TIMING
	// if (op.part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
		baseMLO->timer.toc(baseMLO->TIMING_ESP_DIFF1);
#endif
    CTOC(ctx.accMLO->timer,"getAllSquaredDifferencesCoarsePostPerImgSync");
}

template <class MlClass>
void getAllSquaredDifferencesCoarsePostPerImgGetMin(Context<MlClass>& ctx)
{
    CTIC(ctx.accMLO->timer,"getAllSquaredDifferencesCoarsePostPerImgGetMin");
    int thread_id             = ctx.thread_id;
    OptimisationParamters& op = ctx.op;
    int img_id                = ctx.img_id;
    MlOptimiser* baseMLO      = ctx.baseMLO;
#ifdef NEWMEM
    AccPtrNew<XFLOAT> allWeights = ctx.allWeights;
#else
    AccPtr<XFLOAT> allWeights = ctx.allWeights;
#endif



#ifdef TIMING
	// if (op.part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
		baseMLO->timer.tic(baseMLO->TIMING_ESP_DIFF1);
#endif
    op.min_diff2[img_id] = AccUtilities::getMinOnDevice<XFLOAT>(allWeights);
    // float maxx=AccUtilities::getMaxOnDevice<XFLOAT>(allWeights);
    // printf("#### check coarse op.min_diff2[img_id : %1d] = %8e max%e %d\n", 
    //         img_id, op.min_diff2[img_id],maxx,allWeights.getSize());

    ctx.corr_img.freeIfSet();
    ctx.trans_xyz.freeIfSet();
    ctx.Fimg_.freeIfSet();

#ifdef TIMING
	// if (op.part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
		baseMLO->timer.toc(baseMLO->TIMING_ESP_DIFF1);
#endif
    CTOC(ctx.accMLO->timer,"getAllSquaredDifferencesCoarsePostPerImgGetMin");
}
#endif  // ACC_COARSE_H_