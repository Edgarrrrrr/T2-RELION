#ifndef ACC_FINE_H_
#define ACC_FINE_H_

#include "src/ml_optimiser_mpi.h"
#include "src/acc/acc_context.h"
#include "src/acc/acc_boo.h"
#include "src/acc/cuda/cuda_kernels/fine_splitImg_BCF_PMO.cuh"

#include <chrono>

// ----------------------------------------------------------------------------
// -------------------- getAllSquaredDifferencesFine --------------------------
// ----------------------------------------------------------------------------
template <class MlClass>
void getAllSquaredDifferencesFinePre(Context<MlClass>& ctx)
{ 
	CTIC(ctx.accMLO->timer,"getAllSquaredDifferencesFinePre");
#ifdef TIMING
	// if (op.part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
		ctx.baseMLO->timer.tic(ctx.baseMLO->TIMING_ESP_DIFF2);
#endif

    unsigned exp_ipass = ctx.exp_ipass;
    OptimisationParamters &op = ctx.op;
    SamplingParameters &sp = ctx.sp;
    MlOptimiser *baseMLO = ctx.baseMLO;
    MlClass *accMLO = ctx.accMLO;
    std::vector<std::vector< IndexedDataArrayMask > > &FPCMasks = *ctx.FinePassClassMasks;
    // std::vector<ProjectionParams> &FineProjectionData = *ctx.FineProjectionData;
    int ibody = ctx.ibody;

#ifdef NEWMEM
    AccPtrFactoryNew& ptrFactory = ctx.ptrFactory;
    std::vector<IndexedDataArrayNew > &FinePassWeights = *ctx.FinePassWeights;
    // std::vector<IndexedDataArray > &FinePassWeights = *ctx.FinePassWeights;

    std::vector<AccPtrBundleNew > &bundleD2 = *ctx.bundleD2;
    // std::vector<AccPtrBundle > &bundleD2 = *ctx.bundleD2;
#else
    AccPtrFactory& ptrFactory = ctx.ptrFactory;
    std::vector<IndexedDataArray > &FinePassWeights = *ctx.FinePassWeights;
    std::vector<AccPtrBundle > &bundleD2 = *ctx.bundleD2;
#endif

    int thread_id = ctx.thread_id;

#ifdef TIMING
	// if (op.part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
		ctx.baseMLO->timer.tic(ctx.baseMLO->TIMING_ESP_DIFF2);
#endif

	CUSTOM_ALLOCATOR_REGION_NAME("DIFF_FINE");
	CTIC(accMLO->timer,"diff_pre_gpu");

	CTIC(accMLO->timer,"precalculateShiftedImagesCtfsAndInvSigma2s");
	std::vector<MultidimArray<Complex > > dummy;
	std::vector<std::vector<MultidimArray<Complex > > > dummy2;
	std::vector<MultidimArray<RFLOAT> > dummyRF;
	baseMLO->precalculateShiftedImagesCtfsAndInvSigma2s(false, false, op.part_id, sp.current_oversampling, op.metadata_offset, // inserted SHWS 12112015
			sp.itrans_min, sp.itrans_max, op.Fimg, dummy, op.Fctf, dummy2, dummy2,
			op.local_Fctf, op.local_sqrtXi2, op.local_Minvsigma2, op.FstMulti, dummyRF);
	CTOC(accMLO->timer,"precalculateShiftedImagesCtfsAndInvSigma2s");

	CTOC(accMLO->timer,"diff_pre_gpu");

#ifdef TIMING
	// if (op.part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
		ctx.baseMLO->timer.toc(ctx.baseMLO->TIMING_ESP_DIFF2);
#endif
	CTOC(ctx.accMLO->timer,"getAllSquaredDifferencesFinePre");
}

template <class MlClass>
void getAllSquaredDifferencesFinePostPerImg(Context<MlClass>& ctx)
{
	CTIC(ctx.accMLO->timer,"getAllSquaredDifferencesFinePostPerImg");
#ifdef TIMING
	// if (op.part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
		ctx.baseMLO->timer.tic(ctx.baseMLO->TIMING_ESP_DIFF2);
#endif
    // Reset size without de-allocating: we will append everything significant within
    // the current allocation and then re-allocate the then determined (smaller) volume
    unsigned exp_ipass = ctx.exp_ipass;
    OptimisationParamters &op = ctx.op;
    SamplingParameters &sp = ctx.sp;
    MlOptimiser *baseMLO = ctx.baseMLO;
    MlClass *accMLO = ctx.accMLO;
    std::vector<std::vector< IndexedDataArrayMask > > &FPCMasks = *ctx.FinePassClassMasks;
    std::vector<ProjectionParams> &FineProjectionData = *ctx.FineProjectionData;
    int ibody = ctx.ibody;
#ifdef NEWMEM
    std::vector<AccPtrBundleNew > &bundleD2 = *ctx.bundleD2;
    // std::vector<AccPtrBundle > &bundleD2 = *ctx.bundleD2;
    // std::vector<IndexedDataArray > &FinePassWeights = *ctx.FinePassWeights;
    std::vector<IndexedDataArrayNew > &FinePassWeights = *ctx.FinePassWeights;
    AccPtrFactoryNew& ptrFactory = ctx.ptrFactory;
#else
    std::vector<AccPtrBundle > &bundleD2 = *ctx.bundleD2;
    std::vector<IndexedDataArray > &FinePassWeights = *ctx.FinePassWeights;
    AccPtrFactory& ptrFactory = ctx.ptrFactory;
#endif
    int thread_id = ctx.thread_id;
    int img_id = ctx.img_id;

    int my_metadata_offset = op.metadata_offset + img_id;
    long int group_id = baseMLO->mydata.getGroupId(op.part_id, img_id);
    RFLOAT my_pixel_size = baseMLO->mydata.getImagePixelSize(op.part_id, img_id);
    int optics_group = baseMLO->mydata.getOpticsGroup(op.part_id, img_id);
    // unsigned long image_size = op.local_Minvsigma2[img_id].nzyxdim;

    ctx.image_size = op.local_Minvsigma2[img_id].nzyxdim;
    unsigned long& image_size = ctx.image_size;


    MultidimArray<Complex > Fref;
    Fref.resize(op.local_Minvsigma2[img_id]);

    /*====================================
            Generate Translations
    ======================================*/

    CTIC(accMLO->timer,"translation_2");

    // long unsigned translation_num((sp.itrans_max - sp.itrans_min + 1) * sp.nr_oversampled_trans);
    ctx.translation_num = (sp.itrans_max - sp.itrans_min + 1) * sp.nr_oversampled_trans;
    long unsigned& translation_num = ctx.translation_num;
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

    // ctx.Fimg_.freeIfSet();
    // ctx.trans_xyz.freeIfSet();
    // AccPtr<XFLOAT> Fimg_     = ptrFactory.make<XFLOAT>((size_t)image_size*2);
    // AccPtr<XFLOAT> trans_xyz = ptrFactory.make<XFLOAT>((size_t)translation_num*3);
    ctx.Fimg_ = ptrFactory.make<XFLOAT>((size_t)image_size*2);
    ctx.trans_xyz = ptrFactory.make<XFLOAT>((size_t)translation_num*3);
    ctx.rearranged_trans_xyz = ptrFactory.make<XFLOAT>();
    ctx.TransRearrangedIndex = ptrFactory.make<size_t>();
#ifdef NEWMEM
    AccPtrNew<XFLOAT> &Fimg_ = ctx.Fimg_;
    AccPtrNew<XFLOAT> &trans_xyz = ctx.trans_xyz;
    AccPtrNew<XFLOAT> &rearranged_trans_xyz = ctx.rearranged_trans_xyz;
    AccPtrNew<size_t> &TransRearrangedIndex = ctx.TransRearrangedIndex;
#else
    AccPtr<XFLOAT> &Fimg_ = ctx.Fimg_;
    AccPtr<XFLOAT> &trans_xyz = ctx.trans_xyz;
    AccPtr<XFLOAT> &rearranged_trans_xyz = ctx.rearranged_trans_xyz;
    AccPtr<size_t> &TransRearrangedIndex = ctx.TransRearrangedIndex;
#endif
    // Fimg_.allAlloc();
    // trans_xyz.allAlloc();
    Fimg_.hostAlloc();
    trans_xyz.hostAlloc();

    std::vector<RFLOAT> oversampled_translations_x, oversampled_translations_y, oversampled_translations_z;

    int j = 0;
    for (long int itrans = 0; itrans < (sp.itrans_max - sp.itrans_min + 1); itrans++)
    {
        baseMLO->sampling.getTranslationsInPixel(itrans, baseMLO->adaptive_oversampling, my_pixel_size, oversampled_translations_x,
                oversampled_translations_y, oversampled_translations_z,
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

    XFLOAT scale_correction = baseMLO->do_scale_correction ? baseMLO->mymodel.scale_correction[group_id] : 1;

    int exp_current_image_size = (baseMLO->strict_highres_exp > 0.) ? baseMLO->image_coarse_size[optics_group] : baseMLO->image_current_size[optics_group];
    MultidimArray<Complex > Fimg, Fimg_nomask;
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

    CTOC(accMLO->timer,"translation_2");


    CTIC(accMLO->timer,"kernel_init_1");

    // ctx.corr_img.freeIfSet();
    // AccPtr<XFLOAT> corr_img = ptrFactory.make<XFLOAT>((size_t)image_size);
    ctx.corr_img = ptrFactory.make<XFLOAT>((size_t)image_size);
#ifdef NEWMEM
    AccPtrNew<XFLOAT>& corr_img = ctx.corr_img;
#else
    AccPtr<XFLOAT>& corr_img = ctx.corr_img;
#endif
    // corr_img.allAlloc();
    corr_img.hostAlloc();
    buildCorrImage(baseMLO,op,corr_img,img_id,group_id);

    // trans_xyz.cpToDevice();


    // Fimg_.cpToDevice();
    // corr_img.cpToDevice();

    CTOC(accMLO->timer,"kernel_init_1");

    // std::vector< AccPtr<XFLOAT> > eulers((size_t)(sp.iclass_max-sp.iclass_min+1), ptrFactory.make<XFLOAT>());
    // ctx.eulers = new std::vector< AccPtr<XFLOAT> >((size_t)(sp.iclass_max-sp.iclass_min+1), ptrFactory.make<XFLOAT>());
    // std::vector< AccPtr<XFLOAT> > &eulers = *ctx.eulers;
    

#ifdef NEWMEM
    ctx.eulers = new std::vector< AccPtrNew<XFLOAT> >((size_t)(sp.iclass_max-sp.iclass_min+1), ptrFactory.make<XFLOAT>());
    std::vector< AccPtrNew<XFLOAT> > &eulers = *ctx.eulers;
	ctx.AllEulers = new AccPtrBundleNew(ptrFactory.makeBundle());
    AccPtrBundleNew &AllEulers = *ctx.AllEulers;
    ctx.AllData = new AccPtrBundleNew(ptrFactory.makeBundle());
    AccPtrBundleNew &AllData = *ctx.AllData;
	// ctx.AllEulers = new AccPtrBundle(((AccPtrFactory)ptrFactory).makeBundle());
    // AccPtrBundle &AllEulers = *ctx.AllEulers;
    ctx.rearranged_eulers = new std::vector< AccPtrNew<XFLOAT> >((size_t)(sp.iclass_max-sp.iclass_min+1), ptrFactory.make<XFLOAT>());
    std::vector< AccPtrNew<XFLOAT> > &rearranged_eulers = *ctx.rearranged_eulers;
    ctx.OrientRearrangedIndex = new std::vector< AccPtrNew<size_t> >((size_t)(sp.iclass_max-sp.iclass_min+1), ptrFactory.make<size_t>());
    std::vector< AccPtrNew<size_t> > &OrientRearrangedIndex = *ctx.OrientRearrangedIndex;
    ctx.CoarseIndex2RotId = new std::vector< AccPtrNew<size_t> >((size_t)(sp.iclass_max-sp.iclass_min+1), ptrFactory.make<size_t>());
    std::vector< AccPtrNew<size_t> > &CoarseIndex2RotId = *ctx.CoarseIndex2RotId;
    ctx.blocks64x128 = new std::vector< AccPtrNew<Block<16, 4, 8>> >((size_t)(sp.iclass_max-sp.iclass_min+1), ptrFactory.make<Block<16, 4, 8>>());
    std::vector< AccPtrNew<Block<16, 4, 8>> > &blocks64x128 = *ctx.blocks64x128;
    ctx.blocks32x64 = new std::vector< AccPtrNew<Block<8, 4, 8>> >((size_t)(sp.iclass_max-sp.iclass_min+1), ptrFactory.make<Block<8, 4, 8>>());
    std::vector< AccPtrNew<Block<8, 4, 8>> > &blocks32x64 = *ctx.blocks32x64;
    ctx.blocks16x32 = new std::vector< AccPtrNew<Block<4, 4, 8>> >((size_t)(sp.iclass_max-sp.iclass_min+1), ptrFactory.make<Block<4, 4, 8>>());
    std::vector< AccPtrNew<Block<4, 4, 8>> > &blocks16x32 = *ctx.blocks16x32;
#else
    ctx.eulers = new std::vector< AccPtr<XFLOAT> >((size_t)(sp.iclass_max-sp.iclass_min+1), ptrFactory.make<XFLOAT>());
    std::vector< AccPtr<XFLOAT> > &eulers = *ctx.eulers;
	ctx.AllEulers = new AccPtrBundle(ptrFactory.makeBundle());
    AccPtrBundle &AllEulers = *ctx.AllEulers;
    ctx.AllData = new AccPtrBundle(ptrFactory.makeBundle());
    AccPtrBundle &AllData = *ctx.AllData;
    ctx.rearranged_eulers = new std::vector< AccPtr<XFLOAT> >((size_t)(sp.iclass_max-sp.iclass_min+1), ptrFactory.make<XFLOAT>());
    std::vector< AccPtr<XFLOAT> > &rearranged_eulers = *ctx.rearranged_eulers;
    ctx.OrientRearrangedIndex = new std::vector< AccPtr<size_t> >((size_t)(sp.iclass_max-sp.iclass_min+1), ptrFactory.make<size_t>());
    std::vector< AccPtr<size_t> > &OrientRearrangedIndex = *ctx.OrientRearrangedIndex;
    ctx.CoarseIndex2RotId = new std::vector< AccPtr<size_t> >((size_t)(sp.iclass_max-sp.iclass_min+1), ptrFactory.make<size_t>());
    std::vector< AccPtr<size_t> > &CoarseIndex2RotId = *ctx.CoarseIndex2RotId;
    ctx.blocks64x128 = new std::vector< AccPtr<Block<16, 4, 8>> >((size_t)(sp.iclass_max-sp.iclass_min+1), ptrFactory.make<Block<16, 4, 8>>());
    std::vector< AccPtr<Block<16, 4, 8>> > &blocks64x128 = *ctx.blocks64x128;
    ctx.blocks32x64 = new std::vector< AccPtr<Block<8, 4, 8>> >((size_t)(sp.iclass_max-sp.iclass_min+1), ptrFactory.make<Block<8, 4, 8>>());
    std::vector< AccPtr<Block<8, 4, 8>> > &blocks32x64 = *ctx.blocks32x64;
    ctx.blocks16x32 = new std::vector< AccPtr<Block<4, 4, 8>> >((size_t)(sp.iclass_max-sp.iclass_min+1), ptrFactory.make<Block<4, 4, 8>>());
    std::vector< AccPtr<Block<4, 4, 8>> > &blocks16x32 = *ctx.blocks16x32;
#endif
    // AllEulers.setSize(9*FineProjectionData[img_id].orientationNumAllClasses*sizeof(XFLOAT));
    // AllEulers.allAlloc();

    // unsigned long newDataSize(0);
    ctx.newDataSize = 0;
    unsigned long& newDataSize = ctx.newDataSize;

    size_t Extend_rows(0);
    size_t Extend_cols[sp.iclass_max-sp.iclass_min+1];
    memset(Extend_cols, 0, sizeof(size_t)*(sp.iclass_max-sp.iclass_min+1));

    unsigned long allDataSize(0);

    for (unsigned long exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
    {
        FPCMasks[img_id][exp_iclass].weightNum=0;

        if ((baseMLO->mymodel.pdf_class[exp_iclass] > 0.) && (FineProjectionData[img_id].class_entries[exp_iclass] > 0) )
        {
            // use "slice" constructor with class-specific parameters to retrieve a temporary ProjectionParams with data for this class
            ProjectionParams thisClassProjectionData(	FineProjectionData[img_id],
                                                        FineProjectionData[img_id].class_idx[exp_iclass],
                                                        FineProjectionData[img_id].class_idx[exp_iclass]+FineProjectionData[img_id].class_entries[exp_iclass]);

            // printf("#### original projectiondata : %x  slice projectiondata : %x\n", &FineProjectionData[img_id].rots[FineProjectionData[img_id].class_idx[exp_iclass]], &thisClassProjectionData.rots[0]);
            // since we retrieved the ProjectionParams for *the whole* class the orientation_num is also equal.

            thisClassProjectionData.orientation_num[0] = FineProjectionData[img_id].class_entries[exp_iclass];
            long unsigned orientation_num  = thisClassProjectionData.orientation_num[0];

            if(orientation_num==0)
                continue;

            CTIC(accMLO->timer,"pair_list_1");
            long unsigned significant_num(0);
            long int nr_over_orient = baseMLO->sampling.oversamplingFactorOrientations(sp.current_oversampling);
            long int nr_over_trans = baseMLO->sampling.oversamplingFactorTranslations(sp.current_oversampling);
            // Prepare the mask of the weight-array for this class
            if (FPCMasks[img_id][exp_iclass].weightNum==0)
                FPCMasks[img_id][exp_iclass].firstPos = newDataSize;

            long unsigned ihidden(0);
            std::vector< long unsigned > iover_transes, ihiddens;

            for (long int itrans = sp.itrans_min; itrans <= sp.itrans_max; itrans++, ihidden++)
            {
                for (long int iover_trans = 0; iover_trans < sp.nr_oversampled_trans; iover_trans++)
                {
                    ihiddens.push_back(ihidden);
                    iover_transes.push_back(iover_trans);
                }
            }

            int chunkSize(0);
            if(accMLO->dataIs3D)
                chunkSize = D2F_CHUNK_DATA3D;
            else if(accMLO->refIs3D)
                chunkSize = D2F_CHUNK_DATA3D;
            else
                chunkSize = D2F_CHUNK_2D;

            size_t extend_rows(translation_num / nr_over_trans);
            size_t extend_cols(orientation_num / nr_over_orient);

            significant_num =   cutSignificants2Blocks<4, 8>(op, sp, 
                                orientation_num, translation_num,
                                thisClassProjectionData, 
                                img_id,
                                extend_rows, extend_cols,
                                TransRearrangedIndex, OrientRearrangedIndex[exp_iclass-sp.iclass_min],
                                CoarseIndex2RotId[exp_iclass-sp.iclass_min],
                                blocks64x128[exp_iclass-sp.iclass_min],
                                blocks32x64[exp_iclass-sp.iclass_min],
                                blocks16x32[exp_iclass-sp.iclass_min],
                                FinePassWeights[img_id],
                                FPCMasks[img_id][exp_iclass],
                                chunkSize);

            allDataSize += blocks64x128[exp_iclass-sp.iclass_min].getSize() * sizeof(Block<16, 4, 8>);
            allDataSize += blocks32x64[exp_iclass-sp.iclass_min].getSize() * sizeof(Block<8, 4, 8>);
            allDataSize += blocks16x32[exp_iclass-sp.iclass_min].getSize() * sizeof(Block<4, 4, 8>);
            
            Extend_rows = (extend_rows > Extend_rows) ? extend_rows : Extend_rows;
            Extend_cols[exp_iclass-sp.iclass_min] = extend_cols;

//             // Do more significance checks on translations and create jobDivision
// #ifdef NEWMEM
//             significant_num = makeJobsForDiff2Fine1(	op,	sp,												// alot of different type inputs...
// #else
//             significant_num = makeJobsForDiff2Fine(	op,	sp,												// alot of different type inputs...
// #endif //_xjldebug46
//                                                     orientation_num, translation_num,
//                                                     thisClassProjectionData,
//                                                     iover_transes, ihiddens,
//                                                     nr_over_orient, nr_over_trans, img_id,
//                                                     FinePassWeights[img_id],
//                                                     FPCMasks[img_id][exp_iclass],   // ..and output into index-arrays mask...
//                                                     chunkSize);                    // ..based on a given maximum chunk-size

            // extend size by number of significants found this class
            // printf("%d\n", significant_num);
            newDataSize += significant_num;
            FPCMasks[img_id][exp_iclass].weightNum = significant_num;
            FPCMasks[img_id][exp_iclass].lastPos = FPCMasks[img_id][exp_iclass].firstPos + significant_num;
            CTOC(accMLO->timer,"pair_list_1");

            // CTIC(accMLO->timer,"IndexedArrayMemCp2");
            // bundleD2[img_id].pack(FPCMasks[img_id][exp_iclass].jobOrigin);
            // bundleD2[img_id].pack(FPCMasks[img_id][exp_iclass].jobExtent);
            // CTOC(accMLO->timer,"IndexedArrayMemCp2");

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

            CTIC(accMLO->timer,"generateEulerMatrices");
            eulers[exp_iclass-sp.iclass_min].setSize(9*FineProjectionData[img_id].class_entries[exp_iclass]);
            eulers[exp_iclass-sp.iclass_min].hostAlloc();

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
                    &(eulers[exp_iclass-sp.iclass_min])[0],
                    true,
                    MBL,
                    MBR);

            // AllEulers.pack(eulers[exp_iclass-sp.iclass_min]);

            CTOC(accMLO->timer,"generateEulerMatrices");
            
            rearrangeOrientation(OrientRearrangedIndex[exp_iclass-sp.iclass_min], eulers[exp_iclass-sp.iclass_min], rearranged_eulers[exp_iclass-sp.iclass_min], orientation_num, nr_over_orient, Extend_cols[exp_iclass-sp.iclass_min]);
            allDataSize += rearranged_eulers[exp_iclass-sp.iclass_min].getSize() * sizeof(XFLOAT);
            allDataSize += OrientRearrangedIndex[exp_iclass-sp.iclass_min].getSize() * sizeof(size_t);
            allDataSize += CoarseIndex2RotId[exp_iclass-sp.iclass_min].getSize() * sizeof(size_t);
        }
    }

    rearrangeTranslation(TransRearrangedIndex, trans_xyz, rearranged_trans_xyz, translation_num, sp.nr_oversampled_trans, Extend_rows);
    allDataSize += rearranged_trans_xyz.getSize() * sizeof(XFLOAT);
    allDataSize += TransRearrangedIndex.getSize() * sizeof(size_t);

    allDataSize += Fimg_.getSize() * sizeof(XFLOAT);
    allDataSize += corr_img.getSize() * sizeof(XFLOAT);

    AllData.setSize(allDataSize);
    AllData.allAlloc();
    for (unsigned long exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
    {
        AllData.pack(blocks64x128[exp_iclass-sp.iclass_min]);
        AllData.pack(blocks32x64[exp_iclass-sp.iclass_min]);
        AllData.pack(blocks16x32[exp_iclass-sp.iclass_min]);
        AllData.pack(rearranged_eulers[exp_iclass-sp.iclass_min]);
        AllData.pack(OrientRearrangedIndex[exp_iclass-sp.iclass_min]);
        AllData.pack(CoarseIndex2RotId[exp_iclass-sp.iclass_min]);
    }
    AllData.pack(rearranged_trans_xyz);
    AllData.pack(TransRearrangedIndex);
    AllData.pack(Fimg_);
    AllData.pack(corr_img);

    // trans_xyz.streamSync(); // NO USE
    // Fimg_.streamSync(); // NO USE
    // corr_img.streamSync(); // corr_img, trans_xyz, Fimg_ have the same stream

#ifdef TIMING
	// if (op.part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
		ctx.baseMLO->timer.toc(ctx.baseMLO->TIMING_ESP_DIFF2);
#endif
	CTOC(ctx.accMLO->timer,"getAllSquaredDifferencesFinePostPerImg");

}

template <class MlClass>
void getAllSquaredDifferencesFinePostPerImgMemcpyHtoD(Context<MlClass>& ctx)
{
	CTIC(ctx.accMLO->timer,"getAllSquaredDifferencesFinePostPerImgMemcpyHtoD");
#ifdef TIMING
	// if (op.part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
		ctx.baseMLO->timer.tic(ctx.baseMLO->TIMING_ESP_DIFF2);
#endif
	int img_id = ctx.img_id;
#ifdef NEWMEM
	std::vector<AccPtrBundleNew > &bundleD2 = *ctx.bundleD2;
	AccPtrBundleNew &AllEulers = *ctx.AllEulers;
    AccPtrBundleNew &AllData = *ctx.AllData;

	std::vector< IndexedDataArrayNew > &FinePassWeights = *ctx.FinePassWeights;
    // std::vector<AccPtrBundle > &bundleD2 = *ctx.bundleD2;
	// AccPtrBundle &AllEulers = *ctx.AllEulers;
	// std::vector< IndexedDataArray > &FinePassWeights = *ctx.FinePassWeights;

#else
	std::vector<AccPtrBundle > &bundleD2 = *ctx.bundleD2;
	AccPtrBundle &AllEulers = *ctx.AllEulers;
    AccPtrBundle &AllData = *ctx.AllData;
	std::vector< IndexedDataArray > &FinePassWeights = *ctx.FinePassWeights;
#endif

    // check
	// bundleD2[img_id].cpToDevice();
    // AllEulers.cpToDevice();
    // FinePassWeights[img_id].rot_id.cpToDevice(); //FIXME this is not used
    // FinePassWeights[img_id].rot_idx.cpToDevice();
    // FinePassWeights[img_id].trans_idx.cpToDevice();
    // check end
    AllData.cpToDevice();
    
#ifdef TIMING
	// if (op.part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
		ctx.baseMLO->timer.toc(ctx.baseMLO->TIMING_ESP_DIFF2);
#endif
	CTOC(ctx.accMLO->timer,"getAllSquaredDifferencesFinePostPerImgMemcpyHtoD");
}

template <class MlClass>
void getAllSquaredDifferencesFinePostPerImgMemcpyHtoDSync(Context<MlClass>& ctx)
{
    CTIC(ctx.accMLO->timer,"getAllSquaredDifferencesFinePostPerImgMemcpyHtoDSync");
#ifdef TIMING
	// if (op.part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
		ctx.baseMLO->timer.tic(ctx.baseMLO->TIMING_ESP_DIFF2);
#endif

	SamplingParameters &sp = ctx.sp;
	for (unsigned long exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
		DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.classStreams[exp_iclass]));
	DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.cudaStreamPerTask));

#ifdef TIMING
	// if (op.part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
		ctx.baseMLO->timer.toc(ctx.baseMLO->TIMING_ESP_DIFF2);
#endif		

	CTOC(ctx.accMLO->timer,"getAllSquaredDifferencesFinePostPerImgMemcpyHtoDSync");
	LAUNCH_HANDLE_ERROR(cudaGetLastError());

}

template <class MlClass>
void getAllSquaredDifferencesFinePostPerImgLaunchKernel(Context<MlClass>& ctx)
{
    CTIC(ctx.accMLO->timer,"getAllSquaredDifferencesFinePostPerImgLaunchKernel");
#ifdef TIMING
	// if (op.part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
		ctx.baseMLO->timer.tic(ctx.baseMLO->TIMING_ESP_DIFF2);
#endif

#ifdef NEWMEM
	AccPtrNew<XFLOAT>& Fimg_ = ctx.Fimg_;
    AccPtrNew<XFLOAT>& trans_xyz = ctx.trans_xyz;
    AccPtrNew<XFLOAT>& corr_img = ctx.corr_img;
    // AccPtrNew<XFLOAT>& allWeights = ctx.allWeights;
    AccPtrFactoryNew& ptrFactory = ctx.ptrFactory;
	std::vector <IndexedDataArrayNew >& FinePassWeights = *ctx.FinePassWeights;
    // std::vector<IndexedDataArray > &FinePassWeights = *ctx.FinePassWeights;
	std::vector < std::vector <IndexedDataArrayMask > >& FPCMasks = *ctx.FinePassClassMasks;
	std::vector < ProjectionParams >& FineProjectionData = *ctx.FineProjectionData;
	std::vector < AccPtrNew<XFLOAT> >& eulers = *ctx.eulers;
    std::vector < AccPtrNew<XFLOAT> > &rearranged_eulers = *ctx.rearranged_eulers;
    AccPtrNew<XFLOAT> rearranged_trans_xyz = ctx.rearranged_trans_xyz;
    std::vector < AccPtrNew<Block<16, 4, 8>> >& blocks64x128 = *ctx.blocks64x128;
    std::vector < AccPtrNew<Block<8, 4, 8>> >& blocks32x64 = *ctx.blocks32x64;
    std::vector < AccPtrNew<Block<4, 4, 8>> >& blocks16x32 = *ctx.blocks16x32;
#else
	AccPtr<XFLOAT>& Fimg_ = ctx.Fimg_;
    AccPtr<XFLOAT>& trans_xyz = ctx.trans_xyz;
    AccPtr<XFLOAT>& corr_img = ctx.corr_img;
    // AccPtr<XFLOAT>& allWeights = ctx.allWeights;
    AccPtrFactory& ptrFactory = ctx.ptrFactory;
	std::vector <IndexedDataArray >& FinePassWeights = *ctx.FinePassWeights;
	std::vector < std::vector <IndexedDataArrayMask > >& FPCMasks = *ctx.FinePassClassMasks;
	std::vector < ProjectionParams >& FineProjectionData = *ctx.FineProjectionData;
	std::vector < AccPtr<XFLOAT> >& eulers = *ctx.eulers;
    std::vector < AccPtr<XFLOAT> > &rearranged_eulers = *ctx.rearranged_eulers;
    AccPtr<XFLOAT> rearranged_trans_xyz = ctx.rearranged_trans_xyz;
    std::vector < AccPtr<Block<16, 4, 8>> >& blocks64x128 = *ctx.blocks64x128;
    std::vector < AccPtr<Block<8, 4, 8>> >& blocks32x64 = *ctx.blocks32x64;
    std::vector < AccPtr<Block<4, 4, 8>> >& blocks16x32 = *ctx.blocks16x32;
#endif
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
	int img_id                 = ctx.img_id;



	for (unsigned long iclass = sp.iclass_min; iclass <= sp.iclass_max; iclass++)
	{
		int iproj;
		if (baseMLO->mymodel.nr_bodies > 1) iproj = ibody;
		else                                iproj = iclass;

		if ((baseMLO->mymodel.pdf_class[iclass] > 0.) && (FineProjectionData[img_id].class_entries[iclass] > 0) )
		{
			long unsigned orientation_num  = FineProjectionData[img_id].class_entries[iclass];
			if(orientation_num==0)
				continue;

			long unsigned significant_num(FPCMasks[img_id][iclass].weightNum);
			if(significant_num==0)
				continue;

			CTIC(accMLO->timer,"Diff2MakeKernel");
			AccProjectorKernel projKernel = AccProjectorKernel::makeKernel(
					accMLO->bundle->projectors[iproj],
					op.local_Minvsigma2[img_id].xdim,
					op.local_Minvsigma2[img_id].ydim,
					op.local_Minvsigma2[img_id].zdim,
					op.local_Minvsigma2[img_id].xdim-1);
			CTOC(accMLO->timer,"Diff2MakeKernel");

			// Use the constructed mask to construct a partial class-specific input
#ifdef NEWMEM
				// IndexedDataArray thisClassFinePassWeights(FinePassWeights[img_id],FPCMasks[img_id][iclass]);
				IndexedDataArrayNew thisClassFinePassWeights(FinePassWeights[img_id],FPCMasks[img_id][iclass]);
#else
				IndexedDataArray thisClassFinePassWeights(FinePassWeights[img_id],FPCMasks[img_id][iclass]);
#endif

			CTIC(accMLO->timer,"Diff2CALL");
#ifdef TIMING
if(ctx.thread_id==0)
	ctx.baseMLO->timer.tic(ctx.baseMLO->TIMING_EXTRA2);
    // ctx.baseMLO->timer.tic(ctx.baseMLO->TIMING_EXTRA2_T[ctx.thread_id]);
#endif


            // AccPtrNew<XFLOAT> test_weights = ptrFactory.make<XFLOAT>(thisClassFinePassWeights.weights.getSize());
            // test_weights.allAlloc();
            // deviceInitValue<XFLOAT>(test_weights, 0, test_weights.getSize(), ctx.classStreams[iclass]);
            deviceInitValue<XFLOAT>(thisClassFinePassWeights.weights, 0, thisClassFinePassWeights.weights.getSize(), ctx.classStreams[iclass]);
            // thisClassFinePassWeights.weights.cpToHost();
            // DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.classStreams[iclass]));
            // DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.cudaStreamPerTask));
            // test_weights.copyFrom(thisClassFinePassWeights.weights);
            

            // for (int s = 0; s < test_weights.getSize(); s++)
            // {
            //     test_weights[s] = 0;
            //     // if (test_weights[s] > 1e5) {
            //     //     printf("Error: test_weights[%d] = %f\n", s, test_weights[s]);
            //     // }
            //     // test_weights[s] += op.highres_Xi2_img[img_id] / 2.;
            // }

            // AccPtrNew<XFLOAT> tmp_test_weights = ptrFactory.make<XFLOAT>(thisClassFinePassWeights.weights.getSize());
            // tmp_test_weights.allAlloc();
            // tmp_test_weights.copyFrom(thisClassFinePassWeights.weights);

            // for (int s = 0; s < significant_num; s++)
            // {
            //     auto e = eulers[iclass-sp.iclass_min][thisClassFinePassWeights.rot_idx[s] * 9 + 0];
            //     for (int ss = 1; ss < 8; ss ++){
            //         e += eulers[iclass-sp.iclass_min][thisClassFinePassWeights.rot_idx[s] * 9 + ss];
            //     }
            //     auto t = trans_xyz[(trans_x_offset + thisClassFinePassWeights.trans_idx[s])];
            //     t += trans_xyz[(trans_y_offset + thisClassFinePassWeights.trans_idx[s])];
            //     tmp_test_weights[s] = e * t;
            // }

            // for (int b = 0; b < blocks64x128[iclass-sp.iclass_min].getSize(); b++)
            // {
            //     for (int r = 0; r < 64; r++)
            //     {
            //         for (int c = 0; c < 128; c++)
            //         {
            //             int fine_row_idx = blocks64x128[iclass-sp.iclass_min][b].startRow * 4 + r;
            //             int fine_col_idx = blocks64x128[iclass-sp.iclass_min][b].startCol * 8 + c;

            //             auto e = rearranged_eulers[iclass-sp.iclass_min][fine_col_idx * 9 + 0];
            //             auto t = rearranged_trans_xyz[fine_row_idx * 3 + 0];
            //             if (blocks64x128[iclass-sp.iclass_min][b].result_idx[r * 128 + c] != -1)
            //                 tmp_test_weights[blocks64x128[iclass-sp.iclass_min][b].result_idx[r * 128 + c]] = e * t;

            //         }
            //     }
            // }

            // test_weights.cpToDevice();
            // DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.classStreams[iclass]));
            // DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.cudaStreamPerTask));

			// runDiff2KernelFine(
					// projKernel,
					// ~corr_img,
					// &(~Fimg_)[img_re_offset], //~Fimg_real,
					// &(~Fimg_)[img_im_offset], //~Fimg_imag,
					// &(~trans_xyz)[trans_x_offset], //~trans_x,
					// &(~trans_xyz)[trans_y_offset], //~trans_y,
					// &(~trans_xyz)[trans_z_offset], //~trans_z,
					// ~eulers[iclass-sp.iclass_min],
					// ~thisClassFinePassWeights.rot_id,
					// ~thisClassFinePassWeights.rot_idx,
					// ~thisClassFinePassWeights.trans_idx,
					// ~FPCMasks[img_id][iclass].jobOrigin,
					// ~FPCMasks[img_id][iclass].jobExtent,
					// ~thisClassFinePassWeights.weights,
					// op,
					// baseMLO,
					// orientation_num,
					// translation_num,
					// significant_num,
					// image_size,
					// img_id,
					// iclass,
                    // ctx.classStreams[iclass],
					// FPCMasks[img_id][iclass].jobOrigin.getSize(),
					// ((baseMLO->iter == 1 && baseMLO->do_firstiter_cc) || baseMLO->do_always_cc),
					// accMLO->dataIs3D
					// );

            // DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.classStreams[iclass]));
            // DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.cudaStreamPerTask));
            // thisClassFinePassWeights.weights.cpToHost();
            // printf("weights length : %7d\n",thisClassFinePassWeights.weights.getSize());

            
            FineMatrixKernelIm2colSplitImgBCFProjOverlap<FineTParam64x128_32x64_4_8> kernel(
                translation_num,
                orientation_num,
                image_size,
                blocks64x128[iclass-sp.iclass_min].getSize(),
                108
            );

            kernel.run(
                ~rearranged_eulers[iclass-sp.iclass_min],
                ~rearranged_trans_xyz,
                ~blocks64x128[iclass-sp.iclass_min],
                &(~Fimg_)[img_re_offset], //~Fimg_real,
				&(~Fimg_)[img_im_offset], //~Fimg_imag,
                projKernel,
				~corr_img,
                ~thisClassFinePassWeights.weights,
                op.highres_Xi2_img[img_id] / 2.,
                ctx.classStreams[iclass]
            );

            // // 以1/1000概率随机输出message
            // if (rand() % 1000 == 0) {
            //     printf("fuck relion!\n");
            // }

            // DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.classStreams[iclass]));
            // DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.cudaStreamPerTask));
            
            // test_weights.cpToHost();
            
            // DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.classStreams[iclass]));
            // DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.cudaStreamPerTask));
            
            // double sum_relative_error = 0;

            // for (int i = 0; i < thisClassFinePassWeights.weights.getSize(); i++)
            // {
            //     XFLOAT a = thisClassFinePassWeights.weights[i];
            //     // XFLOAT a = tmp_test_weights[i];
            //     XFLOAT b = test_weights[i];

            //     if (abs(a - b) >= 1e-3 * abs(a)) {
            //         // printf("ERROR weights[%3d] = %12e, test_weights[%3d] = %12e\n", i, a, i, b);
            //         // fflush(stdout);
            //     }
            //     sum_relative_error += abs(a - b) / abs(a);
            // }
            // sum_relative_error /= thisClassFinePassWeights.weights.getSize();
            // printf("imgid : %7d average relative error = %12e\n", (int)img_id,  sum_relative_error);

            

            // computeKernelForOriginalMethod(
            //     trans_xyz,
            //     eulers[iclass-sp.iclass_min],
            //     thisClassFinePassWeights.rot_idx,
            //     thisClassFinePassWeights.trans_idx,
            //     thisClassFinePassWeights.weights,
            //     trans_x_offset, trans_y_offset, trans_z_offset,
            //     significant_num
            //     );

            // computeKernelForMyMethod<4,8>(
            //     blocks64x128[iclass-sp.iclass_min],
            //     blocks32x64[iclass-sp.iclass_min],
            //     blocks16x32[iclass-sp.iclass_min],
            //     rearranged_trans_xyz,
            //     rearranged_eulers[iclass-sp.iclass_min],
            //     FinePassWeights[img_id]
            // );

            
        // if (sum_relative_error >= 1e-4 || isnan(sum_relative_error)) {
        //     printf("block num : %8d\n", blocks64x128[iclass-sp.iclass_min].getSize());
        //     for (int r = 0; r < rearranged_trans_xyz.getSize(); r++)
        //     {
        //         printf("%9.2e ", rearranged_trans_xyz[r]);   
        //     }
        //     printf("\n");
            
        //     for(int b = 0; b < blocks64x128[iclass-sp.iclass_min].getSize(); b++)
        //     {
        //         printf("Block %d, startRow: %d, startCol: %d\n", b, blocks64x128[iclass-sp.iclass_min][b].startRow, blocks64x128[iclass-sp.iclass_min][b].startCol);

        //         for (int r = 0; r < 64; r++)
        //         {
        //             for (int xyz = 0; xyz < 3; xyz++)
        //             {
        //                 assert(((blocks64x128[iclass-sp.iclass_min][b].startRow * 4 + r) * 3 + xyz) < rearranged_trans_xyz.getSize());
        //                 printf("%9.2e ", rearranged_trans_xyz[(blocks64x128[iclass-sp.iclass_min][b].startRow * 4 + r) * 3 + xyz]);
        //             }
        //             printf("| ");
        //         }
        //         printf("\n");

        //         for (int c = 0; c < 128; c++)
        //         {
        //             for (int o = 0; o < 9; o++)
        //             {
        //                 printf("%9.2e ", rearranged_eulers[iclass-sp.iclass_min][(blocks64x128[iclass-sp.iclass_min][b].startCol * 8 + c) * 9 + o]);
        //             }
        //             printf("| ");
        //         }
        //         printf("\n");

        //         for (int r = 0; r < 64; r++)
        //         {
        //             printf("\n--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
        //             for (int c = 0; c < 128; c++)
        //             {
        //                 if (blocks64x128[iclass-sp.iclass_min][b].result_idx[r * 128 + c] == -1)
        //                     printf("          ");
        //                 else
        //                     printf("%9.2e ", test_weights[blocks64x128[iclass-sp.iclass_min][b].result_idx[r * 128 + c]]);
        //             }
        //             printf("\n");
        //             for (int c = 0; c < 128; c++)
        //             {
        //                 if (blocks64x128[iclass-sp.iclass_min][b].result_idx[r * 128 + c] == -1)
        //                     printf("          ");
        //                 else
        //                     printf("%9.2e ", thisClassFinePassWeights.weights[blocks64x128[iclass-sp.iclass_min][b].result_idx[r * 128 + c]]);
        //                     // printf("%9.2e ", tmp_test_weights[blocks64x128[iclass-sp.iclass_min][b].result_idx[r * 128 + c]]);
        //             }
        //             printf("\n");
        //         }   
        //         printf("\n");
        //     }
        // }

        // for(int b = 0; b < blocks32x64[iclass-sp.iclass_min].getSize(); b++)
        // {
        //     for (int r = 0; r < 32; r++)
        //     {
        //         for (int c = 0; c < 64; c++)
        //         {
        //             if (blocks32x64[iclass-sp.iclass_min][b].result_idx[r * 64 + c] == -1)
        //                 printf("%9.2e ", 999.0);
        //             else
        //                 printf("%9.2e ", thisClassFinePassWeights.weights[blocks32x64[iclass-sp.iclass_min][b].result_idx[r * 64 + c]]);
        //         }
        //         printf("\n");
        //     }   
        //     printf("\n");
        // }

        // for(int b = 0; b < blocks16x32[iclass-sp.iclass_min].getSize(); b++)
        // {
        //     for (int r = 0; r < 16; r++)
        //     {
        //         for (int c = 0; c < 32; c++)
        //         {
        //             if (blocks16x32[iclass-sp.iclass_min][b].result_idx[r * 32 + c] == -1)
        //                 printf("%9.2e ", 999.0);
        //             else
        //                 printf("%9.2e ", thisClassFinePassWeights.weights[blocks16x32[iclass-sp.iclass_min][b].result_idx[r * 32 + c]]);
        //         }
        //         printf("\n");
        //     }   
        //     printf("\n");
        // }



#ifdef TIMING
if(ctx.thread_id==0)
	ctx.baseMLO->timer.toc(ctx.baseMLO->TIMING_EXTRA2);
// ctx.baseMLO->timer.toc(ctx.baseMLO->TIMING_EXTRA2_T[ctx.thread_id]);

#endif
//				DEBUG_HANDLE_ERROR(cudaStreamSynchronize(cudaStreamPerThread));
			CTOC(accMLO->timer,"Diff2CALL");

		} // end if class significant
	} // end loop iclass

#ifdef TIMING
	// if (op.part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
		ctx.baseMLO->timer.toc(ctx.baseMLO->TIMING_ESP_DIFF2);
#endif
	CTOC(ctx.accMLO->timer,"getAllSquaredDifferencesFinePostPerImgLaunchKernel");
}

template <class MlClass>
void getAllSquaredDifferencesFinePostPerImgSync(Context<MlClass>& ctx) 
{
	CTIC(ctx.accMLO->timer,"getAllSquaredDifferencesFinePostPerImgSync");
#ifdef TIMING
	// if (op.part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
		ctx.baseMLO->timer.tic(ctx.baseMLO->TIMING_ESP_DIFF2);
#endif

	SamplingParameters &sp = ctx.sp;
	for (unsigned long exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
		// DEBUG_HANDLE_ERROR(cudaStreamSynchronize(accMLO->classStreams[exp_iclass]));
	// DEBUG_HANDLE_ERROR(cudaStreamSynchronize(cudaStreamPerThread));
		DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.classStreams[exp_iclass]));
	DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.cudaStreamPerTask));

#ifdef TIMING
	// if (op.part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
		ctx.baseMLO->timer.toc(ctx.baseMLO->TIMING_ESP_DIFF2);
#endif
	CTOC(ctx.accMLO->timer,"getAllSquaredDifferencesFinePostPerImgSync");
	LAUNCH_HANDLE_ERROR(cudaGetLastError());


}

template <class MlClass>
void getAllSquaredDifferencesFinePostPerImgGetMin(Context<MlClass>& ctx)
{
    CTIC(ctx.accMLO->timer,"getAllSquaredDifferencesFinePostPerImgGetMin");
#ifdef TIMING
	// if (op.part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
		ctx.baseMLO->timer.tic(ctx.baseMLO->TIMING_ESP_DIFF2);
#endif
#ifdef NEWMEM
	std::vector <IndexedDataArrayNew >& FinePassWeights = *ctx.FinePassWeights;
#else
	std::vector <IndexedDataArray >& FinePassWeights = *ctx.FinePassWeights;
#endif
	int img_id = ctx.img_id;
	unsigned long newDataSize = ctx.newDataSize;
	MlOptimiser* baseMLO      = ctx.baseMLO;
	MlClass *accMLO           = ctx.accMLO;
	OptimisationParamters& op = ctx.op;

	FinePassWeights[img_id].setDataSize( newDataSize );

	CTIC(accMLO->timer,"collect_data_1");
	if(baseMLO->adaptive_oversampling!=0)
	{
		op.min_diff2[img_id] = (RFLOAT) AccUtilities::getMinOnDevice<XFLOAT>(FinePassWeights[img_id].weights);
	}
	CTOC(accMLO->timer,"collect_data_1");

	delete ctx.eulers;

#ifdef NEWMEM
    // ctx.AllData->free();
    // ctx.AllEulers->free();
#endif//_xjldebug46
    // ctx.corr_img.freeIfSet();
	ctx.trans_xyz.freeIfSet();
	// ctx.Fimg_.freeIfSet();
    // delete ctx.AllEulers;
    // delete ctx.rearranged_eulers;
    // delete ctx.blocks64x128;
    // delete ctx.blocks32x64;
    // delete ctx.blocks16x32;

#ifdef TIMING
	// if (op.part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
		ctx.baseMLO->timer.toc(ctx.baseMLO->TIMING_ESP_DIFF2);
#endif
	CTOC(ctx.accMLO->timer,"getAllSquaredDifferencesFinePostPerImgGetMin");
}

#endif /* ACC_FINE_H_ */