static omp_lock_t global_mutex;

#include "src/ml_optimiser_mpi.h"
#include "src/acc/acc_context.h"
#include "src/acc/acc_coarse.h"
#include "src/acc/acc_fine.h"
#include "src/acc/acc_store_weight_sum.h"

// ----------------------------------------------------------------------------
// -------------------- getFourierTransformsAndCtfs ---------------------------
// ----------------------------------------------------------------------------
template <class MlClass>
void getFourierTransformsAndCtfs(long int part_id,
		OptimisationParamters &op,
		SamplingParameters &sp,
		MlOptimiser *baseMLO,
		MlClass *accMLO,
#ifdef NEWMEM
		AccPtrFactoryNew ptrFactory,
#else
		AccPtrFactory ptrFactory,
#endif
		Context<MlClass>& ctx, 
		int ibody = 0, int thread_id = 0)
{
#ifdef TIMING
	// if (part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
		baseMLO->timer.tic(baseMLO->TIMING_ESP_FT);
#endif

	CUSTOM_ALLOCATOR_REGION_NAME("GFTCTF");

	for (int img_id = 0; img_id < sp.nr_images; img_id++)
	{
		CTIC(accMLO->timer,"init");
		FileName fn_img;
		Image<RFLOAT> img, rec_img;
		MultidimArray<Complex > Fimg;
		MultidimArray<Complex > Faux;
		MultidimArray<RFLOAT> Fctf, FstMulti;
		Matrix2D<RFLOAT> Aori;
		Matrix1D<RFLOAT> my_projected_com(baseMLO->mymodel.data_dim), my_refined_ibody_offset(baseMLO->mymodel.data_dim);

		// Which group do I belong?
		int group_id =baseMLO->mydata.getGroupId(part_id, img_id);
		RFLOAT my_pixel_size = baseMLO->mydata.getImagePixelSize(part_id, img_id);
		// What is my optics group?
		int optics_group = baseMLO->mydata.getOpticsGroup(part_id, img_id);
		bool ctf_premultiplied = baseMLO->mydata.obsModel.getCtfPremultiplied(optics_group);

		// metadata offset for this image in the particle
		int my_metadata_offset = op.metadata_offset + img_id;

		// Get the right line in the exp_fn_img strings (also exp_fn_recimg and exp_fn_ctfs)
		int istop = 0;
		for (long int ii = baseMLO->exp_my_first_part_id; ii < part_id; ii++)
                    istop += baseMLO->mydata.numberOfImagesInParticle(part_id);
		istop += img_id;

		if (!baseMLO->mydata.getImageNameOnScratch(part_id, img_id, fn_img))
		{
			std::istringstream split(baseMLO->exp_fn_img);
			for (int i = 0; i <= my_metadata_offset; i++)
				getline(split, fn_img);
		}
		sp.current_img = fn_img;

		// Get the norm_correction
		RFLOAT normcorr = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_NORM);

		// Safeguard against gold-standard separation
		if (baseMLO->do_split_random_halves)
		{
			int halfset = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_NR_SIGN);
			if (halfset != baseMLO->my_halfset)
			{
				std::cerr << "BUG!!! halfset= " << halfset << " my_halfset= " << baseMLO->my_halfset << " part_id= " << part_id << std::endl;
				REPORT_ERROR("BUG! Mixing gold-standard separation!!!!");
			}

		}

		// Get the optimal origin offsets from the previous iteration
		// Sjors 5mar18: it is very important that my_old_offset has baseMLO->mymodel.data_dim and not just (3), as transformCartesianAndHelicalCoords will give different results!!!
		Matrix1D<RFLOAT> my_old_offset(baseMLO->mymodel.data_dim), my_prior(baseMLO->mymodel.data_dim), my_old_offset_ori;
		int icol_rot, icol_tilt, icol_psi, icol_xoff, icol_yoff, icol_zoff;
		XX(my_old_offset) = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_XOFF);
		YY(my_old_offset) = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_YOFF);
		XX(my_prior)      = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_XOFF_PRIOR);
		YY(my_prior)      = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_YOFF_PRIOR);
		// Uninitialised priors were set to 999.
		if (XX(my_prior) > 998.99 && XX(my_prior) < 999.01)
			XX(my_prior) = 0.;
		if (YY(my_prior) > 998.99 && YY(my_prior) < 999.01)
			YY(my_prior) = 0.;

		if (accMLO->dataIs3D)
		{
			ZZ(my_old_offset) = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_ZOFF);
			ZZ(my_prior)      = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_ZOFF_PRIOR);
			// Unitialised priors were set to 999.
			if (ZZ(my_prior) > 998.99 && ZZ(my_prior) < 999.01)
				ZZ(my_prior) = 0.;
		}

		if (baseMLO->mymodel.nr_bodies > 1)
		{

			// 17May2017: Shift image to the projected COM for this body!
			// Aori is the original transformation matrix of the consensus refinement
			Euler_angles2matrix(DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_ROT),
					            DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_TILT),
								DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_PSI), Aori, false);
			my_projected_com = Aori * baseMLO->mymodel.com_bodies[ibody];
			// This will have made my_projected_com of size 3 again! resize to mymodel.data_dim
			my_projected_com.resize(baseMLO->mymodel.data_dim);

			// Subtract the projected COM offset, to position this body in the center
			// Also keep the my_old_offset in my_old_offset_ori
			my_old_offset_ori = my_old_offset;
			my_old_offset -= my_projected_com;

			// Also get refined offset for this body
			icol_xoff = 3 + METADATA_LINE_LENGTH_BEFORE_BODIES + (ibody) * METADATA_NR_BODY_PARAMS;
			icol_yoff = 4 + METADATA_LINE_LENGTH_BEFORE_BODIES + (ibody) * METADATA_NR_BODY_PARAMS;
			icol_zoff = 5 + METADATA_LINE_LENGTH_BEFORE_BODIES + (ibody) * METADATA_NR_BODY_PARAMS;
			XX(my_refined_ibody_offset) = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, icol_xoff);
			YY(my_refined_ibody_offset) = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, icol_yoff);
			if (baseMLO->mymodel.data_dim == 3)
				ZZ(my_refined_ibody_offset) = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, icol_zoff);

			// For multi-body refinement: set the priors of the translations to zero (i.e. everything centred around consensus offset)
			my_prior.initZeros();
		}

		CTOC(accMLO->timer,"init");

		CTIC(accMLO->timer,"nonZeroProb");
		// Orientational priors
		if (baseMLO->mymodel.nr_bodies > 1 )
		{

			// Centre local searches around the orientation from the previous iteration, this one goes with overall sigma2_ang
			// On top of that, apply prior on the deviation from (0,0,0) with mymodel.sigma_tilt_bodies[ibody] and mymodel.sigma_psi_bodies[ibody]
			icol_rot  = 0 + METADATA_LINE_LENGTH_BEFORE_BODIES + (ibody) * METADATA_NR_BODY_PARAMS;
			icol_tilt = 1 + METADATA_LINE_LENGTH_BEFORE_BODIES + (ibody) * METADATA_NR_BODY_PARAMS;
			icol_psi  = 2 + METADATA_LINE_LENGTH_BEFORE_BODIES + (ibody) * METADATA_NR_BODY_PARAMS;
			RFLOAT prior_rot  = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, icol_rot);
			RFLOAT prior_tilt = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, icol_tilt);
			RFLOAT prior_psi =  DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, icol_psi);
			baseMLO->sampling.selectOrientationsWithNonZeroPriorProbability(
					prior_rot, prior_tilt, prior_psi,
					sqrt(baseMLO->mymodel.sigma2_rot),
					sqrt(baseMLO->mymodel.sigma2_tilt),
					sqrt(baseMLO->mymodel.sigma2_psi),
					op.pointer_dir_nonzeroprior, op.directions_prior,
					op.pointer_psi_nonzeroprior, op.psi_prior, false, 3.,
					baseMLO->mymodel.sigma_tilt_bodies[ibody],
					baseMLO->mymodel.sigma_psi_bodies[ibody]);

		}
		else if (baseMLO->mymodel.orientational_prior_mode != NOPRIOR && !(baseMLO->do_skip_align ||baseMLO-> do_skip_rotate))
		{
			// First try if there are some fixed prior angles
			// For multi-body refinements, ignore the original priors and get the refined residual angles from the previous iteration
			RFLOAT prior_rot = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_ROT_PRIOR);
			RFLOAT prior_tilt = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_TILT_PRIOR);
			RFLOAT prior_psi = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_PSI_PRIOR);
			RFLOAT prior_psi_flip_ratio = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_PSI_PRIOR_FLIP_RATIO);

			bool do_auto_refine_local_searches = (baseMLO->do_auto_refine) && (baseMLO->sampling.healpix_order >= baseMLO->autosampling_hporder_local_searches);
			bool do_classification_local_searches = (! baseMLO->do_auto_refine) && (baseMLO->mymodel.orientational_prior_mode == PRIOR_ROTTILT_PSI)
					&& (baseMLO->mymodel.sigma2_rot > 0.) && (baseMLO->mymodel.sigma2_tilt > 0.) && (baseMLO->mymodel.sigma2_psi > 0.);
			bool do_local_angular_searches = (do_auto_refine_local_searches) || (do_classification_local_searches);

			// If there were no defined priors (i.e. their values were 999.), then use the "normal" angles
			if (prior_rot > 998.99 && prior_rot < 999.01)
				prior_rot = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_ROT);
			if (prior_tilt > 998.99 && prior_tilt < 999.01)
				prior_tilt = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_TILT);
			if (prior_psi > 998.99 && prior_psi < 999.01)
				prior_psi = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_PSI);
			if (prior_psi_flip_ratio > 998.99 && prior_psi_flip_ratio < 999.01)
				prior_psi_flip_ratio = 0.5;

			////////// How does this work now: each particle has a different sampling object?!!!
			// Select only those orientations that have non-zero prior probability

			if (baseMLO->do_helical_refine && baseMLO->mymodel.ref_dim == 3)
			{
				baseMLO->sampling.selectOrientationsWithNonZeroPriorProbabilityFor3DHelicalReconstruction(prior_rot, prior_tilt, prior_psi,
										sqrt(baseMLO->mymodel.sigma2_rot), sqrt(baseMLO->mymodel.sigma2_tilt), sqrt(baseMLO->mymodel.sigma2_psi),
										op.pointer_dir_nonzeroprior, op.directions_prior, op.pointer_psi_nonzeroprior, op.psi_prior,
										do_local_angular_searches, prior_psi_flip_ratio);
			}
			else
			{
				baseMLO->sampling.selectOrientationsWithNonZeroPriorProbability(prior_rot, prior_tilt, prior_psi,
						sqrt(baseMLO->mymodel.sigma2_rot), sqrt(baseMLO->mymodel.sigma2_tilt), sqrt(baseMLO->mymodel.sigma2_psi),
						op.pointer_dir_nonzeroprior, op.directions_prior, op.pointer_psi_nonzeroprior, op.psi_prior);
			}

			long int nr_orients = baseMLO->sampling.NrDirections(0, &op.pointer_dir_nonzeroprior) * baseMLO->sampling.NrPsiSamplings(0, &op.pointer_psi_nonzeroprior);
			if (nr_orients == 0)
			{
				std::cerr << " sampling.NrDirections()= " << baseMLO->sampling.NrDirections(0, &op.pointer_dir_nonzeroprior)
						<< " sampling.NrPsiSamplings()= " << baseMLO->sampling.NrPsiSamplings(0, &op.pointer_psi_nonzeroprior) << std::endl;
				REPORT_ERROR("Zero orientations fall within the local angular search. Increase the sigma-value(s) on the orientations!");
			}

		}
		CTOC(accMLO->timer,"nonZeroProb");

		// ------------------------------------------------------------------------------------------

		CTIC(accMLO->timer,"readData");
		// Get the image and recimg data
		if (baseMLO->do_parallel_disc_io)
		{

			// If all followers had preread images into RAM: get those now
			if (baseMLO->do_preread_images)
			{

                img().reshape(baseMLO->mydata.particles[part_id].images[img_id].img);
                CTIC(accMLO->timer,"ParaReadPrereadImages");
				FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(baseMLO->mydata.particles[part_id].images[img_id].img)
				{
                	DIRECT_MULTIDIM_ELEM(img(), n) = (RFLOAT)DIRECT_MULTIDIM_ELEM(baseMLO->mydata.particles[part_id].images[img_id].img, n);
				}
				CTOC(accMLO->timer,"ParaReadPrereadImages");
			}
			else
			{
				if (accMLO->dataIs3D)
				{
					CTIC(accMLO->timer,"ParaRead3DImages");
					img.read(fn_img);
					img().setXmippOrigin();
					CTOC(accMLO->timer,"ParaRead3DImages");
				}
				else
				{
					CTIC(accMLO->timer,"ParaRead2DImages");
					img() = baseMLO->exp_imgs[my_metadata_offset];
					CTOC(accMLO->timer,"ParaRead2DImages");
				}
			}
			if (baseMLO->has_converged && baseMLO->do_use_reconstruct_images)
			{
				FileName fn_recimg;
				std::istringstream split2(baseMLO->exp_fn_recimg);
				// Get the right line in the exp_fn_img string
				for (int i = 0; i <= my_metadata_offset; i++)
					getline(split2, fn_recimg);
				rec_img.read(fn_recimg);
				rec_img().setXmippOrigin();
			}
		}
		else
		{
			// Unpack the image from the imagedata
			if (accMLO->dataIs3D)
			{
				CTIC(accMLO->timer,"Read3DImages");
				CTIC(accMLO->timer,"resize");
				img().resize(baseMLO->image_full_size[optics_group], baseMLO->image_full_size[optics_group], baseMLO->image_full_size[optics_group]);
				CTOC(accMLO->timer,"resize");
				// Only allow a single image per call of this function!!! nr_pool needs to be set to 1!!!!
				// This will save memory, as we'll need to store all translated images in memory....
				FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(img())
				{
					DIRECT_A3D_ELEM(img(), k, i, j) = DIRECT_A3D_ELEM(baseMLO->exp_imagedata, k, i, j);
				}
				img().setXmippOrigin();

				if (baseMLO->has_converged && baseMLO->do_use_reconstruct_images)
				{
					rec_img().resize(baseMLO->image_full_size[optics_group], baseMLO->image_full_size[optics_group], baseMLO->image_full_size[optics_group]);
					int offset = (baseMLO->do_ctf_correction) ? 2 * baseMLO->image_full_size[optics_group] : baseMLO->image_full_size[optics_group];
					FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(rec_img())
					{
						DIRECT_A3D_ELEM(rec_img(), k, i, j) = DIRECT_A3D_ELEM(baseMLO->exp_imagedata, offset + k, i, j);
					}
					rec_img().setXmippOrigin();

				}
				CTOC(accMLO->timer,"Read3DImages");

			}
			else
			{
				CTIC(accMLO->timer,"Read2DImages");
				img().resize(baseMLO->image_full_size[optics_group], baseMLO->image_full_size[optics_group]);
				FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(img())
				{
					DIRECT_A2D_ELEM(img(), i, j) = DIRECT_A3D_ELEM(baseMLO->exp_imagedata, my_metadata_offset, i, j);
				}
				img().setXmippOrigin();
				if (baseMLO->has_converged && baseMLO->do_use_reconstruct_images)
				{

					/// TODO: this will be WRONG for multi-image particles, but I guess that's not going to happen anyway...
					int my_nr_particles = baseMLO->exp_my_last_part_id - baseMLO->exp_my_first_part_id + 1;
					rec_img().resize(baseMLO->image_full_size[optics_group], baseMLO->image_full_size[optics_group]);
					FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(rec_img())
					{
						DIRECT_A2D_ELEM(rec_img(), i, j) = DIRECT_A3D_ELEM(baseMLO->exp_imagedata, my_nr_particles + my_metadata_offset, i, j);
					}
					rec_img().setXmippOrigin();
				}
				CTOC(accMLO->timer,"Read2DImages");
			}
		}
		CTOC(accMLO->timer,"readData");

		// ------------------------------------------------------------------------------------------

		size_t current_size_x = baseMLO->image_current_size[optics_group] / 2 + 1;
		size_t current_size_y = baseMLO->image_current_size[optics_group];
		size_t current_size_z = (accMLO->dataIs3D) ? baseMLO->image_current_size[optics_group] : 1;
		accMLO->transformer1.setSize(img().xdim,img().ydim,img().zdim);
		Fimg.initZeros(current_size_z, current_size_y, current_size_x);

		// ------------------------------------------------------------------------------------------

		CTIC(cudaMLO->timer,"makeNoiseMask");
        // Either mask with zeros or noise. Here, make a noise-image that will be optional in the softMask-kernel.
#ifdef NEWMEM
		AccDataTypes::ImageNew<XFLOAT> RandomImage(img(),ptrFactory);
#else
		AccDataTypes::Image<XFLOAT> RandomImage(img(),ptrFactory);
#endif

        if (!baseMLO->do_zero_mask) // prepare a acc-side Random image
        {
        		if(RandomImage.is3D())
        				CRITICAL("Noise-masking not supported with acceleration and 3D input: Noise-kernel(s) is hard-coded 2D");

                // Make a F-space image to hold generate and modulate noise
                RandomImage.accAlloc();

                // Set up scalar adjustment factor and random seed
                XFLOAT temp_sigmaFudgeFactor = baseMLO->sigma2_fudge;
                int seed(baseMLO->random_seed + part_id);


    			// Remap mymodel.sigma2_noise[optics_group] onto remapped_sigma2_noise for this images's size and angpix
    			MultidimArray<RFLOAT > remapped_sigma2_noise;
    			remapped_sigma2_noise.initZeros(XSIZE(img())/2+1);
    			RFLOAT remap_image_sizes = (baseMLO->image_full_size[optics_group] * my_pixel_size) / (baseMLO->mymodel.ori_size * baseMLO->mymodel.pixel_size);
    			FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(baseMLO->mymodel.sigma2_noise[optics_group])
    			{
    				int i_remap = ROUND(remap_image_sizes * i);
    				if (i_remap < XSIZE(remapped_sigma2_noise))
    					DIRECT_A1D_ELEM(remapped_sigma2_noise, i_remap) = DIRECT_A1D_ELEM(baseMLO->mymodel.sigma2_noise[optics_group], i);
    			}


                LAUNCH_PRIVATE_ERROR(cudaGetLastError(),accMLO->errorStatus);
                // construct the noise-image
                AccUtilities::makeNoiseImage<MlClass>(	temp_sigmaFudgeFactor,
                								remapped_sigma2_noise,
												seed,
												accMLO,
												RandomImage,
												RandomImage.is3D());
                LAUNCH_PRIVATE_ERROR(cudaGetLastError(),accMLO->errorStatus);
        }
        CTOC(cudaMLO->timer,"makeNoiseMask");

		// ------------------------------------------------------------------------------------------

		CTIC(accMLO->timer,"HelicalPrep");

		/* FIXME :  For some reason the device-allocation inside "selfTranslate" takes a much longer time than expected.
		 * 			I tried moving it up and placing the size under a bunch of if()-cases, but this simply transferred the
		 * 			allocation-cost to that region. /BjoernF,160129
		 */

		// Apply (rounded) old offsets first
		my_old_offset.selfROUND();

		// Helical reconstruction: calculate old_offset in the system of coordinates of the helix, i.e. parallel & perpendicular, depending on psi-angle!
		// For helices do NOT apply old_offset along the direction of the helix!!
		Matrix1D<RFLOAT> my_old_offset_helix_coords;
		RFLOAT rot_deg = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_ROT);
		RFLOAT tilt_deg = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_TILT);
		RFLOAT psi_deg = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_PSI);
		if ( (baseMLO->do_helical_refine) && (! baseMLO->ignore_helical_symmetry) )
		{
			// Calculate my_old_offset_helix_coords from my_old_offset and psi angle
			transformCartesianAndHelicalCoords(my_old_offset, my_old_offset_helix_coords, rot_deg, tilt_deg, psi_deg, CART_TO_HELICAL_COORDS);
			// We do NOT want to accumulate the offsets in the direction along the helix (which is X in the helical coordinate system!)
			// However, when doing helical local searches, we accumulate offsets
			// Do NOT accumulate offsets in 3D classification of helices
			if ( (! baseMLO->do_skip_align) && (! baseMLO->do_skip_rotate) )
			{
				// TODO: check whether the following lines make sense
				bool do_auto_refine_local_searches = (baseMLO->do_auto_refine) && (baseMLO->sampling.healpix_order >= baseMLO->autosampling_hporder_local_searches);
				bool do_classification_local_searches = (! baseMLO->do_auto_refine) && (baseMLO->mymodel.orientational_prior_mode == PRIOR_ROTTILT_PSI)
						&& (baseMLO->mymodel.sigma2_rot > 0.) && (baseMLO->mymodel.sigma2_tilt > 0.) && (baseMLO->mymodel.sigma2_psi > 0.);
				bool do_local_angular_searches = (do_auto_refine_local_searches) || (do_classification_local_searches);
				if (!do_local_angular_searches)
				{
					if (! accMLO->dataIs3D)
						XX(my_old_offset_helix_coords) = 0.;
					else
						ZZ(my_old_offset_helix_coords) = 0.;
				}
			}
			// TODO: Now re-calculate the my_old_offset in the real (or image) system of coordinate (rotate -psi angle)
			transformCartesianAndHelicalCoords(my_old_offset_helix_coords, my_old_offset, rot_deg, tilt_deg, psi_deg, HELICAL_TO_CART_COORDS);
		}
		CTOC(accMLO->timer,"HelicalPrep");

		// ------------------------------------------------------------------------------------------

		my_old_offset.selfROUND();

		// ------------------------------------------------------------------------------------------

		CTIC(accMLO->timer,"TranslateAndNormCorrect");

#ifdef NEWMEM
		AccDataTypes::ImageNew<XFLOAT> d_img(img.data, ptrFactory);
		AccDataTypes::ImageNew<XFLOAT> d_rec_img(img.data, ptrFactory);
#else
		AccDataTypes::Image<XFLOAT> d_img(img.data, ptrFactory);
		AccDataTypes::Image<XFLOAT> d_rec_img(img.data, ptrFactory);
#endif

		d_img.allAlloc();
		d_img.allInit(0);

		XFLOAT normcorr_val = baseMLO->do_norm_correction ? (XFLOAT)(baseMLO->mymodel.avg_norm_correction / normcorr) : 1;
		AccUtilities::TranslateAndNormCorrect(	img.data,	// input   	host-side 	MultidimArray
												d_img,		// output  	acc-side  	Array
												normcorr_val,
												XX(my_old_offset),
												YY(my_old_offset),
												(accMLO->dataIs3D) ? ZZ(my_old_offset) : 0.,
												accMLO->dataIs3D);
		LAUNCH_PRIVATE_ERROR(cudaGetLastError(),accMLO->errorStatus);

		CTOC(accMLO->timer,"TranslateAndNormCorrect");

		// Set up the UNMASKED image to use for reconstruction, which may be a separate image altogether (rec_img)
		//
		//			d_img has the image information which will be masked
		//
		if(baseMLO->has_converged && baseMLO->do_use_reconstruct_images)
		{
			CTIC(accMLO->timer,"TranslateAndNormCorrect_recImg");
			d_rec_img.allAlloc();
			d_rec_img.allInit(0);
			AccUtilities::TranslateAndNormCorrect(	rec_img.data,	// input   	host-side 	MultidimArray
													d_rec_img,		// output  	acc-side  	Array
													normcorr_val,
													XX(my_old_offset),
													YY(my_old_offset),
													(accMLO->dataIs3D) ? ZZ(my_old_offset) : 0.,
													accMLO->dataIs3D);
			LAUNCH_PRIVATE_ERROR(cudaGetLastError(),accMLO->errorStatus);
			CTOC(accMLO->timer,"TranslateAndNormCorrect_recImg");

			CTIC(cudaMLO->timer,"normalizeAndTransform_recImg");
			// The image used to reconstruct is not masked, so we transform and beam-tilt it
			AccUtilities::normalizeAndTransformImage<MlClass>(d_rec_img,		// input  acc-side  Array
															  Fimg,			// output host-side MultidimArray
															  accMLO,
															  current_size_x,
															  current_size_y,
															  current_size_z);
			LAUNCH_PRIVATE_ERROR(cudaGetLastError(),accMLO->errorStatus);
			CTOC(cudaMLO->timer,"normalizeAndTransform_recImg");
		}
		else // if we don't have special images, just use the same as for alignment. But do it here, *before masking*
		{
			CTIC(cudaMLO->timer,"normalizeAndTransform_recImg");
			// The image used to reconstruct is not masked, so we transform and beam-tilt it
			AccUtilities::normalizeAndTransformImage<MlClass>(	 d_img,		// input  acc-side  Array
																 Fimg,		// output host-side MultidimArray
																 accMLO,
																 current_size_x,
																 current_size_y,
																 current_size_z);
			LAUNCH_PRIVATE_ERROR(cudaGetLastError(),accMLO->errorStatus);
			CTOC(cudaMLO->timer,"normalizeAndTransform_recImg");
		}

		// ------------------------------------------------------------------------------------------

		if ( (baseMLO->do_helical_refine) && (! baseMLO->ignore_helical_symmetry) )
		{
			// Transform rounded Cartesian offsets to corresponding helical ones
			transformCartesianAndHelicalCoords(my_old_offset, my_old_offset_helix_coords, rot_deg, tilt_deg, psi_deg, CART_TO_HELICAL_COORDS);
			op.old_offset[img_id] = my_old_offset_helix_coords;
		}
		else
		{
			// For multi-bodies: store only the old refined offset, not the constant consensus offset or the projected COM of this body
			if (baseMLO->mymodel.nr_bodies > 1)
				op.old_offset[img_id] = my_refined_ibody_offset;
			else
				op.old_offset[img_id] = my_old_offset;  // Not doing helical refinement. Rounded Cartesian offsets are stored.
		}
		// Also store priors on translations
		op.prior[img_id] = my_prior;

		// ------------------------------------------------------------------------------------------

		CTIC(accMLO->timer,"selfApplyBeamTilt");
		baseMLO->mydata.obsModel.demodulatePhase(optics_group, Fimg);
		baseMLO->mydata.obsModel.divideByMtf(optics_group, Fimg);
		CTOC(accMLO->timer,"selfApplyBeamTilt");

		op.Fimg_nomask.at(img_id) = Fimg;

		// ------------------------------------------------------------------------------------------

		MultidimArray<RFLOAT> Mnoise;
		bool is_helical_segment = (baseMLO->do_helical_refine) || ((baseMLO->mymodel.ref_dim == 2) && (baseMLO->helical_tube_outer_diameter > 0.));

		// For multibodies: have the mask radius equal to maximum radius within body mask plus the translational offset search range
		RFLOAT my_mask_radius = (baseMLO->mymodel.nr_bodies > 1 ) ?
						(baseMLO->mymodel.max_radius_mask_bodies[ibody] + baseMLO->sampling.offset_range) / my_pixel_size :
						baseMLO->particle_diameter / (2. * my_pixel_size);

		// ------------------------------------------------------------------------------------------

		// We are now done with the unmasked image used for reconstruction.
		// Now make the masked image used for alignment and classification.

		if (is_helical_segment)
		{
			CTIC(accMLO->timer,"applyHelicalMask");

			// download img...
			d_img.cpToHost();
			d_img.streamSync();
			d_img.getHost(img());

			// ...modify img...
			if(baseMLO->do_zero_mask)
			{
				softMaskOutsideMapForHelix(img(), psi_deg, tilt_deg, my_mask_radius,
						(baseMLO->helical_tube_outer_diameter / (2. * my_pixel_size)),
						baseMLO->width_mask_edge);
			}
			else
			{
				MultidimArray<RFLOAT> Mnoise;
				RandomImage.hostAlloc();
				RandomImage.cpToHost();
				Mnoise.resize(img());
				RandomImage.getHost(Mnoise);
				softMaskOutsideMapForHelix(img(), psi_deg, tilt_deg, my_mask_radius,
						(baseMLO->helical_tube_outer_diameter / (2. * my_pixel_size)),
						baseMLO->width_mask_edge,
						&Mnoise);
			}

			// ... and re-upload img
			d_img.setHost(img());
			d_img.cpToDevice();
			CTOC(accMLO->timer,"applyHelicalMask");
		}
		else // this is not a helical segment
		{
			CTIC(accMLO->timer,"applyMask");

			// Shared parameters for noise/zero masking
			XFLOAT cosine_width = baseMLO->width_mask_edge;
			XFLOAT radius = (XFLOAT) my_mask_radius;
			if (radius < 0)
				radius = ((RFLOAT)img.data.xdim)/2.;
			XFLOAT radius_p = radius + cosine_width;

			// For zero-masking, we need the background-value
			XFLOAT bg_val(0.);
			if(baseMLO->do_zero_mask)
			{
#ifdef NEWMEM
				AccPtrNew<XFLOAT> softMaskSum    = ptrFactory.make<XFLOAT>((size_t)SOFTMASK_BLOCK_SIZE);
				AccPtrNew<XFLOAT> softMaskSum_bg = ptrFactory.make<XFLOAT>((size_t)SOFTMASK_BLOCK_SIZE);
#else
				AccPtr<XFLOAT> softMaskSum    = ptrFactory.make<XFLOAT>((size_t)SOFTMASK_BLOCK_SIZE);
				AccPtr<XFLOAT> softMaskSum_bg = ptrFactory.make<XFLOAT>((size_t)SOFTMASK_BLOCK_SIZE);
#endif				

				softMaskSum.accAlloc();
				softMaskSum_bg.accAlloc();
				softMaskSum.accInit(0);
				softMaskSum_bg.accInit(0);

				// Calculate the background value
				AccUtilities::softMaskBackgroundValue(
						d_img,
						radius,
						radius_p,
						cosine_width,
						softMaskSum,
						softMaskSum_bg);

				LAUNCH_PRIVATE_ERROR(cudaGetLastError(),accMLO->errorStatus);
				softMaskSum.streamSync();

				// Finalize the background value
				bg_val = (RFLOAT) AccUtilities::getSumOnDevice<XFLOAT>(softMaskSum_bg) /
						 (RFLOAT) AccUtilities::getSumOnDevice<XFLOAT>(softMaskSum);
				softMaskSum.streamSync();
			}

			//avoid kernel-calls warning about null-pointer for RandomImage
			if (baseMLO->do_zero_mask)
				RandomImage.setAccPtr(d_img);

			// Apply a cosine-softened mask, using either the background value or the noise-image outside of the radius
			AccUtilities::cosineFilter(
					d_img,
					baseMLO->do_zero_mask,
					RandomImage,
					radius,
					radius_p,
					cosine_width,
					bg_val);

			LAUNCH_PRIVATE_ERROR(cudaGetLastError(),accMLO->errorStatus);
			DEBUG_HANDLE_ERROR(cudaStreamSynchronize(cudaStreamPerThread));
			DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.cudaStreamPerTask));

			CTOC(accMLO->timer,"applyMask");
		}

		// ------------------------------------------------------------------------------------------

		CTIC(cudaMLO->timer,"normalizeAndTransform");
		AccUtilities::normalizeAndTransformImage<MlClass>(	 d_img,		// input
															 Fimg,		// output
															 accMLO,
															 current_size_x,
															 current_size_y,
															 current_size_z);
		LAUNCH_PRIVATE_ERROR(cudaGetLastError(),accMLO->errorStatus);
		CTOC(cudaMLO->timer,"normalizeAndTransform");

		// ------------------------------------------------------------------------------------------

		CTIC(accMLO->timer,"powerClass");
		// Store the power_class spectrum of the whole image (to fill sigma2_noise between current_size and full_size
		if (baseMLO->image_current_size[optics_group] < baseMLO->image_full_size[optics_group])
		{
#ifdef NEWMEM
			AccPtrNew<XFLOAT> spectrumAndXi2 = ptrFactory.make<XFLOAT>((size_t)((baseMLO->image_full_size[optics_group]/2+1)+1)); // last +1 is the Xi2, to remove an expensive memcpy
			// AccPtr<XFLOAT> spectrumAndXi2 = ptrFactory.make<XFLOAT>((size_t)((baseMLO->image_full_size[optics_group]/2+1)+1)); // last +1 is the Xi2, to remove an expensive memcpy
#else
			AccPtr<XFLOAT> spectrumAndXi2 = ptrFactory.make<XFLOAT>((size_t)((baseMLO->image_full_size[optics_group]/2+1)+1)); // last +1 is the Xi2, to remove an expensive memcpy
#endif
			// AccPtr<XFLOAT> spectrumAndXi2 = ptrFactory.make<XFLOAT>((size_t)((baseMLO->image_full_size[optics_group]/2+1)+1)); // last +1 is the Xi2, to remove an expensive memcpy
			spectrumAndXi2.allAlloc();
			spectrumAndXi2.accInit(0);
			spectrumAndXi2.streamSync();

			int gridSize = CEIL((float)(accMLO->transformer1.fouriers.getSize()) / (float)POWERCLASS_BLOCK_SIZE);
			if(accMLO->dataIs3D)
				AccUtilities::powerClass<true>(gridSize,POWERCLASS_BLOCK_SIZE,
					~accMLO->transformer1.fouriers,
					~spectrumAndXi2,
					accMLO->transformer1.fouriers.getSize(),
					spectrumAndXi2.getSize()-1,
					accMLO->transformer1.xFSize,
					accMLO->transformer1.yFSize,
					accMLO->transformer1.zFSize,
					(baseMLO->image_current_size[optics_group]/2)+1, // note: NOT baseMLO->image_full_size[optics_group]/2+1
					&(~spectrumAndXi2)[spectrumAndXi2.getSize()-1],  // last element is the hihgres_Xi2
					spectrumAndXi2.getStream()); 
			else
				AccUtilities::powerClass<false>(gridSize,POWERCLASS_BLOCK_SIZE,
					~accMLO->transformer1.fouriers,
					~spectrumAndXi2,
					accMLO->transformer1.fouriers.getSize(),
					spectrumAndXi2.getSize()-1,
					accMLO->transformer1.xFSize,
					accMLO->transformer1.yFSize,
					accMLO->transformer1.zFSize,
					(baseMLO->image_current_size[optics_group]/2)+1, // note: NOT baseMLO->image_full_size[optics_group]/2+1
					&(~spectrumAndXi2)[spectrumAndXi2.getSize()-1],  // last element is the hihgres_Xi2
					spectrumAndXi2.getStream());
			LAUNCH_PRIVATE_ERROR(cudaGetLastError(),accMLO->errorStatus);

			spectrumAndXi2.streamSync();
			spectrumAndXi2.cpToHost();
			spectrumAndXi2.streamSync();

			op.power_img.at(img_id).resize(baseMLO->image_full_size[optics_group]/2 + 1);

			for (int i = 0; i<(spectrumAndXi2.getSize()-1); i ++)
				op.power_img.at(img_id).data[i] = spectrumAndXi2[i];
			op.highres_Xi2_img.at(img_id) = spectrumAndXi2[spectrumAndXi2.getSize()-1];
		}
		else
		{
			op.highres_Xi2_img.at(img_id) = 0.;
		}
		CTOC(accMLO->timer,"powerClass");

		Fctf.resize(Fimg);
		// Now calculate the actual CTF
		if (baseMLO->do_ctf_correction)
		{
			if (accMLO->dataIs3D)
			{
				Image<RFLOAT> Ictf;
				if (baseMLO->do_parallel_disc_io)
				{
					CTIC(accMLO->timer,"CTFRead3D_disk");
					// Read CTF-image from disc
					FileName fn_ctf;
					if (!baseMLO->mydata.getImageNameOnScratch(part_id, img_id, fn_ctf, true))
					{
						std::istringstream split(baseMLO->exp_fn_ctf);
						// Get the right line in the exp_fn_img string
						for (int i = 0; i <= my_metadata_offset; i++)
							getline(split, fn_ctf);
					}
					Ictf.read(fn_ctf);
					CTOC(accMLO->timer,"CTFRead3D_disk");
				}
				else
				{
					CTIC(accMLO->timer,"CTFRead3D_array");
					// Unpack the CTF-image from the exp_imagedata array
					Ictf().resize(baseMLO->image_full_size[optics_group], baseMLO->image_full_size[optics_group], baseMLO->image_full_size[optics_group]);
					FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(Ictf())
					{
						DIRECT_A3D_ELEM(Ictf(), k, i, j) = DIRECT_A3D_ELEM(baseMLO->exp_imagedata, baseMLO->image_full_size[optics_group] + k, i, j);
					}
					CTOC(accMLO->timer,"CTFRead3D_array");
				}
				// Set the CTF-image in Fctf
				CTIC(accMLO->timer,"CTFSet3D_array");
				baseMLO->get3DCTFAndMulti(Ictf(), Fctf, FstMulti, ctf_premultiplied);
				CTOC(accMLO->timer,"CTFSet3D_array");
			}
			else
			{
				CTIC(accMLO->timer,"CTFRead2D");
				CTF ctf;
				ctf.setValuesByGroup(
                                        &(baseMLO->mydata).obsModel, optics_group,
					DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_CTF_DEFOCUS_U),
					DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_CTF_DEFOCUS_V),
					DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_CTF_DEFOCUS_ANGLE),
					DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_CTF_BFACTOR),
					DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_CTF_KFACTOR),
					DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_CTF_PHASE_SHIFT));

				ctf.getFftwImage(Fctf, baseMLO->image_full_size[optics_group], baseMLO->image_full_size[optics_group], my_pixel_size,
						baseMLO->ctf_phase_flipped, baseMLO->only_flip_phases, baseMLO->intact_ctf_first_peak, true, baseMLO->do_ctf_padding);

				// SHWS 13feb2020: when using CTF-premultiplied, from now on use the normal kernels, but replace ctf by ctf^2
				if (ctf_premultiplied)
				{
					FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fctf)
					{
						DIRECT_MULTIDIM_ELEM(Fctf, n) *= DIRECT_MULTIDIM_ELEM(Fctf, n);
					}
				}

				CTOC(accMLO->timer,"CTFRead2D");
			}
		}
		else
		{
			Fctf.initConstant(1.);
		}

		CTIC(accMLO->timer,"selfApplyBeamTilt");
		baseMLO->mydata.obsModel.demodulatePhase(optics_group, Fimg);
		baseMLO->mydata.obsModel.divideByMtf(optics_group, Fimg);
		CTOC(accMLO->timer,"selfApplyBeamTilt");

		// Store Fimg and Fctf
		op.Fimg.at(img_id) = Fimg;
		op.Fctf.at(img_id) = Fctf;

		// Correct images and CTFs by Multiplicity, if required, and store it
		if ( NZYXSIZE(FstMulti) > 0 )
		{
			baseMLO->applySubtomoCorrection(op.Fimg.at(img_id), op.Fimg_nomask.at(img_id), op.Fctf.at(img_id), FstMulti);
			op.FstMulti.at(img_id) = FstMulti;
		}

		// If we're doing multibody refinement, now subtract projections of the other bodies from both the masked and the unmasked particle
		if (baseMLO->mymodel.nr_bodies > 1)
		{
			MultidimArray<Complex> Fsum_obody;
			Fsum_obody.initZeros(Fimg);

			for (int obody = 0; obody < baseMLO->mymodel.nr_bodies; obody++)
			{
				if (obody != ibody) // Only subtract if other body is not this body....
				{
					// Get the right metadata
					int ocol_rot  = 0 + METADATA_LINE_LENGTH_BEFORE_BODIES + (obody) * METADATA_NR_BODY_PARAMS;
					int ocol_tilt = 1 + METADATA_LINE_LENGTH_BEFORE_BODIES + (obody) * METADATA_NR_BODY_PARAMS;
					int ocol_psi  = 2 + METADATA_LINE_LENGTH_BEFORE_BODIES + (obody) * METADATA_NR_BODY_PARAMS;
					int ocol_xoff = 3 + METADATA_LINE_LENGTH_BEFORE_BODIES + (obody) * METADATA_NR_BODY_PARAMS;
					int ocol_yoff = 4 + METADATA_LINE_LENGTH_BEFORE_BODIES + (obody) * METADATA_NR_BODY_PARAMS;
					int ocol_zoff = 5 + METADATA_LINE_LENGTH_BEFORE_BODIES + (obody) * METADATA_NR_BODY_PARAMS;
					//int ocol_norm = 6 + METADATA_LINE_LENGTH_BEFORE_BODIES + (obody) * METADATA_NR_BODY_PARAMS;

					Matrix2D<RFLOAT> Aresi,  Abody;
					// Aresi is the residual orientation for this obody
					Euler_angles2matrix(DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, ocol_rot),
										DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, ocol_tilt),
										DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, ocol_psi), Aresi, false);
					// The real orientation to be applied is the obody transformation applied and the original one
					Abody = Aori * (baseMLO->mymodel.orient_bodies[obody]).transpose() * baseMLO->A_rot90 * Aresi * baseMLO->mymodel.orient_bodies[obody];

					// Apply anisotropic mag and scaling
					Abody = baseMLO->mydata.obsModel.applyAnisoMag(Abody, optics_group);
					Abody = baseMLO->mydata.obsModel.applyScaleDifference(Abody, optics_group, baseMLO->mymodel.ori_size, baseMLO->mymodel.pixel_size);

					// Get the FT of the projection in the right direction
					MultidimArray<Complex> FTo;
					FTo.initZeros(Fimg);
					// The following line gets the correct pointer to account for overlap in the bodies
					int oobody = DIRECT_A2D_ELEM(baseMLO->mymodel.pointer_body_overlap, ibody, obody);
					baseMLO->mymodel.PPref[oobody].get2DFourierTransform(FTo, Abody);

					/********************************************************************************
					 * Currently CPU-memory for projectors is not deallocated when doing multibody
					 * due to the previous line. See cpu_ml_optimiser.cpp and cuda_ml_optimiser.cu
					 ********************************************************************************/

					// 17May2017: Body is centered at its own COM
					// move it back to its place in the original particle image
					Matrix1D<RFLOAT> other_projected_com(baseMLO->mymodel.data_dim);

					// Projected COM for this body (using Aori, just like above for ibody and my_projected_com!!!)
					other_projected_com = Aori * (baseMLO->mymodel.com_bodies[obody]);
					// This will have made other_projected_com of size 3 again! resize to mymodel.data_dim
					other_projected_com.resize(baseMLO->mymodel.data_dim);

					// Do the exact same as was done for the ibody, but DONT selfROUND here, as later phaseShift applied to ibody below!!!
					other_projected_com -= my_old_offset_ori;

					// Subtract refined obody-displacement
					XX(other_projected_com) -= DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, ocol_xoff);
					YY(other_projected_com) -= DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, ocol_yoff);
					if (baseMLO->mymodel.data_dim == 3)
						ZZ(other_projected_com) -= DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, ocol_zoff);

					// Add the my_old_offset=selfRound(my_old_offset_ori - my_projected_com) already applied to this image for ibody
					other_projected_com += my_old_offset;

					shiftImageInFourierTransform(FTo, Faux, (RFLOAT)baseMLO->image_full_size[optics_group],
							XX(other_projected_com), YY(other_projected_com), (accMLO->dataIs3D) ? ZZ(other_projected_com) : 0.);

					// Sum the Fourier transforms of all the obodies
					Fsum_obody += Faux;

				} // end if obody != ibody
			} // end for obody

			// Now that we have all the summed projections of the obodies, apply CTF, masks etc
			// Apply the CTF to this reference projection
			if (baseMLO->do_ctf_correction)
			{
				FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fsum_obody)
				{
					DIRECT_MULTIDIM_ELEM(Fsum_obody, n) *= DIRECT_MULTIDIM_ELEM(Fctf, n);
				}

				// Also do phase modulation, for beam tilt correction and other asymmetric aberrations
				baseMLO->mydata.obsModel.demodulatePhase(optics_group, Fsum_obody, true); // true means do_modulate_instead
				baseMLO->mydata.obsModel.divideByMtf(optics_group, Fsum_obody, true); // true means do_multiply_instead

			}

			// Subtract the other-body FT from the current image FT
			// First the unmasked one, which will be used for reconstruction
			// Only do this if the flag below is true. Otherwise, use the original particles for reconstruction
			if (baseMLO->do_reconstruct_subtracted_bodies)
				op.Fimg_nomask.at(img_id) -= Fsum_obody;

			// For the masked one, have to mask outside the circular mask to prevent negative values outside the mask in the subtracted image!
			CenterFFTbySign(Fsum_obody);
			windowFourierTransform(Fsum_obody, Faux, baseMLO->image_full_size[optics_group]);
			accMLO->transformer.inverseFourierTransform(Faux, img());

			softMaskOutsideMap(img(), my_mask_radius, (RFLOAT)baseMLO->width_mask_edge);

			// And back to Fourier space now
			accMLO->transformer.FourierTransform(img(), Faux);
			windowFourierTransform(Faux, Fsum_obody, baseMLO->image_current_size[optics_group]);
			CenterFFTbySign(Fsum_obody);

			// Subtract the other-body FT from the masked exp_Fimgs
			op.Fimg.at(img_id) -= Fsum_obody;

			// 23jul17: NEW: as we haven't applied the (nonROUNDED!!)  my_refined_ibody_offset yet, do this now in the FourierTransform
			Faux = op.Fimg.at(img_id);
			shiftImageInFourierTransform(Faux, op.Fimg.at(img_id), (RFLOAT)baseMLO->image_full_size[optics_group],
					XX(my_refined_ibody_offset), YY(my_refined_ibody_offset), (accMLO->dataIs3D) ? ZZ(my_refined_ibody_offset) : 0);
			Faux = op.Fimg_nomask.at(img_id);
			shiftImageInFourierTransform(Faux, op.Fimg_nomask.at(img_id), (RFLOAT)baseMLO->image_full_size[optics_group],
					XX(my_refined_ibody_offset), YY(my_refined_ibody_offset), (accMLO->dataIs3D) ? ZZ(my_refined_ibody_offset) : 0);
		} // end if mymodel.nr_bodies > 1


	} // end loop img_id
	//accMLO->transformer.clear();
#ifdef TIMING
	// if (part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
		baseMLO->timer.toc(baseMLO->TIMING_ESP_FT);
#endif
	GATHERGPUTIMINGS(accMLO->timer);
}

// ----------------------------------------------------------------------------
// -------------- convertAllSquaredDifferencesToWeights -----------------------
// ----------------------------------------------------------------------------
template<class MlClass>
void convertAllSquaredDifferencesToWeights(unsigned exp_ipass,
											OptimisationParamters &op,
											SamplingParameters &sp,
											MlOptimiser *baseMLO,
											MlClass *accMLO,
#ifdef NEWMEM
											std::vector< IndexedDataArrayNew > &PassWeights,
											// std::vector< IndexedDataArray > &PassWeights,

											std::vector< std::vector< IndexedDataArrayMask > > &FPCMasks,
											AccPtrNew<XFLOAT> &Mweight, // FPCMasks = Fine-Pass Class-Masks
											AccPtrFactoryNew ptrFactory,
#else
											std::vector< IndexedDataArray > &PassWeights,
											std::vector< std::vector< IndexedDataArrayMask > > &FPCMasks,
											AccPtr<XFLOAT> &Mweight, // FPCMasks = Fine-Pass Class-Masks
											AccPtrFactory ptrFactory,
#endif
											int ibody,int thread_id, Context<MlClass> &ctx)
{
#ifdef TIMING
	// if (op.part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
	{
		if (exp_ipass == 0) baseMLO->timer.tic(baseMLO->TIMING_ESP_WEIGHT1);
		else baseMLO->timer.tic(baseMLO->TIMING_ESP_WEIGHT2);
	}
#endif

	RFLOAT my_sigma2_offset = (baseMLO->mymodel.nr_bodies > 1) ?
			baseMLO->mymodel.sigma_offset_bodies[ibody]*baseMLO->mymodel.sigma_offset_bodies[ibody] : baseMLO->mymodel.sigma2_offset;

	// Ready the "prior-containers" for all classes (remake every img_id)
#ifdef NEWMEM
	AccPtrNew<XFLOAT>  pdf_orientation       = ptrFactory.make<XFLOAT>((size_t)((sp.iclass_max-sp.iclass_min+1) * sp.nr_dir * sp.nr_psi));
	AccPtrNew<bool>    pdf_orientation_zeros = ptrFactory.make<bool>(pdf_orientation.getSize());
	AccPtrNew<XFLOAT>  pdf_offset            = ptrFactory.make<XFLOAT>((size_t)((sp.iclass_max-sp.iclass_min+1)*sp.nr_trans));
	AccPtrNew<bool>    pdf_offset_zeros      = ptrFactory.make<bool>(pdf_offset.getSize());
#else
	AccPtr<XFLOAT>  pdf_orientation       = ptrFactory.make<XFLOAT>((size_t)((sp.iclass_max-sp.iclass_min+1) * sp.nr_dir * sp.nr_psi));
	AccPtr<bool>    pdf_orientation_zeros = ptrFactory.make<bool>(pdf_orientation.getSize());
	AccPtr<XFLOAT>  pdf_offset            = ptrFactory.make<XFLOAT>((size_t)((sp.iclass_max-sp.iclass_min+1)*sp.nr_trans));
	AccPtr<bool>    pdf_offset_zeros      = ptrFactory.make<bool>(pdf_offset.getSize());
#endif
	pdf_orientation.accAlloc();
	pdf_orientation_zeros.accAlloc();
	pdf_offset.allAlloc();
	pdf_offset_zeros.allAlloc();

	CUSTOM_ALLOCATOR_REGION_NAME("CASDTW_PDF");

	// pdf_orientation is img_id-independent, so we keep it above img_id scope
	CTIC(accMLO->timer,"get_orient_priors");
#ifdef NEWMEM
	AccPtrNew<RFLOAT> pdfs	= ptrFactory.make<RFLOAT>((size_t)((sp.iclass_max-sp.iclass_min+1) * sp.nr_dir * sp.nr_psi));
#else
	AccPtr<RFLOAT> pdfs				= ptrFactory.make<RFLOAT>((size_t)((sp.iclass_max-sp.iclass_min+1) * sp.nr_dir * sp.nr_psi));
#endif
	pdfs.allAlloc();


	for (unsigned long exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
		for (unsigned long idir = sp.idir_min, iorientclass = (exp_iclass-sp.iclass_min) * sp.nr_dir * sp.nr_psi; idir <=sp.idir_max; idir++)
			for (unsigned long ipsi = sp.ipsi_min; ipsi <= sp.ipsi_max; ipsi++, iorientclass++)
			{
				RFLOAT pdf(0);

				if (baseMLO->do_skip_align || baseMLO->do_skip_rotate)
					pdf = baseMLO->mymodel.pdf_class[exp_iclass];
				else if (baseMLO->mymodel.orientational_prior_mode == NOPRIOR)
					pdf = DIRECT_MULTIDIM_ELEM(baseMLO->mymodel.pdf_direction[exp_iclass], idir);
				else
					pdf = op.directions_prior[idir] * op.psi_prior[ipsi];

				pdfs[iorientclass] = pdf;
			}

	pdfs.cpToDevice();
	AccUtilities::initOrientations(pdfs, pdf_orientation, pdf_orientation_zeros);

	CTOC(accMLO->timer,"get_orient_priors");

	if(exp_ipass==0 || baseMLO->adaptive_oversampling!=0)
	{
		op.sum_weight.clear();
		op.sum_weight.resize(sp.nr_images, (RFLOAT)(sp.nr_images));
		op.max_weight.clear();
		op.max_weight.resize(sp.nr_images, (RFLOAT)-1);
	}

	if (exp_ipass==0)
		op.Mcoarse_significant.resizeNoCp(1,1,sp.nr_images, XSIZE(op.Mweight));

	XFLOAT my_significant_weight;
	op.significant_weight.clear();
	op.significant_weight.resize(sp.nr_images, 0.);

	// loop over all images inside this particle
	for (int img_id = 0; img_id < sp.nr_images; img_id++)
	{
		int my_metadata_offset = op.metadata_offset + img_id;
		RFLOAT my_pixel_size = baseMLO->mydata.getImagePixelSize(op.part_id, img_id);

		RFLOAT old_offset_x, old_offset_y, old_offset_z;

		if (baseMLO->mymodel.nr_bodies > 1)
		{
			old_offset_x = old_offset_y = old_offset_z = 0.;
		}
		else
		{
			old_offset_x = XX(op.old_offset[img_id]);
			old_offset_y = YY(op.old_offset[img_id]);
			if (accMLO->dataIs3D)
				old_offset_z = ZZ(op.old_offset[img_id]);
		}

		if ((baseMLO->iter == 1 && baseMLO->do_firstiter_cc) || baseMLO->do_always_cc)
		{
			if(exp_ipass==0)
			{
				int nr_coarse_weights = (sp.iclass_max-sp.iclass_min+1)*sp.nr_images * sp.nr_dir * sp.nr_psi * sp.nr_trans;
				PassWeights[img_id].weights.setAccPtr(&(~Mweight)[img_id*nr_coarse_weights]);
				PassWeights[img_id].weights.setHostPtr(&Mweight[img_id*nr_coarse_weights]);
				PassWeights[img_id].weights.setSize(nr_coarse_weights);
			}
			PassWeights[img_id].weights.doFreeHost=false;

			std::pair<size_t, XFLOAT> min_pair=AccUtilities::getArgMinOnDevice<XFLOAT>(PassWeights[img_id].weights);
			PassWeights[img_id].weights.cpToHost();
			DEBUG_HANDLE_ERROR(cudaStreamSynchronize(cudaStreamPerThread));
			DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.cudaStreamPerTask));

			//Set all device-located weights to zero, and only the smallest one to 1.
#ifdef _CUDA_ENABLED
			DEBUG_HANDLE_ERROR(cudaMemsetAsync(~(PassWeights[img_id].weights), 0.f, PassWeights[img_id].weights.getSize()*sizeof(XFLOAT), ctx.cudaStreamPerTask));

			XFLOAT unity=1;
			DEBUG_HANDLE_ERROR(cudaMemcpyAsync( &(PassWeights[img_id].weights(min_pair.first) ), &unity, sizeof(XFLOAT), cudaMemcpyHostToDevice, ctx.cudaStreamPerTask));

			PassWeights[img_id].weights.cpToHost();
			DEBUG_HANDLE_ERROR(cudaStreamSynchronize(cudaStreamPerThread));
			DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.cudaStreamPerTask));
#else
			deviceInitValue<XFLOAT>(PassWeights[img_id].weights, (XFLOAT)0.0);
			PassWeights[img_id].weights[min_pair.first] = (XFLOAT)1.0;
#endif

			my_significant_weight = 0.999;
			int significant_num = 0;
			DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_NR_SIGN) = (RFLOAT) 1.;
			if (exp_ipass==0) // TODO better memset, 0 => false , 1 => true
				for (int ihidden = 0; ihidden < XSIZE(op.Mcoarse_significant); ihidden++)
					if (DIRECT_A2D_ELEM(op.Mweight, img_id, ihidden) >= my_significant_weight) {
						DIRECT_A2D_ELEM(op.Mcoarse_significant, img_id, ihidden) = true;
						significant_num ++;
					}
					else
						DIRECT_A2D_ELEM(op.Mcoarse_significant, img_id, ihidden) = false;
			else
			{
				std::pair<size_t, XFLOAT> max_pair = AccUtilities::getArgMaxOnDevice<XFLOAT>(PassWeights[img_id].weights);
				op.max_index[img_id].fineIdx = PassWeights[img_id].ihidden_overs[max_pair.first];
				op.max_weight[img_id] = max_pair.second;
			}
			// if (exp_ipass == 0)
			// 	printf("#### significat_num : %8d  Mcoarse_significant.xdim : %8d\n", significant_num, XSIZE(op.Mcoarse_significant));

		}
		else
		{


			long int sumRedSize=0;
			for (unsigned long exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
				sumRedSize+= (exp_ipass==0) ? ceilf((float)(sp.nr_dir*sp.nr_psi)/(float)SUMW_BLOCK_SIZE) : ceil((float)FPCMasks[img_id][exp_iclass].jobNum / (float)SUMW_BLOCK_SIZE);

			// loop through making translational priors for all classes this img_id - then copy all at once - then loop through kernel calls ( TODO: group kernel calls into one big kernel)
			CTIC(accMLO->timer,"get_offset_priors");

			for (unsigned long exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
			{
				RFLOAT myprior_x, myprior_y, myprior_z;
				if (baseMLO->mymodel.nr_bodies > 1)
				{
					myprior_x = myprior_y = myprior_z = 0.;
				}
				else if (baseMLO->mymodel.ref_dim == 2 && !baseMLO->do_helical_refine)
				{
					myprior_x = XX(baseMLO->mymodel.prior_offset_class[exp_iclass]);
					myprior_y = YY(baseMLO->mymodel.prior_offset_class[exp_iclass]);
				}
				else
				{
					myprior_x = XX(op.prior[img_id]);
					myprior_y = YY(op.prior[img_id]);
					if (accMLO->dataIs3D)
						myprior_z = ZZ(op.prior[img_id]);
				}

				for (unsigned long itrans = sp.itrans_min; itrans <= sp.itrans_max; itrans++)
				{

					// If it is doing helical refinement AND Cartesian vector myprior has a length > 0, transform the vector to its helical coordinates
					if ( (baseMLO->do_helical_refine) && (! baseMLO->ignore_helical_symmetry))
					{
						RFLOAT mypriors_len2 = myprior_x * myprior_x + myprior_y * myprior_y;
						if (accMLO->dataIs3D)
							mypriors_len2 += myprior_z * myprior_z;

						if (mypriors_len2 > 0.00001)
						{
							RFLOAT rot_deg = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_ROT);
							RFLOAT tilt_deg = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_TILT);
							RFLOAT psi_deg = DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_PSI);
							transformCartesianAndHelicalCoords(myprior_x, myprior_y, myprior_z, myprior_x, myprior_y, myprior_z, rot_deg, tilt_deg, psi_deg, (accMLO->dataIs3D) ? (3) : (2), CART_TO_HELICAL_COORDS);
						}
					}
					// (For helical refinement) Now offset, old_offset, sampling.translations and myprior are all in helical coordinates

					// To speed things up, only calculate pdf_offset at the coarse sampling.
					// That should not matter much, and that way one does not need to calculate all the OversampledTranslations
					double pdf(0), pdf_zeros(0);
					RFLOAT offset_x = old_offset_x + baseMLO->sampling.translations_x[itrans];
					RFLOAT offset_y = old_offset_y + baseMLO->sampling.translations_y[itrans];
					double tdiff2 = 0.;

					if ( (! baseMLO->do_helical_refine) || (baseMLO->ignore_helical_symmetry) || (accMLO->dataIs3D) )
						tdiff2 += (offset_x - myprior_x) * (offset_x - myprior_x);
					tdiff2 += (offset_y - myprior_y) * (offset_y - myprior_y);
					if (accMLO->dataIs3D)
					{
						RFLOAT offset_z = old_offset_z + baseMLO->sampling.translations_z[itrans];
						if ( (! baseMLO->do_helical_refine) || (baseMLO->ignore_helical_symmetry) )
							tdiff2 += (offset_z - myprior_z) * (offset_z - myprior_z);
					}

					// As of version 3.1, sigma_offsets are in Angstroms!
					tdiff2 *= my_pixel_size * my_pixel_size;

					// P(offset|sigma2_offset)
					// This is the probability of the offset, given the model offset and variance.
					if (my_sigma2_offset < 0.0001)
					{
						pdf_zeros = tdiff2 > 0.;
						pdf = pdf_zeros ? 0. : 1.;

					}
					else
					{
						pdf_zeros = false;
						pdf = tdiff2 / (-2. * my_sigma2_offset);
					}

					pdf_offset_zeros[(exp_iclass-sp.iclass_min)*sp.nr_trans + itrans] = pdf_zeros;
					pdf_offset     [(exp_iclass-sp.iclass_min)*sp.nr_trans + itrans] = pdf;
				}
			}

			pdf_offset_zeros.cpToDevice();
			pdf_offset.cpToDevice();

			CTOC(accMLO->timer,"get_offset_priors");
			CTIC(accMLO->timer,"sumweight1");

			if(exp_ipass==0)
			{
#ifdef NEWMEM
				AccPtrNew<XFLOAT>  ipartMweight(
#else
				AccPtr<XFLOAT>  ipartMweight(
#endif
						Mweight,
						img_id * op.Mweight.xdim + sp.nr_dir * sp.nr_psi * sp.nr_trans * sp.iclass_min,
						(sp.iclass_max-sp.iclass_min+1) * sp.nr_dir * sp.nr_psi * sp.nr_trans);

				pdf_offset.streamSync();

				AccUtilities::kernel_weights_exponent_coarse(
						sp.iclass_max-sp.iclass_min+1,
						pdf_orientation,
						pdf_orientation_zeros,
						pdf_offset,
						pdf_offset_zeros,
						ipartMweight,
						(XFLOAT)op.min_diff2[img_id],
						sp.nr_dir*sp.nr_psi,
						sp.nr_trans);


				XFLOAT weights_max = AccUtilities::getMaxOnDevice<XFLOAT>(ipartMweight);

				/*
				 * Add 50 since we want to stay away from e^88, which approaches the single precision limit.
				 * We still want as high numbers as possible to utilize most of the single precision span.
				 * Dari - 201710
				*/
				AccUtilities::kernel_exponentiate( ipartMweight, 50 - weights_max);

				DEBUG_HANDLE_ERROR(cudaStreamSynchronize(cudaStreamPerThread));
				DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.cudaStreamPerTask));

				unsigned long ipart_length = (sp.iclass_max-sp.iclass_min+1) * sp.nr_dir * sp.nr_psi * sp.nr_trans;
				size_t offset = img_id * op.Mweight.xdim + sp.nr_dir * sp.nr_psi * sp.nr_trans * sp.iclass_min;

				if (ipart_length > 1)
				{
					//Wrap the current ipart data in a new pointer
#ifdef NEWMEM
					AccPtrNew<XFLOAT> unsorted_ipart(
							Mweight,
							offset,
							ipart_length);

					AccPtrNew<XFLOAT> filtered = ptrFactory.make<XFLOAT>((size_t)unsorted_ipart.getSize());
#else
					AccPtr<XFLOAT> unsorted_ipart(
							Mweight,
							offset,
							ipart_length);

					AccPtr<XFLOAT> filtered = ptrFactory.make<XFLOAT>((size_t)unsorted_ipart.getSize());
#endif

					CUSTOM_ALLOCATOR_REGION_NAME("CASDTW_SORTSUM");

					filtered.deviceAlloc();

#ifdef DEBUG_CUDA
					if (unsorted_ipart.getSize()==0)
						ACC_PTR_DEBUG_FATAL("Unsorted array size zero.\n");  // Hopefully Impossible
#endif
					size_t filteredSize = AccUtilities::filterGreaterZeroOnDevice<XFLOAT>(unsorted_ipart, filtered);

					if (filteredSize == 0)
					{
						std::cerr << std::endl;
						std::cerr << " fn_img= " << sp.current_img << std::endl;
						std::cerr << " img_id= " << img_id << " adaptive_fraction= " << baseMLO->adaptive_fraction << std::endl;
						std::cerr << " min_diff2= " << op.min_diff2[img_id] << std::endl;

						pdf_orientation.dumpAccToFile("error_dump_pdf_orientation");
						pdf_offset.dumpAccToFile("error_dump_pdf_offset");
						unsorted_ipart.dumpAccToFile("error_dump_filtered");

						std::cerr << "Dumped data: error_dump_pdf_orientation, error_dump_pdf_orientation and error_dump_unsorted." << std::endl;

						CRITICAL(ERRFILTEREDZERO); // "filteredSize == 0"
					}
					filtered.setSize(filteredSize);
					long int my_nr_significant_coarse_samples;
					XFLOAT significant_weight;

// #ifndef relion_total

					CTIC(accMLO->timer,"sort");

#ifdef NEWMEM
					AccPtrNew<XFLOAT> sorted =         ptrFactory.make<XFLOAT>((size_t)filteredSize);
					AccPtrNew<XFLOAT> cumulative_sum = ptrFactory.make<XFLOAT>((size_t)filteredSize);
#else
					AccPtr<XFLOAT> sorted =         ptrFactory.make<XFLOAT>((size_t)filteredSize);
					AccPtr<XFLOAT> cumulative_sum = ptrFactory.make<XFLOAT>((size_t)filteredSize);
#endif
					sorted.accAlloc();
					cumulative_sum.accAlloc();

					AccUtilities::sortOnDevice<XFLOAT>(filtered, sorted);
					AccUtilities::scanOnDevice<XFLOAT>(sorted, cumulative_sum);



					op.sum_weight[img_id] = cumulative_sum.getAccValueAt(cumulative_sum.getSize() - 1);

					// long int my_nr_significant_coarse_samples;
					size_t thresholdIdx = findThresholdIdxInCumulativeSum<XFLOAT>(cumulative_sum,
							(1 - baseMLO->adaptive_fraction) * op.sum_weight[img_id]);

					my_nr_significant_coarse_samples = filteredSize - thresholdIdx;

					if (my_nr_significant_coarse_samples == 0)
					{
						std::cerr << std::endl;
						std::cerr << " fn_img= " << sp.current_img << std::endl;
						std::cerr << " img_id= " << img_id << " adaptive_fraction= " << baseMLO->adaptive_fraction << std::endl;
						std::cerr << " threshold= " << (1 - baseMLO->adaptive_fraction) * op.sum_weight[img_id] << " thresholdIdx= " << thresholdIdx << std::endl;
						std::cerr << " op.sum_weight[img_id]= " << op.sum_weight[img_id] << std::endl;
						std::cerr << " min_diff2= " << op.min_diff2[img_id] << std::endl;

						unsorted_ipart.dumpAccToFile("error_dump_unsorted");
						filtered.dumpAccToFile("error_dump_filtered");
						sorted.dumpAccToFile("error_dump_sorted");
						cumulative_sum.dumpAccToFile("error_dump_cumulative_sum");

						std::cerr << "Written error_dump_unsorted, error_dump_filtered, error_dump_sorted, and error_dump_cumulative_sum." << std::endl;

						CRITICAL(ERRNOSIGNIFS); // "my_nr_significant_coarse_samples == 0"
					}

					if (baseMLO->maximum_significants > 0 &&
							my_nr_significant_coarse_samples > baseMLO->maximum_significants)
					{
						my_nr_significant_coarse_samples = baseMLO->maximum_significants;
						thresholdIdx = filteredSize - my_nr_significant_coarse_samples;
					}

					significant_weight = sorted.getAccValueAt(thresholdIdx);
					CTOC(accMLO->timer,"sort");

					//xjl
					// FILE *jobs = fopen("./data/iter7.txt", "w");
					// fprintf(jobs,"%ld %d %lf %lf %lf\n", filteredSize, baseMLO->maximum_significants,baseMLO->adaptive_fraction,significant_weight,op.sum_weight[img_id]);
					// for  (int i = 0; i < filteredSize; i++)
					// 	fprintf(jobs, "%e ", filtered[i]);
					// fprintf(jobs, "\n");
					// fclose(jobs);


					CTIC(accMLO->timer,"getArgMaxOnDevice");
					std::pair<size_t, XFLOAT> max_pair = AccUtilities::getArgMaxOnDevice<XFLOAT>(unsorted_ipart);
					CTOC(accMLO->timer,"getArgMaxOnDevice");
					op.max_index[img_id].coarseIdx = max_pair.first;
					op.max_weight[img_id] = max_pair.second;

					// Store nr_significant_coarse_samples for this particle
					// Don't do this for multibody, as it would be overwritten for each body,
					// and we also use METADATA_NR_SIGN in the new safeguard for the gold-standard separation
					if (baseMLO->mymodel.nr_bodies == 1)
						DIRECT_A2D_ELEM(baseMLO->exp_metadata, my_metadata_offset, METADATA_NR_SIGN) = (RFLOAT) my_nr_significant_coarse_samples;

#ifdef NEWMEM
					AccPtrNew<bool> Mcoarse_significant = ptrFactory.make<bool>(ipart_length);
#else
					AccPtr<bool> Mcoarse_significant = ptrFactory.make<bool>(ipart_length);
#endif
					Mcoarse_significant.setHostPtr(&op.Mcoarse_significant.data[offset]);

					CUSTOM_ALLOCATOR_REGION_NAME("CASDTW_SIG");
					Mcoarse_significant.deviceAlloc();

					DEBUG_HANDLE_ERROR(cudaStreamSynchronize(cudaStreamPerThread)); // NO USE
					DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.cudaStreamPerTask));
					arrayOverThreshold<XFLOAT>(unsorted_ipart, Mcoarse_significant, significant_weight);
					Mcoarse_significant.cpToHost();
					DEBUG_HANDLE_ERROR(cudaStreamSynchronize(cudaStreamPerThread)); // NO USE
					DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.cudaStreamPerTask));
				}
				else if (ipart_length == 1)
				{
					op.Mcoarse_significant.data[img_id * op.Mweight.xdim + sp.nr_dir * sp.nr_psi * sp.nr_trans * sp.iclass_min] = 1;
				}
				else
					CRITICAL(ERRNEGLENGTH);
			}
			else
			{
				for (int exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
					DEBUG_HANDLE_ERROR(cudaStreamSynchronize(accMLO->classStreams[exp_iclass]));
				DEBUG_HANDLE_ERROR(cudaStreamSynchronize(cudaStreamPerThread));
				for (unsigned long exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
					DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.classStreams[exp_iclass]));
				DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.cudaStreamPerTask));

				XFLOAT weights_max = -std::numeric_limits<XFLOAT>::max();

				pdf_offset.streamSync();

				for (unsigned long exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++) // TODO could use classStreams
				{
					if ((baseMLO->mymodel.pdf_class[exp_iclass] > 0.) && (FPCMasks[img_id][exp_iclass].weightNum > 0) )
					{
						// Use the constructed mask to build a partial (class-specific) input
						// (until now, PassWeights has been an empty placeholder. We now create class-partials pointing at it, and start to fill it with stuff)
#ifdef NEWMEM
						IndexedDataArrayNew thisClassPassWeights(PassWeights[img_id],FPCMasks[img_id][exp_iclass]);
#else
						IndexedDataArray thisClassPassWeights(PassWeights[img_id],FPCMasks[img_id][exp_iclass]);
#endif
						AccPtr<XFLOAT> pdf_orientation_class =       ptrFactory.make<XFLOAT>(sp.nr_dir*sp.nr_psi),
						               pdf_offset_class =            ptrFactory.make<XFLOAT>(sp.nr_trans);
						AccPtr<bool>   pdf_orientation_zeros_class = ptrFactory.make<bool>(sp.nr_dir*sp.nr_psi),
						               pdf_offset_zeros_class =      ptrFactory.make<bool>(sp.nr_trans);

						pdf_orientation_class      .setAccPtr(&((~pdf_orientation)      [(exp_iclass-sp.iclass_min)*sp.nr_dir*sp.nr_psi]));
						pdf_orientation_zeros_class.setAccPtr(&((~pdf_orientation_zeros)[(exp_iclass-sp.iclass_min)*sp.nr_dir*sp.nr_psi]));

						pdf_offset_class           .setAccPtr(&((~pdf_offset)           [(exp_iclass-sp.iclass_min)*sp.nr_trans]));
						pdf_offset_zeros_class     .setAccPtr(&((~pdf_offset_zeros)     [(exp_iclass-sp.iclass_min)*sp.nr_trans]));

						// thisClassPassWeights.weights.setStream(accMLO->classStreams[exp_iclass]);
						thisClassPassWeights.weights.setStream(ctx.classStreams[exp_iclass]);

						AccUtilities::kernel_exponentiate_weights_fine(
								~pdf_orientation_class,
								~pdf_orientation_zeros_class,
								~pdf_offset_class,
								~pdf_offset_zeros_class,
								~thisClassPassWeights.weights,
								(XFLOAT)op.min_diff2[img_id],
								sp.nr_oversampled_rot,
								sp.nr_oversampled_trans,
								~thisClassPassWeights.rot_id,
								~thisClassPassWeights.trans_idx,
								~FPCMasks[img_id][exp_iclass].jobOrigin,
								~FPCMasks[img_id][exp_iclass].jobExtent,
								FPCMasks[img_id][exp_iclass].jobNum,
								// accMLO->classStreams[exp_iclass]);
								ctx.classStreams[exp_iclass]);

						XFLOAT m = AccUtilities::getMaxOnDevice<XFLOAT>(thisClassPassWeights.weights);

						if (m > weights_max)
							weights_max = m;
					}
				}

				for (unsigned long exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++) // TODO could use classStreams
				{
					if ((baseMLO->mymodel.pdf_class[exp_iclass] > 0.) && (FPCMasks[img_id][exp_iclass].weightNum > 0) )
					{
#ifdef NEWMEM
						IndexedDataArrayNew thisClassPassWeights(PassWeights[img_id],FPCMasks[img_id][exp_iclass]);
#else
						IndexedDataArray thisClassPassWeights(PassWeights[img_id],FPCMasks[img_id][exp_iclass]);
#endif

						// thisClassPassWeights.weights.setStream(accMLO->classStreams[exp_iclass]);
						thisClassPassWeights.weights.setStream(ctx.classStreams[exp_iclass]);
						/*
						 * Add 50 since we want to stay away from e^88, which approaches the single precision limit.
						 * We still want as high numbers as possible to utilize most of the single precision span.
						 * Dari - 201710
						*/
						AccUtilities::kernel_exponentiate( thisClassPassWeights.weights, 50 - weights_max );
					}
				}

				op.min_diff2[img_id] += 50 - weights_max;

				for (unsigned long exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
					DEBUG_HANDLE_ERROR(cudaStreamSynchronize(accMLO->classStreams[exp_iclass]));
				DEBUG_HANDLE_ERROR(cudaStreamSynchronize(cudaStreamPerThread));

				for (unsigned long exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
					DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.classStreams[exp_iclass]));
				DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.cudaStreamPerTask));

				if(baseMLO->is_som_iter) {
					op.sum_weight_class[img_id].resize(baseMLO->mymodel.nr_classes, 0);

					for (unsigned long exp_iclass = sp.iclass_min;
					     exp_iclass <= sp.iclass_max; exp_iclass++) // TODO could use classStreams
					{
						if ((baseMLO->mymodel.pdf_class[exp_iclass] > 0.) &&
						    (FPCMasks[img_id][exp_iclass].weightNum > 0)) {
#ifdef NEWMEM
							IndexedDataArrayNew thisClassPassWeights(PassWeights[img_id], FPCMasks[img_id][exp_iclass]);
#else
							IndexedDataArray thisClassPassWeights(PassWeights[img_id], FPCMasks[img_id][exp_iclass]);
#endif
							op.sum_weight_class[img_id][exp_iclass] = AccUtilities::getSumOnDevice(thisClassPassWeights.weights);
						}
					}
					for (unsigned long exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
						DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.classStreams[exp_iclass]));
					DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.cudaStreamPerTask));
				}

				PassWeights[img_id].weights.cpToHost(); // note that the host-pointer is shared: we're copying to Mweight.


				DEBUG_HANDLE_ERROR(cudaStreamSynchronize(cudaStreamPerThread));
				DEBUG_HANDLE_ERROR(cudaStreamSynchronize(ctx.cudaStreamPerTask));
				size_t weightSize = PassWeights[img_id].weights.getSize();
// #ifndef relion_total

#ifdef NEWMEM
				CTIC(accMLO->timer,"sort");
				AccPtrNew<XFLOAT> sorted =         ptrFactory.make<XFLOAT>((size_t)weightSize);
				AccPtrNew<XFLOAT> cumulative_sum = ptrFactory.make<XFLOAT>((size_t)weightSize);
#else
				AccPtr<XFLOAT> sorted =         ptrFactory.make<XFLOAT>((size_t)weightSize);
				AccPtr<XFLOAT> cumulative_sum = ptrFactory.make<XFLOAT>((size_t)weightSize);
#endif

				CUSTOM_ALLOCATOR_REGION_NAME("CASDTW_FINE");

				sorted.accAlloc();
				cumulative_sum.accAlloc();

				AccUtilities::sortOnDevice<XFLOAT>(PassWeights[img_id].weights, sorted);
				AccUtilities::scanOnDevice<XFLOAT>(sorted, cumulative_sum);


				size_t thresholdIdx ;
				if(baseMLO->adaptive_oversampling!=0)
				{
					op.sum_weight[img_id] = cumulative_sum.getAccValueAt(cumulative_sum.getSize() - 1);

					if (op.sum_weight[img_id]==0)
					{
						std::cerr << std::endl;
						std::cerr << " fn_img= " << sp.current_img << std::endl;
						std::cerr << " op.part_id= " << op.part_id << std::endl;
						std::cerr << " img_id= " << img_id << std::endl;
						std::cerr << " op.min_diff2[img_id]= " << op.min_diff2[img_id] << std::endl;
						int group_id = baseMLO->mydata.getGroupId(op.part_id, img_id);
						std::cerr << " group_id= " << group_id << std::endl;
						int optics_group = baseMLO->mydata.getOpticsGroup(op.part_id, img_id);
						std::cerr << " optics_group= " << optics_group << std::endl;
						std::cerr << " ml_model.scale_correction[group_id]= " << baseMLO->mymodel.scale_correction[group_id] << std::endl;
						std::cerr << " exp_significant_weight[img_id]= " << my_significant_weight << std::endl;
						std::cerr << " exp_max_weight[img_id]= " << op.max_weight[img_id] << std::endl;
						std::cerr << " ml_model.sigma2_noise[optics_group]= " << baseMLO->mymodel.sigma2_noise[optics_group] << std::endl;
						CRITICAL(ERRSUMWEIGHTZERO); //"op.sum_weight[img_id]==0"
					}
    LAUNCH_HANDLE_ERROR(cudaGetLastError());

					thresholdIdx = findThresholdIdxInCumulativeSum<XFLOAT>(cumulative_sum, (1 - baseMLO->adaptive_fraction) * op.sum_weight[img_id]);
    // LAUNCH_HANDLE_ERROR(cudaGetLastError());
					my_significant_weight = sorted.getAccValueAt(thresholdIdx);
    // LAUNCH_HANDLE_ERROR(cudaGetLastError());

					CTIC(accMLO->timer,"getArgMaxOnDevice");
					std::pair<size_t, XFLOAT> max_pair = AccUtilities::getArgMaxOnDevice<XFLOAT>(PassWeights[img_id].weights);
					CTOC(accMLO->timer,"getArgMaxOnDevice");
					op.max_index[img_id].fineIdx = PassWeights[img_id].ihidden_overs[max_pair.first];
					op.max_weight[img_id] = max_pair.second;
				}
				else
				{
					my_significant_weight = sorted.getAccValueAt(0);
				}
				// LAUNCH_HANDLE_ERROR(cudaGetLastError());
				CTOC(accMLO->timer,"sort");

				// //xjl
				// FILE *jobs = fopen("./data/iter7.txt", "a");
				// fprintf(jobs,"%ld %ld %lf %lf %lf\n", weightSize, baseMLO->adaptive_oversampling,baseMLO->adaptive_fraction,my_significant_weight,op.sum_weight[img_id]);
				// for  (int i = 0; i < weightSize; i++)
				// 	fprintf(jobs, "%e ", PassWeights[img_id].weights[i]);
				// fprintf(jobs, "\n");
				// fclose(jobs);
				// exit(-1);

			}
			CTOC(accMLO->timer,"sumweight1");
		}

		op.significant_weight[img_id] = (RFLOAT) my_significant_weight;
		} // end loop img_id

#ifdef TIMING
	// if (op.part_id == baseMLO->exp_my_first_part_id)
	if(ctx.thread_id==0)
	{
		if (exp_ipass == 0) baseMLO->timer.toc(baseMLO->TIMING_ESP_WEIGHT1);
		else baseMLO->timer.toc(baseMLO->TIMING_ESP_WEIGHT2);
	}
#endif
}


// calculate cuda temp memory 
template <class MlClass>
size_t calculate_mem_xjl1(MlOptimiser *baseMLO,MlClass *accMLO,long int part_id,SamplingParameters& sp)
{
	// FILE *file = fopen("cal_mem.txt", "a+"); // 打开文件以写入
	int n_images=sp.nr_images;
	int n_dir = (baseMLO->do_skip_align || baseMLO->do_skip_rotate) ? 1 : baseMLO->sampling.rot_angles.size();
	int n_psi=(baseMLO->do_skip_align || baseMLO->do_skip_rotate) ? 1 :  baseMLO->sampling.psi_angles.size();
	int n_trans=(baseMLO->do_skip_align) ? 1 : baseMLO->sampling.NrTranslationalSamplings();
	int n_oversampled_rot = baseMLO->sampling.oversamplingFactorOrientations(0);
	int n_oversampled_trans = baseMLO->sampling.oversamplingFactorTranslations(0);

	// std::cout<<"xjldebug n_images="<<n_images<<" n_dir="<<n_dir<<" n_psi="<<n_psi
	// 		<<" n_trans="<<n_trans<<std::endl;

	size_t now_memory,max_memory;
	now_memory=max_memory=0;
// fprintf(file, "%lld\n", now_memory);
	for (int img_id = 0; img_id < n_images; img_id++)
	{
		int optics_group = baseMLO->mydata.getOpticsGroup(part_id, img_id);
		int now_full_size=baseMLO->image_full_size[optics_group];
		int idist,odist;
		if(accMLO->dataIs3D)
		{
			idist = now_full_size*now_full_size*now_full_size;
	    	odist = now_full_size*now_full_size*(now_full_size/2+1);
		}
		else
		{
			idist = 1*now_full_size*now_full_size;
	    	odist = 1*now_full_size*(now_full_size/2+1);
		}
		//get img_size:315
// #ifdef ACC_DOUBLE_PRECISION
// 		size_t real_size=sizeof(cufftDoubleReal)*idist;
// 		size_t fouriers_size=sizeof(cufftDoubleComplex)*odist;
// #else
// 		size_t real_size=sizeof(cufftReal)*idist;
// 		size_t fouriers_size=sizeof(cufftComplex)*odist;
// #endif
// 		now_memory=real_size+fouriers_size;
// 		max_memory=now_memory>max_memory?now_memory:max_memory;

		size_t randomimage_size=idist*sizeof(XFLOAT);
		now_memory=randomimage_size;
		// fprintf(file, "%lld\n", now_memory);


		//line:330
		size_t noise_size=(now_full_size/2+1)*sizeof(XFLOAT)+
			RND_BLOCK_NUM*RND_BLOCK_SIZE*sizeof(curandState);
		now_memory+=noise_size;
		// fprintf(file, "%lld\n", now_memory);

		max_memory=now_memory>max_memory?now_memory:max_memory;
		now_memory-=noise_size;
		// fprintf(file, "%lld\n", now_memory);


		//line:417
		size_t dimg_recimg_size=idist*sizeof(XFLOAT)*2;
		now_memory+=dimg_recimg_size;
		// fprintf(file, "%lld\n", now_memory);


		//line:421 441
		size_t tmp_size=idist*sizeof(XFLOAT);
		now_memory+=tmp_size;
		// fprintf(file, "%lld\n", now_memory);

		max_memory=now_memory>max_memory?now_memory:max_memory;
		now_memory-=tmp_size;
		// fprintf(file, "%lld\n", now_memory);


		//line:453 466 616
		size_t dfimg=odist*sizeof(ACCCOMPLEX);
		now_memory+=dfimg;
		// fprintf(file, "%lld\n", now_memory);
		max_memory=now_memory>max_memory?now_memory:max_memory;
		now_memory-=dfimg;
		// fprintf(file, "%lld\n", now_memory);


		//line:570
		size_t softmask_size=SOFTMASK_BLOCK_SIZE*sizeof(XFLOAT)*2;
		now_memory+=softmask_size;
		// fprintf(file, "%lld\n", now_memory);
		max_memory=now_memory>max_memory?now_memory:max_memory;
		now_memory-=softmask_size;
		// fprintf(file, "%lld\n", now_memory);


		//line631
		size_t spectrum_size=(size_t)((now_full_size/2+1)+1)*sizeof(XFLOAT);
		now_memory+=spectrum_size;
		// fprintf(file, "%lld\n", now_memory);
		max_memory=now_memory>max_memory?now_memory:max_memory;
		now_memory-=spectrum_size;
		// fprintf(file, "%lld\n", now_memory);


		now_memory-=dimg_recimg_size;
		// fprintf(file, "%lld\n", now_memory);
		now_memory-=randomimage_size;
		// fprintf(file, "%lld\n\n", now_memory);

		
		// std::cout<<"xjldebug now_mem_after_fft_iter"<<max_memory<<" "<<now_memory<<std::endl;
	}
// fclose(file); // 关闭文件
	return max_memory;

}

template <class MlClass>
size_t calculate_mem_xjl2(MlOptimiser *baseMLO,MlClass *accMLO,long int part_id,SamplingParameters& sp)
{
	// FILE *file = fopen("cal_mem.txt", "a+"); // 打开文件以写入

	int n_images=sp.nr_images;
	int n_classes=baseMLO->mymodel.nr_classes;
	unsigned long n_dir = sp.idir_max-sp.idir_min+1;
	unsigned long n_psi=sp.ipsi_max-sp.ipsi_min+1;
	unsigned long n_trans=sp.itrans_max-sp.itrans_min+1;
	int n_oversampled_rot = baseMLO->sampling.oversamplingFactorOrientations(0);
	int n_oversampled_trans = baseMLO->sampling.oversamplingFactorTranslations(0);

	size_t now_memory,max_memory;
	now_memory=max_memory=0;
	// fprintf(file, "%lld\n", now_memory);
	//line:3390
	size_t weightsPerPart=n_classes*n_dir*n_psi*n_trans*n_oversampled_rot*n_oversampled_trans;
	size_t mweight=n_images*weightsPerPart*sizeof(XFLOAT);
	now_memory+=mweight;
	// fprintf(file, "%lld\n", now_memory);

	//coarse
	{
		//line:970
		int rotation_num=n_dir*n_psi*n_oversampled_rot;
		size_t tmp_size=(rotation_num*3+9*2)*sizeof(XFLOAT);
		size_t euler_iori_size=rotation_num*9*n_classes*sizeof(XFLOAT)+
					rotation_num*n_classes*sizeof(long unsigned);

		now_memory+=tmp_size;
		// fprintf(file, "%lld\n", now_memory);
		now_memory+=euler_iori_size;
		// fprintf(file, "%lld\n", now_memory);
		max_memory=now_memory>max_memory?now_memory:max_memory;
		now_memory-=tmp_size;
		// fprintf(file, "%lld\n", now_memory);

		//line:1012
		size_t allweights_size=rotation_num*n_trans*n_oversampled_trans*n_classes*sizeof(XFLOAT);
		now_memory+=allweights_size;
		// fprintf(file, "%lld\n", now_memory);

		//line:1043
		for (int img_id = 0; img_id < n_images; img_id++)
		{
			int optics_group = baseMLO->mydata.getOpticsGroup(part_id, img_id);
			int exp_current_image_size,image_size;
			if (baseMLO->strict_highres_exp > 0.)
			{
				// Use smaller images in both passes and keep a maximum on coarse_size, just like in FREALIGN
				exp_current_image_size = baseMLO->image_coarse_size[optics_group];
			}
			else if (baseMLO->adaptive_oversampling > 0)
			{
				// Use smaller images in the first pass, larger ones in the second pass
				exp_current_image_size =  baseMLO->image_coarse_size[optics_group];
			}
			else
			{
				exp_current_image_size = baseMLO->image_current_size[optics_group];
			}
			if(accMLO->dataIs3D)
			{
				image_size = exp_current_image_size*exp_current_image_size*(exp_current_image_size/2+1);
			}
			else
			{
				image_size = 1*exp_current_image_size*(exp_current_image_size/2+1);
			}
			size_t iter_size=n_trans*n_oversampled_trans*3*sizeof(XFLOAT)+
					image_size*2*sizeof(XFLOAT)+image_size*sizeof(XFLOAT);
			now_memory+=iter_size;
			// fprintf(file, "%lld\n", now_memory);
			max_memory=now_memory>max_memory?now_memory:max_memory;
			now_memory-=iter_size;
			// fprintf(file, "%lld\n", now_memory);
		}
		now_memory-=allweights_size;
		// fprintf(file, "%lld\n", now_memory);
		now_memory-=euler_iori_size;
		// fprintf(file, "%lld\n", now_memory);
		// std::cout<<"xjldebug max_mem_after_coarse"<<(float)max_memory/1024/1024<<std::endl;
		// std::cout<<"xjldebug now_mem"<<(float)now_memory<<std::endl;
	}

	//covert
	{
		size_t pdfori_size=n_classes*n_dir*n_psi*(sizeof(XFLOAT)+sizeof(bool));
		size_t pdftran_size=n_classes*n_trans*(sizeof(XFLOAT)+sizeof(bool));
		size_t pdf_size=n_classes*n_dir*n_psi*sizeof(RFLOAT);
		size_t pdf_total=pdfori_size+pdftran_size+pdf_size;
		now_memory+=pdf_total;
		// fprintf(file, "%lld\n", now_memory);
		// std::cout<<"xjldebug now_mem:pdf_total"<<(float)now_memory<<std::endl;


		//1887
		size_t local_weights_num=n_dir*n_psi*n_trans*n_classes;
		size_t local_weights_size=3*local_weights_num*sizeof(XFLOAT);	
		now_memory+=local_weights_size;
		// fprintf(file, "%lld\n", now_memory);
				// std::cout<<"xjldebug now_mem:local_weights_size"<<(float)now_memory<<std::endl;


		//1898
		size_t tmpidx_size=sizeof(size_t);
		now_memory+=tmpidx_size;
		// fprintf(file, "%lld\n", now_memory);
		max_memory=now_memory>max_memory?now_memory:max_memory;
		now_memory-=tmpidx_size;
		// fprintf(file, "%lld\n", now_memory);

		//1956
		size_t Mcoarse_size=local_weights_num*sizeof(bool);
		now_memory+=Mcoarse_size;
		// fprintf(file, "%lld\n", now_memory);
		max_memory=now_memory>max_memory?now_memory:max_memory;


		now_memory-=Mcoarse_size;
		// fprintf(file, "%lld\n", now_memory);
		now_memory-=local_weights_size;
		// fprintf(file, "%lld\n", now_memory);
		now_memory-=pdf_total;
		// fprintf(file, "%lld\n", now_memory);

		// std::cout<<"xjldebug max_mem_after_convert"<<max_memory<<std::endl;

	}
	now_memory-=mweight;
	// fprintf(file, "%lld\n\n", now_memory);
	// fclose(file); // 关闭文件
	// std::cout<<std::endl;
	// std::cout<<"xjldebug now_mem_after_section2:"<<
	// 	now_memory<<" max memory:"<<(float)max_memory/1024/1024<<"MB"<<std::endl;
	return max_memory;
}

template <class MlClass>
size_t calculate_mem_xjl3(MlOptimiser *baseMLO,MlClass *accMLO,long int part_id,SamplingParameters& sp,std::vector<ProjectionParams> &FineProjectionData)
{
	// FILE *file = fopen("cal_mem.txt", "a+"); // 打开文件以写入

	int current_oversampling = baseMLO->adaptive_oversampling;
	int n_images=sp.nr_images;
	int n_classes=baseMLO->mymodel.nr_classes;
	int n_dir = sp.nr_dir;
	int n_dir2=(baseMLO->do_skip_align || baseMLO->do_skip_rotate) ? 1 : baseMLO->sampling.rot_angles.size();
	int n_psi= sp.nr_psi;
	int n_psi2=(baseMLO->do_skip_align || baseMLO->do_skip_rotate) ? 1 :  baseMLO->sampling.psi_angles.size();
	int n_trans= sp.nr_trans;
	int n_trans2=(baseMLO->do_skip_align) ? 1 : baseMLO->sampling.NrTranslationalSamplings();
	int n_oversampled_rot = sp.nr_oversampled_rot;
	int n_oversampled_rot2=baseMLO->sampling.oversamplingFactorOrientations(current_oversampling);
	int n_oversampled_trans = sp.nr_oversampled_trans;
	int n_oversampled_trans2=baseMLO->sampling.oversamplingFactorTranslations(current_oversampling);

	size_t now_memory,max_memory;
	now_memory=max_memory=0;
// fprintf(file, "%lld\n", now_memory);

	//line:3434
	size_t data_within_FinePassWeights_size=0,bundleD2_size=0;
	for (int img_id = 0; img_id < n_images; img_id++)
	{
		size_t dataSize=FineProjectionData[img_id].orientationNumAllClasses*n_trans*n_oversampled_trans;
		data_within_FinePassWeights_size+=dataSize*(sizeof(XFLOAT)+sizeof(size_t)*4);	
		bundleD2_size+=dataSize*2*sizeof(unsigned long)*sizeof(unsigned char);//AccPtrBundle里面是对AccPtrBundleBYTE的大小alloc的
	}
	now_memory+=data_within_FinePassWeights_size;//_xjldebug46
	// fprintf(file, "%lld\n", now_memory);
	now_memory+=bundleD2_size;//_xjldebug46
	// fprintf(file, "%lld\n", now_memory);
	// printf("max_mem start:%fKB %lf %lf %lf\n",(float)now_memory/1024,
	// 					(float)now_memory/1024+201,(float)data_within_FinePassWeights_size/1024,(float)bundleD2_size/1024);


	//fine
	size_t iter_size;
	for (int img_id = 0; img_id < n_images; img_id++)
	{
		//1266+1335+1350
		int optics_group = baseMLO->mydata.getOpticsGroup(part_id, img_id);
		int exp_current_image_size,image_size;
		if (baseMLO->strict_highres_exp > 0.)
		{
			// Use smaller images in both passes and keep a maximum on coarse_size, just like in FREALIGN
			exp_current_image_size = baseMLO->image_coarse_size[optics_group];
		}
		else
		{
			// Use smaller images in the first pass, larger ones in the second pass
			exp_current_image_size =  baseMLO->image_current_size[optics_group];
		}

		if(accMLO->dataIs3D)
		{
			image_size = exp_current_image_size*exp_current_image_size*(exp_current_image_size/2+1);
		}
		else
		{
			image_size = 1*exp_current_image_size*(exp_current_image_size/2+1);
		}

		iter_size=n_trans*n_oversampled_trans*3*sizeof(XFLOAT)+
				image_size*3*sizeof(XFLOAT)+
				9*FineProjectionData[img_id].orientationNumAllClasses*sizeof(XFLOAT)*sizeof(unsigned char);
			
		now_memory+=iter_size;
		// fprintf(file, "%lld\n", now_memory);
		max_memory=now_memory>max_memory?now_memory:max_memory;
		now_memory-=iter_size;
		// fprintf(file, "%lld\n", now_memory);
		// std::cout<<"xjldebug now_mem_after_fine_iter"<<max_memory<<" "<<iter_size<<std::endl;
	}

	//convert
	size_t pdf_total;
	{
		size_t pdfori_size=n_classes*n_dir*n_psi*(sizeof(XFLOAT)+sizeof(bool));
		size_t pdftran_size=n_classes*n_trans*(sizeof(XFLOAT)+sizeof(bool));
		size_t pdf_size=n_classes*n_dir*n_psi*sizeof(RFLOAT);
		pdf_total=pdfori_size+pdftran_size+pdf_size;
		now_memory+=pdf_total;
		// fprintf(file, "%lld\n", now_memory);
		// printf("%lu %lu %lu %lf\n",n_classes,n_dir,n_psi,(float)pdf_size/1024);
		// printf("afterpdf %lf %lf KB\n",(float)now_memory/1024,(float)max_memory/1024);

		for (int img_id = 0; img_id < n_images; img_id++)
		{
			size_t dataSize=FineProjectionData[img_id].orientationNumAllClasses*n_trans*n_oversampled_trans;
			size_t sorted_cumu_size=2*dataSize*sizeof(XFLOAT);
			size_t tmpdebug=n_trans2*n_oversampled_trans*n_dir2*n_psi2*2*sizeof(XFLOAT)*n_oversampled_rot2;
			// printf("tmpdebug %e %e %lf %lf\n",(float)sorted_cumu_size,(float)tmpdebug);
			now_memory+=sorted_cumu_size;
			// fprintf(file, "%lld\n", now_memory);

			size_t tmpidx_size=sizeof(size_t);
			now_memory+=tmpidx_size;
			// fprintf(file, "%lld\n", now_memory);
			max_memory=now_memory>max_memory?now_memory:max_memory;
			now_memory-=tmpidx_size;
			// fprintf(file, "%lld\n", now_memory);
			now_memory-=sorted_cumu_size;
			// fprintf(file, "%lld\n", now_memory);
		}
		now_memory-=pdf_total;
		// fprintf(file, "%lld\n", now_memory);

		// std::cout<<"xjldebug now_mem_after_convert"<<(XFLOAT)max_memory<<std::endl;
	}
	size_t bundlesws_size=bundleD2_size;
	now_memory+=bundlesws_size;//_xjldebug46
	// fprintf(file, "%lld\n", now_memory);

	//storeweightedsums
	size_t pweights_size,oo_trans_size,iter_size2;
	{
		//2267 2374
		for (int img_id = 0; img_id < n_images; img_id++)
		{
			oo_trans_size=n_classes*n_trans*n_oversampled_trans*4*sizeof(XFLOAT);
			size_t sumBlockNum=FineProjectionData[img_id].orientationNumAllClasses;
			pweights_size=sumBlockNum*5*sizeof(XFLOAT);
			
			now_memory+=oo_trans_size;
			// fprintf(file, "%lld\n", now_memory);
			now_memory+=pweights_size;
			// fprintf(file, "%lld\n", now_memory);
			max_memory=now_memory>max_memory?now_memory:max_memory;
			now_memory-=oo_trans_size;
			// fprintf(file, "%lld\n", now_memory);
			now_memory-=pweights_size;
			// fprintf(file, "%lld\n", now_memory);
		}
		//2573

		for (int img_id = 0; img_id < n_images; img_id++)
		{
			size_t trans_xyz_size=n_trans*n_oversampled_trans*3*sizeof(XFLOAT);

			int optics_group = baseMLO->mydata.getOpticsGroup(part_id, img_id);
			int exp_current_image_size=baseMLO->image_current_size[optics_group];
			int image_size;
			if(accMLO->dataIs3D)
			{
				image_size = exp_current_image_size*exp_current_image_size*(exp_current_image_size/2+1);
			}
			else
			{
				image_size = 1*exp_current_image_size*(exp_current_image_size/2+1);
			}
			size_t image_realted_size=image_size*6*sizeof(XFLOAT);

			size_t wdiff2s_size=(n_classes*image_size*2+image_size)*sizeof(XFLOAT);
			size_t sorted_weights_size=FineProjectionData[img_id].orientationNumAllClasses * 
							n_trans*n_oversampled_trans*sizeof(XFLOAT);
			size_t eulers_size=FineProjectionData[img_id].orientationNumAllClasses*9*sizeof(XFLOAT);

			iter_size2=trans_xyz_size+image_realted_size+wdiff2s_size+sorted_weights_size+eulers_size;
			// printf("\n\nxjldebug iter_size:%fKB %f %f %f %f\n",(float)(trans_xyz_size+now_memory)/1024,
			// 	(float)image_realted_size/1024,(float)wdiff2s_size/1024,
			// 	(float)sorted_weights_size/1024,(float)eulers_size/1024);
			now_memory+=iter_size2;
			// fprintf(file, "%lld\n", now_memory);
			max_memory=now_memory>max_memory?now_memory:max_memory;
			now_memory-=iter_size2;
			// fprintf(file, "%lld\n", now_memory);
		}
	}
	now_memory-=bundleD2_size;//_xjldebug46
	// fprintf(file, "%lld\n", now_memory);
	now_memory-=data_within_FinePassWeights_size;//_xjldebug46
	// fprintf(file, "%lld\n", now_memory);
	now_memory-=bundlesws_size;//_xjldebug46
	// fprintf(file, "%lld\n", now_memory);



	// std::cout<<"xjldebug now_mem_after_section2:"<<now_memory
	// 	<<" max memory:"<<(float)max_memory/1024/1024<<"MB"<<std::endl;
	// if(max_memory/1024/1024>200)
	// {
	// 	std::cout<<"xjldebug now_mem_after_section2:"<<now_memory
	// 	<<" max memory:"<<(float)max_memory/1024/1024<<"MB"<<std::endl;
	// 	std::cout<<"xjldebug data_within_FinePassWeights_size:"<<data_within_FinePassWeights_size/1024/1024<<"MB"<<std::endl;
	// 	std::cout<<"xjldebug bundleD2_size:"<<bundleD2_size/1024/1024<<"MB"<<std::endl;
	// 	std::cout<<"xjldebug bundlesws_size:"<<bundlesws_size/1024/1024<<"MB"<<std::endl;
	// 	std::cout<<"xjldebug iter_size:"<<iter_size/1024/1024<<"MB"<<std::endl;
	// 	std::cout<<"xjldebug pdf_size:"<<pdf_total/1024/1024<<"MB"<<std::endl;
	// 	std::cout<<"xjldebug iter_size2:"<<iter_size2/1024/1024<<"MB"<<std::endl;
	// 	std::cout<<"xjldebug pweights_size:"<<pweights_size/1024/1024<<"MB"<<std::endl;
	// 	std::cout<<"xjldebug oo_trans_size:"<<oo_trans_size/1024/1024<<"MB"<<std::endl;
	// 	std::cout<<"orinum"<<FineProjectionData[0].orientationNumAllClasses<<" "<<n_trans*n_oversampled_trans<<std::endl;
	// }
	return max_memory;
}




template <class MlClass>
void accDoExpectationOneParticlePre(Context<MlClass>& ctx)
{
	MlOptimiser *baseMLO = ctx.baseMLO;
	MlClass* accMLO = ctx.accMLO;
#ifdef NEWMEM
	AccPtrFactoryNew& ptrFactory = ctx.ptrFactory;//20,21,22
#else
	AccPtrFactory& ptrFactory = ctx.ptrFactory;
#endif
	// AccPtrFactory& ptrFactory = ctx.ptrFactory;
	int& thread_id = ctx.thread_id;

	SamplingParameters& sp = ctx.sp;

	unsigned long& part_id_sorted = ctx.part_id_sorted;


	CTIC(timer,"oneParticlePre");
#ifdef TIMING
	if (thread_id == 0)
		baseMLO->timer.tic(baseMLO->TIMING_ESP_DIFF2_A);
#endif

	ctx.part_id = baseMLO->mydata.sorted_idx[part_id_sorted];
	long int& part_id = ctx.part_id;

	sp.nr_images = baseMLO->mydata.numberOfImagesInParticle(part_id);
	// if(baseMLO->xjl_type==1)
	// 	std::cout<<"exec pic1:"<< part_id_sorted<< " "<<part_id<<std::endl;
	// // std::cout<<<"exec pic1:"<<part_id_sorted<<"  original id="<<part_id<<std::endl;

	// ctx.op.resize(sp.nr_images, part_id);
	ctx.op = OptimisationParamters(sp.nr_images, part_id);
	OptimisationParamters& op = ctx.op;
	// OptimisationParamters op(sp.nr_images, part_id);
	if (baseMLO->mydata.is_3D)
		op.FstMulti.resize(sp.nr_images);

	// In the first iteration, multiple seeds will be generated
	// A single random class is selected for each pool of images, and one does not marginalise over the orientations
	// The optimal orientation is based on signal-product (rather than the signal-intensity sensitive Gaussian)
	// If do_firstiter_cc, then first perform a single iteration with K=1 and cross-correlation criteria, afterwards

	// Decide which classes to integrate over (for random class assignment in 1st iteration)
	sp.iclass_min = 0;
	sp.iclass_max = baseMLO->mymodel.nr_classes - 1;
	// low-pass filter again and generate the seeds
	if (baseMLO->do_generate_seeds)
	{
		if (baseMLO->do_firstiter_cc && baseMLO->iter == 1)
		{
			// In first (CC) iter, use a single reference (and CC)
			sp.iclass_min = sp.iclass_max = 0;
		}
		else if ( (baseMLO->do_firstiter_cc && baseMLO->iter == 2) ||

				(!baseMLO->do_firstiter_cc && baseMLO->iter == 1))
		{
			// In second CC iter, or first iter without CC: generate the seeds
			// Now select a single random class
			// exp_part_id is already in randomized order (controlled by -seed)
			// WARNING: USING SAME iclass_min AND iclass_max FOR SomeParticles!!
			// Make sure random division is always the same with the same seed
			long int idx = part_id_sorted - baseMLO->exp_my_first_part_id;
			if (idx >= baseMLO->exp_random_class_some_particles.size())
				REPORT_ERROR("BUG: expectationOneParticle idx>random_class_some_particles.size()");
			sp.iclass_min = sp.iclass_max = baseMLO->exp_random_class_some_particles[idx];
		}
	}
// #ifdef NEWMEM
// 	size_t max_mem=calculate_mem_xjl1<MlClass>(baseMLO,accMLO,part_id,&sp);
// 	// max_mem=10000;
// 	ptrFactory.getOneTaskAllocator(1, max_mem);

// #endif
	CTOC(timer,"oneParticlePre");
}



template <class MlClass>
void accDoExpectationOneParticlePreMemAlloc(Context<MlClass>& ctx)
{
	MlOptimiser *baseMLO = ctx.baseMLO;
	MlClass* accMLO = ctx.accMLO;
#ifdef NEWMEM
	AccPtrFactoryNew& ptrFactory = ctx.ptrFactory;//20,21,22
#else
	AccPtrFactory& ptrFactory = ctx.ptrFactory;
#endif
	// AccPtrFactory& ptrFactory = ctx.ptrFactory;
	int& thread_id = ctx.thread_id;

	SamplingParameters& sp = ctx.sp;
	long int& part_id = ctx.part_id;

	CTIC(timer,"oneParticlePreMemAlloc");
#ifdef NEWMEM
	size_t max_mem=calculate_mem_xjl1<MlClass>(baseMLO, accMLO, part_id, sp);
	// max_mem=10000;
	ptrFactory.getOneTaskAllocator(1, max_mem);

#endif
	CTOC(timer,"oneParticlePreMemAlloc");
}


template <class MlClass>
void accDoExpectationOneParticlePostPerBodyGetFTAndCtfs(Context<MlClass>& ctx)
{
	CTIC(timer,"GetFTAndCtfs");

	MlOptimiser *baseMLO = ctx.baseMLO;
	MlClass* accMLO = ctx.accMLO;
	MlClass* myInstance = ctx.accMLO;
#ifdef NEWMEM
	AccPtrFactoryNew& ptrFactory = ctx.ptrFactory;
#else
	AccPtrFactory& ptrFactory = ctx.ptrFactory;
#endif
	int& thread_id = ctx.thread_id;
	unsigned long& part_id_sorted = ctx.part_id_sorted;
	long int& part_id = ctx.part_id;

	SamplingParameters& sp = ctx.sp;
	int ibody = ctx.ibody;

	LAUNCH_PRIVATE_ERROR(cudaGetLastError(),accMLO->errorStatus);
	
	// OptimisationParamters op(sp.nr_images, part_id);
	// ctx.op.resize(sp.nr_images, part_id);
	ctx.op = OptimisationParamters(sp.nr_images, part_id);
	OptimisationParamters& op = ctx.op;	
	if (baseMLO->mydata.is_3D)
		op.FstMulti.resize(sp.nr_images);

	// TODO suport continue --fjy
	// // Skip this body if keep_fixed_bodies[ibody] or if it's angular accuracy is worse than 1.5x the sampling rate
	// if ( baseMLO->mymodel.nr_bodies > 1 && baseMLO->mymodel.keep_fixed_bodies[ibody] > 0)
	// 	continue;
	
	if ( baseMLO->mymodel.nr_bodies > 1 && baseMLO->mymodel.keep_fixed_bodies[ibody] > 0)
		printf("[ERROR] !!!!!!!! skip body %d\n",ibody);

	// Global exp_metadata array has metadata of all particles. Where does part_id start?
	for (long int iori = baseMLO->exp_my_first_part_id; iori <= baseMLO->exp_my_last_part_id; iori++)
	{
		if (iori == part_id_sorted) break;
		op.metadata_offset += baseMLO->mydata.numberOfImagesInParticle(iori);
	}
#ifdef TIMING
if (thread_id == 0)
baseMLO->timer.toc(baseMLO->TIMING_ESP_DIFF2_A);
#endif
	CTIC(timer,"getFourierTransformsAndCtfs");
	getFourierTransformsAndCtfs<MlClass>(part_id, op, sp, baseMLO, myInstance, ptrFactory, ctx, ibody,thread_id);
	CTOC(timer,"getFourierTransformsAndCtfs");

	// To deal with skipped alignments/rotations
	if (baseMLO->do_skip_align)
	{
		sp.itrans_min = sp.itrans_max = sp.idir_min = sp.idir_max = sp.ipsi_min = sp.ipsi_max =
				part_id_sorted - baseMLO->exp_my_first_part_id;
	}
	else
	{
		sp.itrans_min = 0;
		sp.itrans_max = baseMLO->sampling.NrTranslationalSamplings() - 1;
	}
	if (baseMLO->do_skip_align || baseMLO->do_skip_rotate)
	{
		sp.idir_min = sp.idir_max = sp.ipsi_min = sp.ipsi_max =
				part_id_sorted - baseMLO->exp_my_first_part_id;
	}
	else if (baseMLO->do_only_sample_tilt)
	{
		sp.idir_min = 0;
		sp.idir_max = baseMLO->sampling.NrDirections(0, &op.pointer_dir_nonzeroprior) - 1;
		sp.ipsi_min = sp.ipsi_max = part_id_sorted - baseMLO->exp_my_first_part_id;

	}
	else
	{
		sp.idir_min = sp.ipsi_min = 0;
		sp.idir_max = baseMLO->sampling.NrDirections(0, &op.pointer_dir_nonzeroprior) - 1;
		sp.ipsi_max = baseMLO->sampling.NrPsiSamplings(0, &op.pointer_psi_nonzeroprior ) - 1;
	}
}


template <class MlClass>
void accDoExpectationOneParticlePostPerBodyGetFTAndCtfsMemAlloc(Context<MlClass>& ctx)
{
	CTIC(timer,"GetFTAndCtfsMemAlloc");

	MlOptimiser *baseMLO = ctx.baseMLO;
	MlClass* accMLO = ctx.accMLO;
	MlClass* myInstance = ctx.accMLO;
#ifdef NEWMEM
	AccPtrFactoryNew& ptrFactory = ctx.ptrFactory;
#else
	AccPtrFactory& ptrFactory = ctx.ptrFactory;
#endif
	int& thread_id = ctx.thread_id;
	unsigned long& part_id_sorted = ctx.part_id_sorted;
	long int& part_id = ctx.part_id;
	OptimisationParamters& op = ctx.op;
	SamplingParameters& sp = ctx.sp;
	int ibody = ctx.ibody;

#ifdef NEWMEM
	size_t max_mem=calculate_mem_xjl2<MlClass>(baseMLO,accMLO,part_id,sp);
	ptrFactory.getOneTaskAllocator(2, max_mem);
#endif

	// Initialise significant weight to minus one, so that all coarse sampling points will be handled in the first pass
	op.significant_weight.resize(sp.nr_images, -1.);

	/// -- This is a iframe-indexed vector, each entry of which is a dense data-array. These are replacements to using
	//    Mweight in the sparse (Fine-sampled) pass, coarse is unused but created empty input for convert ( FIXME )
#ifdef NEWMEM
	ctx.CoarsePassWeights = new std::vector <IndexedDataArrayNew >(1, ptrFactory);
	ctx.FinePassWeights = new std::vector <IndexedDataArrayNew >(sp.nr_images, ptrFactory);
#else
	ctx.CoarsePassWeights = new std::vector <IndexedDataArray >(1, ptrFactory);
	ctx.FinePassWeights = new std::vector <IndexedDataArray >(sp.nr_images, ptrFactory);
#endif
	// -- This is a iframe-indexed vector, each entry of which is a class-indexed vector of masks, one for each
	//    class in FinePassWeights
	ctx.FinePassClassMasks = new std::vector < std::vector <IndexedDataArrayMask > >(sp.nr_images, std::vector <IndexedDataArrayMask >(baseMLO->mymodel.nr_classes, ptrFactory));
	// -- This is a iframe-indexed vector, each entry of which is parameters used in the projection-operations *after* the
	//    coarse pass, declared here to keep scope to storeWS
	ctx.FineProjectionData = new std::vector < ProjectionParams >(sp.nr_images, baseMLO->mymodel.nr_classes);


#ifdef NEWMEM

	ctx.bundleD2 = new std::vector < AccPtrBundleNew >(sp.nr_images, ptrFactory.makeBundle());//应该可以往后移，生命周期没那么长
	ctx.bundleSWS = new std::vector < AccPtrBundleNew >(sp.nr_images, ptrFactory.makeBundle());

#else
	AccPtrFactory& ptrFactoryxjl = ctx.ptrFactory;
	ctx.bundleD2 = new std::vector < AccPtrBundle >(sp.nr_images, ptrFactoryxjl.makeBundle());//应该可以往后移，生命周期没那么长
	ctx.bundleSWS = new std::vector < AccPtrBundle >(sp.nr_images, ptrFactoryxjl.makeBundle());

	// ctx.bundleD2 = new std::vector < AccPtrBundle >(sp.nr_images, ptrFactory.makeBundle());//应该可以往后移，生命周期没那么长
	// ctx.bundleSWS = new std::vector < AccPtrBundle >(sp.nr_images, ptrFactory.makeBundle());
#endif

#ifdef TIMING
if (thread_id == 0)
baseMLO->timer.tic(baseMLO->TIMING_ESP_DIFF2_B);
#endif

	// Use coarse sampling in the first pass, oversampled one the second pass
	sp.current_oversampling = 0;

	sp.nr_dir = (baseMLO->do_skip_align || baseMLO->do_skip_rotate) ? 1 : baseMLO->sampling.NrDirections(0, &op.pointer_dir_nonzeroprior);
	sp.nr_psi = (baseMLO->do_skip_align || baseMLO->do_skip_rotate) ? 1 : baseMLO->sampling.NrPsiSamplings(0, &op.pointer_psi_nonzeroprior);
	sp.nr_trans = (baseMLO->do_skip_align) ? 1 : baseMLO->sampling.NrTranslationalSamplings();
	sp.nr_oversampled_rot = baseMLO->sampling.oversamplingFactorOrientations(sp.current_oversampling);
	sp.nr_oversampled_trans = baseMLO->sampling.oversamplingFactorTranslations(sp.current_oversampling);
#ifdef TIMING
if (thread_id == 0)
baseMLO->timer.toc(baseMLO->TIMING_ESP_DIFF2_B);
#endif

	op.min_diff2.resize(sp.nr_images, 0);

	unsigned long weightsPerPart(baseMLO->mymodel.nr_classes * sp.nr_dir * sp.nr_psi * sp.nr_trans * sp.nr_oversampled_rot * sp.nr_oversampled_trans);

	op.Mweight.resizeNoCp(1,1,sp.nr_images, weightsPerPart);

	ctx.Mweight = ptrFactory.make<XFLOAT>();
#ifdef NEWMEM
	AccPtrNew<XFLOAT>& Mweight = ctx.Mweight;
#else
	AccPtr<XFLOAT>& Mweight = ctx.Mweight;
#endif
	// AccPtr<XFLOAT> Mweight = ptrFactory.make<XFLOAT>();

	Mweight.setSize(sp.nr_images * weightsPerPart);
	Mweight.setHostPtr(op.Mweight.data);
	Mweight.deviceAlloc();
	deviceInitValue<XFLOAT>(Mweight, -std::numeric_limits<XFLOAT>::max());
	Mweight.streamSync();

	ctx.exp_ipass = 0;
	ctx.ibody = ibody;

	CTOC(timer,"GetFTAndCtfsMemAlloc");
}


template <class MlClass>
void accDoExpectationOneParticlePostPerBodyCvtDToWCoarse(Context<MlClass>& ctx)
{
	CTIC(timer,"CvtDToWCoarse");
	MlOptimiser *baseMLO = ctx.baseMLO;
	MlClass* accMLO = ctx.accMLO;
	MlClass* myInstance = ctx.accMLO;
	int& thread_id = ctx.thread_id;
	unsigned long& part_id_sorted = ctx.part_id_sorted;
	long int& part_id = ctx.part_id;

	SamplingParameters& sp = ctx.sp;
	OptimisationParamters& op = ctx.op;

#ifdef NEWMEM
	AccPtrFactoryNew& ptrFactory = ctx.ptrFactory;
	std::vector <IndexedDataArrayNew >& CoarsePassWeights = *ctx.CoarsePassWeights;
	std::vector <IndexedDataArrayNew >& FinePassWeights = *ctx.FinePassWeights;
	// std::vector <IndexedDataArray >& CoarsePassWeights = *ctx.CoarsePassWeights;
	// std::vector <IndexedDataArray >& FinePassWeights = *ctx.FinePassWeights;
	std::vector < std::vector <IndexedDataArrayMask > >& FinePassClassMasks = *ctx.FinePassClassMasks;
	std::vector < ProjectionParams >& FineProjectionData = *ctx.FineProjectionData;
	std::vector < AccPtrBundleNew >& bundleD2 = *ctx.bundleD2;
	std::vector < AccPtrBundleNew >& bundleSWS = *ctx.bundleSWS;
	// std::vector < AccPtrBundle >& bundleD2 = *ctx.bundleD2;
	// std::vector < AccPtrBundle >& bundleSWS = *ctx.bundleSWS;
	AccPtrNew<XFLOAT>& Mweight = ctx.Mweight;	
#else
	AccPtrFactory& ptrFactory = ctx.ptrFactory;
	std::vector <IndexedDataArray >& CoarsePassWeights = *ctx.CoarsePassWeights;
	std::vector <IndexedDataArray >& FinePassWeights = *ctx.FinePassWeights;
	std::vector < std::vector <IndexedDataArrayMask > >& FinePassClassMasks = *ctx.FinePassClassMasks;
	std::vector < ProjectionParams >& FineProjectionData = *ctx.FineProjectionData;
	std::vector < AccPtrBundle >& bundleD2 = *ctx.bundleD2;
	std::vector < AccPtrBundle >& bundleSWS = *ctx.bundleSWS;
	AccPtr<XFLOAT>& Mweight = ctx.Mweight;
#endif

	int ibody = ctx.ibody;

	LAUNCH_PRIVATE_ERROR(cudaGetLastError(),accMLO->errorStatus);

	CTIC(timer,"convertAllSquaredDifferencesToWeightsCoarse");
	convertAllSquaredDifferencesToWeights<MlClass>(0, op, sp, baseMLO, myInstance, CoarsePassWeights, FinePassClassMasks, Mweight, ptrFactory, ibody,thread_id, ctx);
	CTOC(timer,"convertAllSquaredDifferencesToWeightsCoarse");

	ctx.allWeights.freeIfSet();
	for(int i=baseMLO->mymodel.nr_classes-1;i>=0;i--)
	{
		ctx.projectorPlans[i].clear();
	}
	ctx.projectorPlans.clear();
	ctx.Mweight.freeIfSet();


#ifdef TIMING
if (thread_id == 0)
baseMLO->timer.tic(baseMLO->TIMING_ESP_DIFF2_E);
#endif


#ifdef TIMING
if (thread_id == 0)
baseMLO->timer.tic(baseMLO->TIMING_ESP_DIFF2_B);
#endif

	// Use coarse sampling in the first pass, oversampled one the second pass
	sp.current_oversampling = baseMLO->adaptive_oversampling;

	sp.nr_dir = (baseMLO->do_skip_align || baseMLO->do_skip_rotate) ? 1 : baseMLO->sampling.NrDirections(0, &op.pointer_dir_nonzeroprior);
	sp.nr_psi = (baseMLO->do_skip_align || baseMLO->do_skip_rotate) ? 1 : baseMLO->sampling.NrPsiSamplings(0, &op.pointer_psi_nonzeroprior);
	sp.nr_trans = (baseMLO->do_skip_align) ? 1 : baseMLO->sampling.NrTranslationalSamplings();
	sp.nr_oversampled_rot = baseMLO->sampling.oversamplingFactorOrientations(sp.current_oversampling);
	sp.nr_oversampled_trans = baseMLO->sampling.oversamplingFactorTranslations(sp.current_oversampling);
	// printf("#### nr_oversampled_rot %2d  nr_oversampled_trans : %2d\n", sp.nr_oversampled_rot, sp.nr_oversampled_trans);
#ifdef TIMING
if (thread_id == 0)
baseMLO->timer.toc(baseMLO->TIMING_ESP_DIFF2_B);
#endif

	op.min_diff2.resize(sp.nr_images, 0);

#ifdef TIMING
if (thread_id == 0)
baseMLO->timer.tic(baseMLO->TIMING_ESP_DIFF2_D);
#endif
//		// -- go through all classes and generate projectionsetups for all classes - to be used in getASDF and storeWS below --
//		// the reason to do this globally is subtle - we want the orientation_num of all classes to estimate a largest possible
//		// weight-array, which would be insanely much larger than necessary if we had to assume the worst.
	for (int img_id = 0; img_id < sp.nr_images; img_id++)
	{
		FineProjectionData[img_id].orientationNumAllClasses = 0;
		for (int exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
		{
			if(exp_iclass>0)
				FineProjectionData[img_id].class_idx[exp_iclass] = FineProjectionData[img_id].rots.size();
			FineProjectionData[img_id].class_entries[exp_iclass] = 0;

			CTIC(timer,"generateProjectionSetup");
			FineProjectionData[img_id].orientationNumAllClasses += generateProjectionSetupFine(
					op,
					sp,
					baseMLO,
					exp_iclass,
					FineProjectionData[img_id]);
			CTOC(timer,"generateProjectionSetup");

		}
		//set a maximum possible size for all weights (to be reduced by significance-checks)
#ifndef NEWMEM 
		size_t dataSize = FineProjectionData[img_id].orientationNumAllClasses*sp.nr_trans*sp.nr_oversampled_trans;
		FinePassWeights[img_id].setDataSize(dataSize);

		FinePassWeights[img_id].dual_alloc_all();
		// printf("go_xjldebug bundleD2:dataSize %lld \n",2*(FineProjectionData[img_id].orientationNumAllClasses*sp.nr_trans*sp.nr_oversampled_trans)*sizeof(unsigned long));

		bundleD2[img_id].setSize(2*(FineProjectionData[img_id].orientationNumAllClasses*sp.nr_trans*sp.nr_oversampled_trans)*sizeof(unsigned long));
		bundleD2[img_id].allAlloc();
		// printf("go_xjldebug \n");
#endif
	}

#ifdef TIMING
	if (thread_id == 0)
	baseMLO->timer.toc(baseMLO->TIMING_ESP_DIFF2_D);
#endif
	CTOC(timer,"CvtDToWCoarse");

}


template <class MlClass>
void accDoExpectationOneParticlePostPerBodyCvtDToWCoarseMemAlloc(Context<MlClass>& ctx)
{
	MlOptimiser *baseMLO = ctx.baseMLO;
	MlClass* myInstance = ctx.accMLO;
	long int& part_id = ctx.part_id;
	SamplingParameters& sp = ctx.sp;
#ifdef NEWMEM
	AccPtrFactoryNew& ptrFactory = ctx.ptrFactory;
	std::vector <IndexedDataArrayNew >& FinePassWeights = *ctx.FinePassWeights;
	std::vector < ProjectionParams >& FineProjectionData = *ctx.FineProjectionData;
	std::vector < AccPtrBundleNew >& bundleD2 = *ctx.bundleD2;
#else
	AccPtrFactory& ptrFactory = ctx.ptrFactory;
	std::vector <IndexedDataArray >& FinePassWeights = *ctx.FinePassWeights;
	std::vector < ProjectionParams >& FineProjectionData = *ctx.FineProjectionData;
	std::vector < AccPtrBundle >& bundleD2 = *ctx.bundleD2;
#endif
	
	CTIC(timer,"CvtDToWCoarseMemAlloc");
#ifdef NEWMEM
	size_t max_mem=calculate_mem_xjl3<MlClass>(baseMLO,myInstance,part_id,sp,FineProjectionData);
	ptrFactory.getOneTaskAllocator(3, max_mem);

	for(int img_id=0;img_id<sp.nr_images;img_id++)
	{
		size_t dataSize = FineProjectionData[img_id].orientationNumAllClasses*sp.nr_trans*sp.nr_oversampled_trans;
		FinePassWeights[img_id].setDataSize(dataSize,ptrFactory);
		bundleD2[img_id].change_task_allocator(ptrFactory.getAllocator_task());

		// FinePassWeights[img_id].setDataSize(dataSize);
		//_xjldebug46

		FinePassWeights[img_id].dual_alloc_all();
		// printf("go_xjldebug bundleD2:dataSize %lld \n",2*(FineProjectionData[img_id].orientationNumAllClasses*sp.nr_trans*sp.nr_oversampled_trans)*sizeof(unsigned long));

		bundleD2[img_id].setSize(2*(FineProjectionData[img_id].orientationNumAllClasses*sp.nr_trans*sp.nr_oversampled_trans)*sizeof(unsigned long));
		bundleD2[img_id].allAlloc();
		// printf("go_xjldebug \n");
	}
#endif
	CTOC(timer, "CvtDToWCoarseMemAlloc");
}

template <class MlClass>
void accDoExpectationOneParticlePostPerBodyCvtDToWFine(Context<MlClass>& ctx)
{
	CTIC(timer,"CvtDToWFine");
	MlOptimiser *baseMLO = ctx.baseMLO;
	MlClass* accMLO = ctx.accMLO;
	MlClass* myInstance = ctx.accMLO;
	int& thread_id = ctx.thread_id;
	unsigned long& part_id_sorted = ctx.part_id_sorted;
	long int& part_id = ctx.part_id;

	SamplingParameters& sp = ctx.sp;
	OptimisationParamters& op = ctx.op;

#ifdef NEWMEM
	AccPtrFactoryNew& ptrFactory = ctx.ptrFactory;
	std::vector <IndexedDataArrayNew >& CoarsePassWeights = *ctx.CoarsePassWeights;
	std::vector <IndexedDataArrayNew >& FinePassWeights = *ctx.FinePassWeights;
	// std::vector <IndexedDataArray >& CoarsePassWeights = *ctx.CoarsePassWeights;
	// std::vector <IndexedDataArray >& FinePassWeights = *ctx.FinePassWeights;
	std::vector < std::vector <IndexedDataArrayMask > >& FinePassClassMasks = *ctx.FinePassClassMasks;
	std::vector < ProjectionParams >& FineProjectionData = *ctx.FineProjectionData;
	std::vector < AccPtrBundleNew >& bundleD2 = *ctx.bundleD2;
	std::vector < AccPtrBundleNew >& bundleSWS = *ctx.bundleSWS;
	// 	std::vector < AccPtrBundle >& bundleD2 = *ctx.bundleD2;
	// std::vector < AccPtrBundle >& bundleSWS = *ctx.bundleSWS;
	// AccPtrNew<XFLOAT>& Mweight = ctx.Mweight;	
#else
	AccPtrFactory& ptrFactory = ctx.ptrFactory;
	std::vector <IndexedDataArray >& CoarsePassWeights = *ctx.CoarsePassWeights;
	std::vector <IndexedDataArray >& FinePassWeights = *ctx.FinePassWeights;
	std::vector < std::vector <IndexedDataArrayMask > >& FinePassClassMasks = *ctx.FinePassClassMasks;
	std::vector < ProjectionParams >& FineProjectionData = *ctx.FineProjectionData;
	std::vector < AccPtrBundle >& bundleD2 = *ctx.bundleD2;
	std::vector < AccPtrBundle >& bundleSWS = *ctx.bundleSWS;
	// AccPtr<XFLOAT>& Mweight = ctx.Mweight;	
#endif

	int ibody = ctx.ibody;


	FinePassWeights[0].weights.cpToHost();
#ifdef NEWMEM
	AccPtrNew<XFLOAT> Mweight_dummy = ptrFactory.make<XFLOAT>(); //DUMMY
#else
	AccPtr<XFLOAT> Mweight_dummy = ptrFactory.make<XFLOAT>(); //DUMMY
#endif
	LAUNCH_HANDLE_ERROR(cudaGetLastError());


	CTIC(timer,"convertAllSquaredDifferencesToWeightsFine");
	convertAllSquaredDifferencesToWeights<MlClass>(1, op, sp, baseMLO, myInstance, FinePassWeights, FinePassClassMasks, Mweight_dummy, ptrFactory, ibody,thread_id, ctx);
	CTOC(timer,"convertAllSquaredDifferencesToWeightsFine");


#ifdef TIMING
if (thread_id == 0)
baseMLO->timer.tic(baseMLO->TIMING_ESP_DIFF2_E);
#endif

	// For the reconstruction step use mymodel.current_size!
	// as of 3.1, no longer necessary?
	sp.current_image_size = baseMLO->mymodel.current_size;

	for (unsigned long img_id = 0; img_id < sp.nr_images; img_id++)
	{
#ifdef NEWMEM
		bundleSWS[img_id].change_task_allocator(ptrFactory.getAllocator_task());
#endif //_xjldebug46
		bundleSWS[img_id].setSize(2*(FineProjectionData[img_id].orientationNumAllClasses)*sizeof(unsigned long));
		bundleSWS[img_id].allAlloc();
	}

#ifdef TIMING
if (thread_id == 0)
baseMLO->timer.toc(baseMLO->TIMING_ESP_DIFF2_E);
#endif
	// CTOC(timer,"CvtDToWFine");

	// CTIC(timer,"storeWeightedSums");
	// storeWeightedSumsCollectData<MlClass>(ctx);
	// storeWeightedSumsMaximization<MlClass>(ctx);
	// // storeWeightedSums<MlClass>(op, sp, baseMLO, myInstance, FinePassWeights, FineProjectionData, FinePassClassMasks, ptrFactory, ibody, bundleSWS,thread_id, ctx);
	// CTOC(timer,"storeWeightedSums");

	CTOC(timer,"CvtDToWFine");

}

template<class MlClass>
void accDoExpectationOneParticleCleanup(Context<MlClass> &ctx) {
    SamplingParameters& sp = ctx.sp;
    MlClass *accMLO = ctx.accMLO;
    CTIC(accMLO->timer,"accDoExpectationOneParticleCleanup");

#ifdef NEWMEM
	std::vector < AccPtrBundleNew >& bundleD2 = *ctx.bundleD2;
	std::vector < AccPtrBundleNew >& bundleSWS = *ctx.bundleSWS;
	// std::vector < AccPtrBundle >& bundleD2 = *ctx.bundleD2;
	// std::vector < AccPtrBundle >& bundleSWS = *ctx.bundleSWS;
	for (long int img_id = sp.nr_images-1; img_id >=0; img_id--)
	{
		bundleSWS[img_id].free();
	}
	for (long int img_id = sp.nr_images-1; img_id >=0; img_id--)
	{
		bundleD2[img_id].free();
	}
	std::vector <IndexedDataArrayNew >& FinePassWeights = *ctx.FinePassWeights;
    // std::vector <IndexedDataArray >& FinePassWeights = *ctx.FinePassWeights;

#else
    std::vector <IndexedDataArray >& FinePassWeights = *ctx.FinePassWeights;
#endif

    for (long int img_id = 0; img_id < sp.nr_images; img_id++)
	{
		FinePassWeights[img_id].dual_free_all();
	}

	delete ctx.CoarsePassWeights;
	delete ctx.FinePassWeights;
	delete ctx.FinePassClassMasks;
	delete ctx.FineProjectionData;
	delete ctx.bundleD2;
	delete ctx.bundleSWS;

    delete ctx.oversampled_translations_x;
    delete ctx.oversampled_translations_y;
    delete ctx.oversampled_translations_z;

    delete ctx.thr_wsum_pdf_direction;
    delete ctx.thr_wsum_norm_correction;
    delete ctx.thr_sumw_group;
    delete ctx.thr_wsum_pdf_class;
    delete ctx.thr_wsum_prior_offsetx_class;
    delete ctx.thr_wsum_prior_offsety_class;
    delete ctx.thr_metadata;
    delete ctx.zeroArray;

    delete ctx.exp_wsum_norm_correction;
    delete ctx.exp_wsum_scale_correction_XA;
    delete ctx.exp_wsum_scale_correction_AA;
    delete ctx.thr_wsum_signal_product_spectra;
    delete ctx.thr_wsum_reference_power_spectra;
    delete ctx.thr_wsum_sigma2_noise;
    delete ctx.thr_wsum_ctf2;
    delete ctx.thr_wsum_stMulti;
#ifdef NEWMEM
	ctx.ptrFactory.freeTaskAllocator();
#endif
    CTOC(accMLO->timer,"accDoExpectationOneParticleCleanup");
// 	CTOC(accMLO->timer,"store_post_gpu");

}


template <class MlClass>
void forIbodyInit(Context<MlClass>& ctx) {
	ctx.ibody = 0;
}

template <class MlClass>
void forCoarseImgIdInit(Context<MlClass>& ctx) {
	ctx.img_id = 0;
}

template <class MlClass>
void forFineImgIdInit(Context<MlClass>& ctx) {
	ctx.img_id = 0;
}

template <class MlClass>
void forStoreWeightedSumsImgIdInit(Context<MlClass>& ctx) {
	ctx.img_id = 0;
}

template <class MlClass>
bool forIbodyCond(Context<MlClass>& ctx) {
	return ctx.ibody < ctx.baseMLO->mymodel.nr_bodies;
}

template <class MlClass>
bool forCoarseImgIdCond(Context<MlClass>& ctx) {
	return ctx.img_id < ctx.sp.nr_images;
}

template <class MlClass>
bool forFineImgIdCond(Context<MlClass>& ctx) {
	return ctx.img_id < ctx.sp.nr_images;
}

template <class MlClass>
bool forStoreWeightedSumsImgIdCond(Context<MlClass>& ctx) {
	return ctx.img_id < ctx.sp.nr_images;
}

template <class MlClass>
bool forIbodyCondNot(Context<MlClass>& ctx) {
	return !(ctx.ibody < ctx.baseMLO->mymodel.nr_bodies);
}

template <class MlClass>
bool forCoarseImgIdCondNot(Context<MlClass>& ctx) {
	return !(ctx.img_id < ctx.sp.nr_images);
}

template <class MlClass>
bool forFineImgIdCondNot(Context<MlClass>& ctx) {
	return !(ctx.img_id < ctx.sp.nr_images);
}

template <class MlClass>
bool forStoreWeightedSumsImgIdCondNot(Context<MlClass>& ctx) {
	return !(ctx.img_id < ctx.sp.nr_images);
}

template <class MlClass>
void forIbodyUpdate(Context<MlClass>& ctx) {
	ctx.ibody++;
}

template <class MlClass>
void forCoarseImgIdUpdate(Context<MlClass>& ctx) {
	ctx.img_id++;
}

template <class MlClass>
void forFineImgIdUpdate(Context<MlClass>& ctx) {
	ctx.img_id++;
}

template <class MlClass>
void forStoreWeightedSumsUpdate(Context<MlClass>& ctx) {
	ctx.img_id++;
}

template <class MlClass>
bool ifStoreWeightedSumsDoMaximizationCond(Context<MlClass>& ctx) {
	return !ctx.baseMLO->do_skip_maximization;
}

template <class MlClass>
bool ifStoreWeightedSumsDoMaximizationCondNot(Context<MlClass>& ctx) {
	return ctx.baseMLO->do_skip_maximization;
}


