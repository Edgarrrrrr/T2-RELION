#ifndef ACC_BOO_H_
#define ACC_BOO_H_

#include "src/acc/acc_ml_optimiser.h"
// #include "src/ml_optimiser_mpi.h"
// #include "src/acc/acc_context.h"
#include "src/acc/acc_ptr.h"
#include <vector>
#include <algorithm>
#include <cstring>

void sort_indices(const std::vector<size_t> &v, std::vector<size_t> &idx)
{
    idx.resize(v.size());
  // initialize original index locations
  for (size_t i = 0; i < idx.size(); ++i)
    idx[i] = i;

  // sort indexes based on comparing values in v
  std::sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
}

template <size_t SIZE, size_t NR_OVER_TRANS, size_t NR_OVER_ORIENT>
struct Block
{
    size_t startRow;
    size_t startCol;
    int result_idx[SIZE * NR_OVER_ORIENT * SIZE * NR_OVER_TRANS];
    Block(size_t r, size_t c) : startRow(r), startCol(c) 
    {
        memset(result_idx, 0xff, sizeof(result_idx));
    }
};

struct BlockIndex
{
    size_t startRow;
    size_t startCol;
    size_t size;
    int index;
    BlockIndex(size_t r, size_t c, size_t s) : startRow(r), startCol(c), size(s), index(-1) {}
};

void splitBlock(const std::vector< std::vector<bool> > &coarse_map,
    const std::vector<size_t>& trans_rearranged, 
    const std::vector<size_t>& orient_rearranged,
    size_t start_row, size_t start_col, size_t block_size, 
    size_t total_rows, size_t total_cols,
    size_t &extend_rows, size_t &extend_cols,
    std::vector< std::vector<size_t> > &belong_to_which_block,
    std::vector<BlockIndex> &blocks_idx,
    double threshold, int &num4, int &num8, int &num16)
{
    int count = 0;
    size_t end_row = std::min(start_row + block_size, total_rows);
    size_t end_col = std::min(start_col + block_size, total_cols);
    for (unsigned long i = start_row; i < end_row; i++)
    {
        size_t coarse_trans_idx = trans_rearranged[i];
        for (unsigned long j = start_col; j < end_col; j++)
        {
            size_t coarse_orient_idx = orient_rearranged[j];
            if (coarse_map[coarse_trans_idx][coarse_orient_idx])
            {
                count++;
            }
        }
    }

    if (count == 0) return;

    if (block_size == 16 || block_size == 8)
    {
        if (count >= threshold * block_size * block_size)
        {
            blocks_idx.emplace_back(BlockIndex(start_row, start_col, block_size));
            if (block_size == 16)
                num16++;
            else
                num8++;
            size_t block_id = blocks_idx.size();
            for (unsigned long i = start_row; i < end_row; i++)
            {
                size_t coarse_trans_idx = trans_rearranged[i];
                for (unsigned long j = start_col; j < end_col; j++)
                {
                    size_t coarse_orient_idx = orient_rearranged[j];
                    belong_to_which_block[coarse_trans_idx][coarse_orient_idx] = block_id;
                }
            }
            extend_rows = std::max(start_row + block_size, extend_rows);
            extend_cols = std::max(start_col + block_size, extend_cols);
        }
        else
        {
            size_t new_block_size = block_size / 2;
            splitBlock(coarse_map, trans_rearranged, orient_rearranged, start_row, start_col, new_block_size, total_rows, total_cols, extend_rows, extend_cols, belong_to_which_block, blocks_idx, threshold, num4, num8, num16);
            if (start_col + new_block_size < total_cols)
                splitBlock(coarse_map, trans_rearranged, orient_rearranged, start_row, start_col + new_block_size, new_block_size, total_rows, total_cols, extend_rows, extend_cols, belong_to_which_block, blocks_idx, threshold, num4, num8, num16);
            if (start_row + new_block_size < total_rows)
                splitBlock(coarse_map, trans_rearranged, orient_rearranged, start_row + new_block_size, start_col, new_block_size, total_rows, total_cols, extend_rows, extend_cols, belong_to_which_block, blocks_idx, threshold, num4, num8, num16);
            if (start_row + new_block_size < total_rows && start_col + new_block_size < total_cols)
                splitBlock(coarse_map, trans_rearranged, orient_rearranged, start_row + new_block_size, start_col + new_block_size, new_block_size, total_rows, total_cols, extend_rows, extend_cols, belong_to_which_block, blocks_idx, threshold, num4, num8, num16);
        }
    }
    else if (block_size == 4)
    {
        blocks_idx.emplace_back(BlockIndex(start_row, start_col, block_size));
        num4++;
        size_t block_id = blocks_idx.size();
        for (unsigned long i = start_row; i < end_row; i++)
        {
            size_t coarse_trans_idx = trans_rearranged[i];
            for (unsigned long j = start_col; j < end_col; j++)
            {
                size_t coarse_orient_idx = orient_rearranged[j];
                belong_to_which_block[coarse_trans_idx][coarse_orient_idx] = block_id;
            }
        }
        extend_rows = std::max(start_row + block_size, extend_rows);
        extend_cols = std::max(start_col + block_size, extend_cols);
    }
}


template <size_t NR_OVER_TRANS, size_t NR_OVER_ORIENT>
long int cutSignificants2Blocks(
    OptimisationParamters &op,  SamplingParameters &sp,
    long int orientation_num, long int translation_num,
    ProjectionParams &FineProjectionData,
    int img_id,
    size_t &extend_rows, size_t &extend_cols,
    // std::vector <size_t> &trans_rearranged,
    // std::vector <size_t> &orient_rearranged,
#ifdef NEWMEM
    AccPtrNew< size_t > &TransRearrangedIndex,
    AccPtrNew< size_t > &OrientRearrangedIndex,
    AccPtrNew< size_t > &CoarseIndex2RotId,
    AccPtrNew< Block<16, NR_OVER_TRANS, NR_OVER_ORIENT> > &Blocks64x128,
    AccPtrNew< Block<8, NR_OVER_TRANS, NR_OVER_ORIENT> > &Blocks32x64,
    AccPtrNew< Block<4, NR_OVER_TRANS, NR_OVER_ORIENT> > &Blocks16x32,
    IndexedDataArrayNew &FPW,
#else
    AccPtr< size_t > &TransRearrangedIndex,
    AccPtr< size_t > &OrientRearrangedIndex,
    AccPtr< size_t > &CoarseIndex2RotId,
    AccPtr< Block<16, NR_OVER_TRANS, NR_OVER_ORIENT> > &Blocks64x128,
    AccPtr< Block<8, NR_OVER_TRANS, NR_OVER_ORIENT> > &Blocks32x64,
    AccPtr< Block<4, NR_OVER_TRANS, NR_OVER_ORIENT> > &Blocks16x32,
    IndexedDataArray &FPW,
#endif
    IndexedDataArrayMask &dataMask, int chunk)
{
    long int coarse_trans_num = translation_num / NR_OVER_TRANS;
    long int coarse_orient_num = orientation_num / NR_OVER_ORIENT;
    std::vector <size_t> trans_significant_num(coarse_trans_num, 0);
    std::vector <size_t> orient_significant_num(coarse_orient_num, 0);
    std::vector <std::vector <bool> > coarse_map(coarse_trans_num, std::vector<bool>(coarse_orient_num, false));
    for (long unsigned i = 0; i < coarse_trans_num; i++)
    {
        for (long unsigned j = 0; j < coarse_orient_num; j++)
        {
            long int coarse_idx = FineProjectionData.iorientclasses[j * NR_OVER_ORIENT] * sp.nr_trans + i; 
            if (DIRECT_A2D_ELEM(op.Mcoarse_significant, img_id, coarse_idx) == 1)
            {
                trans_significant_num[i]++;
                orient_significant_num[j]++;
                coarse_map[i][j] = true;
            }
        }
    }

    std::vector<size_t> trans_rearranged;
    std::vector<size_t> orient_rearranged;

    if (TransRearrangedIndex.getSize() == 0)
    {
        sort_indices(trans_significant_num, trans_rearranged);
        TransRearrangedIndex.setSize(coarse_trans_num);
        TransRearrangedIndex.hostAlloc();
        for (int i = 0; i < coarse_trans_num; i++)
            TransRearrangedIndex[i] = trans_rearranged[i];
    }
    sort_indices(orient_significant_num, orient_rearranged);
    OrientRearrangedIndex.setSize(coarse_orient_num);
    OrientRearrangedIndex.hostAlloc();
    for (int i = 0; i < coarse_orient_num; i++)
        OrientRearrangedIndex[i] = orient_rearranged[i];


    std::vector <size_t> inv_trans_rearranged(coarse_trans_num);
    std::vector <size_t> inv_orient_rearranged(coarse_orient_num);
    for (long unsigned i = 0; i < coarse_trans_num; i++)
    {
        inv_trans_rearranged[trans_rearranged[i]] = i; 
    }
    for (long unsigned i = 0; i < coarse_orient_num; i++)
    {
        inv_orient_rearranged[orient_rearranged[i]] = i; 
    }

    // // cut into block
    // const double threshold = 0.5;
    // std::vector< std::vector<size_t> > belong_to_which_block(coarse_trans_num, std::vector<size_t>(coarse_orient_num, 0));
    // std::vector<BlockIndex> blocks_idx;
    // int num4 = 0, num8 = 0, num16 = 0;
    // for (long unsigned i = 0; i < coarse_trans_num; i+=16)
    // {
    //     for (long unsigned j = 0; j < coarse_orient_num; j+=16)
    //     {
    //         splitBlock(coarse_map, trans_rearranged, orient_rearranged, i, j, 16, coarse_trans_num, coarse_orient_num, extend_rows, extend_cols, belong_to_which_block, blocks_idx, threshold, num4, num8, num16);
    //     }
    // }

    // Blocks64x128.setSize(num16); Blocks64x128.hostAlloc();
    // Blocks32x64.setSize(num8); Blocks32x64.hostAlloc();
    // Blocks16x32.setSize(num4); Blocks16x32.hostAlloc();

    // num4 = 0; num8 = 0; num16 = 0;
    // for (long unsigned b = 0; b < blocks_idx.size(); b++)
    // {
    //     size_t startRow = blocks_idx[b].startRow;
    //     size_t startCol = blocks_idx[b].startCol;
    //     size_t size = blocks_idx[b].size;

    //     if (size == 16)
    //     {
    //         blocks_idx[b].index = num16;
    //         Blocks64x128[num16++] = Block<16, NR_OVER_TRANS, NR_OVER_ORIENT>(startRow, startCol);
    //     }
    //     else if (size == 8)
    //     {
    //         blocks_idx[b].index = num8;
    //         Blocks32x64[num8++] = Block<8, NR_OVER_TRANS, NR_OVER_ORIENT>(startRow, startCol);
    //     }
    //     else if (size == 4)
    //     {
    //         blocks_idx[b].index = num4;
    //         Blocks16x32[num4++] = Block<4, NR_OVER_TRANS, NR_OVER_ORIENT>(startRow, startCol);
    //     }
    // }

    // only use 64x128 block
    std::vector< std::vector<size_t> > belong_to_which_block(coarse_trans_num, std::vector<size_t>(coarse_orient_num, 0));
    std::vector<BlockIndex> blocks_idx;
    int num = 0;
    for (long int i = 0; i < coarse_trans_num; i+=16)
    {
        for (long int j = 0; j < coarse_orient_num; j+=16)
        {
            size_t end_row = std::min(i + 16, coarse_trans_num);
            size_t end_col = std::min(j + 16, coarse_orient_num);
            bool is_significant = false;
            for (unsigned long ii = i; ii < end_row; ii++)
            {
                size_t coarse_trans_idx = trans_rearranged[ii];
                for (unsigned long jj = j; jj < end_col; jj++)
                {
                    size_t coarse_orient_idx = orient_rearranged[jj];
                    if (coarse_map[coarse_trans_idx][coarse_orient_idx])
                    {
                        is_significant = true;
                        break;
                    }
                }
                if (is_significant) break;
            }
            if (!is_significant) continue;

            blocks_idx.emplace_back(BlockIndex(i, j, 16));
            num++;
            size_t block_id = blocks_idx.size();

            for (long int r = i; r < end_row; r++)
            {
                size_t coarse_trans_idx = trans_rearranged[r];
                for (long int c = j; c < end_col; c++)
                {
                    size_t coarse_orient_idx = orient_rearranged[c];
                    belong_to_which_block[coarse_trans_idx][coarse_orient_idx] = block_id;
                }
            }
            extend_cols = (j + 16 > extend_cols) ? j + 16 : extend_cols;
        }
        extend_rows = (i + 16 > extend_rows) ? i + 16 : extend_rows;
    }
    
    Blocks64x128.setSize(num); Blocks64x128.hostAlloc();

    num = 0;
    for (long unsigned b = 0; b < blocks_idx.size(); b++)
    {
        size_t startRow = blocks_idx[b].startRow;
        size_t startCol = blocks_idx[b].startCol;

        blocks_idx[b].index = num;
        Blocks64x128[num++] = Block<16, NR_OVER_TRANS, NR_OVER_ORIENT>(startRow, startCol);
    }

    long unsigned w_base = dataMask.firstPos, w(0), k(0);
	dataMask.setNumberOfJobs(orientation_num * translation_num);
	dataMask.setNumberOfWeights(orientation_num * translation_num);
	dataMask.jobOrigin.hostAlloc();
	dataMask.jobExtent.hostAlloc();

    dataMask.jobOrigin[k] = 0;
	for (long unsigned i = 0; i < orientation_num; i++)
	{
		dataMask.jobExtent[k] = 0;
		long int tk = 0;
		long int iover_rot = FineProjectionData.iover_rots[i];
        long int coarse_orient_idx = i / NR_OVER_ORIENT;
        // #pragma unroll(4)
		for (long unsigned j = 0; j < translation_num; j+=4)
		{
			// long int iover_trans = j % NR_OVER_TRANS;
			long int ihidden = FineProjectionData.iorientclasses[i] * sp.nr_trans + (j / NR_OVER_TRANS);//coarse矩阵中该角度的index
            long int coarse_trans_idx = j / NR_OVER_TRANS;

			if(DIRECT_A2D_ELEM(op.Mcoarse_significant, img_id, ihidden)==1)
			{
                size_t block_id = belong_to_which_block[coarse_trans_idx][coarse_orient_idx] - 1;
                size_t block_size = blocks_idx[block_id].size;
                size_t block_start_row = blocks_idx[block_id].startRow;
                size_t block_start_col = blocks_idx[block_id].startCol;
                size_t row_in_block = (inv_trans_rearranged[coarse_trans_idx] - block_start_row) * NR_OVER_TRANS;
                size_t col_in_block = (inv_orient_rearranged[coarse_orient_idx] - block_start_col) * NR_OVER_ORIENT + i % NR_OVER_ORIENT;
                for (long unsigned jj = 0; jj < 4; jj++)
                {
                    FPW.rot_id[w_base + w] = FineProjectionData.iorientclasses[i] % (sp.nr_dir * sp.nr_psi); 	// where to look for priors etc
                    FPW.rot_idx[w_base + w] = i;					// which rot for this significant task
                    FPW.trans_idx[w_base + w] = j + jj;					// which trans       - || -
                    FPW.ihidden_overs[w_base + w]= (ihidden * NR_OVER_ORIENT + iover_rot) * NR_OVER_TRANS + jj;


                    size_t index_in_block = (row_in_block + jj) * block_size * NR_OVER_ORIENT + col_in_block;
                    Blocks64x128[blocks_idx[block_id].index].result_idx[index_in_block] = w_base + w;
                    // if (block_size == 16)
                    // {
                    //     Blocks64x128[blocks_idx[block_id].index].result_idx[index_in_block] = w_base + w;
                    // }
                    // else if (block_size == 8)
                    // {
                    //     Blocks32x64[blocks_idx[block_id].index].result_idx[index_in_block] = w_base + w;
                    // }
                    // else if (block_size == 4)
                    // {
                    //     Blocks16x32[blocks_idx[block_id].index].result_idx[index_in_block] = w_base + w;
                    // }
                    
                    if(tk >= chunk)
                    {
                        tk = 0;             // reset counter
                        k++;              // use new element
                        dataMask.jobOrigin[k] = w;
                        dataMask.jobExtent[k] = 0;   // prepare next element for ++ incrementing
                    }
                    tk++;                 		   // increment limit-checker
                    dataMask.jobExtent[k]++;       // increment number of transes this job
                    w++;
                }
			}
			else if(tk != 0) 		  // start a new one with the same rotidx - we expect transes to be sequential.
			{
				tk = 0;             // reset counter
				k++;              // use new element
				dataMask.jobOrigin[k] = w;
				dataMask.jobExtent[k] = 0;   // prepare next element for ++ incrementing
			}
		}

		if(tk>0) // use new element (if tk==0) then we are currently on an element with no signif, so we should continue using this element
		{
			k++;
			dataMask.jobOrigin[k]=w;
			dataMask.jobExtent[k]=0;
		}
	}
	if(dataMask.jobExtent[k] != 0) // if we started putting somehting in last element, then the count is one higher than the index
		k += 1;

	dataMask.setNumberOfJobs(k);
	dataMask.setNumberOfWeights(w);

    CoarseIndex2RotId.setSize(coarse_orient_num);
    CoarseIndex2RotId.hostAlloc();
    for (int i = 0; i < coarse_orient_num; i++)
    {
        CoarseIndex2RotId[i] = FineProjectionData.iorientclasses[orient_rearranged[i] * NR_OVER_ORIENT] % (sp.nr_dir * sp.nr_psi);
    }

    return w;
}

void rearrangeTranslation(
    // std::vector<size_t> &TransRearrangedIndex,
#ifdef NEWMEM
    const AccPtrNew<size_t> &TransRearrangedIndex,
    const AccPtrNew<XFLOAT> &old_trans_xyz, 
    AccPtrNew<XFLOAT> &rearranged_trans_xyz,
#else
    const AccPtr<size_t> &TransRearrangedIndex,
    const AccPtr<XFLOAT> &old_trans_xyz, 
    AccPtr<XFLOAT> &rearranged_trans_xyz, 
#endif   
    long int translation_num, long int nr_over_trans, long int extend_trans_num)
{
    rearranged_trans_xyz.setSize(3 * extend_trans_num * nr_over_trans);
    rearranged_trans_xyz.hostAlloc();
    for (int i = 0; i < rearranged_trans_xyz.getSize(); i++)
    {
        rearranged_trans_xyz[i] = 0.0;
    }

    long int coarse_trans_num = translation_num / nr_over_trans;
    size_t trans_x_offset = 0;
    size_t trans_y_offset = 1 * (size_t)translation_num;
    size_t trans_z_offset = 2 * (size_t)translation_num;

    for (long unsigned i = 0; i < coarse_trans_num; i++)
    {
        for (long unsigned j = 0; j < nr_over_trans; j++)
        {
            size_t old_idx = TransRearrangedIndex[i] * nr_over_trans + j;
            size_t new_idx = (i * nr_over_trans + j) * 3;
            rearranged_trans_xyz[new_idx + 0] = old_trans_xyz[old_idx + trans_x_offset];
            rearranged_trans_xyz[new_idx + 1] = old_trans_xyz[old_idx + trans_y_offset];
            rearranged_trans_xyz[new_idx + 2] = old_trans_xyz[old_idx + trans_z_offset];
        }
    }
}

void rearrangeOrientation(
    // std::vector<size_t> &OrientRearrangedIndex,
#ifdef NEWMEM
    const AccPtrNew<size_t> &OrientRearrangedIndex,
    const AccPtrNew<XFLOAT> &old_eulers, 
    AccPtrNew<XFLOAT> &rearranged_eulers,
#else
    const AccPtr<size_t> &OrientRearrangedIndex,
    const AccPtr<XFLOAT> &old_eulers, 
    AccPtr<XFLOAT> &rearranged_eulers,
#endif
    long int orientation_num, long int nr_over_orient, long int extend_orient_num)
{
    rearranged_eulers.setSize(9 * extend_orient_num * nr_over_orient);
    // rearranged_eulers.allAlloc();
    rearranged_eulers.hostAlloc();

    for (int i = 0; i < rearranged_eulers.getSize(); i++)
    {
        rearranged_eulers[i] = 0.0;
    }

    long int coarse_orient_num = orientation_num / nr_over_orient;
    for (long unsigned i = 0; i < coarse_orient_num; i++)
    {
        for (long unsigned j = 0; j < nr_over_orient; j++)
        {
            size_t old_idx = (OrientRearrangedIndex[i] * nr_over_orient + j) * 9;
            size_t new_idx = (i * nr_over_orient + j) * 9;
            for (long unsigned k = 0; k < 9; k++)
            {
                rearranged_eulers[new_idx + k] = old_eulers[old_idx + k];
            }
        }
    }
}

template <size_t NR_OVER_TRANS, size_t NR_OVER_ORIENT>
void computeKernelForMyMethod(
#ifdef NEWMEM
    AccPtrNew<size_t> &TransRearrangedIndex,
    AccPtrNew<size_t> &OrientRearrangedIndex,
    AccPtrNew< Block<16, NR_OVER_TRANS, NR_OVER_ORIENT> > &Blocks64x128,
    AccPtrNew< Block<8, NR_OVER_TRANS, NR_OVER_ORIENT> > &Blocks32x64,
    AccPtrNew< Block<4, NR_OVER_TRANS, NR_OVER_ORIENT> > &Blocks16x32,
    AccPtrNew<XFLOAT> &trans_xyz,
    AccPtrNew<XFLOAT> &eulers,
    IndexedDataArrayNew &FPW,
#else
    AccPtr<size_t> &TransRearrangedIndex,
    AccPtr<size_t> &OrientRearrangedIndex,
    AccPtr< Block<16, NR_OVER_TRANS, NR_OVER_ORIENT> > &Blocks64x128,
    AccPtr< Block<8, NR_OVER_TRANS, NR_OVER_ORIENT> > &Blocks32x64,
    AccPtr< Block<4, NR_OVER_TRANS, NR_OVER_ORIENT> > &Blocks16x32,
    AccPtr<XFLOAT> &trans_xyz,
    AccPtr<XFLOAT> &eulers,
    IndexedDataArray &FPW,
#endif    
    long unsigned significant_num
)
{
    long unsigned my_significant_num = 0;

    XFLOAT void_ratio = 0.0;
    for (long unsigned b = 0; b < Blocks64x128.getSize(); b++)
    {
        size_t startTrans = Blocks64x128[b].startRow * NR_OVER_TRANS * 3;
        size_t startOrient = Blocks64x128[b].startCol * NR_OVER_ORIENT * 9;
        size_t size = 16;
        size_t void_num = 0;
        
        for (long unsigned i = 0; i < size * NR_OVER_TRANS; i++)
        {
            for (long unsigned j = 0; j < size * NR_OVER_ORIENT; j++)
            {   
                if (Blocks64x128[b].result_idx[i * size * NR_OVER_ORIENT + j] == -1) continue;
                my_significant_num++;
                XFLOAT trans = (trans_xyz[startTrans + i * 3 + 0] + trans_xyz[startTrans + i * 3 + 1] + trans_xyz[startTrans + i * 3 + 2]) / 3;
                XFLOAT orient = (eulers[startOrient + j * 9 + 0] + eulers[startOrient + j * 9 + 1] + eulers[startOrient + j * 9 + 2]    \
                                + eulers[startOrient + j * 9 + 3] + eulers[startOrient + j * 9 + 4] + eulers[startOrient + j * 9 + 5]   \
                                + eulers[startOrient + j * 9 + 6] + eulers[startOrient + j * 9 + 7] + eulers[startOrient + j * 9 + 8]) / 9;
                XFLOAT res = trans * orient;
                XFLOAT answer = FPW.weights[Blocks64x128[b].result_idx[i * size * NR_OVER_ORIENT + j]];
                if (std::isnan(res) || std::isnan(answer))
                {
                    printf("res: %f, answer: %f, index: %ld\n", res, answer, Blocks64x128[b].result_idx[i * size * NR_OVER_ORIENT + j]);
                }
                XFLOAT diff = res - answer;
                if (diff/answer > 1e-6 || diff/answer < -1e-6)
                {
                    printf("res: %f, answer: %f, diff: %f, index: %ld\n", res, answer, diff, Blocks64x128[b].result_idx[i * size * NR_OVER_ORIENT + j]);
                }
                size_t trans_idx_answer = FPW.trans_idx[Blocks64x128[b].result_idx[i * size * NR_OVER_ORIENT + j]];
                size_t orient_idx_answer = FPW.rot_idx[Blocks64x128[b].result_idx[i * size * NR_OVER_ORIENT + j]];
                size_t coarse_trans_idx = TransRearrangedIndex[Blocks64x128[b].startRow + i / NR_OVER_TRANS];
                size_t trans_idx_res = coarse_trans_idx * NR_OVER_TRANS + i % NR_OVER_TRANS;
                size_t coarse_orient_idx = OrientRearrangedIndex[Blocks64x128[b].startCol + j / NR_OVER_ORIENT];
                size_t orient_idx_res = coarse_orient_idx * NR_OVER_ORIENT + j % NR_OVER_ORIENT;
                if (trans_idx_answer != trans_idx_res)
                {
                    printf("trans_idx_res: %f, answer: %f, index:%ld\n", trans_idx_answer, trans_idx_res, Blocks64x128[b].result_idx[i * size * NR_OVER_ORIENT + j]);
                }
                if (orient_idx_answer != orient_idx_res)
                {
                    printf("orient_idx_res: %f, answer: %f, index:%ld\n", orient_idx_answer, orient_idx_res, Blocks64x128[b].result_idx[i * size * NR_OVER_ORIENT + j]);
                }

            }
        }
        void_ratio += (XFLOAT)void_num / (16.0 * 32.0);
    }
    void_ratio /= Blocks16x32.getSize();
    // if(print_void_ratio)
    // printf("void ratio for 16x32 block: %f\n", void_ratio);

    if (my_significant_num != significant_num)
    {
        printf("my_significant_num: %ld, significant_num: %ld\n", my_significant_num, significant_num);
    }


    // for (long unsigned b = 0; b < Blocks32x64.getSize(); b++)
    // {
    //     size_t startTrans = Blocks32x64[b].startRow * NR_OVER_TRANS * 3;
    //     size_t startOrient = Blocks32x64[b].startCol * NR_OVER_ORIENT * 9;
    //     size_t size = 8;
        
    //     for (long unsigned i = 0; i < size * NR_OVER_TRANS; i++)
    //     {
    //         for (long unsigned j = 0; j < size * NR_OVER_ORIENT; j++)
    //         {   
    //             if (Blocks32x64[b].result_idx[i * size * NR_OVER_ORIENT + j] == -1) continue;
    //             my_significant_num++;
    //             XFLOAT trans = (trans_xyz[startTrans + i * 3 + 0] + trans_xyz[startTrans + i * 3 + 1] + trans_xyz[startTrans + i * 3 + 2]) / 3;
    //             XFLOAT orient = (eulers[startOrient + j * 9 + 0] + eulers[startOrient + j * 9 + 1] + eulers[startOrient + j * 9 + 2]    \
    //                             + eulers[startOrient + j * 9 + 3] + eulers[startOrient + j * 9 + 4] + eulers[startOrient + j * 9 + 5]   \
    //                             + eulers[startOrient + j * 9 + 6] + eulers[startOrient + j * 9 + 7] + eulers[startOrient + j * 9 + 8]) / 9;
    //             XFLOAT res = trans * orient;
    //             XFLOAT answer = FPW.weights[Blocks32x64[b].result_idx[i * size * NR_OVER_ORIENT + j]];
    //             XFLOAT diff = res - answer;
    //             if (diff/answer > 1e-6 || diff/answer < -1e-6)
    //             {
    //                 printf("res: %f, answer: %f, diff: %f, index: %ld\n", res, answer, diff, Blocks32x64[b].result_idx[i * size * NR_OVER_ORIENT + j]);
    //             }
    //         }
    //     }
    // }

    // for (long unsigned b = 0; b < Blocks16x32.getSize(); b++)
    // {
    //     size_t startTrans = Blocks16x32[b].startRow * NR_OVER_TRANS * 3;
    //     size_t startOrient = Blocks16x32[b].startCol * NR_OVER_ORIENT * 9;
    //     size_t size = 4;
        
    //     for (long unsigned i = 0; i < size * NR_OVER_TRANS; i++)
    //     {
    //         for (long unsigned j = 0; j < size * NR_OVER_ORIENT; j++)
    //         {   
    //             if (Blocks16x32[b].result_idx[i * size * NR_OVER_ORIENT + j] == -1) continue;
    //             my_significant_num++;
    //             XFLOAT trans = (trans_xyz[startTrans + i * 3 + 0] + trans_xyz[startTrans + i * 3 + 1] + trans_xyz[startTrans + i * 3 + 2]) / 3;
    //             XFLOAT orient = (eulers[startOrient + j * 9 + 0] + eulers[startOrient + j * 9 + 1] + eulers[startOrient + j * 9 + 2]    \
    //                             + eulers[startOrient + j * 9 + 3] + eulers[startOrient + j * 9 + 4] + eulers[startOrient + j * 9 + 5]   \
    //                             + eulers[startOrient + j * 9 + 6] + eulers[startOrient + j * 9 + 7] + eulers[startOrient + j * 9 + 8]) / 9;
    //             XFLOAT res = trans * orient;
    //             XFLOAT answer = FPW.weights[Blocks16x32[b].result_idx[i * size * NR_OVER_ORIENT + j]];
    //             XFLOAT diff = res - answer;
    //             if (diff/answer > 1e-6 || diff/answer < -1e-6)
    //             {
    //                 printf("res: %f, answer: %f, diff: %f, index: %ld\n", res, answer, diff, Blocks16x32[b].result_idx[i * size * NR_OVER_ORIENT + j]);
    //             }
    //         }
    //     }
    // }

    
}

void computeKernelForOriginalMethod(
#ifdef NEWMEM
    AccPtrNew<XFLOAT> &trans_xyz,
    AccPtrNew<XFLOAT> &eulers,
    AccPtrNew<size_t> &orient_idx,
    AccPtrNew<size_t> &trans_idx,
    AccPtrNew<XFLOAT> &weights,
#else
    AccPtr<XFLOAT> &trans_xyz,
    AccPtr<XFLOAT> &eulers,
    AccPtr<size_t> &orient_idx,
    AccPtr<size_t> &trans_idx,
    AccPtr<XFLOAT> &weights,
#endif    
    size_t trans_x_offset, size_t trans_y_offset, size_t trans_z_offset,
    long unsigned significant_num
)
{
    for (long unsigned i = 0; i < significant_num; i++)
    {
        XFLOAT trans = (trans_xyz[trans_idx[i] + trans_x_offset] + trans_xyz[trans_idx[i] + trans_y_offset] + trans_xyz[trans_idx[i] + trans_z_offset]) / 3;
        XFLOAT orient = (eulers[orient_idx[i] * 9 + 0] + eulers[orient_idx[i] * 9 + 1] + eulers[orient_idx[i] * 9 + 2]    \
                        + eulers[orient_idx[i] * 9 + 3] + eulers[orient_idx[i] * 9 + 4] + eulers[orient_idx[i] * 9 + 5]   \
                        + eulers[orient_idx[i] * 9 + 6] + eulers[orient_idx[i] * 9 + 7] + eulers[orient_idx[i] * 9 + 8]) / 9;
        weights[i] = trans * orient;
        // printf("%f ", weights[i]);
    }
    // printf("\n");
}



#endif  /* ACC_BOO_H_ */

