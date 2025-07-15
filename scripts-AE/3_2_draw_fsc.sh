#!/bin/bash

set -e
source ./0_env.sh

work_dir_CNG=$(realpath "$WORK_DIR_CNG")
work_dir_TRPV1=$(realpath "$WORK_DIR_TRPV1")

echo "The results will be saved in $work_dir_CNG/Refine3D-ori/, $work_dir_CNG/Refine3D-opt/ and $work_dir_TRPV1/Refine3D-ori/, $work_dir_TRPV1/Refine3D-opt/, respectively. You can use the ChimeraX to open the run_class001.mrc files and visualize the RELION results in these directories."

# calculate FSC curves for CNG and TRPV1 datasets
# This script assumes that the e2proc3d.py tool is available in the PATH.
if ! command -v e2proc3d.py &> /dev/null; then
    echo "e2proc3d.py could
     not be found. Please ensure eman2 is installed and in your PATH."
    exit 1
fi

cng_ori_fsc_file="./fsc_curve_cng_ori.txt"
cng_opt_fsc_file="./fsc_curve_cng_opt.txt"
trpv_ori_fsc_file="./fsc_curve_trpv_ori.txt"
trpv_opt_fsc_file="./fsc_curve_trpv_opt.txt"

e2proc3d.py --calcfsc "$WORK_DIR_CNG/Refine3D-ori/run_half1_class001_unfil.mrc" "$WORK_DIR_CNG/Refine3D-ori/run_half2_class001_unfil.mrc" $cng_ori_fsc_file

e2proc3d.py --calcfsc "$WORK_DIR_CNG/Refine3D-opt/run_half1_class001_unfil.mrc" "$WORK_DIR_CNG/Refine3D-opt/run_half2_class001_unfil.mrc" $cng_opt_fsc_file


e2proc3d.py --calcfsc "$WORK_DIR_TRPV1/Refine3D-ori/run_half1_class001_unfil.mrc" "$WORK_DIR_TRPV1/Refine3D-ori/run_half2_class001_unfil.mrc" $trpv_ori_fsc_file

e2proc3d.py --calcfsc "$WORK_DIR_TRPV1/Refine3D-opt/run_half1_class001_unfil.mrc" "$WORK_DIR_TRPV1/Refine3D-opt/run_half2_class001_unfil.mrc" $trpv_opt_fsc_file

python 3_2_draw_fsc.py "$cng_ori_fsc_file" "$cng_opt_fsc_file" "CNG"
python 3_2_draw_fsc.py "$trpv_ori_fsc_file" "$trpv_opt_fsc_file" "TRPV1"