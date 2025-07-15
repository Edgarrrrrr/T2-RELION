#!/bin/bash
# This script sets up the environment for running RELION with different versions
# and prepares the necessary data directories.

# step 0: source environment variables
source ./0_env.sh


# step 1: download data
python ./1_1_zenodo_fetch_and_extract.py --record_id 15300133 --target_dir "$DATA_DIR_CNG" #./CNG
python ./1_1_zenodo_fetch_and_extract.py --record_id 15300098 --target_dir "$DATA_DIR_TRPV1" #./trpv1



# step 2: build RELION with different version
bash ./1_2_build_relion_branch.sh


# setup 3: set up working directory
if [ -d "$WORK_DIR_CNG" ]; then
    echo "Removing existing directory: $WORK_DIR_CNG"
    rm -rf "$WORK_DIR_CNG"
fi
if [ -d "$WORK_DIR_TRPV1" ]; then
    echo "Removing existing directory: $WORK_DIR_TRPV1"
    rm -rf "$WORK_DIR_TRPV1"
fi
echo "Creating working directories: $WORK_DIR_CNG and $WORK_DIR_TRPV1"
mkdir -p "$WORK_DIR_CNG" "$WORK_DIR_TRPV1"
chmod a+x ./wrapper.sh


# step 4: copy files and setup links
cp "$DATA_DIR_CNG/Refine3D/initial_model.mrc" "$WORK_DIR_CNG/initial_model.mrc"
cp "$DATA_DIR_CNG/Refine3D/particles.star" "$WORK_DIR_CNG/particles.star"
cp ./wrapper.sh "$WORK_DIR_CNG/wrapper.sh"
mkdir -p "$WORK_DIR_CNG/Refine3D-ori"
mkdir -p "$WORK_DIR_CNG/Refine3D-opt"
Particles_DIR="$DATA_DIR_CNG/CNG/Particles"
if [ ! -d "$Particles_DIR" ]; then
    echo "There is no Particles directory in $DATA_DIR_CNG/CNG. Please check the download data."
    exit 1
fi
Particles_DIR=$(realpath "$Particles_DIR")
ln -s "$Particles_DIR" "$WORK_DIR_CNG/Particles"

cp -r "$DATA_DIR_TRPV1/Refine3D/" "$WORK_DIR_TRPV1/Refine3D/"
cp "$DATA_DIR_TRPV1/Refine3D/model.mrc" "$WORK_DIR_TRPV1/model.mrc"
cp "$DATA_DIR_TRPV1/Refine3D/particles_new.star" "$WORK_DIR_TRPV1/particles_new.star"
cp "$DATA_DIR_TRPV1/Refine3D/mask.mrc" "$WORK_DIR_TRPV1/mask.mrc"
cp ./wrapper.sh "$WORK_DIR_TRPV1/wrapper.sh"
mkdir -p "$WORK_DIR_TRPV1/Refine3D-ori"
mkdir -p "$WORK_DIR_TRPV1/Refine3D-opt"
Particles_DIR="$DATA_DIR_TRPV1/trpv1/Particles"
if [ ! -d "$Particles_DIR" ]; then
    echo "There is no Particles directory in $DATA_DIR_TRPV1/trpv1. Please check the download data."
    exit 1
fi
Particles_DIR=$(realpath "$Particles_DIR")
ln -s "$Particles_DIR" "$WORK_DIR_TRPV1/Particles"

