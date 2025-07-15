#!/bin/bash

set -e
source ./0_env.sh
echo "Starting execution script..., log at $log_dir"
mkdir -p "$log_dir"
output_log_list=$(realpath "$output_log_list")
> "$output_log_list"

# get gpu number
gpu_total=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "available GPU num: $gpu_total"
cpu_total=$(lscpu | awk '/^Core\(s\) per socket:/ {cores=$4} /^Socket\(s\):/ {sockets=$2} END {print cores * sockets}')
CORES_PER_PROCESS=$(($cpu_total / $gpu_total))


# set path and variables
WORK_DIR_CNG=$(realpath "$WORK_DIR_CNG")
WORK_DIR_TRPV1=$(realpath "$WORK_DIR_TRPV1")
# === Executable paths ===
PATH_TO_RELION_BIN_ORI=$(realpath "$PATH_TO_RELION_BIN_ORI")
PATH_TO_RELION_BIN_MEM=$(realpath "$PATH_TO_RELION_BIN_MEM")
PATH_TO_RELION_BIN_TASK=$(realpath "$PATH_TO_RELION_BIN_TASK")
PATH_TO_RELION_BIN_KERNEL=$(realpath "$PATH_TO_RELION_BIN_KERNEL")
executables=("$PATH_TO_RELION_BIN_ORI" "$PATH_TO_RELION_BIN_MEM" "$PATH_TO_RELION_BIN_TASK" "$PATH_TO_RELION_BIN_KERNEL")
declare -A exe_pool_map
declare -A exe_output_prefix_map
exe_pool_map["$PATH_TO_RELION_BIN_ORI"]=4
exe_output_prefix_map["$PATH_TO_RELION_BIN_ORI"]="Refine3D-ori"
exe_pool_map["$PATH_TO_RELION_BIN_MEM"]=4
exe_output_prefix_map["$PATH_TO_RELION_BIN_MEM"]="Refine3D-opt"
exe_pool_map["$PATH_TO_RELION_BIN_TASK"]=50
exe_output_prefix_map["$PATH_TO_RELION_BIN_TASK"]="Refine3D-opt"
exe_pool_map["$PATH_TO_RELION_BIN_KERNEL"]=50
exe_output_prefix_map["$PATH_TO_RELION_BIN_KERNEL"]="Refine3D-opt"

# # xjltest
# PATH_TO_RELION_BIN_KERNEL=$(realpath "$PATH_TO_RELION_BIN_KERNEL")
# executables=("$PATH_TO_RELION_BIN_KERNEL") 
# declare -A exe_pool_map
# declare -A exe_output_prefix_map
# exe_pool_map["$PATH_TO_RELION_BIN_KERNEL"]=50
# exe_output_prefix_map["$PATH_TO_RELION_BIN_KERNEL"]="Refine3D1"

# == tool functions: ==

# generate GPU ID list for relion
gen_gpu_list() {
    local count=$1
    local ids=()
    for ((i = 0; i < count; i++)); do
        ids+=("$i")
    done
    IFS=":"; echo "${ids[*]}"
}

# find the maximum power of 2 that is less than or equal to n
max_pow2_gpu() {
    local n=$1
    local pow=2
    local best=2
    while (( pow <= n )); do
        best=$pow
        pow=$((pow * 2))
    done
    echo "$best"
}

# check if the job has converged on the specified iteration
check_convergence() {
    local logfile=$1
    local target_iter=$2
    local iter_line
    local conv_line

    iter_line=$(grep -n "Auto-refine: Iteration= $target_iter" "$logfile" | cut -d: -f1)
    conv_line=$(grep -n "Auto-refine: Refinement has converged, entering last" "$logfile" | cut -d: -f1)

    if [[ -n "$iter_line" && -n "$conv_line" ]]; then
        if (( conv_line > iter_line && conv_line - iter_line <= 10 )); then
            return 0
        fi
    fi
    return 1
}


run_cng_job() {
    local relion_bin=$1; 
    local np=$2; 
    local gpu_list=$3; 
    local log_file=$4
    local output_dir=$5
    local pool=$6
    local nthread=$CORES_PER_PROCESS;

    cmd="mpirun -np $np --bind-to none ./wrapper.sh $relion_bin \
        --o $output_dir/run --auto_refine --split_random_halves \
        --i particles.star --ref initial_model.mrc --ini_high 50 \
        --dont_combine_weights_via_disc --ctf --ctf_corrected_ref \
        --particle_diameter 160 --flatten_solvent --zero_mask \
        --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4 \
        --offset_range 5 --offset_step 2 --sym C4 --low_resol_join_halves 40 \
        --norm --scale --gpu $gpu_list --random_seed 1744931677 \
        --j $nthread --pool $pool | tee $log_file "
    echo "$cmd"
    eval "$cmd"
}

run_trpv1_job() {
    local relion_bin=$1; 
    local np=$2; 
    local gpu_list=$3; 
    local log_file=$4
    local output_dir=$5
    local pool=$6
    local nthread=$CORES_PER_PROCESS;

    cmd="mpirun -np $np --bind-to none ./wrapper.sh  $relion_bin \
     --o $output_dir/run --auto_refine --split_random_halves --firstiter_cc  \
     --i particles_new.star  --ref model.mrc --ini_high 40 \
     --dont_combine_weights_via_disc --ctf --ctf_corrected_ref \
     --particle_diameter 192 --flatten_solvent --zero_mask \
     --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4  \
     --offset_range 5  --offset_step 2 --sym C4 --low_resol_join_halves 40 \
     --norm --scale --gpu $gpu_list --j $nthread  \
     --pool $pool --pad 2  --skip_gridding \
     --solvent_mask mask.mrc --random_seed 0 --solvent_correct_fsc \
     --continue Refine3D/run_it001_optimiser.star | tee $log_file"
    echo "$cmd"
    eval "$cmd"
}

# === main loop for execution ===

echo "$(date): Running script ..."
for dataset in trpv1 cng; do
    echo "start set on dataset: $dataset"

    if [[ "$dataset" == "cng" ]]; then
        cd "$WORK_DIR_CNG" || { echo "cannot enter $WORK_DIR_CNG"; exit 1; }
        conv_iter=20
        run_job_func="run_cng_job"
    else
        cd "$WORK_DIR_TRPV1" || { echo "cannot enter $WORK_DIR_TRPV1"; exit 1; }
        conv_iter=21
        run_job_func="run_trpv1_job"
    fi

    # # test for different versions of T2-RELION
    max_gpu=$(max_pow2_gpu "$gpu_total")
    if (( max_gpu >= 2 )); then
        gpus=$max_gpu
        gpu_str=$(gen_gpu_list "$gpus")
        np=$((gpus + 1))

        for relion_bin in "${executables[@]}"; do
            # exe_id=$(basename "$relion_bin" | sed 's/relion_refine_mpi_//')
            # exe_id=$(basename "$(dirname "$relion_bin")")
            exe_id=$(basename "$(dirname "$(dirname "$relion_bin")")")

            name="${dataset}_func_${exe_id}"
            pool="${exe_pool_map[$relion_bin]}"
            out_prefix="${exe_output_prefix_map[$relion_bin]}"
            attempt=1

            while true; do
                logfile="./${name}_attempt_${attempt}.log"

                echo "optimization[$dataset][$exe_id] the $attempt attempt, with $gpus GPU"
                $run_job_func "$relion_bin" "$np" "$gpu_str" "$logfile" "$out_prefix" "$pool"
                if check_convergence "$logfile" "$conv_iter"; then
                    echo "optimization[$dataset][$exe_id] done with attempt $attempt"
                    realpath_logfile=$(realpath "$logfile")
                    echo "$realpath_logfile" >> "$output_log_list"
                    break
                fi
                attempt=$((attempt + 1))
            done
        done
    else
        echo "available GPU not enough for test, skip"
        exit 1
    fi

    # test for scalability
    if (( max_gpu < 8 )); then #xjldebug
    # if (( max_gpu < 4 )); then
        echo "available GPU not enough for scalability test, skip"
        continue
    fi
    for gpus in 2 4 8; do
    # for gpus in 2 4 ; do #xjldebug
        name="${dataset}_scale_${gpus}gpu"
        gpu_str=$(gen_gpu_list "$gpus")
        np=$((gpus + 1))
        pool=20
        attempt=1
        out_prefix="${exe_output_prefix_map[$PATH_TO_RELION_BIN_KERNEL]}"

        while true; do
            logfile="./${name}_attempt_${attempt}.log"
            echo "scalability test[$dataset][$gpus GPU] the $attempt attempt"

            $run_job_func "$PATH_TO_RELION_BIN_KERNEL" "$np" "$gpu_str" "$logfile" "$out_prefix" "$pool"
            if check_convergence "$logfile" "$conv_iter"; then
                echo "scaling test on [$dataset][$gpus GPU] done with attempt $attempt"
                realpath_logfile=$(realpath "$logfile")
                echo "$realpath_logfile" >> "$output_log_list"
                break
            fi
            attempt=$((attempt + 1))
        done
    done
done

echo "the log file list is at: $output_log_list"

echo "$(date): Running script end"