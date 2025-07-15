
WORK_DIR_CNG="./work_dir_CNG"
WORK_DIR_TRPV1="./work_dir_TRPV1"
DATA_DIR_CNG="./CNG"
DATA_DIR_TRPV1="./TRPV1"

# List of branches to process
branches=(opt/origin_version opt/vram-management opt/task_parallelism opt/pinned_memory)
# Directories for git worktrees, builds, and results
repo_root="$(git rev-parse --show-toplevel)"
# Parent directory for all git worktrees
work_root="${repo_root}/../worktrees"
build_root="${repo_root}/../build"


PATH_TO_RELION_BIN_ORI="$build_root/build_"${branches[0]}/bin/relion_refine_mpi
PATH_TO_RELION_BIN_MEM="$build_root/build_"${branches[1]}/bin/relion_refine_mpi
PATH_TO_RELION_BIN_TASK="$build_root/build_"${branches[2]}/bin/relion_refine_mpi
PATH_TO_RELION_BIN_KERNEL="$build_root/build_"${branches[3]}/bin/relion_refine_mpi



log_dir="./logs"
output_log_list="./logs/final_logs_list.txt"



# #xjltest
# PATH_TO_RELION_BIN_KERNEL="/home/xujingle/relion/build-gpu2025/bin/relion_refine_mpi_sc_mem_task_kernel_pin"
# WORK_DIR_CNG="/home/xujingle/relion/run/250320paper"
# WORK_DIR_TRPV1="/home/xujingle/relion/run-trpv1/202502"
# SUBDIR_kernel="./Refine3D1"