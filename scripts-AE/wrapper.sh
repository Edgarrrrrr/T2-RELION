#!/bin/bash


# NCPUS=$(nproc --all)  
NCPUS=$(lscpu | awk '/^Core\(s\) per socket:/ {cores=$4} /^Socket\(s\):/ {sockets=$2} END {print cores * sockets}')
# NGPUS=$CUDA_VISIBLE_DEVICES 
NGPUS=$(nvidia-smi -L | wc -l) 
# NGPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l) # 获取GPU数
echo "NCPUS: $NCPUS"
echo "NGPUS: $NGPUS"

LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK # for OpenMPI
# LOCAL_RANK=$MPI_LOCALRANKID # for Intel MPI
# LOCAL_RANK=$SLURM_LOCALID # for SLURM
LOCAL_SIZE=$OMPI_COMM_WORLD_LOCAL_SIZE # for OpenMPI
# LOCAL_SIZE=$MPI_LOCALNRANKS # for Intel MPI
# LOCAL_SIZE=$SLURM_TASKS_PER_NODE # for SLURM

echo "LOCAL_RANK: $LOCAL_RANK"
NUM_NUMA=$(numactl --hardware | grep available | awk '{print $2}')  # 获取 NUMA 节点数
echo "NUM_NUMA: $NUM_NUMA"

# 计算每个进程绑定的核心数（总CPU数 / GPU数 - 1）
CORES_PER_PROCESS=$(($NCPUS / $NGPUS))
echo "CORES_PER_PROCESS: $CORES_PER_PROCESS"

if [ $LOCAL_RANK -eq 0 ]; then
  CORES="0"  # 0号进程绑定到0号核
  echo "Process $LOCAL_RANK on $(hostname) bound to core $CORES"
else
  # 计算该进程对应的CPU核区间
  CORE_START=$((($LOCAL_RANK - 1) * $CORES_PER_PROCESS))  # 核心的开始位置
  CORE_END=$(($CORE_START + $CORES_PER_PROCESS - 1))  # 核心的结束位置
  
  # 获取与该GPU相关的核心
  CORES=$(seq -s, $CORE_START 1 $CORE_END)
  
  echo "Process $LOCAL_RANK on $(hostname) bound to core $CORES"
fi

# 执行命令，指定NUMA绑定的核心
exec numactl -C "$CORES" $@