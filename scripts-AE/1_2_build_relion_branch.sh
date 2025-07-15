#!/usr/bin/env bash

source ./0_env.sh

# Exit on error, undefined variable, or pipe failure
set -euo pipefail                    


if cap_line=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n1); then
  compute_cap=${cap_line//./}
  echo "Detected compute capability: $cap_line → CUDA_ARCH=$compute_cap"
else
  echo "Warning: nvidia-smi cannot get compute_cap, using default CUDA_ARCH=80"
  compute_cap=80
fi

mkdir -p "$work_root" "$build_root"

echo "Preparing to process branches: ${branches[*]}"
echo "  Worktrees Root: $work_root"
echo "  Builds Root: $build_root"

for br in "${branches[@]}"; do
  echo "=============== [$br] ==============="

  wt_dir="${work_root}/${br}"
  build_dir="${build_root}/build_${br}"
  mkdir -p "$build_dir" 
  echo "  Worktree Directory: $wt_dir"
  echo "  Build Directory: $build_dir"

  # 1) Create or update git worktree
  if [[ ! -d "$wt_dir" ]]; then
`2`    git worktree add --force "$wt_dir" "$br"
  fi

  # 2) Configure & build (out-of-source)
  # TODO: xjl Change the CMake command!
  echo "Configuring build for branch $br..."

  cmake -S "$wt_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=profiling \
        -DGUI=OFF -DCUDA=ON -DFETCH_TORCH_MODELS=OFF -DFFTW_DISABLE_DOCS=ON \
        -DCUDA_ARCH="$compute_cap" \
        -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG -D_CUDA_HALF -fPIC -DTIMING" \
        -DCMAKE_C_FLAGS="-O3 -DNDEBUG -fPIC -DTIMING" \
        -DADDITIONAL_LINKER_FLAGS="-lz -lm"

  # TODO: xjl Change the make command if needed!
  echo "Building branch $br..."

  cmake --build "$build_dir" --parallel $(nproc)

done

# 3) Clean up stale worktree metadata

# Run this after manually deleting worktree directories
git worktree prune -v

echo "🎉 All branches processed, executable files are in $build_dir"
