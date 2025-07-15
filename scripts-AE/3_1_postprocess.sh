#!/bin/bash

set -e
source ./0_env.sh

# output_log_list="./logs/final_logs_list_old2.txt"

if [[ ! -f "$output_log_list" ]]; then
    echo "Cannot find $output_log_list"
    exit 1
fi

echo "Start processing $output_log_list"

scale_logs_cng=()
scale_logs_trpv1=()
func_logs_cng=()
func_logs_trpv1=()

while IFS= read -r logfile; do
    if [[ "$logfile" == *"scale"* ]]; then
        if [[ "$logfile" == *"cng"* ]]; then
            scale_logs_cng+=("$logfile")
        elif [[ "$logfile" == *"trpv1"* ]]; then
            scale_logs_trpv1+=("$logfile")
        fi
    elif [[ "$logfile" == *"func"* ]]; then
        if [[ "$logfile" == *"cng"* ]]; then
            func_logs_cng+=("$logfile")
        elif [[ "$logfile" == *"trpv1"* ]]; then
            func_logs_trpv1+=("$logfile")
        fi
    else
        echo "Unknown log type: $logfile"
    fi
done < "$output_log_list"

# draw figure 9:
if (( ${#func_logs_cng[@]} > 0 )); then
    python3 3_1_postprocess.py  0 "${func_logs_cng[@]}"
fi
if (( ${#func_logs_trpv1[@]} > 0 )); then
    python3 3_1_postprocess.py 0 "${func_logs_trpv1[@]}"
fi

# draw scalibility figure :
if (( ${#scale_logs_cng[@]} > 0 )); then
    python3 3_1_postprocess.py  1 "${scale_logs_cng[@]}"
fi
if (( ${#scale_logs_trpv1[@]} > 0 )); then
    python3 3_1_postprocess.py 1 "${scale_logs_trpv1[@]}"
fi


