import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np


def read_log_file(file_path):
    line_lists=[]
    now_lines=""
    flag=0
    with open(file_path, 'r') as f:
        for line in f.readlines():
            l=line.strip('\n')
            if " Expectation iteration " in l:
                if len(now_lines):
                    line_lists.append(now_lines)
                    now_lines=""
                flag=1
            if(flag==1):
                now_lines+=line
        if len(now_lines):
            line_lists.append(now_lines)
    return line_lists

def select_iteration_time(line_lists):
    # items:["iter","E","M","flatten_solvent","writeOutput"]
    data_sum=[]
    data=[]
    iter_num=len(line_lists)        
    for i in range(iter_num-1):
        string_s=line_lists[i]
        # result = re.findall(r'Expectation iteration (.*)',string_s)
        # data.append(result[0])
        for guize in [r'expectation                        : (.*) sec',
                    r'maximization                       : (.*) sec',
                    r"flatten solvent                    : (.*) sec",
                    r'iterate:  writeOutput              : (.*) sec',
                    ]:
            result = re.findall(guize,string_s)
            if len(result) ==0:
                data.append(0)
                continue
            elif len(result) !=1:
                print("ERROR",i,guize)
                break
            data.append(result[0])   
        if len(data):
            data_sum.append(data)
            data=[]
    return data_sum

def get_iteration_time(file_path):
    line_lists=read_log_file(file_path)
    iteration_data=select_iteration_time(line_lists)
    iteration_data=np.array(iteration_data)
    iteration_data = iteration_data.astype(float)
    row_sums = iteration_data.sum(axis=1)
    print(row_sums)
    return row_sums

def draw_iterations(name_time_map):
    custom_palette = [
        (70/255,  120/255, 142/255, 1),  # 2nd color
        (229/255, 139/255, 123/255, 1),  # 1st color
        (120/255, 183/255, 201/255, 1),  # 3rd color
        (246/255, 224/255, 147/255, 1),  # 4th color
    ]
    color=custom_palette
    
    dataset_name = "CNG"
    if "trpv1" in list(name_time_map.keys())[0]:
        dataset_name = "TRPV1"
    
    
    labels=[]
    all_time_data = []
    for name, time_data in name_time_map.items():
        base = os.path.basename(name)
        match = re.search(r'func_(.*?)_attempt', base)
        labels.append(match.group(1) if match else base)
        all_time_data.append(time_data)
    all_time_data = np.column_stack(all_time_data)

    
    # draw bar chart
    fig, ax = plt.subplots(figsize=(2.5, 4))
    bottom = np.zeros(len(labels))
    x = np.arange(len(labels))  # x轴位置
    bar_width = 0.35
    for i in range(all_time_data.shape[0]):
        ax.bar(x, all_time_data[i],width=bar_width,  bottom=bottom, color=color[i%4], edgecolor='black', linewidth=0.5)
        bottom += all_time_data[i]
    
    # add dot lines
    bottom = np.zeros(len(labels))
    for i in range(all_time_data.shape[0]):
        tops = bottom + all_time_data[i]
        for j in range(len(labels) - 1):
            x_start = x[j] + bar_width / 2      # 当前柱右边
            x_end = x[j+1] - bar_width / 2      # 下一柱左边
            ax.plot([x_start, x_end], [tops[j], tops[j+1]], linestyle='--', color='gray',linewidth=0.5, alpha=0.7)
        bottom = tops
    
    # calculate speedup
    speedup=np.ones(len(labels))
    # speedup[0]=1
    for j in range(len(labels)):
        speedup[j]=bottom[0]/bottom[j]
    for j in range(len(labels)):
        total = bottom[j]
        ax.text(x[j], total + 0.05, f'{speedup[j]:.2f}×', ha='center', va='bottom', alpha=1)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels,rotation=20, ha='right')
    ax.set_ylabel("time")

    plt.savefig(f"{dataset_name}_speedup.pdf", bbox_inches='tight')
    # plt.show()
    # plt.close(fig)
    return


def draw_scaling(name_time_map):
    colorset=['#6495ED','#ED6E52']
    dataset_name = "CNG"
    if "trpv1" in list(name_time_map.keys())[0]:
        dataset_name = "TRPV1"
    
    labels=[2,4,8]
    all_time_data = []
    data_draw=[]
    for name, time_data in name_time_map.items():
        base = os.path.basename(name)
        match = re.search(r'func_(.*?)_attempt', base)
        # labels.append(match.group(1) if match else base)
        time_sum= np.sum(time_data)
        all_time_data.append(time_sum)
        print(f"{base} time_sum: {time_sum}")
        data_draw.append(0)
    for i in range(len(all_time_data)):
        data_draw[i] =  all_time_data[0] / all_time_data[i]
    labels=labels[:len(data_draw)]
        
    fig,ax = plt.subplots(figsize = [7, 4])
    ax.plot(labels,data_draw,label=dataset_name,marker='.',markersize=10,color=colorset[0],linewidth=2)
    
    ymin=0
    ymax=max(max(data_draw),4)
    ax.set_xscale('log',base=2)
    ax.set_xticks(labels)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Speedup')
    ax.set_ylim(ymin,ymax)
    plt.savefig(f"{dataset_name}_scale.pdf", bbox_inches='tight')
    return 0


if __name__ == "__main__":
    name_time_map={}
    if len(sys.argv) < 3:
        print("Usage: python3 3_1_postprocess.py [postprocess_type:0(draw bar graph)|1(draw speedup graph)] <log_file1> <log_file2> ...")
        sys.exit(1)
    log_files = sys.argv[2:]
    postprocess_type = int(sys.argv[1])
    if postprocess_type not in [0, 1]:
        print("postprocess_type must be 0 or 1")
        sys.exit(1)
    
    for log_file in log_files:
        iteration_sum_time=get_iteration_time(log_file)
        name_time_map[log_file] = iteration_sum_time
    
    if postprocess_type == 0:
        draw_iterations(name_time_map)
    elif postprocess_type == 1:
        draw_scaling(name_time_map)
    
