import re

# 替换为你的log文件路径

for index in range(1, 38):
    log_file = '../carry/run_log_repeat/run_log_task_pinned_sync_para2_full_{}.txt'.format(index)
# 用于保存结果 {iteration_num: time_in_seconds}
    expectation_times = {}

    with open(log_file, 'r') as f:
        lines = f.readlines()

    current_iter = None

    sum_time = 0.0

    for line in lines:
        # 匹配 iteration 行
        iter_match = re.search(r'Expectation iteration (\d+)', line)
        if iter_match:
            current_iter = int(iter_match.group(1))
            continue

        # 匹配 expectation 时间行
        time_match = re.search(r'expectation\s+: ([\d.]+) sec', line)
        if time_match and current_iter is not None:
            time_sec = float(time_match.group(1))
            expectation_times[current_iter] = time_sec
            current_iter = None  # 清除，避免误关联下一个时间

    print(index)
    # 输出结果
    for iter_num in sorted(expectation_times):
        sum_time += expectation_times[iter_num]
        print(f"Iteration {iter_num}: {expectation_times[iter_num]:.3f} sec")

    print(f"Total time: {sum_time:.3f} sec")
    
    import time
    time.sleep(2)  # 可选，避免输出过快