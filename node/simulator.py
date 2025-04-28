
import argparse
import pandas as pd
import copy
import re
from collections import defaultdict, deque
import csv

QoS_map = {
    'SSD_MobileNet': 202,
    'MnasNet': 62,
    'GooLeNet': 66,
    'DenseNet': 202,
    'LeNet5': 5,
    'resnet50': 108,
    'resnet101': 108,
    'resnet152': 108,
    'vgg16':  142,
    'vgg19': 142,
    'mobilenet_v2': 64,
    'unet': 120,
    'bert': 400,
    'deeplabv3': 300,
    'alexnet': 80,
}
MAP_list = []

path = '/data/zbw/inference_system/MIG_MPS/log/history/our.csv'

with open(path, mode='r') as file:
    reader = csv.reader(file)
    
    headers = next(reader)
    
    history_data = []
    
    for row in reader:
        history_data.append(row)
        
file.close()


log_path = '/data/zbw/inference_system/MIG_MPS/log/utilization'

best_fit_partition_map = {
    'resnet50': '3g.40gb',
    'vgg19':  '3g.40gb',
    'mobilenet_v2': '2g.20gb',
    'GooLeNet': '2g.20gb',
    'bert': '3g.40gb',
    'MnasNet': '2g.20gb',
    'DenseNet': '3g.40gb',
    'SSD_MobileNet': '3g.40gb',
    'LeNet5': '2g.20gb',
}

model_MIG_RPS = {}
utilization_map = {}

config_map = {'1c-1g-10gb': '1g.10gb', '1c-2g-20gb': '2g.20gb', '1c-3g-40gb': '3g.40gb', '1c-4g-40gb': '4g.40gb', '1c-7g-80gb': '7g.80gb'}
SM_partition_size = {'1g.10gb': 1, '2g.20gb':2, '3g.40gb':3, '4g.40gb':4, '7g.80gb': 7}
Memory_partition_size = {'1g.10gb':1, '2g.20gb': 2, '3g.40gb': 4, '4g.40gb': 4 , '7g.80gb': 8}

def read_data_from_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r"Config: (\S+), P99: ([\d.]+), RPS: (\d+)", line.strip())
            if match:
                config = match.group(1)
                p99 = float(match.group(2))
                rps = int(match.group(3))
                data.append((config, p99, rps))
    return data

def genetate_MIG_RPS(tasks):
    dir_path = '/data/zbw/inference_system/MIG_MPS/log/'
    for i in tasks:
        QoS = QoS_map.get(i)/2
        file_path = dir_path+ f'{i}_MIG_RPS'
        data = read_data_from_file(file_path)
        max_rps_for_config = defaultdict(lambda: float('-inf'))

        for config, p99, rps in data:
            if p99 < QoS:
                if rps > max_rps_for_config[config]:
                    max_rps_for_config[config] = rps
        model_MIG_RPS[i] = {}
        for config, max_rps in max_rps_for_config.items():
            translate_config = config_map[config]
            model_MIG_RPS[i][translate_config] = max_rps


def select_partition(p1,p2):
    size_p1 = SM_partition_size.get(p1)
    size_p2 = SM_partition_size.get(p2)

    size_final = min(size_p1, size_p2)

    if size_final == 1:
        return '1g.10gb'
    if size_final == 2:
        return  '2g.20gb'
    if size_final == 3:
        return '3g.40gb'
    if size_final == 4:
        return '4g.40gb'
    if size_final == 7:
        return '7g.80gb'

def handle_data(data):
    new_data = {}
    for i in data.keys():
        SM_value = float(SM_partition_size.get(i))/7
        Memory_value = float(Memory_partition_size.get(i))/8


        SM_utilization = data[i][3]/SM_value
        Memory_utilizaiton = data[i][5]/Memory_value

        new_data[i]  = [SM_utilization, Memory_utilizaiton]
    return new_data
def read_Beta(model1, model2):
    return 0.1

def read_Alpha(model1, model2):
    return 0.8

def generate_utilization_map(tasks):
    utilization_map['memory'] = {}
    utilization_map['SM'] = {}

    for i in tasks:
        file_path = log_path  + f'/{i}'
        data = {}
        current_config = None

        utilization_map['memory'][i] = {}
        utilization_map['SM'][i] = {}

        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                
                if line.endswith('gb'):
                    current_config = line
                elif line.startswith("GPU"):
                    values = line.split() 
                    data_values = list(map(float, values[1:]))  
                    data[current_config] = data_values
        data = handle_data(data)
        
        for config in data.keys():
            utilization_map['memory'][i][config] = data[config][0]
            utilization_map['SM'][i][config]  = data[config][1]


def best_fit_partition(model, RPS):
    partition =  best_fit_partition_map.get(model)

    if  RPS >= int(model_MIG_RPS[model][partition]):
        return partition
    else:
        for partitions in SM_partition_size.keys():
            if int(model_MIG_RPS[model][partitions]) >= RPS:
                return partitions


def read_history(model1, model2, config, demand1, demand2):

    result = None
    
    for i in history_data:
        model1_history = i[0]
        model2_history = i[1]

        config_history = i[2]

        demand1_history = int(i[3])
        demand2_history = int(i[4])

        result1 = int(i[5])
        result2 = int(i[6])

        if model1_history == model1 and model2_history == model2 and config_history==config and demand1_history == demand1 and demand2_history == demand2:
            result = []
            result.append(result1)
            result.append(result2)
        
        if model1_history == model2 and model2_history == model1 and config_history==config and demand1_history == demand2 and demand2_history == demand1:
            result = []
            result.append(result2)
            result.append(result1) 

    if not result:
        print(f"database no data with {model1} {model2} {demand1} {demand2} {config} the max RPS is {model_MIG_RPS[model1][config]} {(model_MIG_RPS[model2][config])}")
        # raise Exception(f"database no data with {model1} {model2} {demand1} {demand2} {config} the max RPS is {model_MIG_RPS[model1][config]} {(model_MIG_RPS[model2][config])}")
        return None
    else:
        return result

def dispatch(tasks, demands, Alpha=0, Beta=1):
    SM_sensitivity_list = []
    Memory_sensitivity_list  = []
    partitions = []

    for i in tasks:
        index = tasks.index(i)
        config = best_fit_partition(i, demands[index])
        SM_sensitivity_list.append(utilization_map['SM'][i][config]/utilization_map['memory'][i][config])
        Memory_sensitivity_list.append(utilization_map['memory'][i][config]/utilization_map['SM'][i][config])

    SM_models = tasks.copy()
    Memory_models = tasks.copy()

    SM_combined = list(zip(SM_sensitivity_list, SM_models))
    Memory_combined = list(zip(Memory_sensitivity_list, Memory_models))

    sorted_combined = sorted(SM_combined, key=lambda x: x[0], reverse=True)
    SM_models, SM_sensitivity_list = zip(*sorted_combined)


    sorted_combined = sorted(Memory_combined, key=lambda x: x[0], reverse=True)
    Memory_models, Memory_sensitivity_list = zip(*sorted_combined)

    i =0 
    j =0 
    satisfied = 0

    while len(Memory_sensitivity_list) >= satisfied :
        model1 = SM_sensitivity_list[i]
        model2 = Memory_sensitivity_list[j]

        demands1 = demands[tasks.index(model1)]
        demands2 = demands[tasks.index(model2)]

        if demands1 <= 0 :
            i = i + 1
            satisfied =  satisfied + 1
            if satisfied >= len(Memory_sensitivity_list):
                break
            j = 0
            continue

        if demands2 <= 0 :
            j = j + 1 
            continue

        partition1 = best_fit_partition(model1, demands1)
        partition2 = best_fit_partition(model2, demands2)

        max_partition = select_partition(partition1, partition2)

        max_util_model1 = max(utilization_map['memory'][model1][max_partition], utilization_map['SM'][model1][max_partition])
        max_util_model2 = max(utilization_map['SM'][model2][max_partition], utilization_map['SM'][model2][max_partition])


        Beta = read_Beta(model1, model2)
        Alpha = read_Alpha(model1, model2)

        if max_util_model1 >= Beta and max_util_model2 >= Beta:
            result = read_history(model1, model2, max_partition, min(demands1, model_MIG_RPS[model1][max_partition]), min(demands2, model_MIG_RPS[model2][max_partition]))
            partitions.append([model1, model2, demands1, demands2, max_partition])
            if result:
                demands[tasks.index(model1)] = demands[tasks.index(model1)] - result[0]
                demands[tasks.index(model2)] = demands[tasks.index(model2)] - result[1]
            else:
                demands[tasks.index(model1)] = demands[tasks.index(model1)] - min(int(Alpha * model_MIG_RPS[model1][max_partition]), demands[tasks.index(model1)]) 
                demands[tasks.index(model2)] = demands[tasks.index(model2)] - min(int(Alpha * model_MIG_RPS[model2][max_partition]), demands[tasks.index(model2)]) 

        elif j == len(SM_sensitivity_list) - 1:
            result = read_history(model1, model2, max_partition, min(demands1, model_MIG_RPS[model1][max_partition]), min(demands2, model_MIG_RPS[model2][max_partition]))
            partitions.append([model1, model2, demands1, demands2, max_partition])
            if result:
                demands[tasks.index(model1)] = demands[tasks.index(model1)] - result[0]
                demands[tasks.index(model2)] = demands[tasks.index(model2)] - result[1]
            else:
                demands[tasks.index(model1)] = demands[tasks.index(model1)] - min(int(Alpha * model_MIG_RPS[model1][max_partition]), demands[tasks.index(model1)]) 
                demands[tasks.index(model2)] = demands[tasks.index(model2)] - min(int(Alpha * model_MIG_RPS[model2][max_partition]), demands[tasks.index(model2)]) 
        else:
            j = j + 1

    return partitions

def is_sublist(sublist, lst):
    for i in range(len(lst) - len(sublist) + 1):
        if lst[i:i+len(sublist)] == sublist:
            return True
    return False

def binary_search_max_true(tasks, demands, num_GPU, left, right):
    global MAP_list
    while left < right:
    
        mid = int((left + right + 1) // 2)
        new_demands = []

        for i in demands:
            new_demands.append(mid * i)

        partitions = dispatch(tasks, new_demands)
        
        

        for i in range(0, len(partitions)):
            partitions[i][4] = SM_partition_size.get(partitions[i][4])

        sorted_partitions = sorted(partitions, key=lambda x: x[4], reverse=True)
        init_MAP_queue(num_GPU)
        
        if valid_MIG_allocation(num_GPU, sorted_partitions):
            left = mid  
        else:
            right = mid - 1  
    return left  



def search(tasks, demands, num_GPU, RPS):
    new_demands = []
    for i in demands:
        new_demands.append(RPS * i)

    partitions = dispatch(tasks, new_demands)
 
    for i in range(0, len(partitions)):
        partitions[i] = SM_partition_size.get(partitions[i])

    partitions_cp = copy.deepcopy(partitions)
    MIG_config_valid_example = [
    [7], [4, 3], [4, 2, 1], [4, 1, 1, 1], [3, 3], [3, 2, 1], [3, 1, 1, 1], [3,2,2],
    [3,2,1,1], [3,1,1,1], [2, 2, 2, 1], [2,2,1,1,1],
    [2, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]]

    GPU_list = [[] for _ in range(num_GPU)]
    remain_P = []
    for i in partitions_cp:
        
        for j in GPU_list:
            tmp_j = j.copy()
            tmp_j.append(i)
           
            find = False
            for configuration in MIG_config_valid_example:
                if is_sublist(tmp_j, configuration): 
                    j.append(i)
                    find = True
                    break
            if find:
                break
        
        if not find:
                remain_P.append(i)
        remain_resource = []

    for i in GPU_list:
        for configuration in MIG_config_valid_example:

            if is_sublist(i, configuration): 
                modified_configuration = []
                modified_configuration = [item for item in configuration if item not in i]

                for item in modified_configuration:
                    remain_resource.append(item)
                break

    for i in remain_P:
        remain_throught = i
        throught = 0
        used = []
        find = False

        for j in range(0, len(remain_resource)):

            if throught + j < remain_throught:
                throught = throught + j
                used.append(j)
            else:
                find = True
                used.append(j)
                for index in sorted(used, reverse=True):
                    del remain_resource[index]
                break

        if not find:
            return False


def init_MAP_queue(num_GPU):
    global MAP_list
    MAP_list = []
    for i in range(0, 5):
        MAP_list.append(deque())

    for i in range(0, num_GPU):
        MAP_list[0].append(i)

def calculate_MAP(gpu):
    MIG_config_valid_example = [
    [7], [4, 3], [4, 2, 1], [4, 1, 1, 1], [3, 3], [3, 2, 1], [3, 1, 1, 1], [3,2,2],
    [3,2,1,1], [3,1,1,1], [2, 2, 2, 1], [2,2,1,1,1],
    [2, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]
    ]

    possible = [7,4,3,2,1]
    for i in possible:
        tmp = copy.deepcopy(gpu)
        tmp.append(i)
        sorted_gpu = sorted(tmp, reverse=True)

        for j in MIG_config_valid_example:
            if is_sublist(sorted_gpu, j):
                return i
    return False

def divide_partitions(patition):
    if patition[4] == 7:
        left_patition = [patition[0], patition[1], patition[2], patition[3], 4]
        right_patition = [patition[0], patition[1], patition[2], patition[3], 3]

    if patition[4] == 4:
        left_patition = [patition[0], patition[1], patition[2], patition[3], 2] 
        right_patition = [patition[0], patition[1], patition[2], patition[3], 2]

    if patition[4] == 3:
        left_patition = [patition[0], patition[1], patition[2], patition[3], 2]
        right_patition = [patition[0], patition[1], patition[2], patition[3], 1]

    if patition[4] == 2:
        left_patition = [patition[0], patition[1], patition[2], patition[3], 1]

        right_patition = [patition[0], patition[1], patition[2], patition[3], 1]
    if patition[4] == 1:
        return False

    return left_patition, right_patition    
    


def valid_MIG_allocation(num_GPU, search_space):
    global MAP_list
    index_map = {7:0, 4:1, 3:2, 2:3, 1:4}
    GPU_list = []
    for i in range(0, num_GPU):
        GPU_list.append([])

    index = 0
    while index < len(search_space):
        flag = False
        for i in range(index_map[search_space[index][4]], -1, -1):
            if len(MAP_list[i]) != 0:
                flag = True
                gpu_index = MAP_list[i].popleft()
                GPU_list[gpu_index].append(search_space[index][4])
                MAP = calculate_MAP(GPU_list[gpu_index])
                if MAP:
                    MAP_list[index_map[MAP]].append(gpu_index)
                break

        if not flag:
            if divide_partitions(search_space[index]):
                left, right = divide_partitions(search_space[index])
                search_space.append(left)
                search_space.append(right)
            else:
                return False
        
        index = index + 1
    return True

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--GPU", type=int)
    args = parser.parse_args()

    
    task = args.task
    GPU = args.GPU
    task = [s.strip() for s in task.split(',')]


    tasks = []
    demands = []

    for i in range(0, len(task),2):
        tasks.append(task[i])

    for i in range(1, len(task), 2):
        demands.append(int(task[i]))


    generate_utilization_map(tasks)

    genetate_MIG_RPS(tasks)

    search_GPU_num = [10]
    
    #Need to much data , cancel!
    for i in range(0, len(search_GPU_num)):
        valid = binary_search_max_true(tasks, demands, num_GPU=search_GPU_num[i], left=0, right=10000)

        print(f"Simulation on {i} GPUs and the result {valid} for scen1")


    