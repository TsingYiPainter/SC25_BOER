import numpy as np
from skopt import gp_minimize
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt import SequentialDomainReductionTransformer
from bayes_opt.util import load_logs
import argparse
import logging
import re
import math
import subprocess
import time
import pickle
from collections import defaultdict
import json
import os
import sys
from scipy.stats import linregress
import pandas as pd

paddingFeedback_dir = '/data/zbw/MIG/MIG/SC25_BOER/script/padding_feedback.sh'
MPS_PID = 354993
logdir = '/data/wyh/MIG_MPS/tmp/bayesian_tmp.txt'
dynamic_path = '/data/wyh/MIG_MPS/tmp/dynamic/dynamic.txt'

sumOfTime2 = 0


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

max_RPS_map = {'resnet50': 1500, 'resnet152': 1100, 'vgg16':1300, 'bert': 200, 'mobilenet_v2': 3200, 'vgg19': 650, 'DenseNet':1500, 'SSD_MobileNet':1000}
min_RPS_map = {'resnet50': 0, 'resnet152': 0, 'vgg16': 0, 'bert': 0, "mobilenet_v2": 10, 'vgg19':0, 'DenseNet':0, 'SSD_MobileNet':0}
SM_map = {'resnet50': (10, 90), 'resnet152': (10, 90), 'vgg16': (10, 90), 'bert': (10, 90), "mobilenet_v2": (10,80), 'vgg19': (10, 90), 'DenseNet': (10, 90), 'SSD_MobileNet': (30,80)}



optimizer =  None
task = None
request = []
test = False
MIGidx = dict()

# The UUID of the current MIG partition needs to be matched with its partition size
# For example, the UUID of the partition currently in use are mig1,mig2, and their sizes are 3g,40gb,4g,40gb. Then you need to set:
# MIGidx["mig1"]=2
# MIGidx["mig2"]=3

MIGTOSM = {0: 100/7, 1: 200/7, 2: 300/7, 3: 400/7, 4:700/7}

class Pruning:
    static_constraint = dict()
    dynamic_constraint = []
    MIG2SM = 0
    @staticmethod
    def set_static_constraint(slope,intercept):
        Pruning.static_constraint["slope"] = slope
        Pruning.static_constraint["intercept"] = intercept

    @staticmethod
    def get_static_constraint(SM,RPS):
        score = (SM * (Pruning.MIG2SM/100) * Pruning.static_constraint["slope"] + Pruning.static_constraint["intercept"])/max(RPS,1)/2
        ## !!!
        return [RPS < SM * (Pruning.MIG2SM/100) * Pruning.static_constraint["slope"] + Pruning.static_constraint["intercept"],score]

    @staticmethod
    def set_dynamic_constraint(SM,RPS,score):
        Pruning.dynamic_constraint.append([SM,RPS,score])
    
    @staticmethod
    def get_dynamic_constraint(SM,RPS):
        for SM_RPS_pair in Pruning.dynamic_constraint:
            if SM<=SM_RPS_pair[0] and RPS>=SM_RPS_pair[1]:
                ## !!!
                return [False,SM_RPS_pair[2]* (SM_RPS_pair[1]/RPS*SM/SM_RPS_pair[0])**(1/2)]
        return [True,None]

class SurfaceHis:
    dict_list = []
    changeFlag = False
    changeParams = None
    changeRes = 0

    @staticmethod
    def changeSuface(bo):
        new_dict_list = []
        dict_list_copy = SurfaceHis.dict_list.copy()
        for dic in dict_list_copy:
            dic_copy = dic
            SurfaceHis.changeParams = [dic["params"]["RPS0"],dic["params"]["RPS1"],dic["params"]["lc0"]]
            bo.probe(
                params={"SM0": dic["params"]["SM0"], "RPS0":dic["params"]["RPS0"]},
                lazy=True,
            )
            print(SurfaceHis.changeParams)
            print()
            bo.maximize(init_points=0, n_iter=0)
            dic_copy["target"]=SurfaceHis.changeRes

            new_dict_list.append(dic_copy)
        SurfaceHis.dict_list = new_dict_list

def get_maxRPSInMIG(modelName,device):
    file_path = '/data/zbw/inference_system/MIG_MPS/log/'
    with open(file_path + modelName+"_MIG_RPS", 'r') as file:
        lines = file.readlines()
    global MIGidx

    parsed_data = []
    for line in lines:
        parts = line.split(", ")
        config = parts[0].split(": ")[1]
        p99 = float(parts[1].split(": ")[1])
        rps = int(parts[2].split(": ")[1])
        parsed_data.append((config, p99, rps))


    df = pd.DataFrame(parsed_data, columns=["Config", "P99", "RPS"])

    max_rps_per_config = df.groupby("Config")["RPS"].max().reset_index()

    rps_list = max_rps_per_config['RPS'].tolist()
    rps = rps_list[MIGidx[device]]
    return rps


def read_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r'task: (\w+), SM: (\d+), batch: (\d+), 99th percentile: ([\d.]+)', line)
            if match:
                task = match.group(1)
                sm = int(match.group(2))
                batch = int(match.group(3))
                percentile = float(match.group(4))
                data.append({"task": task, "SM": sm, "batch": batch, "percentile": percentile})
    return data

def get_configuration_result(configuration_list, serve):
    if serve_num == 1:
        QoS = QoS_map.get(task[0])
        half_QoS = [QoS/2,QoS/2]
    else:
        QoS1 = QoS_map.get(task[0])
        QoS2 = QoS_map.get(task[1])
        half_QoS = [QoS1/2,QoS2/2]

    batch1 = math.floor(float(configuration_list[0]['RPS'])/1000 * half_QoS[0])
    batch2 = math.floor(float(configuration_list[1]['RPS'])/1000 * half_QoS[1])

    file_path = '/data/zbw/inference_system/MIG_MPS/log/'+serve+'_Pairs_MPS_RPS'
    data_list = read_data(file_path)
    for i in range(0, len(data_list)-1, 2):  
        if i + 1 < len(data_list):

            item1 = data_list[i]
            item2 = data_list[i + 1]

            if int(item1['SM']) == int(configuration_list[0]['SM']) and int(item2['SM']) == int(configuration_list[1]['SM']) \
            and int(batch1) == int(item1['batch']) and int(batch2) == int(item2['batch']):
                latency1 = item1['percentile']
                latency2 = item2['percentile']
                return latency1, latency2
            elif int(item2['SM']) == int(configuration_list[0]['SM']) and int(item1['SM']) == int(configuration_list[1]['SM']) \
            and int(batch2) == int(item1['batch']) and int(batch1) == int(item2['batch']):
                latency1 = item2['percentile']
                latency2 = item1['percentile']
                return latency1, latency2


def read_RPS(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r'Config: (\w+), P99: ([\d.]+), RPS: (\d+)', line)
            if match:
                config = int(match.group(1))
                percentile = float(match.group(2))
                RPS = int(match.group(3))
                data.append({"config": config, "RPS": RPS, "percentile": percentile})
    return data


def get_maxRPSInCurSM(serve, sm, halfQoS):

    file_path = '/data/zbw/inference_system/MIG_MPS/log/'+serve+'_MPS_RPS'
    data_list = read_RPS(file_path)
    filtered_data = [item for item in data_list if item['config'] == sm]

    sorted_items = sorted(filtered_data, key=lambda x: x['percentile'])

    max_item = None
    for item in sorted_items:
        if item['percentile'] <= halfQoS:
            max_item = item
        else:
            break

    maxRPS = max_item['RPS']
    return maxRPS


def map_to_range(x, min_val, max_val):
    return 0.5 + ((x - min_val) / (max_val - min_val)) * (1 - 0.5)


def objective_feedback(configuration_list):

    num = len(configuration_list)

    result = 0
    task1 = task[0]
    task2 = task[2]
    
    SM = configuration_list[0]['SM']
    RPS = configuration_list[0]['RPS']
    remain_SM = 100  - SM

    stepDir = dict()
    tmpParams = dict()
    tmpParams["RPS0"] = RPS
    tmpParams["RPS1"] = 0
    tmpParams["SM0"] = SM
    tmpParams["SM1"] = remain_SM
    tmpParams["lc0"] = 1000
    stepDir["params"] = tmpParams

    # !!!
    if not Pruning.get_static_constraint(SM,RPS)[0]:
        stepDir["target"] = Pruning.get_static_constraint(SM,RPS)[1]
        SurfaceHis.dict_list.append(stepDir)
        return stepDir["target"]
    if not Pruning.get_dynamic_constraint(SM,RPS)[0]:
        stepDir["target"] = Pruning.get_dynamic_constraint(SM,RPS)[1]
        SurfaceHis.dict_list.append(stepDir)
        return stepDir["target"]



    half_QoS = QoS_map[task1]/2
    half_QoS2 = QoS_map[task2]/2

    search_SM = (int(remain_SM/10) + 1) * 10
    max_RPS = get_maxRPSInCurSM(task2, search_SM, half_QoS2)

    batch = math.floor(float(RPS)/1000 * half_QoS)

    if SurfaceHis.changeFlag:
        # SurfaceHis format:[dic["params"]["RPS0"],dic["params"]["RPS1"],dic["params"]["lc0"]]
        # SurfaceHis.changeParams[2] is lc0
        if SurfaceHis.changeParams[2]==0:
                SurfaceHis.changeRes = 3
                return 3
        else:
            SurfaceHis.changeRes = 0.5 * min(1, half_QoS/ SurfaceHis.changeParams[2])
            return 0.5 * min(1, half_QoS/ SurfaceHis.changeParams[2])

    server_id = MPS_PID

    script_path = paddingFeedback_dir
    
    BO_args= [task1, task2, SM, remain_SM, batch, max_RPS, server_id, args.device, args.port]
    BO_args = [str(item) for item in BO_args]

    ##time1 end 

    global sumOfTime2
    time2_start = time.time()
    process = subprocess.Popen([script_path] + BO_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()

    process.wait()
    time2_end = time.time()
    sumOfTime2 += (time2_end-time2_start)

    #time3_start = time.time()
    ###!!!
    file_path1 = logdir
    #file_path2 = logdir.replace('.txt', f'_True.txt')
    
    latency = None
    valid_RPS = None

    with open(file_path1, 'r') as file:
        line = file.readline().strip()
        match = re.search(r"model:\s*(\S+)\s+latency:\s*([\d.]+)", line)

        if match:
            model = match.group(1)
            latency = float(match.group(2))
        
        elif line.startswith('valid_RPS:'):
            value = float(line.split(':')[1].strip()) 
            valid_RPS = int(value)

        else:
            print("no result!")

    with open(file_path1, 'w') as file:
        file.write('')

    # with open(file_path2, 'w') as file:
    #     file.write('')

    stepDir = dict()
    tmpParams = dict()
    tmpParams["RPS0"] = RPS
    tmpParams["RPS1"] = valid_RPS if not latency else 0
    tmpParams["SM0"] = SM
    tmpParams["SM1"] = remain_SM
    tmpParams["lc0"] = latency if latency else 0
    stepDir["params"] = tmpParams

    if latency:

        result = 0.5 * min(1, half_QoS/ latency)
        stepDir["target"] = result
        SurfaceHis.dict_list.append(stepDir)

        Pruning.set_dynamic_constraint(SM,RPS,result)
        return result
        
    elif valid_RPS:
        if valid_RPS!=1:
            with open(dynamic_path, 'a') as file: 
                file.write(f"{RPS}\n") 
                file.write('end step\n')
        # half_QoS2 = QoS_map[task2]/2
        # RPS100 = get_maxRPSInCurSM(task2, 100, half_QoS2)
        # result = 0.5 + 0.5/ 2 * (valid_RPS + RPS) / RPS100

        weight0 = 1
        weight1 = 1
        N = 2

        if not test:
            mRPS1 = get_maxRPSInMIG(task1,args.device)
            mRPS2 = get_maxRPSInMIG(task2,args.device)
            result = 0.5 + (1/(2*N))* ( RPS/min(request[0], mRPS1) + valid_RPS/min(request[1], mRPS2))
            #mapped_value = map_to_range(result, 0, num)
            mapped_value = result
            #print(f"RPS IS {valid_RPS + RPS} and result is {mapped_value}")
            stepDir["target"] = mapped_value
            SurfaceHis.dict_list.append(stepDir)
            #time3_end = time.time()

            return mapped_value
        
        else:   
            half_QoS2 = QoS_map[task2]/2
            # RPS100_1 = get_maxRPSInCurSM(task1, 100 ,half_QoS)
            RPS100_2 = get_maxRPSInCurSM(task2, 100, half_QoS2)
            RPS100_1 = get_maxRPSInCurSM(task1, 100, half_QoS)

            RPS100_baseline = min(RPS100_1, RPS100_2)
            relationship = request[1]/request[0]
            unite_RPS = min(RPS * relationship, valid_RPS)

        
            result = 0.5 + 0.5 * (unite_RPS/RPS100_baseline)

            #print(f"RPS IS {RPS} and {valid_RPS} , {unite_RPS} and result is {result}")
            stepDir["target"] = result
            SurfaceHis.dict_list.append(stepDir)
            ##time3 end
            return result


def get_task_num(task):
    return 1

def wrapped_objective_feedback(**kwargs):
    ##time1 start
    configuration_list = []

    for i in range(len(kwargs) // 2):
        sm_key = f'SM{i}'
        rps_key = f'RPS{i}'

        if sm_key in kwargs and rps_key in kwargs:
            sm_value = int(kwargs[sm_key])
            rps_value = int(kwargs[rps_key]) 
            configuration_list.append({'SM': sm_value, 'RPS': rps_value})

    return objective_feedback(configuration_list)


def init_optimizer_feedback(server_num, config):

    search_list = {}
    for i in range(0, server_num):
        if i != serve_num - 1: 

            search_list[f'SM{i}'] = SM_map.get(config[i*2])

            max_RPS = max_RPS_map.get(config[i*2])
            min_RPS = min_RPS_map.get(config[i*2])
            
            if int(config[i*2+1]) < max_RPS_map.get(config[i*2]):
                max_RPS = int(config[i*2+1])
            
            if int(config[i*2+1]) < min_RPS_map.get(config[i*2]):
                min_RPS = 0

            search_list[f'RPS{i}'] = (min_RPS, max_RPS)
            print(search_list)
        else:
            continue


    bounds_transformer = SequentialDomainReductionTransformer(minimum_window=10)
    optimizer =BayesianOptimization(
        f=wrapped_objective_feedback,
        pbounds=search_list,
        verbose = 2,
        random_state = args.seed,
        #allow_duplicate_points=True
        #bounds_transformer=bounds_transformer
    )
    
    return optimizer


def pruningByHistory(historyFile):
    ConfigList_RPS = []
    ConfigList_SM = []
    for fileName in historyFile:
        with open(fileName, 'r') as file:
            data = json.load(file)
            max_target_entry = max(data, key=lambda x: x['target'])
            print(max_target_entry)
            RPS_t =  max_target_entry ['params']['RPS0']
            SM_t =  max_target_entry ['params']['SM0']
            ConfigList_RPS.append(RPS_t)
            ConfigList_SM.append(SM_t)
    min_RPS = min(ConfigList_RPS)
    max_RPS = max(ConfigList_RPS)
    min_SM = min(ConfigList_SM)
    max_SM = max(ConfigList_SM)
    print("bound")
    print([min_SM,max_SM,min_RPS,max_RPS])
    return [min_SM,max_SM,min_RPS,max_RPS]

def changeFileFormat(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    formatted_data = '[\n' + ',\n'.join(line.strip() for line in lines) + '\n]'

    with open(filepath, 'w') as json_file:
        json_file.write(formatted_data)




def start_mps_daemon(gpu_id):

    pipe_dir = f"/tmp/nvidia-mps-{gpu_id}"
    log_dir = f"/tmp/nvidia-log-{gpu_id}"

    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = pipe_dir
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = log_dir
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    process = subprocess.Popen(
        ["nvidia-cuda-mps-control", "-d"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    print(f"MPS Daemon started for GPU {gpu_id}. Waiting for initialization...")
    time.sleep(1)  

    ps_output = subprocess.run(["ps", "-ef"], stdout=subprocess.PIPE, text=True).stdout
    for line in ps_output.splitlines():
        if "nvidia-cuda-mps-control" in line:
            parts = line.split()
            pid = parts[1]
            try:
                with open(f"/proc/{pid}/environ", "r") as env_file:
                    environ = env_file.read()
                    if pipe_dir in environ:
                        print(f"Found MPS Daemon PID for {pipe_dir}: {pid}")
                        return pid
            except PermissionError:
                print(f"Permission denied when accessing /proc/{pid}/environ. Skipping...")
                continue
    print("MPS Daemon not found.")
    return None


def run_simple_cuda_program(cuda_script_path, pipe_dir):

    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = pipe_dir

    try:
        process = subprocess.run(
            ["python", cuda_script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        print("CUDA Program Output:")
        print(process.stdout)
        if process.stderr:
            print("CUDA Program Errors:")
            print(process.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error while running CUDA program: {e.stderr}")


def get_mps_server_pid(mps_pid):

    ps_output = subprocess.run(["ps", "-ef"], stdout=subprocess.PIPE, text=True).stdout
    for line in ps_output.splitlines():
        if "nvidia-cuda-mps-server" in line and f" {mps_pid} " in line:
            parts = line.split()
            server_pid = parts[1]
            print(f"Found MPS Server PID for MPS Daemon {mps_pid}: {server_pid}")
            return server_pid

    print(f"Error: Could not find MPS Server PID for MPS Daemon {mps_pid}.")
    return None



def staticPruning():
    task0 = task[0]
    # !!!
    Pruning.MIG2SM=MIGTOSM[MIGidx[args.device]]
    print(Pruning.MIG2SM)

    if Pruning.MIG2SM % 10 < 5:
        SM_idx = int((Pruning.MIG2SM// 10))
    else:
        SM_idx = int((Pruning.MIG2SM// 10 + 1))

    print(SM_idx)

    with open("/data/wyh/MIG_MPS/log/SM2RPS.json", "r") as file:
        data = json.load(file)


    SM2RPS = data[task0][:SM_idx]
    print(SM2RPS)
    SM2RPS = [0]+SM2RPS
    print(SM2RPS)
    x_axis = [i*10 for i in range(len(SM2RPS))]
    print(x_axis)

    x = np.array(x_axis,dtype=float)
    y = np.array(SM2RPS,dtype=float)

    s, i, r_value, p_value, std_err = linregress(x, y)
    print("res is y={}x+{}".format(s,i))
    Pruning.set_static_constraint(s,i)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_num", type=int)
    parser.add_argument("--task", type=str)
    parser.add_argument("--feedback", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--device",type = str)
    parser.add_argument("--port",type = int)
    parser.add_argument("--idxGI",type = int, default=0)
    parser.add_argument("--numOfGI",type =int,default=1)
    parser.add_argument("--seed",type = int, default=4)
    parser.add_argument("--demand", type =str)
    args = parser.parse_args()


    test = args.test
    serve_num = args.server_num
    task = args.task
    feedback = args.feedback
    logdir = logdir.replace('.txt', f'_{args.device}.txt')
    idxGI = int(args.idxGI)
    numOfGI = int(args.numOfGI)
    task = [s.strip() for s in task.split(',')]
    dynamic_path = dynamic_path.replace('.txt', f'_{task[2]}_{args.device}.txt')

    mps_pid = start_mps_daemon(args.device)
    server_id = None
    run_simple_cuda_program("./start_mps.py","/tmp/nvidia-mps-"+args.device)
    if mps_pid:
        server_id = get_mps_server_pid(mps_pid)
        MPS_PID = server_id

    for i in range(0, serve_num):
        request.append(int(task[i*2 + 1]))


    if not feedback:
        pass

    else:
        start = time.time()

        for idx in range(0,1):
            staticPruning()
            optimizer = init_optimizer_feedback(serve_num, task)
            utility = UtilityFunction(kind="ei", kappa=5, xi=0.2)

            time_path = '/data/wyh/MIG_MPS/tmp/time/time.txt'
            time_path = time_path.replace('.txt', f'_{args.device}.txt')
            with open(time_path, 'w') as file:
                file.write("")
            stepLogDir = "../tmp/"+args.task+"-"+args.device+"_seed"+str(args.seed)+"_"+"dynamic"+".json"

            print("stepLogDir:"+stepLogDir )

            tmpMax = 0
            count = 0
            optimizer.maximize(
                init_points=6,  
                n_iter=0,      
                acquisition_function=utility  
            )
            tmpMax = optimizer.max['target']

            for stepIndx in range(6,20):
                optimizer.maximize(
                    init_points=0,
                    n_iter=1
                )
                if optimizer.res[-1]['target']<=tmpMax:
                    count += 1
                else:
                    tmpMax = optimizer.res[-1]['target']
                    count = 0

                if count >= 5:
                    break

            end = time.time()
            print(end - start)


            with open(stepLogDir, "w") as json_file:
                json.dump(SurfaceHis.dict_list, json_file, indent=4)
            
            with open(stepLogDir, "r") as json_file:
                original_data = json.load(json_file)

            concurrent_time  = 0
            with open('/data/wyh/MIG_MPS/tmp/time/time_'+args.device+".txt", 'r') as file:
                for line in file:
                    concurrent_time += float(line.strip())
            updated_data = {
                "original_data": original_data,
                "elapsed_times": [{"elapsed_time": end - start}],
                "times":end - start - sumOfTime2 + concurrent_time,
            }
            print(end-start-sumOfTime2)
            with open(stepLogDir, "w") as json_file:
                json.dump(updated_data, json_file, indent=4)


