import numpy as np
import sys
import torch
import time
import pandas as pd 
import argparse
from bert import BertModel
from mobilenet_v2 import mobilenet
from vgg_splited import vgg16, vgg19
from resnet import resnet50,resnet101,resnet152
from LeNet5 import load_LeNet5
from DenseNet import DenseNet
from GooLeNet import GooLeNet
from MnasNet  import MnasNet
from SSD_MobileNet import SSD_MobileNet
import signal
import math
import logging
import re
from filelock import FileLock
import os
import socket
import threading

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ['TORCH_HOME'] = '/data/zbw/'
path = "/data/zbw/inference_system/MIG_MPS/jobs/"
sys.path.append(path)
flag_path = "/data/zbw/inference_system/MIG/MIG/MIG_Schedule/flag"
result_path = "/data/zbw/inference_system/MIG_MPS/log/"
bayesTmp_path = '/data/wyh/MIG_MPS/tmp/bayesian_tmp.txt'
time_path = '/data/wyh/MIG_MPS/tmp/time/time.txt'
dynamic_path = '/data/wyh/MIG_MPS/tmp/dynamic/dynamic.txt'
dynamic_list = []
stamp_flag=False

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(42)  

running_tcp_ip = '127.0.0.1'
running_tcp_port = 12335

binary_tcp_ip =  '127.0.0.1'
binary_tcp_port = 12334

sumOfTime = 0
sumOfTimeAlone = 0

model_list = {
    'SSD_MobileNet': SSD_MobileNet,
    'MnasNet': MnasNet,
    'GooLeNet': GooLeNet,
    'DenseNet': DenseNet,
    'LeNet5': load_LeNet5,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "vgg19": vgg19,
    "vgg16": vgg16,
    "inception_v3": inception_v3,
    'unet': unet,
    'deeplabv3':deeplabv3,
    'mobilenet_v2': mobilenet,
    # 'open_unmix':open_unmix,
    'alexnet': alexnet,
    'bert': BertModel,
    'transformer': transformer_layer,
}

input_tensor_list = {
    'SSD_MobileNet': [3, 300, 300],
    'MnasNet': [3, 244, 244],
    'GooLeNet': [3, 244, 244],
    'DenseNet': [3, 244, 244],
    "resnet50": [3, 244, 244],
    "resnet101": [3, 244, 244],
    "resnet152": [3, 244, 244],
    "vgg19": [3, 244, 244],
    "vgg16": [3, 244, 244],
    "inception_v3": [3, 299, 299],
    "unet": [3,256,256],
    'deeplabv3': [3,256,256],
    'mobilenet_v2': [3,244,244],
    # 'open_unmix': [2,100000],
    'alexnet': [3,244,244],
    'bert': [1024,768],
    'LeNet5': [1,28,28],
}

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

max_RPS_map = {
    'LeNet5': 1600000,
    'DenseNet': 1600,
    'GooLeNet':3500,
    'MnasNet': 3400,
    'resnet50': 2000,
    'resnet101': 1500,
    'resnet152': 1000,
    'vgg16': 1500,
    'vgg19': 1300,
    'mobilenet_v2': 4000,
    'unet': 1300,
    'bert': 250, 
    'deeplabv3': 300,
    'alexnet' : 7000,
    'SSD_MobileNet': 900,
}

min_RPS_map = {
    'LeNet5': 1000,
    'DenseNet': 10,
    'GooLeNet':10,
    'MnasNet': 10,
    'SSD_MobileNet': 1,
    'resnet50': 1,
    'resnet101': 1,
    'resnet152': 1,
    'vgg16': 1,
    'vgg19': 1,
    'mobilenet_v2': 200,
    'unet': 1,
    'bert': 1,
    'deeplabv3': 1,
    'alexnet' : 500,
}

def handle_terminate(signum, frame):
    pass


def get_model(model_name):
    return  model_list.get(model_name)

def get_input_tensor(model_name, k):
    input_tensor = input_tensor_list.get(model_name)
    if model_name == 'bert':
        input_tensor = torch.FloatTensor(np.random.rand(k, 1024, 768))
        masks = torch.FloatTensor(np.zeros((k, 1, 1, 1024)))
        return input_tensor,masks
    

    if model_name == 'transformer':
        input_tensor = torch.randn(512, k, 768)
        masks = torch.ones(512, 512)

        return input_tensor,masks
    
    if model_name == 'SSD_MobileNet':
        # utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
        input_shape = torch.randn(k, input_tensor[0], input_tensor[1], input_tensor[2])
        # print(input_shape.shape)
        # input_tensor = utils.prepare_tensor(input_shape)
        # print(input_tensor.shape)
        return input_shape

    if len(input_tensor) == 3:
        return torch.randn(k, input_tensor[0], input_tensor[1], input_tensor[2])
    else:
        return torch.randn(k, input_tensor[0], input_tensor[1])

def handle_concurrent_valid_data(valid_list, task, config, batch):

    file_name = result_path + f"{task}_Pairs_MPS_RPS"

    data = np.array(valid_list)
    percentile_95 = np.percentile(data, 95)
    
    with open(file_name, 'a+') as file:
        file.write(f"task: {task}, SM: {config}, batch: {batch}, 99th percentile: {percentile_95}\n")

def get_p95(data):
    data = np.array(data)
    percentile_95 = np.percentile(data, 95)
    return percentile_95

def get_p99(data):
    data = np.array(data)
    percentile_99 = np.percentile(data, 99)
    return percentile_99

def get_p98(data):
    data = np.array(data)
    percentile_98 = np.percentile(data, 98)
    return percentile_98


def record_result(path, config, RPS ,result):
    filtered_result = result[300:]
    p99 = get_p99(filtered_result)
    with open(path, 'a+') as file:
        file.write(f"Config: {config}, P99: {p99}, RPS: {RPS}\n")
        file.close()

def execute_entry(task, RPS, max_epoch):
    QoS = QoS_map.get(task)
    half_QoS = QoS/2
    batch = math.floor(RPS/1000 * half_QoS)
    valid_list = []

    if task == 'bert':  
        model = get_model(task)
        model = model().half().cuda(0).eval()
    else:
        model = get_model(task)
        model = model().cuda(0).eval()

    with torch.no_grad():
        for i in range(0, max_epoch):
            if task == 'bert':
                input_tensor,masks = get_input_tensor(task, batch)
                input_tensor = input_tensor.half()
                masks = masks.half()
                
            elif task == 'transformer':
                input_tensor,masks = get_input_tensor(task, batch)
            else:
                input_tensor = get_input_tensor(task, batch)

            start_time = time.time()
            if task == 'bert':
                input_tensor = input_tensor.cuda(0)
                masks = masks.cuda(0)
            elif task == 'transformer':
                input_tensor = input_tensor.cuda(0)
                masks = masks.cuda(0)
            else:
                input_tensor = input_tensor.cuda(0)

            if task == 'bert':
                output= model.run(input_tensor,masks,0,12).cpu()
            elif task == 'transformer':

                outputs = model(input_tensor, input_tensor, src_mask=masks, tgt_mask=masks).cpu()
            elif task == 'SSD_MobileNet':
                output=model(input_tensor)
                predictions = output[0].cpu()
            elif task == 'deeplabv3':
                output= model(input_tensor)['out'].cpu()
            else:
                output=model(input_tensor).cpu()
            end_time = time.time()

            valid_list.append((end_time - start_time) * 1000)

        filtered_result = valid_list[200:]
        p99 = get_p95(filtered_result)
        # print(p99, half_QoS, RPS)
        if p99 > half_QoS:
            # print(task, p99, RPS)
            # record_result(path=file_name, config=config, RPS=RPS, result=valid_list)
            return False
        else:
            record_result(path=file_name, config=config, RPS=RPS, result=valid_list)
            return True
        
def binary_search_max_true(task ,min_RPS, max_RPS, max_epoch):
    left = min_RPS
    right = max_RPS

    while left < right:
        mid = (left + right + 1) // 2
        if execute_entry(task=task, RPS=mid, max_epoch=max_epoch):
            left = mid  
        else:
            right = mid - 1  

    return left  

def feedback_execute_entry(task, RPS, remote_half_QoS):

    QoS = QoS_map.get(task)
    half_QoS = QoS/2
    batch = math.floor(RPS/1000 * half_QoS)
    
    valid_list = []

 
    try:

        if task == 'bert':  
            model = get_model(task)
            model = model().half().cuda(0).eval()
        else:
            model = get_model(task)
            model = model().cuda(0).eval()

        if task == 'bert':
                input_tensor,masks = get_input_tensor(task, batch)
                input_tensor = input_tensor.half()
                masks = masks.half()

        elif task == 'transformer':
            input_tensor,masks = get_input_tensor(task, batch)

        else:
            input_tensor = get_input_tensor(task, batch)

        if task == 'bert':
            input_tensor = input_tensor.cuda(0)
            masks = masks.cuda(0)
        elif task == 'transformer':
            input_tensor = input_tensor.cuda(0)
            masks = masks.cuda(0)
        else:
            input_tensor = input_tensor.cuda(0)

        if task == 'bert':
            output= model.run(input_tensor,masks,0,12).cpu()
        elif task == 'transformer':

            outputs = model(input_tensor, input_tensor, src_mask=masks, tgt_mask=masks).cpu()

        elif task == 'SSD_MobileNet':
                output=model(input_tensor)
                predictions = output[0].cpu()

        elif task == 'deeplabv3':
            output= model(input_tensor)['out'].cpu()
        else:
            output=model(input_tensor).cpu()

        sned_flag = False

        global dynamic_list

        timestamp = time.time()
        dynamic_list.append(["modelA","modelAmid",timestamp,"start concurrency"])
        dynamic_list.append(["modelB",RPS,timestamp,"start concurrency"])

        with torch.no_grad():
            global sumOfTime
            for i in range(0, 10):
                if task == 'bert':
                    input_tensor,masks = get_input_tensor(task, batch)
                    input_tensor = input_tensor.half()
                    masks = masks.half()

                elif task == 'transformer':
                    input_tensor,masks = get_input_tensor(task, batch)

                else:
                    input_tensor = get_input_tensor(task, batch)

                start_time = time.time()
                
                if task == 'bert':
                    input_tensor = input_tensor.cuda(0)
                    masks = masks.cuda(0)
                elif task == 'transformer':
                    input_tensor = input_tensor.cuda(0)
                    masks = masks.cuda(0)
                else:
                    input_tensor = input_tensor.cuda(0)

                start_time_concurrent = time.time()

                if task == 'bert':
                    output= model.run(input_tensor,masks,0,12).cpu()
                elif task == 'transformer':

                    outputs = model(input_tensor, input_tensor, src_mask=masks, tgt_mask=masks).cpu()

                elif task == 'SSD_MobileNet':
                        output=model(input_tensor)
                        predictions = output[0].cpu()

                elif task == 'deeplabv3':
                    output= model(input_tensor)['out'].cpu()
                else:
                    output=model(input_tensor).cpu()

                end_time = time.time()
                #print("interval is {}".format(end_time - start_time_concurrent))
                sumOfTime += end_time - start_time_concurrent

                if i == 0 :
                    send_tcp_message(running_tcp_ip, running_tcp_port, 'start')
                    sned_flag = True
                else:
                    valid_list.append((end_time - start_time) * 1000)
        


            send_tcp_message(running_tcp_ip, running_tcp_port, 'finish')

            data = np.array(valid_list[5:])
            percentile_95 = np.percentile(data, 95)


            time.sleep(1)

    except Exception as e:
        print('feedback error')
        if sned_flag == False:
            send_tcp_message(running_tcp_ip, running_tcp_port, 'start')
            time.sleep(5)
        else:
            send_tcp_message(running_tcp_ip, running_tcp_port, 'finish')
        percentile_95 = half_QoS + 100
        

    print(percentile_95, RPS)
    lcA = tcp_control.get_latency()
    lcB = percentile_95
    timestamp = time.time()
    dynamic_list.append(["modelA","modelAmid",timestamp-1,lcA])
    dynamic_list.append(["modelB",RPS,timestamp-1,lcB])
    if percentile_95 > half_QoS or float(tcp_control.get_latency()) > remote_half_QoS:

        # print(f"batch is {batch} latency: {percentile_95}, remote_latency: {float(tcp_control.get_latency())}" )
        return [False,percentile_95]
    else:
        # print(f"batch is {batch} latency: {percentile_95}, remote_latency: {float(tcp_control.get_latency())}" )
        return [True,percentile_95]

def feedback_search_max_true(task, RPS, remote_half_QoS):
    min_RPS = 1
    max_RPS = RPS

    right = max_RPS
    left = min_RPS
    beginT = 0
    global dynamic_list
    gap = int((max_RPS - min_RPS)/100)

    while left+gap < right:
        mid = (left + right + 1) // 2
        if mid == (min_RPS + max_RPS + 1) // 2:
            beginT = time.time()
        returnList = feedback_execute_entry(task=task, RPS=mid, remote_half_QoS=remote_half_QoS)
        valid_flag = returnList[0]
        if valid_flag:
            left = mid
        else:
            right = mid - 1  

    ## !!!
    with open(dynamic_path, 'a') as file:  # 追加模式
        for item in dynamic_list:  # 遍历列表
            file.write(f"{item[0]}, {item[1]}, {item[2]},{item[3]}\n") 

    return left 



class TCPControl:
    def __init__(self):
        self.state = None
        self.latency = None

    def set_state(self, state):
        self.state = state
        # print(f"set state {state}")

    def set_latency(self, latency):
        self.latency = float(latency)
        # print(f"set latency {latency}")

    def get_state(self):
        return self.state

    def get_latency(self):
        return self.latency
    
    def reset_state(self):
        self.state = None
        self.latency = None


def tcp_server(host, port, control):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)
    # print(f"Server is listening on {host}:{port}")

    while True:
        client_socket, addr = server_socket.accept()
        message = client_socket.recv(1024).decode().strip()
        if message in ['start', 'finish']:
            control.set_state(message)

        elif message == 'succeed':
            # print("Received 'succeed', shutting down server.")
            control.set_state(message)
            client_socket.close() 
            break
        else:
            control.set_latency(message)
        client_socket.close()

    server_socket.close()
    # print("Server socket has been closed.")

tcp_control = TCPControl()

def start_server(host, port):
    server_thread = threading.Thread(target=tcp_server, args=(host, port, tcp_control))
    server_thread.start()



def send_tcp_message(host, port, message):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
            # print(f"send message to {host} {port} {message}")
            sock.sendall(message.encode())
    except Exception as e:
        print(f"Error sending TCP message: {e}")






if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--config", default='', type=str)
    parser.add_argument("--file_name", type=str, default='result')
    parser.add_argument("--RPS", type=int)
    parser.add_argument("--test", action='store_false')
    parser.add_argument("--concurrent_profile", action='store_true')
    parser.add_argument("--gpulet", action='store_true')
    parser.add_argument("--worker_id", type=int)
    parser.add_argument("--bayes", action='store_true')
    parser.add_argument("--feedback", action='store_true')
    parser.add_argument("--running", action="store_true")
    parser.add_argument("--port", default=12334, type = int)
    parser.add_argument("--GI",default=None, type =str)
    args = parser.parse_args()

    task = args.task
    concurrent_profile = args.concurrent_profile
    config = args.config
    file_name = args.file_name
    test = args.test
    RPS = args.RPS
    batch = args.batch
    gpulet = args.gpulet
    bayes = args.bayes
    feedback = args.feedback
    running = args.running

    binary_tcp_port = args.port
    running_tcp_port = args.port + 1
    
    if args.GI is not None:
        bayesTmp_path = bayesTmp_path.replace('.txt', f'_{args.GI}.txt')
        time_path = time_path.replace('.txt', f'_{args.GI}.txt')
        dynamic_path = dynamic_path.replace('.txt', f'_{task}_{args.GI}.txt')

    max_epoch = 1000
    min_RPS = min_RPS_map.get(task)
    max_RPS = max_RPS_map.get(task)

    if test:
        QoS = QoS_map.get(task)
        half_QoS = QoS/2
        if batch:
            pass
        else:

            batch = math.floor(RPS/1000 * half_QoS)
      
        if task == 'bert':  
            model = get_model(task)
            model = model().half().cuda(0).eval()
        else:
            model = get_model(task)
            model = model().cuda(0).eval()
    

        with torch.no_grad():
            while True:
                valid_list = []
                for i in range(0, 100):
                    if task == 'bert':
                        input_tensor,masks = get_input_tensor(task, batch)
                        input_tensor = input_tensor.half()
                        masks = masks.half()
                    elif task == 'transformer':
                        input_tensor,masks = get_input_tensor(task, batch)
                    else:
                        input_tensor = get_input_tensor(task, batch)

                    start_time = time.time()

                    if task == 'bert':
                        input_tensor = input_tensor.cuda(0)
                        masks = masks.cuda(0)
                    elif task == 'transformer':
                        input_tensor = input_tensor.cuda(0)
                        masks = masks.cuda(0)
                    else:
                        input_tensor = input_tensor.cuda(0)

                    if task == 'bert':
                        output= model.run(input_tensor,masks,0,12).cpu()

                    elif task == 'SSD_MobileNet':
                        output=model(input_tensor)
                        predictions = output[0].cpu()
                        
                    elif task == 'transformer':

                        outputs = model(input_tensor, input_tensor, src_mask=masks, tgt_mask=masks).cpu()
                        
                    elif task == 'deeplabv3':
                        output= model(input_tensor)['out'].cpu()
                    else:
                        output=model(input_tensor).cpu()

                    end_time = time.time()
                    print((end_time - start_time) * 1000)
                    valid_list.append((end_time - start_time) * 1000)

                print("P99: ", get_p95(valid_list))

              

    elif concurrent_profile:  

        # if task == 'bert':  
        #     model = get_model(task)
        #     model = model().half().cuda(0).eval()
        # else:
        #     model = get_model(task)
        #     model = model().cuda(0).eval()

        # print("finish memory")

        if (not bayes) or (not feedback):
            pass
          
        #     with torch.no_grad():
        #         valid_list = []
                
        #         for i in range(0, 500):
                    
        #             if task == 'bert':
        #                 input_tensor,masks = get_input_tensor(task, batch)
        #                 input_tensor = input_tensor.half()
        #                 masks = masks.half()
        #             elif task == 'transformer':
        #                 input_tensor,masks = get_input_tensor(task, batch)
        #             else:
        #                 input_tensor = get_input_tensor(task, batch)


        #             start_time = time.time()
                    
        #             if task == 'bert':
        #                 input_tensor = input_tensor.cuda(0)
        #                 masks = masks.cuda(0)
        #             elif task == 'transformer':
        #                 input_tensor = input_tensor.cuda(0)
        #                 masks = masks.cuda(0)
        #             else:
        #                 input_tensor = input_tensor.cuda(0)

        #             if task == 'bert':
        #                 output= model.run(input_tensor,masks,0,12).cpu()
        #             elif task == 'transformer':

        #                 outputs = model(input_tensor, input_tensor, src_mask=masks, tgt_mask=masks).cpu()
                        
        #             elif task == 'deeplabv3':
        #                 output= model(input_tensor)['out'].cpu()
        #             else:
        #                 output=model(input_tensor).cpu()

        #             end_time = time.time()

        #             valid_list.append((end_time - start_time) * 1000)

        #         if not bayes:
        #             handle_concurrent_valid_data(valid_list[200:], task, config, batch)

        #         else:
        #             data = np.array(valid_list[200:])
        #             percentile_95 = np.percentile(data, 95)
        #             file_path = bayesTmp_path
        #             lock_path = file_path + '.lock'  

    
        #             lock = FileLock(lock_path)

        #             with lock:
        #                 with open(file_path, 'a+') as file:
        #                     file.write(f"{task} {batch} {config} {percentile_95}\n")
            
        elif running:
            
            start_server(host=running_tcp_ip, port=running_tcp_port)
            try:
               

                QoS = QoS_map.get(task)
                half_QoS = QoS/2
                if task == 'bert':  
                    model = get_model(task)
                    model = model().half().cuda(0).eval()
                else:
                    model = get_model(task)
                    model = model().cuda(0).eval()
                

                tmp_list = []

                percentile_95 = None
                startTime_alone = time.time()
                for i in range(0, 20):
                    if task == 'bert':
                        input_tensor,masks = get_input_tensor(task, batch)
                        input_tensor = input_tensor.half()
                        masks = masks.half()

                    elif task == 'transformer':
                        input_tensor,masks = get_input_tensor(task, batch)
                    else:
                        input_tensor = get_input_tensor(task, batch)

                    
                
                    start_time = time.time()
                    
                    if task == 'bert':
                        input_tensor = input_tensor.cuda(0)
                        masks = masks.cuda(0)
                    elif task == 'transformer':
                        input_tensor = input_tensor.cuda(0)
                        masks = masks.cuda(0)
                    else:
                        input_tensor = input_tensor.cuda(0)
                    if task == 'bert':
                        output= model.run(input_tensor,masks,0,12).cpu()

                    elif task == 'transformer':

                        outputs = model(input_tensor, input_tensor, src_mask=masks, tgt_mask=masks).cpu()
 
                    elif task == 'SSD_MobileNet':
                        output=model(input_tensor) 
                        predictions = output[0].cpu()

                    elif task == 'deeplabv3':
                        output= model(input_tensor)['out'].cpu()
                    else:
                        output=model(input_tensor).cpu()

                    end_time = time.time()
                    tmp_list.append((end_time - start_time) * 1000)
                sumOfTimeAlone += (time.time()-startTime_alone)

                data = np.array(tmp_list[5:])
                percentile_95 = np.percentile(data, 95)
                # file_path = bayesTmp_path
                # lock_path = file_path + '.lock'  
                # lock = FileLock(lock_path)

            except Exception as e:
                print(e)
                print('running error')
                percentile_95 = half_QoS + 100
                time.sleep(10)

            file_path = bayesTmp_path
            lock_path = file_path + '.lock'  
            lock = FileLock(lock_path)

            with open(file_path, 'a+') as file:
                file.write(f"model: {task} latency: {percentile_95}\n")
            # with lock:
            #     with open(file_path, 'a+') as file:
            #         file.write(f"model: {task} latency: {min(half_QoS-1,percentile_95)}\n")
            
            # with open(file_path.replace('.txt', f'_True.txt'), 'a+') as file:
            #     file.write(f"model: {task} latency: {percentile_95}\n")

            #percentile_95 = min(half_QoS-1,percentile_95)        
            if percentile_95 <= half_QoS:
                ## ！！！
                # with open("/data/wyh/MIG_MPS/tmp/dynamic/dynamic_vgg19_MIG-025b3a98-2de6-5a56-9752-ca17816855b4.txt", 'a') as file:  # 追加模式
                #     file.write('model A\n')
                #     file.write(f"{percentile_95}\n") 

                with torch.no_grad():
                    valid_list = []
                    while True:
                        if task == 'bert':
                            input_tensor,masks = get_input_tensor(task, batch)
                            input_tensor = input_tensor.half()
                            masks = masks.half()

                        elif task == 'transformer':
                            input_tensor,masks = get_input_tensor(task, batch)
                        else:
                            input_tensor = get_input_tensor(task, batch)
                        

                        start_time = time.time()
                        
                        if task == 'bert':
                            input_tensor = input_tensor.cuda(0)
                            masks = masks.cuda(0)
                        elif task == 'transformer':
                            input_tensor = input_tensor.cuda(0)
                            masks = masks.cuda(0)
                        else:
                            input_tensor = input_tensor.cuda(0)

                        if task == 'bert':
                            output= model.run(input_tensor,masks,0,12).cpu()
                        elif task == 'transformer':

                            outputs = model(input_tensor, input_tensor, src_mask=masks, tgt_mask=masks).cpu()

                        elif task == 'SSD_MobileNet':
                            output=model(input_tensor)
                            predictions = output[0].cpu()

                        elif task == 'deeplabv3':
                            output= model(input_tensor)['out'].cpu()
                        else:
                            output=model(input_tensor).cpu()

                        end_time = time.time()
                        

                        if tcp_control.get_state() == 'start':
                            valid_list.append((end_time - start_time) * 1000)



                        if tcp_control.get_state() == 'finish':
                            data = np.array(valid_list[5:])
                            if len(data) == 0:
                                percentile_95 = half_QoS + 1
                            else:
                                percentile_95 = np.percentile(data, 95)
                            send_tcp_message(host=binary_tcp_ip, port=binary_tcp_port, message=str(percentile_95))
                            valid_list = []
                            tcp_control.reset_state()  

                        if tcp_control.get_state() == 'succeed':
                            
                            send_tcp_message(host=binary_tcp_ip, port=binary_tcp_port, message='succeed')

                            break
            else:
                time.sleep(10)
                with open(time_path, 'a') as file:
                    file.write(f"{sumOfTimeAlone}\n")
                send_tcp_message(host=running_tcp_ip, port=running_tcp_port, message='succeed')
                    
        else:
            file_path = bayesTmp_path
            while True:

                if os.path.getsize(file_path) == 0:
                    print("wait for latency")
                else:
                    break
                time.sleep(1)

            start_server(host=binary_tcp_ip, port=binary_tcp_port)
            latency = None

            with open(file_path, 'r') as file:
                line = file.readline().strip()
                match = re.search(r"model:\s*(\S+)\s+latency:\s*([\d.]+)", line)

                if match:
                    model = match.group(1)
                    latency = float(match.group(2))
                    print("Model:", model)
                    print("Latency:", latency)

                else:
                    print("error!")

            QoS = QoS_map.get(model)
            half_QoS = QoS/2

            if float(latency) < half_QoS:
                vaild_RPS = feedback_search_max_true(task=task, RPS=RPS, remote_half_QoS=half_QoS)
                
                send_tcp_message(host=running_tcp_ip, port=running_tcp_port, message='succeed')

                print(f"find largest valid RPS: {vaild_RPS}" )
                ## !!!

                # if int(latency) != half_QoS-1:
                #     file_path = bayesTmp_path.replace('.txt', f'_True.txt')
                #     lock_path = file_path + '.lock'  


                #     lock = FileLock(lock_path)

                #     with lock:
                        
                #         with open(file_path, 'w') as file:
                #             file.write(f"valid_RPS: {vaild_RPS}\n")

                file_path = bayesTmp_path
                lock_path = file_path + '.lock'  


                lock = FileLock(lock_path)

                with lock:
                    with open(file_path, 'w') as file:
                        file.write(f"valid_RPS: {vaild_RPS}\n")

            else:
                send_tcp_message(host=binary_tcp_ip, port=binary_tcp_port, message='succeed')
            
            with open(time_path, 'a') as file:
                file.write(f"{sumOfTime}\n")
            

                
    elif gpulet:

        QoS = QoS_map.get(task)
        half_QoS = QoS/2
        print(f"start gpulet worker for {task} {RPS}", flush=True)
        if batch:
            pass
        else:

            batch = math.floor(RPS/1000 * half_QoS) 
      
        if task == 'bert':  
            model = get_model(task)
            model = model().half().cuda(0).eval()
        else:
            model = get_model(task)
            model = model().cuda(0).eval()


        with torch.no_grad():
            while True:
                valid_list = []
                for i in range(0, 200):
                    if task == 'bert':
                        input_tensor,masks = get_input_tensor(task, batch)
                    elif task == 'transformer':
                        input_tensor,masks = get_input_tensor(task, batch)
                    else:
                        input_tensor = get_input_tensor(task, batch)

                    start_time = time.time()

                    if task == 'bert':
                        input_tensor = input_tensor.half().cuda(0)
                        masks = masks.half().cuda(0)
                    elif task == 'transformer':
                        input_tensor = input_tensor.cuda(0)
                        masks = masks.cuda(0)
                    else:
                        input_tensor = input_tensor.cuda(0)

                    if task == 'bert':
                        output= model.run(input_tensor,masks,0,12).cpu()
                    elif task == 'transformer':

                        outputs = model(input_tensor, input_tensor, src_mask=masks, tgt_mask=masks).cpu()
                        
                    elif task == 'deeplabv3':
                        output= model(input_tensor)['out'].cpu()
                    else:
                        output=model(input_tensor).cpu()
                    end_time = time.time()
                    valid_list.append((end_time - start_time) * 1000)

                p99 = get_p99(valid_list[10:])
                if p99 > half_QoS:
                    print(f"{task} {RPS} QoS violate", flush=True)
                    # logging.info(f"{task} {RPS} QoS violate")
            
                else:
                    print(f"{task} {RPS} {p99} {half_QoS}", flush=True)
                    # logging.info(f"{task} {RPS} {p99} {half_QoS}")

       


    else:
        binary_search_max_true(task=task, min_RPS=min_RPS, max_RPS=max_RPS, max_epoch=max_epoch)
    
    