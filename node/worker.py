import grpc
from grpc_tool import server_scherduler_pb2 as server_scherduler_pb2
from grpc_tool import server_scherduler_pb2_grpc as server_scherduler_pb2_grpc
import subprocess
from concurrent import futures
import socket
import json
import socket

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    
    try:
        s.connect(('10.254.254.254', 1))
        ip_address = s.getsockname()[0]
    except Exception:
        ip_address = '127.0.0.1'
    finally:
        s.close()
    
    return ip_address
ip = get_ip()
node = 0
if ip == '10.16.56.14':
    node = 0
else:
    node = 1


UUID_map = {}

def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        result = s.connect_ex(('localhost', port))
        return result != 0  

def find_available_port(start_port):
    port = start_port
    while True:
        if is_port_open(port):
            return port
        port += 2  


class CommandExecutorServicer(server_scherduler_pb2_grpc.CommandExecutorServicer):
    def ExecuteCommand(self, request, context):
        try:
            GI_ID_list = []
            job_list = json.loads(request.command)
            gpu_index = int(job_list[-1])

            for i in job_list:
                GI_ID_list.append(i[4])
            UUID_list = UUID_map[node][gpu_index][GI_ID_list]

            for i in range(len(job_list)):
                UUID = UUID_list[i]
                task1, task2, demand1, demand2 = job_list[i][0],job_list[i][1],str(job_list[i][2]),str(job_list[i][3])
                port = find_available_port(12334)
                pruning_args = "--task "+task1+","+demand1+","+task2+","+demand2+ " --server_num 2 --feedback"+" --device " + UUID + " --port " + str(port)
                command = f"python /data/zbw/MIG/MIG/SC25_BOER/bayesian_pruning.py {pruning_args}"

            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            return server_scherduler_pb2.CommandResponse(result=result.stdout, status_code=result.returncode)
        except server_scherduler_pb2 as e:
            return server_scherduler_pb2.CommandResponse(result=str(e), status_code=1)

def start_worker(worker_id, port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    server_scherduler_pb2_grpc.add_CommandExecutorServicer_to_server(CommandExecutorServicer(), server)
    server.add_insecure_port(f'[::]:{port}')
    print(f"Worker {worker_id} started at port {port}...")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    start_worker("worker1", 50052)