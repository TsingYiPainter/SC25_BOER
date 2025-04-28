import subprocess
import time


def OpenMPS(UUID):
    cmd  = f'export CUDA_VISIBLE_DEVICES={UUID} && export  CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-{UUID} && export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-{UUID} && sudo -E nvidia-cuda-mps-control -d && echo $CUDA_MPS_PIPE_DIRECTORY'
    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
    p.wait()
    read = str(p.stdout.read())

    cmd = f'export CUDA_VISIBLE_DEVICES={UUID} && export  CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-{UUID} && export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-{UUID} && python /data/zbw/inference_system/MIG_MPS/warmup.py'
    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
    p.wait()



def CloseMPS(UUID):
    cmd  = f'export CUDA_VISIBLE_DEVICES={UUID} && export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-{UUID} && export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-{UUID} && echo quit | sudo -E nvidia-cuda-mps-control '
    print(cmd)
    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
    p.wait()
    read = str(p.stdout.read())


def GetPid(UUID):
    cmd = f'export CUDA_VISIBLE_DEVICES={UUID} && export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-{UUID} && export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-{UUID} && echo  get_server_list | sudo -E nvidia-cuda-mps-control'
    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
    p.wait()
    read = str(p.stdout.read().decode())
    server_ID = int(read)

    return server_ID

def SetPercentage(UUID, Percentage):
    cmd = f'export CUDA_VISIBLE_DEVICES={UUID} && export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-{UUID} && export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-{UUID} && echo  get_server_list | sudo -E nvidia-cuda-mps-control'
    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
    p.wait()
    read = str(p.stdout.read().decode())
    server_ID = int(read)

    cmd = f'export CUDA_VISIBLE_DEVICES={UUID} && export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-{UUID} && export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-{UUID} && sudo echo set_active_thread_percentage {server_ID} {Percentage} |sudo -E nvidia-cuda-mps-control'

    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
    p.wait()


def GetPercentage(UUID):
    cmd = f'export CUDA_VISIBLE_DEVICES={UUID} && export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-{UUID} && export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-{UUID} && echo  get_server_list | sudo -E nvidia-cuda-mps-control'
    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
    p.wait()
    read = str(p.stdout.read().decode())
    server_ID = int(read)

    cmd = f'export CUDA_VISIBLE_DEVICES={UUID} && export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-{UUID} && export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-{UUID} && sudo echo get_active_thread_percentage {server_ID}  |sudo -E nvidia-cuda-mps-control'
    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
    p.wait()
    read = str(p.stdout.read().decode().strip())
    SM_percentage = float(read)
    return SM_percentage


# CloseMPS('MIG-409bcca6-a4c6-5f62-a3ee-9ff73b470af8')
# CloseMPS("MIG-e1a6d5d7-52af-5b1d-a897-ce0f71007d15")
# CloseMPS("MIG-fce9dba3-5cd7-59fa-b174-039c6df3e16e")
