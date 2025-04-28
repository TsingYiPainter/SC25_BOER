# BOER: Enhancing Resource Utilization for Deep Learning Inference with Hybrid Spatial GPU Sharing

## What is BOER?
BOER an inference-serving system that
integrates MPS atop MIG partitions while preserving QoS. Also, BOER employs a robust Bayesian optimization framework to efficiently configure MPS and accelerates the optimization process.


## How to run
Assume there are three tasks: model1, model2, and model3, and their loads are load1, load2, load3.

Before the official run, execute the following command on each node:

```
$python node/worker.py
```

Then, run the following command on the master node:

```
$python scheduler.py --task model1,load1,model2,load2,model3,load3
```

## Dependencies
### Hardware
2 servers,
each equipped with 2 * Intel Xeon Gold 5320 CPUs, 4 * NVIDIA A100 80GB GPUs, and 
- CUDA version 12.6
- Ubuntu 22.04

### Software
Refer to `environment.yml`






