#!/bin/bash
workdir=/data/wyh/MIG_MPS/bayesian_optimization
python_path=/data/wyh/conda_env/miniconda3/envs/Abacus/bin/python

model1=DenseNet
model2=vgg19
load1=553
load2=537
device=GPU-08dffabe-6be4-81d7-ba7d-1d96612fb099
port=12334
           
for outer in {1..1}; do  
    echo "Outer Loop Iteration $outer: Starting inner loop..."
        (
            cd "$workdir" && "$python_path" bayesian_pruning.py \
            --task "$model1,$load1,$model2,$load2" \
            --server_num 2 \
            --feedback \
            --device "$device" \
            --port "$port" \
            --idxGI 0 \
            --numOfGI 1 \
            --seed $outer\
        )
        if [ $? -ne 0 ]; then
            echo "    Error in inner iteration $i during outer loop $outer. Exiting..."
            exit 1
        fi
        wait
        sleep 1  
    echo "Outer Loop Iteration $outer completed."
done

echo "All loops completed successfully."