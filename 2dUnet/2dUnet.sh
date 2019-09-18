#!/bin/bash
module load TensorFlow/1.9-gpu-py3.6-cuda90
yhrun -n 1 -c 2 -p gpu_v100  python train2.py 