#!/bin/bash

module load PyTorch/1.1.0-CUDA9.0-py3.6
yhrun -n 1 -c 2 -p gpu_v100 python classifier_train.py --model_type vgg --epochs 50