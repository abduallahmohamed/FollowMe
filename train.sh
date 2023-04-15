# !/bin/bash

CUDA_VISIBLE_DEVICES=1 python3 train.py --w_trip 0.0001 --lr 0.001 --pred_time 50  --tag testModel &
P0=$!
wait $P0 