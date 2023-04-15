# !/bin/bash
CUDA_VISIBLE_DEVICES=1 python3 testgeneric.py   --tag FMSTGCNN_30 &
P0=$!

CUDA_VISIBLE_DEVICES=2 python3 testgeneric.py   --tag FMSTGCNN_50 &
P1=$!

CUDA_VISIBLE_DEVICES=3 python3 testgeneric.py   --tag FMSTGCNN_80 &
P2=$!


wait $P0 $P1 $P2 