#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ python saliency/input_saliency.py \
    --is_train False \
    --dataset VeRiWild \
    --test_model teacher \
    --test_size 256 256 \
    --model_arc resnet50_ibn_a \
    --ssl_dim 16384 \
    --last_stride 1 \
    --test_ckpt '' \
    --query_path '' \
    --gallery_path '' \
    --neck_feat after

