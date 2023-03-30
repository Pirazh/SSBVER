#!/usr/bin/bash

python main.py \
    --is_train False \
    --test_model teacher \
    --dataset VeRi \
    --split small \
    --device_ids 0,1,2,3 \
    --test_size 256 256 \
    --batch_size 256 \
    --model_arc resnet50_ibn_a \
    --use_bn_in_head True \
    --norm_last_layer True \
    --ssl_dim 1024 \
    --last_stride 1 \
    --test_ckpt '' \
    --cython_eval True \
    --plot_dist True \
    --test_hflip False \
    --re_rank False \
    --k1 20 \
    --k2 6 \
    --lambda_value 0.3 \
    --neck_feat after \
    --num_workers 8