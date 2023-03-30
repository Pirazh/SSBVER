#!/usr/bin/bash

python vit_attn_vis.py \
    --is_train False \
    --dataset VeRiWild \
    --test_model teacher \
    --split small \
    --device_ids 0 \
    --test_size 256 256 \
    --batch_size 256 \
    --model_arc vit_base \
    --ssl_dim 16384 \
    --last_stride 1 \
    --test_ckpt './results/VeRiWild_ViTB/ckpt_epoch_118.pth' \
    --image_path '/fs/diva-scratch/pirazhkh/Re-ID/VeRI-Wild/images/00066/000676.jpg' \
    --threshold 0.5 \
    --cython_eval True \
    --plot_dist False \
    --test_hflip False \
    --re_rank False \
    --k1 20 \
    --k2 6 \
    --lambda_value 0.3 \
    --neck_feat after \
    --num_workers 16