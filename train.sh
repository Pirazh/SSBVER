#!/usr/bin/bash

source activate py36-cuda11

python main.py \
    --is_train True \
    --cython_eval True \
    --device_ids 0,1,2,3 \
    --dataset VeRiWild \
    --local_crops_num 4 \
    --global_crop_scale 0.8 1 \
    --pad_size 10 \
    --local_crop_scale 0.1 0.4 \
    --train_global_size 256 256 \
    --train_local_size 112 112\
    --test_size 256 256 \
    --num_instances 4 \
    --batch_size 128 \
    --model_arc resnet50_ibn_a \
    --last_stride 1 \
    --pretrained True \
    --pretrained_method ImageNet \
    --ssl_dim 16384 \
    --use_bn_in_head True \
    --norm_last_layer True \
    --optimizer adam \
    --label_smoothing True \
    --label_smoothing_eps 0.2 \
    --momentum_teacher 0.9995 \
    --warmup_teacher_temp 0.0005 \
    --teacher_temp 0.001 \
    --warmup_teacher_temp_epochs 10 \
    --student_temp 0.1 \
    --weight_decay 0.001 \
    --lr 0.001 \
    --scheduler gamma \
    --epochs 120 \
    --warmup_epochs 10 \
    --warmup_factor 0.01 \
    --clip_grad 0.0 \
    --milestones 40 70 100 \
    --save_ckpt_freq 10 \
    --eval_freq 10 \
    --log_freq 100 \
    --use_margin False \
    --triplet_loss_lambda 1 \
    --ssl_loss_lambda 0.005 \
    --cmpt_loss_lambda 0.0 \
    --output_dir './results/VeRiWild_resnet50_ibn_a'