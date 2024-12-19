#!/bin/bash
DATASET_NAME="CUHK-PEDES"
#CUHK-PEDES
#ICFG-PEDES
#RSTPReid

CUDA_VISIBLE_DEVICES=0 \
python train.py \
--name baseline \
--img_aug \
--batch_size 128 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+aux' \
--num_epoch 60 \
--root_dir '.../dataset_reid' \
--lr 3e-4 \
--num_experts 6 \
--topk 2 \
--reduction 8

