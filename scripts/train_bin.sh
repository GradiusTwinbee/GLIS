#!/bin/bash


TOKENIZERS_PARALLELISM=false python main.py \
--dataset_name scannet \
--max_epoch 10 \
--nqueries 128 \
--base_lr 1e-3 \
--warm_lr_epochs 1 \
--matcher_giou_cost 1 \
--matcher_cls_cost 0 \
--matcher_center_cost 0 \
--matcher_objectness_cost 0 \
--loss_giou_weight 0 \
--loss_no_object_weight 0.1 \
--save_separate_checkpoint_every_epoch -1 \
--checkpoint_dir bin \
--ngpus 4 \
--dataset_num_workers 2 \
--dist_url tcp://localhost:52456 \
--batchsize_per_gpu 4 \
--train_bin
