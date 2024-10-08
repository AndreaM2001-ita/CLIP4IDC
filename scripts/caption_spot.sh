#!/bin/bash

export JAVA_HOME="your_jdk/"
export PATH=$JAVA_HOME/bin:$PATH

DATA_PATH=/content/drive/MyDrive/CLIP4IDC/images
python /content/drive/MyDrive/CLIP4IDC/CLIP4IDCinference.py \
--do_train \
--num_thread_reader=4 \
--epochs=150 \
--batch_size=16 \
--n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/images_test \
--output_dir ckpts/ckpt_spot_caption \
--lr 1e-4 \
--max_words 32 \
--batch_size_val 32 \
--datatype spot \
--coef_lr 1e-3 \
--freeze_layer_num 0 \
--linear_patch 2d \
--pretrained_clip_name ViT-B/32 \
--init_model /content/drive/MyDrive/pretrained1/pytorch_model.bin.30
