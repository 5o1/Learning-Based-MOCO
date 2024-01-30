#!/bin/bash

# 测试预测能不能用

cd /home/assaneko/studio/moco_fastmri/

# expert PKG_CONFIG_PATH="/opt/OpenBLAS/lib/pkgconfig"
# expert LDFLAGS="-L/opt/OpenBLAS/lib"
# expert CPPFLAGS="-I/opt/OpenBLAS/include"
# expert OMP_NUM_THREADS=1

python3 task_testnetac.py \
        --name="networktest" \
        --trainpath="/mnt/nfs_datasets/fastMRI_brain/multicoil_train_sorted/size_320_640_4/" \
        --valpath="/mnt/nfs_datasets/fastMRI_brain/multicoil_val_sorted/size_320_640_4/" \
        --basedir="/home/assaneko/studio/moco_fastmri/" \
        --cache='True' \
        --depth=5 \
        --topchannels=32 \
        --lr="5e-4" \
        --nsample=1\
        --epoch=30
