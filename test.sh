#!/bin/bash

# 测试预测能不能用

cd /home/assaneko/studio/moco/

# expert PKG_CONFIG_PATH="/opt/OpenBLAS/lib/pkgconfig"
# expert LDFLAGS="-L/opt/OpenBLAS/lib"
# expert CPPFLAGS="-I/opt/OpenBLAS/include"
# expert OMP_NUM_THREADS=1

python3 /home/assaneko/studio/moco/task_exp1.py \
        --name="exp_kdiv" \
        --trainpath="/mnt/nfs_datasets/fastMRI_brain/multicoil_train_sorted/size_320_640_4/" \
        --valpath="/mnt/nfs_datasets/fastMRI_brain/multicoil_val_sorted/size_320_640_4/" \
        --basedir="/home/assaneko/studio/moco/" \
        --cache='True' \
        --depth=4 \
        --topchannels=32 \
        --lr="5e-4" \
        --nsample=30\
        --epoch=30
