####################################################################################################
#   This is just a short script to run through the entire model, stage 1 and 2 with cross-validation
#   To get the same effect as running this script, run the following commands:
#       python train.py --exp_name TRAIN --gpus 1 --gpu_4g_limit True --num_epochs 301 --batch_size 2 --validate_epoch 30 --cv 0 --stage 1
#       python train.py --exp_name TRAIN --gpus 1 --gpu_4g_limit True --num_epochs 301 --batch_size 2 --validate_epoch 30 --cv 0 --stage 2 --prob_dir {data_dir}/output/run_TRAIN/exp_0
#       python train.py --exp_name TRAIN --gpus 1 --gpu_4g_limit True --num_epochs 301 --batch_size 2 --validate_epoch 30 --cv 0 --stage 1
#       python train.py --exp_name TRAIN --gpus 1 --gpu_4g_limit True --num_epochs 301 --batch_size 2 --validate_epoch 30 --cv 0 --stage 2 --prob_dir {data_dir}/output/run_TRAIN/exp_1
#       python train.py --exp_name TRAIN --gpus 1 --gpu_4g_limit True --num_epochs 301 --batch_size 2 --validate_epoch 30 --cv 0 --stage 1
#       python train.py --exp_name TRAIN --gpus 1 --gpu_4g_limit True --num_epochs 301 --batch_size 2 --validate_epoch 30 --cv 0 --stage 2 --prob_dir {data_dir}/output/run_TRAIN/exp_2
#       python train.py --exp_name TRAIN --gpus 1 --gpu_4g_limit True --num_epochs 301 --batch_size 2 --validate_epoch 30 --cv 0 --stage 1
#       python train.py --exp_name TRAIN --gpus 1 --gpu_4g_limit True --num_epochs 301 --batch_size 2 --validate_epoch 30 --cv 0 --stage 2 --prob_dir {data_dir}/output/run_TRAIN/exp_3
#       python train.py --exp_name TRAIN --gpus 1 --gpu_4g_limit True --num_epochs 301 --batch_size 2 --validate_epoch 30 --cv 0 --stage 1
#       python train.py --exp_name TRAIN --gpus 1 --gpu_4g_limit True --num_epochs 301 --batch_size 2 --validate_epoch 30 --cv 0 --stage 2 --prob_dir {data_dir}/output/run_TRAIN/exp_4
####################################################################################################
import os, sys
from subprocess import Popen, PIPE
from utils import date_time_stamp

# Args
exp_name = date_time_stamp()
gpus = 1
gpu_4g_limit = "True"
num_epochs = 2
batch_size = 2
validate_epoch = 1

# Loop through all cv folds
for cv in range(5):
    prob_dir = "D:\\Data\\Vessels\\output\\run_%s\\exp_%d" % (exp_name, cv)    
    
    call = "python train.py --exp_name %s --gpus %d --gpu_4g_limit %s --num_epochs %d --batch_size %d \
        --validate_epoch %d --cv %d --stage 1" % (exp_name,gpus,gpu_4g_limit,num_epochs,batch_size,validate_epoch,cv)
    print("%s\n\n" % call)
    p = Popen(call.split(), stderr = PIPE, stdout = PIPE)
    out, err = p.communicate()
    print("OUTPUT: %s" % out)
    print("ERROR: %s" % err)
    
    call = "python train.py --exp_name %s --gpus %d --gpu_4g_limit %s --num_epochs %d --batch_size %d \
        --validate_epoch %d --cv %d --stage 2 --prob_dir %s" % (exp_name,gpus,gpu_4g_limit,num_epochs,batch_size,
        validate_epoch,cv,prob_dir)
    print("%s\n\n" % call)
    p = Popen(call.split(), stderr = PIPE, stdout = PIPE)
    out, err = p.communicate()
    print("OUTPUT: %s" % out)
    print("ERROR: %s" % err)