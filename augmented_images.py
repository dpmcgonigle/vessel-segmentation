import os, sys              
import numpy as np          
import time                 
import argparse             #   parse command-line arguments
import subprocess           #   used for generating/executing subprocesses
import json                 #   handles json-formatted data
from pprint import pprint

# User-defined
from utils import randseed, filepath_to_name, get_memory, print_d, str2bool, normalize_image, augment_imageset
import utils
from data_loader import load_train_test_images, dtype_0_1
from MobileUNet import MobileUNet
from eval_utils import AUC_ROC, AUC_PR, eval_metrics

# Pytorch
import torch
import torch.nn as nn

# Misc modelling and imaging
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2                  #   image-handling functionality

################################### VARIABLES ###################################
STAGE_1_INPUT_CHANNELS = 1
STAGE_2_INPUT_CHANNELS = 3

############################### COMMAND-LINE ARGS ###############################
def get_args():
    """
    returns command-line options from argparse
    """
    parser = argparse.ArgumentParser()
    # do we need these? Not implemented yet.
    # parser.add_argument('--patch_size', type=int, default=384)
    # parser.add_argument('--full_size', type=int, default=512)
    # execution information
    parser.add_argument('--exp_name', type=str, help="Exp name will be used as dir name in data_dir")
    parser.add_argument('--gpu', type=str, default=0, help="0 for GPU device 0, 1 for GPU device 1, -1 for CPU")
    parser.add_argument('--gpu_4g_limit', default=1, type=str2bool, help="set True to shrink MobileUNet, allowing batch size of 2 with 512 x 512 images")
    parser.add_argument('--data_dir', type=str, default="D:\\Data\\Vessels") # expects directory 'training'
    parser.add_argument('--prob_dir', type=str) # expects directory 'probability_maps'; defaults to exp_dir
    # hyper-parameters
    parser.add_argument('--epochs_per_stage', type=int, default=600)
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--validate_epoch', type=int, default=30)
    parser.add_argument('--cv', type=int, default=0) # cross validation, CV=5
    # data augmentation
    parser.add_argument('--augment_data', type=str2bool, default=True, help="Turns on data augmentation(flip, rotate, translate, noise, tophat)")
    parser.add_argument('--augmentation_threshold', type=float, default=0.25, help="randomly perform one of the augmentation procedures this % of the time")
    parser.add_argument('--expand_dataset', type=int, default=5, help="multiply dataset by this number with data augmentation")
    parser.add_argument('--flip', type=str2bool, default=True)
    parser.add_argument('--rotate', type=str2bool, default=True)
    parser.add_argument('--translate', type=str2bool, default=True)
    parser.add_argument('--tophat', type=str2bool, default=True)
    parser.add_argument('--noise', type=str2bool, default=True)
    # Model
    parser.add_argument('--model', type=str, default="MobileUNet-Skip", help="MobileUNet, MobileUNet-Skip")
    parser.add_argument('--load_model', type=str, default=None, help="load [model].pth params, default to None to start fresh")
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--stage', type=int, default=3, help="1 for first stage, 2 for second, 3 for both")
    # If you want to call this get_args() from a Jupyter Notebook, you need to uncomment -f line. Them's the rules.
    # parser.add_argument('-f', '--file', help='Path for input file.')
    return parser.parse_args()

############################### SAVE_IMAGES ###############################
def save_images(args, dirs):
    """
    Save the augmented images to make sure they look right.
    args should be ~ get_args()
    dirs should be a tuple of the experiment dir, stage directory within experiment, checkpoint dir within stage dir
    """
    #
    #   Training variables
    #
    start_epoch = args.start_epoch
    epochs = args.epochs_per_stage
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    print("\nPreparing the model for training on stage %d..." % stage)
    
    # Unpack dirs
    exp_dir, stage_exp_dir, stage_checkpoint_exp_dir = dirs

    cmap = "gray" if stage==1 else None
        
    stage_exp_dir_epoch = os.path.join(stage_exp_dir)
    if not os.path.isdir(stage_exp_dir_epoch):
        os.makedirs(stage_exp_dir_epoch)
    data, filenames = load_train_test_images(data_dir=args.data_dir, prob_dir=args.prob_dir, 
        cv=args.cv, stage=stage)
    
    #
    #   Unpack Images from data; all in (n x c x h x w) format
    #
    train_x_imgs = data["train_x_images"]
    train_y_imgs = data["train_y_images"]
    test_x_imgs = data["test_x_images"]
    test_y_imgs = data["test_y_images"]
    train_filenames  = filenames["train_filenames"]
    test_filenames = filenames["test_filenames"]

    # Store params
    num_train_imgs = train_x_imgs.shape[0]
    num_test_imgs = test_x_imgs.shape[0]
    img_height = train_x_imgs.shape[2]
    img_width = train_x_imgs.shape[3]
    
    print("Number of training images: %d, testing images: %d" % (num_train_imgs, num_test_imgs))
    print("Shape of training images: %s, testing images: %s" % (str(train_x_imgs.shape), str(test_x_imgs.shape)))
    print("\n")
        
    # initialize validation arrays - used to report on statistics of the model
    # Will use np.vstack to add prediction maps to this
    np_train_x_preds = np.empty((0, img_height, img_width), float)
    
    #
    #   Image transformations, if applicable
    #
    input_image_batch = []
    output_image_batch = []
    image_count=1
    for img_index in tqdm(range(0, num_train_imgs)):  
        # Starting images
        input_image = train_x_imgs[img_index].copy()
        output_image = train_y_imgs[img_index].copy()
        
        # Data augmentation
        if args.augment_data:
            for i in range(args.expand_dataset):
                augmented_x, augmented_y = augment_imageset(input_image, output_image,
                    probability_threshold=args.augmentation_threshold, flip=args.flip, rotate=args.rotate,
                    translate=args.translate, tophat=args.tophat, noise=args.noise)
                # Make sure the arrays are not the same
                if not np.array_equal(input_image, augmented_x):
                    #
                    #   Save training images; transpose input image shape to (c x h x w) for imsave (rgb) if stage==2
                    #
                    augmented_x = np.squeeze(np.transpose(augmented_x, axes=(1,2,0))).astype(np.uint8)
                    augmented_y = np.squeeze(augmented_y).astype(np.uint8)
                    plt.imsave(arr=augmented_x,
                               fname=os.path.join(stage_exp_dir_epoch, 
                               "thresh%.0f"%(args.augmentation_threshold * 100), 
                               "%s_%d_x.png" % (train_filenames[img_index], i)), 
                               cmap=cmap)
                    plt.imsave(arr=augmented_y,
                               fname=os.path.join(stage_exp_dir_epoch, 
                               "thresh%.0f"%(args.augmentation_threshold * 100), 
                               "%s_%d_y.png" % (train_filenames[img_index], i)), 
                               cmap='gray')
                    image_count+=1
                    
                    
      
    print("Augmented image_count: %d" % image_count, end=' ')

############################### MAAAAAAAAINS ###############################
if __name__ == "__main__":
    print("main(): starting program at %s" % utils.date_time_stamp())
    args = get_args()
    pprint(args)

    # Provide experiment name if none is provided
    if args.exp_name is None:
        args.exp_name = utils.date_time_stamp()

    num_classes = 2
    
    #
    #   If args.stage is 3, run both stages
    #
    start_stage = args.stage if args.stage < 3 else 1
    end_stage = args.stage if args.stage < 3 else 2
    for stage in range(start_stage, end_stage + 1): 
    
        # Make directories for output
        exp_dir = os.path.join(args.data_dir, "output", "run_%s" % args.exp_name, "exp_%d" % args.cv)
        stage_exp_dir = os.path.join(exp_dir, "stage_%d" % stage)
        stage_checkpoint_exp_dir = os.path.join(exp_dir, "stage_checkpoint_%d" % stage)
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir)
        if not os.path.isdir(stage_exp_dir):
            os.makedirs(stage_exp_dir)
        if not os.path.isdir(stage_checkpoint_exp_dir):
            os.makedirs(stage_checkpoint_exp_dir)
        if not os.path.isdir(os.path.join(stage_exp_dir, "thresh%.0f"%(args.augmentation_threshold * 100))):    
            os.makedirs(os.path.join(stage_exp_dir, "thresh%.0f"%(args.augmentation_threshold * 100)))
        # Pack directories for training
        dirs = (exp_dir, stage_exp_dir, stage_checkpoint_exp_dir)

        #
        #   Training - save the model on keyboard interrupt
        #
        try:        
            save_images(args, dirs)
        except KeyboardInterrupt:
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
    print("\n\nmain(): ending program at %s" % utils.date_time_stamp())
