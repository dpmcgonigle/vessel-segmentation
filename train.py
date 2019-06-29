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
    parser.add_argument('--expand_dataset', type=int, default=5, help="multiply dataset by this number of images per real image data augmentation")
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

############################### TRAIN NETWORK ###############################
def train_network(network, args, dirs, stage):
    """
    Train the model for any stage of the full model cycle.
    As of 6/16/2019, this is what the model looks like:
        Stage 1 => Process grayscale image through 11-Layer Mobile-U-Net for (num_classes) output maps
        Stage 2 => (grayscale img x, Stage 1 output, Stage 1 edge map) through same Mobile-U-Net for (n_classes) o/p maps
    network should be the initialized Mobile-U-Net (or any network that produces [N x num_classes x H x W] torch output)
    args should be ~ get_args()
    dirs should be a tuple of the experiment dir, stage directory within experiment, checkpoint dir within stage dir
    The learning rate with the Mobile-U-Net, Adam, and CrossEntropy loss seems to work well with 0.0001 on vessel imgs
    """
    #
    #   Training variables
    #
    start_epoch = args.start_epoch
    epochs = args.epochs_per_stage
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    
    # Used for graphing - saved every validate_epoch epochs
    training_losses, testing_losses = np.empty((1,0)), np.empty((1,0))
    training_evaluation_metrics = np.empty((6,0)) # of rows are auc_roc, auc_pr, dice_coef, acc, sens, spec
    testing_evaluation_metrics = np.empty((6,0)) # of rows are auc_roc, auc_pr, dice_coef, acc, sens, spec

    print("\nPreparing the model for training on stage %d..." % stage)
    
    # Unpack dirs
    exp_dir, stage_exp_dir, stage_checkpoint_exp_dir = dirs
    exp_prob_dir = os.path.join(exp_dir, "probability_maps")
    # If we're in stage 1, need to create the directory for probability maps
    if stage == 1:
        if not os.path.isdir(exp_prob_dir):
            os.makedirs(exp_prob_dir)
    # If no prob_dir is specified, set it equal to the exp_dir
    if args.prob_dir is None:
        args.prob_dir = exp_dir
            
    # Save the command-line arguments for this stage and the model
    json.dump(args.__dict__, open(os.path.join(stage_exp_dir, "config.txt"), "w"), indent=4)
    model_checkpoint_name = os.path.join(stage_exp_dir, "latest_model_" + args.model + ".ckpt")

    # instantiate loss function
    criterion = nn.CrossEntropyLoss(reduction='mean').cuda(int(args.gpu))
    
    # instantiate optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    utils.count_params(network)

    print("[x] begining training ...")
    global_evaluation = 0.  # DICE
    global_evaluation_epoch = 0
    cmap = "gray" if stage==1 else None
   
    #
    #   wrap epoch counter in tqdm, which is a progress bar (see https://github.com/tqdm/tqdm)
    #
    for epoch in tqdm(range(start_epoch, epochs+1)):
        # I call this function several times throughout training to ensure that garbage has been collected
        torch.cuda.empty_cache()
        
        # If model.eval() was called previously during validation, we need to turn model.train() back on
        network.train()
        
        # initialize epoch_loss to 0
        epoch_loss = 0
        
        #   load data before training
        print("\nepoch %d loading data ......" % epoch)
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
        input_channels = STAGE_1_INPUT_CHANNELS if stage == 1 else STAGE_2_INPUT_CHANNELS
        
        # print out dataset details
        if epoch == 1:
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
        image_count=0
        for img_index in tqdm(range(0, num_train_imgs)):  
            # Starting images
            input_image = train_x_imgs[img_index].copy()
            output_image = train_y_imgs[img_index].copy()
            
            # Boolean to track whether original input image has already been added to batch (data augmentation)
            original_image_used = False
            
            # Data augmentation
            if args.augment_data:
                for i in range(args.expand_dataset):
                    augmented_x, augmented_y = augment_imageset(input_image, output_image,
                        probability_threshold=args.augmentation_threshold, flip=args.flip, rotate=args.rotate,
                        translate=args.translate, tophat=args.tophat, noise=args.noise)
                        
                    # Make sure the original input image is passed into the image batch a maximum of one time
                    if np.array_equal(input_image, augmented_x):
                        if original_image_used:
                            continue # Don't use this image
                        original_image_used = True
                        
                    input_image_batch.append(normalize_image(augmented_x))
                    output_image_batch.append(normalize_image(augmented_y))
                    image_count += 1
                        
            else:
                #   Add normalized images to batch and turn them into pytorch Variable
                input_image_batch.append(normalize_image(input_image))
                output_image_batch.append(normalize_image(output_image))
                image_count += 1

        print("Total image_count (including augmentation, if applicable): %d" % image_count)

        # Need to cast the images to the correct float type for torch FloatTensor to work
        input_image_batch = np.array(input_image_batch, dtype=dtype_0_1())
        output_image_batch = np.array(output_image_batch, dtype=dtype_0_1())
        
        for mini_batch in range(int(np.ceil(image_count / batch_size))):
            training_batch_x = input_image_batch[mini_batch*batch_size : (mini_batch+1)*batch_size]
            training_batch_y = output_image_batch[mini_batch*batch_size : (mini_batch+1)*batch_size]
            torch_training_batch_x = torch.from_numpy(training_batch_x)
            print_d("torch_training_batch_x shape: %s" % str(torch_training_batch_x.shape))
            torch_training_batch_y = torch.from_numpy(training_batch_y)
            print_d("torch_training_batch_y shape: %s" % str(torch_training_batch_y.shape))
            
            if int(args.gpu) >= 0:
                torch_training_batch_x = torch_training_batch_x.cuda(int(args.gpu))
                torch_training_batch_y = torch_training_batch_y.cuda(int(args.gpu))

            #
            #   Training forward pass
            #
            batch_predictions = network.forward(torch_training_batch_x)
            print_d("batch_predictions shape: %s" % str(batch_predictions.shape))
                
            # target needs to be type long, with no singular dimensions
            loss = criterion(batch_predictions, torch.squeeze(torch_training_batch_y,1).long()) # input, target
            
            epoch_loss += float(loss.detach().cpu())
            
            print('\r Epoch {0:d} --- {1:.2f}% complete --- mini-batch loss: {2:.6f}'.format(epoch, mini_batch * batch_size / image_count * 100, loss.item()), end=' ')
            
            #
            #   Zero out gradients and then back-propogate & step forward
            #
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

        print('\rEpoch {} training finished ! Loss: {}'.format(epoch, epoch_loss / (image_count / np.ceil(batch_size))))
            
        #
        #   validate on training and validation data sets
        #
        if epoch % int(args.validate_epoch) == 0 or epoch == 1:
        
            # Turn off batch norm, dropout, and don't worry about storing computation graph
            network.eval()
        
            #
            # Create directories and files
            #
            stage_epoch_exp_dir = os.path.join(stage_exp_dir, "%04d" % (epoch))
            stage_epoch_exp_train_dir = os.path.join(stage_epoch_exp_dir, "train")
            stage_epoch_exp_test_dir = os.path.join(stage_epoch_exp_dir, "test")
            if not os.path.isdir(stage_epoch_exp_dir):
                os.makedirs(stage_epoch_exp_dir)
            if not os.path.isdir(stage_epoch_exp_train_dir):
                os.makedirs(stage_epoch_exp_train_dir)
            if not os.path.isdir(stage_epoch_exp_test_dir):
                os.makedirs(stage_epoch_exp_test_dir)
            with open(os.path.join(stage_epoch_exp_dir, "val_score.csv"), "w") as target:
                # sn = sensitivity; sp = specificity
                target.write("filename,auc_roc,auc_pr,dice,acc,sn,sp\n")
                target.write("training\n")
                
                #
                #   Iterate through training images to save and record metrics for
                #
                print("Saving training images and data from validation epoch %d" % epoch)
                avg_auc_roc , avg_auc_pr , avg_dice , avg_acc , avg_sn , avg_sp = [], [], [], [], [], []
                train_loss = 0
                for img_index in tqdm(range(num_train_imgs)):
                    with torch.no_grad():
                        torch_input_image = torch.from_numpy(np.expand_dims(normalize_image(train_x_imgs[img_index].copy()), 0))
                        torch_label_image = torch.from_numpy(np.expand_dims(normalize_image(train_y_imgs[img_index].copy()), 0))
                        if int(args.gpu) >= 0:
                            torch_input_image = torch_input_image.cuda(int(args.gpu))
                            torch_label_image = torch_label_image.cuda(int(args.gpu))
                        
                        # Run validation images through network
                        pred_image = network.forward(torch_input_image)
                        
                        # Add image loss to train_loss
                        train_loss += criterion(pred_image, torch.squeeze(torch_label_image,1).long()).item() # input, target
                        
                        # argmax is used to compress the classification predictions down to a class prediction map
                        prediction_map = np.argmax(pred_image.detach().cpu().numpy(), 1)
                        del pred_image, torch_input_image # Trying to free up space 
                    
                    # Remove singular dimensions from train_y_imgs (the number of channels in [n x c x h x w])
                    train_y_arr = np.round(np.squeeze(normalize_image(train_y_imgs[img_index].copy(), dtype=np.uint8)))
                    
                    # 
                    #   Save training metrics
                    #
                    train_filename = train_filenames[img_index]
                    auc_roc = AUC_ROC(train_y_arr, prediction_map)
                    auc_pr = AUC_PR(train_y_arr, prediction_map)
                    dice_coef, acc, sens, spec = eval_metrics(train_y_arr, prediction_map)
                    
                    avg_auc_roc.append(auc_roc)
                    avg_auc_pr.append(auc_pr)
                    avg_dice.append(dice_coef)
                    avg_acc.append(acc)
                    avg_sn.append(sens)
                    avg_sp.append(spec)  

                    target.write("%s,%f,%f,%f,%f,%f,%f\n" % (train_filename, auc_roc, auc_pr, dice_coef, acc, sens, spec))

                    #
                    #   Save training images; transpose input image shape to (c x h x w) for imsave (rgb) if stage==2
                    #
                    input_image = np.squeeze(np.transpose(train_x_imgs[img_index], axes=(1,2,0))).astype(np.uint8)
                    output_image = (np.squeeze(prediction_map)*255.0).astype(np.uint8)
                    label_image = np.squeeze(train_y_imgs[img_index]).astype(np.uint8)
                    plt.imsave(arr=input_image,
                               fname=os.path.join(stage_epoch_exp_train_dir, "%s_x.png" % (train_filenames[img_index])), 
                               cmap=cmap)
                    plt.imsave(arr=output_image,
                               fname=os.path.join(stage_epoch_exp_train_dir, "%s_pred.png" % (train_filenames[img_index])),
                               cmap='gray')
                    plt.imsave(arr=label_image,
                               fname=os.path.join(stage_epoch_exp_train_dir, "%s_y.png" % (train_filenames[img_index])), 
                               cmap='gray')
                               
                #   Save training loss
                training_losses = np.append(training_losses, (train_loss / num_train_imgs))
                
                #
                #   Save training evaluation scores
                #
                temp_array = np.array([np.mean(avg_auc_roc),np.mean(avg_auc_roc),
                    np.mean(avg_dice),np.mean(avg_acc),np.mean(avg_sn),np.mean(avg_sp)]).reshape(6,1)
                training_evaluation_metrics = np.hstack([training_evaluation_metrics, temp_array])
                
                #
                #   Iterate through validation images to save and record metrics for
                #
                print("\n[x] testing on validation set for epoch %d" % epoch)
                avg_auc_roc , avg_auc_pr , avg_dice , avg_acc , avg_sn , avg_sp = [], [], [], [], [], []

                target.write("testing data\n")
                test_loss = 0
                for img_index in tqdm(range(num_test_imgs)):
                    with torch.no_grad():
                        torch_input_image = torch.from_numpy(np.expand_dims(normalize_image(test_x_imgs[img_index].copy()), 0))
                        torch_label_image = torch.from_numpy(np.expand_dims(normalize_image(test_y_imgs[img_index].copy()), 0))
                        
                        if int(args.gpu) >= 0:
                            torch_input_image = torch_input_image.cuda(int(args.gpu))
                            torch_label_image = torch_label_image.cuda(int(args.gpu))
                        
                        # Run validation images through network
                        pred_image = network.forward(torch_input_image)
                        
                        # Add image loss to train_loss
                        test_loss += criterion(pred_image, torch.squeeze(torch_label_image,1).long()).item() # input, target
                        
                        # argmax is used to compress the classification predictions down to a class prediction map
                        prediction_map = np.argmax(pred_image.detach().cpu().numpy(), 1)
                        del pred_image, torch_input_image # Trying to free up space 
                    
                    # Remove singular dimensions from train_y_imgs (the number of channels in [n x c x h x w])
                    test_y_arr = np.round(np.squeeze(normalize_image(test_y_imgs[img_index].copy(), dtype=np.uint8)))
                    print_d("test_y_imgs[img_index] shape: %s" % str(train_y_imgs[img_index].shape))
                    print_d("prediction_map shape: %s" % str(prediction_map.shape))
                    
                    # 
                    #   Save validation metrics
                    #
                    test_filename = test_filenames[img_index]
                    auc_roc = AUC_ROC(test_y_arr, prediction_map)
                    auc_pr = AUC_PR(test_y_arr, prediction_map)
                    dice_coef, acc, sens, spec = eval_metrics(test_y_arr, prediction_map)
                    
                    avg_auc_roc.append(auc_roc)
                    avg_auc_pr.append(auc_pr)
                    avg_dice.append(dice_coef)
                    avg_acc.append(acc)
                    avg_sn.append(sens)
                    avg_sp.append(spec)                    

                    #
                    #   Save validation images; transpose input img to (c x h x w) to save as rgb if stage==2
                    #
                    input_image = np.squeeze(np.transpose(test_x_imgs[img_index], axes=(1,2,0))).astype(np.uint8)
                    output_image = (np.squeeze(prediction_map)*255.0).astype(np.uint8)
                    label_image = np.squeeze(test_y_imgs[img_index]).astype(np.uint8)
                    plt.imsave(arr=input_image,
                               fname=os.path.join(stage_epoch_exp_test_dir, "%s_x.png" % (test_filenames[img_index])), 
                               cmap=cmap)
                    plt.imsave(arr=output_image,
                               fname=os.path.join(stage_epoch_exp_test_dir, "%s_pred.png" % (test_filenames[img_index])),
                               cmap='gray')
                    plt.imsave(arr=label_image,
                               fname=os.path.join(stage_epoch_exp_test_dir, "%s_y.png" % (test_filenames[img_index])), 
                               cmap='gray')                   
            
                #   Save validation loss
                testing_losses = np.append(testing_losses, (test_loss / num_test_imgs))
                
                #
                #   Save validation scores
                #
                temp_array = np.array([np.mean(avg_auc_roc),np.mean(avg_auc_roc),
                    np.mean(avg_dice),np.mean(avg_acc),np.mean(avg_sn),np.mean(avg_sp)]).reshape(6,1)
                testing_evaluation_metrics = np.hstack([testing_evaluation_metrics, temp_array])
                
                print("\nAverage validation accuracy for epoch # %04d" % epoch)
                print("Average per class validation accuracies for epoch # %04d:" % epoch)
                print("Validation auc_roc = ", np.mean(avg_auc_roc))
                print("Validation auc_pr = ", np.mean(avg_auc_pr))
                print("Validation dice = ", np.mean(avg_dice))
                print("Validation acc = ", np.mean(avg_acc))
                print("Validation sn = ", np.mean(avg_sn))
                print("Validation sp = ", np.mean(avg_sp))
                target.write("x,%f,%f,%f,%f,%f,%f\n" % (np.mean(avg_auc_roc), np.mean(avg_auc_pr), np.mean(avg_dice),
                                                        np.mean(avg_acc), np.mean(avg_sn), np.mean(avg_sp)))

            
            #
            #   saving model if the score is better than previous scores
            #
            if np.mean(avg_dice) > global_evaluation and epoch > 0:
                print("\n[x] saving model to %s (highest evaluation score so far)" % model_checkpoint_name)
                global_evaluation = np.mean(avg_dice)
                global_evaluation_epoch = epoch
                
                # Save pytorch model
                torch.save(network.state_dict(), os.path.join(stage_checkpoint_exp_dir, "epoch_%d.pth" % epoch))

                #
                #   save probability map images for stage 2
                #
                if stage==1:
                    
                    # replace old prob images
                    print("\nSaving probability maps from training set (highest evaluation score so far)")
                    for img_index in tqdm(range(num_train_imgs)):
                        torch_input_image = torch.from_numpy(np.expand_dims(normalize_image(train_x_imgs[img_index].copy()), 0))
                        if int(args.gpu) >= 0:
                            torch_input_image = torch_input_image.cuda(int(args.gpu))
                        
                        # Run validation images through network
                        pred_image = network.forward(torch_input_image)
                        
                        # argmax is used to compress the classification predictions down to a class prediction map
                        prediction_map = np.argmax(pred_image.detach().cpu().numpy(), 1)
                        del pred_image, torch_input_image # Trying to free up memoryview
                        
                        #
                        #   Save probability map
                        #
                        output_image = np.squeeze(prediction_map)
                        plt.imsave(arr=(np.squeeze(output_image)*255.0).astype(np.uint8), fname=os.path.join(exp_prob_dir, 
                            "%s_pred.png" % (train_filenames[img_index])), cmap='gray')
                                   
                    print("\nSaving probability maps from validation set (highest evaluation score so far)")
                    for img_index in tqdm(range(num_test_imgs)):
                        torch_input_image = torch.from_numpy(np.expand_dims(normalize_image(test_x_imgs[img_index].copy()), 0))
                        
                        if int(args.gpu) >= 0:
                            torch_input_image = torch_input_image.cuda(int(args.gpu))
                            
                        # Run validation images through network
                        pred_image = network.forward(torch_input_image)
                        
                        # argmax is used to compress the classification predictions down to a class prediction map
                        prediction_map = np.argmax(pred_image.detach().cpu().numpy(), 1)
                        del pred_image, torch_input_image # Trying to free up memoryview
                        
                        #
                        #   Save probability map
                        #
                        output_image = np.squeeze(prediction_map)
                        plt.imsave(arr=(np.squeeze(output_image)*255.0).astype(np.uint8), fname=os.path.join(exp_prob_dir, 
                            "%s_pred.png" % (test_filenames[img_index])), cmap='gray')

        #
        #   global_evaluation is mean dice score, and global_epoch is the epoch in which the best eval occurred
        #
        print("global_evaluation = {}".format(global_evaluation))
        print("global_epoch = {}".format(global_evaluation_epoch))
        
    #
    #   MAKE IMAGES! First Loss function, then evaluation metrics
    #
    # x is for x axis (epochs) in the metrics charts
    x = [start_epoch] + [x for x in range(args.validate_epoch, args.epochs_per_stage+1, args.validate_epoch)]
    
    plt.figure()
    plt.plot(x, training_losses.ravel(), label="Training Losses")
    plt.plot(x, testing_losses.ravel(), label="Validation Losses")
    plt.title("Stage %d Loss" % (stage))
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(stage_exp_dir, "losses.png"))
    
    metrics = ['auc_roc', 'auc_pr', 'dice_coef', 'acc', 'sens', 'spec']
    for i in range(len(metrics)):
        plt.figure()
        plt.plot(x, training_evaluation_metrics[i], label="Training %s"%metrics[i])
        plt.plot(x, testing_evaluation_metrics[i], label="Validation %s"%metrics[i])    
        plt.title("Stage %d %s" % (stage, metrics[i]))
        plt.ylabel("%s Value"%metrics[i])
        plt.xlabel("Epoch")
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(stage_exp_dir, "%s.png"%metrics[i]))
    
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
        
        # Pack directories for training
        dirs = (exp_dir, stage_exp_dir, stage_checkpoint_exp_dir)
        
        # Set model input channels
        if stage == 1:
            input_channels = STAGE_1_INPUT_CHANNELS # Grayscale images
        elif stage == 2:
            input_channels = STAGE_2_INPUT_CHANNELS # Grayscale images + Canny edge map + stage 1 output_image
        else:
            raise ValueError("stage error")

        # instantiate model     
        network = MobileUNet(input_channels, preset_model=args.model, num_classes=num_classes, 
            gpu=args.gpu, gpu_4g_limit=args.gpu_4g_limit)

        # assign network to cpu or gpu, and load parameters if applicable
        if int(args.gpu) >= 0:
            print("Using CUDA version of the network, prepare your GPU !")
            network.cuda(int(args.gpu))
            if args.load_model is not None:
                network.load_state_dict(torch.load(args.load_model))
        else:
            print("Using CPU version of the net, this may be very slow")
            network.cpu()
            if args.load_model is not None:
                network.load_state_dict(torch.load(args.model, map_location='cpu'))

        #
        #   Training - save the model on keyboard interrupt
        #
        try:        
            train_network(network, args, dirs, stage)
            del network
            torch.cuda.empty_cache()
        except KeyboardInterrupt:
            torch.save(network.state_dict(), os.path.join(stage_checkpoint_exp_dir, "INTERRUPTED.pth"))
            print("Saved interrupt")
            print("main(): ending program at %s" % utils.date_time_stamp())
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
    print("\n\nmain(): ending program at %s" % utils.date_time_stamp())
