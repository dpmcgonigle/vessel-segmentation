import os, sys              
import numpy as np          
import time                 
import argparse             #   parse command-line arguments
import subprocess           #   used for generating/executing subprocesses
import json                 #   handles json-formatted data
from pprint import pprint

# User-defined
from utils import randseed, filepath_to_name, get_memory, print_d, str2bool
import utils
from data_loader import load_train_test_images
from MobileUNet import MobileUNet
from eval_utils import AUC_ROC, AUC_PR, dice_coefficient, accuracy, sensitivity, specificity

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
    parser.add_argument('--gpus', type=str, default=1, help="0 for no GPU, 1 for single GPU, 2 for double...")
    parser.add_argument('--gpu_4g_limit', default=1, type=str2bool, help="set True to shrink MobileUNet, allowing batch size of 2 with 512 x 512 images")
    parser.add_argument('--data_path', type=str, default="D:\\Data\\Vessels") # expects directory 'training'
    parser.add_argument('--prob_dir', type=str, default="D:\\Data\\Vessels") # expects directory 'probability_maps'    
    # hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=1001)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--validate_epoch', type=int, default=25)
    parser.add_argument('--cv', type=int, default=0) # cross validation, CV=5
    # data augmentation - not implemented yet
    # parser.add_argument('--h_flip', type=str2bool, default=True)
    # parser.add_argument('--v_flip', type=str2bool, default=True)
    # parser.add_argument('--rotation', type=float, default=30)
    # parser.add_argument('--wb_contrast', type=str2bool, default=True) # for stage 1, wb_contrast=True
    # parser.add_argument('--stride', type=int, default=32)
    # Model
    parser.add_argument('--model', type=str, default="MobileUNet-Skip", help="MobileUNet, MobileUNet-Skip")
    # --load_model has not yet been implemented
    # parser.add_argument('--load_model', type=str, default=None, help="load [model].pth params, default to None to start fresh")
    parser.add_argument('--stage', type=int, default=1)
    # If you want to call this get_args() from a Jupyter Notebook, you need to uncomment -f line. Them's the rules.
    # parser.add_argument('-f', '--file', help='Path for input file.')
    return parser.parse_args()

############################### TRAIN NETWORK ###############################
def train_network(network, args, dirs, start_epoch = 0, epochs=5, batch_size=1, learning_rate=0.0001, stage=1):
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

    print("Preparing the model for training ...")
    
    # Unpack dirs
    exp_dir, stage_exp_dir, stage_checkpoint_exp_dir = dirs
    if stage==1:
        if not os.path.isdir(args.prob_dir):
            os.makedirs(args.prob_dir)
    
    # instantiate loss function
    #   Orig func tf.reduce_mean(  tf.nn.softmax_cross_entropy_with_logits_v2(logits=network, labels=net_output)  )
    #   Trying CrossEntropyLoss2d from github
    criterion = nn.CrossEntropyLoss(reduction='mean').cuda()
    #criterion = CrossEntropyLoss2d()
    
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
    for epoch in tqdm(range(start_epoch, epochs)):
        torch.cuda.empty_cache()
        # initialize epoch_loss to 0
        epoch_loss = 0
        
        #   load data before training
        print("\nepoch %d loading data ......" % (epoch + 1))
        data, filenames = load_train_test_images(data_dir=args.data_path, prob_dir=args.prob_dir, 
            cv=args.cv, stage=args.stage)
        
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
        
        if epoch == 0:
            print("Number of training images: %d, testing images: %d" % (num_train_imgs, num_test_imgs))
            print("Shape of training images: %s, testing images: %s" % (str(train_x_imgs.shape), str(test_x_imgs.shape)))
            print("\n")
            
        # initialize validation arrays - used to report on statistics of the model
        # Will use np.vstack to add prediction maps to this
        np_train_x_preds = np.empty((0, img_height, img_width), float)
        
        #
        #   Image transformations
        #
        input_image_batch = []
        output_image_batch = []
        for img_index in tqdm(range(0, num_train_imgs)):            
            #
            #   McGonigle - This is where Chen implemented data augmentation, cropping, etc
            #       Plan is to modularize those functionalities
            #       Using full images for now
            #
            
            #
            #   Add normalized images to batch and turn them into pytorch Variable
            #
            input_image = train_x_imgs[img_index] / 255.0
            output_image = train_y_imgs[img_index] / 255.0
            input_image_batch.append(input_image)
            output_image_batch.append(output_image)

        input_image_batch = np.array(input_image_batch)
        output_image_batch = np.array(output_image_batch)
        
        for mini_batch in range(int(np.ceil(num_train_imgs / batch_size))):
            training_batch_x = input_image_batch[mini_batch*batch_size : (mini_batch+1)*batch_size]
            training_batch_y = output_image_batch[mini_batch*batch_size : (mini_batch+1)*batch_size]
            torch_training_batch_x = torch.from_numpy(training_batch_x)
            print_d("torch_training_batch_x shape: %s" % str(torch_training_batch_x.shape))
            torch_training_batch_y = torch.from_numpy(training_batch_y)
            print_d("torch_training_batch_y shape: %s" % str(torch_training_batch_y.shape))
            
            if args.gpus > 0:
                torch_training_batch_x = torch_training_batch_x.cuda()
                torch_training_batch_y = torch_training_batch_y.cuda()

            #
            #   Training forward pass
            #
            batch_predictions = network.forward(torch_training_batch_x)
            print_d("batch_predictions shape: %s" % str(batch_predictions.shape))

            #
            #   Save the output if this is a validation epoch
            #
            if epoch % args.validate_epoch == 0:
                # argmax is used to compress the classification predictions down to a class prediction map
                batch_prediction_maps = np.argmax(batch_predictions.detach().cpu().numpy(), 1)
                print_d("batch_prediction_maps shape: %s" % str(batch_prediction_maps.shape))
                
                # stack the prediction map batch vertically to np_train_x_preds
                np_train_x_preds = np.vstack((np_train_x_preds, batch_prediction_maps))
                
            # target needs to be type long, with no singular dimensions
            loss = criterion(batch_predictions, torch.squeeze(torch_training_batch_y,1).long()) # input, target
            print_d("mini-batch loss %.03f" % (loss.cpu()))
            
            epoch_loss += float(loss.detach().cpu())
            
            print('\r Epoch {0:d} --- {1:.2f}% complete --- mini-batch loss: {2:.6f}'.format(epoch + 1, mini_batch * batch_size / num_train_imgs * 100, loss.item()), end=' ')
            
            #
            #   Zero out gradients and then back-propogate & step forward
            #
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

        print('\rEpoch {} training finished ! Loss: {}'.format(epoch + 1, epoch_loss / (num_train_imgs / np.ceil(batch_size))))
            
        #
        #   validate on training and validation data sets
        #
        if epoch % args.validate_epoch == 0:
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
                print("Saving training images and data from validation epoch %d" % (epoch+1))
                for img_index in tqdm(range(num_train_imgs)):
                    # Remove singular dimensions from train_y_imgs (the number of channels in [n x c x h x w])
                    train_y_img = np.squeeze(train_y_imgs[img_index])
                    print_d("train_y_img shape: %s" % str(train_y_img.shape))
                    
                    # 
                    #   Save training metrics
                    #
                    train_filename = train_filenames[img_index]
                    auc_roc = AUC_ROC(train_y_img, np_train_x_preds[img_index])
                    auc_pr = AUC_PR(train_y_img, np_train_x_preds[img_index])
                    dice_coef = dice_coefficient(train_y_img, np_train_x_preds[img_index])
                    acc = accuracy(train_y_img, np_train_x_preds[img_index])
                    sens = sensitivity(train_y_img, np_train_x_preds[img_index])
                    spec = specificity(train_y_img, np_train_x_preds[img_index])

                    target.write("%s,%f,%f,%f,%f,%f,%f\n" % (train_filename, auc_roc, auc_pr, dice_coef, acc, sens, spec))

                    #
                    #   Save training images; transpose input image shape to (c x h x w) for imsave (rgb) if stage==2
                    #
                    input_image = np.squeeze(np.transpose(train_x_imgs[img_index], axes=(1,2,0))).astype(int)
                    output_image = (np.squeeze(np_train_x_preds[img_index])*255.0).astype(int)
                    label_image = np.squeeze(train_y_img).astype(int)
                    plt.imsave(arr=input_image,
                               fname=os.path.join(stage_epoch_exp_train_dir, "%s_x.png" % (train_filenames[img_index])), 
                               cmap=cmap)
                    plt.imsave(arr=output_image,
                               fname=os.path.join(stage_epoch_exp_train_dir, "%s_pred.png" % (train_filenames[img_index])),
                               cmap='gray')
                    plt.imsave(arr=label_image,
                               fname=os.path.join(stage_epoch_exp_train_dir, "%s_y.png" % (train_filenames[img_index])), 
                               cmap='gray')

                #
                #   Iterate through validation images to save and record metrics for
                #
                print("\n[x] testing on validation set")
                avg_auc_roc = avg_auc_pr = avg_dice = avg_acc = avg_sn = avg_sp = []

                target.write("testing data\n")

                for img_index in tqdm(range(num_test_imgs)):
                    torch_input_image = torch.from_numpy(np.expand_dims(test_x_imgs[img_index] / 255.0, 0))
                    if args.gpus > 0:
                        torch_input_image = torch_input_image.cuda()
                        
                    output_image = test_y_imgs[img_index] / 255.0
                    
                    # Run validation images through network
                    pred_image = network.forward(torch_input_image)
                    
                    # argmax is used to compress the classification predictions down to a class prediction map
                    prediction_map = np.argmax(pred_image.detach().cpu().numpy(), 1)
                    del pred_image, torch_input_image # Trying to free up space 
                    
                    # Remove singular dimensions from train_y_imgs (the number of channels in [n x c x h x w])
                    test_y_img = np.squeeze(test_y_imgs[img_index])
                    print_d("test_y_imgs[img_index] shape: %s" % str(train_y_imgs[img_index].shape))
                    print_d("prediction_map shape: %s" % str(prediction_map.shape))
                    
                    # 
                    #   Save validation metrics
                    #
                    test_filename = test_filenames[img_index]
                    auc_roc = AUC_ROC(test_y_img, prediction_map)
                    auc_pr = AUC_PR(test_y_img, prediction_map)
                    dice_coef = dice_coefficient(test_y_img, prediction_map)
                    acc = accuracy(test_y_img, prediction_map)
                    sens = sensitivity(test_y_img, prediction_map)
                    spec = specificity(test_y_img, prediction_map)
                    
                    avg_auc_roc.append(auc_roc)
                    avg_auc_pr.append(auc_pr)
                    avg_dice.append(dice_coef)
                    avg_acc.append(acc)
                    avg_sn.append(sens)
                    avg_sp.append(spec)                    

                    #
                    #   Save validation images; transpose input img to (c x h x w) to save as rgb if stage==2
                    #
                    input_image = np.squeeze(np.transpose(test_x_imgs[img_index], axes=(1,2,0))).astype(int)
                    output_image = (np.squeeze(prediction_map)*255.0).astype(int)
                    label_image = np.squeeze(test_y_img).astype(int)
                    plt.imsave(arr=input_image,
                               fname=os.path.join(stage_epoch_exp_test_dir, "%s_x.png" % (test_filenames[img_index])), 
                               cmap=cmap)
                    plt.imsave(arr=output_image,
                               fname=os.path.join(stage_epoch_exp_test_dir, "%s_pred.png" % (test_filenames[img_index])),
                               cmap='gray')
                    plt.imsave(arr=label_image,
                               fname=os.path.join(stage_epoch_exp_test_dir, "%s_y.png" % (test_filenames[img_index])), 
                               cmap='gray')                   
            
                #
                #   Save validation scores
                #
                print("\nAverage validation accuracy for epoch # %04d" % (epoch))
                print("Average per class validation accuracies for epoch # %04d:" % (epoch))
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
                print("\n[x] saving model to %s" % model_checkpoint_name)
                global_evaluation = np.mean(avg_dice)
                global_evaluation_epoch = epoch
                
                # Save pytorch model
                torch.save(network.state_dict(), os.path.join(stage_checkpoint_exp_dir, "epoch_%d.pth" % (epoch+1)))

                #
                #   save probability map images for stage 2
                #
                if stage==2:
                    
                    # replace old prob images
                    print("\nSaving images in training set")
                    for img_index in tqdm(range(num_train_imgs)):
                        torch_input_image = torch.from_numpy(np.expand_dims(train_x_imgs[img_index] / 255.0, 0))
                        if args.gpus > 0:
                            torch_input_image = torch_input_image.cuda()
                            
                        output_image = train_y_imgs[img_index] / 255.0
                        
                        # Run validation images through network
                        pred_image = network.forward(torch_input_image)
                        
                        # argmax is used to compress the classification predictions down to a class prediction map
                        prediction_map = np.argmax(pred_image.detach().cpu().numpy(), 1)
                        del pred_image, torch_input_image # Trying to free up memoryview
                        
                        #
                        #   Save probability map
                        #
                        output_image = np.squeeze(prediction_map)
                        plt.imsave(arr=(np.squeeze(output_image)*255.0).astype(int),
                                   fname=os.path.join(args.prob_dir, "%s_pred.png" % (train_filenames[img_index])), cmap='gray')
                                   
                    print("\nSaving images in validation set")
                    for img_index in tqdm(range(num_test_imgs)):
                        torch_input_image = torch.from_numpy(np.expand_dims(test_x_imgs[img_index] / 255.0, 0))
                        
                        if args.gpus > 0:
                            torch_input_image = torch_input_image.cuda()
                            
                        output_image = test_y_imgs[img_index] / 255.0
                        
                        # Run validation images through network
                        pred_image = network.forward(torch_input_image)
                        
                        # argmax is used to compress the classification predictions down to a class prediction map
                        prediction_map = np.argmax(pred_image.detach().cpu().numpy(), 1)
                        del pred_image, torch_input_image # Trying to free up memoryview
                        
                        #
                        #   Save probability map
                        #
                        output_image = np.squeeze(prediction_map)
                        plt.imsave(arr=(np.squeeze(output_image)*255.0).astype(int),
                                   fname=os.path.join(args.prob_dir, "%s_pred.png" % (test_filenames[img_index])), cmap='gray')

        
        print("global_evaluation = {}".format(global_evaluation))
        print("global_epoch = {}".format(global_evaluation_epoch))

############################### MAAAAAAAAINS ###############################
if __name__ == "__main__":
    print("main(): starting program at %s" % utils.date_time_stamp())
    args = get_args()
    pprint(args)

    #os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpus)
    
    # Make directories for output
    exp_dir = os.path.join(args.data_path, "output", "run_%s" % utils.date_time_stamp(), "exp_%d" % args.cv)
    stage_exp_dir = os.path.join(exp_dir, "stage_%d" % args.stage)
    stage_checkpoint_exp_dir = os.path.join(exp_dir, "stage_checkpoint_%d" % args.stage)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    if not os.path.isdir(stage_exp_dir):
        os.makedirs(stage_exp_dir)
    if not os.path.isdir(stage_checkpoint_exp_dir):
        os.makedirs(stage_checkpoint_exp_dir)
        
    # Pack directories for training
    dirs = (exp_dir, stage_exp_dir, stage_checkpoint_exp_dir)

    json.dump(args.__dict__, open(os.path.join(stage_exp_dir, "config.txt"), "w"), indent=4)
    model_checkpoint_name = os.path.join(stage_exp_dir, "latest_model_" + args.model + ".ckpt")

    num_classes = 2

    if args.stage == 1:
        input_channels = STAGE_1_INPUT_CHANNELS # Grayscale images
    elif args.stage == 2:
        input_channels = STAGE_2_INPUT_CHANNELS # Grayscale images + Canny edge map + stage 1 output_image
    else:
        raise ValueError("args.stage error")

    # instantiate network     
    network = MobileUNet(input_channels, preset_model=args.model, num_classes=num_classes, 
        gpus=args.gpus, gpu_4g_limit=args.gpu_4g_limit)

    # assign network to cpu or gpu, and load parameters if applicable
    if args.gpus > 0:
        print("Using CUDA version of the network, prepare your GPU !")
        network.cuda()
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
        train_network(network, args, dirs, start_epoch = 0, epochs=args.num_epochs, batch_size=args.batch_size, learning_rate=0.0001, stage=args.stage)
    except KeyboardInterrupt:
        torch.save(network.state_dict(), os.path.join(stage_checkpoint_exp_dir, "INTERRUPTED.pth"))
        print("Saved interrupt")
        print("main(): ending program at %s" % utils.date_time_stamp())
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    print("main(): ending program at %s" % utils.date_time_stamp())