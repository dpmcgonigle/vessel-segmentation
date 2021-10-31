# Image Segmentation with two-stage Mobile-U-Net

## Table of Contents
- [Research](#research)
- [Approach](#approach)
- [Setup](#setup)

## Research

### Publication

[A new approach to extracting coronary arteries and detecting stenosis in invasive coronary
angiograms](https://arxiv.org/pdf/2101.09848.pdf)

### Description

This project was created with the goal of image segmentation for fluoroscopic angiograms, and eventually semantic segmentation to identify the specific vessels.  This project was adapted from a Tensorflow version of the model, which was created by Chen Zhao (who in turn had adapted his model from George Seif's Semantic Image Segmentation Suite: https://github.com/GeorgeSeif/Semantic-Segmentation-Suite)

### Poster:

![MCBIOS 2019 Poster](./images/poster.jpg)

[Back to Top](#table-of-contents)

## Approach

Stage 1: Mobile-U-Net to process the original input image, which in our case are 512 x 512 grayscale x-ray images.
Stage 2: Combine the original 512 x 512 grayscale image with the 512 x 512 segmentation prediction map from Stage 1, and a Canny edge detection of the Stage 1 output (3 channels)

### Args
Most of the command-line arguments have defaults built in, so that you can run the program with minimal specification, though I set the defaults to agree with my machine, so have a look-see.

--args(implemented so far): --gpu (which one), --gpu\_4g\_limit (bool; shrink model to fit 4GB GPU with batch\_size 2 for 512x512 images), --data\_path, --exp\_name (experiment name), 
--prob\_dir (directory for probability maps used in Stage 2, which are the output of Stage 1), -epochs\_per\_stage, --batch\_size, --start\_epoch, 
--validate\_epoch (How often to validate training and validation images), --cv-model (MobileUNet-Skip, MobileUNet), --stage (1 or 2, or 3 for both)
Data augmentation:
--augment\_data (bool; Turns on data augmentation like flip, rotate, translate, noise, tophat), --augmentation\_threshold (probability to randomly perform augmentation procedures, 
--expand\_dataset (multiply dataset by this number of images per real image data augmentation), --flip (bool; flips 'augmentation\_threshold' % of images), 
--rotate (bool; rotates 'augmentation\_threshold' % of images), --translate (bool; translates 'augmentation\_threshold' % of images), 
--tophat (bool; tophats or bottomhats 'augmentation\_threshold' % of images), --noise (bool; adds s&p or gaussian noise to 'augmentation\_threshold' % of images), 

### Data Augmentation example:

![Data Augmentation](./images/augmentation.jpg)

### Data

I know we're using Pytorch here, but my data loader is a user-defined implementation of open-cv, which loads the data as Numpy ndarrays.  I may implement Pytorch's dataloader in the future.
The trickiest part of working with arrays is typically the sizing and dimensions, so I took care to comment very thoroughly what was happening in the model, and what dimensions/shapes each function expects.  Don't hesitate to ask me if there is anything unclear to you.

[Back to Top](#table-of-contents)

## Setup

### Modules needed

numpy, cv2 (opencv), tqdm, skimage, scikit-learn, scipy, psutil, torch

[Back to Top](#table-of-contents)

## Authors

Dan McGonigle
dpmcgonigle@gmail.com

[Back to Top](#table-of-contents)
