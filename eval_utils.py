import os, sys

# scikit-learn
from skimage import filters
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from skimage import measure

# scipy for dice
from scipy.spatial import distance
import numpy as np
############################################################################################
#                          EVALUATION UTILITY FUNCTIONS
#   List of functions:
#       AUC_ROC                 - returns Area Under the Curve for ROC
#       AUC_PR                  - returns Area Under the Curve for Precision vs. Recall
#       filepath_to_name        - returns file name from full path, with or without ext (specify)
#       BtoGB                   - converts Bytes to GBs
#       get_memory               - returns string of GPU / CPU usage
#       print_d                 - prints (or writes) debugging information
#       one_hot_it              - Convert label array to one-hot replacing s with a vector of length num_classes
#       reverse_one_hot         - Transform one-hot 3D array (H x W x num_classes) to (H x W x 1), where 1 = class
#       count_params            - Get the total number of parameters that require_grad from the model
#       date_time_stamp         - return YYYYMMDD_HHMM string
############################################################################################
def AUC_ROC(true_vessel_arr, pred_vessel_arr):
    """
    Area under the ROC curve with x axis flipped
    """
    try:
        AUC_ROC = roc_auc_score(true_vessel_arr.flatten(), pred_vessel_arr.flatten())
    except:
        AUC_ROC = 0.
    return AUC_ROC
# END AUC_ROC
############################################################################################

############################################################################################
def AUC_PR(true_vessel_img, pred_vessel_img):
    """
    Precision-recall curve
    """
    try:
        precision, recall, _ = precision_recall_curve(true_vessel_img.flatten(), pred_vessel_img.flatten())
        print("after precision_recall_curve")
        AUC_prec_rec = auc(recall, precision)
        print("after auc")
    except:
        AUC_prec_rec = 0.
    return AUC_prec_rec
# END AUC_PR
############################################################################################

############################################################################################
def best_f1_threshold(precision, recall, thresholds):
    """
    returns the best F1 score and the threshold associated with it
    """
    best_f1 = -1
    for index in range(len(precision)):
        curr_f1 = 2. * precision[index] * recall[index] / (precision[index] + recall[index])
        if best_f1 < curr_f1:
            best_f1 = curr_f1
            best_threshold = thresholds[index]

    return best_f1, best_threshold
# END best_f1_threshold
############################################################################################

############################################################################################
def dice_coefficient(true_vessels, pred_vessels):
    """
    returns the dice coefficient of two flat numpy arrays
    """
    true_vessels = true_vessels.astype(np.bool).flatten()
    pred_vessels = true_vessels.astype(np.bool).flatten()
    
    return 1.0 - np.abs(distance.dice(true_vessels, pred_vessels))
# END dice_coefficient
############################################################################################

############################################################################################
def accuracy(true_vessels, pred_vessels):
    """
    returns the accuracy of two flat numpy arrays
    """
    true_vessels = true_vessels.astype(np.bool).flatten()
    pred_vessels = true_vessels.astype(np.bool).flatten()

    cm = confusion_matrix(true_vessels, pred_vessels)
    return 1. * (cm[0, 0] + cm[1, 1]) / np.sum(cm)
# END accuracy
############################################################################################

############################################################################################
def sensitivity(true_vessels, pred_vessels):
    """
    returns the sensitivity of two flat numpy arrays
    """
    true_vessels = true_vessels.astype(np.bool).flatten()
    pred_vessels = true_vessels.astype(np.bool).flatten()

    cm = confusion_matrix(true_vessels, pred_vessels)
    return 1. * cm[1, 1] / (cm[1, 0] + cm[1, 1]) 
# END sensitivity
############################################################################################

############################################################################################  
def specificity(true_vessels, pred_vessels):
    """
    returns the specifity of two flat numpy arrays
    """
    true_vessels = true_vessels.astype(np.bool).flatten()
    pred_vessels = true_vessels.astype(np.bool).flatten()

    cm = confusion_matrix(true_vessels, pred_vessels)
    return 1. * cm[0, 0] / (cm[0, 1] + cm[0, 0])
# END specificity
############################################################################################

############################################################################################
    
def threshold_by_f1(true_vessels, generated, masks, flatten=True, f1_score=False):
    vessels_in_mask, generated_in_mask = pixel_values_in_mask(true_vessels, generated, masks)
    precision, recall, thresholds = precision_recall_curve(vessels_in_mask.flatten(), generated_in_mask.flatten(),
                                                           pos_label=1)
    best_f1, best_threshold = best_f1_threshold(precision, recall, thresholds)

    pred_vessels_bin = np.zeros(generated.shape)
    pred_vessels_bin[generated >= best_threshold] = 1

    if flatten:
        if f1_score:
            return pred_vessels_bin[masks == 1].flatten(), best_f1
        else:
            return pred_vessels_bin[masks == 1].flatten()
    else:
        if f1_score:
            return pred_vessels_bin, best_f1
        else:
            return pred_vessels_bin





def img_dice(pred_vessel, true_vessel):
    threshold = filters.threshold_otsu(pred_vessel)
    pred_vessels_bin = np.zeros(pred_vessel.shape)
    pred_vessels_bin[pred_vessel >= threshold] = 1
    dice_coeff = dice_coefficient_in_train(true_vessel.flatten(), pred_vessels_bin.flatten())
    return dice_coeff


def vessel_similarity(segmented_vessel_0, segmented_vessel_1):
    try:
        threshold_0 = filters.threshold_otsu(segmented_vessel_0)
        threshold_1 = filters.threshold_otsu(segmented_vessel_1)
        segmented_vessel_0_bin = np.zeros(segmented_vessel_0.shape)
        segmented_vessel_1_bin = np.zeros(segmented_vessel_1.shape)
        segmented_vessel_0_bin[segmented_vessel_0 > threshold_0] = 1
        segmented_vessel_1_bin[segmented_vessel_1 > threshold_1] = 1
        dice_coeff = dice_coefficient_in_train(segmented_vessel_0_bin.flatten(), segmented_vessel_1_bin.flatten())
        return dice_coeff
    except:
        return 0.


def dice_coefficient_in_train(true_vessel_arr, pred_vessel_arr):
    true_vessel_arr = true_vessel_arr.astype(np.bool)
    pred_vessel_arr = pred_vessel_arr.astype(np.bool)

    intersection = np.count_nonzero(true_vessel_arr & pred_vessel_arr)

    size1 = np.count_nonzero(true_vessel_arr)
    size2 = np.count_nonzero(pred_vessel_arr)

    try:
        dc = 2. * intersection / float(size1 + size2)
    except ZeroDivisionError:
        dc = 0.0

    return dc


def operating_pts_human_experts(gt_vessels, pred_vessels, masks):
    gt_vessels_in_mask, pred_vessels_in_mask = pixel_values_in_mask(gt_vessels, pred_vessels, masks, split_by_img=True)

    n = gt_vessels_in_mask.shape[0]
    op_pts_roc, op_pts_pr = [], []
    for i in range(n):
        cm = confusion_matrix(gt_vessels_in_mask[i], pred_vessels_in_mask[i])
        fpr = 1 - 1. * cm[0, 0] / (cm[0, 1] + cm[0, 0])
        tpr = 1. * cm[1, 1] / (cm[1, 0] + cm[1, 1])
        prec = 1. * cm[1, 1] / (cm[0, 1] + cm[1, 1])
        recall = tpr
        op_pts_roc.append((fpr, tpr))
        op_pts_pr.append((recall, prec))

    return op_pts_roc, op_pts_pr


def pixel_values_in_mask(true_vessels, pred_vessels, masks, split_by_img=False):
    assert np.max(pred_vessels) <= 1.0 and np.min(pred_vessels) >= 0.0
    assert np.max(true_vessels) == 1.0 and np.min(true_vessels) == 0.0
    assert np.max(masks) == 1.0 and np.min(masks) == 0.0
    assert pred_vessels.shape[0] == true_vessels.shape[0] and masks.shape[0] == true_vessels.shape[0]
    assert pred_vessels.shape[1] == true_vessels.shape[1] and masks.shape[1] == true_vessels.shape[1]
    assert pred_vessels.shape[2] == true_vessels.shape[2] and masks.shape[2] == true_vessels.shape[2]

    if split_by_img:
        n = pred_vessels.shape[0]
        return np.array([true_vessels[i, ...][masks[i, ...] == 1].flatten() for i in range(n)]), np.array(
            [pred_vessels[i, ...][masks[i, ...] == 1].flatten() for i in range(n)])
    else:
        return true_vessels[masks == 1].flatten(), pred_vessels[masks == 1].flatten()


def remain_in_mask(imgs, masks):
    imgs[masks == 0] = 0
    return imgs


def crop_to_original(imgs, ori_shape):
    pred_shape = imgs.shape
    assert len(pred_shape) < 4

    if ori_shape == pred_shape:
        return imgs
    else:
        if len(imgs.shape) > 2:
            ori_h, ori_w = ori_shape[1], ori_shape[2]
            pred_h, pred_w = pred_shape[1], pred_shape[2]
            return imgs[:, (pred_h - ori_h) // 2:(pred_h - ori_h) // 2 + ori_h,
                   (pred_w - ori_w) // 2:(pred_w - ori_w) // 2 + ori_w]
        else:
            ori_h, ori_w = ori_shape[0], ori_shape[1]
            pred_h, pred_w = pred_shape[0], pred_shape[1]
            return imgs[(pred_h - ori_h) // 2:(pred_h - ori_h) // 2 + ori_h,
                   (pred_w - ori_w) // 2:(pred_w - ori_w) // 2 + ori_w]


def difference_map(ori_vessel, pred_vessel, mask):
    # ori_vessel : an RGB image

    thresholded_vessel = threshold_by_f1(np.expand_dims(ori_vessel, axis=0), np.expand_dims(pred_vessel, axis=0),
                                         np.expand_dims(mask, axis=0), flatten=False)

    thresholded_vessel = np.squeeze(thresholded_vessel, axis=0)
    diff_map = np.zeros((ori_vessel.shape[0], ori_vessel.shape[1], 3))
    diff_map[(ori_vessel == 1) & (thresholded_vessel == 1)] = (0, 255, 0)  # Green (overlapping)
    diff_map[(ori_vessel == 1) & (thresholded_vessel != 1)] = (255, 0, 0)  # Red (false negative, missing in pred)
    diff_map[(ori_vessel != 1) & (thresholded_vessel == 1)] = (0, 0, 255)  # Blue (false positive)

    # compute dice coefficient for a given image
    overlap = len(diff_map[(ori_vessel == 1) & (thresholded_vessel == 1)])
    fn = len(diff_map[(ori_vessel == 1) & (thresholded_vessel != 1)])
    fp = len(diff_map[(ori_vessel != 1) & (thresholded_vessel == 1)])

    return diff_map, 2. * overlap / (2 * overlap + fn + fp)

