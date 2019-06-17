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
#       best_f1_threshold       - 
#       eval_metrics            - returns dice score, accuracy, sensitivity and specificity
#       dice                    - 
#       accuracy                - 
#       sensitivity             - 
#       specificity             - 
############################################################################################
def AUC_ROC(true_vessel_arr, pred_vessel_arr):
    """
    Area under the ROC curve with x axis flipped
    Both arrays need to be flattened (np.flatten())
    """
    try:
        AUC_ROC = roc_auc_score(true_vessel_arr.flatten(), pred_vessel_arr.flatten())
    except Exception as e:
        print("AUC_ROC: ERROR - %s" % str(e))
        AUC_ROC = 0.
    return AUC_ROC
# END AUC_ROC
############################################################################################

############################################################################################
def AUC_PR(true_vessel_img, pred_vessel_img):
    """
    Precision-recall curve
    Both arrays need to be flattened (np.flatten())
    """
    try:
        precision, recall, _ = precision_recall_curve(true_vessel_img.flatten(), pred_vessel_img.flatten())
        AUC_prec_rec = auc(recall, precision)
    except Exception as e:
        print("AUC_PR: ERROR - %s" % str(e))
        AUC_prec_rec = 0.
    return AUC_prec_rec
# END AUC_PR
############################################################################################

############################################################################################
def best_f1_threshold(precision, recall, thresholds):
    """
    returns the best F1 score and the threshold associated with it
    Both arrays need to be flattened (np.flatten())
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
def eval_metrics(true_vessels, pred_vessels):
    """
    returns dice_coefficient, accuracy, specificity and sensitivity
    inputs are binary arrays for target and prediction classification maps
    """
    true_vessels = true_vessels.flatten()
    pred_vessels = pred_vessels.flatten()
    cm = confusion_matrix(true_vessels, pred_vessels)
    
    dice = dice_coefficient(true_vessels, pred_vessels)
    acc = accuracy(cm=cm)
    sens = sensitivity(cm=cm)
    spec = specificity(cm=cm)
    
    return dice, acc, sens, spec
# END eval_metrics
############################################################################################

############################################################################################
def dice_coefficient(true_vessels, pred_vessels):
    """
    returns the dice coefficient of two flat numpy arrays of classification maps
    Both arrays need to be flattened (np.flatten())
    Both arrays need to be flattened (np.flatten()) prior to input
    """
    return 1.0 - np.abs(distance.dice(true_vessels, pred_vessels))
# END dice_coefficient
############################################################################################

############################################################################################
def accuracy(cm=None, true_vessels=None, pred_vessels=None):
    """
    returns the accuracy of a confusion matrix of two flat numpy arrays
    Both arrays need to be flattened (np.flatten())
    can provide two arrays of classification maps (true_vessels and pred_vessels) or the confusion matrix of the two
    """
    if cm is not None:
        return 1. * (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    else:
        cm = confusion_matrix(true_vessels, pred_vessels)
        return 1. * (cm[0, 0] + cm[1, 1]) / np.sum(cm)
# END accuracy
############################################################################################

############################################################################################
def sensitivity(cm=None, true_vessels=None, pred_vessels=None):
    """
    returns the sensitivity of two flat numpy arrays
    Both arrays need to be flattened (np.flatten())
    can provide two arrays of classification maps (true_vessels and pred_vessels) or the confusion matrix of the two
    """
    if cm is not None:
        return 1. * cm[1, 1] / (cm[1, 0] + cm[1, 1])
    else:
        cm = confusion_matrix(true_vessels, pred_vessels)
        return 1. * cm[1, 1] / (cm[1, 0] + cm[1, 1]) 
# END sensitivity
############################################################################################

############################################################################################  
def specificity(cm=None, true_vessels=None, pred_vessels=None):
    """
    returns the specifity of two flat numpy arrays
    Both arrays need to be flattened (np.flatten())
    can provide two arrays of classification maps (true_vessels and pred_vessels) or the confusion matrix of the two
    """
    if cm is not None:
        return 1. * cm[0, 0] / (cm[0, 1] + cm[0, 0])
    else:
        cm = confusion_matrix(true_vessels, pred_vessels)
        return 1. * cm[0, 0] / (cm[0, 1] + cm[0, 0])
# END specificity
############################################################################################