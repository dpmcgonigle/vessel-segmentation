import os,time,cv2, sys, math
import subprocess
import numpy as np
import time
from datetime import datetime
import os, random
import subprocess
import psutil
import torch
from scipy.misc import imread

############################################################################################
#                           UTILITY FUNCTIONS
#   List of functions:
#       install                 - installs a python module; good for lazy installations in jupyter notebook
#       randseed                - sets the random seed for reproducibility throughout suite
#       str2bool
#       filepath_to_name        - returns file name from full path, with or without ext (specify)
#       BtoGB                   - converts Bytes to GBs
#       get_memory               - returns string of GPU / CPU usage
#       print_d                 - prints (or writes) debugging information
#       one_hot_it              - Convert label array to one-hot replacing s with a vector of length num_classes
#       reverse_one_hot         - Transform one-hot 3D array (H x W x num_classes) to (H x W x 1), where 1 = class
#       count_params            - Get the total number of parameters that require_grad from the model
#       date_time_stamp         - return YYYYMMDD_HHMM string
############################################################################################
def install(name):
    """
    installs modules for importing; this is handy when you're in jupyter notebook and don't want to quit for install.
    example: install('opencv-python')
    outputs: standard output and standard error from install process
    """
    p = subprocess.Popen(['pip', 'install', name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    print("Output: " + str(out))
    print("Errors: " + str(err))
# END install
############################################################################################

############################################################################################
def randseed():
    """ returns int(42) intended for np.random.seed(42) for reproducibility in random calls """
    return 42
# END randseed
############################################################################################

############################################################################################
def str2bool(input_str):
    """
    returns boolean value based on input string; i.e. ('yes', 'true', 't', 'y', '1')
    inputs: str(str); output: boolean
    required for command line args, converts values to True/False
    """
    if input_str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input_str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
# END str2bool
############################################################################################

############################################################################################
def filepath_to_name(full_name, ext=False):
    """ returns file name with or without extension based on full path name """
    file_name = os.path.basename(full_name)
    return file_name if ext else os.path.splitext(file_name)[0]
# END filepath_to_name
############################################################################################

############################################################################################
def BtoGB(bytes):
    """
    returns gigabyte representation of a byte value
    input: bytes(integer; any type of number would work)
    output: gigabytes (float)
    """
    return float(bytes / 1e9)
# END filepath_to_name
############################################################################################

############################################################################################
def get_memory(gpu=0):
    """
    returns a string containing your torch.cuda memory usage, CPU usage, and virtual memory information
    >>> print(getMemory())
    out: GPU usage: 3.200 GB / 4.000 GB => 80.000 %, CPU usage: 0.812 GB => 3.301%\n (virtual memory info)
    """
    alloc = BtoGB(torch.cuda.memory_allocated(gpu))
    pct = float(alloc / 4.0) # NVIDIA GeForce GTX 1050Ti has 4GB RAM - McGonigle
    py = psutil.Process(os.getpid())
    cpuUse = BtoGB(py.memory_info()[0])
    return "GPU Memory usage: %.03f GB / 4.000 GB => %.03f %%, CPU usage: %.03f GB => %.03f %%\n%s" % (alloc, pct,
        cpuUse, float(psutil.cpu_percent()), str(psutil.virtual_memory()))
# END getMemory
############################################################################################

############################################################################################
def print_d(input_str, file_path=None, debug=False):
    """ 
    Simple debugging print statements that can be turned off or on with the debug variable. 
    inputs: str, a string to output; file_path, a file to write the output to.
    If a file_path is specified, 
    """
    debug = debug
    time_stamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if debug: 
        print("%s: %s" % (time_stamp, input_str))
        if file_path:
            try:
                with open(file_path, "a") as out_file:
                    out_file.write("%s: %s" % (time_stamp, input_str))
            except Exception as e:
                print("Error opening %s: %s" % (file_path, str(e)))
# END print_d
############################################################################################

############################################################################################
def one_hot_it(label, label_values):
    """
    Convert an image label array to one-hot format by replacing each pixel value with a vector of length num_classes
    inputs: label, The 2D array segmentation image label; label_values, integer
    output: A 2D array with the same width and hieght as the input, but with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    # print("Time 2 = ", time.time() - st)

    return semantic_map
# END one_hot_it
############################################################################################

############################################################################################
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is the classified class key.
    inputs: image, The one-hot format image 
    output: A 2D array with the same height and width as the input, but
        with a depth size of 1, where each pixel value is the classified class key.
    """
    x = np.argmax(image, axis=-1)
    return x
# END reverse_one_hot
############################################################################################

############################################################################################
def count_params(model):
    """
    prints the total number of parameters in the model
    """
    total_parameters = sum(p.data.nelement() for p in model.parameters() if p.requires_grad)
    print("This model has %d trainable parameters"% (total_parameters))
# END count_params
############################################################################################

############################################################################################
def date_time_stamp():
    """returns datetime stamp in YYYYMMDD_HHMM format"""
    now = datetime.now() # current date and time
    return now.strftime("%Y%m%d_%H%M")
 # END date_time_stamp
############################################################################################

############################################################################################   

####################
####################
#   Haven't incorporated functions below yet
####################
####################


# Subtracts the mean images from ImageNet
def mean_image_subtraction(inputs, means=[123.68, 116.78, 103.94]):
    inputs=tf.to_float(inputs)
    num_channels = inputs.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=inputs)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)

def _lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard

def _flatten_probas(probas, labels, ignore=None, order='BHWC'):
    """
    Flattens predictions in the batch
    """
    if order == 'BCHW':
        probas = tf.transpose(probas, (0, 2, 3, 1), name="BCHW_to_BHWC")
        order = 'BHWC'
    if order != 'BHWC':
        raise NotImplementedError('Order {} unknown'.format(order))
    C = probas.shape[3]
    probas = tf.reshape(probas, (-1, C))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return probas, labels
    valid = tf.not_equal(labels, ignore)
    vprobas = tf.boolean_mask(probas, valid, name='valid_probas')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vprobas, vlabels

def _lovasz_softmax_flat(probas, labels, only_present=True):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
    """
    C = probas.shape[1]
    losses = []
    present = []
    for c in range(C):
        fg = tf.cast(tf.equal(labels, c), probas.dtype) # foreground for class c
        if only_present:
            present.append(tf.reduce_sum(fg) > 0)
        errors = tf.abs(fg - probas[:, c])
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort_{}".format(c))
        fg_sorted = tf.gather(fg, perm)
        grad = _lovasz_grad(fg_sorted)
        losses.append(
            tf.tensordot(errors_sorted, tf.stop_gradient(grad), 1, name="loss_class_{}".format(c))
                      )
    losses_tensor = tf.stack(losses)
    if only_present:
        present = tf.stack(present)
        losses_tensor = tf.boolean_mask(losses_tensor, present)
    return losses_tensor

def lovasz_softmax(probas, labels, only_present=True, per_image=False, ignore=None, order='BHWC'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, H, W, C] or [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
      order: use BHWC or BCHW
    """
    probas = tf.nn.softmax(probas, 3)
    labels = helpers.reverse_one_hot(labels)

    if per_image:
        def treat_image(prob, lab):
            prob, lab = tf.expand_dims(prob, 0), tf.expand_dims(lab, 0)
            prob, lab = _flatten_probas(prob, lab, ignore, order)
            return _lovasz_softmax_flat(prob, lab, only_present=only_present)
        losses = tf.map_fn(treat_image, (probas, labels), dtype=tf.float32)
    else:
        losses = _lovasz_softmax_flat(*_flatten_probas(probas, labels, ignore, order), only_present=only_present)
    return losses


# Randomly crop the image to a specific size. For data augmentation
def random_crop(image, label, crop_height, crop_width):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')
        
    if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):
        x = random.randint(0, image.shape[1]-crop_width)
        y = random.randint(0, image.shape[0]-crop_height)
        
        if len(label.shape) == 3:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width, :]
        else:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width]
    else:
        raise Exception('Crop shape exceeds image dimensions!')

# Compute the average segmentation accuracy across all classes
def compute_global_accuracy(pred, label):
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)

# Compute the class-specific segmentation accuracy
def compute_class_accuracies(pred, label, num_classes):
    total = []
    for val in range(num_classes):
        total.append((label == val).sum())

    count = [0.0] * num_classes
    for i in range(len(label)):
        if pred[i] == label[i]:
            count[int(pred[i])] = count[int(pred[i])] + 1.0

    # If there are no pixels from a certain class in the GT, 
    # it returns NAN because of divide by zero
    # Replace the nans with a 1.0.
    accuracies = []
    for i in range(len(total)):
        if total[i] == 0:
            accuracies.append(1.0)
        else:
            accuracies.append(count[i] / total[i])

    return accuracies


def compute_mean_iou(pred, label):

    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels);

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))


    mean_iou = np.mean(I / U)
    return mean_iou

    
def compute_class_weights(labels_dir, label_values):
    '''
    Arguments:
        labels_dir(list): Directory where the image segmentation labels are
        num_classes(int): the number of classes of pixels in all images

    Returns:
        class_weights(list): a list of class weights where each index represents each class label and the element is the class weight for that label.

    '''
    image_files = [os.path.join(labels_dir, file) for file in os.listdir(labels_dir) if file.endswith('.png')]

    num_classes = len(label_values)

    class_pixels = np.zeros(num_classes) 

    total_pixels = 0.0

    for n in range(len(image_files)):
        image = imread(image_files[n])

        for index, colour in enumerate(label_values):
            class_map = np.all(np.equal(image, colour), axis = -1)
            class_map = class_map.astype(np.float32)
            class_pixels[index] += np.sum(class_map)

            
        print("\rProcessing image: " + str(n) + " / " + str(len(image_files)), end="")
        sys.stdout.flush()

    total_pixels = float(np.sum(class_pixels))
    index_to_delete = np.argwhere(class_pixels==0.0)
    class_pixels = np.delete(class_pixels, index_to_delete)

    class_weights = total_pixels / class_pixels
    class_weights = class_weights / np.sum(class_weights)

    return class_weights


