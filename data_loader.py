import numpy as np
import cv2
import os
from glob import glob
from tqdm import tqdm
from utils import randseed, filepath_to_name
from skimage import filters
############################################################################################
#                           DATA LOADING FUNCTIONS
#   List of functions:
#       get_prob_dir            - set the default prob_dir if one isn't provided
#       get_data_dir            - set the default data_dir if one isn't provided
#       get_filenames           - returns the basename of the different image files
#       load_data_filenames     - returns lists of train and test image and label image filenames
#       load_prob_filenames     - returns lists of train and test image probability filenames
#       load_train_test_images  - loads all images and splits them to train/validation bins
#       load_image              - loads one image
#       load_image_batch        - loads a clutch of images
#       reshape_transpose       - reshapes the numpy arrays of images for modelling (N x C x H x W)
#       dtype_0_255             - returns data type to use in numpy arrays that hold 0-255 data
#       dtype_0_1               - returns data type to use in numpy arrays that hold 0-1 data
############################################################################################
def get_prob_dir():
    """ Return default prob_dir if one isn't provided """
    return "D:\\Data\\Vessels\\"
#   End get_prob_dir
############################################################################################

############################################################################################
def get_data_dir():
    """ Return default data_dir if one isn't provided """
    return "D:\\Data\\Vessels\\"
#   End get_prob_dir
############################################################################################

############################################################################################
def get_filenames(data_dir=get_data_dir(), pred=False):
    """
    Retrieving the training and prediction maps with sorted(glob()) in the load_data_filenames was mixing up images
        that were suffixed with a number.  For instance, 3_LCA_LAO.png and 3_LCA_LAO2.png would reverse order in the 
        prob dir because of the _pred extension 3_LCA_LAO2_pred.png, 3_LCA_LAO_pred.png.
    It was therefore necessary to load all files in the same manner, hence this function that gets them from "training".
    if pred==True, addes "_pred" before extension for the probability maps generated in Stage 1.
    """
    # basename returns the filename without path
    filenames = [os.path.basename(f) for f in sorted(glob(os.path.join(data_dir, "training", "*.png")))]
    
    if pred:
        return [os.path.splitext(f)[0] + "_pred" + os.path.splitext(f)[1] for f in filenames]
    else:
        return filenames
#   End get_filenames
############################################################################################

############################################################################################
def load_data_filenames(data_dir=get_data_dir(), cv=0, cv_max=5):
    """
    returns 4 lists of filenames: train_x_image_paths, train_y_image_paths, test_x_image_paths, test_y_image_paths
    data_dir assumes sub-folders named "training" and "label" exist with images and target images, respectively.
    cv specifies which fold to use for training/validation, from 0 to (cv_max - 1)
    The lists will be in a random permutation of the number of images; uses constant random seed for reproducibility.
    """
    # Constant random seed for reproducible results
    np.random.seed(randseed())

    assert 0 <= cv < cv_max, "Choose cv fold between 0 and (cv_max - 1)"
    # os.path.join ensures that this path will work on any os
    filenames = get_filenames(data_dir=data_dir)
    data_image_files = [os.path.join(data_dir, "training", f) for f in filenames]
    data_label_files = [os.path.join(data_dir, "label", f) for f in filenames]
    
    # r_index will be a list of image indexes with the size of the total number of images
    # Ex: 5 images might yield [3,1,2,4,0]
    r_index = np.random.permutation(len(data_image_files))
    num_images = len(data_image_files)
    num_images_per_cv = int(num_images/cv_max)

    train_x_image_paths = []
    train_y_image_paths = []
    test_x_image_paths = []
    test_y_image_paths = []

    for i in range(num_images):
        # If images aren't in the validation fold, add them to train_*; else, add them to test_*
        # EX: If cv==1, cv_max==5, num_images==50, images 10-19 will be validation, all others will be training
        if i not in list(range(cv*num_images_per_cv, (cv+1)*num_images_per_cv)):
            train_x_image_paths.append(np.array(data_image_files)[r_index][i])
            train_y_image_paths.append(np.array(data_label_files)[r_index][i])
        else:   # Validation images
            test_x_image_paths.append(np.array(data_image_files)[r_index][i])
            test_y_image_paths.append(np.array(data_label_files)[r_index][i])

    return train_x_image_paths, train_y_image_paths, test_x_image_paths, test_y_image_paths
# END load_data_filenames
############################################################################################

############################################################################################
def load_prob_filenames(data_dir=get_data_dir(), prob_dir=get_prob_dir(), cv=0, cv_max=5):
    """
    returns 2 lists of filenames: train_prob_image_paths, test_prob_image_paths
    data_dir assumes sub-folder "probability_maps" exists with probability images for stage 2 of the U-Net.
    cv specifies which fold to use for training/validation, from 0 to (cv_max - 1)
    The lists will be in a random permutation of the number of images; uses constant random seed for reproducibility.
    """    
    # Constant random seed for reproducible results
    np.random.seed(randseed())
    assert 0 <= cv < cv_max, "Choose cv fold between 0 and (cv_max - 1)"
    # os.path.join ensures that this path will work on any os
    filenames = get_filenames(data_dir=data_dir, pred=True)
    data_prob_map_files = [os.path.join(prob_dir, "probability_maps", f) for f in filenames]
    
    # r_index will be a list of image indexes with the size of the total number of images
    # Ex: 5 images might yield [3,1,2,4,0]
    r_index = np.random.permutation(len(data_prob_map_files))
    num_images = len(data_prob_map_files)
    num_images_per_cv = int(num_images/cv_max)
    train_prob_image_paths = []
    test_prob_image_paths = []
    
    for i in range(num_images):
        # If images aren't in the validation fold, add them to train_*; else, add them to test_*
        # EX: If cv==1, cv_max==5, num_images==50, images 10-19 will be validation, all others will be training
        if i not in list(range(cv * num_images_per_cv, (cv + 1) * num_images_per_cv)):
            train_prob_image_paths.append(np.array(data_prob_map_files)[r_index][i])
        else:
            test_prob_image_paths.append(np.array(data_prob_map_files)[r_index][i])
                
    return train_prob_image_paths, test_prob_image_paths
# END load_prob_filenames
############################################################################################

############################################################################################
def load_train_test_images(data_dir=get_data_dir(), prob_dir=get_prob_dir(), 
    image_type="grayscale", cv=0, cv_max=5, stage=1):
    """
    returns tuple of data and filename dictionaries:
        data_images: dict_keys(['train_y_images', 'test_x_images', 'test_y_images', 'train_x_images'])
            numpy ndarrays ready for training and validation (dims N x C x H x W)
        data_names: dict_keys(['test_filenames', 'train_filenames']) - lists of basenames without extensions
    image_type is optional and can be str ('grayscale', 'rgb'); default grayscale
    prob_dir only needs to be specified for stage 2, since 
    """
    # Get filenames for X and Y train and val images
    train_x_data_paths, train_y_data_paths, test_x_data_paths, test_y_data_paths = load_data_filenames(
        data_dir=data_dir, cv=cv, cv_max=cv_max
    )
        
    # Load X and Y train and val images as np arrays
    # If grayscale, will come in (n x h x w) format, so it will need to be expanded to (n x c x h x w)
    train_x_images = reshape_transpose( 
        load_image_batch(train_x_data_paths, image_type=image_type) , 
        image_type=image_type)
    train_y_images = reshape_transpose( 
        load_image_batch(train_y_data_paths, image_type=image_type) , 
        image_type=image_type)
    test_x_images = reshape_transpose( 
        load_image_batch(test_x_data_paths, image_type=image_type) , 
        image_type=image_type)
    test_y_images = reshape_transpose( 
        load_image_batch(test_y_data_paths, image_type=image_type) , 
        image_type=image_type)
        
    # ensure all images were of the same size; if not, load_image_batch will return a 1d array of type object
    assert (train_x_images.ndim > 1 and train_y_images.ndim > 1 and test_x_images.ndim > 1 and test_y_images.ndim > 1),\
        "Ensure images are all same size; look in dataset with 1 dimension: train_x=>%d, train_y=>%d, \
        test_x=>%d, test_y=>%d" % (train_x_images.ndim, train_y_images.ndim, test_x_images.ndim, test_y_images.ndim)

    data_images = {}
    if stage == 1:
        data_images = {"train_x_images": train_x_images,
                       "train_y_images": train_y_images,
                       "test_x_images": test_x_images,  
                       "test_y_images": test_y_images,}
    ### END STAGE 1 DATA LOADING                   
    
    elif stage == 2:
        # Shape of training data; 3 channels - one for orig image, one for probability map, one for edge map
        num_channels = 3
        img_height = train_x_images[0,0].shape[0]
        img_width = train_x_images[0,0].shape[1]
        new_shape = [img_height, img_width, num_channels]
        
        # Get filenames for train and val probability maps and load images as np arrays if stage == 2
        train_prob_data_paths, test_prob_data_paths = load_prob_filenames(data_dir=data_dir, prob_dir=prob_dir, cv=0, cv_max=cv_max)
        
        train_prob_images = reshape_transpose( 
            load_image_batch(train_prob_data_paths, image_type=image_type) , 
            image_type=image_type)
        test_prob_images = reshape_transpose( 
            load_image_batch(test_prob_data_paths, image_type=image_type) , 
            image_type=image_type)
    
        num_train_imgs = train_x_images.shape[0]
        # initialize (for scoping) new ndim array to hold orig image, prob map and edge map 
        multidim_train_x_images = np.empty((num_train_imgs, num_channels, img_height, img_width), dtype=dtype_0_255())
        for i in range(num_train_imgs):
            multidim_train_x_images[i,0] = np.squeeze(train_x_images[i])
            multidim_train_x_images[i,1] = np.squeeze(train_prob_images[i])
            # Run an edge detection on the probability map; needs to be int or np.sqrt throws a warning
            # That seems to be a general rule with images: int for (0:255) and float for (0:1) or (-1:1)
            multidim_train_x_images[i,2] = filters.prewitt(np.squeeze(train_prob_images[i].astype(int)))
        
        num_test_imgs = test_x_images.shape[0]
        # initialize (for scoping) new ndim array to hold orig image, prob map and edge map 
        multidim_test_x_images = np.empty((num_test_imgs, num_channels, img_height, img_width), dtype=dtype_0_255())
        for i in range(num_test_imgs):
            multidim_test_x_images[i,0] = np.squeeze(test_x_images[i])
            multidim_test_x_images[i,1] = np.squeeze(test_prob_images[i])
            # Run an edge detection on the probability map; needs to be int or np.sqrt throws a warning
            # That seems to be a general rule with images: int for (0:255) and float for (0:1) or (-1:1)
            multidim_test_x_images[i,2] = filters.prewitt(np.squeeze(test_prob_images[i].astype(int)))

        data_images = {"train_x_images": multidim_train_x_images,
                       "train_y_images": train_y_images,
                       "test_x_images": multidim_test_x_images,
                       "test_y_images": test_y_images}            
    ### END STAGE 2 DATA LOADING
    
    train_filenames = [filepath_to_name(train_file_path) for train_file_path in train_x_data_paths]
    test_filenames = [filepath_to_name(train_file_path) for train_file_path in test_x_data_paths]
    
    data_names = {"train_filenames": train_filenames,
                  "test_filenames": test_filenames}

    return data_images, data_names
# END load_train_test_images
############################################################################################

############################################################################################
def load_image(image_path, image_type="grayscale"):
    """
    use cv2 module to load an image
    inputs: image_path, image_type (optional, str: "grayscale" or "rgb")
    output: numpy ndarray of image if it exists
    """
    if not os.path.isfile(image_path):
        raise OSError("Image %s not found" % image_path)
    if image_type.lower() == "grayscale":
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    elif image_type.lower() == "rgb":
        image = cv2.cvtColor(cv2.imread(image_path, -1), cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("image_type must be \"grayscale\", \"rgb\"...")
    return image.astype(dtype_0_255())
# END load_image
############################################################################################

############################################################################################
def load_image_batch(image_path, image_type="grayscale"):
    """
    use load_image (which uses cv2 module to return a numpy ndarray) to load a numpy array of images.
    inputs: image_paths: list of filenames, image_type (optional, "grayscale" or "rgb")
    output: numpy ndarray of images of shape (Height x Width x Channels)
    """
    images = []
    for path in image_path:
        images.append(load_image(image_path=path, image_type=image_type))
        
    return np.asarray(images)
# END load_image_batch
############################################################################################

############################################################################################
def reshape_transpose(image_array, image_type="grayscale"):
    """
    returns np ndarray images in format for Pytorch model convolutions ( n x c x h x w )
    inputs: image_array after load_image or load_image_batch ([num_images, ]height x width x channels[if > 1]),
    image_type (optional, "grayscale" or "rgb" or number of channels)
    output: np ndarray in shape (batch_size, channels, height, width)
    """
    if image_type.lower() == "grayscale":
        # If image_array has 2 dimensions, it is (H x W); if 3, (num_images x H x W)
        if image_array.ndim == 2: # one image
            return image_array.reshape(1,1,image_array.shape[0], image_array.shape[1])
        elif image_array.ndim == 3: # multiple images
            return image_array.reshape(image_array.shape[0],1,image_array.shape[1], image_array.shape[2])
        else:
            raise ValueError("Unrecognized number of dimensions for grayscale reshape_transpose.  \
            Should be 2 or 3 after using load_image() or load_image_batch(), received %s"%str(image_array.ndim))
    elif image_type.lower() == "rgb" or image_type == 3:
        # If image_array has 3 dimensions, it is (H x W x C); if 4, (num_images x H x W x C)
        if image_array.ndim == 3: # one image
            image_array = image_array.reshape(1,image_array.shape[0], image_array.shape[1], image_array.shape[2])
        elif image_array.ndim == 4: # multiple images
            pass #  continue to return transpose below
        else:
            raise ValueError("Unrecognized number of dimensions for rgb reshape_transpose.  \
            Should be 3 or 4 after using load_image() or load_image_batch(), received %s"%str(image_array.ndim))   
        return np.transpose(image_array, axes=[0,3,1,2])
    else:
        ValueError("image_type must be \"grayscale\", \"rgb\"...")
# END reshape_transpose
############################################################################################

############################################################################################
def dtype_0_255():
    """
    Images loaded in 0-255 format should be integer, while 0-1 should be float
    """
    return np.uint8
# END dtype_0_255
############################################################################################

############################################################################################
def dtype_0_1():
    """
    Images loaded in 0-255 format should be integer, while 0-1 should be float
    Note that float16 corresponds to HalfTensor, which doesn't jive with the network's weights.
    """
    return np.float32
# END dtype_0_1
############################################################################################