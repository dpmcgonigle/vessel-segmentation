from torch.nn import ReLU, Conv2d, BatchNorm2d, ConvTranspose2d, MaxPool2d, Sequential, Softmax
from torch.autograd import Variable
import torch.nn as nn
from torch import add
import os,time,cv2
import numpy as np
import torch
from utils import print_d, get_memory

############################################################################################
#                           MobileUNet Class
#   List of methods:
#       init                        - constructor, sets up model
#       forward                     - processes images through the model, returning segmented array values
#       ConvBlock                   - returns file name from full path, with or without ext (specify)
#       DepthwiseSeparableConvBlock - converts Bytes to GBs
#       ConvTransposeBlock          - returns string of GPU / CPU usage
#       setSkip                     - stores the value of the feature maps produced after down-sampling layer
#       getSkip                     - returns the down-sampling feature map values saved
#       addSkip                     - adds the value of the down-sampling feature maps to the up-sampling layer
############################################################################################ 
class MultiMobileUNet(nn.Module):
    """
    This Mobile U-Net has been adapted from the TensorFlow version on George Sief's semantic segmentation suite:
    https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/blob/master/models/MobileUNet.py
    There are 5 blocks of down-sampling convolutions and 5 blocks of up-sampling with transposed and regular convolutions.
    NOTE: 6/1/2019 - My NVIDIA GeForce GTX 1050 Ti has a capacity of 4GB memory, and the full model was exceeding that, 
    yielding a run-time error.  It may be a memory management issue, but until that time I have commented out some of 
    the depthwise separable convolutions to ensure it runs fine on my machine. - McGonigle
    if gpu is -1, run on CPU.  Otherwise, run on GPU device specified
    
    NOTE: 7/21/2019 - I built in the context block based on: 
    "MULTI-SCALE CONTEXT AGGREGATION BY DILATED CONVOLUTIONS"
    http://vladlen.info/papers/dilated-convolutions.pdf
    Architecture:
    Layer           1   2   3   4       5       6       7       8
    Convolution     3×3 3×3 3×3 3×3     3×3     3×3     3×3     1×1
    Dilation        1   1   2   4       8       16      1       1 
    Truncation      Yes Yes Yes Yes     Yes     Yes     Yes     No
    Receptive field 3×3 5×5 9×9 17×17   33×33   65×65   67×67   67×67
    Convolution feature map size formula: 
    The context module is designed to increase the performance of dense prediction architectures by
    aggregating multi-scale contextual information. The module takes C feature maps as input and
    produces C feature maps as output. The input and output have the same form, thus the module can
    be plugged into existing dense prediction architectures.
    """
    def __init__(self, 
        input_channels=1,
        n_filters=64,
        num_classes=2, 
        gpu=-1, 
        gpu_4g_limit=True,
        dropout=0.0, 
        depth=5,
        skip=True
    ):
        """
        Constructor for MobileUNet
        inputs:
        input_channels: 1 for grayscale, 3 for RGB, etc.
            up-sample.  MobileUNet will not perform this action.
        num_classes: number of classes for semantic segmentation 2 for semantic segmentation
        gpu: Determines if we use cuda for this model. -1 uses CPU 
            NOTE If using GPU, you need to call .cuda() on the variables and model in the driver script.
        gpu_4g_limit: If you have a 4GB Memory limit (like my NVIDIA GeForce GTX 1050 Ti), the full model is too big
            when using 512 x 512 images
        """
        super(MultiMobileUNet, self).__init__()
        
        # self.skips will keep track of down-sampled matrices to add to the up-sampling if self.has_skip == True
        self.skips = {} 
        self.depth = depth
        self.has_skip = skip
        
        # Boolean determines whether to run on CPU
        self.cpu = (gpu == -1)
        # Boolean determines whether GPU is limited to 4GB RAM.  If so, run with max batch size of 2
        self.gpu_4g_limit = gpu_4g_limit
        
        self.dropout=dropout
        
        #####################
        # Downsampling path 
        #####################
        
        #
        #   The example input sizes are going to be based on a 512 x 512 grayscale image as input so you can see what 
        #   each layer is doing. Convolutions and Depthwise Separable Convolutions maintain same size.
        #   Formula: ((W - K - 2P) / S) + 1, where W = input dimension, K = filter size, P = padding, S = stride
        #   EX: W = 512, K = 3, P = 1, S = 1 will yield the same size image after convolution
        #   Input size N x C x H x W (batch_size x channels x height x width); for this example, 1 x 1 x 512 x 512
        #
        self.down1 = Sequential(
            self.ConvBlock(input_channels = input_channels, n_filters = n_filters),
            self.DepthwiseSeparableConvBlock(input_channels = n_filters, n_filters = n_filters, slim=self.gpu_4g_limit), 
            MaxPool2d(kernel_size = [2, 2], stride = [2, 2])
        )
        
        #
        #   Input size 1 x 64 x 256 x 256
        #
        if depth > 1:
            self.down2 = Sequential(
                self.DepthwiseSeparableConvBlock(input_channels = n_filters, n_filters = n_filters*2),
                self.DepthwiseSeparableConvBlock(input_channels = n_filters*2, n_filters = n_filters*2, slim=self.gpu_4g_limit), 
                MaxPool2d(kernel_size = [2, 2], stride = [2, 2])
            )

        #
        #   Input size 1 x 128 x 128 x 128
        #
        if depth > 2:
            self.down3 = Sequential(
                self.DepthwiseSeparableConvBlock(input_channels = n_filters*2, n_filters = n_filters*4),
                self.DepthwiseSeparableConvBlock(input_channels = n_filters*4, n_filters = n_filters*4, slim=self.gpu_4g_limit), 
                self.DepthwiseSeparableConvBlock(input_channels = n_filters*4, n_filters = n_filters*4, slim=self.gpu_4g_limit), 
                MaxPool2d(kernel_size = [2, 2], stride = [2, 2])
            )

        #
        #   Input size 1 x 256 x 64 x 64
        #
        if depth > 3:
            self.down4 = Sequential(
                self.DepthwiseSeparableConvBlock(input_channels = n_filters*4, n_filters = n_filters*8),
                self.DepthwiseSeparableConvBlock(input_channels = n_filters*8, n_filters = n_filters*8, slim=self.gpu_4g_limit), 
                self.DepthwiseSeparableConvBlock(input_channels = n_filters*8, n_filters = n_filters*8, slim=self.gpu_4g_limit), 
                MaxPool2d(kernel_size = [2, 2], stride = [2, 2])
            )

        #
        #   Input size 1 x 512 x 32 x 32
        #
        if depth > 4:
            self.down5 = Sequential(
                self.DepthwiseSeparableConvBlock(input_channels = n_filters*8, n_filters = n_filters*8),
                self.DepthwiseSeparableConvBlock(input_channels = n_filters*8, n_filters = n_filters*8, slim=self.gpu_4g_limit), 
                self.DepthwiseSeparableConvBlock(input_channels = n_filters*8, n_filters = n_filters*8, slim=self.gpu_4g_limit), 
                MaxPool2d(kernel_size = [2, 2], stride = [2, 2])
            )

        #####################
        # Upsampling path 
        #####################
        
        #
        #   Transpose convolutions built to double height and width, while depthwise separable convolutions maintain size.
        #   Formula: S ( W - 1 ) + F - 2P, where W = input dimension, K = filter size, P = padding, S = stride
        #   EX: W = 256, K = 2, P = 0, S = 2 will yield double the same size image after convolution
        #   Input size 1 x 512 x 16 x 16 in the example we started from layer 1
        #
        if depth > 4:
            self.up1 = Sequential(
                self.ConvTransposeBlock(input_channels = n_filters*8, n_filters = n_filters*8),
                self.DepthwiseSeparableConvBlock(input_channels = n_filters*8, n_filters = n_filters*8, slim=self.gpu_4g_limit),
                self.DepthwiseSeparableConvBlock(input_channels = n_filters*8, n_filters = n_filters*8, slim=self.gpu_4g_limit), 
                self.DepthwiseSeparableConvBlock(input_channels = n_filters*8, n_filters = n_filters*8)
            )

        #  
        #   Input size 1 x 512 x 32 x 32
        #
        if depth > 3:
            self.up2 = Sequential(
                self.ConvTransposeBlock(input_channels = n_filters*8, n_filters = n_filters*8),
                self.DepthwiseSeparableConvBlock(input_channels = n_filters*8, n_filters = n_filters*8, slim=self.gpu_4g_limit), 
                self.DepthwiseSeparableConvBlock(input_channels = n_filters*8, n_filters = n_filters*8, slim=self.gpu_4g_limit), 
                self.DepthwiseSeparableConvBlock(input_channels = n_filters*8, n_filters = n_filters*4)
            )

        #  
        #   Input size 1 x 256 x 64 x 64
        #
        if depth > 2:
            self.up3 = Sequential(
                self.ConvTransposeBlock(input_channels = n_filters*4, n_filters = n_filters*4),
                self.DepthwiseSeparableConvBlock(input_channels = n_filters*4, n_filters = n_filters*4, slim=self.gpu_4g_limit), 
                self.DepthwiseSeparableConvBlock(input_channels = n_filters*4, n_filters = n_filters*4, slim=self.gpu_4g_limit), 
                self.DepthwiseSeparableConvBlock(input_channels = n_filters*4, n_filters = n_filters*2)
            )

        #  
        #   Input size 1 x 128 x 128 x 128
        #
        if depth > 1:
            self.up4 = Sequential(
                self.ConvTransposeBlock(input_channels = n_filters*2, n_filters = n_filters*2),
                self.DepthwiseSeparableConvBlock(input_channels = n_filters*2, n_filters = n_filters*2, slim=self.gpu_4g_limit), 
                self.DepthwiseSeparableConvBlock(input_channels = n_filters*2, n_filters = n_filters)
            )

        #  
        #   Input size 1 x 64 x 256 x 256
        #
        self.up5 = Sequential(
            self.ConvTransposeBlock(input_channels = n_filters, n_filters = n_filters),
            self.DepthwiseSeparableConvBlock(input_channels = n_filters, n_filters = n_filters, slim=self.gpu_4g_limit), 
            self.DepthwiseSeparableConvBlock(input_channels = n_filters, n_filters = n_filters)
        )

        #####################
        #   Pre-Softmax output - input size 1 x 64 x 512 x 512, output size 1 x 2 x 512 x 512 for binary classification
        #####################
        self.out = Sequential(
            self.ContextModule(n_filters),
            Conv2d(in_channels = n_filters, out_channels = num_classes, kernel_size = [1, 1])
        )
        #####################
        #      End init     
        #####################
        
    def forward(self, x):
        """
        Pass x forward through the model.
        """
        #   Input size N x C x H x W (batch_size x channels x height x width); for this example, 1 x 1 x 512 x 512
        x = self.down1(x)
        x = self.setSkip(1, x)

        #   Input size 1 x 64 x 256 x 256
        if self.depth > 1:
            x = self.down2(x)
            x = self.setSkip(2, x)

        #   Input size 1 x 128 x 128 x 128
        if self.depth > 2:
            x = self.down3(x)
            x = self.setSkip(3, x)

        #   Input size 1 x 256 x 64 x 64
        if self.depth > 3:
            x = self.down4(x)
            x = self.setSkip(4, x)

        #   Input size 1 x 512 x 32 x 32
        if self.depth > 4:
            x = self.down5(x)

        #   Input size 1 x 512 x 16 x 16 in the example we started from layer 1
        if self.depth > 4:
            x = self.up1(x)
            x = self.addSkip(4, x)

        #   Input size 1 x 512 x 32 x 32
        if self.depth > 3:
            x = self.up2(x)
            x = self.addSkip(3, x)

        #   Input size 1 x 256 x 64 x 64
        if self.depth > 2:
            x = self.up3(x)
            x = self.addSkip(2, x)

        #   Input size 1 x 128 x 128 x 128
        if self.depth > 1:
            x = self.up4(x)
            x = self.addSkip(1, x)

        #   Input size 1 x 64 x 256 x 256
        x = self.up5(x)

        #   Pre-Softmax output - input size 1 x 64 x 512 x 512, output size 1 x 2 x 512 x 512 for binary classification
        x = self.out(x)
        
        return x
        #####################
        #      End forward  
        #####################
    
    def ConvBlock(self, input_channels, n_filters, kernel_size=[3, 3]):
        """
        Builds the conv block for MobileNets
        Apply successivly a 2D convolution, BatchNormalization relu
        Convolution feature map size formula: 
        ((W - K - 2P) / S) + 1, where W = input dimension, K = filter size, P = padding, S = stride
        EX: W = 512, K = 3, P = 1, S = 1 will yield the same size image after convolution
        """
        net = []
        
        # Set padding based on kernel size for same size output
        k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        pad = int(np.floor(k/2))
        
        # Skip pointwise by setting num_outputs=None
        net.append( Conv2d(in_channels = input_channels, out_channels = n_filters, kernel_size=kernel_size, padding=pad) )
        
        net.append( BatchNorm2d(n_filters) )
        
        net.append( ReLU() )
        
        return Sequential(*net)
        #####################
        #      End ConvBlock     
        #####################
    # input 512 , output 256
    
    def ContextModule(self, input_channels, kernel_size=[3, 3]):
        """
        Builds the context module block block for MobileNets
        based on: "MULTI-SCALE CONTEXT AGGREGATION BYDILATED CONVOLUTIONS"
        http://vladlen.info/papers/dilated-convolutions.pdf
        Architecture:
        Layer           1   2   3   4       5       6       7       8
        Convolution     3×3 3×3 3×3 3×3     3×3     3×3     3×3     1×1
        Dilation        1   1   2   4       8       16      1       1 
        Truncation      Yes Yes Yes Yes     Yes     Yes     Yes     No
        Receptive field 3×3 5×5 9×9 17×17   33×33   65×65   67×67   67×67
        Convolution feature map size formula: 
        The context module is designed to increase the performance of dense prediction architectures by
        aggregating multi-scale contextual information. The module takes C feature maps as input and
        produces C feature maps as output. The input and output have the same form, thus the module can
        be plugged into existing dense prediction architectures.
        """
        net = []
        dilations = [1,1,2,4,8,16]
        
        for d in dilations:
            # Get padding to keep output shape same as input shape
            k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
            pad = int(np.floor(k/2))

            # Add Conv layer with dilations
            net.append( Conv2d(in_channels = input_channels, 
                               out_channels = input_channels, 
                               kernel_size=kernel_size, 
                               padding=pad, 
                               dilation=d) )    
        # Pointwise
        net.append( Conv2d(in_channels = input_channels, out_channels = input_channels, kernel_size=[1, 1]) )
        
        #net.append( BatchNorm2d(input_channels) )
        
        #net.append( ReLU() )
        
        return Sequential(*net)
        #####################
        #      End ContextModule     
        #####################
    # input 512 , output 256
    def DepthwiseSeparableConvBlock(self, input_channels, n_filters, kernel_size=[3, 3], slim=False):
        """
        Builds the Depthwise Separable conv block for MobileNets
        Apply successivly a 2D separable convolution, BatchNormalization relu, conv, BatchNormalization, relu
        In order to meet a 4GB Memory limit, this model has been adapted to cut DSW Conv blocks with slim==False
        """
        ##### Replacing slim.separable_convolution2d, which doesn't have an analog in torch
        # How to accomplish depthwise separable convolution:
        # If groups = nInputPlane, then it is Depthwise. 
        # If groups = nInputPlane, kernel=(K, 1), (and before is a Conv2d layer with groups=1 and kernel=(1, K)), then it is separable.
        # https://discuss.pytorch.org/t/depthwise-and-separable-convolutions-in-pytorch/7315
        
        
        net = []
        
        # Sequential(*net) will be empty if self.gpu_4g_limit == True
        if not slim:
        
            # Set padding based on kernel size for same size output
            k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
            pad = int(np.floor(k/2))
            
            # Seperable convolution 
            net.append(
                Conv2d(in_channels = input_channels, out_channels = input_channels, 
                kernel_size=[3,3], groups=input_channels, padding = pad)
            )

            if self.dropout > 0.0:
                net.append( nn.Dropout(self.dropout) )

            net.append( BatchNorm2d(input_channels) )
            net.append( ReLU() )
        
            # Point-wise convolution
            net.append( Conv2d(in_channels = input_channels, out_channels = n_filters, kernel_size=[1, 1]) )

            net.append( BatchNorm2d(n_filters) )
            net.append( ReLU() )

        return Sequential(*net)
        #####################
        #      End DepthwiseSeparableConvBlock
        #####################

    def ConvTransposeBlock(self, input_channels, n_filters, kernel_size=[2, 2]):
        """
        Basic conv transpose block for Encoder-Decoder upsampling
        Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
        Transpose Convolution feature map size formula: 
        S ( W - 1 ) + F - 2P, where W = input dimension, K = filter size, P = padding, S = stride
        EX: W = 256, K = 2, P = 0, S = 2 will yield double the same size image after convolution
        """
        net = []
        
        # Transpose Convolution
        net.append( ConvTranspose2d(in_channels = input_channels, out_channels = n_filters, 
            kernel_size=kernel_size, stride=[2, 2])
        )
        
        # Batch Norm -> Relu
        net.append( ReLU(BatchNorm2d(n_filters)) )
        
        return Sequential(*net)
        #####################
        #      End ConvTransposeBlock
        #####################
                        
    def setSkip(self, layer, x):
        """
        Saves the output of a down-convolution layer to add to corresponding up-convolutions layer.
        If the model is not saving the layer information (!has_skip), simply save a zeros_like variable for the addition.
        The U-Net combines the location information from the downsampling path with the contextual information 
        in the upsampling path to finally obtain a general information combining localisation and context, which is 
        necessary to predict a good segmentation map.
        http://www.deeplearning.net/tutorial/unet.html
        """
        if self.has_skip:
            # in order to add integer key to python dictionary, need to put it inside tuple inside list
            # detach creates a tensor that shares storage with tensor that does not require grad, as opposed to clone
            # https://discuss.pytorch.org/t/clone-and-detach-in-v0-4-0/16861
            self.skips.update([(layer, x.detach())]) 
        else:
            if self.cpu:
                self.skips.update([(layer, (torch.zeros_like(x)).cpu())])
            else:
                self.skips.update([(layer, (torch.zeros_like(x)).cuda(int(gpu)))])

        # Pass x back through to the next module in the model
        
        return x
        #####################
        #      End setSkip
        #####################
        
    def getSkip(self, layer):
        """ Return the saved tensor for a given input layer, specified by layer label """
        return self.skips[layer]
        #####################
        #      End getSkip
        #####################
            
    def addSkip(self, layer, x):
        """ Returns the sum of the saved layer skip variable (zeros_like(x) if !has_skip) and x """
        return x + self.getSkip(layer)
        #####################
        #      End addSkip
        #####################
