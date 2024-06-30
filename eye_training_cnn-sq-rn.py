from __future__ import absolute_import
from __future__ import print_function

import pickle
import time

import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout, Concatenate
from keras.models import Model, load_model,Sequential
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
#from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
#from matplotlib.pyplot import imshow

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json,load_model

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###
    
    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])  
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X
	
def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (f, f), strides = (s,s), padding = 'same', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(F2, (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)


    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(F2, (f, f), strides = (s,s), padding = 'same', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut]) 
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X
	
def ResNet34(input_shape = (100, 100, 1), classes = 1):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64], stage=2, block='b')
    X = identity_block(X, 3, [64, 64], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [128,128], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128,128], stage=3, block='b')
    X = identity_block(X, 3, [128,128], stage=3, block='c')
    X = identity_block(X, 3, [128,128], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [256, 256], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256], stage=4, block='b')
    X = identity_block(X, 3, [256, 256], stage=4, block='c')
    X = identity_block(X, 3, [256, 256], stage=4, block='d')
    X = identity_block(X, 3, [256, 256], stage=4, block='e')
    X = identity_block(X, 3, [256, 256], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters = [512, 512], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512], stage=5, block='b')
    X = identity_block(X, 3, [512, 512], stage=5, block='c')
    print(X.shape)
    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D((3,3), name="avg_pool")(X)
    print(X.shape)
    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    X = Dense(128, activation='relu', name='fc' + str(128), kernel_initializer = glorot_uniform(seed=0))(X)
    X = Dense(classes, activation='sigmoid', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet34')

    return model
	
	
def ResNet18(input_shape = (100, 100, 1), classes = 1):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64], stage=2, block='b')
    #X = identity_block(X, 3, [64, 64], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [128,128], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128,128], stage=3, block='b')
    #X = identity_block(X, 3, [128,128], stage=3, block='c')
    #X = identity_block(X, 3, [128,128], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [256, 256], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256], stage=4, block='b')
    #X = identity_block(X, 3, [256, 256], stage=4, block='c')
    #X = identity_block(X, 3, [256, 256], stage=4, block='d')
    #X = identity_block(X, 3, [256, 256], stage=4, block='e')
    #X = identity_block(X, 3, [256, 256], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters = [512, 512], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512], stage=5, block='b')
    #X = identity_block(X, 3, [512, 512], stage=5, block='c')
    print(X.shape)
    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D((3,3), name="avg_pool")(X)
    print(X.shape)
    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    X = Dense(128, activation='relu', name='fc' + str(128), kernel_initializer = glorot_uniform(seed=0))(X)
    X = Dense(classes, activation='sigmoid', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet18')

    return model
#def ResNet50_B(input_shape = (100, 100, 1), classes = 1):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    #X_input = Input(input_shape)

    
    # Zero-Padding
    #X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    #X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    #X = Activation('relu')(X)
    #X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    #X = convolutional_block(X, f = 3, filters = [64, 64, 128], stage = 2, block='a', s = 1)
    #X = identity_block(X, 3, [64, 64, 128], stage=2, block='b')
    #X = identity_block(X, 3, [64, 64, 128], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    #X = convolutional_block(X, f = 3, filters = [128,128,256], stage = 3, block='a', s = 2)
    #X = identity_block(X, 3, [128,128,256], stage=3, block='b')
    #X = identity_block(X, 3, [128,128,256], stage=3, block='c')
    #X = identity_block(X, 3, [128,128,256], stage=3, block='d')

    # Stage 4 (≈6 lines)
    #X = convolutional_block(X, f = 3, filters = [256, 256, 512], stage = 4, block='a', s = 2)
    #X = identity_block(X, 3, [256, 256, 512], stage=4, block='b')
    #X = identity_block(X, 3, [256, 256, 512], stage=4, block='c')
    #X = identity_block(X, 3, [256, 256, 512], stage=4, block='d')
    #X = identity_block(X, 3, [256, 256, 512], stage=4, block='e')
    #X = identity_block(X, 3, [256, 256, 512], stage=4, block='f')

    # Stage 5 (≈3 lines)
    #X = convolutional_block(X, f = 3, filters = [512, 512, 1024], stage = 5, block='a', s = 2)
    #X = identity_block(X, 3, [512, 512, 1024], stage=5, block='b')
    #X = identity_block(X, 3, [512, 512, 1024], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    #X = AveragePooling2D((2,2), name="avg_pool")(X)
    
    ### END CODE HERE ###

    # output layer
    #X = Flatten()(X)
    #X = Dense(classes, activation='sigmoid', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    #model = Model(inputs = X_input, outputs = X, name='ResNet50')

    #return model
def create_fire_module(x, nb_squeeze_filter, name, use_bypass=False):
    """
    Creates a fire module
    
    Arguments:
        x                 : input
        nb_squeeze_filter : number of filters of squeeze. The filtersize of expand is 4 times of squeeze
        use_bypass        : if True then a bypass will be added
        name              : name of module e.g. fire123
    
    Returns:
        x                 : returns a fire module
    """
    bn_name_base = 'bn' + str(name)  + '_branch'
    nb_expand_filter = 4 * nb_squeeze_filter
    squeeze    = Conv2D(nb_squeeze_filter,(1,1), activation='relu', padding='same', name='%s_squeeze'%name)(x)
    squeeze = BatchNormalization(axis = 3, name = bn_name_base + '2a')(squeeze)
    expand_1x1 = Conv2D(nb_expand_filter, (1,1), activation='relu', padding='same', name='%s_expand_1x1'%name)(squeeze)
    expand_1x1 = BatchNormalization(axis = 3, name = bn_name_base + '2b')(expand_1x1)
    expand_3x3 = Conv2D(nb_expand_filter, (3,3), activation='relu', padding='same', name='%s_expand_3x3'%name)(squeeze)
    expand_3x3 = BatchNormalization(axis = 3, name = bn_name_base + '2c')(expand_3x3)
    axis = get_axis()
    x_ret = Concatenate(axis=axis, name='%s_concatenate'%name)([expand_1x1, expand_3x3])
    
    if use_bypass:
        x_ret = Add(name='%s_concatenate_bypass'%name)([x_ret, x])
        
    return x_ret
def get_axis():
    axis = -1 if K.image_data_format() == 'channels_last' else 1
    return axis	
def SqueezeNet(input_shape = (100, 100, 1), classes = 1):

    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = create_fire_module(X, int(16), name='fire2')
    X = create_fire_module(X, int(16), name='fire3')
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = create_fire_module(X, int(32), name='fire4')
    X = create_fire_module(X, int(32), name='fire5')
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 4 (≈6 lines)
    X = create_fire_module(X, int(48), name='fire6')
    X = create_fire_module(X, int(48), name='fire7')
    X = create_fire_module(X, int(48), name='fire8')
    X = create_fire_module(X, int(48), name='fire9')
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
	
	# Stage 5 (≈6 lines)
    X = create_fire_module(X, int(64), name='fire10')
    X = create_fire_module(X, int(64), name='fire11')
    X = create_fire_module(X, int(64), name='fire12')
    X = create_fire_module(X, int(64), name='fire13')
    #X = MaxPooling2D(strides=(1, 1))(X)
	
    # Stage 6 (≈6 lines)
    #X = create_fire_module(X, int(80), name='fire14')
    #X = create_fire_module(X, int(80), name='fire15')
    #X = create_fire_module(X, int(80), name='fire16')
    #X = create_fire_module(X, int(80), name='fire17')
    #X = MaxPooling2D( strides=(2, 2))(X)
	
	    # Stage 6 (≈6 lines)
    #X = create_fire_module(X, int(96), name='fire10')
    #X = create_fire_module(X, int(96), name='fire11')
    #X = create_fire_module(X, int(96), name='fire12')
    #X = create_fire_module(X, int(96), name='fire13')
    #X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    print(X.shape)
    X = AveragePooling2D((2,2), name="avg_pool")(X)
    X = Flatten()(X)
    #X = Dense(512, name='fc' + str(512), kernel_initializer = glorot_uniform(seed=0))(X)
    #X = BatchNormalization( name = 'bn_conv7')(X)
    #X = Activation('relu')(X)
    X = Dense(128, name='fc' + str(128), kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization( name = 'bn_conv7')(X)
    X = Activation('relu')(X)
    #X = Dense(classes, activation='sigmoid', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    ### END CODE HERE ###

    # output layer
    #X = Flatten()(X)
    X = Dense(classes, activation='sigmoid', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='SqueezeNet')

    return model
def alexnet_model(input_shape = (100, 100, 1), classes = 1):

	# Initialize model
    X_input = Input(input_shape)
    X=X_input
	# Layer 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

	# Layer 2
    X = Conv2D(256, (5, 5), strides = (1,1), padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv2')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

	# Layer 3
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(512, (3, 3), strides = (1,1), padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv3')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

	# Layer 4
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(1024, (3, 3), strides = (1,1), padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv4')(X)
    X = Activation('relu')(X)

	# Layer 5
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(1024, (3, 3), strides = (1,1), padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv5')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
	

	# Layer 6
    X = Flatten()(X)
    X = Dense(3072, kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization( name = 'bn_conv6')(X)
    X = Activation('relu')(X)
    X = Dropout(0.5)(X)
	

	# Layer 7
    X = Dense(4096, kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization( name = 'bn_conv7')(X)
    X = Activation('relu')(X)
    X = Dropout(0.5)(X)
	

	# Layer 8
    X = Dense(classes, activation='sigmoid', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
	#alexnet.add(BatchNormalization())
	#alexnet.add(Activation('softmax'))


    model = Model(inputs = X_input, outputs = X, name='alexnet_model')
    return model
	
def loadModel(model_path, weight_path):
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(weight_path)
    print("Loaded model from disk")
    # evaluate loaded model on test data
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
# Main
if __name__ == '__main__':
    pickle_files = ['open_eyes_all.pickle', 'closed_eyes_all.pickle']
    i = 0
    for pickle_file in pickle_files:
        with open(pickle_file, 'rb') as f:
            save = pickle.load(f)
            if i == 0:
                train_dataset = save['train_dataset']
                train_labels = save['train_labels']
                test_dataset = save['test_dataset']
                test_labels = save['test_labels']
            else:
                print("here")
                train_dataset = np.concatenate((train_dataset, save['train_dataset']))
                train_labels = np.concatenate((train_labels, save['train_labels']))
                test_dataset = np.concatenate((test_dataset, save['test_dataset']))
                test_labels = np.concatenate((test_labels, save['test_labels']))
            del save  # hint to help gc free up memory
        i += 1

    print('Training set', train_dataset.shape, train_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    batch_size = 32
    nb_classes = 1
    nb_epoch = 200

    X_train_orig = train_dataset
    # X_train = X_train.reshape((X_train.shape[0], X_train.shape[3]) + X_train.shape[1:3])
    Y_train_orig = train_labels

    X_test_orig = test_dataset
    # X_test = X_test.reshape((X_test.shape[0], X_test.shape[3]) + X_test.shape[1:3])
    Y_test_orig = test_labels

	# Normalize image vectors
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.

    # Convert training and test labels to one hot matrices
    Y_train = Y_train_orig
    Y_test = Y_test_orig
    # print data shape
    print("{1} train samples, {4} channel{0}, {2}x{3}".format("" if X_train.shape[1] == 1 else "s", *X_train.shape))
    print("{1}  test samples, {4} channel{0}, {2}x{3}".format("" if X_test.shape[1] == 1 else "s", *X_test.shape))
    # input image dimensions
    #datagen = ImageDataGenerator(
			#featurewise_center=False,	# set input mean to 0 over dataset
			#samplewise_center=False,	# set each sample mean to 0
			#featurewise_std_normalization=False,	# divide inputs by std of dataset
			#samplewise_std_normalization=False,	#divide each input by its std
			#zca_whitening=False,	# apply ZCA whitening
			#rotation_range=0,	# randomly roate images in the range (degrees, 0 to 180)
			#width_shift_range=0.1,	# randomly shift image horizontally (fraction of width)
			#height_shift_range=0.1,	# randomly shift image vertically (fraction of height)
			#horizontal_flip=True,	# randomly flip images horizontally
			#vertical_flip=False)	# randomly flip images vertically
    #datagen.fit(X_train)

    model = SqueezeNet(input_shape = (100, 100, 1), classes = 1)

    # let's train the model using SGD + momentum (how original).
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    #model.fit_generator(datagen.flow(X_train, Y_train,
			#batch_size=32),
			#steps_per_epoch=X_train.shape[0] // 32,
			#epochs=20,
			#validation_data=(X_test, Y_test),
			#workers=4)
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test, verbose=0)

    print('Loss score:', score[0])
    print('Test accuracy:', score[1] * 100, '%')

    # Save model to file
    now = time.time()


    print("Save weights to file...")
    model.save('trained_model/weight_' + str(now) + '.h5', overwrite=True)