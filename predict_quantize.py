from tensorflow import keras
from imutils import face_utils
from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
from threading import Thread
import pyglet
from pygame import mixer 
import os
from imutils.video import WebcamVideoStream
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)) 
from keras import backend as K 
K.tensorflow_backend._get_available_gpus() 
import time

import pickle
import time

import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input

from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
#from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc


import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


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
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###
    
    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

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
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(F2, (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(F3, (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut]) 
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X
	
def ResNet50(input_shape = (100, 100, 1), classes = 1):
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
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [128,128,512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128,128,512], stage=3, block='b')
    X = identity_block(X, 3, [128,128,512], stage=3, block='c')
    X = identity_block(X, 3, [128,128,512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D((2,2), name="avg_pool")(X)
    
    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='sigmoid', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model



def sound_alarm(path):
    # play an alarm sound
    mixer.init()
    mixer.music.load(filename)
    mixer.music.play()

def loadModel(model_path, weight_path):
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = ResNet50(input_shape = (100, 100, 1), classes = 1)
    # load weights into new model
    model.load_weights(weight_path)
    #print("Loaded model from disk")
    # evaluate loaded model on test data
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def predictImage(img, model):
   # training_data =np.array([[2],[4],[6],[8]])
    #print(training_data)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255

    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    #abs_weights = np.abs(images)
    #vmax = np.max(abs_weights)
    #s = vmax / 127.
    #images = images / s
    #images = np.round(images)
    #images = images.astype(np.int8)
    #print(images)
    #images=preprocess_input(images)
    #inp = model.input                                           # input placeholder
    #outputs = [layer.output for layer in model.layers][1:]          # all layer outputs
    #functors = [K.function([inp], [out]) for out in outputs]    # evaluation functions

# Testing
    #test = np.random.random([3,100,100])[np.newaxis,...]
    #layer_outs = [func([images]) for func in functors]
    #print (layer_outs)
    classes = model.predict(images)
    return classes


if __name__ == "__main__":
    # Define counter
    COUNTER = 0
    ALARM_ON = True
    MAX_FRAME = 10
    name="Bangau"
    # Load model
    base_dir = os.path.dirname(__file__)
    prototxt_path = os.path.join(base_dir + 'trained_model/model_data/deploy.prototxt')
    caffemodel_path = os.path.join(base_dir + 'trained_model/model_data/weights.caffemodel')
	# Read the model_detect face
    model_face = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
	
    model = loadModel('trained_model/model_resnet50.json', "trained_model/weight_resnet50_int8.h5")
    model.save("trained_model/model_resnet50_int8.h5")
    camera = cv2.VideoCapture("output1.mp4")

    # loop over frames from the video stream
    counter = 0
    count_fps=0
    fps = ""
    sum_fps = 0
    t0 = time.time()
    sum_acc=0
    count_acc=0
    while True:
        t1 = time.time()
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale channels)
        ret, frame = camera.read()
        if ret==True:
        #frame = imutils.resize(frame, width = 400)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            model_face.setInput(blob)
            detections = model_face.forward()
        # loop over the face detections
            for i in range(0, detections.shape[2]):
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                confidence = detections[0, 0, i, 2]
                if (confidence > 0.5):
                    roi = gray[startY:endY, startX:endX]
                    shape = cv2.resize(roi,(100, 100))
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), 2)
                #cv2.putText(frame, name, (x+ (w//2),y-2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)
                    counter += 1
                    predict_img = predictImage(shape, model=model)
                    print (predict_img)

                    if predict_img<0.5:
                        COUNTER += 1
                        cv2.putText(frame, "Closing", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, "NF: {:.2f}".format(COUNTER), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # if the eyes were closed for a sufficient number of
                # then sound the alarm
                        if COUNTER >= MAX_FRAME:
                            filename = 'alarm.wav'
                            sound_alarm(filename)

                        # draw an alarm on the frame
                        #cv2.putText(frame, "DROWSY!", (100, 300),
                                #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    else:
                        COUNTER = 0
                        sum_acc = sum_acc + predict_img
                        count_acc =count_acc+1
                    #ALARM_ON = False
                        cv2.putText(frame, "Opening", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            # show the frame
            cv2.putText(frame, fps, (120, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
            elapsedTime = time.time() - t1
            sum_fps=1/elapsedTime +sum_fps
            if count_fps>10:
                print(sum_fps/10)
                sum_fps=0
                count_fps=0
            count_fps=count_fps+1
            fps = "(Playback) {:.1f} FPS".format(1/elapsedTime)
        else:
            break
    tout=time.time()
    print("Accuracy: ", sum_acc/count_acc)
    print("Time Taken:", tout -t0)
    camera.release()
    cv2.destroyAllWindows()
    

