# -*- coding: utf-8 -*-
import os
os.environ['PYTHONHASHSEED'] = '0'
import numpy as np
np.random.seed(42)
import random as rn
rn.seed(12345)
import tensorflow as tf
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
import keras.backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import SGD
import pdb
import h5py
import sys
sys.path.insert(0, './../essentials/customObjects')
sys.path.insert(0, './../essentials/utils')
import utils
from customLayers import crosschannelnormalization
from customLayers import Softmax4D
from customLayers import splittensor

def AlexNet(weights_path=None, retainTop = True):
    inputs = Input(shape=(227, 227, 3))

    conv_1 = Conv2D(96, (11, 11), strides=(4, 4), activation='relu',
                           name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    conv_2 = crosschannelnormalization(name='convpool_1')(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    concat1 = Concatenate(axis=-1, name='conv_2')
    conv_2 = concat1([
                       Conv2D(128, (5, 5), activation='relu', name='conv_2_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_2)
                       ) for i in range(2)])

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Conv2D(384, (3, 3), activation='relu', name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    concat2 = Concatenate(axis=-1, name='conv_4')
    conv_4 = concat2([
                       Conv2D(192, (3, 3), activation='relu', name='conv_4_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_4)
                       ) for i in range(2)])

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    concat3 = Concatenate(axis=-1, name='conv_5')
    conv_5 = concat3([
                       Conv2D(128, (3, 3), activation='relu', name='conv_5_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_5)
                       ) for i in range(2)])

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name='convpool_5')(conv_5)
    dense_1 = Flatten(name='flatten')(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(1000, name='dense_3')(dense_3)
    prediction = Activation('softmax', name='softmax')(dense_3)
    
    tempModel = Model(inputs=[inputs], outputs=[prediction])
    tempModel.load_weights(weights_path)
    if not retainTop:
      model = Model(inputs=[inputs], outputs=[dense_2])
      lastLayer = dense_2
    else:
      model = tempModel
      lastLayer = prediction
    firstLayer = inputs
    return model, firstLayer, lastLayer

def vgg19(weights_path='./../../../essentials/standardModels/pretrainedWeights/vgg19_weights_tf_dim_ordering_tf_kernels.h5', retainTop = False):
    # -*- coding: utf-8 -*-
    """VGG19 model for Keras.
    # Reference
    - [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
    """



    #WEIGHTS_PATH = 
    #WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'

    classes = 1000
    img_input = Input(shape=(227, 227, 3))
    # Block 1
    conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    conv_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(conv_1)
    max_1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv_2)

    # Block 2
    conv_3 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(max_1)
    conv_4 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(conv_3)
    max_2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv_4)

    # Block 3
    conv_5 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(max_2)
    conv_6 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(conv_5)
    conv_7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(conv_6)
    conv_8 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(conv_7)
    max_3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(conv_8)

    # Block 4
    conv_9 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(max_3)
    conv_10 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(conv_9)
    conv_11 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(conv_10)
    conv_12 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(conv_11)
    max_4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(conv_12)

    # Block 5
    conv_13 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(max_4)
    conv_14 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(conv_13)
    conv_15 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(conv_14)
    conv_16 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(conv_15)
    max_5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(conv_16)
    flat = Flatten(name='flatten')(max_5)
    dense_1 = Dense(4096, activation='relu', name='fc1')(flat)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='fc2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    prediction = Dense(classes, activation='softmax', name='predictions')(dense_3)
    tempModel = Model(img_input, prediction, name='vgg19')

    tempModel.load_weights(weights_path)
    if not retainTop:
      model = Model(inputs=[img_input], outputs=[dense_1])
      lastLayer = dense_2
    else:
      model = tempModel
      lastLayer = prediction
    firstLayer = img_input
    return model, firstLayer, lastLayer