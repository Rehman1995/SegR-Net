#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 15:21:43 2023   

@author: Mobeen 
"""

#%% GPU Selection 
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
!nvidia-smi

#%% Unzip the dataset file
import glob
import zipfile

# importing the zipfile module
from zipfile import ZipFile
  
# loading the temp.zip and creating a zip object
with ZipFile("/home/Mobeen/work_retina/newdatac1n.zip", 'r') as zObject:
  
    # Extracting all the members of the zip 
    # into a specific location.
    zObject.extractall(
        path=("/home/Mobeen/work_retina/"))
#%%DEEPLAB V3 PLUS -- XCEPTION -- RESNET 101
from tensorflow.keras.layers import  Conv2D,BatchNormalization,UpSampling2D,Concatenate,Activation,AveragePooling2D
import os
from tensorflow.keras.layers import Input,Conv2DTranspose
from tensorflow.keras.applications import Xception
from keras.applications.resnet import ResNet50,ResNet101,ResNet152
from tensorflow.keras.models import Model

#Pyramid pooling aspp
#image_pooling->1d conv-->dilated conv with b8=>16=>32
def ASPP(inputs):
    

    #First entire shape pooling
    shape=inputs.shape
    y_pool=AveragePooling2D(pool_size=(shape[1],shape[2]),name='average_pooling')(inputs)
    y_pool=Conv2D(filters=256,kernel_size=1,use_bias=False,padding='same')(y_pool)
    y_pool=BatchNormalization()(y_pool)
    y_pool=Activation(activation='relu')(y_pool)
    y_pool=UpSampling2D(size=(shape[1],shape[2]),interpolation='bilinear')(y_pool)
    #print(y_pool.shape)

    #Now 1-d Channelwise convolution
    y_1=Conv2D(filters=256,kernel_size=1,use_bias=False,padding='same',dilation_rate=1)(inputs)
    y_1=BatchNormalization()(y_1)
    y_1=Activation(activation='relu')(y_1)
    #Now with dilationrate=6
    y_6=Conv2D(filters=256,kernel_size=3,use_bias=False,padding='same',dilation_rate=6)(inputs)
    y_6=BatchNormalization()(y_6)
    y_6=Activation(activation='relu')(y_6)

    #Now with dilationrate=12
    y_12=Conv2D(filters=256,kernel_size=3,use_bias=False,padding='same',dilation_rate=12)(inputs)
    y_12=BatchNormalization()(y_12)
    y_12=Activation(activation='relu')(y_12)

    #Now with dilation rate=18
    y_18=Conv2D(filters=256,kernel_size=3,use_bias=False,padding='same',dilation_rate=18)(inputs)
    y_18=BatchNormalization()(y_18)
    y_18=Activation(activation='relu')(y_18)

    y=Concatenate()([y_pool,y_1,y_6,y_12,y_18])
    #1-d convolution application
    y=Conv2D(filters=256,kernel_size=1,padding='same',dilation_rate=1,use_bias=False)(y)
    y=BatchNormalization()(y)
    y=Activation(activation='relu')(y)
    #print(y.shape)
    return y

def DeepLabv3plusR(shape):
    

    input=Input(shape)
    base_model=ResNet101(include_top=False,weights='imagenet',input_tensor=input)  #Xception
    #base_model.summary()
  
    image_features=base_model.get_layer('conv4_block23_out').output  # block14_sepconv2_act
  
    #Now we will perform atrous asymmetric pyramid pooling
    x_a=ASPP(image_features)
    x_a=UpSampling2D(size=(4,4),interpolation='bilinear')(x_a)
    #Now we will get low level features from our resnet model
    x_b=base_model.get_layer('conv2_block3_out').output # block4_sepconv1_act
    x_b=Conv2D(filters=48,kernel_size=1,padding='same',use_bias=False)(x_b)
    x_b=BatchNormalization()(x_b)
    x_b=Activation(activation='relu')(x_b)
    #Now we will concatenate
    x=Concatenate()([x_a,x_b])
    #print(x.shape)
    #Now apply convolutional layer with 3*3 filter 2 times
    x=Conv2D(filters=256,kernel_size=3,padding='same',use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Activation(activation='relu')(x)

    x=Conv2D(filters=256,kernel_size=3,padding='same',use_bias=False)(x)
    x-BatchNormalization()(x)
    x=Activation(activation='relu')(x)

    x=UpSampling2D(size=(4,4),interpolation='bilinear')(x)
    #print(x.shape)
    #outputs
    x=Conv2D(1,(1,1),name='output_layer')(x)
    x=Activation(activation='sigmoid')(x)
    #print(x.shape)
    #Model
    model=Model(inputs=input,outputs=x)
    return model
     
#%%
from tensorflow.keras.layers import  Conv2D,BatchNormalization,UpSampling2D,Concatenate,Activation,AveragePooling2D
import os
from tensorflow.keras.layers import Input,Conv2DTranspose
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model

#Pyramid pooling aspp
#image_pooling->1d conv-->dilated conv with b8=>16=>32
def ASPP(inputs):
    

    #First entire shape pooling
    shape=inputs.shape
    y_pool=AveragePooling2D(pool_size=(shape[1],shape[2]),name='average_pooling')(inputs)
    y_pool=Conv2D(filters=256,kernel_size=1,use_bias=False,padding='same')(y_pool)
    y_pool=BatchNormalization()(y_pool)
    y_pool=Activation(activation='relu')(y_pool)
    y_pool=UpSampling2D(size=(shape[1],shape[2]),interpolation='bilinear')(y_pool)
    #print(y_pool.shape)

    #Now 1-d Channelwise convolution
    y_1=Conv2D(filters=256,kernel_size=1,use_bias=False,padding='same',dilation_rate=1)(inputs)
    y_1=BatchNormalization()(y_1)
    y_1=Activation(activation='relu')(y_1)
    #Now with dilationrate=6
    y_6=Conv2D(filters=256,kernel_size=3,use_bias=False,padding='same',dilation_rate=6)(inputs)
    y_6=BatchNormalization()(y_6)
    y_6=Activation(activation='relu')(y_6)

    #Now with dilationrate=12
    y_12=Conv2D(filters=256,kernel_size=3,use_bias=False,padding='same',dilation_rate=12)(inputs)
    y_12=BatchNormalization()(y_12)
    y_12=Activation(activation='relu')(y_12)

    #Now with dilation rate=18
    y_18=Conv2D(filters=256,kernel_size=3,use_bias=False,padding='same',dilation_rate=18)(inputs)
    y_18=BatchNormalization()(y_18)
    y_18=Activation(activation='relu')(y_18)

    y=Concatenate()([y_pool,y_1,y_6,y_12,y_18])
    #1-d convolution application
    y=Conv2D(filters=256,kernel_size=1,padding='same',dilation_rate=1,use_bias=False)(y)
    y=BatchNormalization()(y)
    y=Activation(activation='relu')(y)
    #print(y.shape)
    return y

def DeepLabv3plusX(shape):
    

    input=Input(shape)
    base_model=Xception(include_top=False,weights='imagenet',input_tensor=input)
    #base_model.summary()
  
    image_features=base_model.get_layer('block14_sepconv2_act').output
  
    #Now we will perform atrous asymmetric pyramid pooling
    x_a=ASPP(image_features)
    x_a=UpSampling2D(size=(4,4),interpolation='bilinear')(x_a)
    #Now we will get low level features from our resnet model
    x_b=base_model.get_layer('block4_sepconv1_act').output
    x_b=Conv2D(filters=48,kernel_size=1,padding='same',use_bias=False)(x_b)
    x_b=BatchNormalization()(x_b)
    x_b=Activation(activation='relu')(x_b)
    #Now we will concatenate
    x=Concatenate()([x_a,x_b])
    #print(x.shape)
    #Now apply convolutional layer with 3*3 filter 2 times
    x=Conv2D(filters=256,kernel_size=3,padding='same',use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Activation(activation='relu')(x)

    x=Conv2D(filters=256,kernel_size=3,padding='same',use_bias=False)(x)
    x-BatchNormalization()(x)
    x=Activation(activation='relu')(x)

    x=UpSampling2D(size=(8,8),interpolation='bilinear')(x)
    #print(x.shape)
    #outputs
    x=Conv2D(1,(1,1),name='output_layer')(x)
    x=Activation(activation='sigmoid')(x)
    #print(x.shape)
    #Model
    model1=Model(inputs=input,outputs=x)
    return model1
#%% Residual Unet  
#%% this one comparision
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K


#convolutional block
def conv_block(x, kernelsize, filters, dropout, batchnorm=False): 
    conv = layers.Conv2D(filters, (kernelsize, kernelsize), kernel_initializer='he_normal', padding="same")(x)
    if batchnorm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)
    conv = layers.Conv2D(filters, (kernelsize, kernelsize), kernel_initializer='he_normal', padding="same")(conv)
    if batchnorm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)
    return conv


#residual convolutional block
def res_conv_block(x, kernelsize, filters, dropout, batchnorm=False):
    conv1 = layers.Conv2D(filters, (kernelsize, kernelsize), kernel_initializer='he_normal', padding='same')(x)
    if batchnorm is True:
        conv1 = layers.BatchNormalization(axis=3)(conv1)
    conv1 = layers.Activation('relu')(conv1)    
    conv2 = layers.Conv2D(filters, (kernelsize, kernelsize), kernel_initializer='he_normal', padding='same')(conv1)
    if batchnorm is True:
        conv2 = layers.BatchNormalization(axis=3)(conv2)
        conv2 = layers.Activation("relu")(conv2)
    if dropout > 0:
        conv2 = layers.Dropout(dropout)(conv2)
        
    #skip connection    
    shortcut = layers.Conv2D(filters, kernel_size=(1, 1), kernel_initializer='he_normal', padding='same')(x)
    if batchnorm is True:
        shortcut = layers.BatchNormalization(axis=3)(shortcut)
    shortcut = layers.Activation("relu")(shortcut)
    respath = layers.add([shortcut, conv2])       
    return respath


#gating signal for attention unit
def gatingsignal(input, out_size, batchnorm=False):
    x = layers.Conv2D(out_size, (1, 1), padding='same')(input)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

#attention unit/block based on soft attention
def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), kernel_initializer='he_normal', padding='same')(x) 
    shape_theta_x = K.int_shape(theta_x)
    phi_g = layers.Conv2D(inter_shape, (1, 1), kernel_initializer='he_normal', padding='same')(gating)
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3), strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]), kernel_initializer='he_normal', padding='same')(phi_g)
    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), kernel_initializer='he_normal', padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg) 
    upsample_psi = layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': shape_x[3]})(upsample_psi)                          
    y = layers.multiply([upsample_psi, x])
    result = layers.Conv2D(shape_x[3], (1, 1), kernel_initializer='he_normal', padding='same')(y)
    attenblock = layers.BatchNormalization()(result)
    return attenblock

  

#%%Res-UNET
def residualunet(input_shape, dropout=0.2, batchnorm=True):

    filters =[64, 128, 256, 512, 1024]# [16, 32, 64, 128, 256]
    kernelsize = 3
    upsample_size = 2

    inputs = layers.Input(input_shape) 

    # Downsampling layers    
    dn_conv1 = conv_block(inputs, kernelsize, filters[0], dropout, batchnorm)
    dn_pool1 = layers.MaxPooling2D(pool_size=(2,2))(dn_conv1)

    dn_conv2 = res_conv_block(dn_pool1, kernelsize, filters[1], dropout, batchnorm)
    dn_pool2 = layers.MaxPooling2D(pool_size=(2,2))(dn_conv2)

    dn_conv3 = res_conv_block(dn_pool2, kernelsize, filters[2], dropout, batchnorm)
    dn_pool3 = layers.MaxPooling2D(pool_size=(2,2))(dn_conv3)

    dn_conv4 = res_conv_block(dn_pool3, kernelsize, filters[3], dropout, batchnorm)
    dn_pool4 = layers.MaxPooling2D(pool_size=(2,2))(dn_conv4)

    dn_conv5 = res_conv_block(dn_pool4, kernelsize, filters[4], dropout, batchnorm)
   
    # upsampling layers
    up_conv6 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(dn_conv5)
    up_conv6 = layers.concatenate([up_conv6, dn_conv4], axis=3)
    up_conv6 = res_conv_block(up_conv6, kernelsize, filters[3], dropout, batchnorm)

    up_conv7 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv6)
    up_conv7 = layers.concatenate([up_conv7, dn_conv3], axis=3)
    up_conv7 = res_conv_block(up_conv7, kernelsize, filters[2], dropout, batchnorm)

    up_conv8 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv7)
    up_conv8 = layers.concatenate([up_conv8, dn_conv2], axis=3)
    up_conv8 = res_conv_block(up_conv8, kernelsize, filters[1], dropout, batchnorm)

    up_conv9 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv8)
    up_conv9 = layers.concatenate([up_conv9, dn_conv1], axis=3)
    up_conv9 = res_conv_block(up_conv9, kernelsize, filters[0], dropout, batchnorm)


    conv_final = layers.Conv2D(1, kernel_size=(1,1))(up_conv9)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    outputs = layers.Activation('sigmoid')(conv_final) 
    
    model = models.Model(inputs=[inputs], outputs=[outputs])
    #model.summary()
    return model

#%% Attention Unet
import tensorflow as tf 
from keras.models import *
from keras.layers import *

def conv_block(input, filters):
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(input)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out

def up_conv(input, filters):
    out = UpSampling2D()(input)
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out

def Attention_block(input1, input2, filters):
    g1 = Conv2D(filters, kernel_size=1, strides=1, padding='same')(input1)
    g1 = BatchNormalization()(g1)
    x1 = Conv2D(filters, kernel_size=1, strides=1, padding='same')(input2)
    x1 = BatchNormalization()(x1)
    psi = Activation('relu')(add([g1, x1]))
    psi = Conv2D(filters, kernel_size=1, strides=1, padding='same')(psi)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)
    out = multiply([input2, psi])
    return out
    

def AttUNet(nClasses, input_height=224, input_width=224):
    
    inputs = Input(shape=(input_height, input_width, 3))
    n1 = 64
    filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

    e1 = conv_block(inputs, filters[0])

    e2 = MaxPooling2D(strides=2)(e1)
    e2 = conv_block(e2, filters[1])

    e3 = MaxPooling2D(strides=2)(e2)
    e3 = conv_block(e3, filters[2])

    e4 = MaxPooling2D(strides=2)(e3)
    e4 = conv_block(e4, filters[3])

    e5 = MaxPooling2D(strides=2)(e4)
    e5 = conv_block(e5, filters[4])

    d5 = up_conv(e5, filters[3])
    x4 =  Attention_block(d5, e4, filters[3])
    d5 = Concatenate()([x4, d5])
    d5 = conv_block(d5, filters[3])

    d4 = up_conv(d5, filters[2])
    x3 =  Attention_block(d4, e3, filters[2])
    d4 = Concatenate()([x3, d4])
    d4 = conv_block(d4, filters[2])

    d3 = up_conv(d4, filters[1])
    x2 =  Attention_block(d3, e2, filters[1])
    d3 = Concatenate()([x2, d3])
    d3 = conv_block(d3, filters[1])

    d2 = up_conv(d3, filters[0])
    x1 =  Attention_block(d2, e1, filters[0])
    d2 = Concatenate()([x1, d2])
    d2 = conv_block(d2, filters[0])

    o = Conv2D(nClasses, (3, 3), padding='same')(d2)

    out = Activation('sigmoid')(o)

    model = Model(inputs, out)
    

    return model
#%% Unet 
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="UNET")
    return model
#%% 1
from tensorflow.keras.layers import  Conv2D,BatchNormalization,UpSampling2D,Concatenate,Activation,AveragePooling2D
import os
from tensorflow.keras.layers import Input,Conv2DTranspose
from tensorflow.keras.applications import Xception
from keras.applications.resnet import ResNet50,ResNet101,ResNet152
from keras.applications.resnet_v2 import ResNet50V2,ResNet101V2,ResNet152V2
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
#Pyramid pooling aspp
#image_pooling->1d conv-->dilated conv with b8=>16=>32


#Pyramid pooling aspp
#image_pooling->1d conv-->dilated conv with b8=>16=>32
def ASPP(inputs):
    

    #First entire shape pooling
    shape=inputs.shape
    y_pool=AveragePooling2D(pool_size=(shape[1],shape[2]),name='average_pooling')(inputs)
    y_pool=Conv2D(filters=256,kernel_size=1,use_bias=False,padding='same')(y_pool)
    y_pool=BatchNormalization()(y_pool)
    y_pool=Activation(activation='relu')(y_pool)
    y_pool=UpSampling2D(size=(shape[1],shape[2]),interpolation='bilinear')(y_pool)
    #print(y_pool.shape)

    #Now 1-d Channelwise convolution
    y_1=Conv2D(filters=256,kernel_size=1,use_bias=False,padding='same',dilation_rate=1)(inputs)
    y_1=BatchNormalization()(y_1)
    y_1=Activation(activation='relu')(y_1)
    #Now with dilationrate=6
    y_6=Conv2D(filters=256,kernel_size=3,use_bias=False,padding='same',dilation_rate=6)(inputs)
    y_6=BatchNormalization()(y_6)
    y_6=Activation(activation='relu')(y_6)

    #Now with dilationrate=12
    y_12=Conv2D(filters=256,kernel_size=3,use_bias=False,padding='same',dilation_rate=12)(inputs)
    y_12=BatchNormalization()(y_12)
    y_12=Activation(activation='relu')(y_12)

    #Now with dilation rate=18
    y_18=Conv2D(filters=256,kernel_size=3,use_bias=False,padding='same',dilation_rate=18)(inputs)
    y_18=BatchNormalization()(y_18)
    y_18=Activation(activation='relu')(y_18)

    y=Concatenate()([y_pool,y_1,y_6,y_12,y_18])
    #1-d convolution application
    y=Conv2D(filters=256,kernel_size=1,padding='same',dilation_rate=1,use_bias=False)(y)
    y=BatchNormalization()(y)
    y=Activation(activation='relu')(y)
    #print(y.shape)
    return y

def DeepLabv3plus(shape):
    

    input=Input(shape)
    base_model=ResNet50V2(include_top=False,weights='imagenet',input_tensor=input)  #Xception
    #base_model.summary()
  
    image_features=base_model.get_layer('conv4_block6_1_relu').output  # block14_sepconv2_act
  
    #Now we will perform atrous asymmetric pyramid pooling
    x_a=ASPP(image_features)
    x_a=UpSampling2D(size=(4,4),interpolation='bilinear')(x_a)
    #Now we will get low level features from our resnet model
    x_b=base_model.get_layer('conv2_block3_1_relu').output # block4_sepconv1_act
    x_b=Conv2D(filters=48,kernel_size=1,padding='same',use_bias=False)(x_b)
    x_b=BatchNormalization()(x_b)
    x_b=Activation(activation='relu')(x_b)
    #Now we will concatenate
    x=Concatenate()([x_a,x_b])
    #print(x.shape)
    #Now apply convolutional layer with 3*3 filter 2 times
    x=Conv2D(filters=256,kernel_size=3,padding='same',use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Activation(activation='relu')(x)

    x=Conv2D(filters=256,kernel_size=3,padding='same',use_bias=False)(x)
    x-BatchNormalization()(x)
    x=Activation(activation='relu')(x)

    x=UpSampling2D(size=(4,4),interpolation='bilinear')(x)
    #print(x.shape)
    #outputs
    x=Conv2D(1,(1,1),name='output_layer')(x)
    x=Activation(activation='sigmoid')(x)
    #print(x.shape)
    #Model
    model=Model(inputs=input,outputs=x)
    return model
#%% 2
from tensorflow.keras.layers import  Conv2D,BatchNormalization,UpSampling2D,Concatenate,Activation,AveragePooling2D
import os
from tensorflow.keras.layers import Input,Conv2DTranspose
from tensorflow.keras.applications import Xception
from keras.applications.resnet import ResNet50,ResNet101,ResNet152
from keras.applications.resnet_v2 import ResNet50V2,ResNet101V2,ResNet152V2
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
#Pyramid pooling aspp
#image_pooling->1d conv-->dilated conv with b8=>16=>32


def DFM(input):
  x = Conv2D(filters = (16), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x1 = Conv2D(filters = (16), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(x)
  x1 = BatchNormalization()(x1)
  x1 = Activation('relu')(x1)

  x2 = Conv2D(filters = (16), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x)
  x2 = BatchNormalization()(x2)
  x2 = Activation('relu')(x2)
  x2 = Conv2D(filters = (16), kernel_size = (2, 2),kernel_initializer = 'he_normal', padding = 'same')(x2)
  x2= BatchNormalization()(x2)
  x2 = Activation('relu')(x2)
  x2= Dropout(0.3)(x2)

  x3 = concatenate([x1, x2])

  x4 = Conv2D(filters = (16), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x3)
  x4 = BatchNormalization()(x4)
  x4 = Activation('relu')(x4)
  x4 = Conv2D(filters = (16), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x4)
  x4= BatchNormalization()(x4)
  x4 = Activation('relu')(x4)
  x4= Dropout(0.3)(x4)

  x5 = concatenate([x3, x4])  
  return x5


def FEE(input):
  

  x41 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(input)
  x41 = BatchNormalization()(x41)
  x41 = Activation('relu')(x41)
  x41 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x41)
  x41= BatchNormalization()(x41)
  x41 = Activation('relu')(x41)
  x41= Dropout(0.3)(x41)

  x55 = concatenate([input, x41]) 

  x45 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x55)
  x45 = BatchNormalization()(x55)
  x45 = Activation('relu')(x55)

  x56 = concatenate([x45, x55]) 

  x411 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x56)
  x411 = BatchNormalization()(x411)
  x411 = Activation('relu')(x411)
  x411 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x411)
  x411= BatchNormalization()(x411)
  x411 = Activation('relu')(x411)
  x411= Dropout(0.3)(x411)

  return x411


#Pyramid pooling aspp
#image_pooling->1d conv-->dilated conv with b8=>16=>32
def ASPP(inputs):
    

    #First entire shape pooling
    shape=inputs.shape
    y_pool=AveragePooling2D(pool_size=(shape[1],shape[2]),name='average_pooling')(inputs)
    y_pool=Conv2D(filters=256,kernel_size=1,use_bias=False,padding='same')(y_pool)
    y_pool=BatchNormalization()(y_pool)
    y_pool=Activation(activation='relu')(y_pool)
    y_pool=UpSampling2D(size=(shape[1],shape[2]),interpolation='bilinear')(y_pool)
    #print(y_pool.shape)

    #Now 1-d Channelwise convolution
    y_1=Conv2D(filters=256,kernel_size=1,use_bias=False,padding='same',dilation_rate=1)(inputs)
    y_1=BatchNormalization()(y_1)
    y_1=Activation(activation='relu')(y_1)
    #Now with dilationrate=6
    y_6=Conv2D(filters=256,kernel_size=3,use_bias=False,padding='same',dilation_rate=6)(inputs)
    y_6=BatchNormalization()(y_6)
    y_6=Activation(activation='relu')(y_6)

    #Now with dilationrate=12
    y_12=Conv2D(filters=256,kernel_size=3,use_bias=False,padding='same',dilation_rate=12)(inputs)
    y_12=BatchNormalization()(y_12)
    y_12=Activation(activation='relu')(y_12)

    #Now with dilation rate=18
    y_18=Conv2D(filters=256,kernel_size=3,use_bias=False,padding='same',dilation_rate=18)(inputs)
    y_18=BatchNormalization()(y_18)
    y_18=Activation(activation='relu')(y_18)

    y=Concatenate()([y_pool,y_1,y_6,y_12,y_18])

    #y1=FEE(y)
    #y2=DFM(y1)
    #1-d convolution application
    y=Conv2D(filters=256,kernel_size=1,padding='same',dilation_rate=1,use_bias=False)(y)
    y=BatchNormalization()(y)
    y=Activation(activation='relu')(y)
    #print(y.shape)
    return y



def DeepLabv3plus1(shape):
    

    input=Input(shape)
    base_model=ResNet50V2(include_top=False,weights='imagenet',input_tensor=input)  #Xception
    #base_model.summary()
  
    image_features=base_model.get_layer('conv4_block6_1_relu').output  # block14_sepconv2_act
  
    #Now we will perform atrous asymmetric pyramid pooling
    x_a=ASPP(image_features)
    #x_a=UpSampling2D(size=(4,4),interpolation='bilinear')(x_a)

   # x_a1=FEE(image_features)
   # x_a=Concatenate()([x_a,x_a1])
    #x_a=DFM(x_a)
    x_a1=FEE(image_features)
    
    x_a1=DFM(x_a1)

    x_a=Concatenate()([x_a,x_a1])



    x121 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x_a)
    x121 = BatchNormalization()(x121)
    x121 = Activation('relu')(x121)
    x121 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x121)
    x121= BatchNormalization()(x121)
    x121= Activation('relu')(x121)
    x121= Dropout(0.2)(x121)



    x_a=UpSampling2D(size=(4,4),interpolation='bilinear')(x121)
    
    #Now we will get low level features from our resnet model
    x_b=base_model.get_layer('conv2_block3_1_relu').output # block4_sepconv1_act
    x_b=Conv2D(filters=48,kernel_size=1,padding='same',use_bias=False)(x_b)
    x_b=BatchNormalization()(x_b)
    x_b=Activation(activation='relu')(x_b)
    #Now we will concatenate
    x=Concatenate()([x_a,x_b])
    #print(x.shape)
    #Now apply convolutional layer with 3*3 filter 2 times
    x=Conv2D(filters=256,kernel_size=3,padding='same',use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Activation(activation='relu')(x)

    x=Conv2D(filters=256,kernel_size=3,padding='same',use_bias=False)(x)
    x-BatchNormalization()(x)
    x=Activation(activation='relu')(x)

    x=UpSampling2D(size=(4,4),interpolation='bilinear')(x)
    #print(x.shape)
    #outputs
    x=Conv2D(1,(1,1),name='output_layer')(x)
    x=Activation(activation='sigmoid')(x)
    #print(x.shape)
    #Model
    model=Model(inputs=input,outputs=x)
    return model
     
#%% 3

# Initial Model Summary -- DEEPlabv3plus
shape=(256,256,3)
#model=DeepLabv3plusR(shape)
#model=DeepLabv3plusX(shape)
#model=attentionunet(shape, dropout=0.01, batchnorm=True)
#model=att_unet(256, 256, 1, data_format='channels_first')
#model=AttUNet(1, input_height=224, input_width=224)
#model=R2UNet(1, input_height=224, input_width=224)
model=residualunet(shape, dropout=0.01, batchnorm=True)

#input_shape = (512, 512, 3)
#model = build_unet(input_shape)
model.summary()


#%% Model Initial Summary - UNET
H=256
W=256
model=unetmodel(input_shape, dropout=0.05, batchnorm=True)
model.summary()
#%% mRbVseG

# use 0.7 0.5 , 0.6 0.4 , 0.6 0.5 combinations, also try changing lr patience value

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *

def conv2d_block(input_tensor, batchnorm = True):
    
    # 1st layer
    x = Conv2D(filters = 32, kernel_size = (2,2),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 2nd layer
    x = Conv2D(filters = 32, kernel_size = (3,3),\
              kernel_initializer = 'he_normal', padding = 'same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def DFM(input):
  x = Conv2D(filters = (16), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x1 = Conv2D(filters = (16), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(x)
  x1 = BatchNormalization()(x1)
  x1 = Activation('relu')(x1)

  x2 = Conv2D(filters = (16), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x)
  x2 = BatchNormalization()(x2)
  x2 = Activation('relu')(x2)
  x2 = Conv2D(filters = (16), kernel_size = (2, 2),kernel_initializer = 'he_normal', padding = 'same')(x2)
  x2= BatchNormalization()(x2)
  x2 = Activation('relu')(x2)
  x2= Dropout(0.3)(x2)

  x3 = concatenate([x1, x2])

  x4 = Conv2D(filters = (16), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x3)
  x4 = BatchNormalization()(x4)
  x4 = Activation('relu')(x4)
  x4 = Conv2D(filters = (16), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x4)
  x4= BatchNormalization()(x4)
  x4 = Activation('relu')(x4)
  x4= Dropout(0.3)(x4)

  x5 = concatenate([x3, x4])  
  return x5


def DMFF(image_features):
    
    shape = image_features.shape
    
    y_pool = AveragePooling2D(pool_size=(shape[1], shape[2]))(image_features)
    y_pool = Conv2D(filters=96, kernel_size=1, padding='same', use_bias=False)(y_pool)
    y_pool = BatchNormalization()(y_pool)
    y_pool = Activation('relu')(y_pool)
    y_pool = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y_pool)

    y_6 = Conv2D(filters=32, kernel_size=3, padding='same', dilation_rate = 9,use_bias=False)(y_pool)
    y_6 = BatchNormalization()(y_6)
    y_6 = Activation('relu')(y_6)
    
    #y_c11 = Concatenate()([y_pool, y_6])
    
    y_61 = Conv2D(filters=16, kernel_size=3, padding='same', dilation_rate = 6,use_bias=False)(image_features)
    y_61 = BatchNormalization()(y_61)
    y_61 = Activation('relu')(y_61)
    
    y_611 = Conv2D(filters=32, kernel_size=3, padding='same', dilation_rate = 12,use_bias=False)(y_61)
    y_611 = BatchNormalization()(y_611)
    y_611 = Activation('relu')(y_611)
    
    y_6111 = Conv2D(filters=16, kernel_size=3, padding='same', dilation_rate = 6,use_bias=False)(image_features)
    y_6111 = BatchNormalization()(y_6111)
    y_6111 = Activation('relu')(y_6111)
    
    
    y_c21 = Concatenate()([y_6 ,y_61, y_611 ,y_6111 ])
    
    x414 = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(y_c21)
    x414 = BatchNormalization()(x414)
    x414 = Activation('relu')(x414)
    
    x414= Dropout(0.7)(x414) #0.6 for chase  # 0.7 for stare, # 0.7 drive
    
    return x414
    



def FEE(input):

  x41 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(input)
  x41 = BatchNormalization()(x41)
  x41 = Activation('relu')(x41)
  x41 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x41)
  x41= BatchNormalization()(x41)
  x41 = Activation('relu')(x41)
  x41= Dropout(0.3)(x41)

  x55 = concatenate([input, x41]) 

  x45 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x55)
  x45 = BatchNormalization()(x55)
  x45 = Activation('relu')(x55)

  x56 = concatenate([x45, x55]) 

  x411 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x56)
  x411 = BatchNormalization()(x411)
  x411 = Activation('relu')(x411)
  x411 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x411)
  x411= BatchNormalization()(x411)
  x411 = Activation('relu')(x411)
  x411= Dropout(0.3)(x411)

  return x411



def FPI(input):
    xh6 = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(input)
    xh6= BatchNormalization()(xh6)
    xh6 = Activation('relu')(xh6)
    xh6= Dropout(0.3)(xh6)

    xh33 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(input)
    xh33= BatchNormalization()(xh33)
    xh33 = Activation('relu')(xh33)
    xh44 = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(input)
    xh44= BatchNormalization()(xh44)
    xh44 = Activation('relu')(xh44)
    xh44= Dropout(0.5)(xh44) # 0.4 for chase , 0.5 for stare, 0.5 drive
    
    yh4= Concatenate()([xh6,xh33,xh44])
    return yh4



def mRbVseG(input_img, n_filters = 64, dropout = 0.3, batchnorm = True):
    
    # Contracting Path
    #c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    c1=FEE(input_img)

    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    #c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    c2=FEE(p1)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    #c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    c3=FEE(p2)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
   # c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    c4=FEE(p3)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    #c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    #c5=tt(p4)
    
    # Expansive Path
    #u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    #u6 = concatenate([u6, c4])
    c4a = DFM(c4)
    c3a = DFM(c3)
    c2a = DFM(c2)
    c1a = DFM(c1)
    #u6 = concatenate([u6, c4a])
    #u6 = Dropout(dropout)(u6)
    #c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    x121 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(c4a)
    x121 = BatchNormalization()(x121)
    x121 = Activation('relu')(x121)
    x121 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x121)
    x121= BatchNormalization()(x121)
    x121= Activation('relu')(x121)
    x121= Dropout(0.2)(x121)
    
    x122 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(c3a)
    x122 = BatchNormalization()(x122)
    x122 = Activation('relu')(x122)
    x122 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x122)
    x122= BatchNormalization()(x122)
    x122= Activation('relu')(x122)
    x122= Dropout(0.3)(x122)

    
    #c2a = CPT(c2)
   # c1a = CPT(c1)

    x121 = Conv2DTranspose(32, (1, 1), strides = (4, 4), padding = 'same')(x121)
    u11 = concatenate([c2a, x121])

    

    x122= Conv2DTranspose(32, (1, 1), strides = (4, 4), padding = 'same')(x122)
    u12 = concatenate([c1a, x122])
    
    x123 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(u11)# c2a
    x123 = BatchNormalization()(x123)
    x123 = Activation('relu')(x123)
    x123 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x123)
    x123= BatchNormalization()(x123)
    x123= Activation('relu')(x123)
    x123= Dropout(0.3)(x123)

    #u13 = concatenate([c2a, x123])
    u13 = Conv2DTranspose(32, (1, 1), strides = (2, 2), padding = 'same')(x123)

    
    x124 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(u12)#c1a
    x124 = BatchNormalization()(x124)
    x124 = Activation('relu')(x124)
    x124 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x124)
    x124= BatchNormalization()(x124)
    x124= Activation('relu')(x124)
    x124= Dropout(0.3)(x124)

    u14 = concatenate([u13, x124])
    
    xj = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(u14)
    xj= BatchNormalization()(xj)
    xj= Activation('relu')(xj)
    #u11 = Conv2DTranspose(112, (1, 1), strides = (8, 8), padding = 'same')(u11)
    
    u16 = Conv2DTranspose(32, (1, 1), strides = (2, 2), padding = 'same')(x121)
    u17 = Conv2DTranspose(32, (1, 1), strides = (2, 2), padding = 'same')(x123)
    yja = Concatenate()([u16,u17,x122])
    #yja=sap(yja)
    
   
    #yca = Concatenate()([xj, yja])
    yja1= FPI(yja)
    #yja2=CPT(yja)
    #yja3= Concatenate()([yja,yja1,yja2])
    
    
    #xj=sap(xj)
    
  
    
    #yca3=sap1(yca)
    
    ycb = Concatenate()([xj,yja1])
    ycb1=DMFF(ycb)
  
    
    
    #yca=channelattention()

    yca2 = Concatenate()([ycb,ycb1])
    


    outputs = Conv2D(1, (1, 1), activation='sigmoid')(yca2)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
#%% Model Initial Summary nRbVseG
input_img = Input((256, 256, 3), name='img')
#model=AttUNet(nClasses=1, input_height=224, input_width=224)
model=mRbVseG(input_img,dropout = 0.3, batchnorm = True)
model.summary()
#%% ABLATION STUDY
* Conventional Encoder & Conventional Decoder
* FEE & DFM + Conventional Decoder
* Conventional Encoder + FPI & DMFF
* FEE & DFM + FPI
* FEE & DFM + DMFF
* FEE + FPI & DMFF
* DFM + FPI & DMFF
* Proposed mRbVseG
#%% Conventional Encoder & Conventional Decoder
#%%  BASELINE MODEL

# set prefetch =7
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *

def conv2d_block(input_tensor, batchnorm = True):
    
    # 1st layer
    x = Conv2D(filters = 32, kernel_size = (2,2),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 2nd layer
    x = Conv2D(filters = 32, kernel_size = (3,3),\
              kernel_initializer = 'he_normal', padding = 'same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def DFM(input):
  x = Conv2D(filters = (16), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x1 = Conv2D(filters = (16), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(x)
  x1 = BatchNormalization()(x1)
  x1 = Activation('relu')(x1)

  x2 = Conv2D(filters = (16), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x)
  x2 = BatchNormalization()(x2)
  x2 = Activation('relu')(x2)
  x2 = Conv2D(filters = (16), kernel_size = (2, 2),kernel_initializer = 'he_normal', padding = 'same')(x2)
  x2= BatchNormalization()(x2)
  x2 = Activation('relu')(x2)
  x2= Dropout(0.3)(x2)

  x3 = concatenate([x1, x2])

  x4 = Conv2D(filters = (16), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x3)
  x4 = BatchNormalization()(x4)
  x4 = Activation('relu')(x4)
  x4 = Conv2D(filters = (16), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x4)
  x4= BatchNormalization()(x4)
  x4 = Activation('relu')(x4)
  x4= Dropout(0.3)(x4)

  x5 = concatenate([x3, x4])  
  return x5


def DMFF(image_features):
    
    shape = image_features.shape
    
    y_pool = AveragePooling2D(pool_size=(shape[1], shape[2]))(image_features)
    y_pool = Conv2D(filters=96, kernel_size=1, padding='same', use_bias=False)(y_pool)
    y_pool = BatchNormalization()(y_pool)
    y_pool = Activation('relu')(y_pool)
    y_pool = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y_pool)

    y_6 = Conv2D(filters=32, kernel_size=3, padding='same', dilation_rate = 9,use_bias=False)(y_pool)
    y_6 = BatchNormalization()(y_6)
    y_6 = Activation('relu')(y_6)
    
    #y_c11 = Concatenate()([y_pool, y_6])
    
    y_61 = Conv2D(filters=16, kernel_size=3, padding='same', dilation_rate = 6,use_bias=False)(image_features)
    y_61 = BatchNormalization()(y_61)
    y_61 = Activation('relu')(y_61)
    
    y_611 = Conv2D(filters=32, kernel_size=3, padding='same', dilation_rate = 12,use_bias=False)(y_61)
    y_611 = BatchNormalization()(y_611)
    y_611 = Activation('relu')(y_611)
    
    y_6111 = Conv2D(filters=16, kernel_size=3, padding='same', dilation_rate = 6,use_bias=False)(image_features)
    y_6111 = BatchNormalization()(y_6111)
    y_6111 = Activation('relu')(y_6111)
    
    
    y_c21 = Concatenate()([y_6 ,y_61, y_611 ,y_6111 ])
    
    x414 = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(y_c21)
    x414 = BatchNormalization()(x414)
    x414 = Activation('relu')(x414)
    
    x414= Dropout(0.6)(x414)
    
    return x414
    



def FEE(input):

  x41 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(input)
  x41 = BatchNormalization()(x41)
  x41 = Activation('relu')(x41)
  x41 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x41)
  x41= BatchNormalization()(x41)
  x41 = Activation('relu')(x41)
  x41= Dropout(0.3)(x41)

  x55 = concatenate([input, x41]) 

  x45 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x55)
  x45 = BatchNormalization()(x55)
  x45 = Activation('relu')(x55)

  x56 = concatenate([x45, x55]) 

  x411 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x56)
  x411 = BatchNormalization()(x411)
  x411 = Activation('relu')(x411)
  x411 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x411)
  x411= BatchNormalization()(x411)
  x411 = Activation('relu')(x411)
  x411= Dropout(0.3)(x411)

  return x411



def FPI(input):
    xh6 = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(input)
    xh6= BatchNormalization()(xh6)
    xh6 = Activation('relu')(xh6)
    xh6= Dropout(0.3)(xh6)

    xh33 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(input)
    xh33= BatchNormalization()(xh33)
    xh33 = Activation('relu')(xh33)
    xh44 = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(input)
    xh44= BatchNormalization()(xh44)
    xh44 = Activation('relu')(xh44)
    xh44= Dropout(0.4)(xh44)
    
    yh4= Concatenate()([xh6,xh33,xh44])
    return yh4



def R0(input_img, n_filters = 64, dropout = 0.3, batchnorm = True):
    
    # Contracting Path
    #c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    c1=conv2d_block((input_img), batchnorm = True)

    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    #c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    c2=conv2d_block(p1, batchnorm = True)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    #c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    c3=conv2d_block(p2, batchnorm = True)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
   # c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    c4=conv2d_block(p3, batchnorm = True)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    #c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    #c5=tt(p4)
    
    # Expansive Path
    #u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    #u6 = concatenate([u6, c4])
    c4a = conv2d_block(c4, batchnorm = True)
    c3a = conv2d_block(c3, batchnorm = True)
    c2a = conv2d_block(c2, batchnorm = True)
    c1a = conv2d_block(c1, batchnorm = True)
    #u6 = concatenate([u6, c4a])
    #u6 = Dropout(dropout)(u6)
    #c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    x121 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(c4a)
    x121 = BatchNormalization()(x121)
    x121 = Activation('relu')(x121)
    x121 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x121)
    x121= BatchNormalization()(x121)
    x121= Activation('relu')(x121)
    x121= Dropout(0.2)(x121)
    
    x122 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(c3a)
    x122 = BatchNormalization()(x122)
    x122 = Activation('relu')(x122)
    x122 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x122)
    x122= BatchNormalization()(x122)
    x122= Activation('relu')(x122)
    x122= Dropout(0.3)(x122)

    
    #c2a = CPT(c2)
   # c1a = CPT(c1)

    x121 = Conv2DTranspose(32, (1, 1), strides = (4, 4), padding = 'same')(x121)
    u11 = concatenate([c2a, x121])

    

    x122= Conv2DTranspose(32, (1, 1), strides = (4, 4), padding = 'same')(x122)
    u12 = concatenate([c1a, x122])
    
    x123 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(u11)# c2a
    x123 = BatchNormalization()(x123)
    x123 = Activation('relu')(x123)
    x123 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x123)
    x123= BatchNormalization()(x123)
    x123= Activation('relu')(x123)
    x123= Dropout(0.3)(x123)

    #u13 = concatenate([c2a, x123])
    u13 = Conv2DTranspose(32, (1, 1), strides = (2, 2), padding = 'same')(x123)

    
    x124 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(u12)#c1a
    x124 = BatchNormalization()(x124)
    x124 = Activation('relu')(x124)
    x124 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x124)
    x124= BatchNormalization()(x124)
    x124= Activation('relu')(x124)
    x124= Dropout(0.3)(x124)

    u14 = concatenate([u13, x124])
    
    xj = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(u14)
    xj= BatchNormalization()(xj)
    xj= Activation('relu')(xj)
    #u11 = Conv2DTranspose(112, (1, 1), strides = (8, 8), padding = 'same')(u11)
    
    u16 = Conv2DTranspose(32, (1, 1), strides = (2, 2), padding = 'same')(x121)
    u17 = Conv2DTranspose(32, (1, 1), strides = (2, 2), padding = 'same')(x123)
    yja = Concatenate()([u16,u17,x122])
    #yja=sap(yja)
    
   
    #yca = Concatenate()([xj, yja])
    yja1= conv2d_block(yja, batchnorm = True)
    #yja2=CPT(yja)
    #yja3= Concatenate()([yja,yja1,yja2])
    
    
    #xj=sap(xj)
    
  
    
    #yca3=sap1(yca)
    
    ycb = Concatenate()([xj,yja1])
    ycb1=conv2d_block(ycb, batchnorm = True)
  
    
    
    #yca=channelattention()

    yca2 = Concatenate()([ycb,ycb1])
    


    outputs = Conv2D(1, (1, 1), activation='sigmoid')(yca2)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

#%%FEE & DFM + Conventional Decoder
#%% FEE w DFM and CD


import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *

def conv2d_block(input_tensor, batchnorm = True):
    
    # 1st layer
    x = Conv2D(filters = 32, kernel_size = (2,2),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 2nd layer
    x = Conv2D(filters = 32, kernel_size = (3,3),\
              kernel_initializer = 'he_normal', padding = 'same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def DFM(input):
  x = Conv2D(filters = (16), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x1 = Conv2D(filters = (16), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(x)
  x1 = BatchNormalization()(x1)
  x1 = Activation('relu')(x1)

  x2 = Conv2D(filters = (16), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x)
  x2 = BatchNormalization()(x2)
  x2 = Activation('relu')(x2)
  x2 = Conv2D(filters = (16), kernel_size = (2, 2),kernel_initializer = 'he_normal', padding = 'same')(x2)
  x2= BatchNormalization()(x2)
  x2 = Activation('relu')(x2)
  x2= Dropout(0.3)(x2)

  x3 = concatenate([x1, x2])

  x4 = Conv2D(filters = (16), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x3)
  x4 = BatchNormalization()(x4)
  x4 = Activation('relu')(x4)
  x4 = Conv2D(filters = (16), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x4)
  x4= BatchNormalization()(x4)
  x4 = Activation('relu')(x4)
  x4= Dropout(0.3)(x4)

  x5 = concatenate([x3, x4])  
  return x5


def DMFF(image_features):
    
    shape = image_features.shape
    
    y_pool = AveragePooling2D(pool_size=(shape[1], shape[2]))(image_features)
    y_pool = Conv2D(filters=96, kernel_size=1, padding='same', use_bias=False)(y_pool)
    y_pool = BatchNormalization()(y_pool)
    y_pool = Activation('relu')(y_pool)
    y_pool = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y_pool)

    y_6 = Conv2D(filters=32, kernel_size=3, padding='same', dilation_rate = 9,use_bias=False)(y_pool)
    y_6 = BatchNormalization()(y_6)
    y_6 = Activation('relu')(y_6)
    
    #y_c11 = Concatenate()([y_pool, y_6])
    
    y_61 = Conv2D(filters=16, kernel_size=3, padding='same', dilation_rate = 6,use_bias=False)(image_features)
    y_61 = BatchNormalization()(y_61)
    y_61 = Activation('relu')(y_61)
    
    y_611 = Conv2D(filters=32, kernel_size=3, padding='same', dilation_rate = 12,use_bias=False)(y_61)
    y_611 = BatchNormalization()(y_611)
    y_611 = Activation('relu')(y_611)
    
    y_6111 = Conv2D(filters=16, kernel_size=3, padding='same', dilation_rate = 6,use_bias=False)(image_features)
    y_6111 = BatchNormalization()(y_6111)
    y_6111 = Activation('relu')(y_6111)
    
    
    y_c21 = Concatenate()([y_6 ,y_61, y_611 ,y_6111 ])
    
    x414 = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(y_c21)
    x414 = BatchNormalization()(x414)
    x414 = Activation('relu')(x414)
    
    x414= Dropout(0.6)(x414)
    
    return x414
    



def FEE(input):

  x41 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(input)
  x41 = BatchNormalization()(x41)
  x41 = Activation('relu')(x41)
  x41 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x41)
  x41= BatchNormalization()(x41)
  x41 = Activation('relu')(x41)
  x41= Dropout(0.3)(x41)

  x55 = concatenate([input, x41]) 

  x45 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x55)
  x45 = BatchNormalization()(x55)
  x45 = Activation('relu')(x55)

  x56 = concatenate([x45, x55]) 

  x411 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x56)
  x411 = BatchNormalization()(x411)
  x411 = Activation('relu')(x411)
  x411 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x411)
  x411= BatchNormalization()(x411)
  x411 = Activation('relu')(x411)
  x411= Dropout(0.3)(x411)

  return x411



def FPI(input):
    xh6 = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(input)
    xh6= BatchNormalization()(xh6)
    xh6 = Activation('relu')(xh6)
    xh6= Dropout(0.3)(xh6)

    xh33 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(input)
    xh33= BatchNormalization()(xh33)
    xh33 = Activation('relu')(xh33)
    xh44 = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(input)
    xh44= BatchNormalization()(xh44)
    xh44 = Activation('relu')(xh44)
    xh44= Dropout(0.4)(xh44)
    
    yh4= Concatenate()([xh6,xh33,xh44])
    return yh4



def R1(input_img, n_filters = 64, dropout = 0.3, batchnorm = True):
    
    # Contracting Path
    #c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    c1=FEE(input_img)

    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    #c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    c2=FEE(p1)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    #c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    c3=FEE(p2)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
   # c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    c4=FEE(p3)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    #c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    #c5=tt(p4)
    
    # Expansive Path
    #u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    #u6 = concatenate([u6, c4])
    c4a = DFM(c4)
    c3a = DFM(c3)
    c2a = DFM(c2)
    c1a = DFM(c1)
    #u6 = concatenate([u6, c4a])
    #u6 = Dropout(dropout)(u6)
    #c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    x121 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(c4a)
    x121 = BatchNormalization()(x121)
    x121 = Activation('relu')(x121)
    x121 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x121)
    x121= BatchNormalization()(x121)
    x121= Activation('relu')(x121)
    x121= Dropout(0.2)(x121)
    
    x122 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(c3a)
    x122 = BatchNormalization()(x122)
    x122 = Activation('relu')(x122)
    x122 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x122)
    x122= BatchNormalization()(x122)
    x122= Activation('relu')(x122)
    x122= Dropout(0.3)(x122)

    
    #c2a = CPT(c2)
   # c1a = CPT(c1)

    x121 = Conv2DTranspose(32, (1, 1), strides = (4, 4), padding = 'same')(x121)
    u11 = concatenate([c2a, x121])

    

    x122= Conv2DTranspose(32, (1, 1), strides = (4, 4), padding = 'same')(x122)
    u12 = concatenate([c1a, x122])
    
    x123 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(u11)# c2a
    x123 = BatchNormalization()(x123)
    x123 = Activation('relu')(x123)
    x123 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x123)
    x123= BatchNormalization()(x123)
    x123= Activation('relu')(x123)
    x123= Dropout(0.3)(x123)

    #u13 = concatenate([c2a, x123])
    u13 = Conv2DTranspose(32, (1, 1), strides = (2, 2), padding = 'same')(x123)

    
    x124 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(u12)#c1a
    x124 = BatchNormalization()(x124)
    x124 = Activation('relu')(x124)
    x124 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x124)
    x124= BatchNormalization()(x124)
    x124= Activation('relu')(x124)
    x124= Dropout(0.3)(x124)

    u14 = concatenate([u13, x124])
    
    xj = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(u14)
    xj= BatchNormalization()(xj)
    xj= Activation('relu')(xj)
    #u11 = Conv2DTranspose(112, (1, 1), strides = (8, 8), padding = 'same')(u11)
    
    u16 = Conv2DTranspose(32, (1, 1), strides = (2, 2), padding = 'same')(x121)
    u17 = Conv2DTranspose(32, (1, 1), strides = (2, 2), padding = 'same')(x123)
    yja = Concatenate()([u16,u17,x122])
    #yja=sap(yja)
    
   
    #yca = Concatenate()([xj, yja])
    yja1= conv2d_block(yja, batchnorm = True)
    #yja2=CPT(yja)
    #yja3= Concatenate()([yja,yja1,yja2])
    
    
    #xj=sap(xj)
    
  
    
    #yca3=sap1(yca)
    
    ycb = Concatenate()([xj,yja1])
    ycb1=conv2d_block(ycb, batchnorm = True)
  
    
    
    #yca=channelattention()

    yca2 = Concatenate()([ycb,ycb1])
    


    outputs = Conv2D(1, (1, 1), activation='sigmoid')(yca2)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
#%% Conventional Encoder + FPI & DMFF
#%%  CE and FPI w DMFF


import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *

def conv2d_block(input_tensor, batchnorm = True):
    
    # 1st layer
    x = Conv2D(filters = 32, kernel_size = (2,2),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 2nd layer
    x = Conv2D(filters = 32, kernel_size = (3,3),\
              kernel_initializer = 'he_normal', padding = 'same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def DFM(input):
  x = Conv2D(filters = (16), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x1 = Conv2D(filters = (16), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(x)
  x1 = BatchNormalization()(x1)
  x1 = Activation('relu')(x1)

  x2 = Conv2D(filters = (16), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x)
  x2 = BatchNormalization()(x2)
  x2 = Activation('relu')(x2)
  x2 = Conv2D(filters = (16), kernel_size = (2, 2),kernel_initializer = 'he_normal', padding = 'same')(x2)
  x2= BatchNormalization()(x2)
  x2 = Activation('relu')(x2)
  x2= Dropout(0.3)(x2)

  x3 = concatenate([x1, x2])

  x4 = Conv2D(filters = (16), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x3)
  x4 = BatchNormalization()(x4)
  x4 = Activation('relu')(x4)
  x4 = Conv2D(filters = (16), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x4)
  x4= BatchNormalization()(x4)
  x4 = Activation('relu')(x4)
  x4= Dropout(0.3)(x4)

  x5 = concatenate([x3, x4])  
  return x5


def DMFF(image_features):
    
    shape = image_features.shape
    
    y_pool = AveragePooling2D(pool_size=(shape[1], shape[2]))(image_features)
    y_pool = Conv2D(filters=96, kernel_size=1, padding='same', use_bias=False)(y_pool)
    y_pool = BatchNormalization()(y_pool)
    y_pool = Activation('relu')(y_pool)
    y_pool = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y_pool)

    y_6 = Conv2D(filters=32, kernel_size=3, padding='same', dilation_rate = 9,use_bias=False)(y_pool)
    y_6 = BatchNormalization()(y_6)
    y_6 = Activation('relu')(y_6)
    
    #y_c11 = Concatenate()([y_pool, y_6])
    
    y_61 = Conv2D(filters=16, kernel_size=3, padding='same', dilation_rate = 6,use_bias=False)(image_features)
    y_61 = BatchNormalization()(y_61)
    y_61 = Activation('relu')(y_61)
    
    y_611 = Conv2D(filters=32, kernel_size=3, padding='same', dilation_rate = 12,use_bias=False)(y_61)
    y_611 = BatchNormalization()(y_611)
    y_611 = Activation('relu')(y_611)
    
    y_6111 = Conv2D(filters=16, kernel_size=3, padding='same', dilation_rate = 6,use_bias=False)(image_features)
    y_6111 = BatchNormalization()(y_6111)
    y_6111 = Activation('relu')(y_6111)
    
    
    y_c21 = Concatenate()([y_6 ,y_61, y_611 ,y_6111 ])
    
    x414 = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(y_c21)
    x414 = BatchNormalization()(x414)
    x414 = Activation('relu')(x414)
    
    x414= Dropout(0.7)(x414)
    
    return x414
    



def FEE(input):

  x41 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(input)
  x41 = BatchNormalization()(x41)
  x41 = Activation('relu')(x41)
  x41 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x41)
  x41= BatchNormalization()(x41)
  x41 = Activation('relu')(x41)
  x41= Dropout(0.3)(x41)

  x55 = concatenate([input, x41]) 

  x45 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x55)
  x45 = BatchNormalization()(x55)
  x45 = Activation('relu')(x55)

  x56 = concatenate([x45, x55]) 

  x411 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x56)
  x411 = BatchNormalization()(x411)
  x411 = Activation('relu')(x411)
  x411 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x411)
  x411= BatchNormalization()(x411)
  x411 = Activation('relu')(x411)
  x411= Dropout(0.3)(x411)

  return x411



def FPI(input):
    xh6 = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(input)
    xh6= BatchNormalization()(xh6)
    xh6 = Activation('relu')(xh6)
    xh6= Dropout(0.3)(xh6)

    xh33 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(input)
    xh33= BatchNormalization()(xh33)
    xh33 = Activation('relu')(xh33)
    xh44 = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(input)
    xh44= BatchNormalization()(xh44)
    xh44 = Activation('relu')(xh44)
    xh44= Dropout(0.5)(xh44)
    
    yh4= Concatenate()([xh6,xh33,xh44])
    return yh4


def ASPP(inputs):
    

    #First entire shape pooling
    shape=inputs.shape
    y_pool=AveragePooling2D(pool_size=(shape[1],shape[2]))(inputs)
    y_pool=Conv2D(filters=32,kernel_size=1,use_bias=False,padding='same')(y_pool)
    y_pool=BatchNormalization()(y_pool)
    y_pool=Activation(activation='relu')(y_pool)
    y_pool=UpSampling2D(size=(shape[1],shape[2]),interpolation='bilinear')(y_pool)
    #print(y_pool.shape)

    #Now 1-d Channelwise convolution
    y_1=Conv2D(filters=32,kernel_size=1,use_bias=False,padding='same',dilation_rate=1)(inputs)
    y_1=BatchNormalization()(y_1)
    y_1=Activation(activation='relu')(y_1)
    #Now with dilationrate=6
    y_6=Conv2D(filters=32,kernel_size=3,use_bias=False,padding='same',dilation_rate=6)(inputs)
    y_6=BatchNormalization()(y_6)
    y_6=Activation(activation='relu')(y_6)

    #Now with dilationrate=12
    y_12=Conv2D(filters=32,kernel_size=3,use_bias=False,padding='same',dilation_rate=12)(inputs)
    y_12=BatchNormalization()(y_12)
    y_12=Activation(activation='relu')(y_12)

    #Now with dilation rate=18
    y_18=Conv2D(filters=32,kernel_size=3,use_bias=False,padding='same',dilation_rate=18)(inputs)
    y_18=BatchNormalization()(y_18)
    y_18=Activation(activation='relu')(y_18)

    y=Concatenate()([y_pool,y_1,y_6,y_12,y_18])
    #1-d convolution application
    y=Conv2D(filters=32,kernel_size=1,padding='same',dilation_rate=1,use_bias=False)(y)
    y=BatchNormalization()(y)
    y=Activation(activation='relu')(y)
    #print(y.shape)
    return y


def R3(input_img, n_filters = 64, dropout = 0.3, batchnorm = True):
    
    # Contracting Path
    #c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    c1=conv2d_block((input_img), batchnorm = True)

    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    #c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    c2=conv2d_block(p1, batchnorm = True)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    #c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    c3=conv2d_block(p2, batchnorm = True)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
   # c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    c4=conv2d_block(p3, batchnorm = True)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    #c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    #c5=tt(p4)
    
    # Expansive Path
    #u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    #u6 = concatenate([u6, c4])
    c4a = conv2d_block(c4, batchnorm = True)
    c3a = conv2d_block(c3, batchnorm = True)
    c2a = conv2d_block(c2, batchnorm = True)
    c1a = conv2d_block(c1, batchnorm = True)
    #u6 = concatenate([u6, c4a])
    #u6 = Dropout(dropout)(u6)
    #c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    x121 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(c4a)
    x121 = BatchNormalization()(x121)
    x121 = Activation('relu')(x121)
    x121 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x121)
    x121= BatchNormalization()(x121)
    x121= Activation('relu')(x121)
    x121= Dropout(0.2)(x121)
    
    x122 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(c3a)
    x122 = BatchNormalization()(x122)
    x122 = Activation('relu')(x122)
    x122 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x122)
    x122= BatchNormalization()(x122)
    x122= Activation('relu')(x122)
    x122= Dropout(0.3)(x122)

    
    #c2a = CPT(c2)
   # c1a = CPT(c1)

    x121 = Conv2DTranspose(32, (1, 1), strides = (4, 4), padding = 'same')(x121)
    u11 = concatenate([c2a, x121])

    

    x122= Conv2DTranspose(32, (1, 1), strides = (4, 4), padding = 'same')(x122)
    u12 = concatenate([c1a, x122])
    
    x123 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(u11)# c2a
    x123 = BatchNormalization()(x123)
    x123 = Activation('relu')(x123)
    x123 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x123)
    x123= BatchNormalization()(x123)
    x123= Activation('relu')(x123)
    x123= Dropout(0.3)(x123)

    #u13 = concatenate([c2a, x123])
    u13 = Conv2DTranspose(32, (1, 1), strides = (2, 2), padding = 'same')(x123)

    
    x124 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(u12)#c1a
    x124 = BatchNormalization()(x124)
    x124 = Activation('relu')(x124)
    x124 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x124)
    x124= BatchNormalization()(x124)
    x124= Activation('relu')(x124)
    x124= Dropout(0.3)(x124)

    u14 = concatenate([u13, x124])
    
    xj = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(u14)
    xj= BatchNormalization()(xj)
    xj= Activation('relu')(xj)
    #u11 = Conv2DTranspose(112, (1, 1), strides = (8, 8), padding = 'same')(u11)
    
    u16 = Conv2DTranspose(32, (1, 1), strides = (2, 2), padding = 'same')(x121)
    u17 = Conv2DTranspose(32, (1, 1), strides = (2, 2), padding = 'same')(x123)
    yja = Concatenate()([u16,u17,x122])
    #yja=sap(yja)
    
   
    #yca = Concatenate()([xj, yja])
    yja1= FPI(yja)
    #yja2=CPT(yja)
    #yja3= Concatenate()([yja,yja1,yja2])
    
    
    #xj=sap(xj)
    
  
    
    #yca3=sap1(yca)
    
    ycb = Concatenate()([xj,yja1])
    ycb1=DMFF(ycb)
  
    
    
    #yca=channelattention()

    yca2 = Concatenate()([ycb,ycb1])
    


    outputs = Conv2D(1, (1, 1), activation='sigmoid')(yca2)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

#%% FEE & DFM + FPI
#%%  FEE w DFM w FPI


import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *

def conv2d_block(input_tensor, batchnorm = True):
    
    # 1st layer
    x = Conv2D(filters = 32, kernel_size = (2,2),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 2nd layer
    x = Conv2D(filters = 32, kernel_size = (3,3),\
              kernel_initializer = 'he_normal', padding = 'same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def DFM(input):
  x = Conv2D(filters = (16), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x1 = Conv2D(filters = (16), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(x)
  x1 = BatchNormalization()(x1)
  x1 = Activation('relu')(x1)

  x2 = Conv2D(filters = (16), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x)
  x2 = BatchNormalization()(x2)
  x2 = Activation('relu')(x2)
  x2 = Conv2D(filters = (16), kernel_size = (2, 2),kernel_initializer = 'he_normal', padding = 'same')(x2)
  x2= BatchNormalization()(x2)
  x2 = Activation('relu')(x2)
  x2= Dropout(0.3)(x2)

  x3 = concatenate([x1, x2])

  x4 = Conv2D(filters = (16), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x3)
  x4 = BatchNormalization()(x4)
  x4 = Activation('relu')(x4)
  x4 = Conv2D(filters = (16), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x4)
  x4= BatchNormalization()(x4)
  x4 = Activation('relu')(x4)
  x4= Dropout(0.3)(x4)

  x5 = concatenate([x3, x4])  
  return x5


def DMFF(image_features):
    
    shape = image_features.shape
    
    y_pool = AveragePooling2D(pool_size=(shape[1], shape[2]))(image_features)
    y_pool = Conv2D(filters=96, kernel_size=1, padding='same', use_bias=False)(y_pool)
    y_pool = BatchNormalization()(y_pool)
    y_pool = Activation('relu')(y_pool)
    y_pool = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y_pool)

    y_6 = Conv2D(filters=32, kernel_size=3, padding='same', dilation_rate = 9,use_bias=False)(y_pool)
    y_6 = BatchNormalization()(y_6)
    y_6 = Activation('relu')(y_6)
    
    #y_c11 = Concatenate()([y_pool, y_6])
    
    y_61 = Conv2D(filters=16, kernel_size=3, padding='same', dilation_rate = 6,use_bias=False)(image_features)
    y_61 = BatchNormalization()(y_61)
    y_61 = Activation('relu')(y_61)
    
    y_611 = Conv2D(filters=32, kernel_size=3, padding='same', dilation_rate = 12,use_bias=False)(y_61)
    y_611 = BatchNormalization()(y_611)
    y_611 = Activation('relu')(y_611)
    
    y_6111 = Conv2D(filters=16, kernel_size=3, padding='same', dilation_rate = 6,use_bias=False)(image_features)
    y_6111 = BatchNormalization()(y_6111)
    y_6111 = Activation('relu')(y_6111)
    
    
    y_c21 = Concatenate()([y_6 ,y_61, y_611 ,y_6111 ])
    
    x414 = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(y_c21)
    x414 = BatchNormalization()(x414)
    x414 = Activation('relu')(x414)
    
    x414= Dropout(0.6)(x414)
    
    return x414
    



def FEE(input):

  x41 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(input)
  x41 = BatchNormalization()(x41)
  x41 = Activation('relu')(x41)
  x41 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x41)
  x41= BatchNormalization()(x41)
  x41 = Activation('relu')(x41)
  x41= Dropout(0.3)(x41)

  x55 = concatenate([input, x41]) 

  x45 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x55)
  x45 = BatchNormalization()(x55)
  x45 = Activation('relu')(x55)

  x56 = concatenate([x45, x55]) 

  x411 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x56)
  x411 = BatchNormalization()(x411)
  x411 = Activation('relu')(x411)
  x411 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x411)
  x411= BatchNormalization()(x411)
  x411 = Activation('relu')(x411)
  x411= Dropout(0.3)(x411)

  return x411



def FPI(input):
    xh6 = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(input)
    xh6= BatchNormalization()(xh6)
    xh6 = Activation('relu')(xh6)
    xh6= Dropout(0.3)(xh6)

    xh33 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(input)
    xh33= BatchNormalization()(xh33)
    xh33 = Activation('relu')(xh33)
    xh44 = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(input)
    xh44= BatchNormalization()(xh44)
    xh44 = Activation('relu')(xh44)
    xh44= Dropout(0.4)(xh44)
    
    yh4= Concatenate()([xh6,xh33,xh44])
    return yh4



def R4(input_img, n_filters = 64, dropout = 0.3, batchnorm = True):
    
    # Contracting Path
    #c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    c1=FEE((input_img))

    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    #c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    c2=FEE(p1)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    #c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    c3=FEE(p2)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
   # c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    c4=FEE(p3)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    #c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    #c5=tt(p4)
    
    # Expansive Path
    #u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    #u6 = concatenate([u6, c4])
    c4a = DFM(c4)
    c3a = DFM(c3)
    c2a = DFM(c2)
    c1a = DFM(c1)
    #u6 = concatenate([u6, c4a])
    #u6 = Dropout(dropout)(u6)
    #c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    x121 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(c4a)
    x121 = BatchNormalization()(x121)
    x121 = Activation('relu')(x121)
    x121 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x121)
    x121= BatchNormalization()(x121)
    x121= Activation('relu')(x121)
    x121= Dropout(0.2)(x121)
    
    x122 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(c3a)
    x122 = BatchNormalization()(x122)
    x122 = Activation('relu')(x122)
    x122 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x122)
    x122= BatchNormalization()(x122)
    x122= Activation('relu')(x122)
    x122= Dropout(0.3)(x122)

    
    #c2a = CPT(c2)
   # c1a = CPT(c1)

    x121 = Conv2DTranspose(32, (1, 1), strides = (4, 4), padding = 'same')(x121)
    u11 = concatenate([c2a, x121])

    

    x122= Conv2DTranspose(32, (1, 1), strides = (4, 4), padding = 'same')(x122)
    u12 = concatenate([c1a, x122])
    
    x123 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(u11)# c2a
    x123 = BatchNormalization()(x123)
    x123 = Activation('relu')(x123)
    x123 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x123)
    x123= BatchNormalization()(x123)
    x123= Activation('relu')(x123)
    x123= Dropout(0.3)(x123)

    #u13 = concatenate([c2a, x123])
    u13 = Conv2DTranspose(32, (1, 1), strides = (2, 2), padding = 'same')(x123)

    
    x124 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(u12)#c1a
    x124 = BatchNormalization()(x124)
    x124 = Activation('relu')(x124)
    x124 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x124)
    x124= BatchNormalization()(x124)
    x124= Activation('relu')(x124)
    x124= Dropout(0.3)(x124)

    u14 = concatenate([u13, x124])
    
    xj = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(u14)
    xj= BatchNormalization()(xj)
    xj= Activation('relu')(xj)
    #u11 = Conv2DTranspose(112, (1, 1), strides = (8, 8), padding = 'same')(u11)
    
    u16 = Conv2DTranspose(32, (1, 1), strides = (2, 2), padding = 'same')(x121)
    u17 = Conv2DTranspose(32, (1, 1), strides = (2, 2), padding = 'same')(x123)
    yja = Concatenate()([u16,u17,x122])
    #yja=sap(yja)
    
   
    #yca = Concatenate()([xj, yja])
    yja1= FPI(yja)
    #yja2=CPT(yja)
    #yja3= Concatenate()([yja,yja1,yja2])
    
    
    #xj=sap(xj)
    
  
    
    #yca3=sap1(yca)
    
    ycb = Concatenate()([xj,yja1])
    ycb1=conv2d_block((ycb), batchnorm = True)
  
    
    
    #yca=channelattention()

    yca2 = Concatenate()([ycb,ycb1])
    


    outputs = Conv2D(1, (1, 1), activation='sigmoid')(yca2)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
#%% FEE & DFM + DMFF
#%% FEE w DFM w DMFF


import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *

def conv2d_block(input_tensor, batchnorm = True):
    
    # 1st layer
    x = Conv2D(filters = 32, kernel_size = (2,2),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 2nd layer
    x = Conv2D(filters = 32, kernel_size = (3,3),\
              kernel_initializer = 'he_normal', padding = 'same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def DFM(input):
  x = Conv2D(filters = (16), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x1 = Conv2D(filters = (16), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(x)
  x1 = BatchNormalization()(x1)
  x1 = Activation('relu')(x1)

  x2 = Conv2D(filters = (16), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x)
  x2 = BatchNormalization()(x2)
  x2 = Activation('relu')(x2)
  x2 = Conv2D(filters = (16), kernel_size = (2, 2),kernel_initializer = 'he_normal', padding = 'same')(x2)
  x2= BatchNormalization()(x2)
  x2 = Activation('relu')(x2)
  x2= Dropout(0.3)(x2)

  x3 = concatenate([x1, x2])

  x4 = Conv2D(filters = (16), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x3)
  x4 = BatchNormalization()(x4)
  x4 = Activation('relu')(x4)
  x4 = Conv2D(filters = (16), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x4)
  x4= BatchNormalization()(x4)
  x4 = Activation('relu')(x4)
  x4= Dropout(0.3)(x4)

  x5 = concatenate([x3, x4])  
  return x5


def DMFF(image_features):
    
    shape = image_features.shape
    
    y_pool = AveragePooling2D(pool_size=(shape[1], shape[2]))(image_features)
    y_pool = Conv2D(filters=96, kernel_size=1, padding='same', use_bias=False)(y_pool)
    y_pool = BatchNormalization()(y_pool)
    y_pool = Activation('relu')(y_pool)
    y_pool = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y_pool)

    y_6 = Conv2D(filters=32, kernel_size=3, padding='same', dilation_rate = 9,use_bias=False)(y_pool)
    y_6 = BatchNormalization()(y_6)
    y_6 = Activation('relu')(y_6)
    
    #y_c11 = Concatenate()([y_pool, y_6])
    
    y_61 = Conv2D(filters=16, kernel_size=3, padding='same', dilation_rate = 6,use_bias=False)(image_features)
    y_61 = BatchNormalization()(y_61)
    y_61 = Activation('relu')(y_61)
    
    y_611 = Conv2D(filters=32, kernel_size=3, padding='same', dilation_rate = 12,use_bias=False)(y_61)
    y_611 = BatchNormalization()(y_611)
    y_611 = Activation('relu')(y_611)
    
    y_6111 = Conv2D(filters=16, kernel_size=3, padding='same', dilation_rate = 6,use_bias=False)(image_features)
    y_6111 = BatchNormalization()(y_6111)
    y_6111 = Activation('relu')(y_6111)
    
    
    y_c21 = Concatenate()([y_6 ,y_61, y_611 ,y_6111 ])
    
    x414 = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(y_c21)
    x414 = BatchNormalization()(x414)
    x414 = Activation('relu')(x414)
    
    x414= Dropout(0.6)(x414)
    
    return x414
    



def FEE(input):

  x41 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(input)
  x41 = BatchNormalization()(x41)
  x41 = Activation('relu')(x41)
  x41 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x41)
  x41= BatchNormalization()(x41)
  x41 = Activation('relu')(x41)
  x41= Dropout(0.3)(x41)

  x55 = concatenate([input, x41]) 

  x45 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x55)
  x45 = BatchNormalization()(x55)
  x45 = Activation('relu')(x55)

  x56 = concatenate([x45, x55]) 

  x411 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x56)
  x411 = BatchNormalization()(x411)
  x411 = Activation('relu')(x411)
  x411 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x411)
  x411= BatchNormalization()(x411)
  x411 = Activation('relu')(x411)
  x411= Dropout(0.3)(x411)

  return x411



def FPI(input):
    xh6 = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(input)
    xh6= BatchNormalization()(xh6)
    xh6 = Activation('relu')(xh6)
    xh6= Dropout(0.3)(xh6)

    xh33 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(input)
    xh33= BatchNormalization()(xh33)
    xh33 = Activation('relu')(xh33)
    xh44 = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(input)
    xh44= BatchNormalization()(xh44)
    xh44 = Activation('relu')(xh44)
    xh44= Dropout(0.4)(xh44)
    
    yh4= Concatenate()([xh6,xh33,xh44])
    return yh4



def R5(input_img, n_filters = 64, dropout = 0.3, batchnorm = True):
    
    # Contracting Path
    #c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    c1=FEE((input_img))

    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    #c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    c2=FEE(p1)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    #c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    c3=FEE(p2)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
   # c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    c4=FEE(p3)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    #c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    #c5=tt(p4)
    
    # Expansive Path
    #u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    #u6 = concatenate([u6, c4])
    c4a = DFM(c4)
    c3a = DFM(c3)
    c2a = DFM(c2)
    c1a = DFM(c1)
    #u6 = concatenate([u6, c4a])
    #u6 = Dropout(dropout)(u6)
    #c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    x121 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(c4a)
    x121 = BatchNormalization()(x121)
    x121 = Activation('relu')(x121)
    x121 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x121)
    x121= BatchNormalization()(x121)
    x121= Activation('relu')(x121)
    x121= Dropout(0.2)(x121)
    
    x122 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(c3a)
    x122 = BatchNormalization()(x122)
    x122 = Activation('relu')(x122)
    x122 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x122)
    x122= BatchNormalization()(x122)
    x122= Activation('relu')(x122)
    x122= Dropout(0.3)(x122)

    
    #c2a = CPT(c2)
   # c1a = CPT(c1)

    x121 = Conv2DTranspose(32, (1, 1), strides = (4, 4), padding = 'same')(x121)
    u11 = concatenate([c2a, x121])

    

    x122= Conv2DTranspose(32, (1, 1), strides = (4, 4), padding = 'same')(x122)
    u12 = concatenate([c1a, x122])
    
    x123 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(u11)# c2a
    x123 = BatchNormalization()(x123)
    x123 = Activation('relu')(x123)
    x123 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x123)
    x123= BatchNormalization()(x123)
    x123= Activation('relu')(x123)
    x123= Dropout(0.3)(x123)

    #u13 = concatenate([c2a, x123])
    u13 = Conv2DTranspose(32, (1, 1), strides = (2, 2), padding = 'same')(x123)

    
    x124 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(u12)#c1a
    x124 = BatchNormalization()(x124)
    x124 = Activation('relu')(x124)
    x124 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x124)
    x124= BatchNormalization()(x124)
    x124= Activation('relu')(x124)
    x124= Dropout(0.3)(x124)

    u14 = concatenate([u13, x124])
    
    xj = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(u14)
    xj= BatchNormalization()(xj)
    xj= Activation('relu')(xj)
    #u11 = Conv2DTranspose(112, (1, 1), strides = (8, 8), padding = 'same')(u11)
    
    u16 = Conv2DTranspose(32, (1, 1), strides = (2, 2), padding = 'same')(x121)
    u17 = Conv2DTranspose(32, (1, 1), strides = (2, 2), padding = 'same')(x123)
    yja = Concatenate()([u16,u17,x122])
    #yja=sap(yja)
    
   
    #yca = Concatenate()([xj, yja])
    yja1= conv2d_block(yja, batchnorm = True)
    #yja2=CPT(yja)
    #yja3= Concatenate()([yja,yja1,yja2])
    
    
    #xj=sap(xj)
    
  
    
    #yca3=sap1(yca)
    
    ycb = Concatenate()([xj,yja1])
    ycb1=DMFF(ycb)
  
    
    
    #yca=channelattention()

    yca2 = Concatenate()([ycb,ycb1])
    


    outputs = Conv2D(1, (1, 1), activation='sigmoid')(yca2)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

#%% FEE + FPI & DMFF

#%% FEE w FPI w DMFF


import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *

def conv2d_block(input_tensor, batchnorm = True):
    
    # 1st layer
    x = Conv2D(filters = 32, kernel_size = (2,2),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 2nd layer
    x = Conv2D(filters = 32, kernel_size = (3,3),\
              kernel_initializer = 'he_normal', padding = 'same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def DFM(input):
  x = Conv2D(filters = (16), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x1 = Conv2D(filters = (16), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(x)
  x1 = BatchNormalization()(x1)
  x1 = Activation('relu')(x1)

  x2 = Conv2D(filters = (16), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x)
  x2 = BatchNormalization()(x2)
  x2 = Activation('relu')(x2)
  x2 = Conv2D(filters = (16), kernel_size = (2, 2),kernel_initializer = 'he_normal', padding = 'same')(x2)
  x2= BatchNormalization()(x2)
  x2 = Activation('relu')(x2)
  x2= Dropout(0.3)(x2)

  x3 = concatenate([x1, x2])

  x4 = Conv2D(filters = (16), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x3)
  x4 = BatchNormalization()(x4)
  x4 = Activation('relu')(x4)
  x4 = Conv2D(filters = (16), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x4)
  x4= BatchNormalization()(x4)
  x4 = Activation('relu')(x4)
  x4= Dropout(0.3)(x4)

  x5 = concatenate([x3, x4])  
  return x5


def DMFF(image_features):
    
    shape = image_features.shape
    
    y_pool = AveragePooling2D(pool_size=(shape[1], shape[2]))(image_features)
    y_pool = Conv2D(filters=96, kernel_size=1, padding='same', use_bias=False)(y_pool)
    y_pool = BatchNormalization()(y_pool)
    y_pool = Activation('relu')(y_pool)
    y_pool = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y_pool)

    y_6 = Conv2D(filters=32, kernel_size=3, padding='same', dilation_rate = 9,use_bias=False)(y_pool)
    y_6 = BatchNormalization()(y_6)
    y_6 = Activation('relu')(y_6)
    
    #y_c11 = Concatenate()([y_pool, y_6])
    
    y_61 = Conv2D(filters=16, kernel_size=3, padding='same', dilation_rate = 6,use_bias=False)(image_features)
    y_61 = BatchNormalization()(y_61)
    y_61 = Activation('relu')(y_61)
    
    y_611 = Conv2D(filters=32, kernel_size=3, padding='same', dilation_rate = 12,use_bias=False)(y_61)
    y_611 = BatchNormalization()(y_611)
    y_611 = Activation('relu')(y_611)
    
    y_6111 = Conv2D(filters=16, kernel_size=3, padding='same', dilation_rate = 6,use_bias=False)(image_features)
    y_6111 = BatchNormalization()(y_6111)
    y_6111 = Activation('relu')(y_6111)
    
    
    y_c21 = Concatenate()([y_6 ,y_61, y_611 ,y_6111 ])
    
    x414 = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(y_c21)
    x414 = BatchNormalization()(x414)
    x414 = Activation('relu')(x414)
    
    x414= Dropout(0.7)(x414)
    
    return x414
    



def FEE(input):

  x41 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(input)
  x41 = BatchNormalization()(x41)
  x41 = Activation('relu')(x41)
  x41 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x41)
  x41= BatchNormalization()(x41)
  x41 = Activation('relu')(x41)
  x41= Dropout(0.3)(x41)

  x55 = concatenate([input, x41]) 

  x45 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x55)
  x45 = BatchNormalization()(x55)
  x45 = Activation('relu')(x55)

  x56 = concatenate([x45, x55]) 

  x411 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x56)
  x411 = BatchNormalization()(x411)
  x411 = Activation('relu')(x411)
  x411 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x411)
  x411= BatchNormalization()(x411)
  x411 = Activation('relu')(x411)
  x411= Dropout(0.3)(x411)

  return x411



def FPI(input):
    xh6 = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(input)
    xh6= BatchNormalization()(xh6)
    xh6 = Activation('relu')(xh6)
    xh6= Dropout(0.3)(xh6)

    xh33 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(input)
    xh33= BatchNormalization()(xh33)
    xh33 = Activation('relu')(xh33)
    xh44 = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(input)
    xh44= BatchNormalization()(xh44)
    xh44 = Activation('relu')(xh44)
    xh44= Dropout(0.5)(xh44)
    
    yh4= Concatenate()([xh6,xh33,xh44])
    return yh4



def R61(input_img, n_filters = 64, dropout = 0.3, batchnorm = True):
    
    # Contracting Path
    #c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    c1=FEE((input_img))

    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    #c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    c2=FEE(p1)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    #c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    c3=FEE(p2)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
   # c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    c4=FEE(p3)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    #c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    #c5=tt(p4)
    
    # Expansive Path
    #u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    #u6 = concatenate([u6, c4])
    c4a = conv2d_block(c4, batchnorm = True)
    c3a = conv2d_block(c3, batchnorm = True)
    c2a = conv2d_block(c2, batchnorm = True)
    c1a = conv2d_block(c1, batchnorm = True)
    #u6 = concatenate([u6, c4a])
    #u6 = Dropout(dropout)(u6)
    #c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    x121 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(c4a)
    x121 = BatchNormalization()(x121)
    x121 = Activation('relu')(x121)
    x121 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x121)
    x121= BatchNormalization()(x121)
    x121= Activation('relu')(x121)
    x121= Dropout(0.2)(x121)
    
    x122 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(c3a)
    x122 = BatchNormalization()(x122)
    x122 = Activation('relu')(x122)
    x122 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x122)
    x122= BatchNormalization()(x122)
    x122= Activation('relu')(x122)
    x122= Dropout(0.3)(x122)

    
    #c2a = CPT(c2)
   # c1a = CPT(c1)

    x121 = Conv2DTranspose(32, (1, 1), strides = (4, 4), padding = 'same')(x121)
    u11 = concatenate([c2a, x121])

    

    x122= Conv2DTranspose(32, (1, 1), strides = (4, 4), padding = 'same')(x122)
    u12 = concatenate([c1a, x122])
    
    x123 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(u11)# c2a
    x123 = BatchNormalization()(x123)
    x123 = Activation('relu')(x123)
    x123 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x123)
    x123= BatchNormalization()(x123)
    x123= Activation('relu')(x123)
    x123= Dropout(0.3)(x123)

    #u13 = concatenate([c2a, x123])
    u13 = Conv2DTranspose(32, (1, 1), strides = (2, 2), padding = 'same')(x123)

    
    x124 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(u12)#c1a
    x124 = BatchNormalization()(x124)
    x124 = Activation('relu')(x124)
    x124 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x124)
    x124= BatchNormalization()(x124)
    x124= Activation('relu')(x124)
    x124= Dropout(0.3)(x124)

    u14 = concatenate([u13, x124])
    
    xj = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(u14)
    xj= BatchNormalization()(xj)
    xj= Activation('relu')(xj)
    #u11 = Conv2DTranspose(112, (1, 1), strides = (8, 8), padding = 'same')(u11)
    
    u16 = Conv2DTranspose(32, (1, 1), strides = (2, 2), padding = 'same')(x121)
    u17 = Conv2DTranspose(32, (1, 1), strides = (2, 2), padding = 'same')(x123)
    yja = Concatenate()([u16,u17,x122])
    #yja=sap(yja)
    
   
    #yca = Concatenate()([xj, yja])
    yja1= FPI(yja)
    #yja2=CPT(yja)
    #yja3= Concatenate()([yja,yja1,yja2])
    
    
    #xj=sap(xj)
    
  
    
    #yca3=sap1(yca)
    
    ycb = Concatenate()([xj,yja1])
    ycb1=DMFF(ycb)
  
    
    
    #yca=channelattention()

    yca2 = Concatenate()([ycb,ycb1])
    


    outputs = Conv2D(1, (1, 1), activation='sigmoid')(yca2)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

#%% DFM + FPI & DMFF
 
#%% DFM W FPI w DMFF



import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *

def conv2d_block(input_tensor, batchnorm = True):
    
    # 1st layer
    x = Conv2D(filters = 32, kernel_size = (2,2),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 2nd layer
    x = Conv2D(filters = 32, kernel_size = (3,3),\
              kernel_initializer = 'he_normal', padding = 'same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def DFM(input):
  x = Conv2D(filters = (16), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x1 = Conv2D(filters = (16), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(x)
  x1 = BatchNormalization()(x1)
  x1 = Activation('relu')(x1)

  x2 = Conv2D(filters = (16), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x)
  x2 = BatchNormalization()(x2)
  x2 = Activation('relu')(x2)
  x2 = Conv2D(filters = (16), kernel_size = (2, 2),kernel_initializer = 'he_normal', padding = 'same')(x2)
  x2= BatchNormalization()(x2)
  x2 = Activation('relu')(x2)
  x2= Dropout(0.3)(x2)

  x3 = concatenate([x1, x2])

  x4 = Conv2D(filters = (16), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x3)
  x4 = BatchNormalization()(x4)
  x4 = Activation('relu')(x4)
  x4 = Conv2D(filters = (16), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x4)
  x4= BatchNormalization()(x4)
  x4 = Activation('relu')(x4)
  x4= Dropout(0.3)(x4)

  x5 = concatenate([x3, x4])  
  return x5


def DMFF(image_features):
    
    shape = image_features.shape
    
    y_pool = AveragePooling2D(pool_size=(shape[1], shape[2]))(image_features)
    y_pool = Conv2D(filters=96, kernel_size=1, padding='same', use_bias=False)(y_pool)
    y_pool = BatchNormalization()(y_pool)
    y_pool = Activation('relu')(y_pool)
    y_pool = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y_pool)

    y_6 = Conv2D(filters=32, kernel_size=3, padding='same', dilation_rate = 9,use_bias=False)(y_pool)
    y_6 = BatchNormalization()(y_6)
    y_6 = Activation('relu')(y_6)
    
    #y_c11 = Concatenate()([y_pool, y_6])
    
    y_61 = Conv2D(filters=16, kernel_size=3, padding='same', dilation_rate = 6,use_bias=False)(image_features)
    y_61 = BatchNormalization()(y_61)
    y_61 = Activation('relu')(y_61)
    
    y_611 = Conv2D(filters=32, kernel_size=3, padding='same', dilation_rate = 12,use_bias=False)(y_61)
    y_611 = BatchNormalization()(y_611)
    y_611 = Activation('relu')(y_611)
    
    y_6111 = Conv2D(filters=16, kernel_size=3, padding='same', dilation_rate = 6,use_bias=False)(image_features)
    y_6111 = BatchNormalization()(y_6111)
    y_6111 = Activation('relu')(y_6111)
    
    
    y_c21 = Concatenate()([y_6 ,y_61, y_611 ,y_6111 ])
    
    x414 = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(y_c21)
    x414 = BatchNormalization()(x414)
    x414 = Activation('relu')(x414)
    
    x414= Dropout(0.7)(x414)
    
    return x414
    



def FEE(input):

  x41 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(input)
  x41 = BatchNormalization()(x41)
  x41 = Activation('relu')(x41)
  x41 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x41)
  x41= BatchNormalization()(x41)
  x41 = Activation('relu')(x41)
  x41= Dropout(0.3)(x41)

  x55 = concatenate([input, x41]) 

  x45 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x55)
  x45 = BatchNormalization()(x55)
  x45 = Activation('relu')(x55)

  x56 = concatenate([x45, x55]) 

  x411 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(x56)
  x411 = BatchNormalization()(x411)
  x411 = Activation('relu')(x411)
  x411 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x411)
  x411= BatchNormalization()(x411)
  x411 = Activation('relu')(x411)
  x411= Dropout(0.3)(x411)

  return x411



def FPI(input):
    xh6 = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(input)
    xh6= BatchNormalization()(xh6)
    xh6 = Activation('relu')(xh6)
    xh6= Dropout(0.3)(xh6)

    xh33 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(input)
    xh33= BatchNormalization()(xh33)
    xh33 = Activation('relu')(xh33)
    xh44 = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(input)
    xh44= BatchNormalization()(xh44)
    xh44 = Activation('relu')(xh44)
    xh44= Dropout(0.5)(xh44)
    
    yh4= Concatenate()([xh6,xh33,xh44])
    return yh4



def R62(input_img, n_filters = 64, dropout = 0.3, batchnorm = True):
    
    # Contracting Path
    #c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    c1=conv2d_block((input_img), batchnorm = True)

    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    #c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    c2=conv2d_block(p1, batchnorm = True)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    #c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    c3=conv2d_block(p2, batchnorm = True)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
   # c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    c4=conv2d_block(p3, batchnorm = True)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    #c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    #c5=tt(p4)
    
    # Expansive Path
    #u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    #u6 = concatenate([u6, c4])
    c4a = DFM(c4)
    c3a = DFM(c3)
    c2a = DFM(c2)
    c1a = DFM(c1)
    #u6 = concatenate([u6, c4a])
    #u6 = Dropout(dropout)(u6)
    #c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    x121 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(c4a)
    x121 = BatchNormalization()(x121)
    x121 = Activation('relu')(x121)
    x121 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x121)
    x121= BatchNormalization()(x121)
    x121= Activation('relu')(x121)
    x121= Dropout(0.2)(x121)
    
    x122 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(c3a)
    x122 = BatchNormalization()(x122)
    x122 = Activation('relu')(x122)
    x122 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x122)
    x122= BatchNormalization()(x122)
    x122= Activation('relu')(x122)
    x122= Dropout(0.3)(x122)

    
    #c2a = CPT(c2)
   # c1a = CPT(c1)

    x121 = Conv2DTranspose(32, (1, 1), strides = (4, 4), padding = 'same')(x121)
    u11 = concatenate([c2a, x121])

    

    x122= Conv2DTranspose(32, (1, 1), strides = (4, 4), padding = 'same')(x122)
    u12 = concatenate([c1a, x122])
    
    x123 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(u11)# c2a
    x123 = BatchNormalization()(x123)
    x123 = Activation('relu')(x123)
    x123 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x123)
    x123= BatchNormalization()(x123)
    x123= Activation('relu')(x123)
    x123= Dropout(0.3)(x123)

    #u13 = concatenate([c2a, x123])
    u13 = Conv2DTranspose(32, (1, 1), strides = (2, 2), padding = 'same')(x123)

    
    x124 = Conv2D(filters = (32), kernel_size = (3, 3),kernel_initializer = 'he_normal', padding = 'same')(u12)#c1a
    x124 = BatchNormalization()(x124)
    x124 = Activation('relu')(x124)
    x124 = Conv2D(filters = (32), kernel_size = (4, 4),kernel_initializer = 'he_normal', padding = 'same')(x124)
    x124= BatchNormalization()(x124)
    x124= Activation('relu')(x124)
    x124= Dropout(0.3)(x124)

    u14 = concatenate([u13, x124])
    
    xj = Conv2D(filters = (32), kernel_size = (1, 1),kernel_initializer = 'he_normal', padding = 'same')(u14)
    xj= BatchNormalization()(xj)
    xj= Activation('relu')(xj)
    #u11 = Conv2DTranspose(112, (1, 1), strides = (8, 8), padding = 'same')(u11)
    
    u16 = Conv2DTranspose(32, (1, 1), strides = (2, 2), padding = 'same')(x121)
    u17 = Conv2DTranspose(32, (1, 1), strides = (2, 2), padding = 'same')(x123)
    yja = Concatenate()([u16,u17,x122])
    #yja=sap(yja)
    
   
    #yca = Concatenate()([xj, yja])
    yja1= FPI(yja)
    #yja2=CPT(yja)
    #yja3= Concatenate()([yja,yja1,yja2])
    
    
    #xj=sap(xj)
    
  
    
    #yca3=sap1(yca)
    
    ycb = Concatenate()([xj,yja1])
    ycb1=DMFF(ycb)
  
    
    
    #yca=channelattention()

    yca2 = Concatenate()([ycb,ycb1])
    


    outputs = Conv2D(1, (1, 1), activation='sigmoid')(yca2)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

# Proposed mRbVseG
# same as above nRbVseG
#%%MODEL Initial Summary - Abalation Study
from tensorflow.keras import Input

#shape=(320,320,3)
#model=DeepLabv3plus(shape)
#model=unet(pretrained_weights = None, input_size = (256, 256, 3), classes = 1)
#input_img = Input((256, 256, 3), name='img')
#model=Unet2(input_img, n_filters = 64, dropout = 0.3, batchnorm = True)

#model=unetmodel((320,320,3), dropout=0.05, batchnorm=True)
#model.summary()

#%%
input_img = Input((384, 384, 3), name='img')
#model=R62(input_img, n_filters = 64, dropout = 0.3, batchnorm = True)
model=R62(input_img, n_filters = 64, dropout = 0.3, batchnorm = True)
model.summary()
#%% Metrics  
import numpy as np
#import tensorflow as tf
from tensorflow.keras import backend as K
#import keras.backend as K
import sklearn

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

import tensorflow as tf
import numpy as np 
 
#if tf.__version__ == '1.13.1' or tf.__version__ == '1.15.0': 
#    from keras.layers.merge import concatenate
#    from keras.layers import Activation
#    import keras.backend as K 
#if tf.__version__ == '2.2.0' or tf.__version__ == '2.1.0' or tf.__version__ == '2.3.0' or tf.__version__ == '2.5.0': 
import keras.backend as K
from keras.layers import Activation, concatenate
    #import tensorflow_addons as tfa



from keras import backend as K
from keras import backend as keras

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + 
    K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def jacard(y_true, y_pred):

  y_true_f = keras.flatten(y_true)
  y_pred_f = keras.flatten(y_pred)
  intersection = keras.sum ( y_true_f * y_pred_f)
  union = keras.sum ( y_true_f + y_pred_f - y_true_f * y_pred_f)

  return intersection/union

from tensorflow.keras.metrics import binary_accuracy


def accuracy(y_true, y_pred):
    class_nums = y_pred.shape[-1]//2

    y_true = y_true[..., class_nums:]
    y_pred = y_pred[..., class_nums:]
    bi_acc = binary_accuracy(y_true, y_pred)

    return

import numpy as np
import tensorflow as tf
from keras import backend as K



def specificity(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return tn / (tn + fp + K.epsilon())

import numpy as np


def get_pixel_accuracy(pred: np.ndarray, mask: np.ndarray) -> float:
    """
    Pixel accuracy for whole image
    Referenced by：
    Long J , Shelhamer E , Darrell T . Fully Convolutional Networks for Semantic Segmentation[J].
    IEEE Transactions on Pattern Analysis & Machine Intelligence, 2014, 39(4):640-651.
    """
    class_num = np.amax(mask) + 1

    temp_n_ii = 0.0
    temp_t_i = 0.0
    for i_cl in range(class_num):
        temp_n_ii += np.count_nonzero(mask[pred == i_cl] == i_cl)
        temp_t_i  += np.count_nonzero(mask == i_cl)
    value = temp_n_ii / temp_t_i
    return value


def get_mean_accuracy(pred: np.ndarray, mask: np.ndarray) -> float:
    """
    Mean accuracy for each class
    Referenced by：
    Long J , Shelhamer E , Darrell T . Fully Convolutional Networks for Semantic Segmentation[J].
    IEEE Transactions on Pattern Analysis & Machine Intelligence, 2014, 39(4):640-651.
    """
    class_num = np.amax(mask) + 1
    temp = 0.0
    for i_cl in range(class_num):
        n_ii = np.count_nonzero(mask[pred == i_cl] == i_cl)
        t_i = np.count_nonzero(mask == i_cl)
        temp += n_ii / t_i
    value = temp / class_num
    return 

def mean_iou(y_true, y_pred, smooth=1):
    
    
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(y_pred > 0.5, "int32")
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2])
    union = K.sum(y_true,[1,2])+K.sum(y_pred,[1,2])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=[1,0])
    
    return iou

from tensorflow.keras import backend as K
from sklearn.metrics import jaccard_score,confusion_matrix

def jacard(y_true, y_pred):

    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum ( y_true_f * y_pred_f)
    union = keras.sum ( y_true_f + y_pred_f - y_true_f * y_pred_f)

    return intersection/union
#%% Loss Funciton
def weighted_bce_loss(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    #logit_y_pred = y_pred
    
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
    (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)

def weighted_dice_loss(y_true, y_pred, weight=1):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * (m1**2)) + K.sum(w * (m2**2)) + smooth) # Uptill here is Dice Loss with squared
    loss = 1. - K.sum(score)  #Soft Dice Loss
    return loss

def Dice_coef1(y_true, y_pred, weight=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def Dice_loss1(y_true, y_pred):
    loss = 1 - Dice_coef1(y_true, y_pred)
    return loss
        

def Weighted_BCEnDice_loss(y_true, y_pred):
    
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    averaged_mask = K.pool2d(
            y_true, pool_size=(11, 11), strides=(1, 1),padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2 
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss =  weighted_dice_loss(y_true, y_pred, weight) + weighted_bce_loss(y_true, y_pred, weight) 
    return loss
#%% Reset all variables from space
%reset
#%% Training and Evaluation part


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam




H = 320
W = 320

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    x = sorted(glob(os.path.join(path, "image", "*.jpg")))
    y = sorted(glob(os.path.join(path, "mask", "*.jpg")))
    return x, y

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)              ## (512, 512, 1)
    return x




def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
       # x = histo_equalized(x)
        #x=clahe_equalized_single(x)
        #x=adjust_gamma_single(x)

        y = read_mask(y)
       # y = histo_equalized(y)
       # y=clahe_equalized_single(y)
        #y=adjust_gamma_single(y)

        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X, Y, batch_size=8):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(4)
    return dataset


from pathlib import Path
MODEL_PATH = Path("files")
MODEL_PATH.mkdir(parents=True, exist_ok=True)#, exist_ok means that if the parent directory
# already exist then it wont give error in that case, it will recreate it then

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory to save files """
    #create_dir("files")

    """ Hyperparameters """
    batch_size = 8 # 8
    lr = 0.001
    num_epochs = 25#150
    model_path = os.path.join("files", "model.h5")
    csv_path = os.path.join("files", "data.csv")

    """ Dataset """
    dataset_path =  "/home/hasan/work_retina/newdatac1n/"  # "/home/hasan/work_retina/newdatas1n/"  # "/home/hasan/work_retina/newdatac1n/"
    train_path = os.path.join(dataset_path, "train")
    valid_path = os.path.join(dataset_path, "test")

    train_x, train_y = load_data(train_path)
    train_x, train_y = shuffling(train_x, train_y)
    valid_x, valid_y = load_data(valid_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch_size=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch_size=batch_size)

    train_steps = len(train_x)//batch_size
    valid_setps = len(valid_x)//batch_size

    if len(train_x) % batch_size != 0:
        train_steps += 1
    if len(valid_x) % batch_size != 0:
        valid_setps += 1

    """ Model """
    #model = DeeplabV3Plus(image_size=352, num_classes=2)
    #model = build_unet((H, W, 3))
    #input_shape = (H, W, 3)
    #model = DeepLabV3Plus(input_shape)
    #input_shape = (192, 192, 3)
    
   #shape=(320,320,3)
   #model=DeepLabv3plus(shape)
    #model=unet(pretrained_weights = None, input_size = (320, 320, 3), classes = 1)





    
    #input_img = Input((H, W, 3), name='img')
    #model=mRbVseG(input_img, n_filters = 64, dropout = 0.2, batchnorm = True)

    #shape=(320,320,3)
    #model=DeepLabv3plusR(shape)
    #model=residualunet(shape, dropout=0.01, batchnorm=True)
    #model=AttUNet(1, input_height=H, input_width=W)
    input_shape = (H, W, 3)
    model = build_unet(input_shape)
   
    




    #model=unetmodel((320,320,3), dropout=0.05, batchnorm=True)
    #model=FEEwDMFF(input_img, n_filters = 64, dropout = 0.3, batchnorm = True)
    model.summary()
    
    #model = base_unet(64, output_channels=1, width=192, height=192, input_channels=3, conv_layers=2)

    model.compile(loss=Dice_loss1, optimizer=Adam(lr), metrics=[dice_coef, iou,jacard,
                              precision_m,recall_m,f1_m,specificity,mean_iou])
    # model.summary() , dice_loss

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=100, min_lr=1e-6, verbose=1),
        CSVLogger(csv_path),
        TensorBoard(),
        EarlyStopping(monitor="val_loss", patience=100, restore_best_weights=False)
    ]

    history=model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        steps_per_epoch=train_steps,
        validation_steps=valid_setps,
        callbacks=callbacks
    )
# Model Outputs Visulaization
#%%


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope


H =320
W = 320

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    return ori_x, x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.int32)
    return ori_x, x

def load_data(path):
    x = sorted(glob(os.path.join(path, "image", "*.jpg")))
    y = sorted(glob(os.path.join(path, "mask", "*.jpg")))
    return x, y

def save_results(ori_x, ori_y, y_pred, save_image_path):
    line = np.ones((H, 10, 3)) * 255

    ori_y = np.expand_dims(ori_y, axis=-1)
    ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1)

    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) * 255

    cat_images = np.concatenate([ori_x, line, ori_y, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)

if __name__ == "__main__":
    """ Save the results in this folder """
    create_dir("results")

    """ Load the model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'Dice_loss1': Dice_loss1, 'iou':iou,'jacard':jacard,
                              'precision_m':precision_m,'recall_m':recall_m,'f1_m':f1_m,'specificity':specificity,'mean_iou':mean_iou}):
        model = tf.keras.models.load_model("files/model.h5")

    """ Load the dataset """
    dataset_path = os.path.join("/home/hasan/work_retina/newdatac1n/", "test")
    test_x, test_y = load_data(dataset_path)

    """ Make the prediction and calculate the metrics values """
    SCORE = []
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """ Extracting name """
        name = x.split("/")[-1].split(".")[0]

        """ Read the image and mask """
        ori_x, x = read_image(x)
        ori_y, y = read_mask(y)

        """ Prediction """
        y_pred = model.predict(np.expand_dims(x, axis=0))[0]
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.int32)
        y_pred = np.squeeze(y_pred, axis=-1)

        """ Saving the images """
        save_image_path = f"results/{name}.jpg"
        save_results(ori_x, ori_y, y_pred, save_image_path)
####
        fig, ax = plt.subplots(1, 3, figsize = (15, 15))

        plt.imshow(y_pred)

        ax[0].imshow(ori_y)
        ax[0].set_title(f'GT ', fontsize = 15)
        ax[0].axis("off")

        ax[1].imshow(y_pred)
        ax[1].set_title('Prediction', fontsize = 15)
        ax[1].axis("off")

        ax[2].imshow(ori_x, cmap = 'gray')
        ax[2].set_title(f'original ', fontsize = 15)
        ax[2].axis("off")

####

        """ Flatten the array """
        y = y.flatten()
        y_pred = y_pred.flatten()

        y=tf.dtypes.cast(y, tf.float32)
        y_pred=tf.dtypes.cast(y_pred, tf.float32)

y_pred.dtype



