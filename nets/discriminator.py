# -*- encoding utf-8 -*-
import tensorflow as tf
from tensorflow import keras

class vgg_cell_2(keras.layers.Layer):
    def __init__(self,fliters,kernel_size,trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(vgg_cell_2,self).__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        self.conv_1 = keras.layers.Conv2D(fliters,kernel_size,activation="relu",strides=(1,1),padding="same")
        self.conv_2 = keras.layers.Conv2D(fliters,kernel_size,activation="relu",strides=(1,1),padding="same")
        self.max_pool = keras.layers.MaxPool2D(pool_size=(2,2))
        self.bn = keras.layers.BatchNormalization()
    def build(self, input_shape):
        return super().build(input_shape)
    def call(self,x_input):
        features = self.conv_1(x_input)
        features = self.conv_2(features)
        out = self.max_pool(features)
        return self.bn(out)
class vgg_cell_3(keras.layers.Layer):
    def __init__(self,fliters,kernel_size,trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(vgg_cell_3,self).__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        self.conv_1 = keras.layers.Conv2D(fliters,kernel_size,activation="relu",strides=(1,1),padding="same")
        self.conv_2 = keras.layers.Conv2D(fliters,kernel_size,activation="relu",strides=(1,1),padding="same")
        self.conv_3 = keras.layers.Conv2D(fliters,kernel_size,activation="relu",strides=(1,1),padding="same")
        self.max_pool = keras.layers.MaxPool2D(pool_size=(2,2))
        self.bn = keras.layers.BatchNormalization()
    def build(self, input_shape):
        return super().build(input_shape)
    def call(self,x_input):
       
        features = self.conv_1(x_input)
        features = self.conv_2(features)
        features = self.conv_3(features)
        out = self.max_pool(features)
        return self.bn(out)


class Discriminator(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.EVGG_1 = vgg_cell_2(64,3)  #output_shape(128,128,64)
        self.EVGG_2 = vgg_cell_2(128,3) #output_shape(64,64,128)
        self.EVGG_3 = vgg_cell_3(256,3) #output_shape(32,32,256)
        self.EVGG_4 = vgg_cell_3(512,3) #output_shape(16,16,512)
        self.EVGG_5 = vgg_cell_3(512,3) #output_shape(8,8,512)
        self.EVGG_6 = vgg_cell_3(512,3) #output_shape(4,4,512)
        self.flatten = keras.layers.Flatten()
        self.dense_1 = keras.layers.Dense(1024)
        self.dense_2 = keras.layers.Dense(128)
        self.dense_3 = keras.layers.Dense(4)
    def call(self,x_input):
        x_input = tf.reshape(x_input,shape=(-1,3,256,256)) #(batch_size,2,256,256)
        x_input = tf.transpose(x_input,perm=[0,2,3,1])
        E1 = self.EVGG_1(x_input) #output_shape(128,128,64)
        E2 = self.EVGG_2(E1)      #output_shape(64,64,128)
        E3 = self.EVGG_3(E2)      #output_shape(32,32,256)
        E4 = self.EVGG_4(E3)      #output_shape(16,16,512)
        E5 = self.EVGG_5(E4)      #output_shape(8,8,512)
        E6 = self.EVGG_6(E5)      #output_shape(4,4,512)
        out_put = self.flatten(E6)
        out_put = self.dense_1(out_put)
        out_put = self.dense_2(out_put)
        out_put = self.dense_3(out_put)
        return out_put



