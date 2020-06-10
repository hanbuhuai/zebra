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


class Generator(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.EVGG_1 = vgg_cell_2(64,3)  #output_shape(128,128,64)
        self.EVGG_2 = vgg_cell_2(128,3) #output_shape(64,64,128)
        self.EVGG_3 = vgg_cell_3(256,3) #output_shape(32,32,256)
        self.EVGG_4 = vgg_cell_3(512,3) #output_shape(16,16,512)
        self.EVGG_5 = vgg_cell_3(512,3) #output_shape(8,8,512)

        self.TConv_4 = keras.layers.Conv2DTranspose(512,kernel_size=[5,5],strides=[2,2],
        padding="same")#inputship:(8,8,512)output_shape(16,16,512)
        self.TConv_3 = keras.layers.Conv2DTranspose(256,kernel_size=[5,5],strides=[2,2],
        padding="same")#output_shape(32,32,256)
        self.TConv_2 = keras.layers.Conv2DTranspose(128,kernel_size=[5,5],strides=[2,2],
        padding="same")#output_shape(64,64,128)
        self.TConv_1 = keras.layers.Conv2DTranspose(64,kernel_size=[5,5],strides=[2,2],
        padding="same")#output_shape(128,128,64)
        self.TConv_0 = keras.layers.Conv2DTranspose(3,kernel_size=[5,5],strides=[2,2],
        padding="same")#output_shape(256,256,3)


    def call(self,x_input):
        x_input = tf.reshape(x_input,shape=(-1,3,256,256))
        x_input = tf.transpose(x_input,perm=[0,2,3,1])
        E1 = self.EVGG_1(x_input) #output_shape(128,128,64)
        E2 = self.EVGG_2(E1)      #output_shape(64,64,128)
        E3 = self.EVGG_3(E2)      #output_shape(32,32,256)
        E4 = self.EVGG_4(E3)      #output_shape(16,16,512)
        E5 = self.EVGG_5(E4)      #output_shape(8,8,512)
        D4 = self.TConv_4(E5)     #output_shape(16,16,512)
        D3 = self.TConv_3(tf.concat([D4,E4],axis=-1)) 
        D2 = self.TConv_2(tf.concat([D3,E3],axis=-1)) #output_shape(64,64,256)
        D1 = self.TConv_1(tf.concat([D2,E2],axis=-1)) #output_shape(128,128,64)
        D0 = self.TConv_0(tf.concat([D1,E1],axis=-1)) #output_shape(256,256,3)
        out_put = tf.nn.tanh(D0)                       
        out_put = tf.transpose(out_put,perm=[0,3,1,2])
        out_put = tf.reshape(out_put,shape=(-1,256*256*3))
        
        return out_put



