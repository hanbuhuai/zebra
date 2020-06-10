# -*- coding utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from nets.generator import Generator
from nets.discriminator import Discriminator
from data_set import data_maker,batch_feature

"""
创建数据集，初始化网络
"""
#超参数
batch_size = 8

#创建model
modelA2B = Generator(name="a2b")
modelB2A = Generator(name="b2a")
modelD0A = Discriminator(name="d0a")
modelD0B = Discriminator(name="d0b")
#创建训练集
A_TrainSet,A_TimeSteps= data_maker("trainA").get_data_set(batch_size)
B_TrainSet,B_TimeSteps= data_maker("trainB").get_data_set(batch_size)
#创建测试集
A_TestSet,_ = data_maker("testA").get_data_set(batch_size)
B_TestSet,_ = data_maker("testB").get_data_set(batch_size)
for item in A_TestSet.take(1):
    A_FixSet = item
for item in B_TestSet.take(1):
    B_FixSet = item

"""
部署训练方法

训练一个单向网络
"""
class trainA2B():
    def __init__(self,batch_size):
        self.batch_size = batch_size
        self.optimizer = keras.optimizers.Adam()
    def train_gen_step(self,x_input):
        with tf.GradientTape() as tape:
            modelD0B.trainable = False
            modelA2B.trainable = True
            pre_b = modelA2B(x_input)
            d_pre_b = modelD0A(pre_b)
            #计算顺势
            y_true = tf.ones(shape=(self.batch_size,))
            loss = keras.losses.sparse_categorical_crossentropy(y_true,d_pre_b,from_logits=True)
            train_vals = modelA2B.trainable_variables
            grads = tape.gradient(loss,train_vals)
            self.optimizer.apply_gradients(zip(grads,train_vals))
            batch_loss = tf.reduce_mean(loss,axis=0)
        return batch_loss,pre_b
    def train_dis_step(self,x_input_a,x_input_b):
        with tf.GradientTape() as tape:
            modelA2B.trainable = False
            modelD0B.trainable = True
            pre_b = modelA2B(x_input_a)
            x_input = tf.concat([pre_b,x_input_b],axis=0)
            d_pre_b = modelD0B(x_input)
            #计算损失
            label_1 = tf.zeros(shape=(self.batch_size,))
            label_2 = tf.ones(shape=(self.batch_size,))
            y_true = tf.concat([label_1,label_2],axis=0)
            loss = keras.losses.sparse_categorical_crossentropy(y_true,d_pre_b,from_logits=True)
            train_vals = modelD0B.trainable_variables
            grads = tape.gradient(loss,train_vals)
            self.optimizer.apply_gradients(zip(grads,train_vals))
            batch_loss = tf.reduce_mean(loss,axis=0)
        return batch_loss,pre_b
    def save_model(self):
        pass

    def train(self,step):
        data_set = tf.data.Dataset.zip((A_TrainSet,B_TrainSet))
        for k,(setA,setB) in enumerate(data_set.take(step)):
            loss_gen,pre_b_gen = self.train_gen_step(setA)
            los_dis,pre_b_dis = self.train_dis_step(setA,setB)
            print("step {:03d} gen_loss:{:.4f} dis_loss{:.4f}".format(
                k+1,loss_gen,los_dis
            ))
            fviewModel = batch_feature(self.batch_size)
            fviewModel.add_image(setA)
            fviewModel.add_image(pre_b_gen)
            fviewModel.add_image(setB)
            fviewModel.add_image(pre_b_dis)
            fviewModel.save("{:03d}.jpg".format(k+1))
if __name__ == '__main__':
    ta2bModel = trainA2B(8)
    ta2bModel.train(20)
    
        

        




    








