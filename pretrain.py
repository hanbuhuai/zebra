#-*-coding utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from nets.generator import Generator
from nets.discriminator import Discriminator
from data_set import data_recod,batch_feature
import os

def mk_cate_set(data_set_a,data_set_b,category_a,category_b):
    """
        创建一个分类数据集
        args
            data_set_a : tf.data.dataset
            data_set_b : tf.data.dataset
            category_a : int data_set_a 的分类信息
            category_b : int data_set_b 的分类信息
        return 
            data_set : (data_set_a,category_a) + (data_set_b,category_b) 
    """
    category_a = tf.constant(category_a,dtype=tf.int64)
    category_b = tf.constant(category_b,dtype=tf.int64)
    aSet = data_set_a.map(lambda x:(x,category_a))
    bSet = data_set_b.map(lambda x:(x,category_b))
    data_set = aSet.concatenate(bSet)
    opt_set = data_set.shuffle(10000)
    return opt_set
class pretrain():
    mA2B = Generator(name="A2B")
    mB2A = Generator(name="B2A")
    mCategory = Discriminator("Category")
    def __init__(self):
        self._d_root = os.path.abspath(os.path.dirname(__file__))
        self._dckpt = os.path.join(self._d_root,"pretrain_ckpt")
        self.optimizer_category = keras.optimizers.Adam()
        self.optimizer_A2B = keras.optimizers.Adam()
        self.optimizer_B2A = keras.optimizers.Adam()
    def save_model(self,model,prefix):
        if not os.path.isdir(self._dckpt):
            os.makedirs(self._dckpt)
        fpath = os.path.join(self._dckpt,"%s_%s.h5"%(prefix,model.name))
        model.save_weights(fpath,overwrite=True)
        return fpath
    def train_step_category(self,x_input,y_true):
        with tf.GradientTape() as tape:
            self.mCategory.trainable = True
            y_pre_logits = self.mCategory(x_input)
            
            loss = keras.losses.sparse_categorical_crossentropy(y_true,y_pre_logits,from_logits=True)
            gradient_virables = self.mCategory.trainable_variables
            gradients = tape.gradient(loss,gradient_virables)
            self.optimizer_category.apply_gradients(zip(gradients,gradient_virables))
        #返回结果
        batch_loss = tf.reduce_mean(loss).numpy()
        #计算正确率
        predict =tf.argmax(tf.nn.softmax(y_pre_logits),axis=-1) 
        correct = tf.equal(predict,y_true)
        accurancy = tf.reduce_mean(tf.cast(correct,dtype=tf.float32))
        return batch_loss,accurancy
    def train_step_A2B(self,x_img,step):
        with tf.GradientTape() as tape:
            self.mA2B.trainable = True
            gen_image = self.mA2B(x_img)
            loss = keras.losses.mean_squared_error(x_img,gen_image)
            gradient_varialbes = self.mA2B.trainable_variables
            gradients = tape.gradient(loss,gradient_varialbes)
            self.optimizer_A2B.apply_gradients(zip(gradients,gradient_varialbes))
            batch_loss = tf.reduce_mean(loss).numpy()
        return batch_loss,gen_image
    def train_step_B2A(self,x_img,step):
        with tf.GradientTape() as tape:
            self.mB2A.trainable = True
            gen_image = self.mB2A(x_img)
            loss = keras.losses.mean_squared_error(x_img,gen_image)
            gradient_varialbes = self.mB2A.trainable_variables
            gradients = tape.gradient(loss,gradient_varialbes)
            self.optimizer_B2A.apply_gradients(zip(gradients,gradient_varialbes))
            batch_loss = tf.reduce_mean(loss).numpy()
        return batch_loss,gen_image
    def train_category(self,cate_set,steps):
        """
        训练分类器
        args
            cate_set : data_set (x_input,labels)
        """
        for k,(x_input,label) in enumerate(cate_set.take(steps)):
            loss,accurancy = self.train_step_category(x_input,label)
            print("预训练分类器[{:03d}/{:03d}]:loss={:.4f} accurancy={:.4f}".format(
                k+1,steps,loss,accurancy,
            ))
        else:
            self.save_model(self.mCategory,"mCategory")
    def train_A2B(self,img_set,steps,batch_size):

        for k,input_img in enumerate(img_set.take(steps)):
            loss,pre_img = self.train_step_A2B(input_img,k)
            print("预训练生成器A2B[{:03d}/{:03d}]:loss={:.4f}".format(
                k+1,steps,loss
            ))
            ##制作预览
            if k%50==0:
                viewModel = batch_feature(batch_size)
                viewModel.add_image(input_img)
                viewModel.add_image(pre_img)
                viewModel.save("A2B_{:03d}.jpg".format(k+1))
        else:
            self.save_model(self.mA2B,"A2B")
    def train_B2A(self,img_set,steps,batch_size):
    
        for k,input_img in enumerate(img_set.take(steps)):
            loss,pre_img = self.train_step_B2A(input_img,k)
            print("预训练生成器B2A[{:03d}/{:03d}]:loss={:.4f}".format(
                k+1,steps,loss
            ))
            ##制作预览
            if k%50==0:
                viewModel = batch_feature(batch_size)
                viewModel.add_image(input_img)
                viewModel.add_image(pre_img)
                viewModel.save("B2A_{:03d}.jpg".format(k+1))
        else:
            self.save_model(self.mB2A,"B2A")

if __name__ == '__main__':
    BATCH_SIZE = 8 #batch_size
    
    modelTrainSetA = data_recod("records","trainA",6*160+107)
    modelTrainSetB = data_recod("records","trainB",8*160+54)
    """
        创建分类数据集
        cate_set : 分类训练数据集
        cate_set_data_steps : 数据集步数
    """
    cate_set = mk_cate_set(modelTrainSetA.data_set,modelTrainSetB.data_set,0,1).batch(
        batch_size=BATCH_SIZE,
        drop_remainder=True
    )
    cate_set_data_steps = (modelTrainSetA.data_size+modelTrainSetB.data_size)//BATCH_SIZE
    train_model = pretrain()
    """
        预训练分类器
    """
    # for k,(x_input,label) in enumerate(cate_set.take(10)):
    #     #print(x_input.shape,label.shape)
    #     loss,accurancy = train_model.train_step_category(x_input,label)
    #     print("预训练分类器[{:03d}/{:03d}]:loss={:.4f} accurancy={:.4f}".format(
    #         k+1,10,loss,accurancy,
    #     ))

    """
       预训练生成器
    """
    ATRSet,ATRStep = modelTrainSetA.mk_data_set(BATCH_SIZE)
    BTRSet,BTRStep = modelTrainSetB.mk_data_set(BATCH_SIZE)

    train_model.train_B2A(BTRSet,5,BATCH_SIZE)
    
    
        
    



    
    
    




    