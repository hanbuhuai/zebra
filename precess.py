# -*-coding utf-8 -*-
import  os,glob
from os import path
import tensorflow as tf
import threading
class jpg2record():
    def __init__(self,image_path,set_path):
        self._dRoot = path.abspath(path.dirname(__file__))
        self._dImage = path.join(self._dRoot,image_path)
        self.set_path = set_path
        self.data_size = 0
        self.image_set = self._load_file()
    def _read_image(self,fpath):
        image = tf.io.read_file(fpath)
        image_tensor = tf.io.decode_jpeg(image,channels=3)
        image_tensor = tf.transpose(image_tensor,perm=(2,0,1))
        image_tensor = tf.reshape(image_tensor,shape=(256*256*3,))
        #图片归一化
        image_tensor = (tf.cast(image_tensor,dtype=tf.float32)-127.5)/255
        return image_tensor
    def _load_file(self):
        fselect = path.join(self._dImage,"%s/*.jpg"%(self.set_path))
        f_list = glob.glob(fselect)
        self.data_size = len(f_list)
        data_set = tf.data.Dataset.from_tensor_slices(f_list)
        data_set = data_set.map(self._read_image,num_parallel_calls=8)
        data_set = data_set.filter(lambda x:x.shape==(256*256*3))
        return data_set
    def _serialize_example(self,image_tensor):
        image_tensor = tf.train.FloatList(value=image_tensor)
        features = tf.train.Features(feature={
            "image_tensor":tf.train.Feature(float_list=image_tensor),
        })
        example = tf.train.Example(features=features)
        return example.SerializeToString()
    def record(self,output_dir,recodr_size=160):
        opt_dir = path.join(self._dRoot,output_dir)
        if not path.isdir(opt_dir):
            os.makedirs(output_dir)
        data_set = self.image_set.batch(batch_size=recodr_size)
        steps = self.data_size//recodr_size if self.data_size%recodr_size==0 else (self.data_size//recodr_size)+1
        options = tf.io.TFRecordOptions(compression_type="GZIP")
        for k,item in enumerate(data_set.take(steps)):
            data_size = item.shape[0]
            step = k+1
            fname = path.join(opt_dir,"{:03d}_{:03d}.record.zip".format(step,data_size))
            with tf.io.TFRecordWriter(fname,options) as writer:
                for ipk,img_tensor in enumerate(tf.unstack(item)):
                    print("{cur_epoch}/{total_epoch}正在写入文件：{step}/{total}".format(
                        cur_epoch=step,total_epoch=steps,step=ipk+1,total=data_size
                    ),end="\r")
                    serialized_image = self._serialize_example(img_tensor)
                    writer.write(serialized_image)
            writer.close()
"""    
if __name__ == "__main__":
    pmA = jpg2record("images","trainA")
    pmB = jpg2record("images","trainB")
    #pm.record("records/trainB")
    ta = threading.Thread(target=pmA.record,args=("records/trainA",))
    tb = threading.Thread(target=pmB.record,args=("records/trainB",))
    ta.start()
    tb.start()
"""
        