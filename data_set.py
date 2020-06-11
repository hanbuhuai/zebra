import tensorflow as tf
import os
import glob
class data_maker():
    def __init__(self,data_set_path,shuffle=True):
        self._droot = os.path.abspath(os.path.dirname(__file__))
        self._dist = os.path.join(self._droot,"dist/images")
        self._data_set_path = data_set_path
        self._data_size = 0
        self.img_set = self._load_file(shuffle=shuffle)
    def _read_image(self,fpath):
        image = tf.io.read_file(fpath)
        image_tensor = tf.io.decode_jpeg(image,channels=3)
        image_tensor = tf.transpose(image_tensor,perm=(2,0,1))
        image_tensor = tf.reshape(image_tensor,shape=(256*256*3,))
        #图片归一化
        image_tensor = (tf.cast(image_tensor,dtype=tf.float32)-127.5)/255
        return image_tensor
    def _load_file(self,shuffle=True):
        path_selector = os.path.join(self._dist,"{data_set_path}/*.jpg".format(data_set_path=self._data_set_path)) 
        f_list=glob.glob(path_selector)
        data_set = tf.data.Dataset.from_tensor_slices(f_list)
        self._data_size = len(f_list)
        if shuffle:
            data_set.shuffle(10240)
        #读取文件
        data_set = data_set.map(self._read_image)
        data_set = data_set.filter(lambda x:x.shape==(256*256*3))
        return data_set
    def get_data_set(self,batch_size):
        time_step_pre_epoch = self._data_size//batch_size 
        return self.img_set.batch(batch_size,drop_remainder=True),time_step_pre_epoch
class data_recod():
    def __init__(self,record_path,record_name,data_size,shuffle=True):
        self._droot = os.path.abspath(os.path.dirname(__file__))
        self._drecord = os.path.join(self._droot,record_path)
        self.data_set = self._load_record(record_name)
        self.data_size = data_size
    def _unserialize_example(self,serialized):
        example = tf.io.parse_single_example(serialized,features={
            "image_tensor":tf.io.FixedLenFeature([256*256*3],dtype=tf.float32),
        })
        return example['image_tensor']
    def _load_record(self,record_name):
        path_selecter = os.path.join(self._drecord,"{record_name}/*.record.zip".format(record_name=record_name))
        flist = glob.glob(path_selecter)
        data_set = tf.data.TFRecordDataset(flist,compression_type="GZIP")
        data_set = data_set.map(self._unserialize_example)
        return data_set
    def mk_data_set(self,batch_size):
        data_set = self.data_set.batch(batch_size=batch_size)
        time_step = self.data_size//batch_size
        time_step = time_step if self.data_size%batch_size==0 else time_step+1
        return data_set,time_step

class batch_feature():
    """
    组合batch_size 个图片，并输出到文件
    height: batch_size*256,weight:n_batch*256
    """
    def __init__(self,batch_size):
        self._droot = os.path.abspath(os.path.dirname(__file__))
        self.preview_dir = os.path.join(self._droot,"preview")
        self.batch_size = batch_size
    def add_image(self,tensor):
        """
        args
            tensor : tensor shape = (batch_size,256*256*3)
        """
        batch_size = self.batch_size
        if tensor.shape != (batch_size,256*256*3):
            assert Exception("输入图片错误，接受tensor shape=(batch_size,tensor_len)".format(
                batch_size = batch_size,
                tensor_len = 256*256*3
            ))
        image = tf.reshape(tensor,shape=(-1,3,256,256))
        image = tf.transpose(image,perm=(0,2,3,1))
        image = tf.reshape(image,shape=(-1,256,3))
        #image = tf.transpose(tensor,perm=(0,2,3,1))

        # image = tf.reshape(image,shape=(-1,3,256,256))
        # image = tf.reshape(tensor,shape=(batch_size*256,256,3))
        image = tf.cast(image*255+127.5,dtype=tf.uint8)
        if hasattr(self,"image_tensor"):
            self.image_tensor = tf.concat([self.image_tensor,image],axis=1)
        else:
            self.image_tensor = image
        return self
    def save(self,fname):
        image = tf.io.encode_jpeg(self.image_tensor)
        if not os.path.isdir(self.preview_dir):
            os.makedirs(self.preview_dir)
        tf.io.write_file(os.path.join(self.preview_dir,fname),image)
        return os.path.join(self.preview_dir,fname)

        

    

        


        