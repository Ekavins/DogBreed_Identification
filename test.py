import tensorflow as tf
import numpy as np
import model as M
import random
import cv2
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class data_reader():
    def __init__(self):
        print('Reading data...')
        data = []
        #img1 = []
        df = pd.read_csv('labels.csv')
        #label=LabelEncoder().fit_transform(df['breed'])
        #print(label)
        for index, row in df.iterrows():
            image_path = 'train/'+row['id']+'.jpg'
            img = cv2.imread(image_path)
            #print(image_path)
            img = cv2.resize(img,(256,256))
            #print(img)
            #cv2.imshow('image',img)
            #cv2.waitKey(0)
            label=LabelEncoder().fit_transform(df['breed'])
            data_row = [img,label]
            data.append(data_row)
        #    data = (img1,label)
        #print (data)
        self.data = data

    def next_train_batch(self,bsize):
        batch = random.sample(self.data,bsize)
        return batch


def main_structure(inp):
    with tf.variable_scope('mainModel'):
        inp = tf.scan(lambda _,y:tf.image.random_brightness(y,20),inp,initializer=tf.constant(0.0,shape=[256,256,3]))
        inp = tf.scan(lambda _,y:tf.image.random_contrast(y,0.5,2),inp,initializer=tf.constant(0.0,shape=[256,256,3]))
        inp = tf.scan(lambda _,y:tf.image.random_saturation(y,0.5,2),inp,initializer=tf.constant(0.0,shape=[256,256,3]))
        mod = M.Model(inp,[None,256,256,3])
        mod.convLayer(5,16,stride=2,activation=M.PARAM_LRELU)#128_2x2
        mod.convLayer(4,32,stride=2,activation=M.PARAM_LRELU)#64_4x4
        mod.convLayer(3,64,stride=2,activation=M.PARAM_LRELU)#32_8x8
        mod.flatten()
        mod.fcLayer(50000,activation=M.PARAM_RELU)
        mod.fcLayer(10222)
        return mod.get_current_layer()


def build_graph():
	with tf.name_scope('img_holder'):
		img_holder = tf.placeholder(tf.float32,[None,256,256,3])
	with tf.name_scope('lab_holder'):
		lab_holder = tf.placeholder(tf.float32,[None,10222])

	last_layer = main_structure(img_holder)

	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=lab_holder,logits=last_layer))
	accuracy = M.accuracy(last_layer,tf.argmax(lab_holder,1))
	train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

	return img_holder,lab_holder,loss,train_step,accuracy,last_layer

MAXITER = 100000000
BSIZE = 8

reader = data_reader()
img_holder,lab_holder,loss,train_step,accuracy,last_layer = build_graph()

with tf.Session() as sess:
	M.loadSess('./model/',sess,init=True)
	saver = tf.train.Saver()
	reader = data_reader()
	print('Reading finish')
	for iteration in range(MAXITER):
		# print(iteration)
		train_batch = reader.next_train_batch(BSIZE)
		# print(iteration)
		img_batch = [i[0] for i in train_batch]
		lab_batch = [i[1] for i in train_batch]
		img_batch = np.float32(img_batch)
		feeddict = {img_holder:img_batch, lab_holder:lab_batch}
		# print(iteration)
		_,acc,ls = sess.run([train_step,accuracy,loss],feed_dict=feeddict)

		if iteration%10==0:
			print('Iter:',iteration,'\tLoss_b:',ls,'\tAcc:',acc)
			# print(c.max())
			img = img_batch[0].astype(np.uint8)

		if iteration%5000==0 and iteration!=0:
			saver.save(sess,'./model/'+str(iteration)+'.ckpt')
