import tensorflow as tf 
import pandas as pd 
import cv2
import numpy as np
import model as M
import random 
from sklearn.preprocessing import LabelEncoder


def data_reader():
	data =[]
	print('Reading data...')
	df = pd.read_csv('labels.csv')
	#label = LabelEncoder().fit_transform(df['breed'])
	#print (label)
	for index,row in df.iterrows():
		image_path='./train/'+row['id']+'.jpg'
		img = cv2.imread(image_path)
		img = cv2.resize(img,(256,256))
		img = img.reshape([256,256,3])
		#cv2.imshow('img',img)
		#cv2.waitKey(0)
		#print (img)
		label = LabelEncoder().fit_transform(row['breed'])
		data_row = [img,label]
		data.append(data_row)

	return data	


def build_model(inp):
	with tf.variable_scope('mainModel'):
		inp = tf.scan(lambda _,y:tf.image.random_brightness(y,20),inp,initializer=tf.constant(0.0,shape=[256,256,3]))
		inp = tf.scan(lambda _,y:tf.image.random_contrast(y,0.5,2),inp,initializer=tf.constant(0.0,shape=[256,256,3]))
		inp = tf.scan(lambda _,y:tf.image.random_saturation(y,0.5,2),inp,initializer=tf.constant(0.0,shape=[256,256,3]))
		mod = M.Model(inp,[None,256,256,3])
		mod.convLayer(5,16,stride=2,activation=M.PARAM_LRELU)#128_2x2
		mod.convLayer(4,32,stride=2,activation=M.PARAM_LRELU)#64_4x4
		mod.convLayer(3,64,stride=2,activation=M.PARAM_LRELU)#32_8x8
		mod.convLayer(3,128,activation=M.PARAM_RELU)#32_8x8
		mod.flatten()
		mod.fcLayer(256,activation=M.PARAM_SIGMOID)
		mod.fcLayer(120)
		return mod.get_current_layer()

def build_graph():
	img_holder = tf.placeholder(tf.float32,[None,256,256,3])
	lab_holder = tf.placeholder(tf.float32,[None,120])
	last_layer = build_model(img_holder)
	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=lab_holder,logits=last_layer))
	accuracy = M.accuracy(last_layer,tf.argmax(lab_holder,1))
	train_step = tf.train.AdamOptimizer(0.1).minimize(loss)
	return img_holder,lab_holder,loss,train_step,accuracy,last_layer

MAX_ITER = 1000000
BSIZE = 32 #16
img_holder,lab_holder,loss,train_step,accuracy,last_layer = build_graph()
data,label = data_reader()

with tf.Session() as sess:
	saver = tf.train.Saver()
	M.loadSess('./model/',sess,init=True)
	for i in range(MAX_ITER):
		databatch = random.sample(data,BSIZE)
		img_batch = [i[0] for i in databatch]
		label_batch = [i[1] for i in databatch]
		_,acc,ls = sess.run([train_step,accuracy,loss],feed_dict={img_holder:img_batch ,lab_holder:label_batch})
		if i%100==0:
			print('iter',i,'\t|acc:',acc,'\tloss:',ls)

		if i%2000==0 and iteration!=0:
			saver.save(sess,'./model/'+str(iteration)+'.ckpt')
