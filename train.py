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
		#img1 = []
		data = []
		label1 = []
		code = []
		df = pd.read_csv('labels.csv')
		#df1 = pd.read_csv('breeds.csv')
		#code = df1['id'].as_matrix()
		code=LabelEncoder().fit_transform(df['breed'])
		#print(code)
		i=0
		for index, row in df.iterrows():
			image_path = 'train/'+row['id']+'.jpg'
			img = cv2.imread(image_path)
			#print(image_path)
			img = cv2.resize(img,(256,256))
			#print(img)
			#cv2.imshow('image',img)
			#cv2.waitKey(0)
			#label=LabelEncoder().fit_transform(row['breed'])
			label1= code[i]
			i= i+1
			data_row = [img, label1]
			data.append(data_row)
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
		mod.dwconvLayer(5,16,activation=M.PARAM_LRELU)
		#mod.maxpoolLayer(2)
		#mod.incep(4,4,4,4,4,activation=M.PARAM_LRELU)
		#mod.incep(8,8,8,8,8,activation=M.PARAM_LRELU, batch_norm= True)
		mod.incep(1,8,1,8,1,activation=M.PARAM_LRELU, batch_norm= True)
		#mod.incep(8,8,8,8,8,activation=M.PARAM_LRELU, batch_norm= True)
		mod.incep(2,16,2,16,2,activation=M.PARAM_RELU, batch_norm= True)
		mod.flatten()
		mod.fcLayer(120)
		return mod.get_current_layer()


def build_graph():
	with tf.name_scope('img_holder'):
		img_holder = tf.placeholder(tf.float32,[None,256,256,3],name='image')
	with tf.name_scope('lab_holder'):
		lab_holder = tf.placeholder(tf.int32,[None],name='label')
	label = tf.one_hot(lab_holder,120)

	last_layer = main_structure(img_holder)

	#bias_loss = tf.reduce_sum(tf.reduce_mean(tf.square(last_layer - img_holder),axis=0))
	with tf.variable_scope('conf_loss'):
		conf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label,logits=last_layer))
		tf.summary.scalar('loss', conf_loss)

	accuracy = M.accuracy(last_layer,lab_holder)

	with tf.variable_scope('train'):
		train_step = tf.train.AdamOptimizer(0.01).minimize(conf_loss)

	return img_holder,lab_holder,conf_loss,train_step,last_layer,accuracy

MAXITER = 100000000
BSIZE = 16

reader = data_reader()
img_holder,lab_holder,conf_loss,train_step,last_layer,accuracy = build_graph()

with tf.Session() as sess:
	M.loadSess('./model/',sess,init=True)
	saver = tf.train.Saver()	
	reader = data_reader()
	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter('logs/',sess.graph)
	print('Reading finish')
	for iteration in range(MAXITER):
		# print(iteration)
		train_batch = reader.next_train_batch(BSIZE)
		# print(iteration)
		img_batch = [i[0] for i in train_batch]	
		lab_batch = [i[1] for i in train_batch]
		feeddict = {img_holder:img_batch, lab_holder:lab_batch}
		# print(iteration)
		_,acc,loss,result = sess.run([train_step,accuracy,conf_loss, merged],feed_dict=feeddict)
		writer.add_summary(result,iteration)

		if iteration%10==0:
			print('Iter:',iteration,'\tLoss_c:',loss,'\tAcc:',acc)
			# print(c.max())
			#img = img_batch[0].astype(np.uint8)

		if iteration%5000==0 and iteration!=0:
			saver.save(sess,'./model/'+str(iteration)+'.ckpt')
