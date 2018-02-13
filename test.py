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
		#label1 = []
		#code = []
		df = pd.read_csv('sample_submission.csv')
		#df1 = pd.read_csv('labels.csv')
		#df1 = pd.read_csv('breeds.csv')
		#code = df1['id'].as_matrix()
		#code=LabelEncoder().fit_transform(df1['breed'])
		#print(code)
		#i=0
		for index, row in df.iterrows():
			image_path = 'test/'+row['id']+'.jpg'
			img = cv2.imread(image_path)
			#print(image_path)
			img = cv2.resize(img,(256,256))
			#print(img)
			#cv2.imshow('image',img)
			#cv2.waitKey(0)
			#label=LabelEncoder().fit_transform(row['breed'])
			#label1= code[i]
			#i= i+1
			data_row = [img]
			data.append(data_row)
		self.data = data

	def get_i(self,i):
		return [self.data[i]]

	def get_j(self):
		return len(self.data)


def main_structure(inp):
	with tf.variable_scope('mainModel'):
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
	pred = tf.cast(tf.argmax(last_layer,1),tf.int64)
	#bias_loss = tf.reduce_sum(tf.reduce_mean(tf.square(last_layer - img_holder),axis=0))
	with tf.variable_scope('conf_loss'):
		conf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label,logits=last_layer))
		tf.summary.scalar('loss', conf_loss)

	accuracy = M.accuracy(last_layer,lab_holder)

	with tf.variable_scope('train'):
		train_step = tf.train.AdamOptimizer(0.01).minimize(conf_loss)

	return img_holder,lab_holder,conf_loss,train_step,last_layer,accuracy, pred


reader = data_reader()
img_holder,lab_holder,conf_loss,train_step,last_layer,accuracy,pred = build_graph()

with tf.Session() as sess:
	code = []
	saver = tf.train.Saver()	
	reader = data_reader()
	M.loadSess('./model/',sess,init=True)
	#merged = tf.summary.merge_all()
	#writer = tf.summary.FileWriter('logs/',sess.graph)
	MAXITER = reader.get_j()
	df1 = pd.read_csv('labels.csv')
	code=LabelEncoder().fit(df1['breed'])
	print('Reading finish')
	for iteration in range(MAXITER):
		# print(iteration)
		train_batch = reader.get_i(iteration)
		# print(iteration)
		img_batch = [i[0] for i in train_batch]	
		#lab_batch = [i[1] for i in train_batch]
		feeddict = {img_holder:img_batch}
		# print(iteration)
		prediction = sess.run([pred],feed_dict=feeddict)
		#writer.add_summary(result,iteration)

		if iteration%1==0:
			Breed = list(code.inverse_transform ([prediction[0]]))
			print('Prediction:',list(Breed))
			img = img_batch[0].astype(np.uint8)
			cv2.imshow('img',img)
			cv2.waitKey(0)

			# print(c.max())
			#img = img_batch[0].astype(np.uint8)

		#if iteration%5000==0 and iteration!=0:
			#saver.save(sess,'./model/'+str(iteration)+'.ckpt'
