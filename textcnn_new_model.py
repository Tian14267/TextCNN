#coding: utf-8
import tensorflow as tf
import config

def conv2d(input,filter_size_w,filter_size_h,input_channel, out_channel, padding="VALID",stride=1, data_format='NHWC', name=None):
	with tf.variable_scope(name):
		w = tf.get_variable("weight", [filter_size_w, filter_size_h, input_channel, out_channel],
							initializer=tf.truncated_normal_initializer(stddev=0.1))
		b = tf.get_variable("biase", [out_channel], initializer=tf.truncated_normal_initializer(0.0))
		conv = tf.nn.conv2d(input, w,strides=[1, stride, stride, 1], padding=padding, data_format=data_format)
		ret = tf.identity(tf.nn.bias_add(conv, b, data_format=data_format),name=name)
		conv = tf.contrib.layers.batch_norm(ret, is_training=True, scope='cnn_bn')
		relu = tf.nn.relu(features=conv, name='relu')

	return relu

class TextCNN(object):
	def __init__(self,config,vocab_size,keep_prob):
		self.config = config
		self.is_training_flag = True
		self.keep_prob = keep_prob
		self.vocab_size = vocab_size

	def cnn(self,input_x):
		#################  词嵌入: 对输入进行 embedding 嵌入  #########################################
		##### 对词汇表数据进行映射，词汇数 5000 ==> 128
		embedding = tf.get_variable('embedding', [self.vocab_size, self.config.embedding_dim])
		embedding_inputs = tf.nn.embedding_lookup(embedding, input_x)

		self.sentence_embeddings_expanded = tf.expand_dims(embedding_inputs, -1)  ### (?,200,128) ==> (?,200,128,1)
		#filter = [6,self.config.embedding_dim,1,128]

		conv1_1 = conv2d(input=self.sentence_embeddings_expanded, filter_size_w=3, filter_size_h=1,
					   input_channel=1, out_channel=128, padding="VALID", stride=1, data_format='NHWC', name='conv1_1')
		_,w_1,h_1,_ = conv1_1.get_shape().as_list()  ### [?,198,128,128]
		pool1_1 = tf.nn.max_pool(conv1_1, ksize=[1, w_1/2, h_1/2, 1], strides=[1, 1, 1, 1],padding='VALID', name="pool1_1")
		conv1_2 = conv2d(input=pool1_1, filter_size_w=4, filter_size_h=h_1/2+1, input_channel=128, out_channel=256,
						 padding="VALID", stride=1, data_format='NHWC', name='conv1_2') ### [?,97,1,256]
		pool1_2 = tf.nn.max_pool(conv1_2, ksize=[1, w_1/2-2,1, 1], strides=[1,1,1,1], padding='VALID',name="pool1_2")### [?,1,1,256]
		'''
		conv2_1 = conv2d(input=self.sentence_embeddings_expanded, filter_size_w=5, filter_size_h=1,
					   input_channel=1, out_channel=128,padding="VALID", stride=1, data_format='NHWC', name='conv2_1')
		_, w_2, h_2, _ = conv2_1.get_shape().as_list() ### [?,196,128,128]
		pool2_1 = tf.nn.max_pool(conv2_1, ksize=[1, w_2/2, h_2/2, 1], strides=[1, 1, 1, 1],padding='VALID', name="pool2_1")
		conv2_2 = conv2d(input=pool2_1, filter_size_w=6, filter_size_h=h_2/2+1,input_channel=128, out_channel=256, padding="VALID",
						 stride=1, data_format='NHWC',name='conv2_2')### [?,97,1,256]
		pool2_2 = tf.nn.max_pool(conv2_2, ksize=[1, w_2/2-4,1, 1], strides=[1, 1, 1, 1], padding='VALID',name="pool2_2")### [?,1,1,256]

		conv3_1 = conv2d(input=self.sentence_embeddings_expanded, filter_size_w=7, filter_size_h=1,
					   input_channel=1, out_channel=128,padding="VALID", stride=1, data_format='NHWC', name='conv3_1')
		_, w_3, h_3, _ = conv3_1.get_shape().as_list()
		pool3_1 = tf.nn.max_pool(conv3_1, ksize=[1, w_3/2, h_3/2, 1], strides=[1, 1, 1, 1],padding='VALID', name="pool3_1")
		conv3_2 = conv2d(input=pool3_1, filter_size_w=8, filter_size_h=h_3 / 2 + 1, input_channel=128,
						 out_channel=256, padding="VALID",stride=1, data_format='NHWC', name='conv3_2')
		pool3_2 = tf.nn.max_pool(conv3_2, ksize=[1, w_3/2-6, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool3_2")
		'''
		#self.h = tf.concat([pool1_2,pool2_2,pool3_2], 3)  ### (?,1,1,384)
		self.h_flat = tf.reshape(pool1_2, [-1, 256])  ### (?,384)

		with tf.name_scope("dropout"):
			self.h_drop = tf.nn.dropout(self.h_flat, keep_prob=self.keep_prob)

		self.h_dens = tf.layers.dense(self.h_drop, 128, activation=tf.nn.tanh, use_bias=True)
		with tf.name_scope("output"):
			W_projection = tf.get_variable("W_projection",shape=[128, self.config.num_classes],
										   initializer=tf.random_normal_initializer(stddev=0.1))
			b_projection =tf.get_variable("b_projection",shape=[self.config.num_classes])
			logits = tf.matmul(self.h_dens,W_projection) + b_projection

		return logits

def textcnn_loss(logits,label):
	with tf.name_scope("loss"):
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label,logits=logits)
		loss = tf.reduce_mean(cross_entropy)
	return loss

def textcnn_acc(logits,labels): ### 计算 acc 正确率
	##########  输入 logits 和 label都是one_hot编码
	####  labels = [[1,0,0,0,0,...,0],[0,1,0,0,...,0],...]
	####  logits =
	with tf.name_scope("acc"):
		pred = tf.argmax(tf.nn.softmax(logits), 1)  #### 计算预测值 （64，）
		correct_pred = tf.equal(tf.argmax(labels, 1), pred)
		acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	return acc



if __name__=="__main__":
	#input = tf.zeros([4, 200,128], dtype=tf.float32)
	input = tf.placeholder(tf.int32, [4, 200, 128])
	import config
	textcnn = TextCNN(config,5000)
	logits=textcnn.cnn(input)
	print('over!')




