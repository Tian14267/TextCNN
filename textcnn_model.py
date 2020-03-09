#coding: utf-8
import tensorflow as tf
import config

def conv2d(input,filter_size,embedding_dim,input_channel, out_channel, padding="VALID",stride=1, data_format='NHWC', name=None):
	with tf.variable_scope(name):
		w = tf.get_variable("weight_cnn", [filter_size, embedding_dim, input_channel, out_channel],
							initializer=tf.truncated_normal_initializer(stddev=0.1))
		b = tf.get_variable("biase_cnn", [out_channel], initializer=tf.truncated_normal_initializer(0.0))
		conv = tf.nn.conv2d(input, w,strides=[1, stride, stride, 1], padding=padding, data_format=data_format)
		ret = tf.identity(tf.nn.bias_add(conv, b, data_format=data_format),name=name)
		conv = tf.contrib.layers.batch_norm(ret, is_training=True, scope='cnn_bn')
		relu = tf.nn.relu(features=conv, name='relu_cnn')

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
		embedding = tf.get_variable('embedding_cnn', [self.vocab_size, self.config.embedding_dim])
		embedding_inputs = tf.nn.embedding_lookup(embedding, input_x)

		self.sentence_embeddings_expanded = tf.expand_dims(embedding_inputs, -1)  ### (?,200,128) ==> (?,200,128,1)
		#filter = [6,self.config.embedding_dim,1,128]

		conv1 = conv2d(input=self.sentence_embeddings_expanded, filter_size=6, embedding_dim=self.config.embedding_dim,
					   input_channel=1, out_channel=128, padding="VALID", stride=1, data_format='NHWC', name='conv1')
		pool1 = tf.nn.max_pool(conv1, ksize=[1, self.config.seq_length - 6 + 1, 1, 1], strides=[1, 1, 1, 1],
								padding='VALID', name="pool1")

		conv2 = conv2d(input=self.sentence_embeddings_expanded, filter_size=7, embedding_dim=self.config.embedding_dim,
					   input_channel=1, out_channel=128,padding="VALID", stride=1, data_format='NHWC', name='conv2')
		pool2 = tf.nn.max_pool(conv2, ksize=[1, self.config.seq_length - 7 + 1, 1, 1], strides=[1, 1, 1, 1],
								 padding='VALID', name="pool2")

		conv3 = conv2d(input=self.sentence_embeddings_expanded, filter_size=8, embedding_dim=self.config.embedding_dim,
					   input_channel=1, out_channel=128,padding="VALID", stride=1, data_format='NHWC', name='conv3')
		pool3 = tf.nn.max_pool(conv3, ksize=[1, self.config.seq_length - 8 + 1, 1, 1], strides=[1, 1, 1, 1],
								 padding='VALID', name="pool3")

		self.h = tf.concat([pool1,pool2,pool3], 3)  ### (?,1,1,384)
		self.h_flat = tf.reshape(self.h, [-1, 128*3])  ### (?,384)

		with tf.name_scope("dropout_cnn"):
			self.h_drop = tf.nn.dropout(self.h_flat, keep_prob=self.keep_prob)

		self.h_dens = tf.layers.dense(self.h_drop, 128*3, activation=tf.nn.tanh, use_bias=True)
		with tf.name_scope("output_cnn"):
			W_projection = tf.get_variable("W_projection_cnn",shape=[128*3, self.config.num_classes],
										   initializer=tf.random_normal_initializer(stddev=0.1))
			b_projection =tf.get_variable("b_projection_cnn",shape=[self.config.num_classes])
			#logits = tf.matmul(self.h_dens,W_projection) + b_projection
		logits = tf.add(tf.matmul(self.h_dens,W_projection), b_projection, name="output")

		return logits

def textcnn_loss(logits,label):
	with tf.name_scope("loss_cnn"):
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label,logits=logits)
		loss = tf.reduce_mean(cross_entropy)
	return loss

def textcnn_acc(logits,labels): ### 计算 acc 正确率
	##########  输入 logits 和 label都是one_hot编码
	####  labels = [[1,0,0,0,0,...,0],[0,1,0,0,...,0],...]
	####  logits =
	with tf.name_scope("acc_cnn"):
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




