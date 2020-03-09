#coding: utf-8
import numpy as np
import tensorflow as tf
import tools
import os
import time
import logging
import config
from tflearn.data_utils import pad_sequences
from textcnn_model import TextCNN,textcnn_loss,textcnn_acc
import tensorflow.contrib.keras as kr
#from textcnn_new_model import TextCNN,textcnn_loss,textcnn_acc
os.environ['CUDA_VISIBLE_DEVICES']='0'

FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_float("learning_rate",0.001,"learning rate")
tf.app.flags.DEFINE_string("vocab_dir","../data/cnews_vocab.txt","vocab data path")

tf.app.flags.DEFINE_string("train_data","../data/cnews.train.txt","training data path")
tf.app.flags.DEFINE_string("val_data","../data/cnews.val.txt","val data path")
tf.app.flags.DEFINE_string("test_data","../data/cnews.test.txt","test data path")
tf.app.flags.DEFINE_boolean("model_store",False,"If restore model")

def Textcnn_test():
	if FLAGS.vocab_dir is None:
		words = tools.build_vocab(train_data=FLAGS.train_data, vocab_dir=FLAGS.vocab_dir)  ### 制作词汇表
	else:
		words = tools.read_file(FLAGS.vocab_dir)
	vocab_size = len(words)
	print("Test words : ",vocab_size)
	test_X, test_Y = tools.create_voabulary(train_data=FLAGS.test_data, vocab_data=FLAGS.vocab_dir,
											max_length=config.seq_length)

	input_x = tf.placeholder(tf.int32, [None, config.seq_length], name='input_x')
	input_y = tf.placeholder(tf.float32, [None, config.num_classes], name='input_y')

	model_path = 'checkpoints/TextCNNnet_2019-11-01-15-31-50.ckpt-4000'

	save_path = model_path
	sess_config = tf.ConfigProto(allow_soft_placement=True)
	sess_config.gpu_options.allow_growth = True
	sess = tf.Session(config=sess_config)

	textcnn = TextCNN(config, vocab_size, keep_prob=1.0)
	logits = textcnn.cnn(input_x)  ### (?,10)
	loss = textcnn_loss(logits=logits, label=input_y)
	acc = textcnn_acc(logits=logits, labels=input_y)

	saver = tf.train.Saver()
	saver.restore(sess=sess, save_path=save_path)

	batch_test = tools.batch_iter(test_X, test_Y, config.batch_size)  ### 生成批次数据
	i = 0
	all_acc = 0
	for x_batch, y_batch in batch_test:	
		test_loss,test_acc = sess.run([loss,acc],feed_dict = {input_x:x_batch, input_y:y_batch })
		all_acc = all_acc + test_acc
		i += 1

	print("Average acc : ",(all_acc/i))


def create_onehot(one_data,vocab_data,max_length):
	vocab_list = tools.read_file(vocab_data)
	vocabulary_word2index = {}  ### word ：index
	vocabulary_index2word = {}  ### index ：word
	for i, vocab in enumerate(vocab_list):
		vocabulary_word2index[vocab] = i
		vocabulary_index2word[i] = vocab

	if isinstance(one_data, str):###是字符串，即txt
		one_data_list = tools.read_file(one_data)
	else:
		one_data_list = one_data ### 数组
	X = []
	for data in one_data_list:
		content = [vocabulary_word2index.get(e, 0) for e in data]
		X.append(content)
	x_pad = kr.preprocessing.sequence.pad_sequences(X, max_length)  #### 对数据进行定长处理
	return x_pad

def Textcnn_pb(pb_path, inputs):
	test_X = create_onehot(one_data=inputs, vocab_data="../data/cnews_vocab.txt",max_length=config.seq_length)

	with tf.Graph().as_default():
		output_graph_def = tf.GraphDef()
		with open(pb_path, "rb") as f:
			output_graph_def.ParseFromString(f.read())
			tf.import_graph_def(output_graph_def, name="")
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			# 定义输入的张量名称,对应网络结构的输入张量
			# input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数
			input_image_tensor = sess.graph.get_tensor_by_name("input_x:0")

			# 定义输出的张量名称
			output_tensor_name = sess.graph.get_tensor_by_name("output:0")

			out=sess.run(output_tensor_name, feed_dict={input_image_tensor: test_X})

			result = decode_text(out)
			print(result)


def decode_text(labels):
	categories, cat_to_id = tools.label_dict()
	words = []
	for word_num in labels:
		word = categories[word_num]
		words.append(word)

	return words

if __name__=="__main__":
	
	#Textcnn_test()
	inputs = ['三星ST550以全新的拍摄方式超越了以往任何一款数码相机',
                 '热火vs骑士前瞻：皇帝回乡二番战 东部次席唾手可得新浪体育讯北京时间3月30日7:00']
	Textcnn_pb('./freeze/cnn.pb',inputs)
	print("########   Finished!")
