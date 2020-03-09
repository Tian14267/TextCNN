#coding: utf-8
import tensorflow as tf
import tools
import os
import tensorflow.contrib.keras as kr

os.environ['CUDA_VISIBLE_DEVICES']='0'


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
	test_X = create_onehot(one_data=inputs, vocab_data="./cnews_vocab.txt",max_length=200)

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
			textcnn_pred = tf.argmax(tf.nn.softmax(output_tensor_name), 1)
			out=sess.run(textcnn_pred, feed_dict={input_image_tensor: test_X})
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
	Textcnn_pb('./cnn.pb',inputs)
	print("########   Finished!")
