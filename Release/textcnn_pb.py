#coding: utf-8
import tensorflow as tf
import tools
import tensorflow.keras as kr

def create_onehot(one_data,vocab_data,max_length):
	vocab_list = tools.read_file(vocab_data)
	vocabulary_word2index = {}  ### word ：index
	vocabulary_index2word = {}  ### index ：word
	for i, vocab in enumerate(vocab_list):
		vocabulary_word2index[vocab] = i
		vocabulary_index2word[i] = vocab

	if isinstance(one_data, str) and ".txt" in one_data:###是字符串，即txt
		one_data_list = tools.read_file(one_data)
		singleTest = False
	else:
		one_data_list = one_data ### 数组
		singleTest = True
	X = []
	if not singleTest:
		for data in one_data_list:
			content = [vocabulary_word2index.get(e, 0) for e in data]
			X.append(content)
		x_pad = kr.preprocessing.sequence.pad_sequences(X, max_length)  #### 对数据进行定长处理
	else:
		content = [vocabulary_word2index.get(e, 0) for e in one_data_list]
		X.append(content)
		x_pad = kr.preprocessing.sequence.pad_sequences(X, max_length)
	return x_pad

def decode_text(labels):
	categories, cat_to_id = tools.label_dict()
	words = []
	for word_num in labels:
		word = categories[word_num]
		words.append(word)
	return words

class Textcnn_pred(object):
	def __init__(self,pb_path):
		with tf.Graph().as_default():
			self.output_graph_def = tf.GraphDef()
			with open(pb_path, "rb") as f:
				self.output_graph_def.ParseFromString(f.read())
				tf.import_graph_def(self.output_graph_def, name="")
			sess_config = tf.ConfigProto(allow_soft_placement=True)
			sess_config.gpu_options.per_process_gpu_memory_fraction = 0.8
			sess_config.gpu_options.allow_growth = True
			self.sess = tf.Session(config=sess_config)
			self.sess.run(tf.global_variables_initializer())
			# 定义输入的张量名称,对应网络结构的输入张量
			# input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数
			self.input_image_tensor = self.sess.graph.get_tensor_by_name("input_x:0")
			# 定义输出的张量名称
			output_tensor_name = self.sess.graph.get_tensor_by_name("output:0")
			self.textcnn_pred = tf.argmax(tf.nn.softmax(output_tensor_name), 1)
		print("################ load TextCNN model down! ##########################")

	def _close(self):
		self.sess.close()

	def text_pre(self,input):
		test_X = create_onehot(one_data=input, vocab_data="./model_and_data/vocab.txt", max_length=200)
		out=self.sess.run(self.textcnn_pred, feed_dict={self.input_image_tensor: test_X})
		result = decode_text(out)
		return result

if __name__=="__main__":
	textcnn = Textcnn_pred(pb_path="./model_and_data/textcnn_twoClass.pb")
	while 1:
		#inputs = ['龙岗区四联路30号路段多辆车辆违停,私设路障,严重影响车辆和行人通行。',
		#          '馨荔苑业主群，13：44分报警人发来短信：对不起，拨错号了，歉意。',
	    #        ]
		print("开始输入：")
		inputs = input()
		pred = textcnn.text_pre(inputs)[0]
		print("Result: ",pred)