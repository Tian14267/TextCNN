#coding: utf-8
import tensorflow as tf
import tools
import config
import tensorflow.contrib.keras as kr
from textcnn_model import TextCNN

class Textcnn_pred(object):
	def __init__(self,vocab_dir):
		self.input_x = tf.placeholder(tf.int32, [None, config.seq_length], name='input_x')
		self.words = tools.read_file(vocab_dir)
		self.vocab_size = len(self.words)

		self.textcnn = TextCNN(config,self.vocab_size,keep_prob = 1.0)
		self.logits = self.textcnn.cnn(self.input_x)
		self.textcnn_pred = tf.argmax(tf.nn.softmax(self.logits), 1)

		saver = tf.train.Saver()
		sess_config = tf.ConfigProto(allow_soft_placement=True)

		sess_config.gpu_options.per_process_gpu_memory_fraction = 0.8
		sess_config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=sess_config)
		model_path = 'checkpoints/model/TextCNNnet_2019-10-17-14-35-50.ckpt-9000'
		saver.restore(sess=self.sess, save_path=model_path)
		print("################ load TextCNN model down! ##########################")

	def _close(self):
		self.sess.close()

	def text(self,input):
		logit,pred = self.sess.run([self.logits,self.textcnn_pred], feed_dict={self.input_x:input})

		return pred

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

def decode_text(labels):
	categories, cat_to_id = tools.label_dict()
	words = []
	for word_num in labels:
		word = categories[word_num]
		words.append(word)

	return words

if __name__ == '__main__':
	one_text = ['三星ST550以全新的拍摄方式超越了以往任何一款数码相机',
                 '热火vs骑士前瞻：皇帝回乡二番战 东部次席唾手可得新浪体育讯北京时间3月30日7:00']
	test_X = create_onehot(one_data=one_text, vocab_data="./cnews_vocab.txt",
											max_length=config.seq_length)
	textcnn = Textcnn_pred(vocab_dir="./cnews_vocab.txt")
	pred = textcnn.text(test_X)
	result = decode_text(pred)
	print(result)
