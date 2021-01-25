#coding: utf-8
import sys
import numpy as np
from collections import Counter
import tensorflow.contrib.keras as kr

def write_file(file,data): ### 输入数组
	with open(file,'w',encoding='utf-8') as f:
		f.write('\n'.join(data) + '\n')
	f.close()

def read_file(file):
	with open(file,'r',encoding='utf-8') as f:
		lines = f.readlines()
		new_lines = []
		for line in lines:
			new_line = line.strip()
			new_lines.append(new_line)
	f.close()
	return new_lines


''' 制作vocab词汇表'''
def build_vocab(train_data,vocab_dir):
	labels = []
	contents = []
	with open(train_data,'r',encoding="utf-8") as f:
		all_line = f.readlines()
		for line in all_line:
			label,content = line.strip().split(' ')
			if content:
				labels.append(label)
				contents.append(content)
	f.close()
	all_data = []
	for content in all_line:
		content = content.strip().replace("\n","")
		all_data.extend(content) ### 将所有汉字加入到numpy
	counter = Counter(all_data)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
	words, _ = list(zip(*count_pairs))
	words = ['<PAD>'] + list(words)
	#word_to_id = dict(zip(words, range(len(words))))  # 按字频排序后，{字：序号} 高频：0,1...
	write_file(vocab_dir, words)
	print("number of words:",len(words))
	return words

'''制作label的序列'''
def label_dict():
	categories = ['无效', '有效']
	#categories = [x for x in categories]  #### 不是python3的话，需要对汉字编码
	cat_to_id = dict(zip(categories, range(len(categories))))
	return categories, cat_to_id

def create_voabulary(train_data,vocab_data,max_length):
	######  制作vocab 字典 ###################
	vocab_list = read_file(vocab_data)
	vocabulary_word2index = {}  ### word ：index
	vocabulary_index2word = {}  ### index ：word
	for i,vocab in enumerate(vocab_list):
		vocabulary_word2index[vocab] = i
		vocabulary_index2word[i] = vocab

	######## 文本编码  ##############################
	train_data_lines = read_file(train_data)
	categories, cat_to_id = label_dict()
	X = []
	Y = []
	for line in train_data_lines:
		#label,content = line.split('	')
		label,content = line.strip().split(' ')
		content = [vocabulary_word2index.get(e,0) for e in content]
		label = cat_to_id[label]
		X.append(content)
		Y.append(label)
	#print("X[0]:",X[0])
	print("Y[0]:",Y[0])
	x_pad = kr.preprocessing.sequence.pad_sequences(X, max_length) #### 对数据进行定长处理
	y_pad = kr.utils.to_categorical(Y, num_classes=len(cat_to_id)) ### 对label进行onehot处理 ： 0 ==> [1,0,0,0,0,..,0,0]
	#out_data = (X,Y)
	return x_pad,y_pad

"""生成批次数据"""
def batch_iter(x, y, batch_size=64):
	data_len = len(x)
	num_batch = int((data_len - 1) / batch_size) + 1
	#print("Total batch:",num_batch)
	indices = np.random.permutation(np.arange(data_len)) ## 随机打乱原来的元素顺序  ## np.arange 生成的等差一维数组
	#print("Indices:",indices)
	x_shuffle = x[indices]
	y_shuffle = y[indices]

	for i in range(num_batch):
		start_id = i * batch_size
		end_id = min((i + 1) * batch_size, data_len)
		yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

def compute_acc(pred,label): ### (Batch,10)
	batch,class_num = pred.shape
	num = 0
	for i in range(batch):
		if pred[i] == label[i]:
			num += 1
	acc = num / batch
	return acc

if __name__=="__main__":
	#build_vocab(train_data='./data/all.txt',vocab_dir='./data/vocab.txt') ### 制作词汇表
	#categories, cat_to_id=label_dict() ### 制作label词汇表
	#create_voabulary(train_data='../data/cnews.train.txt', vocab_data='./cnews_vocab.txt')

	#indices = np.random.permutation(np.arange(50000))
	print("Stop!")
