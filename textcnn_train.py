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
#from textcnn_new_model import TextCNN,textcnn_loss,textcnn_acc
os.environ['CUDA_VISIBLE_DEVICES']='0'

FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_float("learning_rate",0.001,"learning rate")
tf.app.flags.DEFINE_string("vocab_dir","../data/cnews_vocab.txt","vocab data path")

tf.app.flags.DEFINE_string("train_data","../data/cnews.train.txt","training data path")
tf.app.flags.DEFINE_string("val_data","../data/cnews.val.txt","val data path")
tf.app.flags.DEFINE_string("test_data","../data/cnews.test.txt","test data path")
tf.app.flags.DEFINE_boolean("model_store",False,"If restore model")

def Textcnn_train():
	###########  load data  ###################
	if not os.path.exists(FLAGS.vocab_dir):
		words = tools.build_vocab(train_data=FLAGS.train_data, vocab_dir=FLAGS.vocab_dir)  ### 制作词汇表
	else:
		words = tools.read_file(FLAGS.vocab_dir)
	vocab_size = len(words)
	train_X,train_Y = tools.create_voabulary(train_data=FLAGS.train_data, vocab_data=FLAGS.vocab_dir,
											 max_length=config.seq_length)
	val_X, val_Y = tools.create_voabulary(train_data=FLAGS.val_data, vocab_data=FLAGS.vocab_dir,
											max_length=config.seq_length)

	#trainX = pad_sequences(train_X, maxlen=200, value=0.)  # padding to max length
	#test_X = pad_sequences(test_X, maxlen=200, value=0.)  # padding to max length
	print("Data deal down!")
	###############################################################################

	input_x = tf.placeholder(tf.int32, [None, config.seq_length], name='input_x')
	input_y = tf.placeholder(tf.float32, [None, config.num_classes], name='input_y')

	textcnn = TextCNN(config,vocab_size,keep_prob = config.dropout_keep_prob)
	logits = textcnn.cnn(input_x) ### (?,10)
	loss = textcnn_loss(logits=logits,label=input_y)

	############# 计算 acc ######################################
	acc = textcnn_acc(logits=logits,labels=input_y)
	######################################################

	global_step = tf.Variable(0, name='global_step', trainable=False)
	learning_rate = tf.train.exponential_decay(
		learning_rate=FLAGS.learning_rate,
		global_step=global_step,
		decay_steps=2000,
		decay_rate=0.1,
		staircase=True)

	optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss,global_step=global_step)

	tensorboard_dir = 'tensorboard/textcnn'
	tf.summary.scalar("loss", loss)
	merged_summary = tf.summary.merge_all()
	writer = tf.summary.FileWriter(tensorboard_dir)

	saver = tf.train.Saver(max_to_keep=3) ### 保存模型
	model_save_dir = 'checkpoints/'
	train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
	model_name = 'TextCNNnet_{:s}.ckpt'.format(str(train_start_time))
	model_save_path = os.path.join(model_save_dir, model_name)

	model_restore_path = './checkpoints/TextCNNnet_2019-11-01-15-31-50.ckpt-4000'

	##### 创建日志
	logging.basicConfig(filename='./checkpoints/'+model_name+'.log',format='%(asctime)s - %(pathname)s - %(levelname)s: %(message)s',
						level=logging.DEBUG,filemode='a',datefmt='%Y-%m-%d%I:%M:%S %p')
	logging.info('######  Next Is Training Infomation   ###################')

	sess_config = tf.ConfigProto(allow_soft_placement=True)
	sess_config.gpu_options.allow_growth = True
	sess = tf.Session(config=sess_config)
	
	with sess.as_default():
		if not FLAGS.model_store:
			step = 0
			init = tf.global_variables_initializer()
			sess.run(init)
			writer.add_graph(sess.graph)
		else:
			saver.restore(sess=sess, save_path=model_restore_path)
			step = sess.run(tf.train.get_global_step())
			writer.add_graph(sess.graph)
		print('First step is:', step)
		num_batch = int((len(train_X) - 1) / config.batch_size) + 1 ### 总batch数
		acc_begain = 0
		for epoch in range(config.epochs):
			batch_train = tools.batch_iter(train_X, train_Y, config.batch_size)  ### 生成批次数据
			Begain_learn_rate = FLAGS.learning_rate
			for x_batch, y_batch in batch_train:
				step += 1
				_, learn_rate,train_loss_value, train_pred, train_acc,merge_summary_value = sess.run(
					[optim,learning_rate, loss, logits,acc,merged_summary],feed_dict={input_x: x_batch, input_y: y_batch})
				if Begain_learn_rate != learn_rate:
					information = '############ New Learning_Rate {:6f} in step {:d}  ###########'.format(learn_rate,step) 
					logging.info(information)
					print(information)
					Begain_learn_rate = learn_rate
				if step % 10 == 0:
					information = '## Epoch {:d} Step_Train / Total_Batch: {:d} / {:d}   train_loss= {:5f}  train_acc={:5f}'.\
							format(int(step/num_batch),step, num_batch, train_loss_value, train_acc)
					logging.info(information)
					print(information)

				if step % 500 == 0: ### 每 500 步进行一次验证，并保存最优模型
					val_acc_all = 0
					val_loss_all = 0
					val_step = 0
					batch_val = tools.batch_iter(val_X, val_Y, config.batch_size)  ### 生成批次数据
					for x_val,y_val in batch_val:
						if x_val.shape[0] < config.batch_size:
							pass
						else:
							_, val_loss_value, val_pred, val_acc, merge_summary_value = sess.run(
								[optim, loss, logits, acc, merged_summary], feed_dict={input_x: x_val, input_y: y_val})
							writer.add_summary(merge_summary_value, step)
							val_acc_all = val_acc_all + val_acc
							val_loss_all = val_loss_all + val_loss_value
							val_step += 1
					ave_acc = val_acc_all / val_step
					ave_loss = val_loss_all / val_step
					if (ave_acc - acc_begain)>0.001:
						acc_begain = ave_acc
						saver.save(sess, model_save_path, global_step=step)
						tf.train.write_graph(sess.graph_def, '', './checkpoints/textcnn_graph.pb')
					information ='############   Val_loss = {:5f}   Val_acc = {:5f}   ##################'.format(ave_loss,ave_acc)
					logging.info(information)
					print(information)

			


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


if __name__=="__main__":
	Textcnn_train()
	#Textcnn_test()
	print("########   Finished!")
