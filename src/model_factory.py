import os
from src import cnn3d_net
import tensorflow as tf

class Model(object):
    
	def __init__(self, configs):
		self.configs = configs
        
		self.x = [tf.placeholder(tf.float32, 
                                 [self.configs.batch_size,
                                  configs.input_seq_length + configs.output_seq_length,
                                  self.configs.img_width,
                                  self.configs.img_width,
                                  1]) for i in range(self.configs.n_gpu)]
        
		grads = []
		loss_train = []
		self.pred_seq = []
		self.tf_lr = tf.placeholder(tf.float32, shape=[])
		self.itr = tf.placeholder(tf.float32, shape=[])
		self.params = dict()
		num_hidden = [int(x) for x in self.configs.num_hidden.split(',')]
		num_layers = len(num_hidden)
        
		for i in range(self.configs.n_gpu):
			with tf.device('/gpu:%d' % i):
				with tf.variable_scope(tf.get_variable_scope(), reuse=True if i > 0 else None):
					output_list = cnn3d_net.cnn_net(self.x[i], num_layers, num_hidden)
					gen_ims = output_list[0]
					loss = output_list[1]
					loss_train.append(loss / self.configs.batch_size)
					all_params = tf.trainable_variables()
					grads.append(tf.gradients(loss, all_params))
					self.pred_seq.append(gen_ims)
            
		with tf.device('/gpu:0'):
			for i in range(1, self.configs.n_gpu):
				loss_train[0] += loss_train[i]
				for j in range(len(grads[0])):
					grads[0][j] += grads[i][j]

		train_step = tf.train.AdamOptimizer(self.configs.lr)
		self.loss_train = loss_train[0] / self.configs.n_gpu
        
		variables = tf.global_variables()
		self.saver = tf.train.Saver(variables)
		init = tf.global_variables_initializer()
		config_prot = tf.ConfigProto()
		config_prot.gpu_options.allow_growth = configs.allow_gpu_growth
		config_prot.allow_soft_placement = True
		self.sess = tf.Session(config=config_prot)
		self.sess.run(init)
		if self.configs.pretrained_model:
			self.saver.restore(self.sess, self.configs.pretrained_model)

		def train(self, inputs, lr, itr):
			feed_dict = {self.x[i]: inputs[i] for i in range(self.configs.n_gpu)}
			feed_dict.update({self.tf_lr: lr})
			feed_dict.update({self.itr: float(itr)})
			loss, _ = self.sess.run((self.loss_train, self.train_step), feed_dict)
			return loss
        
		def test(self, inputs):
			feed_dict = {self.x[i]: inputs[i] for i in range(self.configs.n_gpu)}
			gen_ims = self.sess.run(self.pred_seq, feed_dict)
			return gen_ims
        
		def save(self, itr):
			checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt')
			self.saver.save(self.sess, checkpoint_path, global_step=itr)
			print('saved to ' + self.configs.save_dir)

		def load(self, checkpoint_path):
			print('load model:', checkpoint_path)
			self.saver.restore(self.sess, checkpoint_path)

