# Many thanks to daya for modifying the code :)
# ==============================================================================

"""Main function to run the code."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import numpy as np
from src import datasets_factory
from src.model_factory import Model
import src.trainer as trainer
#from src.utils import preprocess
import tensorflow as tf
import argparse


def add_arguments(parser):
	parser.register("type", "bool", lambda v: v.lower() == "true")

	parser.add_argument("--train_data_paths", type=str, default="", help="train data paths")
	parser.add_argument("--valid_data_paths", type=str, default="", help="validation data paths")
	parser.add_argument("--test_data_paths", type=str, default="", help="train data paths")
	parser.add_argument("--save_dir", type=str, default="", help="dir to store trained net")
	parser.add_argument("--gen_frm_dir", type=str, default="", help="dir to store result.")

	parser.add_argument("--is_training", type="bool", nargs="?", const=True,
					default=False,
					help="training or testing")
	parser.add_argument("--dataset_name", type=str, default="milan", help="name of dataset")
	parser.add_argument("--input_seq_length", type=int, default=10, help="number of input snapshots")
	parser.add_argument("--output_seq_length", type=int, default=10, help="number of output snapshots")
	parser.add_argument("--img_width", type=int, default=100, help="input image width.")
    
	parser.add_argument("--model_name", type=str, default="3dcnn", help="The name of the architecture")
	parser.add_argument("--pretrained_model", type=str, default="", help=".ckpt file to initialize from")
	parser.add_argument("--num_hidden", type=str, default="10,10,10,10", help="COMMA separated number of units of 3dcnn layers")
	parser.add_argument("--pool_size", type=str, default="1,5,5", help="Pooling size depth-height-width")
	parser.add_argument("--kernel_size", type=int, default=5, help="filter size of a 3dcnn layer")
	parser.add_argument("--strides", type=int, default="1", help="strides")    
	parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
	parser.add_argument("--batch_size", type=int, default=50, help="batch size for training")
	parser.add_argument("--max_iterations", type=int, default=50, help="max num of steps")
	parser.add_argument("--display_interval", type=int, default=1, help="number of iters showing training loss")
	parser.add_argument("--test_interval", type=int, default=1, help="number of iters for test")
	parser.add_argument("--snapshot_interval", type=int, default=50, help="number of iters saving models")
	parser.add_argument("--n_gpu", type=int, default=1, help="how many GPUs to distribute the training across")
	parser.add_argument("--allow_gpu_growth", type="bool", nargs="?", const=True,
					default=True, 
					help="allow gpu growth")



def main(unused_argv):
	"""Main function."""
	print(FLAGS)
	# print(FLAGS.reverse_input)
	if FLAGS.is_training:
		if tf.gfile.Exists(FLAGS.save_dir):
			tf.gfile.DeleteRecursively(FLAGS.save_dir)
		tf.gfile.MakeDirs(FLAGS.save_dir)
	if not FLAGS.is_training:
		if tf.gfile.Exists(FLAGS.gen_frm_dir):
			tf.gfile.DeleteRecursively(FLAGS.gen_frm_dir)
		tf.gfile.MakeDirs(FLAGS.gen_frm_dir)

	gpu_list = np.asarray(
		os.environ.get('CUDA_VISIBLE_DEVICES', '-1').split(','), dtype=np.int32)
	FLAGS.n_gpu = len(gpu_list)
	print('Initializing models')

	model = Model(FLAGS)

	if FLAGS.is_training:
		train_wrapper(model)
	else:
		test_wrapper(model)


def train_wrapper(model):
	"""Wrapping function to train the model."""
	if FLAGS.pretrained_model:
		model.load(FLAGS.pretrained_model)
  # load data
	train_input_handle, test_input_handle = datasets_factory.data_provider(
		FLAGS.dataset_name,
		FLAGS.train_data_paths,
		FLAGS.valid_data_paths,
		FLAGS.batch_size * FLAGS.n_gpu,
		FLAGS.img_width,
		FLAGS.input_seq_length,
		FLAGS.output_seq_length,
		is_training=True)

	tra_cost = 0.0
	batch_id = 0
	stopping = [10000000000000000]
	for itr in range(1170, FLAGS.max_iterations + 1):
		if itr == 2:
			print('training process started...')
			#model.save(itr)
			#print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'itr: ' + str(itr))
			#print('training loss: ' + str(tra_cost / batch_id))
			#val_cost = trainer.test(model, test_input_handle,FLAGS, itr)
		if train_input_handle.no_batch_left():
			model.save(itr)
			print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'itr: ' + str(itr))
			print('training loss: ' + str(tra_cost / batch_id))
			val_cost = trainer.test(model, test_input_handle,FLAGS, itr)
			if val_cost < min(stopping):
				stopping = [val_cost]
			elif len(stopping) < 5:
				stopping.append(val_cost)
			if len(stopping) == 5:
				break
			train_input_handle.begin(do_shuffle=True)
			tra_cost = 0
			batch_id = 0
		if itr % 50 == 0:
			print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'itr: ' + str(itr))
			print('training loss: ' + str(tra_cost / batch_id))

		ims = train_input_handle.get_batch()
		batch_id += 1

		tra_cost += trainer.train(model, ims, FLAGS, itr)

		train_input_handle.next_batch()


def test_wrapper(model):
	model.load(FLAGS.pretrained_model)
	test_input_handle = datasets_factory.data_provider(
		FLAGS.dataset_name,
		FLAGS.train_data_paths, 
		FLAGS.test_data_paths, # Should use test data rather than training or validation data.
		FLAGS.batch_size * FLAGS.n_gpu,
		FLAGS.img_width,
		FLAGS.input_seq_length,
		FLAGS.output_seq_length,
		is_training=False)
	trainer.test(model, test_input_handle, FLAGS, 'test_result')


if __name__ == '__main__':
	nmt_parser = argparse.ArgumentParser()
	add_arguments(nmt_parser)
	FLAGS, unparsed = nmt_parser.parse_known_args()
	tf.app.run(main=main)

