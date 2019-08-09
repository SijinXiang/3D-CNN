from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np


class InputHandle(object):
	"""Class for handling dataset inputs."""

	def __init__(self, input_param):
		self.is_tra_set = input_param['tra_set']
		self.paths = input_param['paths']
		self.name = input_param['name']
		self.input_data_type = input_param.get('input_data_type', 'float32')
		self.output_data_type = input_param.get('output_data_type', 'float32')
		self.minibatch_size = input_param['minibatch_size']
		self.input_seq_length = input_param['input_seq_length']
		self.output_seq_length = input_param['output_seq_length']

		self.data = {'input_raw_data': [],
					'output_raw_data': []}
		self.indices = []
		self.current_position = 0
		self.current_batch_size = 0
		self.current_batch_indices = []
		self.current_input_length = 0
		self.mean = 0
		self.std = 0
		self.load()


	def load(self):
		"""Load the data."""
		snapshots = np.load(self.paths)
		total_shots = snapshots.shape[0]

		self.mean = np.mean(snapshots)
		self.std = np.std(snapshots)
		norm = snapshots - self.mean
		snapshots = np.nan_to_num(np.divide(norm, self.std))
		if self.is_tra_set == False:
			for i in range(total_shots // (self.input_seq_length+self.output_seq_length)):
				start = int(i * (self.input_seq_length + self.output_seq_length))
				end = int((i+1) * (self.input_seq_length + self.output_seq_length))

				in_raw = snapshots[start:start+self.input_seq_length].reshape((-1, 100, 100, 1))
				out_raw = snapshots[start+self.input_seq_length:end].reshape((-1, 100, 100, 1))

				self.data['input_raw_data'].append(in_raw)
				self.data['output_raw_data'].append(out_raw)

		else:
			total_sequence = total_shots - (self.input_seq_length + self.output_seq_length) + 1
			for i in range(total_sequence):
				end = int(i + (self.input_seq_length + self.output_seq_length))

				in_raw = snapshots[i:i+self.input_seq_length].reshape((-1, 100, 100, 1))
				out_raw = snapshots[i+self.input_seq_length:end].reshape((-1, 100, 100, 1))
                
				self.data['input_raw_data'].append(in_raw)
				self.data['output_raw_data'].append(out_raw)

        

	def total(self):
		"""Returns the total number of clips."""
		return len(self.data['input_raw_data'])


	def begin(self, do_shuffle=True):
		"""Move to the begin of the batch."""
		self.indices = np.arange(self.total(), dtype='int32')
        
		if do_shuffle:
			random.shuffle(self.indices)

		self.current_position = 0
		if self.current_position + self.minibatch_size <= self.total():
			self.current_batch_size = self.minibatch_size
		else:
			self.current_batch_size = self.total() - self.current_position

		self.current_batch_indices = self.indices[self.current_position:self.current_position +self.current_batch_size]


	def next_batch(self):
		"""Move to the next batch."""
		self.current_position += self.current_batch_size
		if self.no_batch_left():
			return None

		if self.current_position + self.minibatch_size <= self.total():
			self.current_batch_size = self.minibatch_size
		else:
			self.current_batch_size = self.total() - self.current_position

		self.current_batch_indices = self.indices[self.current_position:self.current_position +self.current_batch_size]


	def no_batch_left(self):
		if self.current_position >= self.total() - self.current_batch_size:
			return True
		else:
			return False


	def input_batch(self):
		"""Processes for the input batches."""
		if self.no_batch_left():
			return None

		input_batch = np.zeros((self.current_batch_size, int(self.input_seq_length), 100, 100, 1))

		for i in range(self.current_batch_size):
			batch_ind = self.current_batch_indices[i]
			input_batch[i] = self.data['input_raw_data'][batch_ind]

		return input_batch


	def output_batch(self):
		"""Processes for the output batches."""
		if self.no_batch_left():
			return None

		output_batch = np.zeros((self.current_batch_size, int(self.output_seq_length), 100, 100, 1))

		for i in range(self.current_batch_size):
			batch_ind = self.current_batch_indices[i]
			output_batch[i] = self.data['output_raw_data'][batch_ind]

		return output_batch


	def get_batch(self):
		input_seq = self.input_batch()
		output_seq = self.output_batch()
		batch = np.concatenate((input_seq, output_seq), axis=1)
		return batch
