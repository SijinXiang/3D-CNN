from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os.path
import numpy as np


def train(model, ims, configs, itr):
	"""Trains a model."""
	ims_list = np.split(ims, configs.n_gpu)
	cost = model.train(ims_list, configs.lr, itr)
	return cost

def test(model, test_input_handle, configs, save_name):
	if not configs.is_training:
		print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')

	test_input_handle.begin(do_shuffle=False)

	if configs.is_training:
		res_path = os.path.join(configs.save_dir, 'model.ckpt-' + str(save_name))
	else:
		res_path = os.path.join(configs.gen_frm_dir, str(save_name)) # When testing, save_name = 'test_results'
		os.mkdir(res_path)
        
	avg_mse = 0
	batch_id = 0
	img_mse = []
	output_length = int(configs.output_seq_length)

	for i in range(output_length):
		img_mse.append(0)

	while not test_input_handle.no_batch_left():
		batch_id = batch_id + 1
		test_ims = test_input_handle.get_batch()
		test_dat = np.split(test_ims, configs.n_gpu)
		img_out = model.test(test_dat)
		target_out = test_ims[:, -output_length:]
		#print('test_ims: ',np.shape(test_ims))
		#print('img_out: ',np.shape(img_out))
        
		for i in range(output_length):
			x = target_out[:, i]
			gx = img_out[:, int(i)]
			mse = np.square(x - gx).sum()
			img_mse[i] += mse
			avg_mse += mse
        
		if not configs.is_training:
			test_ims = test_ims * test_input_handle.std + test_input_handle.mean
			img_out = img_out * test_input_handle.std + test_input_handle.mean
			target_out = test_ims[:, -output_length:]

			path = os.path.join(res_path, str(batch_id))
			os.mkdir(path)
			name = 'input.npy'
			file_name = os.path.join(path, name)
			np.save(file_name, np.asarray(test_ims)[:,:int(configs.input_seq_length)])
			name = 'target_output.npy'
			file_name = os.path.join(path, name)
			np.save(file_name,np.asarray(target_out))
			name = 'predict_output.npy'
			file_name = os.path.join(path, name)
			np.save(file_name,np.asarray(img_out))

		test_input_handle.next_batch()

	avg_mse = avg_mse / test_input_handle.total()
	avg_mse_snapshot = avg_mse / (configs.output_seq_length)
	img_mse = np.asarray(img_mse) / test_input_handle.total()

	if configs.is_training:
		print('validation loss:\n' + 'mse per seq: ' + str(avg_mse) + '\tmse per snapshot: ' + str(avg_mse_snapshot))
		return avg_mse

	else:
		avg_result = np.zeros(2+len(img_mse))
		avg_result[0] = avg_mse
		avg_result[1] = avg_mse_snapshot
		avg_result[2:] = img_mse
		name = 'test_loss.txt'
		file_name = os.path.join(res_path, name)
		np.savetxt(file_name, avg_result, delimiter=',')

