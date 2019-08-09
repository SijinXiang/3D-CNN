from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from src import milan

def data_provider(dataset_name,
				train_data_paths,
				valid_data_paths,
				batch_size,
				img_width,
				in_seq_length,
				out_seq_length,
				is_training):

	'''Returns a Dataset.

  Args:
    dataset_name: String, the name of the dataset.
    train_data_paths: List, [train_data_path1, train_data_path2...]
    valid_data_paths: List, [val_data_path1, val_data_path2...]
    batch_size: Int, the batch size.
    img_width: Int, the width of input images.
    in_seq_length, number of input snapshots.
    out_seq_length, number of output snapshots.
    dims_3D, dimension of depth channel of 3D encoder
    is_training: Bool, training or testing.

  Returns:
      if is_training is True, it returns two dataset instances for both
      training and evaluation. Otherwise only one dataset instance for
      evaluation.
  Raises:
      ValueError: When `dataset_name` is unknown.
	'''

	if dataset_name != 'milan':
		raise ValueError('Name of dataset unknown %s' % dataset_name)
        
	test_input_param = {
		'tra_set': False,
		'paths': valid_data_paths,
		'minibatch_size': batch_size,
		'input_data_type': 'float32',
		'input_seq_length': in_seq_length,
		'output_seq_length': out_seq_length,
		'name': dataset_name + 'test iterator'
		}
                
	test_input_handle = milan.InputHandle(test_input_param)
	test_input_handle.begin(do_shuffle=False)
    
	if is_training:
		train_input_param = {
			'tra_set': True,
			'paths': train_data_paths,
			'minibatch_size': batch_size,
			'input_data_type': 'float32',
			'input_seq_length': in_seq_length,
			'output_seq_length': out_seq_length,
			'name': dataset_name + ' train iterator'
		}
    
		train_input_handle = milan.InputHandle(train_input_param)
		train_input_handle.begin(do_shuffle=True)
		return train_input_handle, test_input_handle
	else:
		return test_input_handle
