from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from src.cnn3d_cell import conv3d, avgpool3d
import tensorflow as tf

def cnn_net(images, num_layers, num_hidden, configs):

	cnn3d_layer, pooling_layer, hidden = [], [], []
	shape = images.get_shape().as_list()
	batch_size = shape[0]
	ims_depth = shape[1]
	ims_height = shape[2]
	ims_width = shape[3]
	output_channels = shape[-1]
	input_length = configs.input_seq_length
	kernel_size = configs.kernel_size
	pool_size = [int(x) for x in configs.pool_size.split(',')]
	strides = configs.strides

	for i in range(num_layers):
		if i == 0:
			num_hidden_in = output_channels
		else:
			num_hidden_in = num_hidden[i - 1]

		hidden_out = num_hidden[i]
		new_cnn = conv3d(
                name = 'cnn3d'+str(i),
                input_shape = [input_length, ims_height, ims_width, num_hidden_in],
                output_channels = hidden_out,
                kernel_size = kernel_size)
		cnn3d_layer.append(new_cnn)
		new_pooling = avgpool3d(
                    name = 'avgpooling'+str(i),
                    input_shape = [input_length, ims_height, ims_width, hidden_out],
                    pool_size = pool_size,
                    strides = strides)
		pooling_layer.append(new_pooling)
		hidden.append(tf.zeros([input_length, ims_height, ims_width, hidden_out]))
        
	num_hidden_in = num_hidden[-1]
	new_cnn = conv3d(
                name = 'cnn3d'+str(num_layers),
                input_shape = [input_length, ims_height, ims_width, num_hidden_in],
                output_channels = output_channels,
                kernel_size = kernel_size)
	cnn3d_layer.append(new_cnn)  
	new_pooling = avgpool3d(
                    name = 'avgpooling'+str(num_layers),
                    input_shape = [input_length, ims_height, ims_width, output_channels],
                    pool_size = pool_size,
                    strides = strides)
	pooling_layer.append(new_pooling)

    
    
	with tf.variable_scope('generator'):
		reuse = False

		with tf.variable_scope('cnn3d', reuse=reuse):
			for l in range(num_layers):
				if l == 0:
					input_frm = images[:,:input_length]
				else:
					input_frm = hidden[l-1]
				hidden[l] = cnn3d_layer[l](input_frm)
				hidden[l] = pooling_layer[l](hidden[l])
			gen_images = cnn3d_layer[-1](hidden[-1])
			gen_images = pooling_layer[-1](gen_images)

	loss = tf.nn.l2_loss(gen_images - images[:, input_length:])

	loss += tf.reduce_sum(tf.abs(gen_images - images[:, input_length:]))

	return [gen_images, loss]


