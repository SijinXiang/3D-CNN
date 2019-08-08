import tensorflow as tf
import tensorflow.contrib.layers as layers

class conv3d(object):

	def __init__(self, 
                 conv_ndims, 
                 input_shape, 
                 output_channels, 
                 kernel_shape, 
                 name="cnn3d"):

		if conv_ndims != len(input_shape) - 1:
			raise ValueError("Invalid input_shape {} for conv_ndims={}.".format(input_shape, conv_ndims))     

		self._conv_ndims = conv_ndims
		self._input_shape = input_shape
		self._output_channels = output_channels
		self._kernel_shape = kernel_shape

		self._state_size = tf.TensorShape(self._input_shape[:-1] + [self._output_channels])
		self._output_size = tf.TensorShape(self._input_shape[:-1] + [self._output_channels])


	@property
	def state_size(self):
		return self._state_size

	@property
	def output_size(self):
		return self._output_size
    
    
	def __call__(self, inputs):
		return tf.layers.conv3d(inputs, self.output_channels, self.kernel_shape, padding="same")


class avgpool3d(object):

	def __init__(self, 
                 conv_ndims, 
                 input_shape, 
                 pool_size, 
                 strides,
                 name="avgpooling"):
    
		if conv_ndims != len(input_shape) - 1:
			raise ValueError("Invalid input_shape {} for conv_ndims={}.".format(input_shape, conv_ndims))     

		self._conv_ndims = conv_ndims
		self._input_shape = input_shape
		self._pool_size = pool_size
		self._strides = strides
        
	@property
	def state_size(self):
		return tf.TensorShape(self._input_shape)
    
	@property
	def output_size(self):
		return tf.TensorShape(self._input_shape)
    
	def __call__(self, inputs):
		return tf.layers.average_pooling3d(inputs, self.pool_size, self.strides, padding="same")    
    
    
 