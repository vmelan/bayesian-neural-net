import tensorflow as tf

class SimpleNet():

	def __init__(self, config):
		self.config = config 


	def forward(self, input_data, name="SimpleNet"):
		""" Define forward pass of neural net architecture """
		fc_1 = self.fully_connected(input_data, neurons_in=13, neurons_out=50, 
								act=True, name="fc_1")
		fc_2 = self.fully_connected(fc_1, neurons_in=50, neurons_out=2, 
								act=False, name="fc_2")

		return fc_2 
  

	def fully_connected(self, input, neurons_in, neurons_out, act=True, name="fc"):
		""" Fully-connected layer """
		with tf.name_scope(name):
			# Use Glorot/Xavier initialization
			mean = 0.0 
			stddev = tf.sqrt(2 / tf.cast((neurons_in + neurons_out), dtype=tf.float32))
			W = tf.Variable(tf.random_normal([neurons_in, neurons_out], mean, stddev), name=name+"_W")
			b = tf.Variable(tf.constant(0.1, shape=[neurons_out]), name=name+"_b")

			if self.config['trainer']['disp_histogram']:
				tf.summary.histogram("weights", W)
				tf.summary.histogram("biases", b)

			# Compute output 
			output = tf.add(tf.matmul(input, W), b)

			if act: 
				# if True, use relu activation function
				return tf.nn.relu(output)
			else: 
				# else, do not use activation function 
				return output