import tensorflow as tf

import pdb

class SimpleNet():

	def __init__(self, config):
		self.config = config 


	def forward(self, input_data, name="SimpleNet"):
		""" Define forward pass of neural net architecture """
		with tf.name_scope("SimpleNet"):
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
			stddev = tf.sqrt(2 / tf.cast((neurons_in + neurons_out), dtype=tf.float64)) 
			W = tf.Variable(tf.random_normal([neurons_in, neurons_out], mean, stddev, dtype=tf.float64), name=name+"_W")
			# b = tf.Variable(tf.constant(0.2, shape=[neurons_out]), name=name+"_b")
			b = tf.Variable(tf.constant(0.2, shape=[neurons_out], dtype=tf.float64), name=name+"_b")

			# pdb.set_trace()

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

	def train_optimizer(self, loss_value):
		""" Use an optimizer to train the network """
		with tf.name_scope("optimizer"):
			# Create optimizer 
			# optimizer= tf.train.AdamOptimizer(learning_rate=self.config['trainer']['learning_rate'], 
			# 	beta1=self.config['trainer'][''])
			optimizer = getattr(tf.train, self.config['optimizer']['optimizer_type'])(
				**self.config['optimizer']['optimizer_params'])
			# Initialize train step 
			train_step = optimizer.minimize(loss_value)

			return train_step 