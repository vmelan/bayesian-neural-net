import tensorflow as tf 


def compute_loss(y_true, y_pred):
	""" Compute mean-squared error as a loss """
	with tf.name_scope("loss_mse"):
		loss = 1/2 * tf.reduce_mean(tf.square(y_true - y_pred))
		return loss 


def compute_loss_bnn(y_true, y_pred, log_sigma):
	""" Compute loss with uncertainty """ 
	with tf.name_scope("loss_bnn_mse"):
		data_term = 1/2 * tf.reduce_mean(tf.square(y_true - y_pred) 
										 * tf.exp(-log_sigma)) 
		uncertainty_term = 1/2 * tf.reduce_mean(log_sigma)
		loss_bnn = data_term + uncertainty_term 
		return loss_bnn 