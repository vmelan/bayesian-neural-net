import json 
import logging 
import tensorflow as tf 
from utils import clean_events, compute_rmse
from data_loader import DataLoader 
from model import SimpleNet
from losses import compute_loss, compute_loss_bnn

import pdb

def main():
	logger = logging.getLogger(__name__)

	## Load config file
	with open("config.json", "r") as f:
		config = json.load(f)

	## Cleaning TensorBoard events
	clean_events(config)

	## Load data
	data_loader = DataLoader(config)
	X_train, X_test, y_train, y_test = data_loader.get_data()

	## Create placeholders
	X = tf.placeholder(tf.float64, [None, 13])
	# y = tf.placeholder(tf.float32, [None, 2])
	y = tf.placeholder(tf.float64, [None])

	## Create model and outputs
	net = SimpleNet(config)
	net_output = net.forward(X)
	y_pred, log_sigma = net_output[..., 0], net_output[..., 1]
	# tf.Print(y_pred, [y_pred], "y_pred: ")
	tf.summary.scalar("mean_log_sigma", tf.reduce_mean(log_sigma))

	## Define metrics based on experiment
	# Loss 
	type_exp = '_'.join(config['exp_name'].split('_')[:2])
	if type_exp == 'vanilla_loss':
		loss = compute_loss(y_true=y, y_pred=y_pred)
	elif type_exp == 'loss_bnn':
		loss = compute_loss_bnn(y_true=y, y_pred=y_pred, log_sigma=log_sigma)
	# Root Mean Squared Error (RMSE)
	rmse = compute_rmse(y_true=y, y_pred=y_pred)

	## Define optimizer
	optimizer = net.train_optimizer(loss)

	## Merging all summaries
	merged_summary = tf.summary.merge_all()

	## Launching the execution graph for training 
	with tf.Session() as sess:
		# Initializing all variables
		sess.run(tf.global_variables_initializer())
		# Visualizing the Graph 
		writer = tf.summary.FileWriter("./tensorboard/" + config["exp_name"])
		writer.add_graph(sess.graph)

		for epoch in range(config["trainer"]["num_epochs"]):
			for batch in range(config["trainer"]["num_iter_per_epoch"]):
				# Yield next batch of data
				batch_X, batch_y = next(data_loader.get_next_batch(config["trainer"]["batch_size"]))
				# Run the optimizer 
				sess.run(optimizer, feed_dict={X: batch_X, y: batch_y})
				# Compute train loss and rmse 
				train_loss, train_rmse = sess.run([loss, rmse], feed_dict={X: batch_X, y: batch_y})


			# Evaluate test data 
			test_loss, test_rmse = sess.run([loss, rmse], feed_dict={X: batch_X, y: batch_y})


			if (epoch % config["trainer"]["writer_step"] == 0):
				# Run the merged summary and write it to disk 
				# logger.debug("here !")
				s = sess.run(merged_summary, feed_dict={X: batch_X, y: batch_y})
				writer.add_summary(s, (epoch + 1))


			if (epoch % config["trainer"]["display_step"] == 0): 
				print("Epoch: {:03d},".format(epoch + 1), \
      				   "train_loss= {:03f},".format(train_loss), \
      				   "train_rmse= {:03f},".format(train_rmse), \
      				   "test_loss= {:03f},".format(test_loss), \
      				   "test_rmse={:03f}".format(test_rmse)
      				   )

			# pdb.set_trace()

		print("Training complete")


	pdb.set_trace()

if __name__ == '__main__':
	# set logging config 
	logging.basicConfig(level=logging.DEBUG, format="line %(lineno)d: %(message)s")
	
	main()