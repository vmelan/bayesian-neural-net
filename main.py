import json 
import logging 
import tensorflow as tf 
from utils import clean_events
from data_loader import DataLoader 
from model import SimpleNet
from losses import compute_loss, compute_loss_bnn

import pdb

def main():
	## Load config file
	with open("config.json", "r") as f:
		config = json.load(f)

	## Cleaning TensorBoard events
	clean_events(config)

	## Load data
	data_loader = DataLoader(config)
	X_train, X_test, y_train, y_test = data_loader.get_data()

	## Create placeholders
	X = tf.placeholder(tf.float32, [None, 13])
	y = tf.placeholder(tf.float32, [None, 2])

	## Create model and outputs
	net = SimpleNet(config)
	net_output = net.forward(X)
	y_pred, log_sigma = net_output[..., 0], net_output[..., 1]
	
	## Define metrics based on experiment 
	type_exp = '_'.join(config['exp_name'].split('_')[:2])
	if type_exp == 'vanilla_loss':
		loss = compute_loss(y_true=y, y_pred=y_pred)
	elif type_exp == 'loss_bnn':
		loss = compute_loss_bnn(y_true=y, y_pred=y_pred, log_sigma=log_sigma)

	## Define optimizer
	optimizer = net.train_optimizer(loss)

	pdb.set_trace()

if __name__ == '__main__':
	# set logging config 
	logging.basicConfig(level=logging.INFO, format="line %(lineno)d: %(message)s")
	
	main()