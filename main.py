import json 
import tensorflow as tf 
from utils import clean_events
from data_loader import DataLoader 

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


if __name__ == '__main__':
	main()