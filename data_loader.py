import pandas as pd 
import numpy as np


class DataLoader():

	def __init__(self, config):
		self.config = config

		self.X, self.y = self.load_data()
		self.shuffle_data()
		self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()

	def load_data(self):
		""" Load dataset """
		dataframe = pd.read_excel(self.config["data_loader"]["data_path"])
		dataset = dataframe.values 
		X = dataset[:, 0:13]
		y = dataset[:, 13]
		return X, y

	def shuffle_data(self):
		""" Shuffle the data """
		if self.config["data_loader"]["shuffle"]:
			indices = np.arange(self.X.shape[0])
			np.random.shuffle(indices)
			self.X, self.y = self.X[indices], self.y[indices]

	def split_data(self):
		""" Split data into train/test """
		test_split = self.config["data_loader"]["test_split"]
		train_elem = int(self.X.shape[0] * (1 - test_split))
		X_train, y_train = self.X[:train_elem], self.y[:train_elem]
		X_test, y_test = self.X[train_elem:], self.y[train_elem:]

		return X_train, X_test, y_train, y_test

	def get_data(self):
		return self.X_train, self.X_test, self.y_train, self.y_test