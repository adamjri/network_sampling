import tensorflow as tf
from keras import *
from keras.layers import *
from keras import backend as k

import numpy as np

import sys

try:
	from network.utils import *
except:
	from utils import *

def vectorize_weights(weights):
	flattened_weights = []
	for w in weights:
		if len(w)>0:
			flattened_weights.append(w[0].flatten())
		if len(w)==2:
			flattened_weights.append(w[1].flatten())
	vectorized_weights = np.concatenate(flattened_weights)
	return vectorized_weights

def vectorize_gradients(grads):
	flattened_grads = []
	for g in grads:
		flattened_grads.append(g.flatten())
	vectorized_grads = np.concatenate(flattened_grads)
	return vectorized_grads

def devectorize_weights(data_dict, vectorized_weights):
	vector_pointer = 0
	weights = []
	for i, config in enumerate(data_dict["configs"]):
		shape = data_dict["shapes"][i]

		if i==0 and ("input" in config["name"]):
			weights.append([])
		else:
			weight = []
			w_len = shape[0][1]*shape[1][1]
			w_vec = vectorized_weights[vector_pointer:vector_pointer+w_len]
			weight.append(w_vec.reshape((shape[0][1], shape[1][1])))
			vector_pointer+=w_len
			if config["use_bias"]:
				b_len = shape[1][1]
				b_vec = vectorized_weights[vector_pointer:vector_pointer+b_len]
				weight.append(b_vec)
				vector_pointer+=b_len
			weights.append(weight)

	return weights

class NetworkInterface:
	def __init__(self):
		self.model = None

		self.original_dict = None
		self.in_original_mode = False

		self.cached_dict = None

	def get_model(self, model_trials_directory, log_filename="test_log.txt"):
		indexes, values, trial_dirs = get_k_min_index_value_trial(model_trials_directory)
		best_trial_dir = trial_dirs[0]
		best_epoch_index = indexes[0]
		data_dict = load_log_file(os.path.join(best_trial_dir, log_filename))
		best_epoch = data_dict["epoch"][best_epoch_index]
		epoch_dir = os.path.join(best_trial_dir, "Epoch_"+str(int(best_epoch)))
		model_file = os.path.join(epoch_dir, "model.h5")
		self.model = models.load_model(model_file)
		self.original_dict = self.get_config_dict()
		self.in_original_mode = True

	def get_config_dict(self):
		if self.in_original_mode:
			return self.original_dict

		elif self.cached_dict is None:
			configs = []
			weights = []
			shapes = []
			class_names = []
			for layer in self.model.layers:
				configs.append(layer.get_config())
				weight = layer.get_weights()
				weights.append(weight)
				shapes.append([layer.input_shape, layer.output_shape])
				class_names.append(layer.__class__.__name__)

			data_dict = {"configs": configs, "weights": weights,
						"shapes": shapes, "class_names":class_names}

			self.cached_dict = data_dict

		return self.cached_dict

	def set_from_dict(self, data_dict):
		self.in_original_mode = False
		self.cached_dict = data_dict
		for i, layer in enumerate(self.model.layers):
			self.model.layers[i].set_weights(data_dict["weights"][i])

	def restore_to_original(self):
		self.set_from_dict(self.original_dict)
		self.in_original_mode = True

	def set_from_vectorized_weights(self, vectorized_weights):
		new_dict = self.cached_dict
		new_weights = devectorize_weights(self.cached_dict, vectorized_weights)
		new_dict["weights"] = new_weights
		self.set_from_dict(new_dict)
