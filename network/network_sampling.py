# from keras.models import Sequential
# from keras.layers import Dense, Activation

from keras import backend as k
import tensorflow as tf

import numpy as np
import math
import random
import time

import sys

try:
	from network import network_interface as NI
except:
	import network_interface as NI

# model = Sequential()
# model.add(Dense(2, input_dim=2, init='uniform', activation='relu'))
# model.add(Dense(1, init='uniform', activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# sess = tf.InteractiveSession()
#
# outputTensor = model.output #Or model.layers[index].output
#
# listOfVariableTensors = model.trainable_weights
#
# gradients = k.gradients(outputTensor, listOfVariableTensors)
#
# trainingExample = np.random.random((10000,2))
# sess.run(tf.global_variables_initializer())
# evaluated_gradients = sess.run(gradients, feed_dict={model.input:trainingExample} )
# print(evaluated_gradients)


def get_sampling_function(interface, inputs_, outputs_):
	output_vec = np.array([output_[0] for output_ in outputs_])
	outputTensor = interface.model.output
	listOfVariableTensors = interface.model.trainable_weights
	gradients = k.gradients(outputTensor, listOfVariableTensors)
	# states in this case are vectorized_weights
	def sample_conditional_distribution(states, index):
		interface.set_from_vectorized_weights(states)

		sess = tf.InteractiveSession()
		sess.run(tf.global_variables_initializer())
		grad_vec = []
		for input_ in inputs_:
			eval_grads = sess.run(gradients, feed_dict={interface.model.input:[input_]} )
			vectorized_grads = NI.vectorize_gradients(eval_grads)
			grad_vec.append(vectorized_grads[index])

		grad_vec = np.array(grad_vec)

		D = grad_vec.dot(grad_vec)
		N = grad_vec.dot(output_vec)

		mean = -N/D
		var = 1.0/(2.0*D)

		sample = random.gauss(mean, math.sqrt(var))
		states[index] = sample + states[index]
		return states

	return sample_conditional_distribution

if __name__=="__main__":
	model_trials_directory = "/scratch/richards/network_data/sweep_6/d4/train_3500"
	interface = NI.NetworkInterface()
	interface.get_model(model_trials_directory)
	state = NI.vectorize_weights(interface.cached_dict["weights"])
	print(state)

	inputs_ = np.array([[random.uniform(-1.0, 1.0)] for i in range(10)])
	outputs_ = np.array([[random.uniform(-20.0, 20.0)] for i in range(10)])

	sampling_function = get_sampling_function(interface, inputs_, outputs_)

	t0 = time.time()
	new_state = state
	for i in range(len(state)):
		new_state = sampling_function(new_state, i)
	print(new_state)
	t1 = time.time()
	total = t1-t0
	print(total)
