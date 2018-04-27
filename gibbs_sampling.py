import math
import random
import numpy as np


'''
Gibbs sampler needs a way to sample from the conditional probabilities

input k, the number of samples we want to collect (int>0)

input "states" is a list of states with a predetermined order
the non-conditioned state index is given as an arg
	sample_from_cond_prob(states, index):
		return new states


input a lag variable for waiting until beginning sampling (int>=0)
input a skip variable for skipping some number of samples for each sample to be stored (int>0)
'''

def gibbs_sampling(initial_s, k, sample_from_cond_prob, lag=1000, skip=100):
	dimension = len(initial_s)
	samples = []
	current_s = initial_s
	global_counter = 0
	skip_counter = 0
	while len(samples)<k:
		global_counter+=1

		# get the next sample from conditional distributions on each state element
		new_state = current_s
		for i in range(dimension):
			new_state = sample_from_cond_prob(new_state, i)
		current_s = new_state

		if global_counter<lag:
			continue

		if skip_counter==skip:
			skip_counter=0
		if skip_counter==0:
			samples.append(current_s)

	return samples
