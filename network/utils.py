import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
import math

def get_immediate_subdirectories(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def get_avg(value_list):
	d = 0.0
	sum_ = 0.0
	for v in value_list:
		sum_+=v
		d+=1.0
	return sum_/d

def load_log_file(filename):
	f = open(filename, 'r')
	lines = f.readlines()
	f.close()
	header = lines[0]
	header_list = header[:-1].split(",")
	output_dict = {}
	for key in header_list:
		output_dict[key] = []
	for line in lines[1:]:
		value_list = line[:-1].split(",")
		for i in range(len(value_list)):
			key = header_list[i]
			value_str = value_list[i]
			if value_str[0]=="[":
				value = np.fromstring(value_str[1:-1], sep=' ')
			else:
				value = float(value_str)
			output_dict[key].append(value)
	return output_dict

def get_min_index_value(data_dict, key='loss'):
	min_v = float('Inf')
	min_i = -1
	for i in range(len(data_dict[key])):
		if data_dict[key][i]<=min_v:
			min_v = data_dict[key][i]
			min_i = i
	return [min_i, min_v]

def get_k_min_index_value_trial(trials_dir, k=1, fname='test_log.txt', key='loss'):
	trial_dirs = get_immediate_subdirectories(trials_dir)
	best_mins = [float('Inf') for j in range(k)]
	best_i = [-1 for j in range(k)]
	best_trials = ['' for j in range(k)]
	for trial_dir in trial_dirs:
		filename = os.path.join(trial_dir, fname)
		data_dict = load_log_file(filename)
		min_i, min_v = get_min_index_value(data_dict)

		max_min_v = -1
		max_min_j = -1
		for j in range(k):
			if max_min_v<=best_mins[j]:
				max_min_v=best_mins[j]
				max_min_j=j

		if min_v<max_min_v:
			best_mins[max_min_j] = min_v
			best_i[max_min_j] = min_i
			best_trials[max_min_j] = trial_dir

	return [best_i, best_mins, best_trials]
