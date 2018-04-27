# network_sampling

## Dependencies
* Python 3.6
* Tensorflow
* Keras

## Description
This code allows for sampling in the posterior distribution of a neural network's parameters using Gibbs Sampling or Simulated Annealing.

## Usage
* /network contains an interface for manipulating neural networks as Keras models. It is currently designed to interface with the results produced by repositories "iter_enc_dec_training" or "simple_function_test", but can be easily changed to load from any .h5 model file.
* gibbs_sampling.py contains the general gibbs sampling algorithm, which can be run by importing the network sampling function
* simulated_annealing.py contains the general simulated annealing algorithm. The necessary "temperature" and "accept probability" functions must be defined by the user.
