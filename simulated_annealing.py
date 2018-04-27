import math
import random
import numpy as np

'''
simulated annealing requires a neighborhood function that randomly produces a neighber of a state s
within some step size determined by the temperature T:
neighborhood function should try to pick states with close to the same energy value
should have randomness: i.e. collect a neighborhood and randomly sample 1 neighbor
try to avoid "deep basins", i.e. collections of neighboring states that all have much lower
energy than the surrounding states
	neighbor(state, T)
		return new state

an energy function:
smaller energy means more likely
	energy(state)
		return float

an acceptance probability function P(e=energy(s), e'=energy(s'), T):
computes probability of transitioning from state s to state s' given T
probability must be positive even when e'>e
when T->0, probability->0 when e'>e
When T=0, P should degenerate to the greedy algorithm (1 when e'<e, 0 when e'>=e)
Some implementations have P(e, e')=1 when e'<e for all T, but this is not necessary
P typically chosen so it decreases when e'-e increases, but also not necessary
T essentially modulates the sensitivity of P to uphill moves
	accept_probability(e, e', T):
		return float in range [0, 1]


a k_max (int)

a temperature function that takes k/k_max:
temperature should decrease with increasing k and should be 0 when k=k_max
temperature scheduling should be determined empirically
	temperature(k/k_max)
		return float T

'''

def accept_probability_default(e_start, e_next, T):
	if e_next<e_start:
		return 1.0
	else:
		return math.exp((e_start-e_next)/T)
	

def simulated_annealing(initial_s, neighbor, energy, accept_probability, k_max, temperature):
	current_s = initial_s
	for k in range(k_max):
		T = temperature(float(k)/float(k_max))
		new_s = neighbor(current_s, T)
		if accept_probability(energy(current_s), energy(new_s), T) >= random.uniform(0.0,1.0):
			current_s = new_s
	return current_s
