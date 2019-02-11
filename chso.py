from math import ceil, exp
import numpy as np
from random import choice, shuffle
import warnings
import sys
import pandas as pd

data = pd.read_csv('dataset.csv')
data.drop('drop', axis=1, inplace=True)

def function(x):
	s=np.count_nonzero(x)
	return s

def update_relationship(agents, n, function, rn, hn, cn, mn):

	fitness = [(function(agents[i]), i) for i in range(n)]
	fitness.sort()

	chickens = [i[1] for i in fitness]
	roosters = chickens[:rn]
	hines = chickens[rn:-cn]
	chicks = chickens[-cn:]

	shuffle(hines)
	mothers = hines[:mn]

	for i in range(cn):
		chicks[i] = chicks[i], choice(mothers)

	for i in range(hn):
		hines[i] = hines[i], choice(roosters)

	return roosters, hines, chicks

def kill(agents, n, function, dimension):

	for i in range(n):

		fit = None

		try:
			fit = function(agents[i])
		except OverflowError:
			for j in range(dimension):
				agents[i][j] = round(agents[i][j])

		if str(fit) == 'nan':
			agents[i] = np.random.uniform( (1, dimension))

def __init__( n, function, dimension, iteration, G=5, FL=0.5):

	agents=np.array(data.iloc[:,0:dimension])

	rn = ceil(0.15 * n)
	hn = ceil(0.7 * n)
	cn = n - rn - hn
	mn = ceil(0.2 * n)

	pbest = agents
	#print(pbest)
	
	x = 0
	fitness = []
	for i in range(0,n):         
					
		fitness_each = function(agents[i,:]) 
		fitness.append(fitness_each)  
	
	pfit = fitness
	#print(pfit)

	Pbest = agents[np.array(fitness).argmax()]
	Gbest = Pbest

	for t in range(iteration):

		if t % G == 0:

			chickens = update_relationship(agents, n, function, rn, hn,
													  cn, mn)
			roosters, hines, chicks = chickens

		for i in roosters:

			k = choice(roosters)
			while k == i:
				k = choice(roosters)

			if pfit[i] <= pfit[k]:
				sigma = 1
			else:
				sigma = exp((pfit[k] - pfit[i]) / (abs(pfit[i]) + 0.01))

			agents[i] = pbest[i] * (1 + np.random.normal(0, sigma,
																	dimension))

		for i in hines:

			r1 = i[1]
			r2 = choice([choice(roosters), choice(hines)[0]])
			while r2 == r1:
				r2 = choice([choice(roosters), choice(hines)[0]])

			s1 = exp((pfit[i[0]] - pfit[r1]) / (abs(pfit[i[0]]) + 0.01))

			try:
				s2 = exp(pfit[r2] - pfit[i[0]])
			except OverflowError:
				s2 = float('inf')

			rand1 = np.random.random((1, dimension))[0]
			rand2 = np.random.random((1, dimension))[0]

			agents[i[0]] = pbest[i[0]] + s1 * rand1 * (
				pbest[r1] - pbest[i[0]]) + s2 * rand2 * (
				pbest[r2] - pbest[i[0]])

		for i in chicks:
			agents[i[0]] = pbest[i[0]] * FL * (pbest[i[1]] - pbest[i[0]])

		kill(agents, n, function, dimension)
		

		fitness = [function(x) for x in agents[x,:]]
        

		for i in range(n):
			if fitness[i] < pfit[i]:
				pfit[i] = fitness[i]
				pbest[i] = agents[i]

		Pbest = agents[np.array(fitness).argmin()]
		if function(Pbest) < function(Gbest):
			Gbest = Pbest

	#set_Gbest(Gbest)
	print(roosters)

__init__(94, function, 100, 200, G=5, FL=0.5)

