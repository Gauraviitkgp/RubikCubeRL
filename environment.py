import numpy as np 
import pycuber as pc 
import utils

class RubikCube(object):
	"""docstring for env"""

	def __init__(self):
		self.cube = pc.Cube()

	def reset(self,max_scrambles):
		alg = pc.Formula()
		#Random arrangement
		random_alg = alg.random()

		if max_scrambles==None:
			pass
		else:
			random_alg = random_alg[:max_scrambles]

		self.cube(random_alg)

		return utils.flatten_1d_b(self.cube) #return states


	def reward(self, tcube):
		if utils.perc_solved_cube(self.cube)==1:
			return 1000,True

		return -1,False


	def step(self, action):
		lookup     = ["R", "L","D","U","B","F","R''", "L''","D''","U''","B''","F''"] #We are not accounting for half turns

		tcube = self.cube.copy()

		step_taken = pc.Formula(lookup[action])

		self.cube(step_taken) 

		rwd,over = self.reward(tcube)

		return utils.flatten_1d_b(self.cube),rwd,over

