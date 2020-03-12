import numpy as np 
import pycuber as pc 
import utils

class RubikCube(object):
	"""docstring for env"""

	def __init__(self):
		self.cube      = pc.cube()

	def _get_states_(self,cub):


	def reset(self,max_scrambles):
		alg = pc.Formula()
		#Random arrangement
		random_alg = alg.random()

		if max_scrambles==None:
			pass
		else:
			random_alg = random_alg[:max_scrambles]

		self.cube(random_alg)

		return self.cube



	def estimated_position(self):
        """
        Get the estimated cubie of solved pair.
        """
        corner = {"D":self.cube["D"]["D"]}
        edge = {}
        for cubie in (corner, edge):
            for face in self.pair:
                cubie.update({face:self.cube[face][face]})
        return (Corner(**corner), Edge(**edge))

	def get_pair(self):
        colours = (
            self.cube[self.pair[0]].colour, 
            self.cube[self.pair[1]].colour, 
            self.cube["D"].colour
            )
        result_corner = self.cube.children.copy()
        for c in colours[:2]:
            result_corner &= self.cube.has_colour(c)
        result_edge = result_corner & self.cube.select_type("edge")
        result_corner &= self.cube.has_colour(colours[2])
        return (list(result_corner)[0], list(result_edge)[0])	

	def reward(self):
		if self.get_pair()==self.estimated_position:
			return 1000
		else 
			return -1


	def step(self, action):
		lookup     = ["R", "L","D","U","B","F","R2", "L2","D2","U2","B2","F2"]

		step_taken = pc.formula(lookup[action])

		self.cube(step_taken) 

		rwd = self.reward()

		return self.cube
