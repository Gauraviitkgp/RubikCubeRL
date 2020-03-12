import pycuber as pc 

#To declare a cube object
mycube = pc.Cube()

#Formula is for set of actions taken
my_formula = pc.Formula("R U R' U'")

#Reversing the actions
my_formula.reverse()

#A object for Algo
alg = pc.Formula()
#Random arrangement
random_alg = alg.random()
mycube(random_alg)


print(mycube)