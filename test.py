import pycuber as pc 

#https://github.com/adrianliaw/PyCuber/blob/version2/examples/sample_program.py
#To declare a cube object
mycube = pc.Cube()

k=["R","S"]
#Formula is for set of actions taken
my_formula = pc.Formula(k[0])

mycube(my_formula)
print(mycube)
#Reversing the actions
my_formula.reverse()

#A object for Algo
alg = pc.Formula()
#Random arrangement
random_alg = alg.random()

random_alg = random_alg[:10]

print(random_alg)

mycube(random_alg)


print(mycube)