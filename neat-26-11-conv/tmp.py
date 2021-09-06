import CPPN
import time
import random

cppn = CPPN.CPPN(None, None, 10, 1, new_individual = True, plasticity = True)
for i in range(100,150): 
	CPPN.mutate_individual(i, None, cppn)
time_1 = time.time() 
for i in range(50000): 
	cppn.run([0 for i in range(9)])
time_taken = time.time() - time_1
print(time_taken)
