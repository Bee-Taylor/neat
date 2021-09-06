import CPPN
import random
random.seed(0)


database = CPPN.CPPN_Database(1,4,2,number_of_species=1, plasticity = True)
for i in range(10):
    database.mutate_population()
    print(i)
cppns = database.get_cppns()
cppn = cppns[-1]
for i in range(1,10):
    print(cppn.run([i,i,i,i]))