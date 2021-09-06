import random
from functools import reduce

class Dsp_rule:
    def __init__(self, mother, father, new_individual, threshold):
        if new_individual:
            self.delta_ws = [random.choice([-1, 0, 1]) for i in range(32)]
        else:
            self.delta_ws = [mother.delta_ws[i] if random.getrandbits(1) else father.delta_ws[i] for i in range(32)]
        self.threshold = 1

    def get_delta_w(self, nats, reward):
        #print(str(nats) + str(reduce(lambda a,b:a*2+b,
        #                       list(map(lambda c: 1 if c > self.threshold else 0, nats)) + [0 if reward < 0 else 1])))
        return self.delta_ws[reduce(lambda a, b:a*2+b,
                               list(map(lambda c: 1 if c >= self.threshold else 0, nats)) + [0 if reward < 0 else 1])]
        #return 1


    def mutate(self):
        self.delta_ws = [random.choice([-1, 0, 1]) if random.random() < 0.05 else self.delta_ws[i] for i in range(32)]

if __name__ == "__main__":
    mother, father = (Dsp_rule(None, None, True, 0) for i in range(2))
    child = Dsp_rule(mother, father, False, 0)
    print("done")