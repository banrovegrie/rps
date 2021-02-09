from numpy import random

random.seed = 67280421310721

def random_agent(observation, configuration):
    return int(random.randint(3))
