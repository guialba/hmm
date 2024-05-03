from sim import generate_trajectory
from hmm import HiddenMarkovModel
import numpy as np


m = {
    'total_emission_sates': 5,
    'total_hidden_sates': 4, 
    'transition_prob': [
        [.1, .1, .4, .4],
        [.3, .5, .1, .1],
        [.1, .6, .1, .1],
        [.25, .25, .25, .25]
    ],
    'emission_prob': [
        [.2, .3, .3, .1, .1],
        [.1, .1, .1, .1, .6],
        [.1, .1, .1, .4, .3],
        [.1, .1, .5, .1, .2],
    ],
    'inital_prab': [.2, .3, .3, .2]
}

os = '101000100111101010101100100100101'
O = [int(i) for i in os]
# O, S = generate_trajectory(50, **m)

model = None
ll = -np.inf

for n in range(100):
    hmm = HiddenMarkovModel(5, 4)
    hmm.fit(O)
    l = hmm.model.p()
    if l > ll:
        model = hmm
        ll = l

print("Esti", model)