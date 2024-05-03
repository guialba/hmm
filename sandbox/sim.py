import numpy as np
import random


def generate_trajectory(T, random_state=None, **kargs):
    if random_state is not None:
        np.random.seed(random_state)

    def generate(T, total_hidden_sates, total_emission_sates, inital_prab, transition_prob, emission_prob):
        draw = lambda size, p: random.choices(range(size), p)[0]
        
        q = draw(total_hidden_sates, inital_prab)
        for t in range(1, T+1):
            yield draw(total_emission_sates, emission_prob[q]), q
            q = draw(total_hidden_sates, transition_prob[q])

    os = np.fromiter(generate(T, **kargs), dtype=np.dtype((int, 2))) 
    return os[:,0], os[:,1]