import numpy as np

class ForwardBackward:
    def __init__(self, trajectory, total_hidden_sates, inital_prab, transition_prob, emission_prob, **kargs):
        self.O = trajectory
        
        self.N = total_hidden_sates
        self.pi = inital_prab
        self.a = transition_prob
        self.b = emission_prob

        self._alpha = np.full([len(trajectory), self.N], np.nan)
        self._beta = np.full([len(trajectory),  self.N], np.nan)

    def p(self):
        return np.sum([self.alpha(len(self.O)-1, i) for i in range(self.N)])

    def alpha(self, t, i):
        if np.isnan(self._alpha[t, i]):
            self._alpha[t, i] = self.__alpha(t, i)
        return self._alpha[t, i]
    
    def beta(self, t, i):
        if np.isnan(self._beta[t, i]):
            self._beta[t, i] = self.__beta(t, i)
        return self._beta[t, i]
    

    def __alpha(self, t, i):
        if t == 0:
            return self.pi[i] * self.b[i][self.O[t]]
        else:
            return sum([self.alpha(t-1, j) * self.a[j][i] for j in range(self.N)]) * self.b[i][self.O[t]]
        
    def __beta(self, t, i):
        if t == (len(self.O)-1):
            return 1
        else:
            return sum([self.a[i][j] * self.b[j][self.O[t+1]]* self.beta(t+1, j) for j in range(self.N)])
        