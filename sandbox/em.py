import numpy as np
from viterbi import Viterbi

class ExpectationMaximization(Viterbi):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.M = kargs['total_emission_sates']

        self._xi = np.full([len(self.O), self.N, self.N], np.nan)
        self._gamma = np.full([len(self.O), self.N], np.nan)
        # self._pi = np.full([self.M], np.nan)
        # self._a = np.full([self.N, self.N], np.nan)
        # self._b = np.full([self.N, self.M], np.nan)

    def xi(self, t, a, b):
        if np.isnan(self._xi[t,a,b]):
            self._xi[t,a,b] = (self.alpha(t, a) * self.a[a][b] * self.b[b][self.O[t+1]] * self.beta(t+1, b)) / sum([self.alpha(t, i) * self.a[i][j] * self.b[j][self.O[t+1]] * self.beta(t+1, j) for i in range(self.N) for j in range(self.N)])
        
        return self._xi[t,a,b]

    # def gamma(self, t, i):
    #     if np.isnan(self._gamma[t,i]):
    #         self._gamma[t,i] = sum([self.xi(t, i, j) for j in range(self.N)]) 
    #     return self._gamma[t,i]
    
    def gamma(self, t, i):
        if np.isnan(self._gamma[t,i]):
            self._gamma[t,i] = (self.alpha(t, i) * self.beta(t, i)) / sum([self.alpha(t, j) * self.beta(t, j) for j in range(self.N)])
        return self._gamma[t,i]
    
    def estimate_pi(self, i):
        return self.gamma(0, i)

    def estimate_A(self, i,j):
        e = sum([self.xi(t, i,j) for t,_ in enumerate(self.O[:-1])]) / sum([self.gamma(t, i) for t,_ in enumerate(self.O[:-1])])
        return (e or 1e-5)
    def estimate_B(self, j,k):
        e = sum([self.gamma(t, j) for t,o in enumerate(self.O) if o==k]) / sum([self.gamma(t, j) for t,_ in enumerate(self.O)])
        return (e or 1e-5)
        # return sum([self.gamma(t, j) for t,o in enumerate(self.O[:-1]) if o==k]) / sum([self.gamma(t, j) for t,_ in enumerate(self.O[:-1])])