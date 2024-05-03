import numpy as np
from fb import ForwardBackward

class Viterbi(ForwardBackward):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

        self._delta = np.full([len(self.O), self.N], np.nan)
        self._psi = np.full([len(self.O), self.N], np.nan)
        
    def get_value(self, t, i):
        if np.isnan(self._delta[t, i]):
            self._psi[t, i], self._delta[t, i] = self.__get_value(t, i)
        return self._psi[t, i], self._delta[t, i]

    def __get_value(self, t, i):
        if t == 0:
            return 0, self.pi[i] * self.b[i][self.O[t]]
        else:
            arr = np.array([self.get_value(t-1, j)[1] * self.a[j][i] for j in range(self.N)])
            return np.argmax(arr), np.max(arr)*self.b[i][self.O[t]]
    
    def p_star(self):
        T = len(self.O)-1
        if np.isnan(self._delta[T]).any():
            for j in range(self.N):
                self.get_value(T, j)
        return np.max(self._delta[T])
    
    def q_star(self):
        # return np.array([self.__q_star(t) for t,_ in list(enumerate(self.O))[::-1]])
        return np.array([self.__q_star(t) for t,_ in list(enumerate(self.O))])
    
    def __q_star(self, t):
        T = len(self.O)-1
        if np.isnan(self._delta[T]).any():
            for j in range(self.N):
                self.get_value(T, j)
        if t == T:
            return np.argmax(self._delta[T])
        else:
            return self.get_value(t+1, int(self.__q_star(t+1)))[0]
        
