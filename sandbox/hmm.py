import numpy as np
from em import ExpectationMaximization

class HiddenMarkovModel():
    def __init__(self, total_emission_sates, total_hidden_sates, random_state=None):
        self.total_emission_sates = total_emission_sates
        self.total_hidden_sates = total_hidden_sates
        self.model = None

        if random_state is not None:
            self.random_state = np.random.RandomState(random_state)
            # np.random.seed(random_state)
        else:
            self.random_state = np.random.mtrand._rand

    def get_model(self):
        return {
            'total_emission_sates': self.model.M,
            'total_hidden_sates': self.model.N, 
            'transition_prob': self.model.a,
            'emission_prob': self.model.b,
            'inital_prab': self.model.pi
        }
    
    def fit(self, trajectory, initial_model=None, max_iter=100, e=1e-3):        
        if initial_model is None:
            m = {
                'total_emission_sates': self.total_emission_sates,
                'total_hidden_sates': self.total_hidden_sates, 
                ## Probabilities
                'transition_prob': self.random_state.dirichlet(np.full(self.total_hidden_sates, 1/self.total_hidden_sates), size=self.total_hidden_sates),
                'emission_prob': self.random_state.dirichlet(np.full(self.total_emission_sates, 1/self.total_emission_sates), size=self.total_hidden_sates),
                'inital_prab': self.random_state.dirichlet(np.full(self.total_hidden_sates, 1/self.total_hidden_sates), size=1)[0,:]
            } 
        else: 
            m = initial_model

        log_prob = np.inf
        for i in range(max_iter):
            self.model = ExpectationMaximization(trajectory, **m)
            
            self.model.a = self.__get_estimate_for_A()
            self.model.b = self.__get_estimate_for_B()
            self.model.pi = self.__get_estimate_for_Pi()

            new_log_prob =  np.log(self.model.p())

            if (log_prob - new_log_prob) < e:
                return new_log_prob, i
            else:
                log_prob = new_log_prob
                m = {
                    'total_emission_sates': self.total_emission_sates,
                    'total_hidden_sates': self.total_hidden_sates, 
                    ## Probabilities
                    'transition_prob':self.model.a,
                    'emission_prob': self.model.b,
                    'inital_prab': self.model.pi
                } 

    def __get_estimate_for_A(self):
        return [[ round(self.model.estimate_A(i,j), 2) for j in range(self.model.N)] for i in range(self.model.N)]

    def __get_estimate_for_B(self):
        return [[ round(self.model.estimate_B(j,k), 2) for k in range(self.model.M)] for j in range(self.model.N)]

    def __get_estimate_for_Pi(self):
        return [ round(self.model.estimate_pi(i), 2) for i in range(self.model.N) ] 
    
    def __str__(self):
        if self.model is None:
            return 'no fitted model'
        else:
            return str(self.get_model())