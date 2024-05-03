# Forward variable
def alpha(self,
          k,
          estado,
          Q,
          mat_trans,
          pi):

    import numpy as np

    # Obtendo número de estados (dimensão N da matriz de transição)
    N = np.shape(self.mat_trans)[0]

    # Caso base da recursão (alpha_1(state))
    if k == 1:

        return self.Q[estado](self.amostra[1]) * self.pi(estado)
    
    # Regra geral
    else:
        
        # Inicializando soma
        soma = 0

        # Iterando para cada estado de 1 a N (0 a N-1)
        for estado_iter in np.arange(1,N+1):

            # Definindo forward variable para (n-1) considerando o estado da iteração atual
            alpha_ant = self.alpha(k=k-1,
                                   estado=estado_iter,
                                   Q=Q,
                                   mat_trans=mat_trans,
                                   pi=pi)
            
            # Definindo parcela da soma
            parcela = alpha_ant * Q[estado](self.amostra[k-1]) * mat_trans[estado_iter-1,estado-1]

            # Acrescentando parcela na soma
            soma += parcela

        return soma

# Backward variable
def beta(self,
            k,
            estado):
    
    import numpy as np

    # Caso base da recursão (beta_k(estado))
    if k == len(self.amostra):
        return 1
    
    # Caso geral
    else:

        # Inicializa soma
        soma = 0

        # Iterando sobre os estados (1 a N)
        for estado_iter in np.arange(1,self.N+1):

            # Definindo parcela atual da soma
            parcela = self.mat_trans[(estado-1),(estado_iter-1)] * self.Q[estado_iter](self.amostra[k]) * self.beta(k=k+1,
                                                                                                                    estado=estado_iter)
            
            # Incrementando soma
            soma += parcela
        
        return soma

# Função gamma
def gamma(self,
          k,
          estado):

    import numpy as np
    import pandas as pd

    # Obtendo valores das alpha e beta variables
    alphas = pd.Series({estado_iter:self.alpha(k=k,
                                               estado=estado_iter,
                                               Q=self.Q,
                                               mat_trans=self.mat_trans,
                                               pi=self.pi) for estado_iter in np.arange(1,self.N+1)})
    
    betas = pd.Series({estado_iter:self.beta(k=k,
                                             estado=estado_iter) for estado_iter in np.arange(1,self.N+1)})
    
    # Numerador
    num = alphas[estado] * betas[estado]

    # Denominador
    den = (alphas * betas).sum()

    return num/den

# Função delta
def delta(self,
            t,
            estado):

    import numpy as np
    import pandas as pd
    
    # Caso base da recursão (t = 1)
    if t == 1:
        
        return self.pi(estado) * self.Q[estado](self.amostra[t-1])
    
    # Caso geral (t > 1)
    else:
        
        # Computando delta_(t-1) para todos os estados i de 1 a M
        deltas_ant = pd.Series({i:self.delta(t-1,
                                                estado=i)
                                        
                                for i in np.arange(1,self.N+1)})
        
        # Computando probabilidades de transição partindo do estado i
        probs_i = pd.Series(self.mat_trans[:,(estado-1)])
        probs_i.index = probs_i.index + 1

        # Produto entre probabilidades de transição e delta para cada estado i
        delta_x_probs = deltas_ant * probs_i
        
        # Computando valor máximo do produto definindo anteriormente
        max_delta_x_probs = delta_x_probs.idxmax()

        return max_delta_x_probs * self.Q[estado](self.amostra[t-1])

# Função phi
def phi(self,
        t,
        estado):

    import numpy as np
    import pandas as pd
    
    # Deltas 
    deltas_ant = pd.Series({i:self.delta(t-1,
                                            estado=i)
                        
                            for i in np.arange(1,self.N+1)})
    
    # Computando probabilidades de transição partindo do estado i para i = 1,...,M (coluna estado)
    probs_i = pd.Series(self.mat_trans[:,(estado-1)])
    probs_i.index = probs_i.index + 1

    # Produto entre probabilidades de transição e delta para cada estado i
    delta_x_probs = deltas_ant * probs_i

    # Obtendo estado que maximiza produto anterior
    return delta_x_probs.idxmax()