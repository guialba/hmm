# Algoritmo de Viterbi
def viterbi(self):

    import numpy as np
    import pandas as pd

    # Tamanho da amostra (n)
    n = len(self.amostra)

    # Inicializando dicionário para armazenar estados inferidos
    x_hat = {}

    # Inicializando algoritmo com estado inferido no instante n (última observação da amostra)
    x_hat[n] = pd.Series({estado:self.delta(t=n,
                                            estado=estado) 

                            for estado in np.arange(1,self.N+1)}).idxmax()

    # Iterando para t de n-1 a 1 (do "final para o início" da amostra)
    for t in np.arange(n-1,0,-1):
        
        # Obtendo estado inferido no instante t
        x_hat[t] = self.phi(t+1,
                            estado=x_hat[t+1]) 
        
    # Criando série com estados inferidos
    x_hat = pd.Series(x_hat).sort_index()
    
    return x_hat

# Algoritmo de inferência marginal dos regimes por observação
def inf_states_marginal(self):

    import numpy as np
    import pandas as pd
    
    # Tamanho da amostra (n)
    n = len(self.amostra)

    # Dicionário com estados inferidos
    dic_states = {ind:pd.Series({estado:self.gamma(k=ind,
                                                    estado=estado)
                                
                                # Iterando por estado (1 a N)
                                for estado in np.arange(1,self.N+1)}).idxmax()

                    # Iterando para cada valor na amostra (1 a n)              
                for ind in np.arange(1,n+1)}

    # Série com estados inferidos
    inf_states = pd.Series(dic_states)

    return inf_states