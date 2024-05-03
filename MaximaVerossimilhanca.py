def verossimilhança(self,
                    Q,
                    mat_trans,
                    pi,
                    amostra):
    
    import numpy as np
    
    # Obtendo número de estados (dimensão N da matriz de transição)
    N = np.shape(mat_trans)[0]

    # Obtendo tamanho da amostra n
    n = len(amostra)
    
    # Inicializando soma
    soma = 0

    # Iterando para cada estado de 1 até N
    for estado_iter in np.arange(1,N+1):
        
        # Incrementando soma
        soma += self.alpha(k=n,
                           estado=estado_iter,
                           Q=Q,
                           mat_trans=mat_trans,
                           pi=pi)
                      
    return soma
    