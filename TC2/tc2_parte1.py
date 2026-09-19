import numpy as np
import math

def eliminacao_gauss(A, b, eps=1e-8):
    A = np.array(A, dtype=float).copy()
    b = np.array(b, dtype=float).copy()
    n = len(b)
    flops = 0

    # 1. Eliminação Em Frente (Forward Elimination) com Pivotamento Parcial
    for i in range(n):
        p = i + np.argmax(np.abs(A[i:n, i]))
        if p != i:
            A[[i, p]] = A[[p, i]]
            b[[i, p]] = b[[p, i]]
            
        if abs(A[i, i]) < eps:
            raise ValueError(f"O sistema é singular ou quase singular (pivô nulo na coluna {i}).")

        for j in range(i + 1, n):
            m = A[j, i] / A[i, i]
            flops += 1  # 1 divisão
            
            # Operação na linha A[j] completa (n multiplicações + n subtrações)
            A[j] = A[j] - m * A[i]
            flops += 2 * n  
            
            # Operação em b[j] (1 multiplicação + 1 subtração)
            b[j] = b[j] - m * b[i]
            flops += 2  

    # 2. Substituição Regressiva (Back-substitution)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        num_elem = n - 1 - i
        s = np.dot(A[i, i + 1:], x[i + 1:])
        
        if num_elem > 0:
            flops += 2 * num_elem  # produto interno (multiplicações e somas)
            
        x[i] = (b[i] - s) / A[i, i]
        flops += 2  # 1 subtração + 1 divisão

    info_custo = {
        "flops": flops
    }

    return x, info_custo

def criterio_das_linhas(A):
    n = len(A)
    a = 0
    for i in range(n):
        if abs(A[i, i]) < 1e-15:
            # Se houver zero na diagonal, o critério falha/não é aplicável diretamente
            return False
        a_ant = a
        s = 0
        for j in range(n):
            if j != i:
                s+=abs(A[i, j])
        a = s/abs(A[i,i])
        if(a_ant > a):
            a = a_ant
    if(a < 1):
        return True
    return False

def gauss_jacobi(A, b, x0, eps=1e-8, max_iter=200):
    
    A = np.array(A, dtype=float).copy()
    b = np.array(b, dtype=float).copy()
    n = len(b)
    x = np.array(x0, dtype=float).copy()
    x_new = np.array(x0, dtype=float).copy()
    flops = 0

    for k in range(max_iter):
        for i in range(n):
            if abs(A[i, i]) < 1e-15:
                raise ZeroDivisionError(f"Elemento diagonal A[{i},{i}] é zero ou insignificante.")
            
            s = 0.0
            for j in range(n):
                if j != i:
                    s += A[i, j] * x[j]
                    flops += 2  # 1 multiplicação + 1 adição

            x_new[i] = (b[i] - s) / A[i, i]
            flops += 2  # 1 subtração + 1 divisão

        # Cálculo do erro relativo vetorial
        erro = np.max(np.abs(x_new - x) / np.maximum(np.abs(x_new), 1e-15))
        flops += 2 * n  # n subtrações (x_new - x) + n divisões

        x = x_new.copy()

        if erro < eps:
            info_custo = {
                "iteracoes": k + 1,
                "flops": flops
            }
            return x, info_custo

    raise RuntimeError(f"O método não convergiu após {max_iter} iterações.")

def criterio_sassenfeld(A):
    n = len(A)
    beta = np.empty(n)
    for i in range(n):
        if abs(A[i, i]) < 1e-15:
            # Se houver zero na diagonal, o critério falha/não é aplicável diretamente
            return False
        s = 0
        for j in range(n):
            if j <= i-1:
                s+=(abs(A[i, j]) * beta[j])
            elif j >= i+1:
                s+=abs(A[i, j])
        a = s/abs(A[i,i])
        beta[i] = a
    return np.max(beta) < 1

def gauss_seidel(A, b, x0, eps=1e-8, max_iter=200):
    
    A = np.array(A, dtype=float).copy()
    b = np.array(b, dtype=float).copy()
    n = len(b)
    x = np.array(x0, dtype=float).copy()
    flops = 0

    for k in range(max_iter):
        x_ant = x.copy()
        
        for i in range(n):
            if abs(A[i, i]) < 1e-15:
                raise ZeroDivisionError(f"Elemento diagonal A[{i},{i}] é zero ou insignificante.")
            
            s = 0.0
            for j in range(n):
                if j != i:
                    s += A[i, j] * x[j]  # Usa o x[j] atualizado em tempo real para j < i
                    flops += 2  # 1 multiplicação + 1 adição

            x[i] = (b[i] - s) / A[i, i]
            flops += 2  # 1 subtração + 1 divisão

        # Cálculo do erro relativo vetorial
        erro = np.max(np.abs(x - x_ant) / np.maximum(np.abs(x), 1e-15))
        flops += 2 * n  # n subtrações (x - x_ant) + n divisões

        if erro < eps:
            info_custo = {
                "iteracoes": k + 1,
                "flops": flops
            }
            return x, info_custo

    raise RuntimeError(f"O método não convergiu após {max_iter} iterações.")

def jacobiana(F, X, F_X, h=1e-8):
    n = len(X)
    J = np.empty([F_X.shape[0], n])
    for i in range(n):
        # Passo adaptativo para a variável i
        h_i = h * max(abs(X[i]), 1.0)
        
        X_pert = X.copy()
        X_pert[i] += h_i
        
        J[:, i] = (F(X_pert).ravel() - F_X) / h_i
    return J

def decomposicao_LU(A, b, eps=1e-15):
    """Resolve Ax = b"""
    U = np.array(A, dtype=float).copy()
    b = np.array(b, dtype=float).copy()
    L = np.identity(A.shape[0])
    n = U.shape[0]
    b_vec = np.array(b, dtype=float).ravel()
    P = np.arange(0, n)
    #
    for i in range(n):
        p = i + np.argmax(np.abs(U[i:n, i]))
        if p != i:
            U[[i, p]] = U[[p, i]]
            L[[i, p], :i] = L[[p, i], :i]
            P[[i, p]] = P[[p, i]]

        if abs(U[i, i]) < eps:
            raise ValueError(f"Pivô nulo ou muito próximo de zero na posição ({i},{i}).")
        for j in range(i+1, n):
            m = U[j, i]/U[i,i]
            L[j, i] = m
            U[j] = U[j] - m * U[i]
    
    y = np.zeros(n)
    for i in range(n):
        s = 0
        for k in range(0, i):
            s += L[i, k] * y[k]
        y[i] = b_vec[P[i]] - s
        
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        s = 0
        for k in range(i + 1, n):
            s += U[i, k] * x[k]
        x[i] = (y[i] -   s)/U[i, i]
    return(x)

def newton_raphson_multi(F, X0, eps=1e-8, max_iters=200):
    X0 = np.array(X0, dtype=float).ravel()
    for i in range(max_iters):
        F_X = F(X0).ravel()
        if(np.max(abs(F_X)) < eps):
            return X0
        J_X = jacobiana(F, X0, F_X)
        try:
            s = decomposicao_LU(J_X, F_X * -1)
        except ValueError as e:
            raise RuntimeError(
                f"O método de Newton falhou na iteração {i}: a matriz Jacobiana se tornou singular ({e})."
            )
        X0 = X0 + s
        if(np.isnan(X0).any() or np.isinf(X0).any()):
            raise(ValueError(f"O método divergiu levando X ao infinito"))
        if (np.max(abs(s)) < eps):
            return X0
    raise(RuntimeError(f"O método não convergiu após {max_iters} iterações"))

def problema_2():
    print('\n\n\nPROBLEMA 2\n')
    n = 4

    R1 = R2 = R3 = 100

    R4 = R5 = 150

    R6 = R7 = R8 = 200

    V1 = 10

    V2 = 12

    A = np.array([
        [R1 + R2 + R3, -R2, 0, -R4],
        [-R2, R2 + R3 + R5, -R5, 0],
        [0, -R5, R5 + R7 + R8, -R7],
        [-R4, 0, -R7, R4 + R6 + R7]
    ])

    b = np.array([-V1, V2, 0, 0])

    print(f'R1 = {R1}, R2 = {R2}, R3 = {R3}, R4 = {R4}, R5 = {R5}, R6 = {R6}, R7 = {R7}, R8 = {R8}')

    print(f'A = \n{A}')

    print(f'V1 = {V1}, V2 = {V2}')

    print(f'b = {b}')

    print("""\nComo a matriz é densa (a maioria dos elementos nao sao zero ou nulo) e possui um pequeno (n = 4) pela regra de bolso (matriz densa e n até 
    alguns milhares usamos eliminação de gauss com pivoteamento. Matriz esparsa de grande porte e diagonalmente dominante usamos Gauss-Seidel.) o melhor
    método de resolução seria por eliminação de Gauss com pivoteamento. Entretanto executaremos os três métodos para verificar o custo de cada um e 
    a confiabilidade da solução encontrada.\n""")

    g, custo_g = eliminacao_gauss(A, b)

    print('Eliminação de Gauss')
    print(f'x = {g}, custo em flops (floating point operations): {custo_g['flops']}\n')

    print('Gauss-Jacobi')
    print(f'Critério das linhas para a matriz A: {criterio_das_linhas(A)}, logo, com certeza o método converge')
    gj, custo_gj = gauss_jacobi(A, b, b / np.diag(A))
    print(f'x = {gj}, quantidade de iterações: {custo_gj['iteracoes']}, custo em flops (floating point operations): {custo_gj['flops']}\n')

    print('Gauss-Seidel')
    print(f'Critério das linhas para a matriz A: {criterio_das_linhas(A)}, logo, com certeza o método converge')
    print(f'Critério de Sassenfeld para a matriz A: {criterio_sassenfeld(A)}, logo, com certeza o método converge')
    gs, custo_gs = gauss_seidel(A, b, b / np.diag(A))
    print(f'x = {gs}, quantidade de iterações: {custo_gs['iteracoes']}, custo em flops (floating point operations): {custo_gs['flops']}')

    print("""
    A resolução alcançada pelos três métodos é suficientemente satisfatória, entretanto, percebemos que o método de eliminação de Gauss gera menos operações
    com pontos flutuantes sendo assim mais eficiente.
    """)

def problema_4():
    print('\n\n\nPROBLEMA 4\n')
    F = lambda v: np.array([
        6*v[0] - 2*v[1] + np.exp(v[2]) - 2,
        np.sin(v[0]) - v[1] + v[2],
        np.sin(v[0]) + 2*v[1] + 3*v[2] - 1
    ])

    print('Como ele quer saber uma solucao proxima a origem do sistema utilizaremos ela como chute inicial e aplicaremos o algoritmo de Newton-Raphson')
    X0 = np.array([0,0,0])

    print(f'Resultado do metodo = {newton_raphson_multi(F, X0, eps=1e-5)}')

if __name__ == "__main__":
    problema_2()
    problema_4()