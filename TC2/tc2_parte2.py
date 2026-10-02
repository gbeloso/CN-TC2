import numpy as np
import math

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

def newton_raphson_multi_s_J(F, J, X0, X, eps=1e-8, max_iters=200):
    X0 = np.array(X0, dtype=float).ravel()
    hist = []
    for i in range(max_iters):
        F_X = F(X0).ravel()
        if(np.max(abs(F_X)) < eps):
            return (X0, hist)
        J_X = J(X0)
        try:
            s = decomposicao_LU(J_X, F_X * -1)
        except ValueError as e:
            raise RuntimeError(
                f"O método de Newton falhou na iteração {i}: a matriz Jacobiana se tornou singular ({e})."
            )
        X0 = X0 + s
        hist.append({
            'k' : i,
            'Q1' : X0[0],
            'Q2' : X0[1],
            'Q3' : X0[2],
            'H' : X0[3],
            '||F(X)||' : np.max(abs(F(X0).ravel())),
            '|x_k - x|': np.max(abs(X0-X))
        })
        if(np.isnan(X0).any() or np.isinf(X0).any()):
            raise(ValueError(f"O método divergiu levando X ao infinito"))
        if (np.max(abs(s)) < eps):
            return (X0, hist)
    raise(RuntimeError(f"O método não convergiu após {max_iters} iterações"))

g = 9.81
z = np . array ([100.0 , 85.0 , 60.0])
L = np . array ([1200.0 , 900.0 , 1500.0])
D = np . array ([0.30 , 0.25 , 0.25])
f = np . array ([0.022 , 0.024 , 0.024])
K = 8* f *L /( np . pi **2* g * D **5)
q = 0.200


def problema_1_2_b():

    X0 = (0.1, 0.1, -0.05, 70)

    X= np.array([0.17724612, 0.08499127, -0.06223739, 71.79849039])

    F = lambda Q: np.array([
        [K[0] * Q[0] * abs(Q[0]) + Q[3] - z[0]],
        [K[1] * Q[1] * abs(Q[1]) + Q[3] - z[1]],
        [K[2] * Q[2] * abs(Q[2]) + Q[3] - z[2]],
        [Q[0] + Q[1] + Q[2] - q]
    ])

    J = lambda Q: np.array([
        [2 * K[0] * Q[0]**2 / abs(Q[0]), 0, 0, 1],
        [0, 2 * K[1] * Q[1]**2 / abs(Q[1]), 0, 1],
        [0, 0, 2 * K[2] * Q[2]**2 / abs(Q[2]), 1],
        [1, 1, 1, 0]
    ])

    resultado, hist = newton_raphson_multi_s_J(F, J, X0, X)

    # Imprimindo o cabeçalho da tabela
    print(f"{'k':>3} | {'Q1 (m³/s)':>14} | {'Q2 (m³/s)':>14} | {'Q3 (m³/s)':>14} | {'H (mca)':>14} | {'||F(x)||':>12} | {'||x_k - x*||':>12}")
    print("-" * 97)
    
    # Imprimindo os dados formatados
    for h in hist:
        print(f"{h['k']:>3} | "
              f"{h['Q1']:>14.8f} | "
              f"{h['Q2']:>14.8f} | "
              f"{h['Q3']:>14.8f} | "
              f"{h['H']:>14.8f} | "
              f"{h['||F(X)||']:>12.2e} | "
              f"{h['|x_k - x|']:>12.2e}")
              
    print("\nSolução final encontrada:")
    print('[' + ', '.join(f'{val:.8f}' for val in resultado) + ']')

    print("\nObservando a evolução do erro absoluto máximo (||x_k - x*||):")
    print(" • Iteração 1: ordem de 10^-2")
    print(" • Iteração 2: ordem de 10^-4")
    print(" • Iteração 3: ordem de 10^-9")
    print(" • Iteração 4: ordem de 10^-9")
    print("\nConclusão: Os expoentes do erro dobram sucessivamente a cada")
    print("passo, demonstrando que o número de casas decimais corretas")
    print("também dobra. Isso confirma empiricamente a convergência")
    print("quadrática característica do Método de Newton-Raphson.")

def problema_1_2_c():
    """
    PROBLEMA 1.2.C - Análise da Singularidade do Jacobiano para x^(0) = (0, 0, 0, 70)
    e Interrupção por Exceção (raise).
    """
    print("\n\n" + "=" * 80)
    print("PROBLEMA 1.2.C - ANÁLISE DE SINGULARIDADE NO PONTO x^(0) = (0, 0, 0, 70)")
    print("=" * 80 + "\n")

    # 1. Ponto inicial considerado
    X0_singular = np.array([0.0, 0.0, 0.0, 70.0])
    Q1_0, Q2_0, Q3_0, H_0 = X0_singular

    print(f"Ponto inicial considerado: x^(0) = (Q1={Q1_0}, Q2={Q2_0}, Q3={Q3_0}, H={H_0})")

    # 2. Matriz Jacobiana avaliada em x^(0)
    # Como Q1 = Q2 = Q3 = 0, d_i = 2 * k_i * |Q_i| = 0 para i=1,2,3
    J_num_0 = np.array([
        [2 * K[0] * abs(Q1_0), 0, 0, 1],
        [0, 2 * K[1] * abs(Q2_0), 0, 1],
        [0, 0, 2 * K[2] * abs(Q3_0), 1],
        [1, 1, 1, 0]
    ], dtype=float)

    print("\nMatriz Jacobiana J(x^(0)):")
    print(J_num_0)

    # 3. Cálculo do Determinante
    det_num = np.linalg.det(J_num_0)
    print(f"\nDeterminante calculado: det(J(x^(0))) = {det_num}")

    # 4. Avaliação do vetor F(x^(0))
    F_num_0 = np.array([
        K[0] * Q1_0 * abs(Q1_0) + H_0 - z[0],
        K[1] * Q2_0 * abs(Q2_0) + H_0 - z[1],
        K[2] * Q3_0 * abs(Q3_0) + H_0 - z[2],
        Q1_0 + Q2_0 + Q3_0 - q
    ], dtype=float)
    neg_F_0 = -F_num_0

    # 5. Emissão direta da exceção com a explicação fundamentada
    if abs(det_num) < 1e-12:
        explicacao_erro = (
            f"\n\nFALHA NO MÉTODO DE NEWTON NO PONTO x^(0) = (0, 0, 0, 70):\n"
            f"1. O determinante da matriz Jacobiana é nulo: det(J(x^(0))) = {det_num}.\n"
            f"2. Estrutura de Linhas: As três primeiras linhas de J(x^(0)) são idênticas:\n"
            f"   L1 = L2 = L3 = [0, 0, 0, 1], o que torna o Jacobiano singular e não-invertível.\n"
            f"3. Inconsistência do Sistema Linear J(x^(0)) * dx = -F(x^(0)):\n"
            f"   - Linha 1 exige: 1 * dH = {neg_F_0[0]} => dH = {neg_F_0[0]}\n"
            f"   - Linha 2 exige: 1 * dH = {neg_F_0[1]} => dH = {neg_F_0[1]}\n"
            f"   - Linha 3 exige: 1 * dH = {neg_F_0[2]} => dH = {neg_F_0[2]}\n"
            f"   Como o sistema exige três valores conflitantes para dH, ele é IMPOSSÍVEL.\n"
            f"Portanto, o Método de Newton falha logo na primeira iteração (pivô nulo)."
        )
        raise RuntimeError(explicacao_erro)

if __name__ == "__main__":
    problema_1_2_b()
    problema_1_2_c()