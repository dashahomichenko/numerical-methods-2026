import random

def generate_and_save_data(n, filename_A, filename_B):
    A = [[random.uniform(1.0, 10.0) for _ in range(n)] for _ in range(n)]
    
    with open(filename_A, 'w', encoding='utf-8') as fA:
        for row in A:
            fA.write(" ".join(map(str, row)) + "\n")
            
    X_exact = [2.5 for _ in range(n)]
    
    B = [sum(A[i][j] * X_exact[j] for j in range(n)) for i in range(n)]
    
    with open(filename_B, 'w', encoding='utf-8') as fB:
        for val in B:
            fB.write(str(val) + "\n")

def read_matrix(filename):
    A = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                A.append([float(x) for x in line.split()])
    return A

def read_vector(filename):
    B = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                B.append(float(line.strip()))
    return B

def lu_decomposition(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        U[i][i] = 1.0
        
    for k in range(n):
        for i in range(k, n):
            s = sum(L[i][j] * U[j][k] for j in range(k))
            L[i][k] = A[i][k] - s
            
        for i in range(k + 1, n):
            s = sum(L[k][j] * U[j][i] for j in range(k))
            if L[k][k] != 0:
                U[k][i] = (A[k][i] - s) / L[k][k]
            else:
                U[k][i] = 0.0
            
    return L, U

def save_lu_to_file(L, U, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Matrix L:\n")
        for row in L:
            f.write(" ".join(map(str, row)) + "\n")
        f.write("\nMatrix U:\n")
        for row in U:
            f.write(" ".join(map(str, row)) + "\n")

def solve_lu(L, U, B):
    n = len(L)
    Z = [0.0] * n
    for i in range(n):
        s = sum(L[i][j] * Z[j] for j in range(i))
        Z[i] = (B[i] - s) / L[i][i]
        
    X = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(U[i][j] * X[j] for j in range(i + 1, n))
        X[i] = Z[i] - s
        
    return X

def multiply_matrix_vector(A, X):
    n = len(A)
    res = [0.0] * n
    for i in range(n):
        res[i] = sum(A[i][j] * X[j] for j in range(n))
    return res

def vector_norm(V):
    return max(abs(v) for v in V)

def vector_subtract(V1, V2):
    return [v1 - v2 for v1, v2 in zip(V1, V2)]

def vector_add(V1, V2):
    return [v1 + v2 for v1, v2 in zip(V1, V2)]

def main():
    n = 100
    file_A = "matrix_A.txt"
    file_B = "vector_B.txt"
    file_LU = "matrix_LU.txt"
    eps_0 = 1e-14
    
    generate_and_save_data(n, file_A, file_B)
    
    A = read_matrix(file_A)
    B = read_vector(file_B)
    
    L, U = lu_decomposition(A)
    save_lu_to_file(L, U, file_LU)
    
    X_0 = solve_lu(L, U, B)
    
    AX_0 = multiply_matrix_vector(A, X_0)
    diff_0 = vector_subtract(AX_0, B)
    eps_initial = vector_norm(diff_0)
    print(f"Початкова точність розв'язку (eps): {eps_initial}")
    
    X = X_0.copy()
    iterations = 0
    
    while True:
        AX = multiply_matrix_vector(A, X)
        R = vector_subtract(B, AX)
        
        norm_R = vector_norm(R)
        if norm_R <= eps_0:
            break
            
        dX = solve_lu(L, U, R)
        
        X = vector_add(X, dX)
        iterations += 1
        
        norm_dX = vector_norm(dX)
        if norm_dX <= eps_0:
            break
            
        if iterations > 1000:
            print("Досягнуто ліміт ітерацій.")
            break
            
    print(f"Кількість ітерацій для досягнення точності {eps_0}: {iterations}")
    
    AX_final = multiply_matrix_vector(A, X)
    diff_final = vector_subtract(AX_final, B)
    eps_final = vector_norm(diff_final)
    print(f"Точність після уточнення: {eps_final}")

if __name__ == "__main__":
    main()