import random
import math

def generate_system(n, file_a, file_b):
    A = [[0.0] * n for _ in range(n)]
    x_exact = [2.5] * n
    B = [0.0] * n

    for i in range(n):
        row_sum = 0.0
        for j in range(n):
            if i != j:
                A[i][j] = random.uniform(1.0, 9.0)
                row_sum += abs(A[i][j])
        A[i][i] = row_sum + random.uniform(1.0, 5.0)

    for i in range(n):
        b_val = 0.0
        for j in range(n):
            b_val += A[i][j] * x_exact[j]
        B[i] = b_val

    with open(file_a, 'w') as f:
        for row in A:
            f.write('\t'.join([f"{x:.16e}" for x in row]) + '\n')

    with open(file_b, 'w') as f:
        for val in B:
            f.write(f"{val:.16e}\n")

def read_matrix_A(filename):
    A = []
    with open(filename, 'r') as f:
        for line in f:
            A.append([float(x) for x in line.strip().split()])
    return A

def read_vector_B(filename):
    B = []
    with open(filename, 'r') as f:
        for line in f:
            B.append(float(line.strip()))
    return B

def mat_vec_mult(A, x):
    n = len(A)
    res = [0.0] * n
    for i in range(n):
        for j in range(n):
            res[i] += A[i][j] * x[j]
    return res

def vector_norm(x1, x2):
    return max(abs(x1[i] - x2[i]) for i in range(len(x1)))

def matrix_norm(A):
    max_sum = 0.0
    for row in A:
        current_sum = sum(abs(x) for x in row)
        if current_sum > max_sum:
            max_sum = current_sum
    return max_sum

def simple_iteration(A, B, x0, eps):
    n = len(A)
    tau = 1.0 / matrix_norm(A)
    x_k = x0[:]
    iters = 0
    while True:
        iters += 1
        x_next = [0.0] * n
        for i in range(n):
            sum_ax = 0.0
            for j in range(n):
                sum_ax += A[i][j] * x_k[j]
            x_next[i] = x_k[i] - tau * (sum_ax - B[i])

        if vector_norm(x_next, x_k) < eps or iters > 6000:
            break
        x_k = x_next[:]
    return x_next, iters

def jacobi(A, B, x0, eps):
    n = len(A)
    x_k = x0[:]
    iters = 0
    while True:
        iters += 1
        x_next = [0.0] * n
        for i in range(n):
            s = 0.0
            for j in range(n):
                if i != j:
                    s += A[i][j] * x_k[j]
            x_next[i] = (B[i] - s) / A[i][i]

        if vector_norm(x_next, x_k) < eps:
            break
        x_k = x_next[:]
    return x_next, iters

def seidel(A, B, x0, eps):
    n = len(A)
    x_k = x0[:]
    iters = 0
    while True:
        iters += 1
        x_next = [0.0] * n
        max_diff = 0.0
        for i in range(n):
            s = 0.0
            for j in range(n):
                if j < i:
                    s += A[i][j] * x_next[j]
                elif j > i:
                    s += A[i][j] * x_k[j]
            x_new = (B[i] - s) / A[i][i]
            diff = abs(x_new - x_k[i])
            if diff > max_diff:
                max_diff = diff
            x_next[i] = x_new

        x_k = x_next[:]
        if max_diff < eps:
            break
    return x_k, iters

if __name__ == "__main__":
    n = 100
    file_a = "matrix_A.txt"
    file_b = "matrix_B.txt"
    
    eps = 1e-14

    generate_system(n, file_a, file_b)
    
    A = read_matrix_A(file_a)
    B = read_vector_B(file_b)
    
    x0 = [1.0] * n 
    
    x_simple, iters_simple = simple_iteration(A, B, x0, eps)
    x_jacobi, iters_jacobi = jacobi(A, B, x0, eps)
    x_seidel, iters_seidel = seidel(A, B, x0, eps)
    
    print(f"Метод простої ітерації: {iters_simple} ітерацій")
    print(f"Метод Якобі: {iters_jacobi} ітерацій")
    print(f"Метод Зейделя: {iters_seidel} ітерацій")
    print(f"Перші 5 елементів розв'язку (Зейдель): {x_seidel[:5]}")