import math
import csv
import numpy as np
import matplotlib.pyplot as plt

def f_lab3(x):
    return math.sin(x)

def run_lab_3_1():
    a = 0.0
    b = 1.0
    n = 20
    h = (b - a) / n
    
    with open("in.txt", "w") as file1:
        for i in range(n + 1):
            x = a + i * h
            y = f_lab3(x)
            file1.write(f"{x:e} \t {y:e}\n")

x_arr = [0.0] * 100
f_arr = [0.0] * 100

def fact(k):
    if k <= 1: return 1
    return k * fact(k - 1)

def Cnk(n, k):
    if k < 0 or k > n: return 0
    return fact(n) / (fact(k) * fact(n - k))

def step(n):
    return 1 if n % 2 == 0 else -1

def deltaf(n):
    r = 0.0
    for k in range(n + 1):
        r += f_arr[k] * step(n - k) * Cnk(n, k)
    return r

def factmn(t, k):
    mn = 1.0
    if k == 0: return 1.0
    for i in range(k):
        mn *= (t - i)
    return mn

def fappr(n, t):
    res = 0.0
    for k in range(n + 1):
        res += (deltaf(k) * factmn(t, k) / fact(k))
    return res

def Eps(appr, in_val):
    return abs(appr - in_val)

def run_lab_3_2():
    n = 20
    a = 0.0
    b = 1.0
    h = (b - a) / n
    
    try:
        with open("in.txt", "r") as file1:
            lines = file1.readlines()
            for i in range(n + 1):
                if i < len(lines):
                    parts = lines[i].split()
                    x_arr[i] = float(parts[0])
                    f_arr[i] = float(parts[1])
    except FileNotFoundError:
        print("Файл in.txt не знайдено.")
        return

    with open("fappr.txt", "w") as file2, open("R.txt", "w") as file3:
        t = 0.0
        nt = 1.0 / 20.0
        
        for j in range(20 * n + 1):
            current_x = a + (t * h)
            appr_val = fappr(n, t)
            real_val = f_lab3(current_x)
            
            file2.write(f"{current_x:e} \t {appr_val:e}\n")
            file3.write(f"{current_x:e} \t {Eps(appr_val, real_val):e}\n")
            t += nt

def form_matrix(x, m):
    matrix = np.zeros((m + 1, m + 1))
    for i in range(m + 1):
        for j in range(m + 1):
            matrix[i, j] = sum(xi**(i + j) for xi in x)
    return matrix

def form_vector(x, y, m):
    vector = np.zeros(m + 1)
    for i in range(m + 1):
        vector[i] = sum(yi * (xi**i) for xi, yi in zip(x, y))
    return vector

def gauss_solve(A, b):
    n = len(b)
    A = A.astype(float)
    b = b.astype(float)
    
    for k in range(n):
        max_row = k + np.argmax(np.abs(A[k:, k]))
        A[[k, max_row]] = A[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]
        
        for i in range(k + 1, n):
            if A[k, k] == 0: continue
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]
            
    x_sol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x_sol[i] = (b[i] - np.dot(A[i, i + 1:], x_sol[i + 1:])) / A[i, i]
    return x_sol

def polynomial(x, coef):
    x = np.array(x)
    y_poly = np.zeros_like(x, dtype=float)
    for i, c in enumerate(coef):
        y_poly += c * (x**i)
    return y_poly

def variance(y_true, y_approx):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_approx))**2))

def run_main():
    x_data = []
    y_data = []

    try:
        with open('data.csv', mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                x_data.append(float(row['Month']))
                y_data.append(float(row['Temp']))
    except FileNotFoundError:
        print("Файл data.csv не знайдено. Пропускаю візуалізацію main.py.")
        return

    variances = []
    degrees = list(range(1, 11))
    
    all_y_approx = {}
    all_errors = {}

    for m in degrees:
        A = form_matrix(x_data, m)
        b_vec = form_vector(x_data, y_data, m)
        coef = gauss_solve(A, b_vec)
        y_approx = polynomial(x_data, coef)
        var = variance(y_data, y_approx)
        variances.append(var)
        all_y_approx[m] = y_approx
        all_errors[m] = np.array(y_data) - y_approx

    optimal_m = degrees[np.argmin(variances)]
    print(f"Дисперсії для m=1..10: {variances}")
    print(f"Оптимальний ступінь m: {optimal_m}")

    A_opt = form_matrix(x_data, optimal_m)
    b_opt = form_vector(x_data, y_data, optimal_m)
    coef_opt = gauss_solve(A_opt, b_opt)
    
    
    x0 = min(x_data) if x_data else 0
    xn = max(x_data) if x_data else 1
    n_points = len(x_data)
    h1 = (xn - x0) / (20 * n_points) if n_points > 0 else 0.1
    x_dense = np.arange(x0, xn + h1, h1)
    y_dense_opt = polynomial(x_dense, coef_opt)

    x_future = [25, 26, 27]
    y_future = polynomial(x_future, coef_opt)
    print(f"Прогноз на місяці 25, 26, 27: {y_future}")

    plt.figure(figsize=(14, 10))


    plt.subplot(2, 2, 1)
    plt.scatter(x_data, y_data, color='red', label='Фактичні дані')
    plt.plot(x_dense, y_dense_opt, label=f'Апроксимація (m={optimal_m})')
    plt.plot(x_future, y_future, 'go--', label='Прогноз')
    plt.legend()
    plt.title('Температура: Дані та Апроксимація')


    plt.subplot(2, 2, 2)
    plt.plot(degrees, variances, 'bo-')
    plt.title('Залежність дисперсії від ступеня m')
    plt.xlabel('m')
    plt.ylabel('Дисперсія')


    plt.subplot(2, 2, 3)
    for m in degrees:
        plt.plot(x_data, all_errors[m], label=f'm={m}')
    plt.title('Похибки для m=1..10')
    plt.xlabel('Місяць')
    plt.ylabel('Похибка')
    plt.legend(fontsize=8)


    plt.subplot(2, 2, 4)
    plt.bar(x_data, all_errors[optimal_m])
    plt.title(f'Похибка апроксимації (m={optimal_m})')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_lab_3_1()
    run_lab_3_2()
    run_main()