import math
import os

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def F(x):
    return math.exp(-x) - math.sin(x)

def F_prime(x):
    return -math.exp(-x) - math.cos(x)

def F_double_prime(x):
    return math.exp(-x) + math.sin(x)

def plot_transcendental_function(a, b):
    
    x_vals = np.linspace(a, b, 400)
    y_vals = [F(x) for x in x_vals]

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, label="F(x) = exp(-x) - sin(x)", color="blue")
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.title("Графік трансцендентної функції F(x)")
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.grid(True)
    plt.legend()
    plt.show()

def tabulate_and_find_roots(a, b, h, filename="tabulation.txt"):
    roots = []
    with open(filename, "w") as f:
        f.write("x\tF(x)\n")
        x = a
        prev_x = x
        prev_y = F(x)
        while x <= b + h/2:
            y = F(x)
            f.write(f"{x:.4f}\t{y:.6f}\n")
            
            if prev_y * y <= 0 and x != a:
                approx_x = prev_x - prev_y * (x - prev_x) / (y - prev_y)
                behavior = "зростає" if y > prev_y else "спадає"
                roots.append((prev_x, x, approx_x, behavior))
            
            prev_x = x
            prev_y = y
            x += h
    return roots


def check_criterion(x_new, x_old, eps):
    return abs(F(x_new)) < eps and abs(x_new - x_old) < eps

def method_simple_iteration(x0, tau, eps=1e-10):
    x = x0
    iters = 0
    while True:
        iters += 1
        x_new = x + tau * F(x)
        if check_criterion(x_new, x, eps):
            return x_new, iters
        x = x_new
        if iters > 10000: break
    return x, iters

def method_newton(x0, eps=1e-10):
    x = x0
    iters = 0
    while True:
        iters += 1
        x_new = x - F(x) / F_prime(x)
        if check_criterion(x_new, x, eps):
            return x_new, iters
        x = x_new
        if iters > 10000: break
    return x, iters

def method_chebyshev(x0, eps=1e-10):
    x = x0
    iters = 0
    while True:
        iters += 1
        f_x = F(x)
        f_p = F_prime(x)
        f_dp = F_double_prime(x)
        
        x_new = x - f_x / f_p - 0.5 * (f_x**2 * f_dp) / (f_p**3)
        if check_criterion(x_new, x, eps):
            return x_new, iters
        x = x_new
        if iters > 10000: break
    return x, iters

def method_chord(x0, x1, eps=1e-10):
    iters = 0
    while True:
        iters += 1
        f_x1 = F(x1)
        f_x0 = F(x0)
        
        if f_x1 - f_x0 == 0: break
        
        x_new = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        if check_criterion(x_new, x1, eps):
            return x_new, iters
            
        x0, x1 = x1, x_new
        if iters > 10000: break
    return x1, iters

def method_parabola(x0, x1, x2, eps=1e-10):
    iters = 0
    while True:
        iters += 1
        f0, f1, f2 = F(x0), F(x1), F(x2)
        
        div1 = (f1 - f0) / (x1 - x0)
        div2 = (f2 - f1) / (x2 - x1)
        a_coef = (div2 - div1) / (x2 - x0)
        b_coef = div2 + (x2 - x1) * a_coef
        c_coef = f2
        
        disc = b_coef**2 - 4 * a_coef * c_coef
        if disc < 0: disc = 0 
            
        sqrt_disc = math.sqrt(disc)
        den1 = b_coef + sqrt_disc
        den2 = b_coef - sqrt_disc
        
        den = den1 if abs(den1) > abs(den2) else den2
        if den == 0: break
            
        delta = -2 * c_coef / den
        x_new = x2 + delta
        
        if check_criterion(x_new, x2, eps):
            return x_new, iters
            
        x0, x1, x2 = x1, x2, x_new
        if iters > 10000: break
    return x2, iters

def method_inverse_interpolation(x0, x1, x2, eps=1e-10):
    iters = 0
    while True:
        iters += 1
        y0, y1, y2 = F(x0), F(x1), F(x2)
        
        try:
            term1 = (y1 * y2) / ((y0 - y1) * (y0 - y2)) * x0
            term2 = (y0 * y2) / ((y1 - y0) * (y1 - y2)) * x1
            term3 = (y0 * y1) / ((y2 - y0) * (y2 - y1)) * x2
            x_new = term1 + term2 + term3
        except ZeroDivisionError:
            break
            
        if check_criterion(x_new, x2, eps):
            return x_new, iters
            
        x0, x1, x2 = x1, x2, x_new
        if iters > 10000: break
    return x2, iters


def write_polynomial_coeffs(filename, coeffs):
    with open(filename, "w") as f:
        f.write(" ".join(map(str, coeffs)))

def read_polynomial_coeffs(filename):
    with open(filename, "r") as f:
        return list(map(float, f.read().split()))

def horner_evaluate(coeffs, x):
    n = len(coeffs) - 1
    b = [0] * (n + 1)
    b[0] = coeffs[0]
    
    for i in range(1, n + 1):
        b[i] = coeffs[i] + x * b[i-1]
        
    c = [0] * n
    c[0] = b[0]
    for i in range(1, n):
        c[i] = b[i] + x * c[i-1]
        
    return b[-1], c[-1]

def plot_algebraic_function(coeffs, a, b):
    if not HAS_MATPLOTLIB:
        print("Увага: бібліотеки matplotlib/numpy не встановлені. Графік алгебраїчного рівняння не побудовано.")
        return
    
    x_vals = np.linspace(a, b, 400)
    y_vals = [horner_evaluate(coeffs, x)[0] for x in x_vals]

    plt.figure(figsize=(8, 5))
    equation_str = f"P(x) = {coeffs[0]}x^3 + ({coeffs[1]})x^2 + ({coeffs[2]})x + ({coeffs[3]})"
    plt.plot(x_vals, y_vals, label=equation_str, color="red")
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.title("Графік алгебраїчного рівняння 3-го порядку")
    plt.xlabel("x")
    plt.ylabel("P(x)")
    plt.grid(True)
    plt.legend()
    plt.show()

def method_newton_horner(coeffs, x0, eps=1e-10):
    x = x0
    iters = 0
    while True:
        iters += 1
        p_val, p_prime_val = horner_evaluate(coeffs, x)
        
        if p_prime_val == 0: break
        
        x_new = x - p_val / p_prime_val
        if abs(p_val) < eps and abs(x_new - x) < eps:
            return x_new, iters
            
        x = x_new
        if iters > 10000: break
    return x, iters

def method_lin(coeffs, p0, q0, eps=1e-10):
    p, q = p0, q0
    iters = 0
    
    alpha_old = -p / 2
    disc_old = q - alpha_old**2
    beta_old = math.sqrt(disc_old) if disc_old > 0 else 0

    while True:
        iters += 1
        b3 = coeffs[0]
        b2 = coeffs[1] - p * b3
        
        if abs(b2) < 1e-12:
            break
            
        a1 = coeffs[2]
        a0 = coeffs[3]
        
        p_new = (a1 * b2 - a0 * b3) / (b2**2)
        q_new = a0 / b2
        
        alpha_new = -p_new / 2
        disc = q_new - alpha_new**2
        beta_new = math.sqrt(disc) if disc > 0 else 0
        
        if abs(alpha_new - alpha_old) <= eps and abs(beta_new - beta_old) <= eps:
            return alpha_new, beta_new, iters
            
        p, q = p_new, q_new
        alpha_old, beta_old = alpha_new, beta_new
        
        if iters > 10000: break
        
    return alpha_new, beta_new, iters


def main():
    print("="*60)
    print("1. ТАБУЛЯЦІЯ, ЛОКАЛІЗАТОР КОРЕНІВ ТА ГРАФІК")
    print("="*60)
    
    # Побудова графіка трансцендентної функції на проміжку [0, 4]
    plot_transcendental_function(0, 4)
    
    roots = tabulate_and_find_roots(a=0, b=4, h=0.1, filename="tabulation.txt")
    print("Результати табуляції записано у файл tabulation.txt")
    
    root_inc = None
    root_dec = None
    
    for r in roots:
        print(f"Знайдено інтервал: [{r[0]:.1f}, {r[1]:.1f}], наближений корінь: {r[2]:.4f}, функція {r[3]}")
        if r[3] == "зростає" and not root_inc:
            root_inc = r[2]
        if r[3] == "спадає" and not root_dec:
            root_dec = r[2]

    eps = 1e-10
    print("\n" + "="*60)
    print("2-4. УТОЧНЕННЯ КОРЕНІВ ТРАНСЦЕНДЕНТНОГО РІВНЯННЯ")
    print("="*60)

    if root_dec:
        print(f"\n--- Корінь 1 (функція спадає), початкове наближення: x0 = {root_dec:.4f} ---")
        tau1 = 0.5 
        
        res, itrs = method_simple_iteration(root_dec, tau1, eps)
        print(f"Метод простої ітерації:    x = {res:.10f}, ітерацій = {itrs}")
        res, itrs = method_newton(root_dec, eps)
        print(f"Метод Ньютона:             x = {res:.10f}, ітерацій = {itrs}")
        res, itrs = method_chebyshev(root_dec, eps)
        print(f"Метод Чебишева:            x = {res:.10f}, ітерацій = {itrs}")
        res, itrs = method_chord(root_dec - 0.1, root_dec + 0.1, eps)
        print(f"Метод хорд:                x = {res:.10f}, ітерацій = {itrs}")
        res, itrs = method_parabola(root_dec - 0.1, root_dec, root_dec + 0.1, eps)
        print(f"Метод парабол:             x = {res:.10f}, ітерацій = {itrs}")
        res, itrs = method_inverse_interpolation(root_dec - 0.1, root_dec, root_dec + 0.1, eps)
        print(f"Метод оберненої інтерпол.: x = {res:.10f}, ітерацій = {itrs}")

    if root_inc:
        print(f"\n--- Корінь 2 (функція зростає), початкове наближення: x0 = {root_inc:.4f} ---")
        tau2 = -0.5 
        
        res, itrs = method_simple_iteration(root_inc, tau2, eps)
        print(f"Метод простої ітерації:    x = {res:.10f}, ітерацій = {itrs}")
        res, itrs = method_newton(root_inc, eps)
        print(f"Метод Ньютона:             x = {res:.10f}, ітерацій = {itrs}")
        res, itrs = method_chebyshev(root_inc, eps)
        print(f"Метод Чебишева:            x = {res:.10f}, ітерацій = {itrs}")
        res, itrs = method_chord(root_inc - 0.1, root_inc + 0.1, eps)
        print(f"Метод хорд:                x = {res:.10f}, ітерацій = {itrs}")
        res, itrs = method_parabola(root_inc - 0.1, root_inc, root_inc + 0.1, eps)
        print(f"Метод парабол:             x = {res:.10f}, ітерацій = {itrs}")
        res, itrs = method_inverse_interpolation(root_inc - 0.1, root_inc, root_inc + 0.1, eps)
        print(f"Метод оберненої інтерпол.: x = {res:.10f}, ітерацій = {itrs}")


    print("\n" + "="*60)
    print("5-9. АЛГЕБРАЇЧНІ РІВНЯННЯ ТРЕТЬОГО ПОРЯДКУ ТА ГРАФІК")
    print("="*60)
    
    poly_coeffs = [1.0, -6.0, 10.0, -8.0]
    filename_poly = "poly_coeffs.txt"
    
    write_polynomial_coeffs(filename_poly, poly_coeffs)
    print(f"Коефіцієнти рівняння x^3 - 6x^2 + 10x - 8 = 0 записано у файл {filename_poly}")
    
    loaded_coeffs = read_polynomial_coeffs(filename_poly)
    print(f"Зчитані коефіцієнти: {loaded_coeffs}")
    
    # Побудова графіка алгебраїчного рівняння на проміжку [-1, 6] (корінь x=4)
    plot_algebraic_function(loaded_coeffs, -1, 6)
    
    print("\n--- Дійсний корінь (Метод Ньютона + Схема Горнера) ---")
    x_real, itrs_real = method_newton_horner(loaded_coeffs, x0=4.5, eps=eps)
    print(f"Дійсний корінь: x = {x_real:.10f}, ітерацій = {itrs_real}")
    
    print("\n--- Комплексні корені (Метод Ліна) ---")
    p_guess = -2.1
    q_guess = 1.9
    alpha, beta, itrs_comp = method_lin(loaded_coeffs, p_guess, q_guess, eps=eps)
    print(f"Комплексні корені: x = {alpha:.10f} +/- {beta:.10f}*i, ітерацій = {itrs_comp}")

if __name__ == "__main__":
    main()