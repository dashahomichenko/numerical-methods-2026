import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)


x_vals = np.linspace(0, 24, 1000)
y_vals = f(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label=r'$f(x)=50+20\sin\left(\frac{\pi x}{12}\right)+5e^{-0.2(x-12)^2}$')
plt.title('Графік функції навантаження на сервер')
plt.xlabel('Час, x (год)')
plt.ylabel('Навантаження, f(x)')
plt.grid(True)
plt.legend()
plt.show()

a, b = 0, 24
I0, _ = quad(f, a, b, epsabs=1e-14, epsrel=1e-14)
print(f"Точне значення інтегралу I0: {I0}")

def simpson(f, a, b, N):
    if N % 2 != 0:
        N += 1  
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    

    integral = (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])
    return integral

N_values = np.arange(10, 1002, 2)
errors = []

target_eps = 1e-12
N_opt = None

for N in N_values:
    I_N = simpson(f, a, b, N)
    err = abs(I_N - I0)
    errors.append(err)
    if N_opt is None and err <= target_eps:
        N_opt = N

if N_opt is None:
    N_opt = N_values[-1]

epsopt = abs(simpson(f, a, b, N_opt) - I0)

print(f"N_opt (для точності 1e-12): {N_opt}")
print(f"epsopt: {epsopt}")

plt.figure(figsize=(10, 6))
plt.plot(N_values, errors, color='red')
plt.yscale('log')
plt.title('Залежність похибки методу Сімпсона від кількості вузлів N')
plt.xlabel('Кількість розбиттів N')
plt.ylabel(r'Абсолютна похибка $\epsilon(N)$')
plt.grid(True, which="both", ls="--")
plt.show()

N0_target = N_opt / 10
N0 = max(8, int(np.round(N0_target / 8.0)) * 8)

I_N0 = simpson(f, a, b, N0)
eps0 = abs(I_N0 - I0)

print(f"N0 (кратне 8): {N0}")
print(f"Похибка при N0 (eps0): {eps0}")

I_N0_2 = simpson(f, a, b, N0 // 2)

I_R = I_N0 + (I_N0 - I_N0_2) / 15
epsR = abs(I_R - I0)


print(f"Уточнене значення I_R: {I_R}")
print(f"Похибка методу Рунге-Ромберга (epsR): {epsR}")

I_N0_4 = simpson(f, a, b, N0 // 4)

numerator = (I_N0_2)**2 - I_N0 * I_N0_4
denominator = 2 * I_N0_2 - (I_N0 + I_N0_4)

if denominator != 0:
    I_E = numerator / denominator
else:
    I_E = I_N0

epsE = abs(I_E - I0)

# Порядок методу
if (I_N0_2 - I_N0) != 0:
    p = (1 / np.log(2)) * np.log(abs((I_N0_4 - I_N0_2) / (I_N0_2 - I_N0)))
else:
    p = 0


print(f"Уточнене значення I_E: {I_E}")
print(f"Похибка методу Ейткена (epsE): {epsE}")
print(f"Порядок методу за Ейткеном p: {p}")


print("Порівняння похибок:")
print(f"Початкова похибка (Сімпсон, N={N0}): {eps0}")
print(f"Похибка Рунге-Ромберга:            {epsR}")
print(f"Похибка Ейткена:                   {epsE}")

function_evals = 0

def adaptive_simpson(f, a, b, tol):
    global function_evals
    c = (a + b) / 2
    h = b - a
    
    fa = f(a)
    fc = f(c)
    fb = f(b)
    f_mid_left = f((a + c) / 2)
    f_mid_right = f((c + b) / 2)
    
    function_evals += 5
    
    S1 = (h / 6) * (fa + 4 * fc + fb)
    S2_left = (h / 12) * (fa + 4 * f_mid_left + fc)
    S2_right = (h / 12) * (fc + 4 * f_mid_right + fb)
    S2 = S2_left + S2_right
    
    if abs(S1 - S2) <= 15 * tol:
        return S2 + (S2 - S1) / 15
    else:
        left_int = adaptive_simpson(f, a, c, tol / 2)
        right_int = adaptive_simpson(f, c, b, tol / 2)
        return left_int + right_int


delta = 1e-8
function_evals = 0
I_adapt = adaptive_simpson(f, a, b, delta)
eps_adapt = abs(I_adapt - I0)

print(f"Адаптивний інтеграл (delta={delta}): {I_adapt}")
print(f"Похибка адаптивного алгоритму: {eps_adapt}")
print(f"Кількість обчислень функції: {function_evals}")