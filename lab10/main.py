import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return -y + x + 1

def y_exact(x):
    return x + np.exp(-x)

# Початкові умови та параметри
x0, y0 = 0.0, 1.0
x_end = 5.0
h_init = 0.2
eps_target = 1e-4

def adams_pc2_step(x, y, h, f_prev):
    f_curr = f(x, y)
    # Прогноз (Адамс-Башфорт 2-го порядку)
    y_p = y + h/2 * (3*f_curr - f_prev)
    # Корекція (Адамс-Моултон 2-го порядку)
    f_next_p = f(x + h, y_p)
    y_c = y + h/2 * (f_next_p + f_curr)
    return y_c, y_p, f_curr

# 2. Чисельний розв'язок методом Адамса зі заданим кроком
x_adams = [x0]
y_adams = [y0]
f_prev = f(x0, y0)

# Робимо перший крок методом Ейлера для запуску Адамса
y1 = y0 + h_init * f(x0, y0)
x_adams.append(x0 + h_init)
y_adams.append(y1)

err_exact_adams = [0.0, abs(y1 - y_exact(x0 + h_init))]
err_est_adams = [0.0, 0.0]

x_curr = x0 + h_init
y_curr = y1

while x_curr < x_end:
    y_c, y_p, f_curr = adams_pc2_step(x_curr, y_curr, h_init, f_prev)
    x_curr += h_init
    
    # 3. Локальна похибка за точним розв'язком
    err_ex = abs(y_c - y_exact(x_curr))
    
    # 4. Локальна похибка за оцінкою
    err_est = abs(y_c - y_p) / 6.0
    
    x_adams.append(x_curr)
    y_adams.append(y_c)
    err_exact_adams.append(err_ex)
    err_est_adams.append(err_est)
    
    y_curr = y_c
    f_prev = f_curr

plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.plot(x_adams, err_exact_adams, label='За точним розв.', marker='o', markersize=4)
plt.plot(x_adams, err_est_adams, label='За оцінкою', linestyle='--')
plt.title('Ч.1. Похибка методу Адамса (сталий крок)')
plt.xlabel('x')
plt.ylabel('Похибка')
plt.legend()
plt.grid(True)

# 5. Автоматичний вибір кроку для Адамса
x_auto_adams = [x0]
h_auto_adams = [h_init]
y_curr_auto = y0
x_curr_auto = x0
h_curr = h_init

# Перший крок РК2 для запуску
k1 = f(x_curr_auto, y_curr_auto)
k2 = f(x_curr_auto + h_curr, y_curr_auto + h_curr * k1)
y_next_auto = y_curr_auto + h_curr/2 * (k1 + k2)
f_prev_auto = k1

x_curr_auto += h_curr
x_auto_adams.append(x_curr_auto)
h_auto_adams.append(h_curr)
y_curr_auto = y_next_auto

while x_curr_auto < x_end:
    y_c, y_p, f_curr = adams_pc2_step(x_curr_auto, y_curr_auto, h_curr, f_prev_auto)
    err_est = abs(y_c - y_p) / 6.0
    
    if err_est > eps_target:
        h_curr /= 2
        continue
    
    x_curr_auto += h_curr
    y_curr_auto = y_c
    f_prev_auto = f_curr
    
    x_auto_adams.append(x_curr_auto)
    h_auto_adams.append(h_curr)
    
    if err_est < eps_target / 10:
        h_curr *= 2
        
    if x_curr_auto + h_curr > x_end:
        h_curr = x_end - x_curr_auto
        
    if h_curr < 1e-8:
        break

plt.subplot(2, 2, 2)
plt.step(x_auto_adams, h_auto_adams, where='post', color='green')
plt.title('Ч.1. Залежність кроку h від x (Адамс)')
plt.xlabel('x')
plt.ylabel('Крок h')
plt.grid(True)



# 6. Функція чисельного розв'язку методом РК4
def rk4_step(x, y, h):
    k1 = f(x, y)
    k2 = f(x + h/2, y + h/2 * k1)
    k3 = f(x + h/2, y + h/2 * k2)
    k4 = f(x + h, y + h * k3)
    return y + h/6 * (k1 + 2*k2 + 2*k3 + k4)

x_rk = [x0]
y_rk = [y0]
err_exact_rk = [0.0]
err_runge_rk = [0.0]

x_curr = x0
y_curr = y0

while x_curr < x_end:
    # 7. Один крок з кроком h
    y_next = rk4_step(x_curr, y_curr, h_init)
    
    # 8. Два кроки з кроком h/2 для оцінки за правилом Рунге
    y_half_1 = rk4_step(x_curr, y_curr, h_init/2)
    y_half_2 = rk4_step(x_curr + h_init/2, y_half_1, h_init/2)
    
    err_runge = abs(y_next - y_half_2) / 15.0
    
    x_curr += h_init
    y_curr = y_next
    
    err_ex = abs(y_curr - y_exact(x_curr))
    
    x_rk.append(x_curr)
    y_rk.append(y_curr)
    err_exact_rk.append(err_ex)
    err_runge_rk.append(err_runge)

plt.subplot(2, 2, 3)
plt.plot(x_rk, err_exact_rk, label='За точним розв.', marker='o', markersize=4)
plt.plot(x_rk, err_runge_rk, label='За правилом Рунге', linestyle='--')
plt.title('Ч.2. Похибка методу РК4 (сталий крок)')
plt.xlabel('x')
plt.ylabel('Похибка')
plt.legend()
plt.grid(True)

# 9. Автоматичний вибір кроку для РК4
x_auto_rk = [x0]
h_auto_rk = [h_init]
x_curr = x0
y_curr = y0
h_curr = h_init

while x_curr < x_end:
    y_next = rk4_step(x_curr, y_curr, h_curr)
    y_half_1 = rk4_step(x_curr, y_curr, h_curr/2)
    y_half_2 = rk4_step(x_curr + h_curr/2, y_half_1, h_curr/2)
    
    err_runge = abs(y_next - y_half_2) / 15.0
    
    if err_runge > eps_target:
        h_curr /= 2
        continue
        
    x_curr += h_curr
    y_curr = y_half_2 # беремо точніше значення
    
    x_auto_rk.append(x_curr)
    h_auto_rk.append(h_curr)
    
    # Оцінка необхідної величини кроку
    if err_runge > 0:
        h_new = h_curr * 0.9 * (eps_target / err_runge)**0.2
        h_curr = min(h_new, 2 * h_curr)
    else:
        h_curr *= 2
        
    if x_curr + h_curr > x_end:
        h_curr = x_end - x_curr
        
    if h_curr < 1e-8:
        break

plt.subplot(2, 2, 4)
plt.step(x_auto_rk, h_auto_rk, where='post', color='purple')
plt.title('Ч.2. Залежність кроку h від x (РК4)')
plt.xlabel('x')
plt.ylabel('Крок h')
plt.grid(True)

plt.tight_layout()
plt.show()
