import numpy as np
import matplotlib.pyplot as plt

def hooke_jeeves(func, x0, delta, eps1, eps2, q=2, p=2.0, max_iter=10000):
    def exploratory_search(x_base, d):
        x = x_base.copy()
        for i in range(len(x)):
            x_new = x.copy()
            x_new[i] = x[i] + d[i]
            if func(x_new) < func(x):
                x = x_new
            else:
                x_new[i] = x[i] - d[i]
                if func(x_new) < func(x):
                    x = x_new
        return x

    x_base = np.array(x0, dtype=float)
    d = np.array(delta, dtype=float)
    trajectory = [x_base.copy()]
    
    k = 0
    while k < max_iter:
        x_new = exploratory_search(x_base, d)
        
        if not np.array_equal(x_new, x_base):
            # Перевірка умови закінчення перед пошуком по зразку
            if np.linalg.norm(d) < eps1 and abs(func(x_new) - func(x_base)) < eps2:
                return x_new, k, trajectory
            
            # Пошук по зразку
            while True:
                x_pattern = x_new + p * (x_new - x_base)
                x_after_expl = exploratory_search(x_pattern, d)
                
                if func(x_after_expl) < func(x_new):
                    x_base = x_new
                    x_new = x_after_expl
                    trajectory.append(x_new.copy())
                    k += 1
                else:
                    x_base = x_new
                    break
        else:
            if np.linalg.norm(d) < eps1:
                return x_base, k, trajectory
            d = d / q
            k += 1
            
    return x_base, k, trajectory

# 1. Система нелінійних рівнянь
def system_f1(x):
    return x[0]**2 + x[1]**2 - 4

def system_f2(x):
    return np.exp(x[0]) + x[1] - 1

# Цільова функція Phi(X) = sum(fi^2)
def target_phi(x):
    return system_f1(x)**2 + system_f2(x)**2

#  Тестова функція Розенброка
def rosenbrock(x):
    return 100 * (x[0]**2 - x[1])**2 + (x[0] - 1)**2


def plot_equations_and_solution(x0_sys, res_sys):
    x_val = np.linspace(-3, 3, 400)
    
    # Рівняння 
    x_circle = np.linspace(-2, 2, 400) # Обмежуємо область визначення для кореня
    y_circle_pos = np.sqrt(4 - x_circle**2)
    y_circle_neg = -np.sqrt(4 - x_circle**2)
    
    # Рівняння 2
    y_exp = 1 - np.exp(x_val)
    
    plt.figure(figsize=(8, 6))
    
    # Графіки рівнянь
    plt.plot(x_circle, y_circle_pos, 'b-', label='x1^2 + x2^2 = 4')
    plt.plot(x_circle, y_circle_neg, 'b-')
    plt.plot(x_val, y_exp, 'g-', label='exp(x1) + x2 = 1')
    
    # Точки: початкове наближення та знайдений розв'язок
    plt.plot(x0_sys[0], x0_sys[1], 'ro', label=f'Початкове наближення: {x0_sys}')
    plt.plot(res_sys[0], res_sys[1], 'm*', markersize=10, label=f'Знайдений розв\'язок: [{res_sys[0]:.4f}, {res_sys[1]:.4f}]')
    
    # Налаштування графіка
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.title('Графіки рівнянь та розв\'язок системи')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()


# Параметри
eps1 = 1e-6
eps2 = 1e-6
q = 2.0
p = 2.0

# Тестування на функції Розенброка
x0_rosen = [-1.2, 0.0]
delta_rosen = [0.1, 0.1]
res_rosen, steps_rosen, _ = hooke_jeeves(rosenbrock, x0_rosen, delta_rosen, eps1, eps2, q, p)

# Розв'язання системи рівнянь
x0_sys = [-1.0, 1.0] # Початкове наближення
delta_sys = [0.2, 0.2]
res_sys, steps_sys, traj_sys = hooke_jeeves(target_phi, x0_sys, delta_sys, eps1, eps2, q, p)

# Вивід траєкторії в файл та результатів в консоль
with open("trajectory.txt", "w", encoding="utf-8") as f:
    f.write("Траєкторія спуску (x1, x2):\n")
    for pt in traj_sys:
        f.write(f"{pt[0]:.6f}\t{pt[1]:.6f}\n")

print(f"--- Тест: Функція Розенброка ---")
print(f"Мінімум: {res_rosen}")
print(f"Значення функції: {rosenbrock(res_rosen)}")
print(f"Кількість кроків: {steps_rosen}\n")

print(f"--- Розв'язання системи рівнянь ---")
print(f"Розв'язок: {res_sys}")
print(f"Значення Phi(X): {target_phi(res_sys)}")
print(f"Кількість кроків: {steps_sys}")
print(f"Траєкторія збережена у файл 'trajectory.txt'\n")

plot_equations_and_solution(x0_sys, res_sys)