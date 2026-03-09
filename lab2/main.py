import csv
import matplotlib.pyplot as plt

with open('data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['n', 't'])
    writer.writerow(['1000', '3'])
    writer.writerow(['2000', '5'])
    writer.writerow(['4000', '11'])
    writer.writerow(['8000', '28'])
    writer.writerow(['16000', '85'])

def read_data(filename):
    x = []
    y = []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['n']))
            y.append(float(row['t']))
    return x, y

def wkx(k, x, X):
    p = 1.0
    for i in range(k + 1):
        p *= (x - X[i])
    return p

def rr(k, X, Y):
    S = 0.0
    for i in range(k + 1):
        p = 1.0
        for j in range(k + 1):
            if j != i:
                p *= (X[i] - X[j])
        S += Y[i] / p
    return S

def newton_predict(x, N, X, Y):
    S = Y[0]
    for k in range(1, N):
        S += rr(k, X, Y) * wkx(k - 1, x, X)
    return S

x_data, y_data = read_data("data.csv")
nodes_to_test = [3, 4, 5]
target_n = 6000.0

plt.figure(figsize=(10, 6))
for n_nodes in nodes_to_test:
    X_sub = x_data[:n_nodes]
    Y_sub = y_data[:n_nodes]
    
    prediction = newton_predict(target_n, len(X_sub), X_sub, Y_sub)
    print(f"Прогноз для n={target_n} ({n_nodes} вузлів): {prediction:.2f} мс")
    
    calc_x = []
    calc_y = []
    steps = 100
    x_start, x_end = min(x_data), max(x_data)
    step_size = (x_end - x_start) / steps
    
    for i in range(steps + 1):
        current_x = x_start + i * step_size
        calc_x.append(current_x)
        calc_y.append(newton_predict(current_x, len(X_sub), X_sub, Y_sub))
        
    plt.plot(calc_x, calc_y, label=f'Newton ({n_nodes} вузлів)')

plt.scatter(x_data, y_data, color='red', s=50, zorder=5, label='Експериментальні дані')
plt.axvline(x=target_n, color='grey', linestyle='--', alpha=0.7, label=f'Прогноз n={target_n}')
plt.xlabel('Розмір вхідних даних (n)')
plt.ylabel('Час виконання, t (мс)')
plt.title('Прогнозування часу виконання алгоритму')
plt.legend()
plt.grid(True)
plt.show()

def runge_func(x):
    return 1.0 / (1.0 + 25.0 * x**2)

runge_nodes = [5, 10, 20]
a, b = -1.0, 1.0

plt.figure(figsize=(10, 6))

orig_x = []
orig_y = []
steps_plot = 200
step_plot_size = (b - a) / steps_plot
for i in range(steps_plot + 1):
    cx = a + i * step_plot_size
    orig_x.append(cx)
    orig_y.append(runge_func(cx))
plt.plot(orig_x, orig_y, 'k-', linewidth=2, label='f(x) = 1/(1+25x^2)')

colors = ['blue', 'green', 'orange']
for idx, n in enumerate(runge_nodes):
    X_runge = []
    Y_runge = []
    h = (b - a) / n
    for i in range(n + 1):
        xi = a + i * h
        X_runge.append(xi)
        Y_runge.append(runge_func(xi))
    
    calc_runge_x = []
    calc_runge_y = []
    for i in range(steps_plot + 1):
        cx = a + i * step_plot_size
        calc_runge_x.append(cx)
        calc_runge_y.append(newton_predict(cx, len(X_runge), X_runge, Y_runge))
    
    plt.plot(calc_runge_x, calc_runge_y, color=colors[idx], label=f'Newton (n={n})')

plt.ylim(-0.5, 1.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Аналіз ефекту Рунге (рівномірні вузли)')
plt.legend()
plt.grid(True)
plt.show()