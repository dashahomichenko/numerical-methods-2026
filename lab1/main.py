import requests 
import numpy as np 
import matplotlib.pyplot as plt 

url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106" 
response = requests.get(url) 
data = response.json() 
results = data["results"] 
n = len(results) 

print("Кількість вузлів:", n)
print("\nТабуляція вузлів:") 
with open("tabulation.txt", "w", encoding="utf-8") as f:
    header = " | Latitude | Longitude | Elevation (m)"
    print(header)
    f.write(header + "\n")
    for i, point in enumerate(results): 
        line = f"{i:2d} | {point['latitude']:.6f} | {point['longitude']:.6f} | {point['elevation']:.2f}"
        print(line)
        f.write(line + "\n")

def haversine(lat1, lon1, lat2, lon2): 
    R = 6371000  
    phi1, phi2 = np.radians(lat1), np.radians(lat2) 
    dphi = np.radians(lat2 - lat1) 
    dlambda = np.radians(lon2 - lon1) 
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2 
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a)) 

coords = [(p["latitude"], p["longitude"]) for p in results] 
elevations = [p["elevation"] for p in results] 
distances = [0] 

for i in range(1, n): 
    d = haversine(*coords[i-1], *coords[i]) 
    distances.append(distances[-1] + d) 

print("\nЗагальна довжина маршруту (м):", distances[-1]) 
total_ascent = sum(max(elevations[i]-elevations[i-1], 0) for i in range(1, n)) 
print("Сумарний набір висоти (м):", total_ascent) 
total_descent = sum(max(elevations[i-1]-elevations[i], 0) for i in range(1, n)) 
print("Сумарний спуск (м):", total_descent) 

mass = 80 
g = 9.81 
energy = mass * g * total_ascent 
print("Механічна робота (Дж):", energy)
print("Механічна робота (кДж):", energy/1000)

def progonka(x, y, quiet=False):
    N = len(x) - 1
    h = [0.0] * (N + 1)
    for i in range(1, N + 1):
        h[i] = x[i] - x[i-1]

    alfa = [0.0] * (N + 1)
    beta = [0.0] * (N + 1)
    hamma = [0.0] * (N + 1)
    delta = [0.0] * (N + 1)
    A = [0.0] * (N + 1)
    B = [0.0] * (N + 1)
    c = [0.0] * (N + 1)

    alfa[1] = hamma[1] = delta[1] = 0.0
    beta[1] = 1.0

    for i in range(2, N + 1):
        alfa[i] = h[i-1]
        beta[i] = 2 * (h[i-1] + h[i])
        hamma[i] = h[i]
        delta[i] = 3 * (((y[i] - y[i-1]) / h[i]) - ((y[i-1] - y[i-2]) / h[i-1]))

    if N > 0:
        hamma[N] = 0.0

    if N > 1:
        A[1] = -hamma[1] / beta[1]
        B[1] = delta[1] / beta[1]
        for i in range(2, N):
            denom = alfa[i] * A[i-1] + beta[i]
            A[i] = -hamma[i] / denom
            B[i] = (delta[i] - alfa[i] * B[i-1]) / denom
        
        c[N] = (delta[N] - alfa[N] * B[N-1]) / (alfa[N] * A[N-1] + beta[N])
        for i in range(N, 1, -1):
            c[i-1] = A[i-1] * c[i] + B[i-1]

    if not quiet:
        print("\nКоефіцієнти методу прогонки:")
        for i in range(1, N + 1):
            print(f"i={i}: alfa={alfa[i]:.6e}, beta={beta[i]:.6e}, hamma={hamma[i]:.6e}, delta={delta[i]:.6e}, A={A[i]:.6e}, B={B[i]:.6e}")

        print("\nКоефіцієнти c:")
        for i in range(1, N + 1):
            print(f"c[{i}]={c[i]:.6e}")

    a = [0.0] * (N + 1)
    b = [0.0] * (N + 1)
    d = [0.0] * (N + 1)

    for i in range(1, N):
        a[i] = y[i-1]
        b[i] = (y[i] - y[i-1]) / h[i] - (h[i] / 3.0) * (c[i+1] + 2.0 * c[i])
        d[i] = (c[i+1] - c[i]) / (3.0 * h[i])

    if N > 0:
        a[N] = y[N-1]
        b[N] = (y[N] - y[N-1]) / h[N] - (2.0 / 3.0) * h[N] * c[N]
        d[N] = -c[N] / (3.0 * h[N])

    if not quiet:
        print("\nКоефіцієнти a, b, d:")
        for i in range(1, N + 1):
            print(f"i={i}: a={a[i]:.6e}, b={b[i]:.6e}, d={d[i]:.6e}")

    return a, b, c, d

a_etalon, b_etalon, c_etalon, d_etalon = progonka(distances, elevations, quiet=False)


x_dense = np.linspace(min(distances), max(distances), 500)
y_etalon_dense = np.interp(x_dense, distances, elevations)

grad_full = np.gradient(y_etalon_dense, x_dense) * 100
print("\nМаксимальний підйом (%):", np.max(grad_full))
print("Максимальний спуск (%):", np.min(grad_full))
print("Середній градієнт (%):", np.mean(np.abs(grad_full)))

def evaluate_spline(x_nodes, y_nodes, x_eval):
    a, b, c, d = progonka(x_nodes, y_nodes, quiet=True)
    y_eval = []
    N = len(x_nodes) - 1
    for x in x_eval:
        idx = 1
        for i in range(1, N + 1):
            if x <= x_nodes[i] or i == N:
                idx = i
                break
        dx = x - x_nodes[idx-1]
        y_eval.append(a[idx] + b[idx]*dx + c[idx]*dx**2 + d[idx]*dx**3)
    return np.array(y_eval)

plt.figure(figsize=(10, 5))
plt.plot(distances, elevations, label='21 вузол')

splines_y = {}
for num_nodes in [10, 15, 20]:
    indices = np.linspace(0, n - 1, num_nodes, dtype=int)
    dist_sub = [distances[i] for i in indices]
    elev_sub = [elevations[i] for i in indices]
    
    y_spline_dense = evaluate_spline(dist_sub, elev_sub, x_dense)
    splines_y[num_nodes] = y_spline_dense
    plt.plot(x_dense, y_spline_dense, label=f'{num_nodes} вузлів')

plt.title('Вплив кількості вузлів')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
for num_nodes in [10, 15, 20]:
    error = np.abs(y_etalon_dense - splines_y[num_nodes])
    plt.plot(x_dense, error, label=f'{num_nodes} вузлів')

plt.title('Похибка апроксимації')
plt.legend()
plt.show()