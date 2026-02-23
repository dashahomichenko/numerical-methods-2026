import requests
import numpy as np
import matplotlib.pyplot as plt

# 1. Запит до Open-Elevation API для отримання реальних висот маршруту
url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"

response = requests.get(url)
data = response.json()
results = data["results"]
n = len(results)

# Вивід отриманих даних у консоль
print("Кількість вузлів:", n)
print("\nТабуляція вузлів:")
print(" | Latitude | Longitude | Elevation (m)")
for i, point in enumerate(results):
    print(f"{i:2d} | {point['latitude']:.6f} | {point['longitude']:.6f} | {point['elevation']:.2f}")

# Функція для розрахунку відстані між GPS-координатами
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Радіус Землі в метрах
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))

# Обчислення кумулятивної відстані для кожної точки
coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = [p["elevation"] for p in results]
distances = [0]
for i in range(1, n):
    d = haversine(*coords[i-1], *coords[i])
    distances.append(distances[-1] + d)

# Вивід основних характеристик маршруту
print("\nЗагальна довжина маршруту (м):", distances[-1])
total_ascent = sum(max(elevations[i]-elevations[i-1], 0) for i in range(1, n))
print("Сумарний набір висоти (м):", total_ascent)
total_descent = sum(max(elevations[i-1]-elevations[i], 0) for i in range(1, n))
print("Сумарний спуск (м):", total_descent)

# Розрахунок механічної роботи (енергії)
mass = 80
g = 9.81
energy = mass * g * total_ascent
print("Механічна робота (кДж):", energy/1000)

# Створення файлу input.txt для C++ програми
with open("input.txt", "w") as f_out:
    for i in range(n):
        # Формат: індекс, відстань (x), висота (y), крок (h)
        h_val = distances[i] - distances[i-1] if i > 0 else 0
        f_out.write(f"{i}\t{distances[i]:.6e}\t{elevations[i]:.6e}\t{h_val:.6e}\n")
print("\nФайл input.txt успішно створено!")