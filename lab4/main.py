import math
import matplotlib.pyplot as plt

t0 = 1.0

def M(t):
    return 50 * math.exp(-0.1 * t) + 5 * math.sin(t)

def dM_exact(t):
    return -5 * math.exp(-0.1 * t) + 5 * math.cos(t)

exact_val = dM_exact(t0)
print(f"1. Точне значення похідної в точці t0={t0}: {exact_val}")

def dM_approx(t, h):
    if h == 0: return float('inf')
    return (M(t + h) - M(t - h)) / (2 * h)

h_values = [10**(-i) for i in range(20, -4, -1)]
best_h = 1.0
min_error = float('inf')

errors = []
valid_hs = []

for h in h_values:
    try:
        approx_val = dM_approx(t0, h)
        error = abs(approx_val - exact_val)
        if error < min_error:
            min_error = error
            best_h = h
        if h >= 1e-15: 
            errors.append(error)
            valid_hs.append(h)
    except Exception:
        pass

print(f"2. Оптимальний крок h0: {best_h:.1e}, досягнута точність R0: {min_error:.1e}")

h_step = 1e-3
print(f"3. Прийнято значення кроку h = {h_step}")

dM_h = dM_approx(t0, h_step)
dM_2h = dM_approx(t0, 2 * h_step)
print(f"4. Значення похідної: для h -> {dM_h}, для 2h -> {dM_2h}")

R1 = abs(dM_h - exact_val)
print(f"5. Похибка при кроці h (R1): {R1}")

dM_RR = dM_h + (dM_h - dM_2h) / 3
R2 = abs(dM_RR - exact_val)
print(f"6. Уточнене значення (Рунге-Ромберг): {dM_RR}, Похибка (R2): {R2}")
if R2 != 0:
    print(f"   Характер зміни: похибка зменшилась у {R1/R2:.2f} разів.")
else:
    print("   Характер зміни: похибка стала нульовою (в межах точності).")

dM_4h = dM_approx(t0, 4 * h_step)
print(f"7. Значення похідної для 4h: {dM_4h}")

numerator = dM_2h**2 - dM_4h * dM_h
denominator = 2 * dM_2h - (dM_4h + dM_h)

if denominator != 0:
    dM_E = numerator / denominator
else:
    dM_E = dM_h

ratio = abs((dM_4h - dM_2h) / (dM_2h - dM_h)) if (dM_2h - dM_h) != 0 else 0
if ratio > 0:
    p = (1 / math.log(2)) * math.log(ratio)
else:
    p = 0

R3 = abs(dM_E - exact_val)

print(f"   Уточнене значення (Ейткен): {dM_E}")
print(f"   Порядок точності p: {p:.2f}")
print(f"   Похибка (R3): {R3}")
if R3 != 0:
    print(f"   Характер зміни: похибка за методом Ейткена зменшилась у {R1/R3:.2f} разів порівняно з R1.")

print("\n--- ВИСНОВОК ---")
print("Оптимальні режими поливу: оскільки швидкість зміни вологості (похідна) від'ємна,")
print("ґрунт втрачає вологу. Систему автоматичного поливу необхідно вмикати, коли")
print("значення похідної перевищує допустимий поріг швидкості висихання, щоб")
print("компенсувати втрати до того, як вологість M(t) впаде нижче критичного мінімуму.")

t_vals = [i * 0.1 for i in range(201)] 
M_vals = [M(t) for t in t_vals]

plt.figure(figsize=(10, 5))
plt.plot(t_vals, M_vals, label='M(t) = 50*exp(-0.1t) + 5*sin(t)')
plt.title('Модель вологості ґрунту M(t)')
plt.xlabel('t (Час)')
plt.ylabel('M(t) (Вологість)')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.loglog(valid_hs, errors, marker='o', linestyle='-')
plt.title('Залежність похибки чисельного диференціювання R від кроку h')
plt.xlabel('Крок h (логарифмічна шкала)')
plt.ylabel('Похибка R (логарифмічна шкала)')
plt.grid(True, which="both", ls="--")
plt.gca().invert_xaxis() 
plt.show()