import matplotlib.pyplot as plt

def read_results(filename):
    x, y = [], []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split()
            x.append(float(parts[0]))
            y.append(float(parts[1]))
    return x, y

# Дані Варіанта 1 для точок
orig_x = [1000, 2000, 4000, 8000, 16000]
orig_y = [3, 5, 11, 28, 85]

try:
    calc_x, calc_y = read_results("results.txt")
    plt.plot(calc_x, calc_y, label='Newton Interpolation', color='blue')
    plt.scatter(orig_x, orig_y, color='red', label='Experimental points')
    plt.xlabel('n (Data size)')
    plt.ylabel('t (ms)')
    plt.title('Execution Time Prediction (Option 1)')
    plt.legend()
    plt.grid(True)
    plt.show()
except FileNotFoundError:
    print("Error: results.txt not found. Run C++ code first.")