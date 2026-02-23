#include <stdio.h>
#include <math.h>

double X[1000];
double Y[1000];

int readdata() {
    FILE *file = fopen("in.txt", "rt");
    if (file == NULL) return 0;
    int i = 0;
    while(fscanf(file, "%le %le", &X[i], &Y[i]) != EOF) {
        i++;
    }
    fclose(file);
    return i;
}

double wkx(int k, double x) {
    double p = 1.0;
    for (int i = 0; i <= k; i++) {
        p = p * (x - X[i]);
    }
    return p;
}

double rr(int k) {
    double S = 0;
    for (int i = 0; i <= k; i++) {
        double p = 1.0;
        for (int j = 0; j <= k; j++) {
            if (j != i) {
                p = p * (X[i] - X[j]);
            }
        }
        S += Y[i] / p;
    }
    return S;
}

double Nn(double x, int N) {
    double S = Y[0];
    for (int k = 1; k < N; k++) {
        S = S + wkx(k - 1, x) * rr(k);
    }
    return S;
}

int main() {
    int N = readdata();
    if (N == 0) return 1;
    
    double target = 6000.0;
    double result = Nn(target, N);
    
    printf("N = %d\n", N);
    printf("Prediction for n=6000: %f ms\n", result);
    
    FILE *out = fopen("results.txt", "wt");
    for(double x = X[0]; x <= X[N-1]; x += 100) {
        fprintf(out, "%f\t%f\n", x, Nn(x, N));
    }
    fclose(out);
    
    return 0;
}