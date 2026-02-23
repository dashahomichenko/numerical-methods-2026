#include <stdio.h>
#include <math.h>
#include <stdlib.h>

double f(double x) {
    return sin(x);
}

void progonka(double *y, double *h, const int N, double *c) {
    int i;
    double *alfa = (double*)malloc((N+1)*sizeof(double));
    double *beta = (double*)malloc((N+1)*sizeof(double));
    double *hamma = (double*)malloc((N+1)*sizeof(double));
    double *delta = (double*)malloc((N+1)*sizeof(double));
    double *A = (double*)malloc((N+1)*sizeof(double));
    double *B = (double*)malloc((N+1)*sizeof(double));

    alfa[1] = hamma[1] = delta[1] = 0.0;
    beta[1] = 1.0;

    for (i = 2; i <= N; i++) {
        alfa[i] = h[i-1];
        beta[i] = 2 * (h[i-1] + h[i]);
        hamma[i] = h[i];
        delta[i] = 3 * (((y[i] - y[i-1]) / h[i]) - ((y[i-1] - y[i-2]) / h[i-1]));
    }
    hamma[N] = 0.0;

    A[1] = -hamma[1] / beta[1];
    B[1] = delta[1] / beta[1];
    for (i = 2; i <= N - 1; i++) {
        A[i] = -hamma[i] / (alfa[i] * A[i-1] + beta[i]);
        B[i] = (delta[i] - alfa[i] * B[i-1]) / (alfa[i] * A[i-1] + beta[i]);
    }
    c[N] = (delta[N] - alfa[N] * B[N-1]) / (alfa[N] * A[N-1] + beta[N]);
    for (i = N; i > 1; i--) {
        c[i-1] = A[i-1] * c[i] + B[i-1];
    }
    free(alfa); free(beta); free(hamma); free(delta); free(A); free(B);
}

int main() {
    FILE *fdata = fopen("input.txt", "r");
    if(!fdata) return 1;
    int N = 0, i = 0, j = 0;
    char one_char;
    while ((one_char = fgetc(fdata)) != EOF)
        if (one_char == '\n') ++N;
    N = N - 1;
    rewind(fdata);

    double *x = (double*)malloc((N+1)*sizeof(double));
    double *y = (double*)malloc((N+1)*sizeof(double));
    double *h = (double*)malloc((N+1)*sizeof(double));
    double *a = (double*)malloc((N+1)*sizeof(double));
    double *b = (double*)malloc((N+1)*sizeof(double));
    double *c = (double*)malloc((N+1)*sizeof(double));
    double *d = (double*)malloc((N+1)*sizeof(double));

    for (i = 0; i <= N; i++) {
        fscanf(fdata, "%i\t%le\t%le\t%le\n", &j, &x[i], &y[i], &h[i]);
    }
    fclose(fdata);

    progonka(y, h, N, c);

    for (i = 1; i < N; i++) {
        a[i] = y[i-1];
        b[i] = (y[i] - y[i-1]) / h[i] - (h[i] / 3.0) * (c[i+1] + 2.0 * c[i]);
        d[i] = (c[i+1] - c[i]) / (3.0 * h[i]);
    }
    a[N] = y[N-1];
    b[N] = (y[N] - y[N-1]) / h[N] - (2.0 / 3.0) * h[N] * c[N];
    d[N] = -c[N] / (3.0 * h[N]);

    FILE *foutput = fopen("output.txt", "w");
    int k = 1;
    int total_points = 20 * N;
    double hh = (x[N] - x[0]) / (double)total_points;

    for (i = 0; i <= 20 * (N - 1); i++) {
        double xm_i = x[0] + (double)i * hh;
        double ym_i = f(xm_i);
        double s = a[k] + b[k]*(xm_i - x[k-1]) + c[k]*pow(xm_i - x[k-1], 2) + d[k]*pow(xm_i - x[k-1], 3);
        fprintf(foutput, "%le\t%le\t%le\t%le\n", xm_i, ym_i, s, fabs(s - ym_i));
        if ((i != 0) && (i % 20 == 0)) k++;
    }
    fclose(foutput);
    return 0;
}