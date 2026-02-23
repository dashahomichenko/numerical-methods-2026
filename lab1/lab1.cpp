#include <stdio.h>
#include <math.h>

double f(double x) {
    return sin(x);
}

int main(int argc, char **argv) {
    int N = 50, i = 0;
    double x0 = 0.0, xn = 1.0;
    double x[51], y[51], h[51]; // N+1
    double hh = (xn - x0) / (N);

    for (i = 0; i <= N; i++) {
        x[i] = x0 + (double)i * hh;
        h[i] = hh;
        y[i] = f(x[i]);
    }

    FILE *finput = fopen("input.txt", "w");
    for (i = 0; i <= N; i++) {
        fprintf(finput, "%i\t%e\t%e\t%e\n", i, x[i], y[i], h[i]);
    }
    printf("DONE");
    fclose(finput);
    return 0;
}