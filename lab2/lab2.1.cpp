#include <stdio.h>

int main() {
    FILE *file1 = fopen("in.txt", "wt");
    if (file1 == NULL) return 1;
    
    double n[] = {1000, 2000, 4000, 8000, 16000};
    double t[] = {3, 5, 11, 28, 85};
    
    for(int i = 0; i < 5; i++) {
        fprintf(file1, "%le %le\n", n[i], t[i]);
    }
    
    fclose(file1);
    return 0;
}