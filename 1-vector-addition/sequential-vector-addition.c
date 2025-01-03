#include <stdio.h>


void addVectors(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {

    int n = 4;
    float a[4] = {1, 2, 3, 4};
    float b[4] = {1, 2, 3, 4};
    float c[4];

    addVectors(a, b, c, n);
    for (int i = 0; i < n; i++) {
        printf("%f\n", c[i]);
    }

    return 0;
}
