#include <stdio.h>

__global__ void matMulKernel(float* a, float* b, float* c, int n, int m, int p) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // p: columns of B, columns of C
    int row = blockIdx.y * blockDim.y + threadIdx.y; // n: rows of A, rows of C

    if ((row < n) && (col < p)) {
        float val = 0;
        for (int k = 0; k < m; k++) {
            val += a[row * m + k] * b[k * p + col];
        }
        c[row * p + col] = val;
    }
}

void matMul(float* a_h, float* b_h, float* c_h, int n, int m, int p) {
    // allocate memory on the host
    int sizeA = sizeof(float) * n * m; // Size of matrix A
    int sizeB = sizeof(float) * m * p; // Size of matrix B
    int sizeC = sizeof(float) * n * p; // Size of matrix C

    float *a_d, *b_d, *c_d;

    // allocate memory on the device
    cudaMalloc((void **) &a_d, sizeA);
    cudaMalloc((void **) &b_d, sizeB);
    cudaMalloc((void **) &c_d, sizeC);

    // move data from host to device
    cudaMemcpy(a_d, a_h, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, sizeB, cudaMemcpyHostToDevice);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((m + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);

    matMulKernel<<<gridSize, blockSize>>>(a_d, b_d, c_d, n, m, p);

    cudaMemcpy(c_h, c_d, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}


int main() {
    // define a 2d matrix a and b
    int n = 2; // Rows of A
    int m = 2; // Columns of A, Rows of B
    int p = 2; // Columns of B

    float a[2][2] = {{1, 2}, {3, 4}};
    float b[2][2] = {{1, 2}, {3, 4}};
    float c[2][2];

    // matrix multiplication
    matMul((float*)a, (float*)b, (float*)c, n, m, p);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%.1f \n", c[i][j]);
        }
    }

    return 0;
}
