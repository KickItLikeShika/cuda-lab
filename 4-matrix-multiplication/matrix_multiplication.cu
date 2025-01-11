#include <stdio.h>


__global__ void matMulKernel(float* a, float* b, float* c, int n, int m) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if ((row < n) && (col < m)) {
        float val = 0;
        for int (k = 0; k < n, k++) {
            val += a[row * n + k] * b[k * m + col];
        }
        c[row * m + col] = val;
    }
}

void matMul(float* a_h, float* b_h, float* c_h, int n, int m) {
    // allocate memory on the host
    int size = sizeof(float) * n;
    float *a_d, *b_d, *c_d;
    
    // allocate memory on the device
    cudaMalloc((void **) &a_d, size);
    cudaMalloc((void **) &b_d, size);
    cudaMalloc((void **) &c_d, size);

    // move data from host to device
    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);
    

    // launch the grid, ceil(n/256) blocks of 256 threads each
    // and execute on device
    
}


int main() {
    // rows and columns of the matrices
    // int n, m;
    // float *a;
    // float *b;

    // define a 2d matrix a and b
    int n = 2;
    int m = 2;
    float a[2][2] = {{1, 2}, {3, 4}};
    float b[2][2] = {{1, 2}, {3, 4}};
    float c[2][2];

    // // take input for both matrices
    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < m; j++) {
    //         std::cin >> a[i * m + j];
    //     }
    // }

    


    return 0;
}