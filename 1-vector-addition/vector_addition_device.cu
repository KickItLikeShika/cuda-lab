#include <stdio.h>


__global__
void vecAddKernel(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // initiated grid will have blocks of same thread size, but threads in last block might not be used as vector size might be smaller,
    // so that's why we have this if conidtion
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void addVectors(float* a_h, float* b_h, float* c_h, int n) {
    int size = n * sizeof(float);
    float *a_d, *b_d, *c_d;

    // allocate memory on gpu/device for the new vectors
    cudaMalloc((void **) &a_d, size);
    cudaMalloc((void **) &b_d, size);
    cudaMalloc((void **) &c_d, size);

    // move vectors from cpu/host to gpu/device
    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

    // launch the grid, ceil(n/256) blocks of 256 threads each
    // and execute on device
    vecAddKernel<<<ceil(n/256.0), 256>>>(a_d, b_d, c_d, n);

    // move vector from gpu to cpu
    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

    // free gpu/device memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
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
