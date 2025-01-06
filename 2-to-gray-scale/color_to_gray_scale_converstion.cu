#include <iostream>
#include <fstream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"  // Include the stb_image library
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


__global__ void rgbToGrayKernel(unsigned char* d_in, unsigned char* d_out, int width, int height, int channels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        // get 1d offset for the grayscale image
        int grayOffset = row * width + col;
        
        // one can think of the RGB image having CHANNEL
        // times more columns than the gray scale image
        int rgbOffset = grayOffset * channels;

        unsigned char r = d_in[rgbOffset]; // red value
        unsigned char g = d_in[rgbOffset + 1]; // green value
        unsigned char b = d_in[rgbOffset + 2]; // blue value

        d_out[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

void rgbToGray(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    size_t numPixels = width * height;

    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in, numPixels * channels);
    cudaMalloc(&d_out, numPixels);

    cudaMemcpy(d_in, input, numPixels * channels, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    rgbToGrayKernel<<<gridSize, blockSize>>>(d_in, d_out, width, height, channels);

    cudaMemcpy(output, d_out, numPixels, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

int main() {
    int width, height, channels;

    // load the image
    unsigned char *image_data = stbi_load("example.jpg", &width, &height, &channels, 0);

    if (image_data == nullptr) {
        std::cerr << "Error: Could not open input image." << std::endl;
        return -1;
    }

    // allocate memory for the output grayscale image
    unsigned char *gray_image = (unsigned char*)malloc(width * height);

    // convert the image to grayscale
    rgbToGray(image_data, gray_image, width, height, channels);

    // save the grayscale image
    stbi_write_jpg("output.jpg", width, height, 1, gray_image, 100);

    // free the allocated memory
    stbi_image_free(image_data);
    cudaFree(gray_image);

    return 0;
}
