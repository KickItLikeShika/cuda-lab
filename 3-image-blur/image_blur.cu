#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const int BLUR_SIZE = 10;

__global__ void blurKernel(unsigned char* d_in, unsigned char* d_out, int width, int height, int channels) {    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        for (int c = 0; c < channels; c++) {
            int pixVal = 0;
            int pixels = 0;

            // get average of the surrounding BLUR_SIZE x BLUR_SIZE box
            for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; blurRow++) {
                for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; blurCol++) {
                    int curRow = row + blurRow;
                    int curCol = col + blurCol;

                    // make sure we have a valid image pixel
                    if (curCol >= 0 && curCol < width && curRow >= 0 && curRow < height) {
                        pixVal += d_in[(curRow * width + curCol) * channels + c];
                        pixels++;
                    }
                }
            }

            d_out[(row * width + col) * channels + c] = (unsigned char)(pixVal / pixels);
        }
    }
}

void blurImage(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    size_t numPixels = width * height * channels;

    unsigned char *d_in, *d_out;
    cudaMalloc((void **) &d_in, numPixels);
    cudaMalloc((void **) &d_out, numPixels);

    cudaMemcpy(d_in, input, numPixels, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    blurKernel<<<gridSize, blockSize>>>(d_in, d_out, width, height, channels);

    cudaMemcpy(output, d_out, numPixels, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

int main() {
    int width, height, channels;

    // load the image
    unsigned char *image_data = stbi_load("example.jpg", &width, &height, &channels, 0);

    if (image_data == NULL) {
        fprintf(stderr, "Error: Could not open input image.\n");
        return -1;
    }

    printf("width: %d\n", width);
    printf("height: %d\n", height);
    printf("channels: %d\n", channels);

    // allocate memory for the output blurred image
    unsigned char *blurred_image = (unsigned char *)malloc(width * height * channels);

    // blur the image
    blurImage(image_data, blurred_image, width, height, channels);

    // save the blurred image
    stbi_write_jpg("output.jpg", width, height, channels, blurred_image, 100);

    // free the allocated memory
    stbi_image_free(image_data);
    free(blurred_image);

    return 0;
}
