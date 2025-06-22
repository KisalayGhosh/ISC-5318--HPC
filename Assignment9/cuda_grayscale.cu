#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// CUDA kernel: 2D grid and block mapping for image pixels
// ===========================================
// Each thread processes one pixel using 2D indexing
// ===========================================
__global__ void grayscale_kernel(unsigned char *input, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        int i = idx * 3;  // 3 channels: RGB

        unsigned char r = input[i];
        unsigned char g = input[i + 1];
        unsigned char b = input[i + 2];
        // Apply luminosity formula for grayscale conversion
        output[idx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s input.jpg output.jpg\n", argv[0]);
        return 1;
    }

    char *input_file = argv[1];
    char *output_file = argv[2];

    int width, height, channels;

    // I avoided flip on load
    unsigned char *h_image = stbi_load(input_file, &width, &height, &channels, 3);
    if (!h_image) {
        printf("Failed to load image: %s\n", input_file);
        return 1;
    }

    size_t img_size = width * height * 3;
    size_t gray_size = width * height;
     // Allocate host memory for grayscale image
    unsigned char *h_gray = (unsigned char *)malloc(gray_size);
    if (!h_gray) {
        fprintf(stderr, "Host memory allocation failed!\n");
        stbi_image_free(h_image);
        return 1;
    }
    // Allocate GPU device memory for input and output
    unsigned char *d_image, *d_gray;
    cudaMalloc(&d_image, img_size);
    cudaMalloc(&d_gray, gray_size);
    cudaMemcpy(d_image, h_image, img_size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_gray, width, height);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_gray, d_gray, gray_size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUDA Execution Time: %.4f ms\n", milliseconds);

    // erroron flip on write thats why avoided
    if (!stbi_write_jpg(output_file, width, height, 1, h_gray, 100)) {
        printf("Failed to write image: %s\n", output_file);
    } else {
        printf("Grayscale image saved to %s\n", output_file);
    }
     // Free all memory allocations
    
    cudaFree(d_image);
    cudaFree(d_gray);
    stbi_image_free(h_image);
    free(h_gray);

    return 0;
}
