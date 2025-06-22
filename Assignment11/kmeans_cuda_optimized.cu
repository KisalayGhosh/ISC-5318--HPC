// Optimized CUDA K-Means for large k > 256 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

extern "C" {
    #define STB_IMAGE_IMPLEMENTATION
    #define STB_IMAGE_WRITE_IMPLEMENTATION
    #include "stb_image.h"
    #include "stb_image_write.h"
}

#define BLOCK_SIZE 256
#define MAX_K 1024

typedef struct {
    float r, g, b;
} Color;

__device__ float color_distance(Color a, Color b) {
    return sqrtf((a.r - b.r) * (a.r - b.r) +
                 (a.g - b.g) * (a.g - b.g) +
                 (a.b - b.b) * (a.b - b.b));
}

__global__ void assign_clusters_shared(Color* pixels, Color* centroids, int* labels,
                                       Color* sums, int* counts, int num_pixels, int k) {
    __shared__ Color shared_centroids[MAX_K];
    int tid = threadIdx.x;
    for (int i = tid; i < k; i += blockDim.x) {
        shared_centroids[i] = centroids[i];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= num_pixels) return;

    float min_dist = 1e20f;
    int best_k = 0;

    for (int i = 0; i < k; i++) {
        float dist = color_distance(pixels[idx], shared_centroids[i]);
        if (dist < min_dist) {
            min_dist = dist;
            best_k = i;
        }
    }

    labels[idx] = best_k;
    atomicAdd(&sums[best_k].r, pixels[idx].r);
    atomicAdd(&sums[best_k].g, pixels[idx].g);
    atomicAdd(&sums[best_k].b, pixels[idx].b);
    atomicAdd(&counts[best_k], 1);
}

__global__ void update_centroids(Color* centroids, Color* sums, int* counts, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k) return;

    if (counts[idx] > 0) {
        centroids[idx].r = sums[idx].r / counts[idx];
        centroids[idx].g = sums[idx].g / counts[idx];
        centroids[idx].b = sums[idx].b / counts[idx];
    }
}

__global__ void color_pixels(Color* pixels, Color* centroids, int* labels, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;
    int label = labels[idx];
    pixels[idx] = centroids[label];
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        printf("Usage: %s input.jpg output.png k iterations\n", argv[0]);
        return 1;
    }

    int width, height, channels;
    unsigned char* img_data = stbi_load(argv[1], &width, &height, &channels, 3);
    if (!img_data) {
        fprintf(stderr, "Failed to load image.\n");
        return 1;
    }

    int num_pixels = width * height;
    int k = atoi(argv[3]);
    int iterations = atoi(argv[4]);
    if (k > MAX_K) {
        fprintf(stderr, "Error: k exceeds MAX_K (%d).\n", MAX_K);
        return 1;
    }

    Color* h_pixels = (Color*)malloc(sizeof(Color) * num_pixels);
    Color* h_centroids = (Color*)malloc(sizeof(Color) * k);
    srand(time(NULL));

    for (int i = 0; i < num_pixels; i++) {
        h_pixels[i].r = img_data[i * 3];
        h_pixels[i].g = img_data[i * 3 + 1];
        h_pixels[i].b = img_data[i * 3 + 2];
    }

    bool* used = (bool*)calloc(num_pixels, sizeof(bool));
    for (int i = 0; i < k; i++) {
        int idx;
        do {
            idx = rand() % num_pixels;
        } while (used[idx]);
        used[idx] = true;
        h_centroids[i] = h_pixels[idx];
    }
    free(used);

    Color *d_pixels, *d_centroids, *d_sums;
    int *d_labels, *d_counts;
    cudaMalloc(&d_pixels, sizeof(Color) * num_pixels);
    cudaMalloc(&d_labels, sizeof(int) * num_pixels);
    cudaMalloc(&d_centroids, sizeof(Color) * k);
    cudaMalloc(&d_sums, sizeof(Color) * k);
    cudaMalloc(&d_counts, sizeof(int) * k);

    cudaMemcpy(d_pixels, h_pixels, sizeof(Color) * num_pixels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, sizeof(Color) * k, cudaMemcpyHostToDevice);

    int blocks = (num_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int k_blocks = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float elapsed;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int it = 0; it < iterations; it++) {
        cudaMemset(d_sums, 0, sizeof(Color) * k);
        cudaMemset(d_counts, 0, sizeof(int) * k);

        assign_clusters_shared<<<blocks, BLOCK_SIZE>>>(d_pixels, d_centroids, d_labels, d_sums, d_counts, num_pixels, k);
        update_centroids<<<k_blocks, BLOCK_SIZE>>>(d_centroids, d_sums, d_counts, k);
    }

    color_pixels<<<blocks, BLOCK_SIZE>>>(d_pixels, d_centroids, d_labels, num_pixels);
    cudaMemcpy(h_pixels, d_pixels, sizeof(Color) * num_pixels, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("CUDA Execution Time: %.4f ms\n", elapsed);

    unsigned char* segmented = (unsigned char*)malloc(sizeof(unsigned char) * 3 * num_pixels);
    for (int i = 0; i < num_pixels; i++) {
        segmented[i * 3]     = min(max((int)(h_pixels[i].r + 0.5f), 0), 255);
        segmented[i * 3 + 1] = min(max((int)(h_pixels[i].g + 0.5f), 0), 255);
        segmented[i * 3 + 2] = min(max((int)(h_pixels[i].b + 0.5f), 0), 255);
    }

    stbi_write_png(argv[2], width, height, 3, segmented, width * 3);

    free(segmented);
    free(h_pixels);
    free(h_centroids);
    stbi_image_free(img_data);
    cudaFree(d_pixels);
    cudaFree(d_labels);
    cudaFree(d_centroids);
    cudaFree(d_sums);
    cudaFree(d_counts);

    return 0;
}
