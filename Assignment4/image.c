#include <stdio.h>      
#include <stdlib.h>     // For memory allocation and general utilities
#include <omp.h>        // OpenMP library for parallel programming
#include <math.h>       
#include <string.h>     

// Including the STB libraries for image loading and writing
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Maximum number of iterations for the k-means algorithm
#define MAX_ITER 20

// Structure to represent a color (pixel) with R, G, B components
typedef struct {
    float r, g, b;
} Color;

// Function to calculate the Euclidean distance between two colors
float color_distance(Color c1, Color c2) {
    return sqrt((c1.r - c2.r) * (c1.r - c2.r) + 
                (c1.g - c2.g) * (c1.g - c2.g) + 
                (c1.b - c2.b) * (c1.b - c2.b));
}

// Function to initialize the cluster centroids by selecting evenly spaced pixels
void initialize_clusters(Color *centroids, Color *pixels, int num_pixels, int k) {
    int step = num_pixels / k;
    for (int i = 0; i < k; i++) {
        centroids[i] = pixels[i * step];  // Select every (num_pixels/k)-th pixel as a centroid
    }
}

// Function to assign each pixel to the nearest cluster centroid
void assign_clusters(int *labels, Color *pixels, Color *centroids, int num_pixels, int k, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)  
    for (int i = 0; i < num_pixels; i++) {
        float min_dist = INFINITY;  // Initialize minimum distance to a large value
        int cluster_index = 0;
        for (int j = 0; j < k; j++) {  // Calculate distance to each centroid
            float dist = color_distance(pixels[i], centroids[j]);
            if (dist < min_dist) {
                min_dist = dist;  // Update minimum distance
                cluster_index = j;  // Update the cluster label
            }
        }
        labels[i] = cluster_index;  // Assign the pixel to the closest cluster
    }
}

// Function to compute new centroids by averaging the colors of pixels in each cluster
void compute_new_centroids(Color *centroids, Color *pixels, int *labels, int num_pixels, int k, int num_threads) {
    Color *sum = calloc(k, sizeof(Color));    // Array to store the sum of pixel colors for each cluster
    int *count = calloc(k, sizeof(int));      // Array to store the number of pixels in each cluster

    #pragma omp parallel num_threads(num_threads)  // Parallel section
    {
        Color *local_sum = calloc(k, sizeof(Color));   // Local arrays for each thread
        int *local_count = calloc(k, sizeof(int));

        #pragma omp for  // Distribute the loop across threads
        for (int i = 0; i < num_pixels; i++) {
            int cluster = labels[i];  // Get the cluster label of the pixel
            local_sum[cluster].r += pixels[i].r;
            local_sum[cluster].g += pixels[i].g;
            local_sum[cluster].b += pixels[i].b;
            local_count[cluster]++;  // Increment the count for the cluster
        }

        #pragma omp critical  // Critical section to update the shared arrays
        {
            for (int i = 0; i < k; i++) {
                sum[i].r += local_sum[i].r;
                sum[i].g += local_sum[i].g;
                sum[i].b += local_sum[i].b;
                count[i] += local_count[i];
            }
        }

        free(local_sum);  
        free(local_count);
    }

    // Calculate the new centroids by averaging the sums
    for (int i = 0; i < k; i++) {
        if (count[i] > 0) {
            centroids[i].r = sum[i].r / count[i];
            centroids[i].g = sum[i].g / count[i];
            centroids[i].b = sum[i].b / count[i];
        }
    }

    free(sum);   // Free the allocated memory
    free(count);
}

// Main function to read image, perform k-means segmentation, and write output image
int main(int argc, char *argv[]) {
    if (argc < 6) {  
        printf("Usage: %s input.jpg output.jpg k num_threads iterations\n", argv[0]);
        return 1;
    }

    // Parse command line arguments
    char *input_file = argv[1];
    char *output_file = argv[2];
    int k = atoi(argv[3]);
    int num_threads = atoi(argv[4]);
    int iterations = atoi(argv[5]);

    // Load the input image using stb_image
    int width, height, channels;
    unsigned char *image = stbi_load(input_file, &width, &height, &channels, 0);
    if (image == NULL) {
        printf("Error in loading the image\n");
        return 1;
    }

    int num_pixels = width * height;  // Total number of pixels in the image
    Color *pixels = malloc(num_pixels * sizeof(Color));  // Allocate memory for pixel data

    // Convert image data to Color structure
    for (int i = 0; i < num_pixels; i++) {
        pixels[i].r = image[channels * i];
        pixels[i].g = image[channels * i + 1];
        pixels[i].b = image[channels * i + 2];
    }

    Color *centroids = malloc(k * sizeof(Color));  // Allocate memory for centroids
    int *labels = malloc(num_pixels * sizeof(int));  // Allocate memory for pixel labels

    initialize_clusters(centroids, pixels, num_pixels, k);  // Initialize centroids

    double start_time = omp_get_wtime();  // Record start time
    for (int iter = 0; iter < iterations; iter++) {  // Perform k-means iterations
        assign_clusters(labels, pixels, centroids, num_pixels, k, num_threads);
        compute_new_centroids(centroids, pixels, labels, num_pixels, k, num_threads);
    }
    double end_time = omp_get_wtime();  // Record end time

    
    printf("Execution time with k=%d, threads=%d, iterations=%d: %f seconds\n", k, num_threads, iterations, end_time - start_time);

    // Assign the final centroid colors to each pixel in the image
    for (int i = 0; i < num_pixels; i++) {
        image[channels * i] = (unsigned char)centroids[labels[i]].r;
        image[channels * i + 1] = (unsigned char)centroids[labels[i]].g;
        image[channels * i + 2] = (unsigned char)centroids[labels[i]].b;
    }

    // Save the segmented image to output file
    stbi_write_jpg(output_file, width, height, channels, image, 100);

    printf("Image segmentation completed with k=%d, threads=%d.\n", k, num_threads);

    // Free all allocated memory
    free(pixels);
    free(centroids);
    free(labels);
    stbi_image_free(image);

    return 0;  
}
