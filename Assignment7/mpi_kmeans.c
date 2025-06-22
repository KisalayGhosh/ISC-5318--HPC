#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

typedef struct {
    float r, g, b;
} Color;

// Compute Euclidean distance between two colors
float color_distance(Color c1, Color c2) {
    return sqrt((c1.r - c2.r) * (c1.r - c2.r) +
                (c1.g - c2.g) * (c1.g - c2.g) +
                (c1.b - c2.b) * (c1.b - c2.b));
}

// Initialize k cluster centroids
void initialize_clusters(Color *centroids, Color *pixels, int num_pixels, int k) {
    int step = num_pixels / k;
    for (int i = 0; i < k; i++) {
        centroids[i] = pixels[i * step];
    }
}

// Assign pixels to the nearest centroid
void assign_clusters(int *labels, Color *pixels, Color *centroids, int num_pixels, int k) {
    for (int i = 0; i < num_pixels; i++) {
        float min_dist = INFINITY;
        int cluster_index = 0;
        for (int j = 0; j < k; j++) {
            float dist = color_distance(pixels[i], centroids[j]);
            if (dist < min_dist) {
                min_dist = dist;
                cluster_index = j;
            }
        }
        labels[i] = cluster_index;
    }
}

// Compute new centroids (parallel reduction)
void compute_new_centroids(Color *centroids, Color *pixels, int *labels, int num_pixels, int k, int rank) {
    Color *sum = calloc(k, sizeof(Color));
    int *count = calloc(k, sizeof(int));

    for (int i = 0; i < num_pixels; i++) {
        int cluster = labels[i];
        sum[cluster].r += pixels[i].r;
        sum[cluster].g += pixels[i].g;
        sum[cluster].b += pixels[i].b;
        count[cluster]++;
    }

    // Reduce sum and count to rank 0 (Master)
    Color *global_sum = NULL;
    int *global_count = NULL;
    if (rank == 0) {
        global_sum = calloc(k, sizeof(Color));
        global_count = calloc(k, sizeof(int));
    }

    MPI_Reduce(sum, global_sum, k * 3, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(count, global_count, k, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < k; i++) {
            if (global_count[i] > 0) {
                centroids[i].r = global_sum[i].r / global_count[i];
                centroids[i].g = global_sum[i].g / global_count[i];
                centroids[i].b = global_sum[i].b / global_count[i];
            }
        }
        free(global_sum);
        free(global_count);
    }

    MPI_Bcast(centroids, k * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);

    free(sum);
    free(count);
}

int main(int argc, char *argv[]) {
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc < 6) {
        if (rank == 0) {
            printf("Usage: %s input.jpg output.jpg k num_iterations num_procs\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    char *input_file = argv[1];
    char *output_file = argv[2];
    int k = atoi(argv[3]);
    int iterations = atoi(argv[4]);

    int width, height, channels;
    Color *pixels = NULL;
    unsigned char *image = NULL;

    if (rank == 0) {
        image = stbi_load(input_file, &width, &height, &channels, 0);
        if (image == NULL) {
            printf("Error loading image.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int num_pixels = width * height;
        pixels = malloc(num_pixels * sizeof(Color));

        for (int i = 0; i < num_pixels; i++) {
            pixels[i].r = image[channels * i];
            pixels[i].g = image[channels * i + 1];
            pixels[i].b = image[channels * i + 2];
        }

        MPI_Bcast(&num_pixels, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        int num_pixels;
        MPI_Bcast(&num_pixels, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
        pixels = malloc(num_pixels * sizeof(Color));
    }

    MPI_Bcast(pixels, width * height * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);

    Color *centroids = malloc(k * sizeof(Color));
    int *labels = malloc(width * height * sizeof(int));

    if (rank == 0) {
        initialize_clusters(centroids, pixels, width * height, k);
    }

    MPI_Bcast(centroids, k * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();

    for (int iter = 0; iter < iterations; iter++) {
        MPI_Scatter(pixels, width * height / num_procs * 3, MPI_FLOAT, pixels, width * height / num_procs * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);
        assign_clusters(labels, pixels, centroids, width * height / num_procs, k);

        compute_new_centroids(centroids, pixels, labels, width * height / num_procs, k, rank);
    }

    double end_time = MPI_Wtime();
    if (rank == 0) {
        printf("Execution Time: %f seconds\n", end_time - start_time);
    }

    MPI_Gather(labels, width * height / num_procs, MPI_INT, labels, width * height / num_procs, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < width * height; i++) {
            image[channels * i] = (unsigned char)centroids[labels[i]].r;
            image[channels * i + 1] = (unsigned char)centroids[labels[i]].g;
            image[channels * i + 2] = (unsigned char)centroids[labels[i]].b;
        }
        stbi_write_jpg(output_file, width, height, channels, image, 100);
    }

    free(pixels);
    free(centroids);
    free(labels);
    if (rank == 0) stbi_image_free(image);
    MPI_Finalize();
    return 0;
}
