
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#define MASTER 0

/// Function to convert RGB to grayscale using the luminosity method
unsigned char rgb_to_grayscale(unsigned char r, unsigned char g, unsigned char b) {
    return (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
}

int main(int argc, char *argv[]) {
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc < 3) {
        if (rank == MASTER) {
            printf("Usage: %s input.jpg output.jpg\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    char *input_file = argv[1];
    char *output_file = argv[2];
    int width, height, channels;
    unsigned char *image = NULL, *gray_image = NULL, *local_data = NULL, *local_gray = NULL;

    double start_time, end_time; 

    if (rank == MASTER) {
        // Start  execution time
        start_time = MPI_Wtime();

        // Load the image in the master process
        image = stbi_load(input_file, &width, &height, &channels, 0);
        if (image == NULL) {
            printf("Error loading image\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // To Ensure the image has at least 3 channels (RGB)
        if (channels < 3) {
            printf("Image does not have enough channels\n");
            stbi_image_free(image);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Allocate memory for the grayscale output image
        gray_image = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    }

    // Broadcast image dimensions to all processes
    MPI_Bcast(&width, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    // number of rows each process will handle
    int rows_per_proc = height / num_procs;
    int remaining_rows = height % num_procs;
    int local_rows = (rank < remaining_rows) ? rows_per_proc + 1 : rows_per_proc;
    int local_size = local_rows * width * channels;

    
    local_data = (unsigned char *)malloc(local_size * sizeof(unsigned char));
    local_gray = (unsigned char *)malloc(local_rows * width * sizeof(unsigned char));

    //  scatter operation
    int *sendcounts = NULL, *displs = NULL;
    if (rank == MASTER) {
        sendcounts = (int *)malloc(num_procs * sizeof(int));
        displs = (int *)malloc(num_procs * sizeof(int));

        int offset = 0;
        for (int i = 0; i < num_procs; i++) {
            sendcounts[i] = ((i < remaining_rows) ? (rows_per_proc + 1) : rows_per_proc) * width * channels;
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    // Scatter image data from the master to all processes
    MPI_Scatterv(image, sendcounts, displs, MPI_UNSIGNED_CHAR, local_data, local_size, MPI_UNSIGNED_CHAR, MASTER, MPI_COMM_WORLD);

    // Convert the received portion to grayscale
    for (int i = 0; i < local_rows * width; i++) {
        local_gray[i] = rgb_to_grayscale(local_data[i * channels], local_data[i * channels + 1], local_data[i * channels + 2]);
    }

    // gather operation
    int *recvcounts = NULL, *recvdispls = NULL;
    if (rank == MASTER) {
        recvcounts = (int *)malloc(num_procs * sizeof(int));
        recvdispls = (int *)malloc(num_procs * sizeof(int));

        int offset = 0;
        for (int i = 0; i < num_procs; i++) {
            recvcounts[i] = ((i < remaining_rows) ? (rows_per_proc + 1) : rows_per_proc) * width;
            recvdispls[i] = offset;
            offset += recvcounts[i];
        }
    }

    // Gather the grayscale data from all processes
    MPI_Gatherv(local_gray, local_rows * width, MPI_UNSIGNED_CHAR, gray_image, recvcounts, recvdispls, MPI_UNSIGNED_CHAR, MASTER, MPI_COMM_WORLD);

    if (rank == MASTER) {
       
        stbi_write_jpg(output_file, width, height, 1, gray_image, 100);
        printf("Grayscale image saved to %s\n", output_file);

        // Stop execution time
        end_time = MPI_Wtime();
        printf("Execution Time: %f seconds\n", end_time - start_time);

        free(gray_image);
        free(sendcounts);
        free(displs);
        free(recvcounts);
        free(recvdispls);
    }

    free(local_data);
    free(local_gray);
    if (rank == MASTER) stbi_image_free(image);

    MPI_Finalize();
    return 0;
}