#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <time.h>  



int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s input.jpg output.jpg\n", argv[0]);
        return 1;
    }

    int width, height, channels;
    unsigned char *input_image = stbi_load(argv[1], &width, &height, &channels, 3);
    if (!input_image) {
        fprintf(stderr, "Error loading image.\n");
        return 1;
    }

    size_t img_size = width * height;
    unsigned char *gray_image = (unsigned char *)malloc(img_size);
    clock_t start = clock();
    #pragma acc data copyin(input_image[0:img_size*3]) copyout(gray_image[0:img_size])
    {
        #pragma acc parallel loop
        for (int i = 0; i < img_size; i++) {
            int r = input_image[3*i + 0];
            int g = input_image[3*i + 1];
            int b = input_image[3*i + 2];
            gray_image[i] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
        }
    }
    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("OpenACC Execution Time: %.6f seconds\n", time_taken);
    stbi_write_jpg(argv[2], width, height, 1, gray_image, 100);
    stbi_image_free(input_image);
    free(gray_image);
    return 0;
}
