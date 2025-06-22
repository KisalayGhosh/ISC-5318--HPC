#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>

#define NX 100
#define NY 100
#define DX 0.01
#define DY 0.01
#define ALPHA 0.01
#define MAX_ITER 20000 
#define DT 0.001

void decompose1d(int n, int m, int i, int* s, int* e) {
    const int length = n / m;
    const int deficit = n % m;
    *s = i * length + (i < deficit ? i : deficit);
    *e = *s + length - 1 + (i < deficit ? 1 : 0);
    if (*e >= n || i == m - 1) *e = n - 1;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int numProcesses, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dims[2], coords[2];
    int periods[2] = {0, 0};
    MPI_Comm comm2d;
    MPI_Dims_create(numProcesses, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm2d);
    MPI_Cart_get(comm2d, 2, dims, periods, coords);

    int x0, x1, y0, y1;
    decompose1d(NX, dims[0], coords[0], &x0, &x1);
    decompose1d(NY, dims[1], coords[1], &y0, &y1);

    int left, right, down, up;
    MPI_Cart_shift(comm2d, 0, 1, &left, &right);
    MPI_Cart_shift(comm2d, 1, 1, &down, &up);

    if (left >= 0) x0--;
    if (right >= 0) x1++;
    if (down >= 0) y0--;
    if (up >= 0) y1++;
    int nx = x1 - x0 + 1;
    int ny = y1 - y0 + 1;

    double **u = new double*[nx];
    double **unew = new double*[nx];
    double *p = new double[nx * ny];
    double *pnew = new double[nx * ny];
    for (int i = 0; i < nx; ++i) {
        u[i] = &p[i * ny];
        unew[i] = &pnew[i * ny];
    }

    // Initialize with boundary conditions
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            u[i][j] = 0.0;

    //Dirichlet boundary conditions 
    for (int i = 0; i < nx; ++i) {
        int global_x = x0 + i;
        if (y1 == NY - 1) u[i][ny - 1] = 1.0; // Top boundary
    }
    for (int j = 0; j < ny; ++j) {
        int global_y = y0 + j;
        if (x0 == 0) u[0][j] = 1.0;           // Left boundary
        if (x1 == NX - 1) u[nx - 1][j] = 1.0; // Right boundary
    }


    MPI_Datatype xSlice, ySlice;
    MPI_Type_vector(nx, 1, ny, MPI_DOUBLE, &xSlice);
    MPI_Type_commit(&xSlice);
    MPI_Type_vector(ny, 1, 1, MPI_DOUBLE, &ySlice);
    MPI_Type_commit(&ySlice);

    double start_time = MPI_Wtime();

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        
        MPI_Sendrecv(&u[0][ny - 2], 1, xSlice, up, 0, &u[0][0], 1, xSlice, down, 0, comm2d, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&u[0][1], 1, xSlice, down, 0, &u[0][ny - 1], 1, xSlice, up, 0, comm2d, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&u[nx - 2][0], 1, ySlice, right, 0, &u[0][0], 1, ySlice, left, 0, comm2d, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&u[1][0], 1, ySlice, left, 0, &u[nx - 1][0], 1, ySlice, right, 0, comm2d, MPI_STATUS_IGNORE);

       /
        for (int i = 1; i < nx - 1; ++i) {
            for (int j = 1; j < ny - 1; ++j) {
                unew[i][j] = u[i][j] + DT * ALPHA * (
                    (u[i+1][j] - 2*u[i][j] + u[i-1][j]) / (DX*DX) +
                    (u[i][j+1] - 2*u[i][j] + u[i][j-1]) / (DY*DY)
                );
            }
        }

        
        double **tmp = u; u = unew; unew = tmp;

        
        if (iter % 5000 == 0 && rank == 0) {
            printf("Iteration %d complete.\n", iter);
        }
    }

    double end_time = MPI_Wtime();
    if (rank == 0) {
        printf("Execution Time: %.6f seconds\n", end_time - start_time);
    }

    
    char filename[64];
    sprintf(filename, "output_rank_%d.txt", rank);
    std::ofstream fout(filename);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j)
            fout << u[i][j] << " ";
        fout << "\n";
    }
    fout.close();

    delete[] p;
    delete[] pnew;
    delete[] u;
    delete[] unew;

    MPI_Type_free(&xSlice);
    MPI_Type_free(&ySlice);
    MPI_Finalize();
    return 0;
}
