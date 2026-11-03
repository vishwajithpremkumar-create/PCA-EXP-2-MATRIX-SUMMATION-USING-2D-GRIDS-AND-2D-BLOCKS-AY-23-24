# PCA-EXP-2-Matrix-Summation-using-2D-Grids-and-2D-Blocks-AY-23-24

<h3>ENTER YOUR NAME</h3> VISHWAJITH P
<h3>ENTER YOUR REGISTER NO</h3> 212225220122
<h3>EX. NO</h3> 2
<h3>DATE</h3> 03/11/26
<h1> <align=center> MATRIX SUMMATION WITH A 2D GRID AND 2D BLOCKS </h3>
i.  Use the file sumMatrixOnGPU-2D-grid-2D-block.cu
ii. Matrix summation with a 2D grid and 2D blocks. Adapt it to integer matrix addition. Find the best execution configuration. </h3>

## AIM:
To perform  matrix summation with a 2D grid and 2D blocks and adapting it to integer matrix addition.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler




## PROCEDURE:

1.	Initialize the data: Generate random data for two input arrays using the initialData function.
2.	Perform the sum on the host: Use the sumMatrixOnHost function to calculate the sum of the two input arrays on the host (CPU) for later verification of the GPU results.
3.	Allocate memory on the device: Allocate memory on the GPU for the two input arrays and the output array using cudaMalloc.
4.	Transfer data from the host to the device: Copy the input arrays from the host to the device using cudaMemcpy.
5.	Set up the execution configuration: Define the size of the grid and blocks. Each block contains multiple threads, and the grid contains multiple blocks. The total number of threads is equal to the size of the grid times the size of the block.
6.	Perform the sum on the device: Launch the sumMatrixOnGPU2D kernel on the GPU. This kernel function calculates the sum of the two input arrays on the device (GPU).
7.	Synchronize the device: Use cudaDeviceSynchronize to ensure that the device has finished all tasks before proceeding.
8.	Transfer data from the device to the host: Copy the output array from the device back to the host using cudaMemcpy.
9.	Check the results: Use the checkResult function to verify that the output array calculated on the GPU matches the output array calculated on the host.
10.	Free the device memory: Deallocate the memory that was previously allocated on the GPU using cudaFree.
11.	Free the host memory: Deallocate the memory that was previously allocated on the host.
12.	Reset the device: Reset the device using cudaDeviceReset to ensure that all resources are cleaned up before the program exits.

## PROGRAM:
```
%%cuda
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <sys/time.h>

#define CHECK(call) \
{ \
    cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        fprintf(stderr, "CUDA Error: %s\n", \
                cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
}

double seconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

void initialData(int *ip, const int size)
{
    for (int i = 0; i < size; i++)
    {
        ip[i] = rand() % 256;
    }
}

void sumMatrixOnHost(int *A, int *B, int *C,
                     const int nx, const int ny)
{
    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            int idx = iy * nx + ix;
            C[idx] = A[idx] + B[idx];
        }
    }
}

void checkResult(int *hostRef, int *gpuRef, const int N)
{
    bool match = true;

    for (int i = 0; i < N; i++)
    {
        if (hostRef[i] != gpuRef[i])
        {
            match = false;
            printf("Mismatch at %d: Host=%d GPU=%d\n",
                   i, hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (match)
        printf("Arrays match successfully!\n");
    else
        printf("Arrays do not match!\n");
}

// CUDA 2D matrix addition kernel
__global__ void sumMatrixOnGPU2D(
    int *A, int *B, int *C, int NX, int NY)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int idx = iy * NX + ix;

    if (ix < NX && iy < NY)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    printf("CUDA Matrix Addition Starting...\n");

    // Select CUDA device
    int dev = 0;
    cudaDeviceProp deviceProp;

    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    CHECK(cudaSetDevice(dev));

    printf("Using Device %d: %s\n", dev, deviceProp.name);

    // Matrix dimensions
    int nx = 1024;
    int ny = 1024;

    int nxy = nx * ny;
    size_t nBytes = (size_t)nxy * sizeof(int);

    printf("Matrix size: %d x %d\n", nx, ny);

    // Allocate host memory
    int *h_A = (int *)malloc(nBytes);
    int *h_B = (int *)malloc(nBytes);
    int *hostRef = (int *)malloc(nBytes);
    int *gpuRef = (int *)malloc(nBytes);

    if (!h_A || !h_B || !hostRef || !gpuRef)
    {
        printf("Host memory allocation failed!\n");
        return EXIT_FAILURE;
    }

    // Initialize input matrices
    srand(123);

    double iStart = seconds();

    initialData(h_A, nxy);
    initialData(h_B, nxy);

    double iElaps = seconds() - iStart;

    printf("Matrix initialization time: %f sec\n", iElaps);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // CPU matrix addition
    iStart = seconds();

    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);

    iElaps = seconds() - iStart;

    printf("CPU matrix addition time: %f sec\n", iElaps);

    // Allocate GPU memory
    int *d_MatA, *d_MatB, *d_MatC;

    CHECK(cudaMalloc((void **)&d_MatA, nBytes));
    CHECK(cudaMalloc((void **)&d_MatB, nBytes));
    CHECK(cudaMalloc((void **)&d_MatC, nBytes));

    // Copy matrices from host to device
    CHECK(cudaMemcpy(d_MatA, h_A, nBytes,
                     cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(d_MatB, h_B, nBytes,
                     cudaMemcpyHostToDevice));

    // Configure CUDA grid and block
    dim3 block(16, 16);

    dim3 grid(
        (nx + block.x - 1) / block.x,
        (ny + block.y - 1) / block.y
    );

    // Execute CUDA kernel
    iStart = seconds();

    sumMatrixOnGPU2D<<<grid, block>>>(
        d_MatA, d_MatB, d_MatC, nx, ny
    );

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    iElaps = seconds() - iStart;

    printf("GPU matrix addition time: %f sec\n", iElaps);

    // Copy result from GPU to host
    CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes,
                     cudaMemcpyDeviceToHost));

    // Verify results
    checkResult(hostRef, gpuRef, nxy);

    // Free GPU memory
    CHECK(cudaFree(d_MatA));
    CHECK(cudaFree(d_MatB));
    CHECK(cudaFree(d_MatC));

    // Free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // Reset CUDA device
    CHECK(cudaDeviceReset());

    printf("CUDA Matrix Addition Completed!\n");

    return 0;
}
```
## OUTPUT:
<img width="597" height="213" alt="image" src="https://github.com/user-attachments/assets/85ec43b3-5be2-4421-8f3d-9ae2219991b9" />

## RESULT:
The host took 0.003556 seconds to complete it’s computation, while the GPU outperforms the host and completes the computation in 0.000627 seconds. Therefore, float variables in the GPU will result in the best possible result. Thus, matrix summation using 2D grids and 2D blocks has been performed successfully.
