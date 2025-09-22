#include <cuda.h>
#include <curand_kernel.h>

#define N 1024

__global__ void addVectors(float *A, float *B, float *C) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

__local__ float tempStorage[N];

__global__ void addVectorsWithSharedMemory(float *A, float *B, float *C) {
    int i = threadIdx.x;
    float localSum = 0.0f;

    if (i < N) {
        tempStorage[i] = A[i];
    }

    int blockSize = blockDim.x;
    int offset = i - ((i / blockSize) * blockSize);

    __syncthreads();

    for (int j = offset; j < N; j += blockSize) {
        if ((threadIdx.x == 0 || threadIdx.x == blockSize) && j + blockSize <= N) {
            localSum += tempStorage[j+1] + A[j+blockSize];
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            B[i] = localSum;
            localSum = 0.0f;
        }
        if (j + blockSize < N) {
            tempStorage[j] = B[j+blockSize];
        }
    }
}

void host_vector_addition(float *A, float *B, float *C, int N) {
    dim3 gridDim(1 + (N - 1) / 32); // Choose a suitable block size for your GPU.
    dim3 blockDim(32);

    addVectorsWithSharedMemory<<<gridDim, blockDim>>>(A, B, C);
    cudaDeviceSynchronize();
}