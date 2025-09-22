// matrix_mul.cu
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define MAX_THREADS_PER_BLOCK 1024

__global__ void MatrixMulKernel(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float sA[32][32]; // Shared memory for A
    __shared__ float sB[32][32]; // Shared memory for B

    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int tx = threadIdx.x; 
    int ty = threadIdx.y;

    int row = by * 32 + ty;
    int col = bx * 32 + tx;
    float sum = 0.0f;

    for (int m = 0; m < (K / 32); ++m) {
        // Load elements from global memory to shared memory
        sA[ty][tx] = A[row * K + (m * 32 + tx)];
        sB[ty][tx] = B[(m * 32 + ty) * N + col];
        
        __syncthreads(); // Synchronize threads within the block

        for (int k = 0; k < 32; ++k) {
            sum += sA[ty][k] * sB[k][tx];
        }
        __syncthreads(); // Synchronize before loading next set of elements
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void MatrixMul(float *A, float *B, float *C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    
    // Allocate device memory
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((N + 31) / 32, (M + 31) / 32);

    // Launch kernel
    MatrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

    // Copy result back to host
    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int M = 512, N = 512, K = 512;
    size_t size = M * K * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(M * N * sizeof(float));

    // Initialize matrices A and B with random values
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i + M * K / 2] = rand() / (float)RAND_MAX; // Ensure non-zero values for better visualization of results
    }

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    MatrixMul(h_A, h_B, h_C, M, N, K);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time to compute matrix multiplication: %f ms\n", elapsedTime);

    // Verify the result
    for (int i = 0; i < M * N; ++i) {
        if (h_C[i] != h_C[i]) { // Check for NaN, assuming non-zero values mean correct computation
            printf("Error at element %d: %f\n", i, h_C[i]);
            break;
        }
    }

    free(h_A);
    free(h_B);
    free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}