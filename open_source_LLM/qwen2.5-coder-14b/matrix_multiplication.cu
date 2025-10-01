#include <iostream>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matrixMulKernel(float* A, float* B, float* C, int M, int N, int K) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the current C element being computed
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Csub = 0.0f;
    for (int k = 0; k < K; ++k) {
        float Aelem = A[row * K + k];
        float Belem = B[k * N + col];
        Csub += Aelem * Belem;
    }

    // Write the result to device memory
    if (row < M && col < N) {
        C[row * N + col] = Csub;
    }
}

void matrixMul(float* h_A, float* h_B, float* h_C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define dimensions for the grid and block
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch kernel
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int M = 1024, N = 1024, K = 1024;
    float *h_A, *h_B, *h_C;

    // Allocate host memory
    h_A = (float*)malloc(M * K * sizeof(float));
    h_B = (float*)malloc(K * N * sizeof(float));
    h_C = (float*)malloc(M * N * sizeof(float));

    // Initialize matrices with random values
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Perform matrix multiplication
    matrixMul(h_A, h_B, h_C, M, N, K);

    // Print results (for debugging purposes)
    // for (int i = 0; i < M; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         std::cout << h_C[i * N + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
