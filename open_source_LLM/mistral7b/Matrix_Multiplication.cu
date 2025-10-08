#include <iostream>
#include <cuda_runtime.h>

constexpr int M = 16; // Rows in matrix A
constexpr int K = 8;  // Columns common to matrices A and B
constexpr int N = 24; // Columns in matrix B

__global__ void matrixMultiplication(float *d_C, const float *d_A, const float *d_B) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += d_A[idx + k * M] * d_B[k + K * blockIdx.y]; // Matrix A row indexed by idx, Matrix B column indexed by blockIdx.y * K + k
    }

    int idx2 = threadIdx.x + (blockDim.x * blockIdx.x) * N;
    d_C[idx2] = sum; // Store the result in the output matrix C
}

int main() {
    // Initialize host and device memory for matrices A, B, and C
    float *h_A, *h_B, *h_C;
    size_t size = M * K * sizeof(float);
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, N * M * sizeof(float));

    // Set up host memory for matrices A and B
    h_A = new float[size];
    h_B = new float[size];
    h_C = new float[N * M * sizeof(float)];

    // Initialize matrices A and B with random values
    srand(time(NULL));
    for (int i = 0; i < size; ++i) {
        h_A[i] = rand() % 1000;
        h_B[i] = rand() % 1000;
    }

    // Copy matrices A and B to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Set up grid and block dimensions for the matrix multiplication kernel
    int blocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 numBlocks(blocks, blocks);

    // Launch the matrixMultiplication kernel on the GPU
    matrixMultiplication<<<numBlocks, threadsPerBlock>>>(d_C, d_A, d_B);

    // Copy the result back to host memory and print it
    cudaMemcpy(h_C, d_C, N * M * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N * M; ++i) {
        std::cout << h_C[i] << " ";
        if ((i + 1) % M == 0) {
            std::cout << std::endl;
        }
    }

    // Clean up device and host memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}