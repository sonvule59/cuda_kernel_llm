#include <iostream>
#include <cuda_runtime.h>

// CUDA Kernel function to add vectors in parallel
__global__ void vecAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 1024; // Size of the vectors
    float h_A[N], h_B[N], h_C[N]; // Host copies of A, B, and C
    float *d_A, *d_B, *d_C; // Device copies of A, B, and C

    // Initialize host vectors A and B
    for (int i = 0; i < N; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel with one block of threads (adjust block size for optimal performance)
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize; // Number of blocks needed to cover all threads
    vecAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify the result
    for (int i = 0; i < N; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            std::cerr << "Error at index " << i << ": " << h_A[i] << " + " << h_B[i] << " != " << h_C[i] << std::endl;
            return -1;
        }
    }
    std::cout << "Success!" << std::endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}