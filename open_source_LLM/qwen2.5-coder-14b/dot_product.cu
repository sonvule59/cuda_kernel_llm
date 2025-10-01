#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel to compute dot product
__global__
void dotProduct(float* A, float* B, float* C, int n) {
    // Shared memory to store intermediate results
    __shared__ float partialSum[512];
    
    int tid = threadIdx.x;  // Thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread index

    // Initialize shared memory
    partialSum[tid] = 0.0f;

    if (idx < n) {
        // Partially compute the dot product
        partialSum[tid] = A[idx] * B[idx];
    }

    // Synchronize to ensure all partial sums are computed
    __syncthreads();

    // Reduce the results in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partialSum[tid] += partialSum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        C[blockIdx.x] = partialSum[0];
    }
}

// Host function to compute dot product
float computeDotProduct(float* A, float* B, int n) {
    float C[1], c = 0.0f;
    float* d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc((void**)&d_A, n * sizeof(float));
    cudaMalloc((void**)&d_B, n * sizeof(float));
    cudaMalloc((void**)&d_C, (n / 512 + 1) * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, A, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    int blockSize = 512;
    int numBlocks = (n + blockSize - 1) / blockSize;
    dotProduct<<<numBlocks, blockSize>>>(d_A, d_B, d_C, n);

    // Copy result from device to host
    cudaMemcpy(C, d_C, sizeof(float), cudaMemcpyDeviceToHost);

    // Sum up the results from blocks
    for (int i = 0; i < numBlocks; ++i) {
        c += C[i];
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return c;
}

int main() {
    int n = 1024; // Size of the vectors
    float A[n], B[n];
    
    // Initialize the vectors with some values
    for (int i = 0; i < n; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(i + 1);
    }

    float result = computeDotProduct(A, B, n);

    std::cout << "Dot Product: " << result << std::endl;

    return 0;
}
