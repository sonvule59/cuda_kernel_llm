// sum_reduction.cu
#include <iostream>
#include <cuda_runtime.h>

#define N 1024 // Size of the array (must be a power of two for simplicity)

__global__ void reduceKernel(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (i < n) {
        sdata[tid] = input[i];
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    const int numBlocks = N / 256; // Number of blocks in the grid, each with 256 threads
    const int numThreads = 256; // Each block has 256 threads

    float h_input[N];
    float h_outputCPU[numBlocks];
    float* d_input;
    float* d_output;

    // Initialize input array with some values
    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(i + 1); // Example data
    }

    // Allocate memory on the device
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, numBlocks * sizeof(float));

    // Copy input array to the device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    reduceKernel<<<numBlocks, numThreads, numThreads * sizeof(float)>>>(d_input, d_output, N);

    // Copy output from device to host
    cudaMemcpy(h_outputCPU, d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Final reduction on the CPU for verification
    float h_outputGPU[numBlocks];
    for (int i = 0; i < numBlocks; i++) {
        h_outputGPU[i] = h_outputCPU[i];
    }
    while (numBlocks > 1) {
        int nextLevelSize = (numBlocks + 1) / 2;
        for (int i = 0; i < numBlocks / 2; i++) {
            h_outputGPU[i] = h_outputGPU[i * 2] + h_outputGPU[i * 2 + 1];
        }
        if (numBlocks % 2 != 0) {
            h_outputGPU[(numBlocks / 2)] = h_outputGPU[numBlocks - 1];
        }
        numBlocks = nextLevelSize;
    }

    // Check correctness
    float sumCPU = 0;
    for (int i = 0; i < N; i++) {
        sumCPU += h_input[i];
    }
    if (fabs(sumCPU - h_outputGPU[0]) / sumCPU > 1e-5) {
        std::cerr << "Result verification failed!" << std::endl;
        exit(-1);
    } else {
        std::cout << "Sum reduction is correct: " << h_outputGPU[0] << std::endl;
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}