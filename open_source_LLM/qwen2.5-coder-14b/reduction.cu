#include <iostream>
#include <cuda_runtime.h>

__global__ void parallelReduce(float* d_in, float* d_out, int n) {
    external __shared__ float sdata[256];  // Shared memory for partial sums
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory from global memory
    if (i < n) {
        sdata[tid] = d_in[i];
    } else {
        sdata[tid] = 0.0f;
    }

    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

float blockReduce(float* d_in, int n) {
    // Number of blocks
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Allocate memory for the intermediate results
    float* d_temp;
    cudaMalloc((void**)&d_temp, numBlocks * sizeof(float));

    // Perform the first level of reduction
    parallelReduce<<<numBlocks, blockSize>>>(d_in, d_temp, n);
    cudaDeviceSynchronize();

    // If there are still multiple blocks to sum, repeat the process
    if (numBlocks > 1) {
        blockReduce(d_temp, numBlocks);
    }

    // Copy the final result back to host
    float h_result;
    cudaMemcpy(&h_result, d_temp, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_temp);

    return h_result;
}

int main() {
    int n = 1024;  // Size of the input array
    float* h_in = new float[n];  // Host memory for input
    float* d_in;  // Device memory for input

    // Initialize the input array with some values
    for (int i = 0; i < n; ++i) {
        h_in[i] = 1.0f;
    }

    // Allocate device memory and copy input array to device
    cudaMalloc((void**)&d_in, n * sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);

    // Perform the reduction
    float result = blockReduce(d_in, n);

    // Output the result
    std::cout << "Sum of elements: " << result << std::endl;

    // Free the allocated memory
    delete[] h_in;
    cudaFree(d_in);

    return 0;
}
