// histogram_kernel.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NUM_BINS 256

__global__ void computeHistogram(const int *input, int *histogram, const int numElements) {
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numElements) {
        atomicAdd(&histogram[input[idx]], 1);
    }
}

void computeHistogramHost(const int *h_input, int *h_histogram, const int numElements) {
    for (int i = 0; i < NUM_BINS; ++i) {
        h_histogram[i] = 0;
    }
    
    // Allocate memory on the device
    int *d_input, *d_histogram;
    cudaMalloc((void**)&d_input, numElements * sizeof(int));
    cudaMalloc((void**)&d_histogram, NUM_BINS * sizeof(int));
    
    // Copy data to the device
    cudaMemcpy(d_input, h_input, numElements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, NUM_BINS * sizeof(int));
    
    // Define block and grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch the kernel
    computeHistogram<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_histogram, numElements);
    
    // Copy results back to host
    cudaMemcpy(h_histogram, d_histogram, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_histogram);
}

void validateHistogram(const int *h_histogram) {
    for (int i = 0; i < NUM_BINS; ++i) {
        if (h_histogram[i] != 0) {
            printf("Bin %d: %d elements\n", i, h_histogram[i]);
        }
    }
}

int main() {
    const int numElements = 1024; // Example number of elements
    int h_input[numElements];
    
    // Initialize input array with random integers in the range 0-255
    for (int i = 0; i < numElements; ++i) {
        h_input[i] = rand() % 256;
    }
    
    int h_histogram[NUM_BINS];
    
    // Compute the histogram on the device
    computeHistogramHost(h_input, h_histogram, numElements);
    
    // Validate and print the histogram
    validateHistogram(h_histogram);
    
    return 0;
}