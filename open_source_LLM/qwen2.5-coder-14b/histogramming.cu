#include <iostream>
#include <cuda_runtime.h>

// Kernel function to compute the histogram
__global__ void histogramKernel(const int* input, const int N, int* hist, const int numBins) {
    // Shared memory to store local histogram counts
    extern __shared__ int sharedHist[];

    // Initialize shared memory for this thread block
    int tid = threadIdx.x;
    if (tid < numBins) {
        sharedHist[tid] = 0;
    }
    __syncthreads();

    // Compute the global index
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        atomicAdd(&sharedHist[input[idx]], 1);
    }
    __syncthreads();

    // Write the results to global memory
    if (tid < numBins) {
        atomicAdd(&hist[tid], sharedHist[tid]);
    }
}

void computeHistogram(const int* input, int N, int* hist, int numBins) {
    // Allocate memory on the device
    int* dev_input;
    int* dev_hist;
    cudaMalloc((void**)&dev_input, N * sizeof(int));
    cudaMalloc((void**)&dev_hist, numBins * sizeof(int));

    // Copy input data to the device
    cudaMemcpy(dev_input, input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize the histogram to zero
    cudaMemset(dev_hist, 0, numBins * sizeof(int));

    // Define the block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    histogramKernel<<<blocksPerGrid, threadsPerBlock, numBins * sizeof(int)>>>(dev_input, N, dev_hist, numBins);

    // Copy the results back to the host
    cudaMemcpy(hist, dev_hist, numBins * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dev_input);
    cudaFree(dev_hist);
}

int main() {
    // Example usage
    const int N = 1024; // Size of the input array
    const int numBins = 256; // Number of bins in the histogram

    // Initialize input data on the host
    int* input = new int[N];
    for (int i = 0; i < N; ++i) {
        input[i] = rand() % numBins; // Random values in the range [0, numBins)
    }

    // Allocate memory for the histogram on the host
    int* hist = new int[numBins];

    // Compute the histogram
    computeHistogram(input, N, hist, numBins);

    // Print the histogram
    for (int i = 0; i < numBins; ++i) {
        std::cout << "Histogram[" << i << "] = " << hist[i] << std::endl;
    }

    // Clean up
    delete[] input;
    delete[] hist;

    return 0;
}
