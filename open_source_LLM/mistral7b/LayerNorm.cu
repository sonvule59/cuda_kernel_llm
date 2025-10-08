// **.cu file (histogram_cuda.cu)**

#include <stddef.h>
#include <cuda_runtime.h>

#define BINS 256
#define N 1000000

__global__ void histogramKernel(unsigned char *data, int *histogram) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
        atomicAdd(&histogram[data[index]], 1);
}

void checkResult(unsigned char *data, int *deviceHistogram, int *hostHistogram) {
    cudaMemcpy(hostHistogram, deviceHistogram, BINS * sizeof(int), cudaMemcpyDeviceToHost);
    // Your validation logic here, e.g., comparing with the expected distribution or other reference results
}

int main() {
    unsigned char *data;
    size_t dataSize = N * sizeof(unsigned char);
    cudaMalloc((void**)&data, dataSize);

    // Initialize data here, e.g., with random values or a specific distribution

    int *histogram;
    cudaMalloc((void**)&histogram, BINS * sizeof(int));

    dim3 blocks(BINS);
    dim3 threadsPerBlock(128);

    histogramKernel<<<blocks, threadsPerBlock>>>(data, histogram);

    int *deviceResult;
    cudaMalloc((void**)&deviceResult, BINS * sizeof(int));

    checkResult(data, histogram, deviceResult);

    // ... Perform further operations with the device result if needed

    cudaFree(data);
    cudaFree(histogram);
    cudaFree(deviceResult);
}


// **.cpp file (main.cpp)**

#include <iostream>
#include "histogram_cuda.cu"
#include <vector>
#include <random>
#include <algorithm>

int main() {
    // Generate random data on the CPU and copy it to the GPU
    std::mt19937 generator(time(nullptr));
    std::uniform_int_distribution<unsigned char> distribution(0, 255);
    std::vector<unsigned char> data(N);
    std::generate(data.begin(), data.end(), [&]() { return distribution(generator); });

    unsigned char *cudaData;
    cudaMalloc((void**)&cudaData, data.size() * sizeof(unsigned char));
    cudaMemcpy(cudaData, data.data(), data.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Perform histogram computation on the GPU
    // ... (the CUDA code you provided)

    int hostHistogram[BINS];
    checkResult(cudaData, deviceHistogram, hostHistogram);

    std::cout << "Histogram:" << std::endl;
    for (int i = 0; i < BINS; ++i) {
        std::cout << "Bin " << i << ": " << hostHistogram[i] << std::endl;
    }

    cudaFree(cudaData);
}