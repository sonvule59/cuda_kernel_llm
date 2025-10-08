// sigmoid_kernel.h
#ifndef SIGMOID_KERNEL_H
#define SIGMOID_KERNEL_H

#include <cuda.h>
#include <curand_kernel.h>

__global__ void sigmoidKernel(float *output, const float *input, int batchSize, int dim);

#endif //SIGMOID_KERNEL_H

// sigmoid_kernel.cu
#include "sigmoid_kernel.h"

__device__ float exp(float x) {
    return expf(x);
}

__device__ float invExp(float x) {
    return 1.0f / expf(x);
}

__global__ void sigmoidKernel(float *output, const float *input, int batchSize, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batchSize * dim) {
        output[idx] = invExp(-input[idx]);
    }
}

// main.cu
#include <iostream>
#include "sigmoid_kernel.h"

#define BATCH_SIZE 128
#define DIM 3072

int main() {
    float *inputDataHost, *outputDataHost;
    float *inputDataDevice, *outputDataDevice;
    FILE *outputFile;

    // Allocate host memory for input and output data
    inputDataHost = (float *)malloc(BATCH_SIZE * DIM * sizeof(float));
    outputDataHost = (float *)malloc(BATCH_SIZE * DIM * sizeof(float));

    // Open output file for writing the results
    outputFile = fopen("output.txt", "w");

    // Allocate device memory for input and output data
    cudaMalloc((void **)&inputDataDevice, BATCH_SIZE * DIM * sizeof(float));
    cudaMalloc((void **)&outputDataDevice, BATCH_SIZE * DIM * sizeof(float));

    // Fill the input data with random values (replace this part with your actual input data)
    for (int i = 0; i < BATCH_SIZE * DIM; ++i) {
        inputDataHost[i] = rand() / (float)RAND_MAX - 1.0f;
    }

    // Copy the input data to device memory
    cudaMemcpy(inputDataDevice, inputDataHost, BATCH_SIZE * DIM * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BATCH_SIZE, 1);
    dim3 blocksPerGrid(ceil(BATCH_SIZE * 1.0f / (float)threadsPerBlock.x), 1);

    // Launch the kernel function on the GPU
    sigmoidKernel<<<blocksPerGrid, threadsPerBlock>>>(outputDataDevice, inputDataDevice, BATCH_SIZE, DIM);

    // Copy the output data from device memory to host memory and write it to the output file
    cudaMemcpy(outputDataHost, outputDataDevice, BATCH_SIZE * DIM * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < BATCH_SIZE * DIM; ++i) {
        fprintf(outputFile, "%.16f\n", outputDataHost[i]);
    }
    fclose(outputFile);

    // Free GPU memory
    cudaFree(inputDataDevice);
    cudaFree(outputDataDevice);

    // Free host memory
    free(inputDataHost);
    free(outputDataHost);

    return 0;
}