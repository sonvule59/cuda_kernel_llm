#include <iostream>
#include <cuda_runtime.h>

// Define the global constants
const unsigned int batchSize = 16;
const unsigned int channels = 32;
const unsigned int height = 64;
const unsigned int width = 128;

// Define the ReLU kernel function
__global__ void reluKernel(float* d_in, float* d_out) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < batchSize * channels * height * width) {
        d_out[idx] = max(d_in[idx], 0.0f);
    }
}

int main(void) {
    // Allocate device memory for input and output tensors
    float* h_in;
    float* h_out;
    cudaMalloc((void**)&h_in, batchSize * channels * height * width * sizeof(float));
    cudaMalloc((void**)&h_out, batchSize * channels * height * width * sizeof(float));

    // Initialize input tensor on the host
    for (int i = 0; i < batchSize * channels * height * width; ++i) {
        h_in[i] = static_cast<float>(rand() - rand() / RAND_MAX * 2);
    }

    // Transfer input tensor to the device
    cudaMemcpy(d_in, h_in, batchSize * channels * height * width * sizeof(float), cudaMemcpyHostToDevice);

    // Set up the grid and block dimensions for the kernel launch
    unsigned int threadsPerBlock = 256;
    unsigned int blocksPerGrid = (batchSize * channels * height * width + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the ReLU kernel on the device
    reluKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out);

    // Transfer output tensor back to the host for verification
    cudaMemcpy(h_out, d_out, batchSize * channels * height * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Verification of the ReLU function results
    for (int i = 0; i < batchSize * channels * height * width; ++i) {
        if (h_out[i] < 0.0f || h_in[i] >= 0.0f && h_out[i] == h_in[i]) {
            std::cout << "Error: ReLU function did not produce correct results." << std::endl;
            return -1;
        }
    }

    // Print success message
    std::cout << "ReLU kernel execution successful!" << std::endl;

    // Free the device memory and deallocate the pointers
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}