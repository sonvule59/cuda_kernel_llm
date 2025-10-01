#include <iostream>
#include <cuda_runtime.h>

// Host function to initialize the input vector
void initInput(float* h_input, int n) {
    for (int i = 0; i < n; ++i) {
        h_input[i] = (float)(-5 + rand() % 10); // Random values between -5 and 4
    }
}

// CUDA kernel to apply ReLU activation function
__global__ void reluKernel(float* d_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_input[idx] = max(0.0f, d_input[idx]);
    }
}

// Host function to copy data from device to host
void printOutput(float* d_output, int n) {
    float* h_output = new float[n];
    cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Output vector after ReLU Activation: ";
    for (int i = 0; i < n; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    delete[] h_output;
}

int main() {
    const int n = 10; // Size of the input vector
    float* h_input = new float[n];

    initInput(h_input, n);

    // Allocate memory on the device
    float* d_input;
    cudaMalloc((void**)&d_input, n * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    reluKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, n);

    // Print the output vector
    printOutput(d_input, n);

    // Free device memory
    cudaFree(d_input);
    delete[] h_input;

    return 0;
}
