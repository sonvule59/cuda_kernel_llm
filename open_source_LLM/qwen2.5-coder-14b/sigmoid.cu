#include <cuda_runtime.h>
#include <iostream>

// Define the sigmoid kernel
__global__ void sigmoid_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + exp(-input[idx]));
    }
}

// Main function
int main() {
    // Define input size and dimensions
    int batch_size = 1024;
    int dim = 1024;
    int size = batch_size * dim;

    // Allocate host memory
    float* h_input = (float*)malloc(size * sizeof(float));
    float* h_output = (float*)malloc(size * sizeof(float));

    // Initialize input data with random values
    for (int i = 0; i < size; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX * 20.0f - 10.0f;
    }

    // Allocate device memory
    float* d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    // Launch the sigmoid kernel
    sigmoid_kernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, size);

    // Copy the result back to host
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Print the first few elements of the output
    std::cout << "Sigmoid Output:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Free host memory
    free(h_input);
    free(h_output);

    return 0;
}
