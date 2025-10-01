#include <iostream>
#include <cuda_runtime.h>

// Define the dimensions of the matrix
#define ROWS 4
#define COLS 5

__global__ void transposeKernel(float* input, float* output, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        output[y * cols + x] = input[x * cols + y];
    }
}
int main() {
    // Host matrices
    float h_input[ROWS * COLS] = {0};
    float h_output[COLS * ROWS];

    // Initialize input matrix
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            h_input[i * COLS + j] = i * COLS + j;
        }
    }

    // Device matrices
    float* d_input = nullptr;
    float* d_output = nullptr;

    // Allocate memory on the device
    cudaMalloc(&d_input, ROWS * COLS * sizeof(float));
    cudaMalloc(&d_output, COLS * ROWS * sizeof(float));

    // Copy input matrix from host to device
    cudaMemcpy(d_input, h_input, ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the transpose kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((COLS + blockSize.x - 1) / blockSize.x, (ROWS + blockSize.y - 1) / blockSize.y);
    transposeKernel<<<gridSize, blockSize>>>(d_input, d_output, ROWS, COLS);

    // Copy the transposed matrix back to host
    cudaMemcpy(h_output, d_output, COLS * ROWS * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the transposed matrix
    std::cout << "Transposed Matrix:" << std::endl;
    for (int i = 0; i < COLS; ++i) {
        for (int j = 0; j < ROWS; ++j) {
            std::cout << h_output[i * ROWS + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
