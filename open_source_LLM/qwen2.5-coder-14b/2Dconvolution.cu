#include <iostream>
#include <cuda_runtime.h>

// Define the kernel for 2D convolution
__global__ void convolve(float* input, float* kernel, float* output, int inputRows, int inputCols, int kernelRows, int kernelCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < inputRows - kernelRows + 1 && col < inputCols - kernelCols + 1) {
        float sum = 0.0f;
        for (int m = 0; m < kernelRows; ++m) {
            for (int n = 0; n < kernelCols; ++n) {
                sum += input[(row + m) * inputCols + col + n] * kernel[m * kernelCols + n];
            }
        }
        output[row * (inputCols - kernelCols + 1) + col] = sum;
    }
}

void performConvolution(float* input, float* kernel, float* output, int inputRows, int inputCols, int kernelRows, int kernelCols) {
    // Allocate memory on the device
    float* d_input;
    float* d_kernel;
    float* d_output;

    cudaMalloc(&d_input, inputRows * inputCols * sizeof(float));
    cudaMalloc(&d_kernel, kernelRows * kernelCols * sizeof(float));
    cudaMalloc(&d_output, (inputRows - kernelRows + 1) * (inputCols - kernelCols + 1) * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input, input, inputRows * inputCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelRows * kernelCols * sizeof(float), cudaMemcpyHostToDevice);

    // Define the block and grid size
    dim3 blockSize(16, 16);
    dim3 gridSize((inputCols - kernelCols + 1 + blockSize.x - 1) / blockSize.x,
                  (inputRows - kernelRows + 1 + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    convolve<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, inputRows, inputCols, kernelRows, kernelCols);

    // Copy the result back to the host
    cudaMemcpy(output, d_output, (inputRows - kernelRows + 1) * (inputCols - kernelCols + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

int main() {
    // Example usage
    int inputRows = 5, inputCols = 5;
    int kernelRows = 3, kernelCols = 3;

    float input[] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
        4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        5.0f, 6.0f, 7.0f, 8.0f, 9.0f
    };

    float kernel[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };

    float output[(inputRows - kernelRows + 1) * (inputCols - kernelCols + 1)];

    performConvolution(input, kernel, output, inputRows, inputCols, kernelRows, kernelCols);

    // Print the result
    std::cout << "Convolution Output:" << std::endl;
    for (int i = 0; i < (inputRows - kernelRows + 1); ++i) {
        for (int j = 0; j < (inputCols - kernelCols + 1); ++j) {
            std::cout << output[i * (inputCols - kernelCols + 1) + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
