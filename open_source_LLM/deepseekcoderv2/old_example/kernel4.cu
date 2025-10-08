#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << " code: " << error << " msg: " << cudaGetErrorString(error) << "\n"; \
        exit(-1); \
    } \
}

__global__ void relu_kernel(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

void relu_host(float *input, float *output, int batch_size, int channels) {
    float *d_input;
    float *d_output;
    CHECK(cudaMalloc(&d_input, batch_size * channels * sizeof(float)));
    CHECK(cudaMalloc(&d_output, batch_size * channels * sizeof(float)));

    // Copy input data to the device
    CHECK(cudaMemcpy(d_input, input, batch_size * channels * sizeof(float), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (batch_size * channels + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, batch_size * channels);
    CHECK(cudaGetLastError());

    // Copy the result back to host
    CHECK(cudaMemcpy(output, d_output, batch_size * channels * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    const int batch_size = 3;
    const int channels = 4;
    float input[batch_size * channels] = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f, 9.0f, -10.0f, 11.0f, -12.0f};
    float output[batch_size * channels];

    relu_host(input, output, batch_size, channels);

    // Print the result for verification
    std::cout << "Input: ";
    for (int i = 0; i < batch_size * channels; ++i) {
        std::cout << input[i] << " ";
    }
    std::cout << "\nOutput: ";
    for (int i = 0; i < batch_size * channels; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << "\n";

    return 0;
}