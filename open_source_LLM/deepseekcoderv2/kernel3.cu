// sigmoid_kernel.cu
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

__global__ void sigmoid(float *input, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = 1.0f / (1.0f + exp(-input[idx]));
    }
}

void reference_sigmoid(const float *input, float *output, int N) {
    for (int i = 0; i < N; ++i) {
        output[i] = 1.0f / (1.0f + exp(-input[i]));
    }
}

int main() {
    const int N = 1024; // Size of the input array
    float h_input[N];   // Host input array
    float h_output[N];  // Host output array
    float *d_input, *d_output;

    // Initialize host input array with random values
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f; // Random values between -5 and 5
    }

    // Allocate memory on the device
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));

    // Copy input data to the device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    sigmoid<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);

    // Copy the result back to host
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Validate against the reference implementation
    reference_sigmoid(h_input, h_output, N);

    bool passed = true;
    const float epsilon = 1e-6f;
    for (int i = 0; i < N; ++i) {
        if (std::abs(h_output[i] - h_input[i]) > epsilon) {
            std::cerr << "Validation failed at index " << i << ": " << h_output[i] << " vs " << h_input[i] << std::endl;
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "Validation passed." << std::endl;
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}