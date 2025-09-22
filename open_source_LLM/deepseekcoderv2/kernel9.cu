// ### CUDA Kernel (convolution.cu)
#include <cuda_runtime.h>
#include <iostream>

// Kernel function for convolution
__global__ void convolve(float *input, float *filter, float *output, int N, int M) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= N) return; // Ensure we don't access out-of-bounds memory

    for (int i = 0; i < M; ++i) {
        output[index] += input[index + i] * filter[M - 1 - i];
    }
}

// Host function to set up and launch the kernel
void convolveHost(float *input, float *filter, float *output, int N, int M) {
    float *d_input, *d_filter, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_filter, M * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, N * sizeof(float)); // Initialize output to zero

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    convolve<<<numBlocks, blockSize>>>(d_input, d_filter, d_output, N, M);

    cudaMemcpy(output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
}
```

// ### Host Code (host.py)

import numpy as np
from convolve import convolveHost

# Parameters
N = 1024  # Length of input signal
M = 3     # Length of filter

# Generate random input and filter
input_signal = np.random.rand(N).astype(np.float32)
filter_coeffs = np.random.rand(M).astype(np.float32)
output_signal = np.zeros(N, dtype=np.float32)

# Call the CUDA function
convolveHost(input_signal, filter_coeffs, output_signal, N, M)

print("Input Signal:", input_signal)
print("Filter Coefficients:", filter_coeffs)
print("Output Signal after convolution:", output_signal)