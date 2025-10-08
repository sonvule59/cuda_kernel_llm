#include <iostream>
#include <curand_kernel.h>
#include <cuda_runtime.h>

const int N = 1024;
__global__ void prefixSum(float *input, float *output) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        output[idx] = (idx == 0) ? input[idx] : output[idx - 1] + input[idx];
    }
}

int main(void) {
    float *h_input, *h_output;
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));

    h_input = new float[N];
    h_output = new float[N];

    // Initialize the input array with random numbers for demonstration purposes
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_RANDOM, NULL);
    curandSetStream(generator, 0);
    curand_kernel(d_input, N, generator);

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(32);
    dim3 grid((N + block.x - 1) / block.x);
    prefixSum<<<grid, block>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Input array:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << h_input[i] << ", ";
    }
    std::cout << "\nCumulative sum array: " << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << h_output[i] << ", ";
    }

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;

    return 0;
}