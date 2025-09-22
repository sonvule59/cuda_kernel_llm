// **convolution_kernel.cu**

#include <cuda.h>
#include <curand_kernel.h>

__global__ void convolve(const float *input, const float *filter, float *output) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= N && idx < N + M) {
        float sum = 0.f;
        for (int i = max(0, idx - M + 1); i <= min(N, idx); ++i) {
            sum += input[i] * filter[idx - i];
        }
        output[blockIdx.x] = sum;
    }
}

// **main.cu**

#include <stdio.h>
#include "convolution_kernel.h"

void checkCudaErrors(cudaError_t err, const char *name) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Fatal error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    int N = 1024;
    int M = 32;

    float *input_d, *filter_d, *output_d, *output_h;
    float *padded_input_d;
    size_t bufferSize;

    checkCudaErrors(cudaMalloc((void **)&input_d, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&filter_d, M * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&output_d, (N - M + 1) * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&output_h, (N - M + 1) * sizeof(float)));
    bufferSize = N * sizeof(float);
    checkCudaErrors(cudaMalloc((void **)&padded_input_d, bufferSize));

    // Zero-pad input signal and copy it to the device
    float paddingValue = 0.f;
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_RANDOM, NULL);
    checkCudaErrors(curandGenerateNormal(generator, input_d, N)); // Fill input with random values for this example
    cudaMemcpy(padded_input_d, input_d, bufferSize, cudaMemcpyDeviceToHost);
    for (int i = N; i < bufferSize; ++i) {
        padded_input_d[i] = paddingValue;
    }
    cudaMemcpy(input_d, padded_input_d, bufferSize, cudaMemcpyHostToDevice);

    // Copy filter to the device
    float hann[M] = {0.5f, 0.5f};
    for (int i = 2; i < M; ++i) {
        hann[i] = 0.5f * cos(2.0f * M_PI * (i - 1) / (M - 1)); // Hann window
    }
    cudaMemcpy(filter_d, hann, M * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N - M + 1) / threadsPerBlock + ((N - M + 1) % threadsPerBlock != 0 ? 1 : 0);

    convolve<<<blocksPerGrid, threadsPerBlock>>>(input_d, filter_d, output_d);
    cudaMemcpy(output_h, output_d, (N - M + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    // Output computation and printing
    float sum = 0.f;
    for (int i = 0; i < N - M + 1; ++i) {
        sum += output_h[i];
    }
    printf("Sum of the convolved signal: %f\n", sum);

    cudaFree(input_d);
    cudaFree(filter_d);
    cudaFree(output_d);
    cudaFree(padded_input_d);
    cudaFreeHost(output_h);
    curandDestroyGenerator(generator);

    return 0;
}