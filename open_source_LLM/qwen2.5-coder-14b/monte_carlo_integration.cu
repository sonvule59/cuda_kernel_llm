#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <cmath>

// Define the function f(x)
__device__ float f(float x) {
    return x * x + sinf(x);
}

// CUDA kernel to compute function values and store in an array
__global__ void monte_carlo_kernel(float* xi, float* yi, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        yi[idx] = f(xi[idx]);
    }
}

int main() {
    // Interval [a, b]
    float a = 0.0f;
    float b = 1.0f;

    // Number of samples
    int n = 1000000;

    // Allocate memory on the host
    float* xi_host = new float[n];
    float* yi_host = new float[n];

    // Allocate memory on the device
    float* xi_device;
    float* yi_device;
    cudaMalloc(&xi_device, n * sizeof(float));
    cudaMalloc(&yi_device, n * sizeof(float));

    // Initialize random number generator
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    // Generate random points uniformly distributed in [a, b]
    curandGenerateUniform(gen, xi_device, n);
    for (int i = 0; i < n; ++i) {
        xi_host[i] = a + (b - a) * xi_device[i];
    }

    // Copy the generated points to the device
    cudaMemcpy(xi_device, xi_host, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    monte_carlo_kernel<<<blocksPerGrid, threadsPerBlock>>>(xi_device, yi_device, n);

    // Copy the results back to the host
    cudaMemcpy(yi_host, yi_device, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute the average of yi
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += yi_host[i];
    }
    float average_yi = sum / n;

    // Estimate the integral
    float integral_estimate = (b - a) * average_yi;

    // Print the result
    std::cout << "Estimated integral: " << integral_estimate << std::endl;

    // Free resources
    delete[] xi_host;
    delete[] yi_host;
    cudaFree(xi_device);
    cudaFree(yi_device);
    curandDestroyGenerator(gen);

    return 0;
}
