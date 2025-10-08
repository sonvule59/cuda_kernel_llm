#include <cuda_runtime.h>
#include <vector_types.h>
#include <curand_kernel.h>

__device__ double f(double x) {
    // Define your function here
    return x * x;
}

void integrateKernel(curandState* state, double* output, int N, double h, double a, double b) {
    double y = 0.0;
    double x = a + h * curand_uniform(state);
    for (int i = 0; i < N; ++i) {
        y += f(x);
        x += 2 * h;
    }
    __syncthreads();
    if (blockDim.x * blockIdx.x + threadIdx.x >= N) output[blockIdx.x] += y / N / h;
}

double integrate(int N, double a, double b) {
    int blocks = (b - a) * 1024 / (N * sizeof(double)) + 1;
    int threadsPerBlock = 1024;

    // Allocate device memory
    double* d_output;
    cudaMalloc(&d_output, blocks * sizeof(double));

    curandGenerator_t generator;
    curandCreate(&generator);
    curandState state;
    curand_init(generator, 0, 0, &state);

    // Launch the kernel on the GPU
    integrateKernel<<<blocks, threadsPerBlock>>>(state, d_output, N, (b - a) / N, a, b);

    double result;
    cudaMemcpy(&result, d_output, blocks * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_output);
    curandDestroy(generator);

    return result;
}