#include <iostream>
#include <vector>
#include <curand_kernel.h>

const int N = 1000000; // Input array size
const int numBins = 1024; // Number of bins

__global__ void kernel(int *input, int *output, int numBins) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        output[input[index]] += 1;
    }
}

int main(void) {
    int *input, *output;
    curandGenerator_t generator;
    curandState_t state;
    int *d_output;

    std::vector<int> h_input(N), h_output(numBins);
    cudaMalloc((void **)&input, N * sizeof(int));
    cudaMalloc((void **)&d_output, numBins * sizeof(int));

    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_RANDOM);
    curandInit(generator, CURAND_RNG_ALG_AES, CURAND_DETERMINISTIC);
    curandSetPseudoRandomGeneratorSeed(generator, 0);

    for (int i = 0; i < N; ++i) {
        h_input[i] = curandGenInteger(generator, 32767) + 1; // Generate random integers in range [1, 32768]
    }

    cudaMemcpy(input, h_input.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 1, 1);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x);
    kernel<<<numBlocks, threadsPerBlock>>>(input, d_output, numBins);

    cudaMemcpy(h_output.data(), d_output, numBins * sizeof(int), cudaMemcpyDeviceToHost);

    curandDestroyGenerator(generator);
    cudaFree(input);
    cudaFree(d_output);

    return 0;
}