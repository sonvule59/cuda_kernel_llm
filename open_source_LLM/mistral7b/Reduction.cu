#include <iostream>
#include <vector_types.h>
#include <curand_kernel.h>

const int BLOCK_SIZE = 32;
const int NUM_BLOCKS = ceil(sizeof(float) * num / BLOCK_SIZE);

__global__ void kernel(float* data, float* tempSum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num)
        tempSum[blockIdx.x] += data[idx];
}

int main() {
    int num = 1024;
    float* h_data;
    float* d_data;
    float* d_tempSum;
    float sum;

    cudaMalloc((void**)&d_data, num * sizeof(float));
    cudaMalloc(&d_tempSum, NUM_BLOCKS * sizeof(float));

    curand_init(0, NULL);
    for (int i = 0; i < num; ++i) {
        h_data[i] = static_cast<float>(curand_generate(&globalRng));
        d_data[i] = h_data[i];
    }

    kernel<<<1, BLOCK_SIZE>>>(d_data, d_tempSum);

    for (int i = 1; i < NUM_BLOCKS; ++i) {
        if (i % 64 == 0) {
            cudaDeviceSynchronize();
        }
        d_tempSum[i] += d_tempSum[i - 1];
    }

    sum = d_tempSum[NUM_BLOCKS - 1];

    cudaFree(d_data);
    cudaFree(d_tempSum);

    std::cout << "The sum of the elements is: " << sum << std::endl;

    return 0;
}