#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>

const int BATCH_SIZE = 1;
const int CHANNELS = 32;
const int DIM1 = 64;
const int DIM2 = 64;
const int DIM3 = 64;
const int K = 2; // pooling stride in each dimension

__global__ void max_pool(unsigned char4 *input, unsigned char4 *output) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= BATCH_SIZE * CHANNELS * DIM1 * DIM2 * DIM3) return;

    int c = index / (DIM1 * DIM2 * DIM3);
    int d1 = (index % (DIM1 * DIM2 * DIM3)) / (DIM2 * DIM3);
    int d2 = (index % (DIM1 * DIM2 * DIM3)) % (DIM2 * DIM3) / DIM3;
    int d3 = index % DIM3;

    int pooled_start = c * K + d1 * K * K + d2 * K + d3;

    unsigned char4 maxValue = input[pooled_start];

    for (int kk = 0; kk < K; ++kk) {
        for (int ll = 0; ll < K; ++ll) {
            for (int mm = 0; mm < K; ++mm) {
                int pooled_index = c * K + d1 + kk + (d2 + ll) * DIM3 + (d3 + mm);
                if (input[pooled_index].x > maxValue.x) {
                    maxValue = input[pooled_index];
                }
            }
        }
    }

    output[index] = maxValue;
}

int main() {
    unsigned char4 *d_input, *d_output, *h_output;
    size_t bytesPerElement = sizeof(unsigned char4);
    cudaMalloc((void**)&d_input, BATCH_SIZE * CHANNELS * DIM1 * DIM2 * DIM3 * bytesPerElement);
    cudaMalloc((void**)&d_output, BATCH_SIZE * CHANNELS * (DIM1 / K) * (DIM2 / K) * (DIM3 / K) * bytesPerElement);
    h_output = (unsigned char4 *)malloc(BATCH_SIZE * CHANNELS * (DIM1 / K) * (DIM2 / K) * (DIM3 / K) * sizeof(unsigned char4));

    // Initialize input data here...

    dim3 threadsPerBlock(DIM1, DIM2, DIM3);
    dim3 blocksPerGrid((DIM1 + threadsPerBlock.x - 1) / threadsPerBlock.x, (DIM2 + threadsPerBlock.y - 1) / threadsPerBlock.y, (DIM3 + threadsPerBlock.z - 1) / threadsPerBlock.z);
    max_pool<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, BATCH_SIZE * CHANNELS * (DIM1 / K) * (DIM2 / K) * (DIM3 / K) * bytesPerElement, cudaMemcpyDeviceToHost);

    // Check output here...

    return 0;
}