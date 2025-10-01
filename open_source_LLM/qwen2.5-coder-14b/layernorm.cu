#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void layerNorm(float* input, float* output, int batchSize, int features, int dim1, int dim2) {
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int index = bx * blockDim.x + tx;

    if (index < batchSize * features) {
        int batch = index / features;
        int feature = index % features;

        // Calculate mean
        float sum = 0.0f;
        for (int i = 0; i < dim1 * dim2; ++i) {
            sum += input[(batch * features + feature) * dim1 * dim2 + i];
        }
        float mean = sum / (dim1 * dim2);

        // Calculate variance
        float varSum = 0.0f;
        for (int i = 0; i < dim1 * dim2; ++i) {
            float diff = input[(batch * features + feature) * dim1 * dim2 + i] - mean;
            varSum += diff * diff;
        }
        float variance = varSum / (dim1 * dim2);
        float epsilon = 1e-6f;
        float stddevInv = rsqrt(variance + epsilon);

        // Normalize
        for (int i = 0; i < dim1 * dim2; ++i) {
            output[(batch * features + feature) * dim1 * dim2 + i] = (input[(batch * features + feature) * dim1 * dim2 + i] - mean) * stddevInv;
        }
    }
}

void runLayerNorm(float* input, float* output, int batchSize, int features, int dim1, int dim2) {
    int numThreads = 256;
    int numBlocks = (batchSize * features + numThreads - 1) / numThreads;

    layerNorm<<<numBlocks, numThreads>>>(input, output, batchSize, features, dim1, dim2);
    cudaDeviceSynchronize();
}

int main() {
    int batchSize = 10;
    int features = 5;
    int dim1 = 32;
    int dim2 = 32;

    size_t inputSize = batchSize * features * dim1 * dim2 * sizeof(float);
    float* h_input = (float*)malloc(inputSize);
    float* d_input, *d_output;
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_output, inputSize);

    // Initialize h_input with some values
    for (int i = 0; i < batchSize * features * dim1 * dim2; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice);

    runLayerNorm(d_input, d_output, batchSize, features, dim1, dim2);

    cudaMemcpy(h_input, d_output, inputSize, cudaMemcpyDeviceToHost);

    // Clean up
    free(h_input);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
