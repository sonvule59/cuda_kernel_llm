#include <iostream>
#include <cuda_runtime.h>

// Kernel function to compute the squared differences
__global__ void computeSquaredDifferences(const float* predictions, const float* targets, float* squaredDifferences, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        squaredDifferences[idx] = (predictions[idx] - targets[idx]) * (predictions[idx] - targets[idx]);
    }
}

// Kernel function to sum up the squared differences
__global__ void sumSquaredDifferences(const float* squaredDifferences, float* sum, int n) {
    extern __shared__ float sharedMem[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        sharedMem[threadIdx.x] = squaredDifferences[idx];
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedMem[threadIdx.x] += sharedMem[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        sum[blockIdx.x] = sharedMem[0];
    }
}

int main() {
    // Example data
    int N = 1024;
    float *predictions, *targets, *d_predictions, *d_targets, *d_squaredDifferences, *d_sum;
    float h_mse = 0.0f;

    // Allocate host memory
    predictions = new float[N];
    targets = new float[N];

    // Initialize data (for simplicity, using random values)
    for (int i = 0; i < N; ++i) {
        predictions[i] = static_cast<float>(rand()) / RAND_MAX;
        targets[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    cudaMalloc(&d_predictions, N * sizeof(float));
    cudaMalloc(&d_targets, N * sizeof(float));
    cudaMalloc(&d_squaredDifferences, N * sizeof(float));
    cudaMalloc(&d_sum, (N + 255) / 256 * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_predictions, predictions, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, targets, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel to compute squared differences
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    computeSquaredDifferences<<<numBlocks, blockSize>>>(d_predictions, d_targets, d_squaredDifferences, N);

    // Launch kernel to sum squared differences
    numBlocks = (N + 255) / 256; // Each block will sum up to one value
    blockSize = 256;
    sumSquaredDifferences<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_squaredDifferences, d_sum, N);

    // Copy result back to host
    float *h_sum = new float[numBlocks];
    cudaMemcpy(h_sum, d_sum, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Sum up the results from each block
    for (int i = 0; i < numBlocks; ++i) {
        h_mse += h_sum[i];
    }

    // Compute the mean squared error
    h_mse /= N;

    // Output the result
    std::cout << "Mean Squared Error: " << h_mse << std::endl;

    // Free device and host memory
    cudaFree(d_predictions);
    cudaFree(d_targets);
    cudaFree(d_squaredDifferences);
    cudaFree(d_sum);
    delete[] predictions;
    delete[] targets;
    delete[] h_sum;

    return 0;
}
