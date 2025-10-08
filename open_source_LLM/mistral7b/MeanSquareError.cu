#include <cuda_runtime.h>
#include <vector_types.h>
#include <stddef.h>

__device__ float mse(float * predictions, float * targets, int N) {
    register float sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += (predictions[i] - targets[i]) * (predictions[i] - targets[i]);
    }
    return sum / N;
}

__global__ void mseKernel(float * d_predictions, float * d_targets, float * d_mse, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        d_mse[i] = mse(d_predictions, d_targets, N);
}

void calculateMSE(float * h_predictions, float * h_targets, float * d_predictions, float * d_targets, float * d_mse, int N, size_t bytesPerElement) {
    dim3 grid(N / 256 + (N % 256 > 0 ? 1 : 0), 1, 1);
    dim3 block(256, 1, 1);
    cudaMemcpy(d_predictions, h_predictions, N * bytesPerElement, cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_targets, N * bytesPerElement, cudaMemcpyHostToDevice);
    mseKernel<<<grid, block>>>(d_predictions, d_targets, d_mse, N);
    cudaMemcpy(h_mse, d_mse, N * sizeof(float), cudaMemcpyDeviceToHost);
}

int main() {
    float * h_predictions;
    float * h_targets;
    float * h_mse;

    // Allocate memory for host and device
    cudaMallocManaged(&h_predictions, N * sizeof(float));
    cudaMallocManaged(&h_targets, N * sizeof(float));
    cudaMallocManaged(&h_mse, N * sizeof(float));

    // Initialize predictions and targets arrays on the host
    // ...

    calculateMSE(h_predictions, h_targets, d_predictions, d_targets, h_mse, N, sizeof(float));

    // Print the MSE result
    printf("Mean Squared Error: %f\n", h_mse[0]);

    cudaFree(h_predictions);
    cudaFree(h_targets);
    cudaFree(h_mse);
    return 0;
}