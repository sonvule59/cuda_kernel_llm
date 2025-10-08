#include <iostream>
#include <cuda_runtime.h>

__device__ void lossFunction(float* z, float* y, float* dL, int c) {
    float L = 0.f;
    for (int j = 0; j < c; ++j) {
        L += logf(expf(z[j]) - expf(dL[j]));
    }
    L -= y[0] * dL[0];
    for (int j = 1; j < c; ++j) {
        L -= y[j] * (logf(expf(dL[j])) - logf(expf(z[j]) - expf(dL[j])));
    }
    __syncthreads();
    L /= c;
    if (threadIdx.x == 0) {
        dL[0] = L;
    }
}

void calculateCrossEntropyLoss(float* predictedLogits, float* trueLabels, float* loss, int N, int C) {
    float* dL = new float[C];

    dim3 blockSize(32);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 1);

    float* dev_predictedLogits;
    float* dev_trueLabels;
    float* dev_dL;
    cudaMalloc((void**)&dev_predictedLogits, N * C * sizeof(float));
    cudaMalloc((void**)&dev_trueLabels, N * sizeof(float));
    cudaMalloc((void**)&dev_dL, C * sizeof(float));

    cudaMemcpy(dev_predictedLogits, predictedLogits, N * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_trueLabels, trueLabels, N * sizeof(float), cudaMemcpyHostToDevice);

    lossFunction<<<gridSize, blockSize>>>(dev_predictedLogits, dev_trueLabels, dev_dL, C);
    cudaMemcpy(loss, dev_dL, C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_predictedLogits);
    cudaFree(dev_trueLabels);
    cudaFree(dev_dL);
}
// Usage:
// Prepare input arrays
float predictedLogits[N][C];
float trueLabels[N];

// Call the function to compute the loss
float loss;
calculateCrossEntropyLoss(predictedLogits, trueLabels, &loss, N, C);