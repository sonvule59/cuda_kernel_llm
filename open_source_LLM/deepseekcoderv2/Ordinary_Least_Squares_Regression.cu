#include <iostream>
#include <vector>
#include <cuda_runtime.h>

const int BLOCK_SIZE = 32;

void matVecMult(float* d_A, float* d_X, int rows, int cols);
void matMatMultInvTrans(float* d_A, float* d_Beta, float* d_AT, int cols);

__global__ void matVecMult(float* d_A, float* d_X, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows) {
        float sum = 0.f;
        for (int i = 0; i < cols; ++i) {
            int jdx = idx * cols + i;
            sum += d_X[jdx] * d_X[(cols + i) * rows + idx];
        }
        d_A[idx * (cols + 1)] = sum;
        for (int k = 1; k < idx; ++k) {
            d_A[idx * (cols + 1) + k] = sum;
        }
    }
}

__global__ void matMatMultInvTrans(float* d_A, float* d_Beta, float* d_AT, int cols) {
    // Inverse of a matrix and its transpose are the same when it's square and symmetric.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < cols) {
        float sum = 0.f;
        for (int i = 0; i < cols; ++i) {
            sum += d_AT[idx * cols + i] * d_A[(cols + i) * (cols + 1) + idx];
        }
        d_Beta[idx] = d_y[idx] / sum;
    }
}

void cudaOls(const std::vector<std::vector<float>>& X, const std::vector<float>& y, std::vector<float>& beta) {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((X.size() + blockDim.x - 1) / blockDim.x, (X[0].size() + blockDim.y - 1) / blockDim.y);

    float* d_X;
    float* d_y;
    float* d_A;
    float* d_beta;

    cudaMalloc((void**)&d_X, X.size() * X[0].size() * sizeof(float));
    cudaMemcpy(d_X, X.data(), X.size() * X[0].size() * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_y, X.size() * sizeof(float));
    cudaMemcpy(d_y, y.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_A, (X[0].size() + 1) * (X[0].size() + 1) * sizeof(float));
    cudaMalloc((void**)&d_beta, X[0].size() * sizeof(float));

    matVecMult<<<gridDim, blockDim>>>(d_A, d_X, X.size(), X[0].size());
    matMatMultInvTrans<<<gridDim, blockDim>>>(d_A, d_beta, d_A, X[0].size());

    cudaMemcpy(beta.data(), d_beta, X[0].size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_A);
    cudaFree(d_beta);
}

int main() {
    // Initialize the X and y data...
    std::vector<std::vector<float>> X = ...;
    std::vector<float> y = ...;

    std::vector<float> beta(X[0].size());

    cudaOls(X, y, beta);

    // Print the solution...
    for (const auto& b : beta) {
        std::cout << b << ' ';
    }

    return 0;
}