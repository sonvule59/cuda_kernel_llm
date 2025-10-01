#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuBLAS_v2.h>

// CUDA kernel for matrix multiplication
__global__ void matMulKernel(const float* A, const float* B, float* C, int widthA, int heightA, int widthB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < heightA && col < widthB) {
        float sum = 0.0;
        for (int k = 0; k < widthA; ++k) {
            sum += A[row * widthA + k] * B[k * widthB + col];
        }
        C[row * widthB + col] = sum;
    }
}

// CUDA kernel for matrix inversion using Gauss-Jordan elimination
__global__ void matrixInversionKernel(float* A, float* invA, int n) {
    __shared__ float s_A[16][16];

    // Load block of A into shared memory
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    s_A[ty][tx] = A[(by * blockDim.y + ty) * n + bx * blockDim.x + tx];

    __syncthreads();

    // Perform Gauss-Jordan elimination
    for (int p = 0; p < n; ++p) {
        if (tx == ty && bx == by) invA[p * n + p] = 1.0 / s_A[ty][tx];

        __syncthreads();

        if (p == tx) {
            for (int j = 0; j < n; ++j) s_A[ty][j] *= invA[p * n + p];
        }

        __syncthreads();

        for (int j = 0; j < n; ++j) {
            if (p != tx) s_A[ty][j] -= s_A[tx][j] * s_A[ty][p];
        }

        __syncthreads();

        if (p == ty && bx == by) invA[p * n + tx] = s_A[ty][tx];
    }

    // Write block of invA from shared memory to global memory
    invA[(by * blockDim.y + ty) * n + bx * blockDim.x + tx] = s_A[ty][tx];
}

int main() {
    // Example data: replace these with your actual data
    int n_samples = 1000;
    int n_features = 50;

    // Allocate host memory
    std::vector<float> X_host(n_samples * n_features);
    std::vector<float> y_host(n_samples);

    // Allocate device memory
    float* X_device, *y_device, *XT_X_device, *XT_X_inv_device, *X_transpose_device, *beta_device;
    cudaMalloc(&X_device, n_samples * n_features * sizeof(float));
    cudaMalloc(&y_device, n_samples * sizeof(float));

    // Copy data to device
    cudaMemcpy(X_device, X_host.data(), n_samples * n_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_device, y_host.data(), n_samples * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate additional device memory
    cudaMalloc(&XT_X_device, n_features * n_features * sizeof(float));
    cudaMalloc(&X_transpose_device, n_samples * n_features * sizeof(float));
    cudaMalloc(&XT_X_inv_device, n_features * n_features * sizeof(float));
    cudaMalloc(&beta_device, n_features * sizeof(float));

    // Matrix transpose using cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n_samples, n_features, &alpha, X_device, n_features, &beta, X_transpose_device, n_samples);
    cublasDestroy(handle);

    // Matrix multiplication (X^T X)
    int widthA = n_features, heightA = n_samples, widthB = n_features;
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((widthB + threadsPerBlock.x - 1) / threadsPerBlock.x, (heightA + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matMulKernel<<<blocksPerGrid, threadsPerBlock>>>(X_transpose_device, X_device, XT_X_device, widthA, heightA, widthB);

    // Matrix inversion (XT_X_inv)
    matrixInversionKernel<<<blocksPerGrid, threadsPerBlock>>>(XT_X_device, XT_X_inv_device, n_features);

    // Matrix multiplication ((X^T X)^-1 X^T)
    cuBLASHandle_t handle2;
    cublasCreate(&handle2);
    cublasSgeam(handle2, CUBLAS_OP_T, CUBLAS_OP_N, n_features, n_samples, &alpha, XT_X_inv_device, n_features, &beta, X_transpose_device, n_samples);
    cublasDestroy(handle2);

    // Matrix multiplication ((X^T X)^-1 X^T y)
    matMulKernel<<<blocksPerGrid, threadsPerBlock>>>(XT_X_inv_device, X_transpose_device, beta_device, n_features, n_samples, 1);

    // Copy result back to host
    cudaMemcpy(beta_host.data(), beta_device, n_features * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(X_device);
    cudaFree(y_device);
    cudaFree(XT_X_device);
    cudaFree(X_transpose_device);
    cudaFree(XT_X_inv_device);
    cudaFree(beta_device);

    // Print solution
    std::cout << "Coefficients β:" << std::endl;
    for (int i = 0; i < n_features; ++i) {
        std::cout << beta_host[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
