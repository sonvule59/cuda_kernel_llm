#include <iostream>
#include <cuda_runtime.h>

const int BLOCK_SIZE = 32;
const int GRID_SIZE = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;

__global__ void transposeKernel(float* A, float* AT, int rows, int cols) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = gid * cols;

    if (gid < rows) {
        for (int j = 0; j < cols; ++j) {
            int aid = lid + j;
            AT[aid] = A[gid];
            AT[(rows * j) + gid] = A[aid];
        }
    }
}

int main() {
    int rows = 1024;
    int cols = 1024;
    size_t bytes = rows * cols * sizeof(float);

    float* d_A, *d_AT;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_AT, bytes);
    transposeKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_A, d_AT, rows, cols);

    cudaDeviceSynchronize();
    cudaFree(d_A);
    cudaFree(d_AT);

    return 0;
}