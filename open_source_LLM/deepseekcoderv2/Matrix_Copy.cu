#include <cuda.h>
#include <curand_kernel.h>

#define N 1024 // Change this value to your desired matrix size

__global__ void copyMatrix(float* A, float* B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= N || j >= N) return;

    B[i * N + j] = A[i * N + j];
}

int main() {
    float* A_device, *B_device;
    float* A_host, *B_host;
    size_t size = N * N * sizeof(float);

    // Allocate and copy input matrix A to host memory
    A_host = (float*)malloc(size);
    cudaMalloc((void**)&A_device, size);
    cudaMemcpy(A_device, A_host, size, cudaMemcpyHostToDevice);

    // Allocate output matrix B on both host and device
    B_host = (float*)malloc(size);
    cudaMalloc((void**)&B_device, size);

    // Set all elements of output matrix B to zero
    cudaMemset(B_device, 0, size);

    dim3 blocks(sqrt(N), sqrt(N));
    dim3 threads(blocks.x * 2, blocks.y * 2); // Launch multiple threads per block for better coalesced memory access
    copyMatrix<<<blocks, threads>>>(A_device, B_device);

    // Copy the result from device to host
    cudaMemcpy(B_host, B_device, size, cudaMemcpyDeviceToHost);

    // Free device memory and clean up
    free(A_host);
    free(B_host);
    cudaFree(A_device);
    cudaFree(B_device);

    return 0;
}