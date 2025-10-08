#include <iostream>
#include <vector_types.h>
#include <cuda_runtime.h>

const int N = 1024;

__global__ void dotProductKernel(float* A, float* B, float* C) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < N) {
        int reductionIndex = log2(N);
        int localIndex = threadIdx.x;
        int localBase = blockDim.x * blockIdx.x;
        float temp = 0;
        while (reductionIndex > 0) {
            if (localIndex % 2 == 0)
                temp += A[index] * B[index] + C[(localBase + localIndex) / 2];
            index >>= reductionIndex;
            localIndex >>= reductionIndex;
            reductionIndex--;
        }
        C[index] = temp;
    }
}

int main(void) {
    float* A_host, *B_host, *C_host, *D_device;
    size_t size = N * sizeof(float);

    // Allocate memory on the host for input vectors and result array
    cudaMalloc((void**)&A_host, size);
    cudaMalloc((void**)&B_host, size);
    cudaMalloc((void**)&C_host, size);

    // Fill in your input data here (for simplicity, I'll use all 1s)
    for (int i = 0; i < N; ++i) {
        A_host[i] = 1.f;
        B_host[i] = 1.f;
    }

    // Allocate device memory and copy host data to the device
    cudaMalloc((void**)&D_device, size);
    cudaMemcpy(D_device, A_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(D_device + size, B_host, size, cudaMemcpyHostToDevice);

    // Set up the kernel launch configuration
    dim3 threadsPerBlock(64);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Launch the CUDA kernel
    dotProductKernel<<<blocksPerGrid, threadsPerBlock>>>(D_device, D_device + N, D_device);

    // Copy result back to the host
    float* C_device;
    cudaMalloc((void**)&C_device, size);
    cudaMemcpy(C_device, D_device, size, cudaMemcpyDeviceToHost);

    // Sum up the result on the host since there might be some carry-over from reduction
    float sum = 0.f;
    for (int i = 0; i < N; ++i) {
        sum += C_device[i];
    }

    std::cout << "Dot Product: " << sum << std::endl;

    // Cleanup memory allocations
    cudaFree(A_host);
    cudaFree(B_host);
    cudaFree(C_host);
    cudaFree(D_device);

    return 0;
}