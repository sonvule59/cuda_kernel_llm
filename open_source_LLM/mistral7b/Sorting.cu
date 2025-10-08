#include <iostream>
#include <vector>
#define N 1024

__global__ void multiply(float* C, const float* A, const float* B) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N)
        C[idx] = A[idx] * B[idx];
}

int main() {
    float* h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    size_t size = N * sizeof(float);

    // Allocate host memory for input and output arrays
    h_A = new float[N];
    h_B = new float[N];
    h_C = new float[N];

    // Allocate device memory for input and output arrays
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy host data to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Set up input arrays with random values
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Set up kernel grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel on the GPU
    multiply<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B);

    // Copy device data back to host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Clean up memory and print the results
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    for (int i = 0; i < N; ++i) {
        std::cout << "Result[" << i << "] = " << h_C[i] << std::endl;
    }

    return 0;
}