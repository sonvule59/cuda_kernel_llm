#include <iostream>
#include <cuda_runtime.h>

// Kernel function to copy elements from A to B
__global__ void matrixCopy(float* A, float* B, int N) {
    // Calculate the global index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    intidy = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure the index is within bounds
    if (idx < N && idy < N) {
        B[idy * N + idx] = A[idy * N + idx];
    }
}

int main() {
    int N; // Size of the matrix
    std::cout << "Enter the size of the matrix (N): ";
    std::cin >> N;

    // Allocate memory for matrices A and B
    float* h_A = new float[N * N];
    float* h_B = new float[N * N];

    // Initialize matrix A with some values (for example, you can initialize it with zeros or random values)
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate memory on the device
    float* d_A, *d_B;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define block size and grid size
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    matrixCopy<<<gridSize, blockSize>>>(d_A, d_B, N);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy data back from device to host
    cudaMemcpy(h_B, d_B, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify the result (optional)
    bool isCorrect = true;
    for (int i = 0; i < N * N; ++i) {
        if (h_A[i] != h_B[i]) {
            isCorrect = false;
            break;
        }
    }

    if (isCorrect) {
        std::cout << "Matrix copy successful!" << std::endl;
    } else {
        std::cout << "Matrix copy failed." << std::endl;
    }

    // Free allocated memory
    delete[] h_A;
    delete[] h_B;
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}
