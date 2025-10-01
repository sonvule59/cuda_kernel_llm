#include <iostream>
#include <cuda_runtime.h>

// Kernel function to reverse an array in-place
__global__ void reverseArray(float* arr, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n / 2) {
        int otherIdx = n - idx - 1;
        float temp = arr[idx];
        arr[idx] = arr[otherIdx];
        arr[otherIdx] = temp;
    }
}

int main() {
    // Example array of 32-bit floating point numbers
    float arr[] = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f};
    int n = sizeof(arr) / sizeof(float);

    // Allocate device memory
    float* d_arr;
    cudaMalloc((void**)&d_arr, n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_arr, arr, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    reverseArray<<<numBlocks, blockSize>>>(d_arr, n);

    // Copy data back to host
    cudaMemcpy(arr, d_arr, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_arr);

    // Print the reversed array
    std::cout << "Reversed array: ";
    for (int i = 0; i < n; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
