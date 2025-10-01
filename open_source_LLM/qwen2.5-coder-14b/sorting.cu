#include <iostream>
#include <cuda_runtime.h>

__global__ void bubbleSort(float* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j, temp;

    // Each thread will perform multiple passes over the array
    for (j = 0; j < n - i - 1; ++j) {
        // Perform the bubble sort swap operation
        if (arr[j] > arr[j + 1]) {
            temp = arr[j];
            arr[j] = arr[j + 1];
            arr[j + 1] = temp;
        }
    }
}

int main() {
    // Example array to be sorted
    float arr[] = {5.3f, 2.1f, 8.6f, 1.4f, 9.7f};
    int n = sizeof(arr) / sizeof(arr[0]);

    // Allocate memory on the device
    float* dev_arr;
    cudaMalloc((void**)&dev_arr, n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(dev_arr, arr, n * sizeof(float), cudaMemcpyHostToDevice);

    // Define the number of blocks and threads
    int numBlocks = 1;
    int blockSize = n;

    // Launch the kernel
    bubbleSort<<<numBlocks, blockSize>>>(dev_arr, n);

    // Copy the sorted data back from device to host
    cudaMemcpy(arr, dev_arr, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the sorted array
    std::cout << "Sorted array: ";
    for (int i = 0; i < n; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(dev_arr);

    return 0;
}
