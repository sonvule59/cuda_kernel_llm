#include <iostream>
#include <cuda_runtime.h>

const int BLOCK_SIZE = 256;

__global__ void bubbleSort(float *arr, int n) {
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (arr[i] > arr[j]) {
                float temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
    }
}

int main(void) {
    int n = 1024; // number of elements in the array
    float *d_arr, *h_arr;
    cudaMalloc((void **)&d_arr, n * sizeof(float));
    h_arr = new float[n];

    for (int i = 0; i < n; ++i) {
        h_arr[i] = static_cast<float>(rand()) / RAND_MAX; // initialize array with random values
    }

    cudaMemcpy(d_arr, h_arr, n * sizeof(float), cudaMemcpyHostToDevice);

    bubbleSort<<<1, BLOCK_SIZE>>>(d_arr, n);

    cudaMemcpy(h_arr, d_arr, n * sizeof(float), cudaMemcpyDeviceToHost);

    delete[] h_arr;

    // Verify the sorted array
    bool isSorted = true;
    for (int i = 0; i < n - 1; ++i) {
        if (h_arr[i] > h_arr[i + 1]) {
            isSorted = false;
            break;
        }
    }

    std::cout << (isSorted ? "Sorted!" : "Not sorted!") << std::endl;

    cudaFree(d_arr);
    return 0;
}