// main.cu - Contains the main function that initializes CUDA, allocates memory for input and output arrays, launches the kernel function, copies results back to the CPU, and frees the allocated memory.

#include <iostream>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const int N = 1024;
const int k = 5;

void checkCudaErrors(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << '\n';
    }
}

__global__ void kLargest(float* d_input, float* d_output, int* d_index, int k);

int main() {
    float *d_input;
    float *d_output;
    float *h_input;
    float *h_output;
    int *d_index;
    int *h_index;

    // Allocate device memory for input, output, and index arrays
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, k * sizeof(float));
    cudaMalloc((void**)&d_index, N * sizeof(int));

    // Allocate host memory for input, output, and index arrays
    h_input = new float[N];
    h_output = new float[k];
    h_index = new int[N];

    // Fill in the host input array with data
    // For example:
    // for (int i = 0; i < N; ++i) {
    //     h_input[i] = static_cast<float>(i);
    // }

    // Copy input array to device memory
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Launch the kernel function to find k largest elements
    kLargest<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, d_index, k);

    // Copy results from device memory to host memory
    cudaMemcpy(h_output, d_output, k * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < k; ++i) {
        std::cout << h_output[i] << " ";
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_index);

    // Free host memory
    delete[] h_input;
    delete[] h_output;
    delete[] h_index;

    return 0;
}


// kLargest.cu - Contains the CUDA kernel function that finds the k largest elements in a given array. This file includes the sort_k_largest.cu source file for the mergesort implementation.

__global__ void kLargest(float* d_input, float* d_output, int* d_index, int k) {
    sort_k_largest(d_input, d_index, 0, N - 1, k);

    // Copy sorted elements to output array
    for (int i = 0; i < k; ++i) {
        d_output[i] = d_input[d_index[i]];
    }
}

// sort_k_largest.cu - Contains the mergesort implementation used in the CUDA kernel function.
__device__ void merge(float* d_left, float* d_right, float* d_temp, int* d_idx_left, int* d_idx_right, int len) {
    // Merge sort implementation goes here (see the previous answer for reference)
}

__device__ void mergeSort(float* d_input, int* d_index, int lo, int hi, int k) {
    if (lo >= hi) return;

    int mid = lo + (hi - lo) / 2;
    __syncthreads();

    mergeSort<<<1, 32>>>(d_input, d_index, lo, mid, k);
    mergeSort<<<1, 32>>>(d_input, d_index, mid + 1, hi, k);

    int idxLeft = blockIdx.x * 32 + threadIdx.x;
    __syncthreads();

    if (idxLeft < k) merge(d_input + d_index[lo + idxLeft], d_input + d_index[mid + idxLeft + 1], d_temp, d_idx_left + idxLeft, d_idx_right + idxLeft, hi - mid - 1);

    __syncthreads();

    if (idxLeft < k && lo + idxLeft < mid + idxLeft + 1) {
        if (d_input[d_index[lo + idxLeft]] > d_input[d_index[mid + idxLeft + 1]]) {
            swap(d_index[lo + idxLeft], d_index[mid + idxLeft + 1]);
            swap(d_index[mid + idxLeft + 1], d_index[hi - (idxLeft - lo)]);
        }
    }
}