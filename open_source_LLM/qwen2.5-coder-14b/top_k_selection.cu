#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << err << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }
    __global__ void findKlargest(float *input, float *output, int N, int k) {
        __shared__ float shared_mem[k];
    
        // Load data into shared memory
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            shared_mem[threadIdx.x] = input[idx];
        }
    
        __syncthreads();
    
        // Use a simple bubble sort to arrange the elements in shared memory
        for (int i = 0; i < k; ++i) {
            for (int j = 1; j < k - i; ++j) {
                if (shared_mem[j] > shared_mem[j - 1]) {
                    float temp = shared_mem[j];
                    shared_mem[j] = shared_mem[j - 1];
                    shared_mem[j - 1] = temp;
                }
            }
        }
    
        // Write the sorted elements to the output array
        if (threadIdx.x < k) {
            output[threadIdx.x] = shared_mem[threadIdx.x];
        }
    }

void kLargestOnGPU(float *h_input, float *h_output, int N, int k) {
    float *d_input, *d_output;

    // Allocate memory on the device
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, k * sizeof(float)));

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch the kernel
    int threadsPerBlock = k;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    findKlargest<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N, k);

    // Copy the result back to the host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, k * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
}
int main() {
    const int N = 1024;
    const int k = 5;

    float h_input[N];
    float h_output[k];

    // Initialize input data (for example, random numbers)
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Call the GPU function
    kLargestOnGPU(h_input, h_output, N, k);

    // Print the result
    std::cout << "Top " << k << " largest elements:" << std::endl;
    for (int i = 0; i < k; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
