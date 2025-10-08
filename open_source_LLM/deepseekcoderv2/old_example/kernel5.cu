#include <iostream>
#include <cuda_runtime.h>

// Define a macro for CUDA error checking
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t status = call;                                           \
        if (status != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(status)); \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// CUDA kernel to perform reduction on threads within a block
__global__ void dotProductKernel(const float *A, const float *B, float *C, int N) {
    __shared__ float sdata[256];  // Shared memory for each thread block

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int tidx = threadIdx.x;

    if (tid < N) {
        sdata[tidx] = A[tid] * B[tid];  // Each thread computes the dot product for one element
    } else {
        sdata[tidx] = 0.0f;  // Out of bounds threads don't contribute to result
    }

    __syncthreads();  // Synchronize all threads in the block

    // Reduction: sum elements of 'sdata' array using shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tidx < s) {
            sdata[tidx] += sdata[tidx + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tidx == 0) {
        C[blockIdx.x] = sdata[0];
    }
}

int main() {
    const int N = 1 << 24;  // Example size: 2^24 elements
    float *h_A, *h_B, *h_C;  // Host arrays
    float *d_A, *d_B, *d_C;  // Device arrays

    // Allocate memory on host
    CUDA_CHECK(cudaMallocHost((void**)&h_A, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&h_B, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&h_C, (N / 256 + 1) * sizeof(float)));  // Sum of each block's result

    // Allocate memory on device
    CUDA_CHECK(cudaMalloc((void**)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, (N / 256 + 1) * sizeof(float)));

    // Initialize host arrays with some values
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel: N elements per block
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    dotProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result from device to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, (N / 256 + 1) * sizeof(float), cudaMemcpyDeviceToHost));

    // Sum up all partial sums in h_C to get final dot product
    float finalResult = 0.0f;
    for (int i = 0; i <= blocksPerGrid; ++i) {
        finalResult += h_C[i];
    }

    std::cout << "Dot Product: " << finalResult << std::endl;

    // Free memory on host and device
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}