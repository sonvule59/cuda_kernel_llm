#include <stdio.h>
#include <cuda_runtime.h>

__global__ void prefix_sum_kernel(float* input, float* output, int n) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;

    // Load data into shared memory
    if (idx < n) {
        shared[tid] = input[idx];
    } else {
        shared[tid] = 0.0f;
    }
    __syncthreads();

    // Up-sweep
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        if (tid >= offset && idx + offset < n) {
            shared[tid] += shared[tid - offset];
        }
        __syncthreads();
    }

    // Down-sweep
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid >= offset && idx + offset < n) {
            shared[tid] += shared[tid - offset];
        }
        __syncthreads();
    }

    // Write result to output
    if (idx < n) {
        output[idx] = shared[tid];
    }
}

void prefix_sum(float* input, float* output, int n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice);

    prefix_sum_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_input, d_output, n);

    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    int n = 1024;
    float h_input[1024];
    float h_output[1024];

    // Initialize input array
    for (int i = 0; i < n; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    prefix_sum(h_input, h_output, n);

    // Print output
    for (int i = 0; i < n; ++i) {
        printf("%f ", h_output[i]);
    }
    printf("\n");

    return 0;
}
