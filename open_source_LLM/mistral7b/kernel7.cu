#include <iostream>
#define N 1024

__global__ void reduce(int *d_data, int *d_scratch) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        __shared__ int localData[N];
        int blockStart = idx - (blockDim.x >> 1) * blockIdx.x;

        localData[threadIdx.x] = d_data[idx];
        __syncthreads();

        for (int i = blockStart; i < N && i < blockStart + (blockDim.x >> 1); ++i) {
            if (localData[i] > localData[threadIdx.x])
                localData[threadIdx.x] = localData[i];
            __syncthreads();
        }

        if (threadIdx.x == 0)
            d_scratch[blockIdx.x] = localData[0];
        __syncthreads();
    }
}

__global__ void sumReduction(int *d_scratch, int *d_output) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        d_output[0] = d_scratch[0];
        for (int i = 1; i < blockDim.x; ++i) {
            d_output[0] += d_scratch[i];
        }
    }
}

void host_code() {
    int *h_data, *d_data, *d_scratch, *d_output;
    size_t bytesPerInt = sizeof(int);
    cudaMalloc((void **)&d_data, N * bytesPerInt);
    cudaMalloc((void **)&d_scratch, (blockDim.x * bytesPerInt));
    cudaMalloc((void **)&d_output, bytesPerInt);

    h_data = new int[N];
    for (int i = 0; i < N; ++i) {
        h_data[i] = i;
    }

    cudaMemcpy(d_data, h_data, N * bytesPerInt, cudaMemcpyHostToDevice);

    int blockCount = (N + blockDim.x - 1) / blockDim.x;
    reduce<<<blockCount, blockDim.x>>>(d_data, d_scratch);
    sumReduction<<<1, 1>>>(d_scratch, d_output);

    int output;
    cudaMemcpy(&output, d_output, bytesPerInt, cudaMemcpyDeviceToHost);
    std::cout << "Sum of the array is: " << output << std::endl;

    delete[] h_data;
    cudaFree(d_data);
    cudaFree(d_scratch);
    cudaFree(d_output);
}

int main() {
    host_code();
    return 0;
}