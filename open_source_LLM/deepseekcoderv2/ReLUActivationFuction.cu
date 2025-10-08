#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

__global__ void reluKernel(float *d_input, float *d_output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    d_output[idx] = (d_input[idx] > 0) ? d_input[idx] : 0;
}

int main() {
    const int size = 32;
    float *h_input, *h_output, *d_input, *d_output;

    h_input = new float[size];
    h_output = new float[size];

    cudaMalloc((void**)&d_input, size*sizeof(float));
    cudaMalloc((void**)&d_output, size*sizeof(float));

    for (int i = 0; i < size; ++i) {
        h_input[i] = -3.5 + i * 0.7; // Generate example input vector
    }

    cudaMemcpy(d_input, h_input, size*sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(size);
    dim3 blocksPerGrid(1, 1, 1);

    reluKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size);

    cudaMemcpy(h_output, d_output, size*sizeof(float), cudaMemcpyDeviceToHost);

    delete [] h_input;
    delete [] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}