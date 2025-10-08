#include <stdio.h>
#include <cuda_runtime.h>

__global__ void convolveKernel(float *d_input, float *d_kernel, float *d_output, int inputRows, int inputCols, int kernelRows, int kernelCols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= inputRows || j >= inputCols || i < 0 || j < 0) return;

    float sum = 0.0f;
    for (int m = max(0, i - kernelRows + 1); m < min(inputRows, i + kernelRows); ++m) {
        for (int n = max(0, j - kernelCols + 1); n < min(inputCols, j + kernelCols); ++n) {
            sum += d_input[i * inputCols + j] * d_kernel[(m - i) * kernelCols + (n - j)];
        }
    }
    if (i * inputCols + j < outputRows * outputCols) {
        d_output[i * outputCols + j] = sum;
    }
}

int main() {
    // Set device and stream
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cudaSetDevice(prop.major > 3 ? 0 : 1);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocate memory on the GPU
    int inputRows = ...;
    int inputCols = ...;
    int kernelRows = ...;
    int kernelCols = ...;
    float *d_input, *d_kernel, *d_output;
    cudaMalloc((void **)&d_input, inputRows * inputCols * sizeof(float));
    cudaMalloc((void **)&d_kernel, kernelRows * kernelCols * sizeof(float));
    cudaMalloc((void **)&d_output, (outputRows = inputRows - kernelRows + 1) * (outputCols = inputCols - kernelCols + 1) * sizeof(float));

    // Copy data from the host to the device
    float *h_input = ...;
    float *h_kernel = ...;
    cudaMemcpy(d_input, h_input, inputRows * inputCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelRows * kernelCols * sizeof(float), cudaMemcpyHostToDevice);

    // Set block and grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (inputRows - kernelRows + 1) / threadsPerBlock + ((inputRows - kernelRows + 1) % threadsPerBlock != 0 ? 1 : 0);
    blocksPerGrid *= (inputCols - kernelCols + 1) / threadsPerBlock + ((inputCols - kernelCols + 1) % threadsPerBlock != 0 ? 1 : 0);

    // Launch the kernel on the GPU
    convolveKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_kernel, d_output, inputRows, inputCols, kernelRows, kernelCols);

    // Synchronize and copy data from the device to the host
    cudaStreamSynchronize(stream);
    float *h_output = (float *)malloc((outputRows * outputCols) * sizeof(float));
    cudaMemcpy(h_output, d_output, (outputRows * outputCols) * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory on the device and stream
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    cudaStreamDestroy(stream);

    return 0;
}