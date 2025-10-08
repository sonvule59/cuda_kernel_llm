// ### Device Code (.cu file)

// define the CUDA kernel for max pooling.

#include <cuda_runtime.h>
#include <iostream>

// Kernel function to perform 2D max pooling
__global__ void maxPoolingKernel(float* input, float* output, int height, int width, int channels) {
    // Calculate global thread index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;

    // Check if the current thread is within bounds
    if (x < width && y < height && c < channels) {
        // Calculate output indices
        int pooledWidth = width / 2;
        int pooledHeight = height / 2;
        int outX = x / 2;
        int outY = y / 2;
        int outC = c;

        // Calculate input index
        int inIdx = (y * width + x) * channels + c;
        int outIdx = (outY * pooledWidth + outX) * channels + outC;

        // Perform max pooling
        float currentMax = input[inIdx];
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                int neighborIdx = ((y * width + x) * channels + c);
                float value = input[neighborIdx];
                if (value > currentMax) {
                    currentMax = value;
                }
            }
        }
        output[outIdx] = currentMax;
    }
}
```

// ### Host Code

// Host code to handle tensor operations and allocate memory on both the host and device.

#include <cuda_runtime.h>
#include <iostream>

// Function prototypes
void maxPooling(float* input, float* output, int height, int width, int channels);
__global__ void maxPoolingKernel(float* input, float* output, int height, int width, int channels);
void checkCudaError(cudaError_t err, const char* msg);

int main() {
    // Define input and output tensor dimensions
    int height = 4;
    int width = 4;
    int channels = 3;

    // Allocate memory for the input tensor on host
    float* h_input = new float[height * width * channels];
    // Initialize the input tensor with some values (example)
    for (int i = 0; i < height * width * channels; ++i) {
        h_input[i] = static_cast<float>(rand() % 100);
    }

    // Allocate memory for the output tensor on host
    float* h_output = new float[(height / 2) * (width / 2) * channels];

    // Allocate memory for the input tensor on device
    float* d_input;
    checkCudaError(cudaMalloc(&d_input, height * width * channels * sizeof(float)), "Failed to allocate device memory for input");

    // Allocate memory for the output tensor on device
    float* d_output;
    checkCudaError(cudaMalloc(&d_output, (height / 2) * (width / 2) * channels * sizeof(float)), "Failed to allocate device memory for output");

    // Copy input data from host to device
    checkCudaError(cudaMemcpy(d_input, h_input, height * width * channels * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy data from host to device");

    // Define block and grid dimensions
    dim3 blockSize(8, 8, 1); // Adjust based on your GPU capabilities
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, channels);

    // Launch the kernel
    maxPoolingKernel<<<gridSize, blockSize>>>(d_input, d_output, height, width, channels);

    // Check for any CUDA errors
    checkCudaError(cudaGetLastError(), "Kernel launch failed");

    // Copy output data from device to host
    checkCudaError(cudaMemcpy(h_output, d_output, (height / 2) * (width / 2) * channels * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy data from device to host");

    // Print the output tensor for verification (optional)
    std::cout << "Output:" << std::endl;
    for (int i = 0; i < (height / 2) * (width / 2) * channels; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    delete[] h_input;
    delete[] h_output;

    return 0;
}

// Function to perform max pooling
void maxPooling(float* input, float* output, int height, int width, int channels) {
    cudaError_t err;
    float *d_input, *d_output;

    // Allocate memory for the input tensor on device
    err = cudaMalloc(&d_input, height * width * channels * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for input: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Allocate memory for the output tensor on device
    err = cudaMalloc(&d_output, (height / 2) * (width / 2) * channels * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for output: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        return;
    }

    // Copy input data from host to device
    err = cudaMemcpy(d_input, input, height * width * channels * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy data from host to device: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }

    // Define block and grid dimensions
    dim3 blockSize(8, 8, 1); // Adjust based on your GPU capabilities
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, channels);

    // Launch the kernel
    maxPoolingKernel<<<gridSize, blockSize>>>(d_input, d_output, height, width, channels);

    // Check for any CUDA errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }

    // Copy output data from device to host
    err = cudaMemcpy(output, d_output, (height / 2) * (width / 2) * channels * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy data from device to host: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

// Function to check CUDA errors
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}