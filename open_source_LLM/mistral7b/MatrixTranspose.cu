#include <stdio.h>
#include <cuda_runtime.h>

typedef struct {
    int height;
    int width;
    int channels;
    float *data;
} Tensor;

Tensor input;
Tensor output;


// kernel function:
__global__ void max_pooling_kernel(const float *inputData, float *outputData, int kernelSize, int stride) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int channelId = threadIdx.z;

    if (tx >= kernelSize || ty >= kernelSize) return;

    float maxVal = inputData[(ty * input.height + tx) * input.channels + channelId];
    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            int newTx = tx + i - stride / 2;
            int newTy = ty + j - stride / 2;

            if (newTx >= 0 && newTy >= 0 && newTx < input.height && newTy < input.width) {
                float currentVal = inputData[(newTy * input.height + newTx) * input.channels + channelId];
                if (currentVal > maxVal) maxVal = currentVal;
            }
        }
    }

    outputData[(ty * output.height / stride + tx / stride) * output.channels + channelId] = maxVal;
}

// host code for tensor handling:

void initialize_tensors(int height, int width, int channels, float *inputData, Tensor &inputTensor, Tensor &outputTensor) {
    inputTensor.height = height;
    inputTensor.width = width;
    inputTensor.channels = channels;
    inputTensor.data = inputData;

    outputTensor.height = (height + 2) / 2 * 2;
    outputTensor.width = (width + 2) / 2 * 2;
    outputTensor.channels = channels;
}

void max_pooling(int kernelSize, int stride, float *inputData, Tensor &inputTensor, Tensor &outputTensor) {
    constexpr int threadsPerBlock = 256;
    constexpr int blocksPerGridX = (inputTensor.width + threadsPerBlock - 1) / threadsPerBlock;
    constexpr int blocksPerGridY = (inputTensor.height + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc((void **)&outputTensor.data, outputTensor.height * outputTensor.width * outputTensor.channels * sizeof(float));

    max_pooling_kernel<<<blocksPerGridY, blocksPerGridX, threadsPerBlock>>>(inputData, (float *)outputTensor.data, kernelSize, stride);
}
// Call the function
int main() {
    int height = 10;
    int width = 10;
    int channels = 3;

    float *inputData = new float[height * width * channels]; // Allocate input tensor on the host

    // Populate the input data with your values here

    Tensor inputTensor, outputTensor;
    initialize_tensors(height, width, channels, inputData, inputTensor, outputTensor);
    max_pooling(2, 2, inputData, inputTensor, outputTensor);

    // Now the output tensor (outputTensor) contains the pooled data

    return 0;
}