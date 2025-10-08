#include <cuda_runtime.h>
#include <math.h>

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

void kernel_sigmoid(float *d_input, float *d_output, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N)
        d_output[idx] = sigmoid(d_input[idx]);
}

void host_sigmoid(float *h_input, float *h_output, int N) {
    cudaMalloc((void**)&d_input, sizeof(float) * N);
    cudaMemcpy(d_input, h_input, sizeof(float) * N, cudaMemcpyHostToDevice);

    dim3 blocks(N / 256 + (N % 256 > 0 ? 1 : 0), 1, 1);
    dim3 threads(256, 1, 1);

    float *d_output;
    cudaMalloc((void**)&d_output, sizeof(float) * N);

    kernel_sigmoid<<<blocks, threads>>>(d_input, d_output, N);

    cudaMemcpy(h_output, d_output, sizeof(float) * N, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    const int N = 1024;
    float h_input[N];
    float h_output[N], ref_output[N];

    // Fill input array with random values for testing
    for (int i = 0; i < N; ++i) {
        h_input[i] = rand() / RAND_MAX * 10 - 5;
        ref_output[i] = 1.0f / (1.0f + exp(-h_input[i]));
    }

    host_sigmoid(h_input, h_output, N);

    // Check if results are close to reference implementation
    float max_error = 0.0f;
    for (int i = 0; i < N; ++i) {
        float error = fabs(h_output[i] - ref_output[i]);
        max_error = std::max(max_error, error);
    }

    printf("Max error: %.8f\n", max_error);

    return 0;
}