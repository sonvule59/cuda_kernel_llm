#include <cuda_runtime.h>
#include <iostream>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        std::cerr << "Error: " << cudaGetErrorString(error) << std::endl;     \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

template<typename T>
using Matrix = std::vector<std::vector<T>>;

// Function to compute softmax
__device__ T softmax(T x, T* max) {
    *max = fmaxf(*max, x);
    return exp(x - *max);
}

// CUDA kernel for single head attention
__global__ void attention_kernel(float* Q, float* K, float* V, float* attn_output, int N, int dmodel) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        float max_val = -1e38f;
        float sum = 0.0f;

        for (int k = 0; k < dmodel; ++k) {
            sum += __expf(Q[row * dmodel + k] * K[col * dmodel + k] - max_val);
        }

        attn_output[row * N + col] = sum;
    }
}

// CUDA kernel for multi-head attention
__global__ void multi_head_attention_kernel(float* Q, float* K, float* V, float* attn_output, int N, int dmodel, int h) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        for (int head = 0; head < h; ++head) {
            int offset = head * N * dmodel;
            float max_val = -1e38f;

            // Compute attention for each head
            float attn_value = 0.0;
            for (int k = 0; k < dmodel / h; ++k) {
                attn_value += __expf(Q[row * (dmodel / h) + k + offset] *
                                        K[col * (dmodel / h) + k + offset] - max_val);
            }

            // Store the result
            attn_output[(row * N + col) * h + head] = attn_value;
        }
    }
}

// Function to launch the CUDA kernel
void multi_head_attention(Matrix<float>& Q, Matrix<float>& K, Matrix<float>& V, Matrix<float>& attn_output, int h) {
    int N = Q.size();
    int dmodel = Q[0].size();

    float* d_Q, *d_K, *d_V, *d_attn_output;
    CHECK(cudaMalloc(&d_Q, N * dmodel * sizeof(float)));
    CHECK(cudaMalloc(&d_K, N * dmodel * sizeof(float)));
    CHECK(cudaMalloc(&d_V, N * dmodel * sizeof(float)));
    CHECK(cudaMalloc(&d_attn_output, N * N * h * sizeof(float)));

    // Copy data to device
    CHECK(cudaMemcpy(d_Q, Q.data(), N * dmodel * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_K, K.data(), N * dmodel * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_V, V.data(), N * dmodel * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    multi_head_attention_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_Q, d_K, d_V, d_attn_output, N, dmodel, h);

    // Copy result back to host
    CHECK(cudaMemcpy(attn_output.data(), d_attn_output, N * N * h * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK(cudaFree(d_Q));
    CHECK(cudaFree(d_K));
    CHECK(cudaFree(d_V));
    CHECK(cudaFree(d_attn_output));
}

int main() {
    int N = 10; // Batch size
    int dmodel = 32; // Model dimension
    int h = 8; // Number of heads

    Matrix<float> Q(N, std::vector<float>(dmodel));
    Matrix<float> K(N, std::vector<float>(dmodel));
    Matrix<float> V(N, std::vector<float>(dmodel));
    Matrix<float> attn_output(N, std::vector<float>(N * h));

    // Initialize Q, K, V with some values
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < dmodel; ++j) {
            Q[i][j] = i + j;
            K[i][j] = i - j;
            V[i][j] = i * j;
        }
    }

    // Run multi-head attention
    multi_head_attention(Q, K, V, attn_output, h);

    // Print the output
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N * h; ++j) {
            std::cout << attn_output[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
