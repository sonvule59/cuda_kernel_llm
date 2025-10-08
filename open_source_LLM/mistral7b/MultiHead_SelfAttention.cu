#include <cub/cub.cuh>

template<typename T>
__global__ void attention_kernel(const T* q, const T* k, const T* v, T* output, int num_heads, int seq_len, int dmodel) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= num_heads * seq_len) return;

    int head = threadId / seq_len;
    int queryIndex = threadId % seq_len;

    T* qi = q + head * seq_len;
    T* ki = k + head * seq_len;
    T* vi = v + head * seq_len;
    T* scoreBuffer = new T[seq_len];
    T* attentionScores = new T[seq_len];

    // Calculate scores
    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSetMatrix(handle, CUBLAS_ROW_MAJOR, CUBLAS_DOUBLE, seq_len, seq_len, 1, qi, 1, ki, 1, scoreBuffer, 1);
    cublasSqrts(handle, CUBLAS_N, seq_len, scoreBuffer, 1, attentionScores, 1);

    // Normalize scores with softmax
    T sum = 0.0;
    for (int i = 0; i < seq_len; ++i) {
        sum += exp(attentionScores[i]);
    }

    cublasSetMatrix(handle, CUBLAS_ROW_MAJOR, CUBLAS_DOUBLE, seq_len, 1, 1.0 / sum, attentionScores, 1, attentionScores, 1);

    // Compute weighted sum of values
    cublasSetMatrix(handle, CUBLAS_ROW_MAJOR, CUBLAS_DOUBLE, seq_len, dmodel, 1, attentionScores, 1, vi, dmodel, output + head * dmodel);

    delete[] scoreBuffer;
    delete[] attentionScores;
    cublasDestroy(handle);
}

void multiHeadAttention(T* Q, T* K, T* V, int num_heads, int seq_len, int dmodel, int batch_size) {
    dim3 blockSize(num_heads * 64, 1, 1); // Adjust the number of threads per block to optimize for your hardware
    dim3 gridSize((seq_len + blockSize.x - 1) / blockSize.x, 1, 1); // Adjust the number of blocks based on the input size

    T* output = new T[num_heads * dmodel * batch_size];
    attention_kernel<<<gridSize, blockSize>>>(Q, K, V, output, num_heads, seq_len, dmodel);

    // Reshape the output matrix into a 3-D tensor (batch_size, num_heads, dmodel)
    T* reshapedOutput = new T[batch_size * num_heads * dmodel];
    cudaMemcpy(reshapedOutput, output, batch_size * num_heads * dmodel * sizeof(T), cudaMemcpyDeviceToHost);

    // Reshape the output tensor
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int i = 0; i < dmodel; ++i) {
                reshapedOutput[b * num_heads * dmodel + h * dmodel + i] = output[b * dmodel + h + b * num_heads];
            }
        }
    }

    delete[] output;
    return reshapedOutput;
}