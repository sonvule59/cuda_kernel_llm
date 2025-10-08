// **matrix_multiplication.cu:**
#include <sm_35.h>

__global__ void matMul(float* A, float* B, float* C, int M, int K, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float temp[M * BLOCK_SIZE];

    if (i < M) {
        for (int k = j * BLOCK_SIZE + min(BLOCK_SIZE, N); k < min((j + 1) * BLOCK_SIZE, N); k++) {
            int index = i * N + k;
            temp[i] += A[i * K + k] * B[k * M + i];
        }
    }

    if (i == 0) __syncthreads();

    if (i < M) {
        temp[i] += temp[i + blockDim.x];
        __syncthreads();
    }

    if (i == 0) {
        C[i * N + j] = temp[0];
        for (int p = 1; p < M; ++p) {
            C[i * N + j] += temp[p];
            __syncthreads();
        }
    }
}

void launchKernel(float* A, float* B, float* C, int M, int K, int N, int blockSize) {
    int blocks = ((M + blockSize - 1) / blockSize);
    dim3 grid(blocks, ceil((float)N / (float)blockSize));
    dim3 threads(blockSize, blockSize);
    matMul<<<grid, threads>>>(A, B, C, M, K, N);
}

// **main.cpp:**


#include <iostream>
#include <vector>
#include <cuda_runtime.h>

void checkCudaError(cudaError_t err, const char* operation) {
    if (err != cudaSuccess) {
        std::cerr << "Error during " << operation << ": " << cudaGetErrorString(err) << "\n";
    }
}

int main() {
    int M = 128;
    int K = 32;
    int N = 64;

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N);

    // Initialize matrices A and B
    for (size_t i = 0; i < A.size(); ++i) {
        A[i] = (rand() % 100) / 10.0f;
        B[i] = (rand() % 100) / 10.0f;
    }

    float* d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, A.size() * sizeof(float));
    cudaMalloc((void**)&d_B, B.size() * sizeof(float));
    cudaMalloc((void**)&d_C, C.size() * sizeof(float));
    checkCudaError(cudaMemcpy(d_A, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_B, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = 32; // Adjust this value for optimal shared memory usage and memory coalescing
    launchKernel<matMul>(d_A, d_B, d_C, M, K, N, blockSize);
    checkCudaError(cudaDeviceSynchronize());

    cudaMemcpy(C.data(), d_C, C.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Matrix multiplication completed\n";

    // Check the result for small matrices
    if (M < 10 && N < 10) {
        bool correct = true;
        float* d_CRef(nullptr);
        cudaMalloc((void**)&d_CRef, C.size() * sizeof(float));

        for (size_t i = 0; i < C.size(); ++i) {
            float ref = 0.0f;
            for (int k = 0; k < K; ++k) {
                ref += A[i * K + k] * B[k * M + i];
            }
            correct &= fabs(C[i] - ref) < 1e-5f;
        }
        checkCudaError(cudaFree(d_CRef));

        if (correct) {
            std::cout << "Result is correct\n";
        } else {
            std::cout << "Result is incorrect\n";
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}