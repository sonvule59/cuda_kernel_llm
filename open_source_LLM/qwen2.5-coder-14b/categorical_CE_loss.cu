#include <iostream>
#include <cuda_runtime.h>

__global__ void compute_loss(float* logits, int* true_labels, float* loss, int N, int C) {
    extern __shared__ float shmem[];

    // Thread index within the block
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Each thread calculates the loss for one sample
    if (bid < N) {
        float* z = logits + bid * C;
        int y = true_labels[bid];

        // Numerical stability: find max logit value
        float z_max = z[0];
        for (int k = 1; k < C; ++k) {
            if (z[k] > z_max) {
                z_max = z[k];
            }
        }

        // Compute the sum of exp(logits) and the log softmax
        float sum_exp = 0.0;
        for (int k = 0; k < C; ++k) {
            sum_exp += exp(z[k] - z_max);
        }

        // Compute the loss for this sample
        float log_softmax = z[y] - z_max;
        float loss_value = -log_softmax - log(sum_exp);

        // Store the loss value in shared memory
        shmem[tid] = loss_value;

        __syncthreads();

        // Reduce the loss values within the block
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shmem[tid] += shmem[tid + s];
            }
            __syncthreads();
        }

        // Write the result for this block to the global loss array
        if (tid == 0) {
            loss[bid] = shmem[0];
        }
    }
}

__host__ void compute_average_loss(float* h_logits, int* h_true_labels, float* d_loss, int N, int C) {
    // Allocate device memory for logits and true labels
    float* d_logits;
    int* d_true_labels;
    cudaMalloc(&d_logits, N * C * sizeof(float));
    cudaMalloc(&d_true_labels, N * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_logits, h_logits, N * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_true_labels, h_true_labels, N * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate device memory for the loss values
    float* d_loss_sum;
    cudaMalloc(&d_loss_sum, sizeof(float));

    // Launch the kernel
    int threadsPerBlock = 128;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemSize = threadsPerBlock * sizeof(float);
    compute_loss<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_logits, d_true_labels, d_loss, N, C);

    // Reduce the block-wise losses to get the total loss
    float* h_loss = new float[N];
    cudaMemcpy(h_loss, d_loss, N * sizeof(float), cudaMemcpyDeviceToHost);

    float total_loss = 0.0;
    for (int i = 0; i < N; ++i) {
        total_loss += h_loss[i];
    }

    // Compute the average loss
    float* h_average_loss = new float[1];
    *h_average_loss = total_loss / N;
    cudaMemcpy(d_loss_sum, h_average_loss, sizeof(float), cudaMemcpyHostToDevice);

    // Cleanup
    delete[] h_loss;
    delete[] h_average_loss;
    cudaFree(d_logits);
    cudaFree(d_true_labels);
    cudaFree(d_loss_sum);

    // Output the average loss
    std::cout << "Average Loss: " << *h_average_loss << std::endl;
}

int main() {
    // Example input
    int N = 4; // Number of samples
    int C = 3; // Number of classes
    float h_logits[] = {1.0, 2.0, 3.0,
                       4.0, 5.0, 6.0,
                       7.0, 8.0, 9.0,
                       10.0, 11.0, 12.0};
    int h_true_labels[] = {0, 1, 2, 0};

    // Allocate device memory for the loss
    float* d_loss;
    cudaMalloc(&d_loss, N * sizeof(float));

    // Compute the average loss
    compute_average_loss(h_logits, h_true_labels, d_loss, N, C);

    // Cleanup
    cudaFree(d_loss);

    return 0;
}
