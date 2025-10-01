#include <iostream>
#include <cuda_runtime.h>

template<typename T, int kernel_size_x, int kernel_size_y, int kernel_size_z>
__global__ void max_pooling3d_kernel(T *input, T *output, int batch_size, int channels, int dim1, int dim2, int dim3) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    int i = blockIdx.z * kernel_size_x + threadIdx.x;
    int j = blockIdx.w * kernel_size_y + threadIdx.y;
    int k = blockIdx.v * kernel_size_z + threadIdx.z;

    if (i < dim1 && j < dim2 && k < dim3) {
        T max_value = input[(b * channels + c) * dim1 * dim2 * dim3 + i * dim2 * dim3 + j * dim3 + k];
        for (int r = 0; r < kernel_size_x; ++r) {
            for (int s = 0; s < kernel_size_y; ++s) {
                for (int t = 0; t < kernel_size_z; ++t) {
                    int x = i + r;
                    int y = j + s;
                    int z = k + t;
                    if (x < dim1 && y < dim2 && z < dim3) {
                        T value = input[(b * channels + c) * dim1 * dim2 * dim3 + x * dim2 * dim3 + y * dim3 + z];
                        if (value > max_value) {
                            max_value = value;
                        }
                    }
                }
            }
        }
        output[(b * channels + c) * dim1 / kernel_size_x * dim2 / kernel_size_y * dim3 / kernel_size_z + 
               blockIdx.z * (dim1 / kernel_size_x) * (dim2 / kernel_size_y) + 
               blockIdx.w * (dim1 / kernel_size_x) + 
               blockIdx.v] = max_value;
    }
}

template<typename T, int kernel_size_x, int kernel_size_y, int kernel_size_z>
void max_pooling3d(T *input_d, T *output_d, int batch_size, int channels, int dim1, int dim2, int dim3) {
    dim3 blocks(batch_size, channels);
    dim3 threads(kernel_size_x, kernel_size_y, kernel_size_z);

    max_pooling3d_kernel<T, kernel_size_x, kernel_size_y, kernel_size_z><<<blocks, threads>>>(input_d, output_d, batch_size, channels, dim1, dim2, dim3);
    cudaDeviceSynchronize();
}

int main() {
    int batch_size = 1;
    int channels = 1;
    int dim1 = 32;
    int dim2 = 32;
    int dim3 = 32;
    int kernel_size_x = 2;
    int kernel_size_y = 2;
    int kernel_size_z = 2;

    size_t input_size = batch_size * channels * dim1 * dim2 * dim3 * sizeof(float);
    size_t output_size = batch_size * channels * (dim1 / kernel_size_x) * (dim2 / kernel_size_y) * (dim3 / kernel_size_z) * sizeof(float);

    float *input_h = new float[input_size];
    float *output_h = new float[output_size];

    // Initialize input
    for (int i = 0; i < batch_size * channels * dim1 * dim2 * dim3; ++i) {
        input_h[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *input_d, *output_d;
    cudaMalloc(&input_d, input_size);
    cudaMalloc(&output_d, output_size);

    cudaMemcpy(input_d, input_h, input_size, cudaMemcpyHostToDevice);

    max_pooling3d<float, 2, 2, 2>(input_d, output_d, batch_size, channels, dim1, dim2, dim3);

    cudaMemcpy(output_h, output_d, output_size, cudaMemcpyDeviceToHost);

    // Output some results for verification
    std::cout << "Output:\n";
    for (int i = 0; i < 16; ++i) {
        std::cout << output_h[i] << " ";
    }
    std::cout << "\n";

    delete[] input_h;
    delete[] output_h;

    cudaFree(input_d);
    cudaFree(output_d);

    return 0;
}
