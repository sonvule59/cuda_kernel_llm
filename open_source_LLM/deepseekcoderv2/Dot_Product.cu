#include <cuda.h>
#include <stdio.h>

int main() {
    const int len = 32;
    float *a_device, *b_device, *result_device;
    float *a_host = new float[len];
    float *b_host = new float[len];
    float *c_host = new float[len];

    cudaMalloc((void **)&a_device, len * sizeof(float));
    cudaMalloc((void **)&b_device, len * sizeof(float));
    cudaMalloc((void **)&result_device, len * sizeof(float));

    // Initialize host vectors a and b
    for (int i = 0; i < len; ++i) {
        a_host[i] = i + 1.f;
        b_host[i] = 2.f * i + 3.f;
    }

    cudaMemcpy(a_device, a_host, len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_host, len * sizeof(float), cudaMemcpyHostToDevice);

    // Set up kernel launch configuration
    dim3 threadsPerBlock(32);
    dim3 blocksPerGrid((len + threadsPerBlock.x - 1) / threadsPerBlock.x, 1);

    dotProduct<<<blocksPerGrid, threadsPerBlock>>>(a_device, b_device, result_device, len);

    // Copy result back to the host
    cudaMemcpy(c_host, result_device, len * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Dot product: \n");
    for (int i = 0; i < len; ++i) {
        printf("%f ", c_host[i]);
    }
    printf("\n");

    // Clean up resources
    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(result_device);
    delete[] a_host;
    delete[] b_host;
    delete[] c_host;

    return 0;
}