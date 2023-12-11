
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define BLOCK_DIM 16

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float* A,
    const float* B, float beta, float* C) {
    // compute position in C that this thread is responsible for
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // `if` condition is necessary for when M or N aren't multiples of 32.
    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        // C = α*(A@B)+β*C
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

int main() {
    size_t N = 4092;
    size_t size = N * N * sizeof(float);
    float* h_A, * h_B, * h_C;
    float* d_A, * d_B, * d_C;

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(N / BLOCK_DIM, N / BLOCK_DIM);
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);

    sgemm_naive << <dimGrid, dimBlock >> > (N, N, N, 1, d_A, d_B, 0, d_C);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Validate the result
    // ... (validation code here)

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
