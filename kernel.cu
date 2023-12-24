
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define CEIL_DIV(x, y) ((x + y - 1) / y)

#define BLOCKSIZE 32

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float* A,
    const float* B, float beta, float* C) {
    // compute position in C that this thread is responsible for
    const int threadRow = threadIdx.x / BLOCKSIZE;
    const int threadCol = threadIdx.x % BLOCKSIZE;
    const int cRow = blockIdx.x;
    const int cCol = blockIdx.y;
    const int x = blockIdx.x * BLOCKSIZE + threadRow;
    const int y = blockIdx.y * BLOCKSIZE + threadCol;

    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    // `if` condition is necessary for when M or N aren't multiples of 32.
    if (x < M && y < N) {
        // advance pointers to the starting positions
        A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
        B += cCol * BLOCKSIZE;                        // row=0, col=cCol
        C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

        float tmp = 0.0;
        // the outer loop advances A along the columns and B along
        // the rows until we have fully calculated the result in C.
        for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
            // Have each thread load one of the elements in A & B from
            // global memory into shared memory.
            // Make the threadCol (=threadIdx.x) the consecutive index
            // to allow global memory access coalescing
            As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
            Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

            // block threads in this block until cache is fully populated
            __syncthreads();

            // advance pointers onto next chunk
            A += BLOCKSIZE;
            B += BLOCKSIZE * N;

            // execute the dotproduct on the currently cached block
            for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
                tmp += As[threadRow * BLOCKSIZE + dotIdx] *
                    Bs[dotIdx * BLOCKSIZE + threadCol];
            }
            // need to sync again at the end, to avoid faster threads
            // fetching the next block into the cache before slower threads are done
            __syncthreads();
        }
        C[threadRow * N + threadCol] =
            alpha * tmp + beta * C[threadRow * N + threadCol];
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

    dim3 dimGrid(CEIL_DIV(N, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));
    dim3 dimBlock(BLOCKSIZE * BLOCKSIZE);

    sgemm_naive << <dimGrid, dimBlock >> > (N, N, N, 1, d_A, d_B, 0, d_C);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Validate the result
    bool validationFailed = false;
    for (int i = 0; i < N * N; ++i) {
        float expected = 0;
        for (int k = 0; k < N; ++k) {
            expected += h_A[i / N * N + k] * h_B[k * N + i % N];
        }
        expected = 1 * expected + 0 * h_C[i]; // Apply alpha and beta

        if (std::fabs(expected - h_C[i]) > 1e-5) {
            printf("Validation failed at index %d: expected %f, got %f\n", i, expected, h_C[i]);
            validationFailed = true;
        }
    }

    if (!validationFailed) {
        printf("Validation passed!\n");
    }
    else {
        printf("Validation failed!\n");
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
