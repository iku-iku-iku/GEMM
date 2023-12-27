
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define CEIL_DIV(x, y) ((x + y - 1) / y)

#define DEBUG 0
#define BM 64
#define BK 8
#define BN 64
#define TM 8
#define BLOCKDIM_X (BM * BK)

__global__ void sgemm_multi_res(int M, int N, int K, float alpha, const float* A,
    const float* B, float beta, float* C) {
    // compute position in C that this thread is responsible for
    const int threadRow = threadIdx.x / BN;
    const int threadCol = threadIdx.x % BN;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int x = blockIdx.x * BN + threadRow * TM;
    const int y = blockIdx.y * BM + threadCol;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    float threadResults[TM] = { 0.0 };

    const int innerRowA = threadIdx.x / BK, innerColA = threadIdx.x % BK;
    const int innerRowB = threadIdx.x / BN, innerColB = threadIdx.x % BN;

    // `if` condition is necessary for when M or N aren't multiples of 32.
    if (x < M && y < N) {
        // advance pointers to the starting positions
        A += bx * BM * K;                   
        B += by * BN;                      
        C += bx * BM * N + by * BN; 

        // the outer loop advances A along the columns and B along
        // the rows until we have fully calculated the result in C.
        for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
            // Have each thread load one of the elements in A & B from
            // global memory into shared memory.
            As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
            Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];

            // block threads in this block until cache is fully populated
            __syncthreads();

            // advance pointers onto next chunk
            A += BK;
            B += BK * N;

            // execute the dotproduct on the currently cached block
            for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
				float Btmp = Bs[dotIdx * BN + threadCol];
                for (int resIdx = 0; resIdx < TM; ++resIdx) {
                    threadResults[resIdx] += As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;
                }
            }
            // need to sync again at the end, to avoid faster threads
            // fetching the next block into the cache before slower threads are done
            __syncthreads();
        }

        for (int resIdx = 0; resIdx < TM; ++resIdx) {
			C[(threadRow * TM + resIdx) * N + threadCol] =
				alpha * threadResults[resIdx] + beta * C[(threadRow * TM + resIdx) * N + threadCol];
        }
    }
}

int main() {
#if DEBUG
    size_t N = 1024;
#else
    size_t N = 4092;
#endif
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

    dim3 dimGrid(CEIL_DIV(N, BM), CEIL_DIV(N, BN));
    dim3 dimBlock(BLOCKDIM_X);

    sgemm_multi_res << <dimGrid, dimBlock >> > (N, N, N, 1, d_A, d_B, 0, d_C);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

#if DEBUG
    // Validate the result
    bool validationFailed = false;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; j++) {
			float expected = 0;
			for (int k = 0; k < N; ++k) {
				expected += h_A[i * N + k] * h_B[k * N + j];
			}

			if (std::fabs(expected - h_C[i * N + j]) > 1e-3) {
				printf("Validation failed at index %d: expected %f, got %f\n", i * N + j, expected, h_C[i * N + j]);
				validationFailed = true;
			}
        }
    }

    if (!validationFailed) {
        printf("Validation passed!\n");
    }
    else {
        printf("Validation failed!\n");
    }
#endif
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
