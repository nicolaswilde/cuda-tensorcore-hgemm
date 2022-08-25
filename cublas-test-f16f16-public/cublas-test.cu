#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

#define checkCudaErrors(func)                                                    \
{                                                                                \
    cudaError_t e = (func);                                                      \
    if(e != cudaSuccess)                                                         \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e)); \
}

#define checkCublasErrors(func)                                                                     \
{                                                                                                   \
    cublasStatus_t e = (func);                                                                      \
    if (e != CUBLAS_STATUS_SUCCESS) {                                                               \
        printf ("%s %d CUDA: ", __FILE__,  __LINE__);                                               \
        switch (e) {                                                                                \
            case CUBLAS_STATUS_NOT_INITIALIZED : printf("CUBLAS_STATUS_NOT_INITIALIZED\n"); break;  \
            case CUBLAS_STATUS_ALLOC_FAILED    : printf("CUBLAS_STATUS_NOT_INITIALIZED\n"); break;  \
            case CUBLAS_STATUS_INVALID_VALUE   : printf("CUBLAS_STATUS_INVALID_VALUE\n"); break;    \
            case CUBLAS_STATUS_ARCH_MISMATCH   : printf("CUBLAS_STATUS_ARCH_MISMATCH\n"); break;    \
            case CUBLAS_STATUS_MAPPING_ERROR   : printf("CUBLAS_STATUS_MAPPING_ERROR\n"); break;    \
            case CUBLAS_STATUS_EXECUTION_FAILED: printf("CUBLAS_STATUS_EXECUTION_FAILED\n"); break; \
            case CUBLAS_STATUS_INTERNAL_ERROR  : printf("CUBLAS_STATUS_INTERNAL_ERROR\n"); break;   \
            case CUBLAS_STATUS_NOT_SUPPORTED   : printf("CUBLAS_STATUS_NOT_SUPPORTED\n"); break;    \
            case CUBLAS_STATUS_LICENSE_ERROR   : printf("CUBLAS_STATUS_LICENSE_ERROR\n"); break;    \
            default: break;                                                                         \
        }                                                                                           \
    }                                                                                               \
}

void cpuF16F16Gemm(
    const half *a, const half *b, half *c, int M, int N, int K) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += (float)a[OFFSET(m, k, K)] * (float)b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = (half)psum;
        }
    }
}

template <cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP>
void cublasF16F16Gemm(
    const half *a, const half *b, half *c, int M, int N, int K) {

    cublasHandle_t handle;
    cublasCreate(&handle);
    half alpha = 1.0;
    half beta = 0.0;
    checkCublasErrors(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
        &alpha, b, CUDA_R_16F, N, a, CUDA_R_16F, K, &beta, c, CUDA_R_16F, N,
        CUBLAS_COMPUTE_16F, algo));
    cublasDestroy(handle);
}

float testF16F16GemmMaxError(
    void (*gpuF16F16Gemm) (const half *, const half *, half *, int, int, int),
    int M, int N, int K) {

    size_t size_a = M * K * sizeof(half);
    size_t size_b = K * N * sizeof(half);
    size_t size_c = M * N * sizeof(half);

    half *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_d_c;
    h_a = (half *)malloc(size_a);
    h_b = (half *)malloc(size_b);
    h_c = (half *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (half *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = (half)(rand() / float(RAND_MAX));
    for (int i = 0; i < K * N; i++)
        h_b[i] = (half)(rand() / float(RAND_MAX));

    cpuF16F16Gemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    gpuF16F16Gemm(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs((float)h_d_c[i] - (float)h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); free(h_d_c);

    return max_error;
}

float testF16F16GemmPerformance(
    void (*gpuF16F16Gemm) (const half *, const half *, half *, int, int, int),
    int M, int N, int K, const int repeat) {

    size_t size_a = M * K * sizeof(half);
    size_t size_b = K * N * sizeof(half);
    size_t size_c = M * N * sizeof(half);

    half *d_a, *d_b;
    half *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        gpuF16F16Gemm(d_a, d_b, d_c, M, N, K);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return sec;
}

int main () {
    const int algo_num = 44;
    char algo_name[algo_num][30] = {
        "CUBLAS_GEMM_DFALT",
        "CUBLAS_GEMM_DEFAULT",
        "CUBLAS_GEMM_ALGO0",
        "CUBLAS_GEMM_ALGO1",
        "CUBLAS_GEMM_ALGO2",
        "CUBLAS_GEMM_ALGO3",
        "CUBLAS_GEMM_ALGO4",
        "CUBLAS_GEMM_ALGO5",
        "CUBLAS_GEMM_ALGO6",
        "CUBLAS_GEMM_ALGO7",
        "CUBLAS_GEMM_ALGO8",
        "CUBLAS_GEMM_ALGO9",
        "CUBLAS_GEMM_ALGO10",
        "CUBLAS_GEMM_ALGO11",
        "CUBLAS_GEMM_ALGO12",
        "CUBLAS_GEMM_ALGO13",
        "CUBLAS_GEMM_ALGO14",
        "CUBLAS_GEMM_ALGO15",
        "CUBLAS_GEMM_ALGO16",
        "CUBLAS_GEMM_ALGO17",
        "CUBLAS_GEMM_ALGO18",
        "CUBLAS_GEMM_ALGO19",
        "CUBLAS_GEMM_ALGO20",
        "CUBLAS_GEMM_ALGO21",
        "CUBLAS_GEMM_ALGO22",
        "CUBLAS_GEMM_ALGO23",
        "CUBLAS_GEMM_DEFAULT_TENSOR_OP",
        "CUBLAS_GEMM_DFALT_TENSOR_OP",
        "CUBLAS_GEMM_ALGO0_TENSOR_OP",
        "CUBLAS_GEMM_ALGO1_TENSOR_OP",
        "CUBLAS_GEMM_ALGO2_TENSOR_OP",
        "CUBLAS_GEMM_ALGO3_TENSOR_OP",
        "CUBLAS_GEMM_ALGO4_TENSOR_OP",
        "CUBLAS_GEMM_ALGO5_TENSOR_OP",
        "CUBLAS_GEMM_ALGO6_TENSOR_OP",
        "CUBLAS_GEMM_ALGO7_TENSOR_OP",
        "CUBLAS_GEMM_ALGO8_TENSOR_OP",
        "CUBLAS_GEMM_ALGO9_TENSOR_OP",
        "CUBLAS_GEMM_ALGO10_TENSOR_OP",
        "CUBLAS_GEMM_ALGO11_TENSOR_OP",
        "CUBLAS_GEMM_ALGO12_TENSOR_OP",
        "CUBLAS_GEMM_ALGO13_TENSOR_OP",
        "CUBLAS_GEMM_ALGO14_TENSOR_OP",
        "CUBLAS_GEMM_ALGO15_TENSOR_OP"};
    cublasGemmAlgo_t algo_list[algo_num] = {
        CUBLAS_GEMM_DFALT,
        CUBLAS_GEMM_DEFAULT,
        CUBLAS_GEMM_ALGO0,
        CUBLAS_GEMM_ALGO1,
        CUBLAS_GEMM_ALGO2,
        CUBLAS_GEMM_ALGO3,
        CUBLAS_GEMM_ALGO4,
        CUBLAS_GEMM_ALGO5,
        CUBLAS_GEMM_ALGO6,
        CUBLAS_GEMM_ALGO7,
        CUBLAS_GEMM_ALGO8,
        CUBLAS_GEMM_ALGO9,
        CUBLAS_GEMM_ALGO10,
        CUBLAS_GEMM_ALGO11,
        CUBLAS_GEMM_ALGO12,
        CUBLAS_GEMM_ALGO13,
        CUBLAS_GEMM_ALGO14,
        CUBLAS_GEMM_ALGO15,
        CUBLAS_GEMM_ALGO16,
        CUBLAS_GEMM_ALGO17,
        CUBLAS_GEMM_ALGO18,
        CUBLAS_GEMM_ALGO19,
        CUBLAS_GEMM_ALGO20,
        CUBLAS_GEMM_ALGO21,
        CUBLAS_GEMM_ALGO22,
        CUBLAS_GEMM_ALGO23,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        CUBLAS_GEMM_DFALT_TENSOR_OP,
        CUBLAS_GEMM_ALGO0_TENSOR_OP,
        CUBLAS_GEMM_ALGO1_TENSOR_OP,
        CUBLAS_GEMM_ALGO2_TENSOR_OP,
        CUBLAS_GEMM_ALGO3_TENSOR_OP,
        CUBLAS_GEMM_ALGO4_TENSOR_OP,
        CUBLAS_GEMM_ALGO5_TENSOR_OP,
        CUBLAS_GEMM_ALGO6_TENSOR_OP,
        CUBLAS_GEMM_ALGO7_TENSOR_OP,
        CUBLAS_GEMM_ALGO8_TENSOR_OP,
        CUBLAS_GEMM_ALGO9_TENSOR_OP,
        CUBLAS_GEMM_ALGO10_TENSOR_OP,
        CUBLAS_GEMM_ALGO11_TENSOR_OP,
        CUBLAS_GEMM_ALGO12_TENSOR_OP,
        CUBLAS_GEMM_ALGO13_TENSOR_OP,
        CUBLAS_GEMM_ALGO14_TENSOR_OP,
        CUBLAS_GEMM_ALGO15_TENSOR_OP};
    void (*cublasF16F16Gemm_list[algo_num]) (const half *, const half *, half *, int, int, int) = {
        cublasF16F16Gemm<CUBLAS_GEMM_DFALT>,
        cublasF16F16Gemm<CUBLAS_GEMM_DEFAULT>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO0>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO1>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO2>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO3>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO4>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO5>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO6>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO7>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO8>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO9>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO10>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO11>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO12>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO13>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO14>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO15>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO16>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO17>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO18>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO19>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO20>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO21>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO22>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO23>,
        cublasF16F16Gemm<CUBLAS_GEMM_DEFAULT_TENSOR_OP>,
        cublasF16F16Gemm<CUBLAS_GEMM_DFALT_TENSOR_OP>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO0_TENSOR_OP>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO1_TENSOR_OP>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO2_TENSOR_OP>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO3_TENSOR_OP>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO4_TENSOR_OP>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO5_TENSOR_OP>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO6_TENSOR_OP>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO7_TENSOR_OP>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO8_TENSOR_OP>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO9_TENSOR_OP>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO10_TENSOR_OP>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO11_TENSOR_OP>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO12_TENSOR_OP>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO13_TENSOR_OP>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO14_TENSOR_OP>,
        cublasF16F16Gemm<CUBLAS_GEMM_ALGO15_TENSOR_OP>};

    /*
    const int test_num = 8;
    const int M_list[test_num] = {128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    const int N_list[test_num] = {128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    const int K_list[test_num] = {128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    */

    const int test_num = 64;
    int M_list[test_num];
    int N_list[test_num];
    int K_list[test_num];
    for (int i = 0; i < test_num; i++) {
        M_list[i] = (i + 1) * 256;
        N_list[i] = (i + 1) * 256;
        K_list[i] = (i + 1) * 256;
    }

    const int outer_repeat = 10, inner_repeat = 1;

    for (int i = 0; i < algo_num; i++) {
        printf("\nalgo = %s\n", algo_name[i]);

        {
            const int M = 256, N = 256, K = 256;
            float max_error = testF16F16GemmMaxError(
                cublasF16F16Gemm_list[i], M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        for (int j = 0; j < test_num; j++) {
            int M = M_list[j], N = N_list[j], K = K_list[j];

            double max_sec = 0.0;
            double min_sec = DBL_MAX;
            double total_sec = 0.0;

            double sec[outer_repeat];
            for (int k = 0; k < outer_repeat; k++) {
                double this_sec = testF16F16GemmPerformance(
                    cublasF16F16Gemm_list[i], M, N, K, inner_repeat);
                sec[k] = this_sec;
                max_sec = max(max_sec, this_sec);
                min_sec = min(min_sec, this_sec);
                // total_sec += this_sec;
            }

            int valid_sec = 0;

            for (int k = 0; k < outer_repeat; k++) {
                if ((sec[k] > 2 * min_sec) && (sec[k] >= 0.5)) {
                    continue;
                }
                valid_sec++;
                total_sec += sec[k];
            }

            // if (valid_sec < outer_repeat)
            //     printf("%d test invalid!\n", outer_repeat - valid_sec);

            // double avg_sec = total_sec / outer_repeat;
            double avg_sec = total_sec / valid_sec;
            double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

            printf("M N K = %6d %6d %6d, ", M, N, K);
            printf("Time = %12.8lf %12.8lf %12.8lf s, ", min_sec, avg_sec, max_sec);
            printf("AVG Performance = %10.4lf Gflops\n", avg_Gflops);
        }
    }
}
