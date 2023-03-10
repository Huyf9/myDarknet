int gpu_index = 0;

#ifdef GPUS

#include "cuda.h"
#include "utils.h"
#include "blas.h"
#include <assert.h>
#include <iostream>
#include <time.h>

void cuda_set_device(int n) 
{
    gpu_index = n;
    cudaError_t status = cudaSetDevice(n);
    check_error(status);
}

int cuda_get_device()
{
    int n = 0;
    cudaError_t status = cudaGetDevice(&n);
    check_error(status);
    return n;
}

void check_error(cudaError_t status)
{
    cudaError_t status2 = cudaGetLastError();
    if(status != cudaSuccess)
        std::cerr << "CUDA error: " << cudaGetErrorString(status) << '\n';
    if(status2 != cudaSuccess)
        std::cerr << "CUDA error: " << cudaGetErrorString(status2) << '\n';
}   


dim3 cuda_gridsize(size_t n)
{
    size_t k = (n-1) / BLOCK + 1;
    size_t x = k;
    size_t y = 1;
    if(x > 65535) {
        x = ceil(sqrt(k));
        y = (n-1)/(x*BLOCK) + 1;
    }
    dim3 d = {x, y, 1};
    return d;
}

#ifdef CUDNN
cudnnHandle_t cudnn_handle()
{
    static int init[16] = {0};
    static cudnnHandle_t handle[16];
    int i = cuda_get_device();
    if(!init[i]) {
        cudnnCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
}
#endif

cublasHandle_t blas_handle()
{
    static int init[16] = {0};
    static cublasHandle_t handle[16];
    int i = cuda_get_device();
    if(!init[i]) {
        cudnnCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
}

float *cuda_make_array(float *x, size_t n)
{
    float *x_gpu;
    size_t size = sizeof(float) * n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
    if(x) {
        status = cudaMemcpy(x_gpu, size, cudaMemcpyHostToDevice);
        check_error(status);
    } else {
        fill_gpu(n, 0, x_gpu, 1);
    }
    if(!x_gpu) std::cerr << ("Cuda malloc failed!\n");
    return x_gpu;
}

void cuda_random(float *x_gpu, size_t n)
{
    static curandGenerator_t gen[16];
    static int init[16] = {0};
    int i = cuda_get_device();
    if(!init[i]) {
        curandCreateGenerator(&gen[i], CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen[i], time(0));
        init[i] = 1;
    }
    curandGenerateUniform(gen[i], x_gpu, n);
    check_error(cudaPeekAtLastError());
}

void cuda_compare(float *x_gpu, float *x, size_t n, char *s) 
{
    float *tmp = new float[n];
    cuda_pull_array(x_gpu, tmp, n);

    axpy_cpu(n, -1, x, 1, tmp, 1);
    float err = dot_cpu(n, tmp, 1, tmp, 1);
    std::cout << "Error " << s << ": " << sqrt(err/n) << '\n';
    delete[] tmp;
    return err;
}

int *cuda_make_int_array(int *x, size_t n)
{
    int *x_gpu;
    size_t size = sizeof(int) * n;
    cudaErrot_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
    if(x) {
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        check_error(status);
    }
    if(!x_gpu) std::cerr << ("Cuda malloc failed!\n");
    return x_gpu;
}

void cuda_free(float *x_gpu) 
{
    cudaError_t status = cudaFree(x_gpu);
    check_error(status);
}

void cuda_push_array(float *xgpu, float *x, size_t n)
{
    size_t size = sizeof(float) * n;
    cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyHostToDevice);
    check_error(status);
}

float cuda_mag_array(float *x_gpu, size_t n)
{
    float *temp = new float[n];
    cuda_pull_array(x_gpu, temp, n);
    float m = mag_array(temp, n);
    delete[] temp;
    return m;
}

#else
void cuda_set_device(int n) {}

#endif