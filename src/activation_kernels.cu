#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "activations.h"
#include "cuda.h"
}

/*
 * light tanh activate function: 
 *   y = 0.001 * x              if x < 0;
 *   y = x                      if 0 <= x <= 1;
 *   y = 0.001 * (x - 1) + 1    else.
 *
 * the grident of light tanh:
 *   y' = 1                     if 0 <= x <= 1;
 *   y' = 0.001                 else.
*/
__device__ float lhtan_activate_kernel(float x) 
{
    if(x < 0) return 0.001f * x;
    if(x > 1) return 0.001f * (x - 1) + 1;
    return x; 
}

__device__ float lhtan_gradient_kernel(float x)
{
    if(x >= 0 && x <= 1) return 1;
    return 0.001f;
}

/*
 * hard tan activate function:
 *   y = -1                     if x < -1;
 *   y = 1                      if x > 1;
 *   y = x                      else.
 *
 * the grident of hard tan:
 *   y = 1                      if -1 <= x <= 1;
 *   y = 0                      else.
*/
__device__ float hardtan_activate_kernel(float x)
{
    if(x < -1) return -1;
    if(x > 1) return 1;
    return x;
    if(x >= -1 && x <= 1) return 1;
    return 0;
}

__device__ float hardtan_gradient_kernel(float x)
{
    if(x >= -1 && x <= 1) return 1;
    return 0;
}

/*
 * linear activate function:
 *   y = x
 *
 * the grident of linear:
 *   y' = 1
*/
__device__ float linear_activate_kernel(float x) {return x;}
__device__ float linear_gradient_kernel(float x) {return 1;}

__device__ float logistic_activate_kernel(float x) {return 1.f/(1.f + expf(-x));}
__device__ float logistic_gradient_kernel(float x) {return expf(-x) / ((1 + expf(-x)) * (1 + expf(-x)));}

__device__ float relu_activate_kernel(float x) {return x*(x > 0);}
__device__ float relu_gradient_kernel(float x) {return x > 0;}

__device__ float elu_activate_kernel(float x) {return (x >= 0)*x + (x < 0)*(expf(x)-1);}
__device__ float elu_gradient_kernel(float x) {return (x >= 0) + (x < 0)*(x + 1);}

__device__ float selu_activate_kernel(float x) {return (x >= 0)*1.0507f*x + (x < 0)*1.0507f*1.6732*(expf(x)-1);}
__device__ float selu_gradient_kernel(float x) {return (x >= 0)*1.0507f + (x < 0)*(1.0507f*1.6732 + x);}


__device__ float leaky_activate_kernel(float x) {return (x > 0)*x + (x <= 0)*0.1f*x;}
__device__ float leaky_gradient_kernel(float x) {return (x > 0) + (x <= 0)*0.1f;}

__device__ float tanh_activate_kernel(float x) {return (exp(2*x)-1)/(exp(2*x)+1);}
__device__ float tanh_gradient_kernel(float x) {return 1 - x*x;}


// 根据传入的激活函数选项选择具体的激活函数
__device__ float activate_kernel(float x, ACTIVATION a)
{
    switch (a)
    {
    case LHTAN:         return lhtan_activate_kernel(x);
    case HARDTAN:       return hardtan_activate_kernel(x);
    case LINEAR:        return linear_activate_kernel(x);
    case LOGISTIC:      return logistic_activate_kernel(x);
    case RELU:          return relu_activate_kernel(x);
    case ELU:           return elu_activate_kernel(x);
    case SELU:          return selu_activate_kernel(x);
    case LEAKY:         return leaky_activate_kernel(x);
    case TANH:          return tanh_activate_kernel(x);
    }
}
// 同时计算其梯度
__device__ float gradient_kernel(float x, ACTIVATION a)
{
    switch (a)
    {
    case LHTAN:         return lhtan_gradient_kernel(x);
    case HARDTAN:       return hardtan_gradient_kernel(x);
    case LINEAR:        return linear_gradient_kernel(x);
    case LOGISTIC:      return logistic_gradient_kernel(x);
    case RELU:          return relu_gradient_kernel(x);
    case ELU:           return elu_gradient_kernel(x);
    case SELU:          return selu_gradient_kernel(x);
    case LEAKY:         return leaky_gradient_kernel(x);
    case TANH:          return tanh_gradient_kernel(x);
    }
}

/*
* x: 某一层网络的输入
* dy: x 对应的微分
* dx: x对应的输入的微分
* s: 二维矩阵x的第一个维度
* 
*/
__global__ void binary_gradient_array_kernel(float *x, float *dy, int n, int s, float *dx)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    int i = id % s; //获取当前的
    int b = id / s;
    float x1 = x[b*s + i];
    float x2 = x[b*s + s/2 + i];
    if(id < n) {
        float de = dy[id];
        dx[b*s + i] = x2 * de;
        dx[b*s + s/2 + i] = x1 * de;
    }
}

extern "C" void binary_gradient_array_gpu(float *x, float *dx, int n, int s, float *y)
{
    binary_gradient_array_kernel<<<cuda_gridsize(n/2), BLOCK>>>(x, dx, n, s, y);
    check_error(cudaPeekAtLastError());
}


__global__ void binary_activate_array_kernel(float *x, int n, int s, float *y)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    int i = id % s;
    int b = id / s;
    float x1 = x[b*s + i];
    float x2 = x[b*s + s/2 + i];
    if(id < n) y[id] = x1 * x2;
}

extern "C" void binary_activate_array_gpu(float *x, float *dx, int n, int s, float *y)
{
    binary_activate_array_kernel<<<cuda_gridsize(n/2), BLOCK>>>(x, dx, n, s, y);
    check_error(cudaPeekAtLastError());
}
