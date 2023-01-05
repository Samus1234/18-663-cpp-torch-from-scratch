#include <Eigen/Core>
#include <vector>
#include <memory>
#include <cstring>
#include "utility.hpp"

/*
    Specify the number of threads per block to run on the GPU
    Please refer to your specific GPU architecture for information on number of threads
    These experiments were carried out on an GeForce RTX 3080Ti 
*/



#define NUM_THREADS (1 << 8)

/*
    Global functions, for the GPU device
    Matrices are in column-major order (due to Eigen)
*/

// GPU array copy function -> 1 dim

__global__ void gpucopy(size_t N, float* x_in, float* x_out)
{
    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        x_out[idx] = x_in[idx];
    }
}

// GPU forward exponential linear unit function -> 1 dim

__global__ void elu(size_t N, float alpha, float* x_in, float* x_out)
{
    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        x_out[idx] = fmaxf(x_in[idx], 0.0f) + alpha * (expf(x_in[idx]) - 1) * (x_in[idx] <= 0);
    }
}

// GPU backward exponential linear unit function -> 1 dim

__global__ void eluGrad(size_t N, float alpha, float* x_in, float* grad_in, float* grad_out)
{
    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        grad_out[idx] = grad_in[idx] * ( alpha * expf(x_in[idx])  + (x_in[idx] > 0) * (1 - alpha * expf(x_in[idx])) );
    }
}

// GPU matrix addition/subtraction function ( Z = X + sign*Y ) -> 1 dim

__global__ void matadd(size_t n, size_t m, float sign, float* X, float* Y, float* Z)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n*m)
    {
        Z[idx] = fmaf(sign, Y[idx], X[idx]);
    }
}

// GPU matrix broadcasted addtion/subtraction function ( X += sign*y ) -> 2 dim

__global__ void replicateAdd(size_t n, size_t m, float sign, float* X, float* Y)
{
    unsigned col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if ( (row_idx < n) && (col_idx < m) )
    {
        X[row_idx + n*col_idx] = fmaf(sign, Y[col_idx], X[row_idx + n*col_idx]);
    }
}

// GPU matrix transpose function ( Y = X.T ) -> 2 dim

__global__ void transpose(size_t n, size_t m, float* X, float* Y)
{
    unsigned col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if ( (row_idx < n) && (col_idx < m) )
    {
        Y[row_idx + n*col_idx] = X[m*row_idx + col_idx];
    }
}

// GPU matrix multiplication function ( Z = X * Y ) -> 2 dim

__global__ void matmul(size_t n, size_t m, size_t d, float* X, float* Y, float* Z)
{
    unsigned col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if ( (row_idx < n) && (col_idx < d) )
    {
        for (int k = 0; k < m; k++)
        {
            Z[row_idx + n*col_idx] = fmaf(X[row_idx + n*k], Y[k + m*col_idx], Z[row_idx + n*col_idx]);
        }
    }
}

// GPU matrix einsum batched product function ( Z = X * Y.T ) -> 2 dim

__global__ void batchedProd(size_t n, size_t m, size_t d, float* X, float* Y, float* Z)
{
    unsigned col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if ( (row_idx < n) && (col_idx < d) )
    {
        for (int k = 0; k < m; k++)
        {
            Z[row_idx + n*col_idx] = fmaf(X[row_idx + n*k], Y[d*k + col_idx], Z[row_idx + n*col_idx]);
        }
    }
}

// GPU weight gradient compute function ( Z = X.T * Y ) -> 2 dim

__global__ void weightGrad(size_t n, size_t m, size_t d, float* X, float* Y, float* Z)
{
    unsigned col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    float batch = (float) m;

    if ( (row_idx < n) && (col_idx < d) )
    {
        for (int k = 0; k < m; k++)
        {
            Z[row_idx + n*col_idx] = fmaf(fdividef(X[k + m*row_idx], batch), Y[k + m*col_idx], Z[row_idx + n*col_idx]);
        }
    }
}

// GPU bias gradient compute function ( Y = sum(X.T, axis = 0) ) -> 2 dim

__global__ void biasGrad(size_t n, size_t m, float* X, float* Y)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

    float batch = (float) m;

    if ( idx < n )
    {
        for (unsigned k = 0; k < m; k++)
        {
            Y[idx] = fmaf(fdividef(1.0f, batch), X[k + m*idx], Y[idx]);
        }
    }
}

// GPU device single element sigmoid function

__device__ float expit(float x)
{
    return fdividef(1, 1 + expf(-x));
}

// GPU forward sigmoid activation unit function -> 1 dim

__global__ void sigmoidActivation(size_t N, float* x_in, float* x_out)
{
    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        x_out[idx] = expit(x_in[idx]);
    }
}

// GPU backward sigmoid activation unit function -> 1 dim

__global__ void sigmoidGrad(size_t N, float* x_in, float* grad_in, float* grad_out)
{
    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        grad_out[idx] = grad_in[idx] * expit(x_in[idx]) * (1 - expit(x_in[idx]));
    }
}

// GPU scalar element-wise division -> 1 dim

__global__ void divide(size_t N, float divisor, float* x_in)
{
    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        x_in[idx] = fdividef(x_in[idx], divisor);
    }
}

// Uniform weight randomizer for Glorot initialization scheme

Eigen::MatrixXf uniformRV(float u, size_t dim_in, size_t dim_out)
{
    int N = dim_in * dim_out;

    unsigned seed = 0;

    std::default_random_engine generator(seed);
    std::uniform_real_distribution<float> distribution(-u, u);

    auto uniform = [&] (float) {return distribution(generator);};
    Eigen::VectorXf v = Eigen::VectorXf::NullaryExpr(N, uniform);

    return v.matrix().reshaped<Eigen::RowMajor>(dim_out, dim_in);
}

// GPU Variable structure, hold both the data and gradients -> similiar to tf.variable

struct Variable
{
    float* data;
    float* grad;
    size_t n, m, bytes;

    // Overloaded constructor (must specify dimensions)
    // Allocates data and gradient on CPU/GPU shared memory pool (cudaMallocManaged)

    Variable(size_t n, size_t m) : n(n), m(m)
    {
        bytes = n * m * sizeof(float);
        cudaMallocManaged(&data, bytes);
        cudaMallocManaged(&grad, bytes);
    }

    // Overloaded constructor (must specify dimensions and provide data)
    // Allocates data and gradient on CPU/GPU shared memory pool (cudaMallocManaged)

    Variable(size_t n, size_t m, float* data_in) : n(n), m(m)
    {
        bytes = n * m * sizeof(float);
        cudaMallocManaged(&data, bytes);
        cudaMallocManaged(&grad, bytes);
        std::memcpy(data, data_in, bytes);
    }

    // Free data and gradient on CPU/GPU shared memory pool

    ~Variable()
    {
        cudaFree(data);
        cudaFree(grad);
    }
};

// Base module class; with inheritable interfaces

class Module
{
    public:

    size_t dim_in, batch_size;

    // Contains mutable list of trainable weights in the network

    std::vector< std::shared_ptr<Variable> > trainable_weights;

    // Forward pass function for module

    virtual void forward(float* x_in, float* x_out) = 0;

    // Backward pass function for module

    virtual void backward(float* grad_in, float* grad_out) = 0;

    virtual ~Module() {}
};

// Base optimizer class; with inheritable interfaces

class Optimizer
{
    public:

    // Initialize optimizer states

    float learning_rate;

    virtual void initialize(std::vector< std::shared_ptr<Variable> > params) = 0;

    // Apply gradients for all parameters

    virtual void applyGradients(std::vector< std::shared_ptr<Variable> > params) = 0;

    virtual ~Optimizer() {}
};