#include "base.cuh"

// Exponential Linear Unit activation module -> inherits from Module class

class ELU : public Module
{
    public:

    float alpha;

    size_t dim_in, batch_size;

    size_t N, bytes;

    dim3 threads, blocks;

    float* x;

    ELU(size_t dim_in, size_t batch_size, float alpha = 0.9) : dim_in(dim_in), batch_size(batch_size), alpha(alpha)
    {
        N = dim_in * batch_size;
        bytes = N * sizeof(float);

        // Compute the number of blocked based on the size of the matrix

        size_t NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

        // Create the 1-dimensional threads and blocks dim3 CUDA structures
        
        threads = dim3(NUM_THREADS);
        blocks = dim3(NUM_BLOCKS);

        // Allocate CPU/GPU shared memory on the layer (cudaMallocManaged)

        cudaMallocManaged(&x, bytes);
    };

    // Free CPU/GPU shared memory on the layer

    ~ELU()
    {
        cudaFree(x);
    }

    // Copy input to the layer into the allocated shared memory and compute the ELU forward pass

    void forward(float* x_in, float* x_out)
    {
        gpucopy<<<blocks, threads>>>(N, x_in, x);

        elu<<<blocks, threads>>>(N, alpha, x_in, x_out);
    }

    // Compute the ELU backward pass

    void backward(float* grad_in, float* grad_out)
    {
        eluGrad<<<blocks, threads>>>(N, alpha, x, grad_in, grad_out);
    }
};

// Densely connected module -> inherits from Module class

class Dense : public Module
{
    public:

    size_t dim_in, dim_out, batch_size;

    dim3 threads1d, blocks1d, threads2d, blocks2d;

    float *x;

    Eigen::MatrixXf eig_grad, eig_x, eig_W_grad, eig_b_grad;

    Dense(size_t dim_in, size_t dim_out, size_t batch_size) : dim_in(dim_in), dim_out(dim_out), batch_size(batch_size)
    {
        // Initialize the weights and biases

        float u = std::sqrt(6.0 / (float) (dim_in + dim_out));

        Eigen::MatrixXf W = uniformRV(u, dim_in, dim_out);
        Eigen::MatrixXf b = Eigen::MatrixXf::Zero(dim_out, 1);

        // Insert the weight and bias onto the trainable weights list

        trainable_weights.push_back(std::make_shared<Variable>(dim_in, dim_out, W.data()));
        trainable_weights.push_back(std::make_shared<Variable>(dim_out, 1, b.data()));

        // Allocate CPU/GPU shared memory on the layer (cudaMallocManaged)

        cudaMallocManaged(&x, batch_size * dim_in * sizeof(float));

        // Compute the number of blocked based on the size of the matrix

        size_t NUM_BLOCKS = (dim_in * dim_out * batch_size + NUM_THREADS - 1) / NUM_THREADS;

        // Create the 1-dimensional threads and blocks dim3 CUDA structures

        threads1d = dim3(NUM_THREADS);
        blocks1d = dim3(NUM_BLOCKS);

        // Create the 2-dimensional threads and blocks dim3 CUDA structures

        threads2d = dim3(NUM_THREADS, NUM_THREADS);
        blocks2d = dim3(NUM_BLOCKS, NUM_BLOCKS);
    }

    // Free CPU/GPU shared memory on the layer

    ~Dense()
    {
        cudaFree(x);
    }

    // Copy input to the layer into the allocated shared memory and compute the dense forward pass

    void forward(float* x_in, float* x_out)
    {
        gpucopy<<<blocks1d, threads1d>>>(dim_in * batch_size, x_in, x);

        batchedProd<<<blocks2d, threads2d>>>(batch_size, dim_in, dim_out, x_in, trainable_weights[0] -> data, x_out);

        replicateAdd<<<blocks2d, threads2d>>>(batch_size, dim_out, 1.0f, x_out, trainable_weights[1] -> data);
    }

    // Compute the dense backward pass and set weight and bias gradients

    void backward(float* grad_in, float* grad_out)
    {
        weightGrad<<<blocks2d, threads2d>>>(dim_out, batch_size, dim_in, grad_in, x, trainable_weights[0] -> grad);

        biasGrad<<<blocks1d, threads1d>>>(dim_out, batch_size, grad_in, trainable_weights[1] -> grad);

        matmul<<<blocks2d, threads2d>>>(batch_size, dim_out, dim_in, grad_in, trainable_weights[0] -> data, grad_out);
    }

};

// Sigmoid layer + Cross Entropy loss module -> inherits from Module class

class SigmoidCrossEntropy : public Module
{
    public:

    size_t batch_size, dim_in;

    float* y_pred;

    dim3 threads, blocks;

    SigmoidCrossEntropy(size_t dim_in, size_t batch_size) : dim_in(dim_in), batch_size(batch_size)
    {

        // Compute the number of blocked based on the size of the matrix

        size_t NUM_BLOCKS = (dim_in * batch_size + NUM_THREADS - 1) / NUM_THREADS;

        // Create the 1-dimensional threads and blocks dim3 CUDA structures

        threads = dim3(NUM_THREADS);
        blocks = dim3(NUM_BLOCKS);

        // Allocate CPU/GPU shared memory on the layer (cudaMallocManaged)

        cudaMallocManaged(&y_pred, dim_in * batch_size * sizeof(float));
    }

    // Free CPU/GPU shared memory on the layer

    ~SigmoidCrossEntropy()
    {
        cudaFree(y_pred);
    }

    // Copy input to the layer into the allocated shared memory and compute the sigmoid forward pass

    void forward(float* logits, float* predictions)
    {
        sigmoidActivation<<<blocks, threads>>>(dim_in * batch_size, logits, predictions);

        gpucopy<<<blocks, threads>>>(dim_in * batch_size, predictions, y_pred);
    }

    // Compute the sigmoid backward pass

    void backward(float* labels, float* grad_out)
    {
        matadd<<<blocks, threads>>>(batch_size, dim_in, -1, y_pred, labels, grad_out);

        sigmoidGrad<<<blocks, threads>>>(dim_in * batch_size, y_pred, grad_out, grad_out);
    }

};



