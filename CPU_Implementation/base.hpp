#include <iostream>
#include <Eigen/Core>
#include <vector>
#include <chrono>
#include <memory>

// Uniform weight randomizer for Glorot initialization scheme

Eigen::MatrixXf uniformRV(float u, size_t dim_in, size_t dim_out)
{
    int N = dim_in * dim_out;

    // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    unsigned seed = 0;

    std::default_random_engine generator(seed);

    std::uniform_real_distribution<float> distribution(-u, u);

    auto uniform = [&] (float) {return distribution(generator);};

    Eigen::VectorXf v = Eigen::VectorXf::NullaryExpr(N, uniform);

    return v.matrix().reshaped<Eigen::RowMajor>(dim_out, dim_in);
}

// Variable structure, hold both the data and gradients -> similiar to tf.variable

struct Variable
{
    Eigen::MatrixXf data;
    Eigen::MatrixXf grad;

    Variable(Eigen::MatrixXf data)
    {
        this -> data = data;
        grad = Eigen::MatrixXf::Zero(data.rows(), data.cols());
    }
};

// Base module class; with inheritable interfaces

class Module
{
    public:

    // Contains mutable list of trainable weights in the network

    std::vector< std::shared_ptr<Variable> > trainable_weights;

    // Forward pass function for module

    virtual Eigen::MatrixXf forward(Eigen::MatrixXf x) = 0;

    // Backward pass function for module

    virtual Eigen::MatrixXf backward(Eigen::MatrixXf grad) = 0;

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