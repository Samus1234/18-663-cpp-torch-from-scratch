#include "base.hpp"

// Flatten module; reshapes forward and backward passes -> inherits from Module class

class Flatten : public Module
{
    public:

    size_t n, m;
    
    Eigen::MatrixXf forward(Eigen::MatrixXf x)
    {
        n = x.rows();
        m = x.cols();

        return x.reshaped<Eigen::RowMajor>(n*m, 1);
    }

    Eigen::MatrixXf backward(Eigen::MatrixXf grad)
    {
        return grad.reshaped<Eigen::RowMajor>(n, m);
    }
};

// Exponential Linear Unit activation module -> inherits from Module class

class ELU : public Module
{
    public:

    float alpha;

    Eigen::MatrixXf x;

    ELU(float alpha = 0.9) : alpha(alpha) {};

    Eigen::MatrixXf forward(Eigen::MatrixXf x)
    {
        this -> x = x;

        Eigen::MatrixXf x_arr;

        x_arr = (x.array() > 0).select(x.array(), 0) + alpha * (x.array() <= 0).select((exp(x.array()) - 1), 0);

        return x_arr;
    }

    Eigen::MatrixXf backward(Eigen::MatrixXf grad)
    {
        Eigen::MatrixXf dLdx = grad.array() * ( alpha * exp(x.array()) - (x.array() > 0).select((alpha * exp(x.array()) - 1), 0)).array();

        return dLdx;
    }
};

// Densely connected module -> inherits from Module class

class Dense : public Module
{
    public:

    size_t dim_in, dim_out;

    int batch;

    Eigen::MatrixXf x;

    Dense(size_t dim_in, size_t dim_out)
    {
        this -> dim_in = dim_in;
        this -> dim_out = dim_out;

        float u = std::sqrt(6.0 / (float) (dim_in + dim_out));

        Eigen::MatrixXf W = uniformRV(u, dim_in, dim_out);

        Eigen::MatrixXf b = Eigen::MatrixXf::Zero(dim_out, 1);

        trainable_weights.push_back(std::make_shared<Variable>(W));
        trainable_weights.push_back(std::make_shared<Variable>(b));
    }

    Eigen::MatrixXf forward(Eigen::MatrixXf x)
    {
        this -> x = x;

        batch = x.rows();

        return x * trainable_weights[0] -> data.transpose() + trainable_weights[1] -> data.transpose().replicate(batch, 1);
    }

    Eigen::MatrixXf backward(Eigen::MatrixXf grad)
    {
        trainable_weights[0] -> grad = (grad.transpose() * x) / ( (float) batch );
        trainable_weights[1] -> grad = grad.transpose().rowwise().sum() / ( (float) batch );

        Eigen::MatrixXf dx = grad * trainable_weights[0] -> data;

        return dx;
    }
};

// Softmax layer + Cross Entropy loss module -> inherits from Module class

class SoftmaxCrossEntropy : public Module
{
    public:

    Eigen::MatrixXf y_pred;

    int classes;

    Eigen::MatrixXf forward(Eigen::MatrixXf logits)
    {
        classes = logits.cols();

        Eigen::MatrixXf exp_logits = exp( (logits - logits.rowwise().maxCoeff().replicate(1, classes) ).array() );

        Eigen::MatrixXf sum_exp_logits = exp_logits.rowwise().sum().replicate(1, classes);

        y_pred = exp_logits.cwiseProduct(sum_exp_logits.cwiseInverse());

        return y_pred;
    }

    Eigen::MatrixXf backward(Eigen::MatrixXf labels)
    {
        return y_pred - labels;
    }

};

// Sigmoid layer + Cross Entropy loss module -> inherits from Module class

class SigmoidCrossEntropy : public Module
{
    public:

    Eigen::MatrixXf y_pred;

    int classes;

    Eigen::MatrixXf forward(Eigen::MatrixXf logits)
    {
        y_pred = expit(logits);

        return y_pred;
    }

    Eigen::MatrixXf backward(Eigen::MatrixXf labels)
    {
        return (y_pred - labels).cwiseProduct(expitGrad(y_pred));
    }

    private:

    Eigen::MatrixXf expit(Eigen::MatrixXf x)
    {
        return (1 + exp(-x.array())).cwiseInverse(); 
    }

    Eigen::MatrixXf expitGrad(Eigen::MatrixXf x)
    {
        Eigen::MatrixXf S = expit(x);
        Eigen::MatrixXf S_ = Eigen::MatrixXf::Constant(S.rows(), S.cols(), 1.0f) - S;
        return S.cwiseProduct(S_);
    }
};

