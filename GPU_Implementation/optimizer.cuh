// Stochastic Gradient Descent optimizer -> inherits from the Optimizer class

class SGD : public Optimizer
{
    public:

    float learning_rate;

    Eigen::MatrixXf data, grad;

    size_t num_params;

    dim3 threads;
    dim3* blocks;

   // Create the 1-dimensional thread dim3 CUDA structure

    SGD(float learning_rate = 0.1) : learning_rate(learning_rate)
    {
        threads = dim3(NUM_THREADS);
    }

    // Allocate and fill 1-dimensional block dim3 CUDA structure for rach parameter

    void initialize(std::vector< std::shared_ptr<Variable> > params)
    {
        num_params = params.size();

        blocks = new dim3 [num_params];

        unsigned idx = 0;

        for (std::shared_ptr<Variable> param : params)
        {
            size_t N = param -> n * param -> m;

            size_t NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

            blocks[idx] = dim3(NUM_BLOCKS);

            idx++;
        }
    }

    // Apply gradients to each parameter

    void applyGradients(std::vector< std::shared_ptr<Variable> > params)
    {
        unsigned idx = 0;

        for (std::shared_ptr<Variable> param : params)
        {
            matadd<<<blocks[idx], threads>>>(param -> n, param -> m, -learning_rate, param -> data, param -> grad, param -> data);

            idx++;
        }
    }

};
