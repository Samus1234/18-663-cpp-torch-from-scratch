// Stochastic Gradient Descent optimizer -> inherits from the Optimizer class

class SGD : public Optimizer
{
    public:

    float learning_rate;

    SGD(float learning_rate = 0.1) : learning_rate(learning_rate) { }

    void initialize(std::vector< std::shared_ptr<Variable> > params)
    {

    }

    // Apply gradients to each parameter

    void applyGradients(std::vector< std::shared_ptr<Variable> > params)
    {
        for (std::shared_ptr<Variable> param : params)
        {
            param -> data -= learning_rate * param -> grad;
        }
    }

};

// Adaptive Moment (Adam) optimizer -> inherits from the Optimizer class

class Adam : public Optimizer
{
    public:

    float learning_rate, beta1, beta2, epsilon;

    Eigen::MatrixXf* M;
    Eigen::MatrixXf* V;

    Adam(float learning_rate = 0.0001, float beta1 = 0.900, float beta2 = 0.999, float epsilon = 1e-7) : learning_rate(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon) { }

    ~Adam()
    {
        delete [] M;
        delete [] V;
    }

    // Allocate and initialize optimizer states

    void initialize(std::vector< std::shared_ptr<Variable> > params)
    {
        M = new Eigen::MatrixXf [params.size()];
        V = new Eigen::MatrixXf [params.size()];

        int idx = 0;

        for (std::shared_ptr<Variable> param : params)
        {
            int n = param -> data.rows();
            int m = param -> data.cols();

            M[idx] = Eigen::MatrixXf::Zero(n, m);
            V[idx] = Eigen::MatrixXf::Zero(n, m);

            idx++;
        }
    }

    // Apply gradients to each parameter

    void applyGradients(std::vector< std::shared_ptr<Variable> > params)
    {
        int idx = 0;

        for (std::shared_ptr<Variable> param : params)
        {

            M[idx] = beta1 * M[idx] + (1 - beta1) * param -> grad;
            V[idx] = beta2 * V[idx] + (1 - beta2) * param -> grad.cwiseSqrt();

            param -> data -= learning_rate * M[idx].cwiseProduct( (Eigen::MatrixXf::Constant(V[idx].rows(), V[idx].cols(), epsilon) + V[idx].cwiseSqrt()).cwiseInverse() );

            idx++;
        }
    }

};