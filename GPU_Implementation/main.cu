#include "model.cuh"

int main()
{
    // Specify batch size and learning rate

    size_t batch_size = 250;

    float learning_rate = 0.1;

    // Create list of shared pointer of sequential modules

    std::vector<std::shared_ptr<Module>> modules =
    {
        std::make_shared<Dense>(784, 256, batch_size), 
        std::make_shared<ELU>(256, batch_size, 0.9), 
        std::make_shared<Dense>(256, 64, batch_size), 
        std::make_shared<ELU>(64, batch_size, 0.9), 
        std::make_shared<Dense>(64, 10, batch_size)
    };

    // Create shared pointer to the optimizer module

    std::shared_ptr<Optimizer> optimizer = std::make_shared<SGD>(learning_rate);

    // Create shared pointer to the loss module

    std::shared_ptr<Module> loss = std::make_shared<SigmoidCrossEntropy>(10, batch_size);

    // Create Sequential model

    Sequential model(modules, loss, optimizer, batch_size);

    // Run training + testing epochs

    unsigned epochs = 25;

    Eigen::MatrixXf metrics = Eigen::MatrixXf::Zero(epochs, 4);

    long *train_metrics = new long[epochs];

    long *test_metrics = new long[epochs];

    for (unsigned epoch = 0; epoch < epochs; epoch++)
    {
        train_metrics[epoch] = model.train();

        test_metrics[epoch] = model.test();
    }

    long mask = (( (long)1 << (sizeof(float)*8 + 1)) - 1);

    for (unsigned epoch = 0; epoch < epochs; epoch++)
    {
        long train_loss_long = (~mask & train_metrics[epoch]) >> sizeof(float)*8;

        long train_acc_long = (mask & train_metrics[epoch]);

        long test_loss_long = (~mask & train_metrics[epoch]) >> sizeof(float)*8;

        long test_acc_long = (mask & train_metrics[epoch]);

        float* train_loss_ = (float*) &train_loss_long;

        float* train_acc_ = (float*) &train_acc_long;

        float* test_loss_ = (float*) &test_loss_long;

        float* test_acc_ = (float*) &test_acc_long;

        metrics(epoch, 0) = *train_loss_;
        metrics(epoch, 1) = *train_acc_;
        metrics(epoch, 2) = *test_loss_;
        metrics(epoch, 3) = *test_acc_;

    }

    // Store training + testing metrics in csv file

    CSVData saveMetrics("cuda_NN_Metrics.csv", metrics);

    saveMetrics.writeToCSVfile();

    std::cout << "Done" << std::endl;

    return 0;  
}