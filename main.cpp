#include "model.hpp"


int main()
{

    // Specify batch size and learning rate

    unsigned batch_size = 250;

    float learning_rate = 0.1;

    // Create list of shared pointer of sequential modules

    std::vector< std::shared_ptr<Module> > modules =
    {
        std::make_shared<Dense>(784, 256), 
        std::make_shared<ELU>(0.9), 
        std::make_shared<Dense>(256, 64), 
        std::make_shared<ELU>(0.9), 
        std::make_shared<Dense>(64, 10)
    };

    // Create shared pointer to the loss module

    std::shared_ptr<Module> loss = std::make_shared<SoftmaxCrossEntropy>();

    // Create shared pointer to the optimizer module

    std::shared_ptr<Optimizer> sgd = std::make_shared<SGD>(learning_rate);

    // Create Sequential model

    Sequential model(modules, loss, sgd);

    int epochs = 25;

    Eigen::MatrixXf metrics(epochs, 4);

    // Run training + testing epochs

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        std::cout << "Epoch# " << epoch << "\n";

        float *train_metrics = model.train(batch_size);

        float *test_metrics = model.test(batch_size);

        metrics(epoch, 0) = train_metrics[0];
        metrics(epoch, 1) = train_metrics[1];

        metrics(epoch, 2) = test_metrics[0];
        metrics(epoch, 3) = test_metrics[1];

        std::cout << "Train Loss: " << train_metrics[0] << "\n";
        std::cout << "Train Accuracy: " << train_metrics[1] << "\n\n";

        std::cout << "Test Loss: " << test_metrics[0] << "\n";
        std::cout << "Test Accuracy: " << test_metrics[1] << "\n\n";
    }

    // Store training + testing metrics in csv file

    CSVData saveMetrics("NN_Metrics.csv", metrics);

    saveMetrics.writeToCSVfile();

    std::cout << "Completed!" << std::endl;
    return 0;
}