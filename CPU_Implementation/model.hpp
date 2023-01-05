#include "module.hpp"
#include "optimizer.hpp"
#include "dataset.hpp"
#include "tqdm.h"

// Categorical Cross Entropy loss metric for training/testing loss evaluation

float categorical_cross_entropy(Eigen::MatrixXf y_pred, Eigen::MatrixXf labels, float epsilon = 1e-10)
{
    return ( -1*(labels.array() * log(y_pred.array() + epsilon)).colwise().sum() ).mean();
}

// Categorical accuracy metric for training/testing accuracy evaluation

float categorical_accuracy(Eigen::MatrixXf y_pred, Eigen::MatrixXf labels)
{
    int batch_size = y_pred.rows();
    int num_classes = y_pred.cols();

    Eigen::ArrayXi argmax_preds(batch_size);
    Eigen::ArrayXi argmax_labels(batch_size);

    Eigen::MatrixXf maxVal(batch_size, 2);

    for (int i = 0; i < batch_size; i++)
    {
        maxVal(i, 0) = y_pred.row(i).maxCoeff( &argmax_preds(i) );
        maxVal(i, 1) = labels.row(i).maxCoeff( &argmax_labels(i) );
    }

    return (argmax_preds == argmax_labels).cast<float>().mean();
}

// Sequential Layer -> sequentially passes through all specified modules in both forward and backward directions

/*
    Note the use of shared_ptr to the module/variable objects instead of the objects directly.
    This ensures that modifications to these objects in the sequential or optimizer 
    class are reflected throughout the entire program. We create a vector list of these shared pointers
    and pass them around; memory management is automated by reference counting, so freeing them is not required.
*/

class Sequential
{
    public:

    int num_modules;

    // List of shared pointers of modules 

    std::vector< std::shared_ptr<Module> > modules;

    // List of shared pointers of trainable parameters of all modules

    std::vector< std::shared_ptr<Variable> > params;

    // Dataset class to load training and testing datasets

    Dataset dataset;

    // Loss Module shared pointer

    std::shared_ptr<Module> loss;

    // Optimizer Module shared pointer

    std::shared_ptr<Optimizer> optimizer;

    // Raw array of Eigen Matrices to store forward module activations and backward gradients

    Eigen::MatrixXf* activations;

    Eigen::MatrixXf* gradients;

    // Constructor takes in list of modules, loss, and optimizer

    Sequential(std::vector< std::shared_ptr<Module> > modules, std::shared_ptr<Module> loss, std::shared_ptr<Optimizer> optimizer) : modules(modules), loss(loss), optimizer(optimizer)
    {   
        num_modules = modules.size();

        // Append all trainable weights in each module to the list of trainable weights

        for (std::shared_ptr<Module> module : modules)
        {
            std::vector< std::shared_ptr<Variable> > trainable_weights = module -> trainable_weights;

            params.insert(params.end(), trainable_weights.begin(), trainable_weights.end());
        }

        optimizer -> initialize(params);

        // Allocate raw array of activations and gradients

        activations = new Eigen::MatrixXf[num_modules + 1];

        gradients = new Eigen::MatrixXf[num_modules + 1];
    }

    // The destructor is needed to free the heap allocated activations and gradients

    ~Sequential()
    {
        delete [] activations;
        delete [] gradients;
    }

    // Sequential forward function -> fills all the activations

    Eigen::MatrixXf forward(Eigen::MatrixXf X)
    {
        int idx = 0;
        activations[idx] = X;
        for (std::shared_ptr<Module> module : modules)
        {
            activations[idx + 1] = module -> forward(activations[idx]);
            idx++;
        }
        return loss -> forward(activations[idx]);
    }

    // Sequential backward function -> fills all the gradients

    void backward(Eigen::MatrixXf y)
    {
        int idx = 0;
        gradients[idx] = loss -> backward(y);

        for (std::vector< std::shared_ptr<Module> >::reverse_iterator module_ptr = modules.rbegin(); module_ptr != modules.rend(); ++module_ptr)
        {
            gradients[idx + 1] = (*module_ptr) -> backward(gradients[idx]);
            idx++;
        }
    }

    // Training function; returns the training loss and training accuracy

    float* train(int batch_size)
    {
        float* train_metrics = new float[2];

        // Load training dataset

        std::vector< std::vector<Eigen::MatrixXf> > trainingData = dataset.trainingDataset(batch_size);

        std::vector<Eigen::MatrixXf> X_train = trainingData[0];

        std::vector<Eigen::MatrixXf> y_train = trainingData[1];

        Eigen::MatrixXf y_pred;

        int num_batches = X_train.size();

        // Array of losses and accuracies

        Eigen::ArrayXf cross_entropy_loss(num_batches);

        Eigen::ArrayXf accuracy(num_batches);

        std::cout << "\n Training: \n";

        // Create a progress bar

        tqdm bar;

        bar.reset();

        for (int i = 0; i < num_batches; i++)
        {
            y_pred = forward(X_train[i]);

            backward(y_train[i]);

            optimizer -> applyGradients(params);

            cross_entropy_loss(i) = categorical_cross_entropy(y_pred, y_train[i]);

            accuracy(i) = categorical_accuracy(y_pred, y_train[i]);

            bar.progress(i, num_batches);
        }

        bar.finish();

        train_metrics[0] = cross_entropy_loss.mean();

        train_metrics[1] = accuracy.mean();

        return train_metrics;

    }

    // Test function; returns the test loss and test accuracy

    float* test(int batch_size)
    {
        float* test_metrics = new float[2];

        // Load test dataset

        std::vector< std::vector<Eigen::MatrixXf> > testData = dataset.testDataset(batch_size);

        std::vector<Eigen::MatrixXf> X_test = testData[0];

        std::vector<Eigen::MatrixXf> y_test = testData[1];

        Eigen::MatrixXf y_pred;

        int num_batches = X_test.size();

        // Array of losses and accuracies

        Eigen::ArrayXf cross_entropy_loss(num_batches);

        Eigen::ArrayXf accuracy(num_batches);

        std::cout << "\n Testing: \n";

        // Create a progress bar

        tqdm bar;

        bar.reset();

        for (int i = 0; i < num_batches; i++)
        {
            y_pred = forward(X_test[i]);

            cross_entropy_loss(i) = categorical_cross_entropy(y_pred, y_test[i]);

            accuracy(i) = categorical_accuracy(y_pred, y_test[i]);

            bar.progress(i, num_batches);
        }

        bar.finish();

        test_metrics[0] = cross_entropy_loss.mean();

        test_metrics[1] = accuracy.mean();

        return test_metrics;
    }
};

