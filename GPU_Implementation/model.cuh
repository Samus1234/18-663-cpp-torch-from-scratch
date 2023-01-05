#include "module.cuh"
#include "optimizer.cuh"
#include "dataset.hpp"

// Categorical Cross Entropy loss metric for training/testing loss evaluation -> casts the raw arrays to Eigen matrices 

float categorical_cross_entropy(size_t batch_size, size_t classes, float* y_pred, float* labels, float epsilon = 1e-10)
{
    Eigen::MatrixXf labels_ = Eigen::Map<Eigen::MatrixXf>(labels, batch_size, classes);
    Eigen::MatrixXf y_pred_ = Eigen::Map<Eigen::MatrixXf>(y_pred, batch_size, classes);
    return ( -1*(labels_.array() * log(y_pred_.array() + epsilon)).colwise().sum() ).mean();
}

// Categorical accuracy metric for training/testing accuracy evaluation -> casts the raw arrays to Eigen matrices 

float categorical_accuracy(size_t batch_size, size_t classes, float* y_pred, float* labels)
{
    Eigen::MatrixXf labels_ = Eigen::Map<Eigen::MatrixXf>(labels, batch_size, classes);
    Eigen::MatrixXf y_pred_ = Eigen::Map<Eigen::MatrixXf>(y_pred, batch_size, classes);

    Eigen::ArrayXi argmax_preds(batch_size);
    Eigen::ArrayXi argmax_labels(batch_size);

    Eigen::MatrixXf maxVal(batch_size, 2);

    for (int i = 0; i < batch_size; i++)
    {
        maxVal(i, 0) = y_pred_.row(i).maxCoeff( &argmax_preds(i) );
        maxVal(i, 1) = labels_.row(i).maxCoeff( &argmax_labels(i) );
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

    float weight_size;

    size_t num_modules, batch_size, training_batches, test_batches;

    // List of shared pointers of modules 

    std::vector< std::shared_ptr<Module> > modules;

    // List of shared pointers of trainable parameters of all modules

    std::vector< std::shared_ptr<Variable> > params;

    // Loss Module shared pointer

    std::shared_ptr<Module> loss;

    // Optimizer Module shared pointer

    std::shared_ptr<Optimizer> optimizer;

    // Dataset Module shared pointer

    std::shared_ptr<Dataset> dataset;

    Sequential(std::vector< std::shared_ptr<Module> > modules, std::shared_ptr<Module> loss, std::shared_ptr<Optimizer> optimizer, size_t batch_size) : modules(modules), loss(loss), optimizer(optimizer), batch_size(batch_size)
    {   
        num_modules = modules.size();

        // Store the dimensions of each module layer

        sizes = new size_t [num_modules + 1];

        // Store the number of bytes needed to be allocated for each module layer

        bytes = new size_t [num_modules + 1];

        // Allocate raw array of activations and gradients

        activations = new float* [num_modules + 1];

        gradients = new float* [num_modules + 1];

        // Load all trainable parameters and allocate CPU/CPU shared memory

        unsigned idx = 0;

        sizes[idx] = modules[0] -> trainable_weights[0] -> n;
        bytes[idx] = sizes[idx] * batch_size * sizeof(float);

        idx++;

        for (std::shared_ptr<Module> module : modules)
        {
            std::vector< std::shared_ptr<Variable> > trainable_weights = module -> trainable_weights;

            params.insert(params.end(), trainable_weights.begin(), trainable_weights.end());

            if (!trainable_weights.empty())
            {
                sizes[idx] = trainable_weights[0] -> m;
                bytes[idx] = trainable_weights[0] -> m * batch_size * sizeof(float);
            }

            else
            {
                sizes[idx] = sizes[idx - 1];
                bytes[idx] = bytes[idx - 1];
            }

            idx++;
        }

        for (unsigned i = 0; i <= num_modules; i++)
        {
            cudaMallocManaged(&activations[i], bytes[i]);
            cudaMallocManaged(&gradients[i], bytes[num_modules - i]);
        }

        cudaMallocManaged(&y_pred, bytes[num_modules]);
        cudaMallocManaged(&y_label, bytes[num_modules]);

        // Initialize optimizer and load training/testing datasets

        optimizer -> initialize(params);

        dataset = std::make_shared<Dataset>(batch_size);

        training_batches = dataset -> num_training_batches;
        test_batches = dataset -> num_test_batches;

        X_train = dataset -> trainingX();
        X_test = dataset -> testX();
        y_train = dataset -> trainingY();
        y_test = dataset -> testY();
        
    }

    // Free all heap allocations on CPU/CPU shared memory

    ~Sequential()
    {
        for (unsigned i = 0; i <= num_modules; i++)
        {
            cudaFree(activations[i]);
            cudaFree(gradients[i]);           
        }
        cudaFree(y_pred);
        cudaFree(y_label);

        delete [] eigenActivations;
        delete [] eigenGradients;
        delete [] paramData;
        delete [] paramGrad;
    }

    // Sequential forward function -> fills all the activations

    void forward(float *x_in)
    {
        unsigned idx = 0;

        std::memcpy(activations[idx], x_in, bytes[idx]);

        while(idx < num_modules)
        {
            modules[idx] -> forward(activations[idx], activations[idx + 1]);
            idx++;
        }

        loss -> forward(activations[idx], y_pred);
    }

    // Sequential backward function -> fills all the gradients

    void backward(float* y)
    {
        unsigned idx = 0;

        std::memcpy(y_label, y, bytes[num_modules - idx]);

        loss -> backward(y_label, gradients[idx]);

        while(idx < num_modules)
        {
            modules[(num_modules - 1) - idx] -> backward(gradients[idx], gradients[idx + 1]);
            idx++;
        }
    }

    // Training function; returns the training loss and training accuracy in bitwise long format

    long train()
    {
        float mean_accuracy = 0;
        float mean_loss = 0;
        float mean_factor = (float) training_batches;

        std::cout << "\n Training: \n";

        tqdm bar;

        bar.reset();

        // Run training epoch

        for (unsigned b = 0; b < training_batches; b++)
        {
            forward(X_train[b]);

            backward(y_train[b]);

            optimizer -> applyGradients(params);

            cudaDeviceSynchronize();

            mean_loss += categorical_cross_entropy(batch_size, sizes[num_modules], y_pred, y_train[b]);

            mean_accuracy += categorical_accuracy(batch_size, sizes[num_modules], y_pred, y_train[b]);

            bar.progress(b, iters);
        }

        bar.finish();

        // Compute mean training loss and accuracy over epoch

        mean_loss /= mean_factor;

        mean_accuracy /= mean_factor;

        // Covert raw floating point number to long from memory address

        long *mean_loss_ptr = (long*) &mean_loss;

        long *mean_accuracy_ptr = (long*) &mean_accuracy;

        // Perform bitwise masking to store both loss and accuracy in a single long variable

        long returnVal = ( (*mean_loss_ptr) << sizeof(float) * 8 ) | ( *mean_accuracy_ptr );

        printf("Final Training Loss: %f, Training Accuracy: %f \n", mean_loss, mean_accuracy);

        return returnVal;
    }

    long test()
    {
        float mean_accuracy = 0;
        float mean_loss = 0;
        float mean_factor = (float) test_batches;

        std::cout << "\n Testing: \n";

        tqdm bar;

        bar.reset();

        // Run test epoch

        for (unsigned b = 0; b < test_batches; b++)
        {
            forward(X_test[b]);

            cudaDeviceSynchronize();

            mean_loss += categorical_cross_entropy(batch_size, sizes[num_modules], y_pred, y_test[b]) / mean_factor;

            mean_accuracy += categorical_accuracy(batch_size, sizes[num_modules], y_pred, y_test[b]) / mean_factor;

            bar.progress(b, test_batches);
        }

        bar.finish();

        printf("Test Loss: %f, Test Accuracy: %f \n", mean_loss, mean_accuracy);

        // Covert raw floating point number to long from memory address

        long *mean_loss_ptr = (long*) &mean_loss;

        long *mean_accuracy_ptr = (long*) &mean_accuracy;

        // Perform bitwise masking to store both loss and accuracy in a single long variable

        long returnVal = ( (*mean_loss_ptr) << sizeof(float) * 8 ) | ( *mean_accuracy_ptr );

        return returnVal;
    }

    private:

    // Train and test metric arrays

    float metricsTrain[2];
    float metricsTest[2];

    // Raw array of raw arrays to store forward module activations and backward gradients

    float *y_pred, *y_label;
    float **activations;
    float **gradients;

    // Array of number of bytes and layer dimensions 

    size_t *bytes, *sizes;

    // Raw array of training and test batches

    float **X_train, **X_test, **y_train, **y_test;

};