#include <fstream>
const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

// Read and write csv data

class CSVData
{
    public:
    Eigen::MatrixXf data;
    std::string filename;

    CSVData(std::string filename_)
    {
        filename = filename_;
    }

    CSVData(std::string filename_, Eigen::MatrixXf data_)
    {
        filename = filename_;
        data = data_;
    }

    void writeToCSVfile()
    {
        std::ofstream file(filename.c_str());
        file << data.format(CSVFormat);
        file.close();
    }

    Eigen::MatrixXf readFromCSVfile()
    {
        std::vector<float> matrixEntries;
        std::ifstream matrixDataFile(filename);
        std::string matrixRowString;
        std::string matrixEntry;
        int matrixRowNumber = 0;
    
        while (getline(matrixDataFile, matrixRowString))
        {
            std::stringstream matrixRowStringStream(matrixRowString);
            while (getline(matrixRowStringStream, matrixEntry, ','))
            {
                matrixEntries.push_back(stod(matrixEntry));
            }
            matrixRowNumber++;
        }
        
        return Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
    }
};

// Load training and testing datasets and create batches

class Dataset
{
    Eigen::MatrixXf X_train;
    Eigen::MatrixXf y_train;

    Eigen::MatrixXf X_test;
    Eigen::MatrixXf y_test;

    public:

    Dataset()
    {
        CSVData featuresTrain("X_train.csv");
        CSVData labelsTrain("y_train.csv");

        X_train = featuresTrain.readFromCSVfile();
        y_train = labelsTrain.readFromCSVfile();

        CSVData featuresTest("X_test.csv");
        CSVData labelsTest("y_test.csv");

        X_test = featuresTest.readFromCSVfile();
        y_test = labelsTest.readFromCSVfile();
    }

    std::vector< std::vector<Eigen::MatrixXf> > trainingDataset(int batch_size)
    {
        int num_samples = X_train.rows();
        int num_batches = num_samples / batch_size;

        std::vector<Eigen::MatrixXf> x_train_data;
        std::vector<Eigen::MatrixXf> y_train_data;

        std::vector< std::vector<Eigen::MatrixXf> > trainingData;

        for (int i = 0; i < num_batches; i++)
        {
            x_train_data.push_back( X_train(Eigen::seq(i*(batch_size-1), (i+1)*(batch_size-1)), Eigen::all) );

            y_train_data.push_back( y_train(Eigen::seq(i*(batch_size-1), (i+1)*(batch_size-1)), Eigen::all) );
        }

        trainingData.push_back(x_train_data);

        trainingData.push_back(y_train_data);

        return trainingData;
    }

    std::vector< std::vector<Eigen::MatrixXf> > testDataset(int batch_size)
    {

        int num_samples = X_test.rows();
        int num_batches = num_samples / batch_size;

        std::vector<Eigen::MatrixXf> x_test_data;
        std::vector<Eigen::MatrixXf> y_test_data;

        std::vector< std::vector<Eigen::MatrixXf> > testData;

        for (int i = 0; i < num_batches; i++)
        {
            x_test_data.push_back( X_test(Eigen::seq(i*(batch_size-1), (i+1)*(batch_size-1)), Eigen::all) );

            y_test_data.push_back( y_test(Eigen::seq(i*(batch_size-1), (i+1)*(batch_size-1)), Eigen::all) );
        }

        testData.push_back(x_test_data);

        testData.push_back(y_test_data);

        return testData;
    }

};