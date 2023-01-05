#include <iostream>
#include <stdio.h>
#include "tqdm.h"

// Print raw CUDA array

template<class T>
void printCudaArray(size_t n, size_t m, const T* arr)
{
    for (unsigned i = 0; i < n; i++)
    {
        for (unsigned j = 0; j < m; j++)
        {
            std::cout << arr[i*m + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}