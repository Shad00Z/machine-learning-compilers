#!/bin/bash

/opt/homebrew/opt/llvm/bin/clang++ \
    -std=c++20 \
    -o EinsumExample \
    EinsumExample.cpp \
    -I../include \
    -L../lib \
    -lmlc \
    -I/opt/homebrew/opt/libomp/include \
    -L/opt/homebrew/opt/libomp/lib \
    -Xpreprocessor \
    -fopenmp \
    -lomp

/opt/homebrew/opt/llvm/bin/clang++ \
    -std=c++20 \
    -o TensorOperationExample \
    TensorOperationExample.cpp \
    -I../include \
    -L../lib \
    -lmlc \
    -I/opt/homebrew/opt/libomp/include \
    -L/opt/homebrew/opt/libomp/lib \
    -Xpreprocessor \
    -fopenmp \
    -lomp

/opt/homebrew/opt/llvm/bin/clang++ \
    -std=c++20 \
    -o OptimizerExample \
    OptimizerExample.cpp \
    -I../include \
    -L../lib \
    -lmlc \
    -I/opt/homebrew/opt/libomp/include \
    -L/opt/homebrew/opt/libomp/lib \
    -Xpreprocessor \
    -fopenmp \
    -lomp

# Check if the compilation was successful
if [ $? -eq 0 ]; then
    # Run the examples
    ./EinsumExample
    ./TensorOperationExample
    ./OptimizerExample
    echo "Compilation and execution successful."

    # Clean up the generated executables
    rm -f EinsumExample TensorOperationExample OptimizerExample
    rm -rf build
    echo "Cleaned up generated executables."
    exit 0
else
    echo "Compilation failed."
    exit 1
fi